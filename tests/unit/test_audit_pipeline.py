"""RED-phase failing tests for issue #24 (bba.audit_pipeline).

Each ``class`` maps to one acceptance criterion in the issue body or to
one explicit user constraint. Tests assert contracts (the WHY), not
implementation choices — see PRD §"Testing Decisions".

No implementation exists yet beyond the model layer. Every public
function in :mod:`bba.audit_pipeline.state_machine`,
:mod:`bba.audit_pipeline.pipeline`, :mod:`bba.audit_pipeline.resume`,
:mod:`bba.audit_pipeline.replay`, :mod:`bba.audit_pipeline.cost_guard`,
and :class:`bba.audit_pipeline.InMemoryBatchRunStore` raises
``NotImplementedError("RED-phase scaffold; see issue #24")``. A test that
fails with that string is "well-failed"; an ``ImportError`` or
``AttributeError`` means the scaffold drifted, not that the test
discovered a real regression.

Acceptance-criterion → test-class map:

* AC ① "Implementation in ``src/bba/audit_pipeline/``" → covered by
  the public-API import block (collection-time surface pin).
* AC ② "State-machine transitions tested" → :class:`TestStateMachine`,
  :class:`TestInMemoryBatchRunStore`.
* AC ③ "Resume scenario: kill process mid-batch, restart, verify no
  duplicate writes and no lost work" → :class:`TestResumeOnStartup`,
  :class:`TestSigtermMidBatchResume`.
* AC ④ "Replay test: applying same batch result twice is a no-op
  (property test)" → :class:`TestReplayIdempotencyProperty`.
* AC ⑤ "End-to-end smoke test: tiny synthetic dataset flows through
  full pipeline → audit_results table" → :class:`TestEndToEndSmoke`.
* AC ⑥ "Coverage ≥ 70%; ruff + mypy clean" → structural; the public
  surface is locked by the import block.

User-constraint → test-class map:

* #6 winning-attempt rule (last verifier-passed wins; else
  NEEDS_REVIEW + hallucination_suspect) → :class:`TestWinningAttemptRule`.
* #7 replay idempotency property → :class:`TestReplayIdempotencyProperty`.
* #8 end-to-end smoke (5 synthetic orders, 4 Hb-tier branches +
  INSUFFICIENT_EVIDENCE) → :class:`TestEndToEndSmoke`.
* #10 cost guard (live Anthropic API never called in tests) →
  :class:`TestCostGuardRejectsLiveTransport`.

Adversarial fixtures: the issue body singles out ``quote_grounder`` and
``vitals_extractor`` for adversarial coverage. The pipeline layer
re-asserts that contract end-to-end in
:class:`TestAdversarialQuoteGrounderRoutesToNeedsReview` and
:class:`TestAdversarialVitalsExtractorPropagates`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bba.audit_orders import AuditOrder
from bba.audit_store import LlmCall
from bba.audit_store.models import Classification
from bba.audit_pipeline import (
    VALID_TRANSITIONS,
    AuditPipelineConfig,
    AuditPipelineError,
    BatchRun,
    BatchRunState,
    BatchRunStore,
    BatchStateTransitionError,
    InMemoryBatchRunStore,
    LiveAnthropicApiError,
    PipelineRowContext,
    PipelineRunResult,
    ResumeReconciliationError,
    ResumeReport,
    apply_batch_results,
    assert_test_safe_transport,
    is_terminal,
    process_audit_order,
    resume_on_startup,
    run_pipeline,
    select_winning_attempt,
    transition,
)
from bba.cohort_detector import CohortAssignment, CohortLabel
from bba.hb_lookup import DeltaHbWindow, HbLookupResult
from bba.prompt_builder import EvidenceChunk
from bba.vitals_extractor import SourceProvenance, VitalSigns, VitalsResult
from bba.llm_client import (
    AnthropicBatchTransport,
    AnthropicTransport,
    CassetteTransport,
    LlmClientConfig,
    RawBatchResponse,
    SONNET_MODEL_ID,
)


# =============================================================================
# Public-surface pin
#
# Importing the full surface at module scope is the collection-time
# regression test: if a public name is renamed or removed in a refactor
# the test module fails to load and pytest reports the drift before any
# test body runs. The tuple below pins names that the test bodies do
# not directly reference, so ruff does not strip them as unused.
# =============================================================================


_PUBLIC_SURFACE: tuple[object, ...] = (
    VALID_TRANSITIONS,
    AuditPipelineConfig,
    AuditPipelineError,
    BatchRun,
    BatchRunState,
    BatchRunStore,
    BatchStateTransitionError,
    InMemoryBatchRunStore,
    LiveAnthropicApiError,
    PipelineRunResult,
    ResumeReconciliationError,
    ResumeReport,
    apply_batch_results,
    assert_test_safe_transport,
    is_terminal,
    process_audit_order,
    resume_on_startup,
    run_pipeline,
    select_winning_attempt,
    transition,
)


# =============================================================================
# Fixtures — minimal valid models so tests state only the fields they exercise.
# =============================================================================


_RUN_TS = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)


def _batch_run(
    *,
    batch_id: str = "batch-001",
    state: BatchRunState = BatchRunState.PENDING,
    audit_ids: tuple[str, ...] = ("audit-001",),
    run_id: str = "run-aaa",
    anthropic_batch_id: str | None = None,
    submitted_at: datetime | None = None,
    updated_at: datetime = _RUN_TS,
    error_message: str | None = None,
    code_version: str = "v0.1.0+test",
) -> BatchRun:
    return BatchRun(
        batch_id=batch_id,
        state=state,
        run_id=run_id,
        code_version=code_version,
        audit_ids=audit_ids,
        anthropic_batch_id=anthropic_batch_id,
        submitted_at=submitted_at,
        updated_at=updated_at,
        error_message=error_message,
    )


def _pipeline_config(**overrides: object) -> AuditPipelineConfig:
    base: dict[str, object] = {
        "db_url": "postgresql://localhost/test",
        "code_version": "v0.1.0+test",
        "max_batch_size": 100,
        "poll_interval_seconds": 30.0,
    }
    base.update(overrides)
    return AuditPipelineConfig.model_validate(base)


def _llm_config() -> LlmClientConfig:
    return LlmClientConfig(code_version="v0.1.0+test")


def _cassette_transport() -> CassetteTransport:
    return CassetteTransport(interactions=())


def _cassette_for_contexts(
    contexts: Sequence[PipelineRowContext],
    *,
    classification: str = "APPROPRIATE",
    review_reason: str | None = None,
) -> CassetteTransport:
    """Build a CassetteTransport pre-loaded with one successful response
    per order.

    Used by the smoke + adversarial integration tests so the pipeline's
    Anthropic submission step lands a recorded cassette interaction
    (PRD §22 contract). The cassette is keyed on
    ``(model, sorted_tuple(custom_ids))``; submitting any subset of
    the orders to the matching model returns the recorded responses.

    ``classification`` controls the structured-output payload baked
    into each result so a test can pin "all rows NEEDS_REVIEW" (the
    adversarial quote-grounder case) or "all rows APPROPRIATE" (the
    happy-path smoke).
    """
    from bba.llm_client.models import (
        BatchSubmissionResult,
        CassetteInteraction,
        RawBatchResponse,
    )

    # Only include LLM-required contexts: deterministic-final
    # classifications (APPROPRIATE / INSUFFICIENT_EVIDENCE / INAPPROPRIATE)
    # never hit Anthropic, so they don't belong in the cassette key.
    # We re-run the deterministic engine to mirror the pipeline's
    # partitioning logic.
    from bba.audit_pipeline.replay import _classify_from_context

    llm_contexts = tuple(
        ctx
        for ctx in contexts
        if _classify_from_context(ctx).classification
        not in {"APPROPRIATE", "INSUFFICIENT_EVIDENCE", "INAPPROPRIATE"}
    )
    if not llm_contexts:
        return CassetteTransport(interactions=())

    results = tuple(
        BatchSubmissionResult(
            custom_id=ctx.order.audit_id,
            model_id=SONNET_MODEL_ID,
            raw_response_json={
                "id": f"msg_smoke_{i:03d}",
                "type": "message",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "classify_audit",
                        "input": {
                            "classification": classification,
                            "indications": [],
                            "negative_evidence": [],
                            "reasoning_summary_en": f"smoke-{i}",
                            "reasoning_summary_th": "smoke-th",
                            **(
                                {"review_reason": review_reason}
                                if review_reason
                                else {}
                            ),
                        },
                    }
                ],
                "stop_reason": "tool_use",
            },
            request_json={"messages": [{"role": "user", "content": "..."}]},
            response_headers={"anthropic-version": "2023-06-01"},
            request_timestamp=_RUN_TS,
            latency_ms=1234,
            anthropic_version="2023-06-01",
            prompt_cache_id=None,
            extended_thinking_blocks=None,
        )
        for i, ctx in enumerate(llm_contexts)
    )
    interaction = CassetteInteraction(
        model=SONNET_MODEL_ID,
        custom_ids=tuple(ctx.order.audit_id for ctx in llm_contexts),
        response=RawBatchResponse(batch_id="msgbatch_smoke", results=results),
    )
    return CassetteTransport(interactions=(interaction,))


def _orphan_llm_call(ctx: PipelineRowContext, *, run_id: str) -> LlmCall:
    """Build a stand-alone :class:`LlmCall` for the orphan-re-emit test.

    Carries an Anthropic-shaped response payload so the resume
    reconciler can reconstruct a :class:`BatchSubmissionResult` and
    feed it through :func:`apply_batch_results`. Headers are folded
    in via the ``__bba_response_headers__`` namespaced key (same
    contract the real writer uses)."""
    return LlmCall(
        call_id=f"call-{ctx.order.audit_id}-orphan",
        audit_id=ctx.order.audit_id,
        run_id=run_id,
        model_id=SONNET_MODEL_ID,
        anthropic_version="2023-06-01",
        prompt_cache_id=None,
        request_json={"messages": [{"role": "user", "content": "..."}]},
        response_json={
            "id": "msg_orphan",
            "type": "message",
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "name": "classify_audit",
                    "input": {
                        "classification": "APPROPRIATE",
                        "indications": [],
                        "negative_evidence": [],
                        "reasoning_summary_en": "orphan-replay",
                        "reasoning_summary_th": "orphan-th",
                    },
                }
            ],
            "__bba_response_headers__": {"anthropic-version": "2023-06-01"},
        },
        request_timestamp=_RUN_TS,
        latency_ms=900,
        extended_thinking_blocks=None,
        cold_storage_uri=None,
    )


def _row_context(
    *,
    audit_id: str,
    classification: Classification = "POTENTIALLY_INAPPROPRIATE",
    hb_value: float | None = 7.5,
    hb_source: str = "HEMATOLOGY",
    hb_freshness: str = "fresh",
    vitals_source: SourceProvenance = SourceProvenance.IPDADMPROGRESS,
    cohort_label: CohortLabel = CohortLabel.DEFAULT,
    cohort_threshold: float | None = 7.0,
    evidence_text: str = "Hb 7.5 with symptomatic chest pain",
) -> PipelineRowContext:
    """Build a PipelineRowContext whose upstream data drives the
    deterministic_classifier to produce the requested ``classification``.

    The pipeline now calls :func:`bba.deterministic_classifier.classify`
    itself; the test fixture sets ``hb_value`` and ``hb_freshness`` so
    that classifier output matches the requested classification.
    """
    # Drive the classifier output via Hb tier:
    # * APPROPRIATE — Hb strictly below cohort threshold (default 7.0)
    # * POTENTIALLY_INAPPROPRIATE — Hb in the [threshold, 10) tier
    # * INAPPROPRIATE — the deterministic engine never emits this, so
    #   the LLM-path simulation is responsible (the classifier returns
    #   POTENTIALLY_INAPPROPRIATE for Hb >= 10 and the cassette
    #   downgrades it). We map "INAPPROPRIATE" the same as
    #   "POTENTIALLY_INAPPROPRIATE" at the context layer so the
    #   classifier routes via LLM, and the caller sets cassette
    #   classification="INAPPROPRIATE" to exercise the branch.
    # * INSUFFICIENT_EVIDENCE — Hb missing (None).
    threshold = cohort_threshold if cohort_threshold is not None else 7.0
    if classification == "APPROPRIATE":
        hb = hb_value if hb_value is not None else max(1.0, threshold - 1.0)
        if hb >= threshold:
            hb = max(1.0, threshold - 1.0)
        freshness = hb_freshness
    elif classification == "INSUFFICIENT_EVIDENCE":
        hb = None
        freshness = "missing"
    elif classification == "INAPPROPRIATE":
        # Routes through LLM (deterministic returns
        # POTENTIALLY_INAPPROPRIATE); the test cassette downgrades to
        # INAPPROPRIATE. Use Hb >= 10 so the deterministic path is
        # POTENTIALLY_INAPPROPRIATE.
        hb = hb_value if hb_value is not None and hb_value >= 10.0 else 11.0
        freshness = hb_freshness
    else:  # POTENTIALLY_INAPPROPRIATE / NEEDS_REVIEW
        hb = hb_value if hb_value is not None else max(threshold, 7.5)
        if hb < threshold:
            hb = threshold + 0.5
        freshness = hb_freshness
    order = AuditOrder(
        audit_id=audit_id,
        hn=f"HN-{audit_id}",
        an=f"AN-{audit_id}",
        reqno=f"REQ-{audit_id}",
        order_datetime=_RUN_TS,
        anchor_imputed=False,
        products_ordered=("LPRC",),
        diagnosis_codes=("D62",),
    )
    hb_result = HbLookupResult(
        value_g_dl=hb,
        datetime_utc=_RUN_TS if hb is not None else None,
        source=hb_source if hb is not None else None,
        freshness=freshness,
        delta_hb_bypass=False,
        delta_hb_windows=(
            DeltaHbWindow(
                window_hours=6,
                threshold_g_dl=2.0,
                prior_value_g_dl=None,
                prior_datetime_utc=None,
                drop_g_dl=None,
                triggered=False,
            ),
        ),
        needs_review_single_low_hb=False,
    )
    vitals = VitalsResult(
        vitals=VitalSigns(sbp=110.0, hr=85.0, dbp=70.0, bt=37.0, rr=16.0),
        source=vitals_source,
        flags=frozenset(),
        note_timestamp=_RUN_TS,
    )
    cohort = CohortAssignment(
        label=cohort_label,
        threshold=cohort_threshold,
        evidence_code=None,
        evidence_name=None,
    )
    evidence_chunks = (
        EvidenceChunk(
            evidence_id="E1",
            source="IPDNRFOCUSDT",
            text=evidence_text,
        ),
    )
    return PipelineRowContext(
        order=order,
        hb_result=hb_result,
        vitals_result=vitals,
        cohort_assignment=cohort,
        procedure_proximity_hours=None,
        crystalloid_liters_prior_4h=0.0,
        hn_hash=f"hn_hash_{audit_id}",
        an_hash=f"an_hash_{audit_id}",
        prior_rbc_units_24h=0,
        prior_rbc_units_7d=0,
        redactor_version="0.4.1+test",
        redactor_model_sha=f"redactor_sha_{audit_id}",
        policy_version="kcmh-pr17.2-2024",
        prompt_hash=f"prompt_hash_{audit_id}",
        evidence_bundle_hash=f"bundle_hash_{audit_id}",
        evidence_chunks=evidence_chunks,
    )


# =============================================================================
# AC ② — State-machine transitions
#
# WHY: the state machine is the single load-bearing primitive that makes
# resume-on-startup safe. If PENDING could jump to COMPLETE, resume would
# stop polling Anthropic for a row whose batch was never submitted, and
# the audit_results row would never land. The tests below pin every legal
# move and every illegal move so a refactor cannot silently relax the
# contract.
# =============================================================================


class TestStateMachine:
    """``transition()`` is the only legal way to advance a ``BatchRun``."""

    def test_pending_can_transition_to_submitted(self) -> None:
        run = _batch_run(state=BatchRunState.PENDING)
        moved = transition(
            run,
            to_state=BatchRunState.SUBMITTED,
            anthropic_batch_id="msgbatch_01abc",
            now=_RUN_TS,
        )
        assert moved.state is BatchRunState.SUBMITTED
        assert moved.anthropic_batch_id == "msgbatch_01abc"
        # Source row is immutable — the original is unchanged.
        assert run.state is BatchRunState.PENDING

    def test_pending_can_transition_to_failed(self) -> None:
        run = _batch_run(state=BatchRunState.PENDING)
        moved = transition(
            run,
            to_state=BatchRunState.FAILED,
            error_message="anthropic batch shape rejected",
            now=_RUN_TS,
        )
        assert moved.state is BatchRunState.FAILED
        assert moved.error_message == "anthropic batch shape rejected"

    def test_submitted_can_transition_to_partial(self) -> None:
        run = _batch_run(
            state=BatchRunState.SUBMITTED,
            anthropic_batch_id="msgbatch_01abc",
            submitted_at=_RUN_TS,
        )
        moved = transition(run, to_state=BatchRunState.PARTIAL, now=_RUN_TS)
        assert moved.state is BatchRunState.PARTIAL
        assert moved.anthropic_batch_id == "msgbatch_01abc"  # preserved

    def test_submitted_can_transition_to_complete(self) -> None:
        run = _batch_run(
            state=BatchRunState.SUBMITTED,
            anthropic_batch_id="msgbatch_01abc",
            submitted_at=_RUN_TS,
        )
        moved = transition(run, to_state=BatchRunState.COMPLETE, now=_RUN_TS)
        assert moved.state is BatchRunState.COMPLETE

    def test_partial_can_transition_to_complete(self) -> None:
        run = _batch_run(
            state=BatchRunState.PARTIAL,
            anthropic_batch_id="msgbatch_01abc",
            submitted_at=_RUN_TS,
        )
        moved = transition(run, to_state=BatchRunState.COMPLETE, now=_RUN_TS)
        assert moved.state is BatchRunState.COMPLETE

    def test_pending_cannot_skip_to_complete(self) -> None:
        """PENDING -> COMPLETE would orphan an unpolled Anthropic batch."""
        run = _batch_run(state=BatchRunState.PENDING)
        with pytest.raises(BatchStateTransitionError):
            transition(run, to_state=BatchRunState.COMPLETE, now=_RUN_TS)

    def test_pending_cannot_skip_to_partial(self) -> None:
        run = _batch_run(state=BatchRunState.PENDING)
        with pytest.raises(BatchStateTransitionError):
            transition(run, to_state=BatchRunState.PARTIAL, now=_RUN_TS)

    def test_complete_is_terminal(self) -> None:
        run = _batch_run(
            state=BatchRunState.COMPLETE,
            anthropic_batch_id="msgbatch_01abc",
            submitted_at=_RUN_TS,
        )
        for target in BatchRunState:
            if target is BatchRunState.COMPLETE:
                continue
            with pytest.raises(BatchStateTransitionError):
                transition(run, to_state=target, now=_RUN_TS)

    def test_failed_is_terminal(self) -> None:
        run = _batch_run(
            state=BatchRunState.FAILED,
            anthropic_batch_id="msgbatch_01abc",
            submitted_at=_RUN_TS,
            error_message="boom",
        )
        for target in BatchRunState:
            if target is BatchRunState.FAILED:
                continue
            with pytest.raises(BatchStateTransitionError):
                transition(run, to_state=target, now=_RUN_TS)

    def test_transition_to_submitted_requires_anthropic_batch_id(self) -> None:
        """PENDING -> SUBMITTED with no anthropic_batch_id would leave the
        row with state=SUBMITTED and anthropic_batch_id=None — the resume
        reconciler has no batch to poll. Reject before persistence."""
        run = _batch_run(state=BatchRunState.PENDING)
        with pytest.raises((BatchStateTransitionError, ValueError)):
            transition(run, to_state=BatchRunState.SUBMITTED, now=_RUN_TS)

    def test_transition_to_failed_requires_error_message(self) -> None:
        run = _batch_run(state=BatchRunState.PENDING)
        with pytest.raises((BatchStateTransitionError, ValueError)):
            transition(run, to_state=BatchRunState.FAILED, now=_RUN_TS)


# =============================================================================
# AC ② extended — BatchRunStore round-trip on the in-memory implementation
#
# WHY: state-machine logic lives one level above the store. The Postgres
# implementation (GREEN) and the in-memory one must round-trip identical
# data so a unit test using InMemoryBatchRunStore validates the same
# contract that production exercises.
# =============================================================================


class TestInMemoryBatchRunStore:
    def test_create_then_get_round_trips(self) -> None:
        store = InMemoryBatchRunStore()
        run = _batch_run()
        store.create(run)
        assert store.get(run.batch_id) == run

    def test_get_missing_raises_keyerror(self) -> None:
        store = InMemoryBatchRunStore()
        with pytest.raises(KeyError):
            store.get("nonexistent")

    def test_create_duplicate_raises(self) -> None:
        store = InMemoryBatchRunStore()
        run = _batch_run()
        store.create(run)
        with pytest.raises((ValueError, KeyError)):
            store.create(run)

    def test_update_replaces_existing_row(self) -> None:
        store = InMemoryBatchRunStore()
        run = _batch_run(state=BatchRunState.PENDING)
        store.create(run)
        moved = transition(
            run,
            to_state=BatchRunState.SUBMITTED,
            anthropic_batch_id="msgbatch_01abc",
            now=_RUN_TS,
        )
        store.update(moved)
        assert store.get(run.batch_id).state is BatchRunState.SUBMITTED

    def test_list_by_state_filters(self) -> None:
        store = InMemoryBatchRunStore()
        a = _batch_run(batch_id="batch-a", state=BatchRunState.PENDING)
        b = _batch_run(
            batch_id="batch-b",
            state=BatchRunState.SUBMITTED,
            anthropic_batch_id="msgbatch_b",
            submitted_at=_RUN_TS,
        )
        store.create(a)
        store.create(b)
        submitted = store.list_by_state(BatchRunState.SUBMITTED)
        assert len(submitted) == 1
        assert submitted[0].batch_id == "batch-b"


# =============================================================================
# AC ③ — Resume-on-startup
#
# WHY: the pipeline must survive a SIGTERM mid-batch. The resume
# reconciler is the load-bearing primitive that delivers this. Tests
# below pin the three failure modes it must handle: SUBMITTED batch
# whose results are now available, PARTIAL batch with mixed status,
# orphan llm_calls whose audit_results row was never written.
# =============================================================================


class TestResumeOnStartup:
    def test_submitted_batch_is_polled(self) -> None:
        """A SUBMITTED row is the resume's primary target. Polling
        Anthropic for its batch_id is what advances the state."""
        store = InMemoryBatchRunStore()
        report = resume_on_startup(
            batch_run_store=store,
            audit_store=pytest.importorskip("bba.audit_store").AuditStore(
                pytest.importorskip("bba.audit_store").AuditStoreConfig(
                    root_dir=pytest.importorskip("pathlib").Path("/tmp/x"),
                    code_version="v0.1.0+test",
                )
            ),
        )
        assert isinstance(report, ResumeReport)

    def test_terminal_rows_are_skipped(self, tmp_path: object) -> None:
        """COMPLETE / FAILED rows are sinks — reconciler must leave them
        alone or it would re-submit work and double-write."""
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        store = InMemoryBatchRunStore()
        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        report = resume_on_startup(
            batch_run_store=store,
            audit_store=audit_store,
        )
        # No rows in the store → no work to do → empty report
        # (asserted once GREEN lands).
        assert isinstance(report, ResumeReport)

    def test_resume_is_idempotent(self, tmp_path: object) -> None:
        """A second resume against the same Postgres state must produce
        zero new writes — the audit_store's idempotency contract is the
        load-bearing primitive."""
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        store = InMemoryBatchRunStore()
        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        first = resume_on_startup(
            batch_run_store=store,
            audit_store=audit_store,
        )
        second = resume_on_startup(
            batch_run_store=store,
            audit_store=audit_store,
        )
        # Same input → same output. Both reports must list the same
        # completed_audit_ids and reemitted_audit_ids; zero divergence.
        assert first.completed_audit_ids == second.completed_audit_ids
        assert first.reemitted_audit_ids == second.reemitted_audit_ids


class TestSigtermMidBatchResume:
    """The acceptance criterion calls out a SIGTERM-then-restart scenario
    explicitly. The full integration test stands up a subprocess and
    sends a real SIGTERM; here we pin the contract surface so the
    integration test can be wired in GREEN."""

    def test_orphan_llm_calls_are_re_emitted(self, tmp_path: object) -> None:
        """A crash after llm_calls phase 1 but before audit_results
        phase 2 leaves an orphan call. The reconciler re-emits it
        through the verifier + winning-attempt rule and persists the
        audit row. PRD §10 names this as the expected failure mode.

        Codex review HIGH #6: assert behaviour, not just the typed
        return shape. Seeds a real orphan LlmCall, runs resume, and
        verifies a committed audit_row + zero duplicate writes on a
        second resume pass.
        """
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        run_id = "run-orphan"
        ctx = _row_context(
            audit_id="audit-orphan-001",
            classification="POTENTIALLY_INAPPROPRIATE",
        )

        # Phase-1-only state: seed an orphan llm_call via the
        # audit_store's leading-underscore seam (the same seam tests
        # for audit_store itself use to stage "crashed between phases"
        # states). No audit_results row is written.
        orphan_call = _orphan_llm_call(ctx, run_id=run_id)
        audit_store._persist_llm_calls([orphan_call])  # type: ignore[reportPrivateUsage]

        batch_run_store = InMemoryBatchRunStore()
        batch_run_store.create(
            BatchRun(
                batch_id="batch-orphan",
                state=BatchRunState.SUBMITTED,
                run_id=run_id,
                code_version="v0.1.0+test",
                audit_ids=(ctx.order.audit_id,),
                anthropic_batch_id="msgbatch_orphan",
                submitted_at=_RUN_TS,
                updated_at=_RUN_TS,
            )
        )

        # First resume: re-emits the orphan + commits the audit_row.
        first = resume_on_startup(
            batch_run_store=batch_run_store,
            audit_store=audit_store,
            contexts={ctx.order.audit_id: ctx},
        )
        assert ctx.order.audit_id in first.reemitted_audit_ids
        assert ctx.order.audit_id in first.completed_audit_ids
        rows = audit_store.read_audit_results(run_id=run_id)
        assert len(rows) == 1
        assert rows[0].audit_id == ctx.order.audit_id
        # The batch_runs row must have advanced to COMPLETE.
        assert batch_run_store.get("batch-orphan").state is BatchRunState.COMPLETE
        # No orphan llm_calls remain after the reconcile.
        reconcile = audit_store.reconcile(run_id=run_id)
        assert reconcile.orphan_call_ids == ()

        # Second resume: every state already settled. The batch_run
        # is COMPLETE (terminal) so it's skipped — the report carries
        # NO new completions or re-emissions. The audit_results row
        # remains a single committed row.
        second = resume_on_startup(
            batch_run_store=batch_run_store,
            audit_store=audit_store,
            contexts={ctx.order.audit_id: ctx},
        )
        assert second.reemitted_audit_ids == ()
        assert second.completed_audit_ids == ()
        assert second.polled_batch_ids == ()
        # Still exactly one audit_row — no double-write.
        rows_after = audit_store.read_audit_results(run_id=run_id)
        assert len(rows_after) == 1

    def test_pending_batch_is_marked_failed(self, tmp_path: object) -> None:
        """PENDING + no Anthropic batch_id is unrecoverable.

        Resume must surface the audit_ids as failed and transition the
        batch_run to FAILED with an operator-visible error_message
        (Codex review HIGH #4: PENDING rows previously dropped silently).
        """
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        batch_run_store = InMemoryBatchRunStore()
        batch_run_store.create(
            BatchRun(
                batch_id="batch-stuck",
                state=BatchRunState.PENDING,
                run_id="run-stuck",
                code_version="v0.1.0+test",
                audit_ids=("audit-stuck-001",),
                updated_at=_RUN_TS,
            )
        )

        report = resume_on_startup(
            batch_run_store=batch_run_store,
            audit_store=audit_store,
        )

        assert "audit-stuck-001" in report.failed_audit_ids
        failed_run = batch_run_store.get("batch-stuck")
        assert failed_run.state is BatchRunState.FAILED
        assert failed_run.error_message is not None
        assert "Anthropic" in failed_run.error_message

    def test_pipeline_persists_submitted_state_before_polling(
        self, tmp_path: object
    ) -> None:
        """Codex PR #54 P1: pipeline must persist (SUBMITTED, batch_id)
        BEFORE waiting for results. A transport that asserts the
        batch_run is already SUBMITTED at fetch time validates the
        ordering."""
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig
        from bba.llm_client.models import (  # noqa: F401 used in type strings
            BatchSubmissionRequest,
            RawBatchResponse,
        )

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        batch_run_store = InMemoryBatchRunStore()
        contexts = tuple(
            _row_context(
                audit_id=f"audit-order-{i:03d}",
                classification="POTENTIALLY_INAPPROPRIATE",
            )
            for i in range(2)
        )

        # Build a wrapping transport that asserts batch_run is in
        # SUBMITTED state at fetch_batch_results time. This is the
        # SIGTERM-recovery guarantee: by the time we wait for the
        # Anthropic poll, the batch_id is durable on disk.
        cassette = _cassette_for_contexts(contexts)
        observed_states_at_fetch: list[BatchRunState] = []

        class _OrderingProbe:
            def submit_batch_only(
                self,
                *,
                model: str,
                requests: "Sequence[BatchSubmissionRequest]",
                prompt_cache_enabled: bool,
            ) -> str:
                return cassette.submit_batch_only(
                    model=model,
                    requests=requests,
                    prompt_cache_enabled=prompt_cache_enabled,
                )

            def fetch_batch_results(
                self,
                batch_id: str,
                *,
                model: str,
                requests: "Sequence[BatchSubmissionRequest]",
                prompt_cache_enabled: bool,
            ) -> RawBatchResponse:
                # At this point every batch_run created in this
                # pipeline run MUST be in SUBMITTED (or later) state.
                for run in batch_run_store.list_all():
                    observed_states_at_fetch.append(run.state)
                return cassette.fetch_batch_results(
                    batch_id,
                    model=model,
                    requests=requests,
                    prompt_cache_enabled=prompt_cache_enabled,
                )

            def submit_batch(
                self,
                *,
                model: str,
                requests: "Sequence[BatchSubmissionRequest]",
                prompt_cache_enabled: bool,
            ) -> RawBatchResponse:
                return cassette.submit_batch(
                    model=model,
                    requests=requests,
                    prompt_cache_enabled=prompt_cache_enabled,
                )

        run_pipeline(
            contexts,
            transport=_OrderingProbe(),
            audit_store=audit_store,
            batch_run_store=batch_run_store,
            llm_config=_llm_config(),
            pipeline_config=_pipeline_config(),
            run_id="run-ordering",
        )

        # At fetch time, the row was SUBMITTED — i.e., the batch_id
        # was already persisted before polling. PENDING at fetch time
        # would mean a SIGTERM during polling could orphan the batch.
        assert observed_states_at_fetch
        assert all(
            state is BatchRunState.SUBMITTED for state in observed_states_at_fetch
        )

    def test_submitted_with_no_cached_calls_polls_anthropic(
        self, tmp_path: object
    ) -> None:
        """Codex PR #54 P1: when the pipeline persisted SUBMITTED + a
        batch_id but crashed before any results landed, resume must
        poll Anthropic via transport.fetch_batch_results — not
        declare the rows failed.

        Stage a SUBMITTED batch_run with a known anthropic_batch_id;
        pre-populate the cassette with the response keyed by THAT
        batch_id. resume must retrieve the response and commit
        audit_rows; the batch_run transitions to COMPLETE."""
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig
        from bba.llm_client import CassetteTransport
        from bba.llm_client.models import (
            BatchSubmissionResult,
            CassetteInteraction,
            RawBatchResponse,
        )

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        run_id = "run-poll-recover"
        ctx = _row_context(
            audit_id="audit-poll-001",
            classification="POTENTIALLY_INAPPROPRIATE",
        )

        # Build a cassette response that the resume reconciler will
        # find via fetch_batch_results(anthropic_batch_id="msgbatch_in_flight").
        response = RawBatchResponse(
            batch_id="msgbatch_in_flight",
            results=(
                BatchSubmissionResult(
                    custom_id=ctx.order.audit_id,
                    model_id=SONNET_MODEL_ID,
                    raw_response_json={
                        "id": "msg_poll",
                        "type": "message",
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "classify_audit",
                                "input": {
                                    "classification": "APPROPRIATE",
                                    "indications": [],
                                    "negative_evidence": [],
                                    "reasoning_summary_en": "poll-recovered",
                                    "reasoning_summary_th": "poll-th",
                                },
                            }
                        ],
                        "stop_reason": "tool_use",
                    },
                    request_json={"messages": []},
                    response_headers={"anthropic-version": "2023-06-01"},
                    request_timestamp=_RUN_TS,
                    latency_ms=2200,
                    anthropic_version="2023-06-01",
                    prompt_cache_id=None,
                    extended_thinking_blocks=None,
                ),
            ),
        )
        cassette = CassetteTransport(
            interactions=(
                CassetteInteraction(
                    model=SONNET_MODEL_ID,
                    custom_ids=(ctx.order.audit_id,),
                    response=response,
                ),
            )
        )

        batch_run_store = InMemoryBatchRunStore()
        batch_run_store.create(
            BatchRun(
                batch_id="batch-poll",
                state=BatchRunState.SUBMITTED,
                run_id=run_id,
                code_version="v0.1.0+test",
                audit_ids=(ctx.order.audit_id,),
                anthropic_batch_id="msgbatch_in_flight",
                submitted_at=_RUN_TS,
                updated_at=_RUN_TS,
            )
        )

        # Resume reconciler: must poll Anthropic, write audit_row,
        # transition batch_run to COMPLETE — no failed audit_ids.
        report = resume_on_startup(
            batch_run_store=batch_run_store,
            audit_store=audit_store,
            contexts={ctx.order.audit_id: ctx},
            transport=cassette,
            llm_config=_llm_config(),
        )
        assert report.failed_audit_ids == ()
        assert ctx.order.audit_id in report.completed_audit_ids
        rows = audit_store.read_audit_results(run_id=run_id)
        assert len(rows) == 1
        assert rows[0].audit_id == ctx.order.audit_id
        assert batch_run_store.get("batch-poll").state is BatchRunState.COMPLETE

    def test_resume_walks_submitted_batches_and_classifies_audit_ids(
        self,
        tmp_path: object,
    ) -> None:
        """Coverage for the non-empty reconcile path.

        Stage a real SUBMITTED batch with three audit_ids. Run the
        full smoke pipeline against the first two (so audit_results
        rows exist for them) and seed the SUBMITTED batch_run with
        all three. resume_on_startup must:

        * Walk the SUBMITTED row (polled_batch_ids includes it).
        * Classify the two audited rows as completed_audit_ids.
        * Classify the third (no llm_call, no audit_row) as failed.
        """
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )

        audited_contexts = _build_synthetic_contexts(n=2)
        run_pipeline(
            audited_contexts,
            transport=_cassette_for_contexts(audited_contexts),
            audit_store=audit_store,
            batch_run_store=InMemoryBatchRunStore(),
            llm_config=_llm_config(),
            pipeline_config=_pipeline_config(),
            run_id="run-recon",
        )

        store = InMemoryBatchRunStore()
        submitted_batch = BatchRun(
            batch_id="batch-recon",
            state=BatchRunState.SUBMITTED,
            run_id="run-recon",
            code_version="v0.1.0+test",
            audit_ids=(
                audited_contexts[0].order.audit_id,
                audited_contexts[1].order.audit_id,
                "audit-unaudited",
            ),
            anthropic_batch_id="msgbatch_recon",
            submitted_at=_RUN_TS,
            updated_at=_RUN_TS,
        )
        store.create(submitted_batch)

        report = resume_on_startup(
            batch_run_store=store,
            audit_store=audit_store,
        )

        assert submitted_batch.batch_id in report.polled_batch_ids
        assert audited_contexts[0].order.audit_id in report.completed_audit_ids
        assert audited_contexts[1].order.audit_id in report.completed_audit_ids
        assert "audit-unaudited" in report.failed_audit_ids

    def test_resume_advances_batch_run_to_complete_when_all_audited(
        self, tmp_path: object
    ) -> None:
        """Once every audit_id in a batch carries an audit_results row,
        the batch_runs row must transition to COMPLETE. Without this,
        the reconciler would re-poll forever."""
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        store = InMemoryBatchRunStore()
        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        report = resume_on_startup(
            batch_run_store=store,
            audit_store=audit_store,
        )
        # Concrete COMPLETE-transition assertion lands in GREEN once the
        # cassette-replay fixture is wired; here we pin the contract.
        assert isinstance(report.completed_audit_ids, tuple)


# =============================================================================
# AC ④ + user constraint #7 — Replay idempotency property test
#
# WHY: applying the same batch result set twice must be a no-op. This
# is the structural defense against double-writes during resume: the
# reconciler will apply results, possibly crash, and apply them again
# on the next boot — the audit_results table must remain unchanged.
# =============================================================================


_AUDIT_ID = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=48, max_codepoint=122
    ),
    min_size=3,
    max_size=12,
).map(lambda s: f"audit-{s}")


_CLASSIFICATIONS_STRAT = st.sampled_from(
    [
        "APPROPRIATE",
        "INAPPROPRIATE",
        "NEEDS_REVIEW",
        "INSUFFICIENT_EVIDENCE",
        "POTENTIALLY_INAPPROPRIATE",
    ]
)


class TestReplayIdempotencyProperty:
    @given(
        n_results=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=10_000),
        classification=_CLASSIFICATIONS_STRAT,
    )
    @settings(max_examples=15, deadline=None)
    def test_applying_same_batch_response_twice_writes_no_new_rows(
        self,
        n_results: int,
        seed: int,
        classification: str,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """For any synthetic RawBatchResponse of N results varying over
        every valid Classification, applying it twice produces zero
        new audit_results rows on the second pass.

        Codex LOW #11: classification varies across the Literal set so
        the property covers more than the happy-path APPROPRIATE
        responses."""
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        tmp_path: Path = tmp_path_factory.mktemp(f"replay_{seed}_{n_results}")
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        synthetic_response = _build_synthetic_raw_batch_response(
            n_results=n_results, seed=seed, classification=classification
        )
        # apply_batch_results requires a context per custom_id (no
        # silent fabrication of clinical data — Codex HIGH #5).
        contexts = {
            result.custom_id: _row_context(
                audit_id=result.custom_id,
                classification="POTENTIALLY_INAPPROPRIATE",
            )
            for result in synthetic_response.results
        }
        first = apply_batch_results(
            synthetic_response,
            audit_store=audit_store,
            run_id="run-replay",
            contexts=contexts,
        )
        second = apply_batch_results(
            synthetic_response,
            audit_store=audit_store,
            run_id="run-replay",
            contexts=contexts,
        )
        assert first.audit_ids_persisted  # at least one row written first pass
        assert second.audit_ids_persisted == (), (
            "second application must be a no-op (zero new rows)"
        )


def _build_synthetic_raw_batch_response(
    *, n_results: int, seed: int, classification: str = "APPROPRIATE"
) -> RawBatchResponse:
    """Build a synthetic RawBatchResponse for property testing.

    Each :class:`BatchSubmissionResult` is shaped just enough that
    :func:`apply_batch_results` can extract a deterministic
    ``audit_id`` and persist one minimal :class:`AuditRow` plus one
    backing :class:`LlmCall`. The seed makes the fixture reproducible
    without leaking across hypothesis examples (each n + seed pair
    produces a distinct set of ``audit_id``\\s)."""
    from bba.llm_client.models import BatchSubmissionResult

    results = tuple(
        BatchSubmissionResult(
            custom_id=f"audit-replay-{seed}-{i:03d}",
            model_id=SONNET_MODEL_ID,
            raw_response_json={
                "id": f"msg_{seed:05d}_{i:03d}",
                "type": "message",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "classify_audit",
                        "input": {
                            "classification": classification,
                            "indications": [],
                            "negative_evidence": [],
                            "reasoning_summary_en": f"synthetic-{seed}-{i}",
                            "reasoning_summary_th": "synthetic-th",
                        },
                    }
                ],
                "stop_reason": "tool_use",
            },
            request_json={"messages": [{"role": "user", "content": "..."}]},
            response_headers={"anthropic-version": "2023-06-01"},
            request_timestamp=_RUN_TS,
            latency_ms=1234,
            anthropic_version="2023-06-01",
            prompt_cache_id=None,
            extended_thinking_blocks=None,
        )
        for i in range(n_results)
    )
    return RawBatchResponse(batch_id=f"msgbatch_{seed:08x}", results=results)


# =============================================================================
# AC ⑤ + user constraint #8 — End-to-end smoke test
#
# WHY: integration coverage for the full composition chain. Five
# synthetic AuditOrders covering all four Hb-tier branches +
# INSUFFICIENT_EVIDENCE; the pipeline must produce five audit_results
# rows with the right classifications and zero orphan llm_calls.
# =============================================================================


class TestEndToEndSmoke:
    def test_five_synthetic_orders_flow_through_pipeline(
        self, tmp_path: object
    ) -> None:
        """The smoke test verifies the composition path, not any single
        module — every upstream module is already independently tested."""
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        synthetic_contexts = _build_synthetic_contexts(n=5)
        result = run_pipeline(
            synthetic_contexts,
            transport=_cassette_for_contexts(synthetic_contexts),
            audit_store=audit_store,
            batch_run_store=InMemoryBatchRunStore(),
            llm_config=_llm_config(),
            pipeline_config=_pipeline_config(),
            run_id="run-smoke",
        )
        assert isinstance(result, PipelineRunResult)
        assert len(result.audit_ids_persisted) == 5

        # AC ⑤ + Codex MEDIUM #9: assert classifications + provenance
        # per branch — every row, not just the count.
        from bba.audit_pipeline.replay import _classify_from_context

        rows = {
            row.audit_id: row
            for row in audit_store.read_audit_results(run_id="run-smoke")
        }
        # The synthetic set is built such that the deterministic
        # engine produces a mix of branches covering APPROPRIATE,
        # INSUFFICIENT_EVIDENCE, INAPPROPRIATE (via the LLM cassette),
        # and POTENTIALLY_INAPPROPRIATE.
        observed_final_classes: set[str] = set()
        observed_models: set[str] = set()
        for ctx in synthetic_contexts:
            persisted = rows[ctx.order.audit_id]
            classifier = _classify_from_context(ctx)
            observed_final_classes.add(persisted.final_classification)
            observed_models.add(persisted.model_id)
            if classifier.classification in {
                "APPROPRIATE",
                "INSUFFICIENT_EVIDENCE",
                "INAPPROPRIATE",
            }:
                # deterministic-final path: final == rule
                assert persisted.final_classification == classifier.classification
                assert persisted.model_id == "deterministic"
            else:
                # LLM path: final classification comes from the cassette;
                # the smoke cassette emits APPROPRIATE by default but
                # the INAPPROPRIATE branch test below uses a dedicated
                # cassette to exercise the downgrade.
                assert persisted.rule_classification == classifier.classification
                assert "sonnet" in persisted.model_id
        # The cohort of synthetic rows must exercise BOTH paths
        # (deterministic + LLM) — Codex MEDIUM #9 demanded multi-branch
        # coverage rather than a single happy path.
        assert "deterministic" in observed_models
        assert any("sonnet" in m for m in observed_models)

    def test_smoke_emits_no_orphan_llm_calls(self, tmp_path: object) -> None:
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        synthetic_contexts = _build_synthetic_contexts(n=5)
        result = run_pipeline(
            synthetic_contexts,
            transport=_cassette_for_contexts(synthetic_contexts),
            audit_store=audit_store,
            batch_run_store=InMemoryBatchRunStore(),
            llm_config=_llm_config(),
            pipeline_config=_pipeline_config(),
            run_id="run-smoke",
        )
        assert result.orphan_audit_ids == ()
        # No orphan llm_calls in the audit_store reconcile path either.
        reconcile_report = audit_store.reconcile(run_id="run-smoke")
        assert reconcile_report.orphan_call_ids == ()

    def test_inappropriate_branch_from_llm_downgrade(self, tmp_path: object) -> None:
        """Codex MEDIUM #9: explicit coverage of the INAPPROPRIATE
        final classification via the LLM downgrade path.

        Deterministic returns POTENTIALLY_INAPPROPRIATE for Hb >= 10;
        the cassette response carries classification=INAPPROPRIATE,
        so the persisted row's final_classification is INAPPROPRIATE
        with rule_classification = POTENTIALLY_INAPPROPRIATE."""
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        contexts = tuple(
            _row_context(
                audit_id=f"audit-inappropriate-{i:03d}",
                classification="INAPPROPRIATE",
            )
            for i in range(2)
        )
        run_pipeline(
            contexts,
            transport=_cassette_for_contexts(contexts, classification="INAPPROPRIATE"),
            audit_store=audit_store,
            batch_run_store=InMemoryBatchRunStore(),
            llm_config=_llm_config(),
            pipeline_config=_pipeline_config(),
            run_id="run-inappropriate",
        )
        rows = audit_store.read_audit_results(run_id="run-inappropriate")
        assert len(rows) == 2
        for row in rows:
            assert row.final_classification == "INAPPROPRIATE"
            assert row.rule_classification == "POTENTIALLY_INAPPROPRIATE"
            assert "sonnet" in row.model_id


class TestParseFailureBranches:
    """Codex MEDIUM #7: every parse-failure mode must persist a typed
    review_reason. The four branches mirror :class:`bba.llm_client.ParseFailureReason`."""

    @pytest.mark.parametrize(
        ("response_shape", "expected_reason"),
        [
            (
                # Empty content array → empty_response
                {
                    "id": "msg_x",
                    "type": "message",
                    "content": [],
                    "stop_reason": "end_turn",
                },
                "empty_response",
            ),
            (
                # First content block is not type=tool_use → tool_use_missing
                {
                    "id": "msg_x",
                    "type": "message",
                    "content": [{"type": "text", "text": "I cannot do that"}],
                    "stop_reason": "end_turn",
                },
                "tool_use_missing",
            ),
            (
                # tool_use input is not a mapping → schema_mismatch
                {
                    "id": "msg_x",
                    "type": "message",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "classify_audit",
                            "input": "not-a-dict",
                        }
                    ],
                    "stop_reason": "tool_use",
                },
                "schema_mismatch",
            ),
            (
                # classification value outside the allowed Literal →
                # classification_out_of_set
                {
                    "id": "msg_x",
                    "type": "message",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "classify_audit",
                            "input": {
                                "classification": "MAYBE_FINE",
                                "indications": [],
                                "negative_evidence": [],
                                "reasoning_summary_en": "x",
                                "reasoning_summary_th": "y",
                            },
                        }
                    ],
                    "stop_reason": "tool_use",
                },
                "classification_out_of_set",
            ),
        ],
    )
    def test_parse_failure_persists_typed_review_reason(
        self,
        response_shape: dict[str, object],
        expected_reason: str,
        tmp_path: object,
    ) -> None:
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig
        from bba.llm_client.models import BatchSubmissionResult, RawBatchResponse

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        ctx = _row_context(
            audit_id="audit-parse-001",
            classification="POTENTIALLY_INAPPROPRIATE",
        )
        result = BatchSubmissionResult(
            custom_id=ctx.order.audit_id,
            model_id=SONNET_MODEL_ID,
            raw_response_json=response_shape,
            request_json={"messages": []},
            response_headers={"anthropic-version": "2023-06-01"},
            request_timestamp=_RUN_TS,
            latency_ms=100,
            anthropic_version="2023-06-01",
            prompt_cache_id=None,
            extended_thinking_blocks=None,
        )
        response = RawBatchResponse(batch_id="msgbatch_parse", results=(result,))
        apply_batch_results(
            response,
            audit_store=audit_store,
            run_id="run-parse",
            contexts={ctx.order.audit_id: ctx},
        )
        rows = audit_store.read_audit_results(run_id="run-parse")
        assert len(rows) == 1
        assert rows[0].final_classification == "NEEDS_REVIEW"
        assert rows[0].review_reason == expected_reason


def _build_synthetic_contexts(*, n: int) -> Sequence[PipelineRowContext]:
    """Build n synthetic :class:`PipelineRowContext` rows covering
    APPROPRIATE (deterministic), INSUFFICIENT_EVIDENCE (deterministic),
    and POTENTIALLY_INAPPROPRIATE (LLM path).

    The pipeline runs the deterministic engine on each context and
    routes APPROPRIATE / INSUFFICIENT_EVIDENCE / INAPPROPRIATE directly
    to persistence (no LLM call). POTENTIALLY_INAPPROPRIATE /
    NEEDS_REVIEW route through the LLM stage, where the cassette
    transport supplies the response.

    Codex MEDIUM #9: the cycled set covers BOTH paths so the smoke
    test asserts multi-branch coverage rather than a single happy
    path. The dedicated INAPPROPRIATE branch is exercised in
    :class:`TestEndToEndSmoke.test_inappropriate_branch_from_llm_downgrade`.
    """
    classifications: tuple[Classification, ...] = (
        "POTENTIALLY_INAPPROPRIATE",  # routes via LLM
        "APPROPRIATE",  # deterministic-final
        "INSUFFICIENT_EVIDENCE",  # deterministic-final (missing Hb)
        "POTENTIALLY_INAPPROPRIATE",  # routes via LLM
        "POTENTIALLY_INAPPROPRIATE",  # routes via LLM
    )
    return tuple(
        _row_context(
            audit_id=f"audit-smoke-{i:03d}",
            classification=classifications[i % len(classifications)],
        )
        for i in range(n)
    )


# =============================================================================
# User constraint #6 — Winning-attempt rule
#
# WHY: a single audit_id may carry multiple llm_calls (retry,
# Sonnet→Opus escalation). The winning attempt is the last one whose
# verifier passed; if none passed, classification = NEEDS_REVIEW with
# hallucination_suspect=True. This rule must be the deterministic
# tiebreaker; without it, two re-runs could pick different attempts and
# produce different audit_results for the same input.
# =============================================================================


class TestWinningAttemptRule:
    def test_last_verifier_passed_attempt_wins(self) -> None:
        """Three attempts: attempt 1 passes, attempt 2 passes, attempt 3
        fails. Winner is attempt 2 — last one whose verifier passed."""
        calls = [
            {"attempt_id": 1, "verifier_pass": True, "classification": "APPROPRIATE"},
            {"attempt_id": 2, "verifier_pass": True, "classification": "INAPPROPRIATE"},
            {"attempt_id": 3, "verifier_pass": False, "classification": "APPROPRIATE"},
        ]
        winner = select_winning_attempt(calls)
        assert winner is not None
        # The winning attempt must be the latest verifier_pass=True row.
        assert winner["attempt_id"] == 2  # type: ignore[index]

    def test_no_passing_attempt_returns_none(self) -> None:
        """Caller routes None → NEEDS_REVIEW + hallucination_suspect."""
        calls = [
            {"attempt_id": 1, "verifier_pass": False, "classification": "APPROPRIATE"},
            {"attempt_id": 2, "verifier_pass": False, "classification": "APPROPRIATE"},
        ]
        assert select_winning_attempt(calls) is None

    def test_single_passing_attempt_wins(self) -> None:
        calls = [
            {"attempt_id": 1, "verifier_pass": True, "classification": "APPROPRIATE"},
        ]
        winner = select_winning_attempt(calls)
        assert winner is not None
        assert winner["attempt_id"] == 1  # type: ignore[index]


# =============================================================================
# User constraint #10 — Cost guard
#
# WHY: the live Anthropic API must NEVER be called from unit tests.
# Cost regression risk is real (a misconfigured ralph-loop iteration
# can spike thousands of dollars). The guard rejects the production
# transport class identity at the pipeline boundary.
# =============================================================================


class TestCostGuardRejectsLiveTransport:
    def test_live_anthropic_transport_is_rejected(self) -> None:
        """``AnthropicBatchTransport`` is the live HTTP wrapper. The cost
        guard must raise :class:`LiveAnthropicApiError`."""
        live: AnthropicTransport = AnthropicBatchTransport.__new__(
            AnthropicBatchTransport
        )
        with pytest.raises(LiveAnthropicApiError):
            assert_test_safe_transport(live)

    def test_cassette_transport_passes_guard(self) -> None:
        """The cassette transport is the test-safe alternative; the
        guard must pass it through silently."""
        cassette = _cassette_transport()
        # No exception — implementation lands GREEN. RED phase: this
        # currently raises NotImplementedError, which is the correct
        # failure mode for the scaffold.
        assert_test_safe_transport(cassette)


# =============================================================================
# Adversarial fixtures called out in the issue body
# =============================================================================


class TestAdversarialQuoteGrounderRoutesToNeedsReview:
    """A quote_grounder rejection on every Tier-1 citation must surface
    as NEEDS_REVIEW with ``hallucination_suspect`` at the pipeline
    level. The grounder module is independently tested (#18); this
    test pins the orchestration contract."""

    def test_all_citations_rejected_routes_to_needs_review(
        self, tmp_path: object
    ) -> None:
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        # Adversarial verifier: every Tier-1 citation rejected. The
        # winning-attempt rule then surfaces NEEDS_REVIEW with
        # ``hallucination_suspect`` (user constraint #6).
        contexts = _build_adversarial_quote_grounder_contexts()

        def reject_all(result: object, ctx: object) -> bool:
            _ = (result, ctx)
            return False

        run_pipeline(
            contexts,
            transport=_cassette_for_contexts(contexts),
            audit_store=audit_store,
            batch_run_store=InMemoryBatchRunStore(),
            llm_config=_llm_config(),
            pipeline_config=_pipeline_config(),
            run_id="run-adv-grounder",
            verifier=reject_all,
        )
        rows = audit_store.read_audit_results(run_id="run-adv-grounder")
        assert len(rows) == len(contexts)
        assert all(row.final_classification == "NEEDS_REVIEW" for row in rows)
        assert all(row.review_reason == "hallucination_suspect" for row in rows)
        assert all(row.verifier_pass is False for row in rows)


class TestAdversarialVitalsExtractorPropagates:
    """A vitals_extractor failure must not silently default to "vitals
    absent"; the pipeline must propagate the LLM-fallback flag through
    to the audit row so reviewers can see the data quality signal."""

    def test_vitals_extractor_failure_surfaces_in_audit_row(
        self, tmp_path: object
    ) -> None:
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        contexts = _build_adversarial_vitals_contexts()
        run_pipeline(
            contexts,
            transport=_cassette_for_contexts(contexts),
            audit_store=audit_store,
            batch_run_store=InMemoryBatchRunStore(),
            llm_config=_llm_config(),
            pipeline_config=_pipeline_config(),
            run_id="run-adv-vitals",
        )
        rows = audit_store.read_audit_results(run_id="run-adv-vitals")
        # vitals_source must reflect the LLM-fallback provenance, not
        # default to "regex" (which would mask the quality signal).
        # Codex MEDIUM #8: every row asserted (not filtered out).
        assert len(rows) == len(contexts)
        assert all(row.vitals_source is not None for row in rows)
        assert all(
            row.vitals_source == SourceProvenance.LLM_EXTRACTED.value for row in rows
        )


def _build_adversarial_quote_grounder_contexts() -> Sequence[PipelineRowContext]:
    """Build contexts for the adversarial-quote-grounder scenario.

    Every context routes via the LLM (POTENTIALLY_INAPPROPRIATE), and
    the paired verifier (injected into run_pipeline) rejects every
    attempt — the winning-attempt rule then surfaces NEEDS_REVIEW
    with ``hallucination_suspect`` (user constraint #6)."""
    return tuple(
        _row_context(
            audit_id=f"audit-adv-grounder-{i:03d}",
            classification="POTENTIALLY_INAPPROPRIATE",
            hb_value=8.0,
        )
        for i in range(3)
    )


def _build_adversarial_vitals_contexts() -> Sequence[PipelineRowContext]:
    """Build contexts for the adversarial-vitals-extractor scenario.

    The vitals source on every context is :attr:`SourceProvenance.LLM_EXTRACTED`
    so the persisted audit row carries the fallback provenance — the
    pipeline must propagate it verbatim, not default to the regex
    provenance."""
    return tuple(
        _row_context(
            audit_id=f"audit-adv-vitals-{i:03d}",
            classification="POTENTIALLY_INAPPROPRIATE",
            vitals_source=SourceProvenance.LLM_EXTRACTED,
        )
        for i in range(2)
    )


# =============================================================================
# Missing-Hb positive-evidence bypass — pipeline persistence
#
# WHY: when the deterministic classifier fires MTP or peri-procedural
# bypass with hb_result.value_g_dl == None, the resulting APPROPRIATE
# verdict is deterministic-final and must flow through the persistence
# path without the ValueError that previously guarded non-INSUFFICIENT
# classifications from missing Hb. These tests confirm the row lands in
# the audit store with the correct model_id and final_classification.
# =============================================================================


def _missing_hb_context(
    *,
    audit_id: str,
    cohort_label: CohortLabel,
    procedure_proximity_hours: float | None,
    enable_missing_hb_positive_evidence: bool = False,
) -> PipelineRowContext:
    """Build a PipelineRowContext with no Hb measurement.

    The caller controls cohort and procedure proximity to steer the
    classifier into the desired Hb-independent bypass branch. The
    ``enable_missing_hb_positive_evidence`` flag is forwarded onto the
    context so the persistence tests can pin both the SEED-on (bypass
    fires → APPROPRIATE) and SEED-off (bypass dark → INSUFFICIENT_EVIDENCE)
    branches end-to-end.
    """
    order = AuditOrder(
        audit_id=audit_id,
        hn=f"HN-{audit_id}",
        an=f"AN-{audit_id}",
        reqno=f"REQ-{audit_id}",
        order_datetime=_RUN_TS,
        anchor_imputed=False,
        products_ordered=("LPRC",),
        diagnosis_codes=("D62",),
    )
    hb_result = HbLookupResult(
        value_g_dl=None,
        datetime_utc=None,
        source=None,
        freshness="missing",
        delta_hb_bypass=False,
        delta_hb_windows=(
            DeltaHbWindow(
                window_hours=6,
                threshold_g_dl=2.0,
                prior_value_g_dl=None,
                prior_datetime_utc=None,
                drop_g_dl=None,
                triggered=False,
            ),
        ),
        needs_review_single_low_hb=False,
    )
    vitals = VitalsResult(
        vitals=VitalSigns(sbp=110.0, hr=85.0, dbp=70.0, bt=37.0, rr=16.0),
        source=SourceProvenance.IPDADMPROGRESS,
        flags=frozenset(),
        note_timestamp=_RUN_TS,
    )
    cohort = CohortAssignment(
        label=cohort_label,
        threshold=None,
        evidence_code=None,
        evidence_name=None,
    )
    evidence_chunks = (
        EvidenceChunk(
            evidence_id="E1",
            source="IPDNRFOCUSDT",
            text="active MTP protocol, no Hb available",
        ),
    )
    return PipelineRowContext(
        order=order,
        hb_result=hb_result,
        vitals_result=vitals,
        cohort_assignment=cohort,
        procedure_proximity_hours=procedure_proximity_hours,
        crystalloid_liters_prior_4h=0.0,
        hn_hash=f"hn_hash_{audit_id}",
        an_hash=f"an_hash_{audit_id}",
        prior_rbc_units_24h=0,
        prior_rbc_units_7d=0,
        redactor_version="0.4.1+test",
        redactor_model_sha=f"redactor_sha_{audit_id}",
        policy_version="kcmh-pr17.2-2024",
        prompt_hash=f"prompt_hash_{audit_id}",
        evidence_bundle_hash=f"bundle_hash_{audit_id}",
        evidence_chunks=evidence_chunks,
        enable_missing_hb_positive_evidence=enable_missing_hb_positive_evidence,
    )


class TestMissingHbBypassPersistence:
    """APPROPRIATE rows from Hb-independent bypasses must persist without error."""

    def test_mtp_bypass_missing_hb_persists_when_flag_enabled(
        self, tmp_path: object
    ) -> None:
        """MTP bypass with no Hb produces a persisted APPROPRIATE row
        ONLY when the SEED policy flag is on. With the flag enabled,
        _deterministic_audit_row must accept APPROPRIATE + missing Hb
        (the widened guard) and write the 0.0 / freshness=missing
        sentinel row.

        WHY: pre-flag-gating, _deterministic_audit_row raised ValueError
        on every APPROPRIATE-with-missing-Hb row, aborting the run.
        Post-gating, the row must still persist correctly whenever the
        operator has signed off on the bypass (flag=True).
        """
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        ctx = _missing_hb_context(
            audit_id="audit-mtp-missing-hb",
            cohort_label=CohortLabel.MTP,
            procedure_proximity_hours=None,
            enable_missing_hb_positive_evidence=True,
        )
        result = run_pipeline(
            (ctx,),
            transport=_cassette_transport(),
            audit_store=audit_store,
            batch_run_store=InMemoryBatchRunStore(),
            llm_config=_llm_config(),
            pipeline_config=_pipeline_config(),
            run_id="run-mtp-bypass",
        )
        assert result.audit_ids_persisted == ("audit-mtp-missing-hb",)
        rows = list(audit_store.read_audit_results(run_id="run-mtp-bypass"))
        assert len(rows) == 1
        row = rows[0]
        assert row.final_classification == "APPROPRIATE"
        assert row.model_id == "deterministic"
        assert row.hb_value == 0.0
        assert row.hb_freshness == "missing"

    def test_peri_procedural_bypass_missing_hb_persists_when_flag_enabled(
        self, tmp_path: object
    ) -> None:
        """Peri-procedural bypass (≤ 6 h) with no Hb produces a persisted
        APPROPRIATE row when the SEED policy flag is on.

        WHY: same guard as the MTP case — peri-procedural bypass is the
        second Hb-independent path that requires the widened persistence
        guard once the operator has signed off (flag=True).
        """
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        ctx = _missing_hb_context(
            audit_id="audit-peri-proc-missing-hb",
            cohort_label=CohortLabel.DEFAULT,
            procedure_proximity_hours=3.0,
            enable_missing_hb_positive_evidence=True,
        )
        result = run_pipeline(
            (ctx,),
            transport=_cassette_transport(),
            audit_store=audit_store,
            batch_run_store=InMemoryBatchRunStore(),
            llm_config=_llm_config(),
            pipeline_config=_pipeline_config(),
            run_id="run-peri-proc-bypass",
        )
        assert result.audit_ids_persisted == ("audit-peri-proc-missing-hb",)
        rows = list(audit_store.read_audit_results(run_id="run-peri-proc-bypass"))
        assert len(rows) == 1
        row = rows[0]
        assert row.final_classification == "APPROPRIATE"
        assert row.model_id == "deterministic"
        assert row.hb_value == 0.0
        assert row.hb_freshness == "missing"

    def test_mtp_missing_hb_default_off_persists_as_insufficient(
        self, tmp_path: object
    ) -> None:
        """SEED policy default-OFF: MTP + missing Hb routed through the
        full pipeline persists as INSUFFICIENT_EVIDENCE, NOT APPROPRIATE.

        WHY (Codex P1): the bypass is gated until clinical sign-off.
        End-to-end regression: without an operator opt-in on the
        PipelineRowContext, no production run can auto-approve an
        undocumented-Hb case via MTP cohort label alone.
        """
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
        )
        ctx = _missing_hb_context(
            audit_id="audit-mtp-missing-hb-off",
            cohort_label=CohortLabel.MTP,
            procedure_proximity_hours=4.0,
            # enable_missing_hb_positive_evidence defaults to False
        )
        result = run_pipeline(
            (ctx,),
            transport=_cassette_transport(),
            audit_store=audit_store,
            batch_run_store=InMemoryBatchRunStore(),
            llm_config=_llm_config(),
            pipeline_config=_pipeline_config(),
            run_id="run-mtp-bypass-off",
        )
        assert result.audit_ids_persisted == ("audit-mtp-missing-hb-off",)
        rows = list(audit_store.read_audit_results(run_id="run-mtp-bypass-off"))
        assert len(rows) == 1
        row = rows[0]
        assert row.final_classification == "INSUFFICIENT_EVIDENCE"
        assert row.model_id == "deterministic"
        assert row.hb_value == 0.0
        assert row.hb_freshness == "missing"
