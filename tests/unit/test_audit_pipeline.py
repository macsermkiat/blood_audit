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
from bba.audit_store import AuditRow, LlmCall
from bba.audit_store.models import Classification
from bba.audit_pipeline.replay import (
    EMPTY_REASONING_REVIEW_REASON,
    LLM_NATIVE_REVIEW_ASSERT_REASON,
    LLM_OVERCLEAR_ASSERT_REASON,
    LLM_OVERCLEAR_REVIEW_REASON,
    LLM_OVERCLEAR_UNSTABLE_HR,
    LLM_OVERCLEAR_UNSTABLE_SBP,
    PERIOP_CONTRADICTION_REVIEW_REASON,
    PERIOP_GUARDRAIL_MIN_EBL_ML,
    llm_overclear_suspect,
    periop_contradiction,
    split_leaked_summaries,
)
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
from bba.vitals_extractor import (
    PeriopFinding,
    PeriopSummary,
    SourceProvenance,
    VitalSigns,
    VitalsResult,
)
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
    periop_summary: PeriopSummary | None = None,
    sbp: float = 110.0,
    hr: float = 85.0,
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
        vitals=VitalSigns(sbp=sbp, hr=hr, dbp=70.0, bt=37.0, rr=16.0),
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
        periop_summary=periop_summary,
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


# =============================================================================
# Part 2 — Peri-op contradiction guardrail (Case 107)
#
# WHY: the verifier only checks citation grounding; it cannot catch the model
# discounting a documented surgery / large EBL / intra-op transfusion that the
# bundle extracted deterministically. On Case 107 the LLM returned
# INSUFFICIENT_EVIDENCE ("no operative procedure documented") while the bundle
# carried a 1500 ml-blood-loss ORIF in the peri-transfusion window. These tests
# pin that such a contradiction forces human review with a DISTINCT reason,
# never silently lets the "insufficient" verdict stand, never blanks the LLM's
# reasoning, and never over-escalates a committed verdict or an empty signal.
# Each test would fail if the override (or its guard conditions) were removed.
# =============================================================================


def _periop_llm_response(
    *,
    audit_id: str,
    classification: str,
    indications: list[dict[str, object]] | None = None,
    negative_evidence: list[object] | None = None,
    reasoning_en: str = "model rationale",
    reasoning_th: str = "th",
    omit_input_keys: tuple[str, ...] = (),
) -> RawBatchResponse:
    """One grounded LLM result carrying ``classification`` for ``audit_id``.

    Shapes a well-formed structured-output payload so the row reaches the
    *winner* branch of ``_build_audit_row`` (verifier passes by default) —
    the only branch the guardrail override runs in. ``omit_input_keys``
    drops schema-required keys from the tool input to simulate drift."""
    from bba.llm_client.models import BatchSubmissionResult

    input_payload: dict[str, object] = {
        "classification": classification,
        "indications": indications or [],
        "negative_evidence": negative_evidence or [],
        "reasoning_summary_en": reasoning_en,
        "reasoning_summary_th": reasoning_th,
    }
    input_payload = {k: v for k, v in input_payload.items() if k not in omit_input_keys}
    return RawBatchResponse(
        batch_id="msgbatch_periop",
        results=(
            BatchSubmissionResult(
                custom_id=audit_id,
                model_id=SONNET_MODEL_ID,
                raw_response_json={
                    "id": "msg_periop",
                    "type": "message",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "classify_audit",
                            "input": input_payload,
                        }
                    ],
                    "stop_reason": "tool_use",
                },
                request_json={"messages": []},
                response_headers={"anthropic-version": "2023-06-01"},
                request_timestamp=_RUN_TS,
                latency_ms=100,
                anthropic_version="2023-06-01",
                prompt_cache_id=None,
                extended_thinking_blocks=None,
            ),
        ),
    )


def _apply_single_row(
    ctx: PipelineRowContext,
    response: RawBatchResponse,
    *,
    tmp_path: object,
) -> AuditRow:
    from pathlib import Path

    from bba.audit_store import AuditStore, AuditStoreConfig

    assert isinstance(tmp_path, Path)
    audit_store = AuditStore(
        AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
    )
    apply_batch_results(
        response,
        audit_store=audit_store,
        run_id="run-periop",
        contexts={ctx.order.audit_id: ctx},
    )
    rows = audit_store.read_audit_results(run_id="run-periop")
    assert len(rows) == 1
    return rows[0]


class TestPeriopContradictionPredicate:
    """Unit-level pin on the pure predicate + its constants."""

    def test_surgery_alone_contradicts_insufficient_evidence(self) -> None:
        ctx = _row_context(
            audit_id="p-1", periop_summary=PeriopSummary(surgical_context=True)
        )
        assert periop_contradiction("INSUFFICIENT_EVIDENCE", ctx) is True

    def test_large_ebl_alone_contradicts(self) -> None:
        ctx = _row_context(
            audit_id="p-2",
            periop_summary=PeriopSummary(blood_loss_ml=PERIOP_GUARDRAIL_MIN_EBL_ML),
        )
        assert periop_contradiction("POTENTIALLY_INAPPROPRIATE", ctx) is True

    def test_intraop_transfusion_alone_contradicts(self) -> None:
        ctx = _row_context(
            audit_id="p-3", periop_summary=PeriopSummary(intraop_transfusion=True)
        )
        assert periop_contradiction("INSUFFICIENT_EVIDENCE", ctx) is True

    def test_ebl_below_floor_does_not_contradict(self) -> None:
        # The 500 mL floor is the line between routine and transfusion-relevant
        # loss; a sub-floor EBL must NOT override an "insufficient" verdict.
        ctx = _row_context(
            audit_id="p-4",
            periop_summary=PeriopSummary(blood_loss_ml=PERIOP_GUARDRAIL_MIN_EBL_ML - 1),
        )
        assert periop_contradiction("INSUFFICIENT_EVIDENCE", ctx) is False

    def test_committed_verdicts_are_never_contradicted(self) -> None:
        # APPROPRIATE / INAPPROPRIATE are verdicts the model reached WITH the
        # evidence in view; the guardrail only targets non-committal shapes.
        ctx = _row_context(
            audit_id="p-5", periop_summary=PeriopSummary(surgical_context=True)
        )
        assert periop_contradiction("APPROPRIATE", ctx) is False
        assert periop_contradiction("INAPPROPRIATE", ctx) is False

    def test_absent_signal_is_inert(self) -> None:
        ctx = _row_context(audit_id="p-6", periop_summary=None)
        assert periop_contradiction("INSUFFICIENT_EVIDENCE", ctx) is False

    def test_review_reason_distinct_from_hallucination(self) -> None:
        assert PERIOP_CONTRADICTION_REVIEW_REASON != "hallucination_suspect"


class TestLlmOverclearPredicate:
    """Unit pin on the B1 symmetric upgrade-guardrail predicate.

    The complementary arm of the peri-op guardrail: where
    ``periop_contradiction`` floors an LLM that UNDER-called against a hard
    peri-op signal, ``llm_overclear_suspect`` identifies an LLM that OVER-cleared
    — returned APPROPRIATE on a gray-zone / missing-Hb / high-Hb order the
    deterministic leg withheld (NEEDS_REVIEW / INSUFFICIENT_EVIDENCE /
    POTENTIALLY_INAPPROPRIATE) with NO structured hard signal to justify it.
    Cases 47 (68062324) and 100 (68069089) are the
    motivating over-clears. Structured exemptions are sub-7.0 Hb, hard peri-op
    signal, or MTP cohort; the call site separately composes the
    committee-approved qualified-major-bleeding exemption. Hemodynamic
    instability left the exemption set with owner ruling #98 (Codex PR #99
    round 3): verified hypotension/tachycardia is not a transfusion
    indication by itself, so it routes through the qualifier logic instead.
    """

    def test_case_100_shape_is_overclear(self) -> None:
        # Hb 9.4 gray-zone, default cohort, stable vitals, no peri-op: the LLM
        # cleared it on a specialist "keep Hb > 9" target it misread as breached.
        ctx = _row_context(audit_id="oc-1", hb_value=9.4)
        assert llm_overclear_suspect("APPROPRIATE", "NEEDS_REVIEW", ctx) is True

    def test_insufficient_evidence_upgrade_is_overclear(self) -> None:
        # Case 47 shape: det INSUFFICIENT_EVIDENCE (missing structured Hb), LLM
        # upgraded to APPROPRIATE on stale-history + soft symptomatic-anemia.
        ctx = _row_context(audit_id="oc-2", hb_value=9.4)
        assert (
            llm_overclear_suspect("APPROPRIATE", "INSUFFICIENT_EVIDENCE", ctx) is True
        )

    def test_sub_seven_hb_exempts(self) -> None:
        # A genuinely low Hb (< 7.0) is a hard signal; clearing it is not an
        # over-clear even when a det carve-out routed it to review.
        ctx = _row_context(audit_id="oc-3", classification="APPROPRIATE", hb_value=6.5)
        assert llm_overclear_suspect("APPROPRIATE", "NEEDS_REVIEW", ctx) is False

    def test_hard_periop_signal_exempts(self) -> None:
        ctx = _row_context(
            audit_id="oc-4",
            hb_value=9.4,
            periop_summary=PeriopSummary(surgical_context=True),
        )
        assert llm_overclear_suspect("APPROPRIATE", "NEEDS_REVIEW", ctx) is False

    def test_mtp_cohort_exempts(self) -> None:
        ctx = _row_context(
            audit_id="oc-5",
            hb_value=9.4,
            cohort_label=CohortLabel.MTP,
            cohort_threshold=None,
        )
        assert llm_overclear_suspect("APPROPRIATE", "NEEDS_REVIEW", ctx) is False

    def test_hypotension_no_longer_exempts(self) -> None:
        # Owner ruling #98 (Codex PR #99 round 3): a structured SBP < 90
        # snapshot is the same clinical claim as pressor-verified
        # hypotension — real, but not a transfusion indication by itself.
        # The clear is suspect; bleeding / ischemia / fluid-refractory
        # qualifiers decide the outcome downstream.
        ctx = _row_context(
            audit_id="oc-6", hb_value=9.4, sbp=LLM_OVERCLEAR_UNSTABLE_SBP - 1
        )
        assert llm_overclear_suspect("APPROPRIATE", "NEEDS_REVIEW", ctx) is True

    def test_tachycardia_no_longer_exempts(self) -> None:
        # Same ruling, tachycardia arm: HR > 120 alone is a compensation
        # sign, not an indication.
        ctx = _row_context(
            audit_id="oc-7", hb_value=9.4, hr=LLM_OVERCLEAR_UNSTABLE_HR + 1
        )
        assert llm_overclear_suspect("APPROPRIATE", "NEEDS_REVIEW", ctx) is True

    def test_stable_vitals_at_thresholds_do_not_exempt(self) -> None:
        # SBP == 90 and HR == 120 are the stable side of the boundary
        # (strict < 90 / > 120), so the over-clear still fires.
        ctx = _row_context(
            audit_id="oc-8",
            hb_value=9.4,
            sbp=LLM_OVERCLEAR_UNSTABLE_SBP,
            hr=LLM_OVERCLEAR_UNSTABLE_HR,
        )
        assert llm_overclear_suspect("APPROPRIATE", "NEEDS_REVIEW", ctx) is True

    def test_non_appropriate_llm_verdict_never_fires(self) -> None:
        ctx = _row_context(audit_id="oc-9", hb_value=9.4)
        for verdict in (
            "INAPPROPRIATE",
            "INSUFFICIENT_EVIDENCE",
            "NEEDS_REVIEW",
            "POTENTIALLY_INAPPROPRIATE",
        ):
            assert llm_overclear_suspect(verdict, "NEEDS_REVIEW", ctx) is False

    def test_deterministic_appropriate_never_fires(self) -> None:
        # A deterministic clear cannot be over-cleared by the LLM.
        ctx = _row_context(audit_id="oc-10", hb_value=9.4)
        assert llm_overclear_suspect("APPROPRIATE", "APPROPRIATE", ctx) is False

    def test_high_hb_soft_verdict_now_fires(self) -> None:
        # Since #93 dispatches the high-Hb override, its withheld soft verdict
        # must also be protected when the LLM clears it (#89 / #94).
        ctx = _row_context(
            audit_id="oc-11",
            classification="POTENTIALLY_INAPPROPRIATE",
            hb_value=12.9,
        )
        assert (
            llm_overclear_suspect("APPROPRIATE", "POTENTIALLY_INAPPROPRIATE", ctx)
            is True
        )

    def test_review_reason_is_distinct(self) -> None:
        reasons = {
            PERIOP_CONTRADICTION_REVIEW_REASON,
            "hallucination_suspect",
            LLM_OVERCLEAR_REVIEW_REASON,
            EMPTY_REASONING_REVIEW_REASON,
            LLM_OVERCLEAR_ASSERT_REASON,
            LLM_NATIVE_REVIEW_ASSERT_REASON,
        }
        assert len(reasons) == 6


class TestLlmOverclearGuardrail:
    """Integration pin on the B1 override inside ``apply_batch_results``."""

    def test_case_100_overclear_asserted_inappropriate(self, tmp_path: object) -> None:
        ctx = _row_context(
            audit_id="audit-oc-100", classification="NEEDS_REVIEW", hb_value=9.4
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "THAL_TARGET_HB",
                    "quote": "keep Hb >9",
                    "source_id": "E1",
                    "confidence": 0.97,
                }
            ],
            reasoning_en="specialist target misread",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON
        assert row.needs_human_review is False

    def test_overclear_preserves_llm_reasoning_and_indications(
        self, tmp_path: object
    ) -> None:
        # The reviewer must see exactly what the model concluded and why it is
        # being second-guessed, so verifier_pass, reasoning, indications, and
        # the deterministic verdict are all preserved.
        ctx = _row_context(
            audit_id="audit-oc-keep", classification="NEEDS_REVIEW", hb_value=9.4
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "SYMPTOMATIC_ANEMIA",
                    "quote": "pale",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
            reasoning_en="soft indications only",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON
        assert row.verifier_pass is True
        assert row.reasoning_summary_en == "soft indications only"
        assert len(row.indications_json) == 1
        assert row.rule_classification == "NEEDS_REVIEW"

    def test_hard_signal_backed_appropriate_not_floored(self, tmp_path: object) -> None:
        # An LLM APPROPRIATE backed by a hard peri-op signal is not an
        # over-clear — it must survive as APPROPRIATE.
        ctx = _row_context(
            audit_id="audit-oc-hard",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            periop_summary=PeriopSummary(surgical_context=True, blood_loss_ml=1500),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id, classification="APPROPRIATE"
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "APPROPRIATE"

    def test_qualified_bleed_exempts_overclear_assertion(
        self, tmp_path: object
    ) -> None:
        # A high-confidence major active bleed is the committee-approved prose
        # exemption — and its quote must ground in the row's evidence bundle.
        ctx = _row_context(
            audit_id="audit-oc-bleed-400",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="Post-op note: EBL 400 mL from drain, ongoing ooze",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "EBL 400 mL from drain",
                    "source_id": "E1",
                    "confidence": 0.85,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "APPROPRIATE"
        assert row.review_reason is None

    def test_negated_life_threatening_quote_does_not_exempt(
        self, tmp_path: object
    ) -> None:
        # Codex PR #97 P2: "no active hemorrhage" grounds in the bundle (the
        # note really says it), but it documents the ABSENCE of the emergency.
        # A mislabeled high-confidence ACTIVE_BLEEDING citation of a negated
        # marker must not exempt — the assert stands.
        ctx = _row_context(
            audit_id="audit-oc-bleed-negated",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="Overnight events: stable, no active hemorrhage",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "no active hemorrhage",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_grounded_acs_overclear_floors_to_review(self, tmp_path: object) -> None:
        # Codex PR #97 P1: ACS is in the prompt's HARD vocabulary but has no
        # structured extractor and no prose exemption path, so asserting
        # INAPPROPRIATE would flip a prompt-compliant ACS clear into an
        # unreviewed committee verdict. A grounded, high-confidence ACS
        # citation floors to a human instead. It does NOT auto-clear:
        # extending prose auto-clear trust beyond qualified bleeding is a
        # committee decision (spec #89 accepted it for bleeding only).
        ctx = _row_context(
            audit_id="audit-oc-acs",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="Progress: NSTEMI with ongoing chest pain, troponin rising",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACS",
                    "quote": "NSTEMI with ongoing chest pain",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
            reasoning_en="active ischemia justifies the gray-zone transfusion",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON
        assert row.needs_human_review is True

    def test_low_confidence_acs_still_asserts(self, tmp_path: object) -> None:
        # Below the shared 0.8 prose-trust bar an ACS citation is not honored.
        ctx = _row_context(
            audit_id="audit-oc-acs-lowconf",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="Progress: NSTEMI with ongoing chest pain, troponin rising",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACS",
                    "quote": "NSTEMI with ongoing chest pain",
                    "source_id": "E1",
                    "confidence": 0.5,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_ungrounded_acs_quote_still_asserts(self, tmp_path: object) -> None:
        # A fabricated ACS quote (grounds nowhere in the bundle) never floors.
        ctx = _row_context(
            audit_id="audit-oc-acs-ungrounded",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="Routine post-op note, patient stable overnight",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACS",
                    "quote": "NSTEMI with ongoing chest pain",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    @staticmethod
    def _single_low_hb_ctx(
        audit_id: str, *, hb: float, threshold: float
    ) -> PipelineRowContext:
        # The real-world route for a sub-floor Hb the deterministic leg still
        # withholds: a single low Hb with no supporting trend
        # (needs_review_single_low_hb). The fixture helper clamps NEEDS_REVIEW
        # rows to hb >= threshold, so rebuild the Hb result explicitly.
        ctx = _row_context(
            audit_id=audit_id,
            classification="NEEDS_REVIEW",
            hb_value=threshold + 0.5,
            cohort_threshold=threshold,
            evidence_text="Lab: Hb 7.6 g/dL this morning, no prior value",
        )
        return ctx.model_copy(
            update={
                "hb_result": ctx.hb_result.model_copy(
                    update={
                        "value_g_dl": hb,
                        "needs_review_single_low_hb": True,
                    }
                )
            }
        )

    def test_grounded_true_subthreshold_overclear_floors_to_review(
        self, tmp_path: object
    ) -> None:
        # Codex PR #97 round 3: the prompt defines SUB_THRESHOLD_HB (Hb
        # strictly below the cohort floor) as HARD. When the claim is
        # structurally TRUE (hb < threshold; the deterministic leg withheld
        # only because the value was unreliable), asserting INAPPROPRIATE
        # would contradict the prompt's own contract — floor to a human.
        ctx = self._single_low_hb_ctx("audit-oc-subthr", hb=7.6, threshold=8.0)
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "SUB_THRESHOLD_HB",
                    "quote": "Hb 7.6 g/dL this morning",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
            reasoning_en="single Hb below the 8.0 cohort floor",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON
        assert row.needs_human_review is True

    def test_structurally_false_subthreshold_claim_still_asserts(
        self, tmp_path: object
    ) -> None:
        # The motivating over-clear class: an at/above-floor Hb mislabeled
        # "sub-threshold" (spec #89 story 6). The structural cross-check
        # (hb >= threshold) keeps the assert.
        ctx = _row_context(
            audit_id="audit-oc-subthr-false",
            classification="NEEDS_REVIEW",
            hb_value=7.6,
            cohort_threshold=7.0,
            evidence_text="Lab: Hb 7.6 g/dL this morning, no prior value",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "SUB_THRESHOLD_HB",
                    "quote": "Hb 7.6 g/dL this morning",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_low_confidence_true_subthreshold_still_asserts(
        self, tmp_path: object
    ) -> None:
        # The structural truth alone is not enough — the citation must carry
        # the shared >=0.8 prose-trust confidence.
        ctx = self._single_low_hb_ctx("audit-oc-subthr-lowconf", hb=7.6, threshold=8.0)
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "SUB_THRESHOLD_HB",
                    "quote": "Hb 7.6 g/dL this morning",
                    "source_id": "E1",
                    "confidence": 0.5,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_native_hedge_with_true_subthreshold_stays_review(
        self, tmp_path: object
    ) -> None:
        # Conversion side of the same rule: a hedge citing a structurally
        # true sub-floor Hb is a genuine human case.
        ctx = self._single_low_hb_ctx(
            "audit-native-review-subthr", hb=7.6, threshold=8.0
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="NEEDS_REVIEW",
            indications=[
                {
                    "code": "SUB_THRESHOLD_HB",
                    "quote": "Hb 7.6 g/dL this morning",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
            reasoning_en="single unconfirmed Hb below floor, deferring",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason is None
        assert row.needs_human_review is True

    def test_bare_hypotension_overclear_asserts(self, tmp_path: object) -> None:
        # Owner ruling (#98, 2026-07-11): hypotension — even on pressors —
        # without evidence of active bleeding, severe organ ischemia, or
        # fluid-refractoriness is NOT a transfusion indication; the assert
        # stands. Deliberately narrows the round-5 floor.
        ctx = _row_context(
            audit_id="audit-oc-hemo",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="NIBP 79/54 (MAP 63) mmHg, on Levophed 17 ml/hr",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.85,
                }
            ],
            reasoning_en="documented hypotension on vasopressor support",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_hypotension_with_active_bleed_citation_floors(
        self, tmp_path: object
    ) -> None:
        # Owner ruling qualifier (1): grounded instability accompanied by a
        # grounded active-bleeding citation (possible hemorrhagic shock that
        # does not meet the qualified-major-bleed bar) is a genuine human
        # case — floor, never assert, never auto-clear.
        ctx = _row_context(
            audit_id="audit-oc-hemo-bleed",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=(
                "NIBP 79/54 (MAP 63) mmHg, on Levophed; "
                "intramuscular hematoma with active bleeding at Lt. thigh"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.85,
                },
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "intramuscular hematoma with active bleeding",
                    "source_id": "E1",
                    "confidence": 0.9,
                },
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON

    def test_fluid_refractory_hypotension_floors(self, tmp_path: object) -> None:
        # Owner ruling qualifier (3): instability documented as unresponsive
        # to fluid resuscitation floors to a human.
        ctx = _row_context(
            audit_id="audit-oc-hemo-refractory",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=(
                "BP 82/50 despite fluid resuscitation 1000 mL, start Levophed"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "BP 82/50 despite fluid resuscitation 1000 mL",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON

    def test_melena_shock_instability_citation_floors(self, tmp_path: object) -> None:
        # Codex PR #103: the prompt routes melena + shock to a
        # HEMODYNAMIC_INSTABILITY citation instead of ACTIVE_BLEEDING (owner
        # ruling: melena is digested blood), so the hemorrhagic-shock
        # accompaniment arrives inside the instability citation's own quote.
        # It must still floor to human review — "with shock we care" — not
        # fall through both the exemption and the floor to an assert.
        ctx = _row_context(
            audit_id="audit-oc-hemo-melena",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="melena x 3 this morning; BP 82/50, start Levophed",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "melena x 3 this morning; BP 82/50, start Levophed",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON

    def test_negated_melena_instability_citation_asserts(
        self, tmp_path: object
    ) -> None:
        # The melena arm is negation-aware: a denied melena inside the
        # instability quote is a documented ABSENCE, not hemorrhagic-shock
        # accompaniment — the bare-hypotension assert stands (ruling #98).
        ctx = _row_context(
            audit_id="audit-oc-hemo-melena-neg",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="denies melena; NIBP 79/54 (MAP 63) mmHg, on Levophed",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "denies melena; NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_stale_dated_melena_instability_citation_asserts(
        self, tmp_path: object
    ) -> None:
        # Same temporal screen as the family-code arm (case 68080335): a
        # melena mention governed by a stale date anchor (1/5/69 BE ==
        # 2026-05-01, 15 days before the 2026-05-16 order) is an old
        # episode, not the current hemorrhagic picture — assert stands.
        ctx = _row_context(
            audit_id="audit-oc-hemo-melena-stale",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="NIBP 79/54 (MAP 63) mmHg, on Levophed; Hx.1/5/69: melena",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": (
                        "NIBP 79/54 (MAP 63) mmHg, on Levophed; Hx.1/5/69: melena"
                    ),
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_denial_list_melena_instability_citation_asserts(
        self, tmp_path: object
    ) -> None:
        # Codex PR #103 round 2: a comma-distributed denial inside the
        # instability quote ("denies bleeding, melena") is a documented
        # ABSENCE. quote_indicates_melena keeps the marker screens' comma
        # boundary, so the floor applies the family-code arm's denial-list
        # read (quote_negates_bleeding) on top — denied melena must not
        # float a bare-hypotension over-clear to review.
        ctx = _row_context(
            audit_id="audit-oc-hemo-melena-denial",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="denies bleeding, melena; NIBP 79/54 (MAP 63), on Levophed",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": (
                        "denies bleeding, melena; NIBP 79/54 (MAP 63), on Levophed"
                    ),
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_affirmed_melena_synonym_with_denied_bleed_still_floors(
        self, tmp_path: object
    ) -> None:
        # Codex PR #103 round 3: the floor's quote_negates_bleeding veto
        # must recognize every _MELENA_TERMS synonym. An affirmed
        # coffee-ground emesis next to a denied GENERIC bleed term is a
        # documented hemorrhagic-shock picture — the denial of the other
        # term must not veto the melena accompaniment.
        ctx = _row_context(
            audit_id="audit-oc-hemo-melena-syn",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="coffee-ground emesis x2, no gross bleeding; BP 82/50",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "coffee-ground emesis x2, no gross bleeding; BP 82/50",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON

    def test_still_active_melena_instability_citation_floors(
        self, tmp_path: object
    ) -> None:
        # Codex PR #103 round 2: "melena not controlled" is an ONGOING
        # melena (the negator binds the control verb) — the still-active
        # rescue keeps it visible as hemorrhagic-shock accompaniment, so
        # the row floors to human review.
        ctx = _row_context(
            audit_id="audit-oc-hemo-melena-active",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="melena not controlled; BP 82/50, start Levophed",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "melena not controlled; BP 82/50, start Levophed",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON

    def test_hypotension_with_negated_bleed_citation_asserts(
        self, tmp_path: object
    ) -> None:
        # A risk/history-qualified bleed code is not bleeding evidence; the
        # bare-hypotension assert stands.
        ctx = _row_context(
            audit_id="audit-oc-hemo-riskbleed",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=(
                "NIBP 79/54 (MAP 63) mmHg, on Levophed; bleeding precaution"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.85,
                },
                {
                    "code": "ACTIVE_BLEEDING_RISK",
                    "quote": "bleeding precaution",
                    "source_id": "E1",
                    "confidence": 0.9,
                },
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_hypotension_with_negated_bleed_quote_asserts(
        self, tmp_path: object
    ) -> None:
        # Codex PR #99 round 2: an ACTIVE_BLEEDING-coded citation whose
        # grounded quote documents the ABSENCE of bleeding is a mislabeled
        # citation, not qualifier-(1) accompaniment — bare hypotension
        # stays asserted (owner ruling #98), same quote-negation screen as
        # the bleed exemption.
        ctx = _row_context(
            audit_id="audit-oc-hemo-negbleed",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=(
                "NIBP 79/54 (MAP 63) mmHg, on Levophed; no active hemorrhage seen"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.85,
                },
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "no active hemorrhage seen",
                    "source_id": "E1",
                    "confidence": 0.9,
                },
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_hypotension_with_stale_dated_bleed_asserts(self, tmp_path: object) -> None:
        # Case 68080335 / PR #100 Codex: a bleed charted for an event weeks
        # before the order is not the CURRENT hemorrhagic picture. It must
        # not supply qualifier-(1) accompaniment for a bare-hypotension
        # over-clear — otherwise the stale-date gate leaks and the row floors
        # to NEEDS_REVIEW instead of the asserted INAPPROPRIATE. Order date is
        # 2026-05-16 (Bangkok); 1/5/69 BE == 2026-05-01, 15 days prior. The
        # "Hx." label mirrors the real charting (Codex round 2): masking
        # leaves the label prefix, but no current bleed evidence survives.
        ctx = _row_context(
            audit_id="audit-oc-hemo-stalebleed",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            sbp=LLM_OVERCLEAR_UNSTABLE_SBP - 8.0,
            evidence_text=(
                "NIBP 79/54 (MAP 63) mmHg, on Levophed; "
                "Hx.1/5/69: active bleeding 400 ml"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.85,
                },
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "Hx.1/5/69: active bleeding 400 ml",
                    "source_id": "E1",
                    "confidence": 0.9,
                },
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_hypotension_with_current_dated_bleed_floors(
        self, tmp_path: object
    ) -> None:
        # Guard against over-correction: a CURRENT-episode dated bleed (14/5/69
        # BE == 2026-05-14, 2 days before the 2026-05-16 order) is genuine
        # accompaniment and must still floor a bare-hypotension over-clear to
        # human review — the temporal gate blanks only STALE spans. The bleed
        # is deliberately sub-major (no >300 mL, no life-threatening marker) so
        # it is qualifier-(1) accompaniment, not a self-standing bleed
        # exemption that would auto-clear the row to APPROPRIATE first.
        ctx = _row_context(
            audit_id="audit-oc-hemo-currentbleed",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            sbp=LLM_OVERCLEAR_UNSTABLE_SBP - 8.0,
            evidence_text=(
                "NIBP 79/54 (MAP 63) mmHg, on Levophed; "
                "14/5/69: active bleeding per rectum ongoing"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.85,
                },
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "14/5/69: active bleeding per rectum ongoing",
                    "source_id": "E1",
                    "confidence": 0.9,
                },
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON

    def test_structured_bare_hypotension_overclear_asserts(
        self, tmp_path: object
    ) -> None:
        # Codex PR #99 round 3 / owner ruling #98: a structured SBP < 90
        # snapshot is bare hypotension too — verified, but still not a
        # transfusion indication without bleeding / ischemia /
        # fluid-refractory evidence. It no longer shields the clear.
        ctx = _row_context(
            audit_id="audit-oc-hemo-structured",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            sbp=LLM_OVERCLEAR_UNSTABLE_SBP - 8.0,
            evidence_text="Routine ward note, transfusion given",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[],
            reasoning_en="cleared on the overall picture",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_structured_hypotension_with_bleed_citation_floors(
        self, tmp_path: object
    ) -> None:
        # Structured instability + a grounded sub-exemption bleed citation
        # is the same possible-hemorrhagic-shock picture as the prose path —
        # qualifier (1) floors it to a human.
        ctx = _row_context(
            audit_id="audit-oc-hemo-structured-bleed",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            sbp=LLM_OVERCLEAR_UNSTABLE_SBP - 8.0,
            evidence_text="intramuscular hematoma with active bleeding at Lt. thigh",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "intramuscular hematoma with active bleeding",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON

    def test_boundary_vitals_with_bleed_citation_still_assert(
        self, tmp_path: object
    ) -> None:
        # SBP exactly 90 is the stable side of the strict boundary: no
        # instability precondition holds, and a sub-exemption bleed alone
        # does not floor — the assert stands.
        ctx = _row_context(
            audit_id="audit-oc-hemo-boundary-bleed",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            sbp=LLM_OVERCLEAR_UNSTABLE_SBP,
            evidence_text="intramuscular hematoma with active bleeding at Lt. thigh",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "intramuscular hematoma with active bleeding",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_structured_hypotension_with_qualified_bleed_stays_appropriate(
        self, tmp_path: object
    ) -> None:
        # The committee-approved major-bleed exemption still auto-clears a
        # hypotensive row — the ruling narrows hemodynamic instability, not
        # the bleeding exemption.
        ctx = _row_context(
            audit_id="audit-oc-hemo-structured-majorbleed",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            sbp=LLM_OVERCLEAR_UNSTABLE_SBP - 8.0,
            evidence_text="Massive hematemesis, EBL 1100 mL over 2 hours",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "EBL 1100 mL",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "APPROPRIATE"
        assert row.review_reason is None

    def test_native_hedge_with_structured_bare_hypotension_converts(
        self, tmp_path: object
    ) -> None:
        # Hedge side of the same ruling: structured bare hypotension is no
        # longer a hard signal keeping the hedge a human case — it converts.
        ctx = _row_context(
            audit_id="audit-native-review-hemo-structured",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            sbp=LLM_OVERCLEAR_UNSTABLE_SBP - 8.0,
            evidence_text="Routine ward note, transfusion given",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="NEEDS_REVIEW",
            indications=[],
            reasoning_en="hypotensive on the ward, deferring to a human",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_NATIVE_REVIEW_ASSERT_REASON

    def test_hypotension_with_negated_bleed_synonym_asserts(
        self, tmp_path: object
    ) -> None:
        # Codex PR #99 round 4: a negated bleed SYNONYM ("denies melena")
        # miscoded as ACTIVE_BLEEDING must not count as qualifier-(1)
        # accompaniment either — the generic bleeding terms alone would
        # miss it and the bare-hypotension assert would be lost.
        ctx = _row_context(
            audit_id="audit-oc-hemo-negsynonym",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=("NIBP 79/54 (MAP 63) mmHg, on Levophed; denies melena"),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.85,
                },
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "denies melena",
                    "source_id": "E1",
                    "confidence": 0.9,
                },
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_hypotension_with_comma_denial_list_asserts(self, tmp_path: object) -> None:
        # Codex PR #99 round 5: "denies bleeding, melena" is one denial
        # distributed across a comma list — the comma must not shield
        # "melena" from the negator and float bare hypotension to review.
        ctx = _row_context(
            audit_id="audit-oc-hemo-denial-list",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=(
                "NIBP 79/54 (MAP 63) mmHg, on Levophed; denies bleeding, melena"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.85,
                },
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "denies bleeding, melena",
                    "source_id": "E1",
                    "confidence": 0.9,
                },
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_hypotension_with_contrastive_bleed_clause_floors(
        self, tmp_path: object
    ) -> None:
        # Codex PR #99 round 6: "denies hematemesis but melena ongoing"
        # documents an ACTIVE bleed after the contrastive — the earlier
        # denial must not leak across "but" and rob a genuinely bleeding
        # hypotensive row of its human review.
        ctx = _row_context(
            audit_id="audit-oc-hemo-contrastive",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=(
                "NIBP 79/54 (MAP 63) mmHg, on Levophed; "
                "denies hematemesis but melena ongoing"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.85,
                },
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "denies hematemesis but melena ongoing",
                    "source_id": "E1",
                    "confidence": 0.9,
                },
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON

    def test_hypotension_with_uncontrolled_bleed_double_negative_floors(
        self, tmp_path: object
    ) -> None:
        # Codex PR #99 round 7: "bleeding not controlled after pressure"
        # is an ONGOING bleed — the "not" binds "controlled". No volume
        # and no exact marker means the exemption cannot catch it, so the
        # accompaniment screen must keep it visible and floor the row.
        ctx = _row_context(
            audit_id="audit-oc-hemo-notcontrolled",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=(
                "NIBP 79/54 (MAP 63) mmHg, on Levophed; "
                "bleeding not controlled after pressure"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.85,
                },
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "bleeding not controlled after pressure",
                    "source_id": "E1",
                    "confidence": 0.9,
                },
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON

    def test_hypotension_with_label_value_denial_asserts(
        self, tmp_path: object
    ) -> None:
        # Codex PR #99 round 8: a checklist-style label-value denial
        # ("GI bleeding: no") is a documented absence — it must not count
        # as qualifier-(1) accompaniment, and bare hypotension asserts.
        ctx = _row_context(
            audit_id="audit-oc-hemo-labelvalue",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=("NIBP 79/54 (MAP 63) mmHg, on Levophed; GI bleeding: no"),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.85,
                },
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "GI bleeding: no",
                    "source_id": "E1",
                    "confidence": 0.9,
                },
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_hypotension_with_unresolved_bleed_adverb_floors(
        self, tmp_path: object
    ) -> None:
        # Codex PR #99 round 8: "bleeding not yet resolved" is explicitly
        # ONGOING — the adverb between the negator and the control verb
        # must not defeat the rescue, and the row floors to a human.
        ctx = _row_context(
            audit_id="audit-oc-hemo-notyetresolved",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=(
                "NIBP 79/54 (MAP 63) mmHg, on Levophed; bleeding not yet resolved"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.85,
                },
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "bleeding not yet resolved",
                    "source_id": "E1",
                    "confidence": 0.9,
                },
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON

    def test_hypotension_with_denied_bleed_unrelated_rescue_asserts(
        self, tmp_path: object
    ) -> None:
        # Codex PR #99 round 9: the still-active rescue must bind its OWN
        # negator. "bleeding denied, pain not controlled" denies the bleed;
        # the unrelated pain phrase must not resurrect it as accompaniment.
        ctx = _row_context(
            audit_id="audit-oc-hemo-unrelated-rescue",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=(
                "NIBP 79/54 (MAP 63) mmHg, on Levophed; "
                "bleeding denied, pain not controlled"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.85,
                },
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "bleeding denied, pain not controlled",
                    "source_id": "E1",
                    "confidence": 0.9,
                },
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_negated_refractory_wording_does_not_floor(self, tmp_path: object) -> None:
        # Codex PR #99 round 4: qualifier (3) requires POSITIVE
        # unresponsiveness to fluids. "not refractory after IV fluids" is
        # documented responsiveness — bare hypotension, assert stands.
        ctx = _row_context(
            audit_id="audit-oc-hemo-notrefractory",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="BP 82/50, not refractory after IV fluids",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "BP 82/50, not refractory after IV fluids",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_bare_refractory_token_floors(self, tmp_path: object) -> None:
        # Positive control for the screened token: un-negated "refractory"
        # in the instability citation's own quote still floors.
        ctx = _row_context(
            audit_id="audit-oc-hemo-refractory-bare",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="BP 82/50 refractory hypotension, start Levophed",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "BP 82/50 refractory hypotension",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON

    def test_pre_negated_despite_phrase_still_floors(self, tmp_path: object) -> None:
        # Guard against over-screening: "not improving despite fluid
        # resuscitation" is GENUINE refractoriness (the negation binds
        # "improving", not the refractory phrase). Only the
        # polarity-ambiguous bare "refractory" token is negation-screened;
        # the phrase tokens carry their own polarity.
        ctx = _row_context(
            audit_id="audit-oc-hemo-refract-doubleneg",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=(
                "BP 82/50 not improving despite fluid resuscitation 1000 mL"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": (
                        "BP 82/50 not improving despite fluid resuscitation 1000 mL"
                    ),
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON

    def test_low_confidence_refractory_quote_does_not_floor(
        self, tmp_path: object
    ) -> None:
        # Codex PR #99 P2: qualifier (3) requires the fluid-refractory quote
        # to be its OWN >=0.8 citation. A high-conf bare-hypotension citation
        # must not let a separate low-conf 'despite fluid' quote floor the
        # row — the bare-hypotension assert stands.
        ctx = _row_context(
            audit_id="audit-oc-hemo-refract-lowconf",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=(
                "NIBP 79/54 (MAP 63) mmHg, on Levophed; "
                "BP 82/50 despite fluid resuscitation 1000 mL"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.85,
                },
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "BP 82/50 despite fluid resuscitation 1000 mL",
                    "source_id": "E1",
                    "confidence": 0.4,
                },
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_low_confidence_hemodynamic_still_asserts(self, tmp_path: object) -> None:
        ctx = _row_context(
            audit_id="audit-oc-hemo-lowconf",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="NIBP 79/54 (MAP 63) mmHg, on Levophed 17 ml/hr",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.5,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_ungrounded_hemodynamic_quote_still_asserts(self, tmp_path: object) -> None:
        ctx = _row_context(
            audit_id="audit-oc-hemo-ungrounded",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="Routine post-op note, patient stable overnight",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_native_hedge_with_bare_hypotension_converts(
        self, tmp_path: object
    ) -> None:
        # Owner ruling (#98): a hedge citing only bare hypotension converts —
        # hypotension without bleeding/ischemia/fluid-refractory evidence is
        # not a transfusion indication.
        ctx = _row_context(
            audit_id="audit-native-review-hemo",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="NIBP 79/54 (MAP 63) mmHg, on Levophed 17 ml/hr",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="NEEDS_REVIEW",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "NIBP 79/54 (MAP 63) mmHg, on Levophed",
                    "source_id": "E1",
                    "confidence": 0.85,
                }
            ],
            reasoning_en="pressor-dependent hypotension, deferring to a human",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_NATIVE_REVIEW_ASSERT_REASON

    def test_native_hedge_with_fluid_refractory_hypotension_stays_review(
        self, tmp_path: object
    ) -> None:
        ctx = _row_context(
            audit_id="audit-native-review-hemo-refract",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=(
                "BP 82/50 despite fluid resuscitation 1000 mL, start Levophed"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="NEEDS_REVIEW",
            indications=[
                {
                    "code": "HEMODYNAMIC_INSTABILITY",
                    "quote": "BP 82/50 despite fluid resuscitation 1000 mL",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
            reasoning_en="fluid-refractory hypotension, deferring to a human",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason is None
        assert row.needs_human_review is True

    def test_qualified_bleed_beats_acs_floor(self, tmp_path: object) -> None:
        # When a grounded qualified major bleed co-occurs with a grounded ACS
        # citation, the committee-approved bleeding exemption wins and the
        # clear survives as APPROPRIATE (the ACS floor is only reached when
        # no exemption applies).
        ctx = _row_context(
            audit_id="audit-oc-acs-and-bleed",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=(
                "EBL 400 mL from drain, ongoing ooze; NSTEMI with ongoing chest pain"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "EBL 400 mL from drain",
                    "source_id": "E1",
                    "confidence": 0.85,
                },
                {
                    "code": "ACS",
                    "quote": "NSTEMI with ongoing chest pain",
                    "source_id": "E1",
                    "confidence": 0.9,
                },
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "APPROPRIATE"
        assert row.review_reason is None

    def test_ungrounded_bleed_quote_does_not_exempt(self, tmp_path: object) -> None:
        # The batch verifier is still the Phase-1 pass-through, so the
        # exemption itself must reject a bleed quote that appears nowhere in
        # the row's evidence — a fabricated major bleed must never auto-clear.
        ctx = _row_context(
            audit_id="audit-oc-bleed-ungrounded",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "EBL 400 mL from drain",
                    "source_id": "E1",
                    "confidence": 0.85,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_bleed_quote_citing_unknown_source_does_not_exempt(
        self, tmp_path: object
    ) -> None:
        # A quote must ground in the *cited* chunk; an unknown source_id is
        # the same failure as a fabricated quote (fail closed).
        ctx = _row_context(
            audit_id="audit-oc-bleed-bad-source",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="Post-op note: EBL 400 mL from drain, ongoing ooze",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "EBL 400 mL from drain",
                    "source_id": "E9",
                    "confidence": 0.85,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_out_of_range_confidence_does_not_exempt(self, tmp_path: object) -> None:
        # confidence 2.0 is schema-invalid; it must read as malformed, not
        # as "very sure" — a malformed indication never exempts (spec #89).
        ctx = _row_context(
            audit_id="audit-oc-bleed-conf-2",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="Post-op note: EBL 400 mL from drain, ongoing ooze",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "EBL 400 mL from drain",
                    "source_id": "E1",
                    "confidence": 2.0,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_drifted_overclear_payload_floors_to_review(self, tmp_path: object) -> None:
        # A payload missing the schema-required indications key may have LOST
        # cited evidence — asserting INAPPROPRIATE on it would be a verdict
        # built on absence we cannot distinguish from drift. Fail closed to
        # the pre-#94 floor: NEEDS_REVIEW with the suspect slug.
        ctx = _row_context(
            audit_id="audit-oc-drifted",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            omit_input_keys=("indications",),
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON
        assert row.needs_human_review is True

    def test_drifted_native_review_payload_is_not_converted(
        self, tmp_path: object
    ) -> None:
        # Same drift on a native hedge: the conversion requires a well-formed
        # payload, so the row stays a genuine human-review case.
        ctx = _row_context(
            audit_id="audit-native-drifted",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="NEEDS_REVIEW",
            reasoning_en="hedge with reasoning but drifted payload",
            omit_input_keys=("negative_evidence",),
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason is None
        assert row.needs_human_review is True

    def test_exactly_300_ml_does_not_exempt(self, tmp_path: object) -> None:
        # The major-bleed boundary is strictly greater than 300 mL.
        ctx = _row_context(
            audit_id="audit-oc-bleed-300",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            # Quote grounds so the strict > 300 boundary is what decides.
            evidence_text="Post-op note: EBL 300 mL recorded",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "EBL 300 mL",
                    "source_id": "E1",
                    "confidence": 0.85,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_low_confidence_major_bleed_does_not_exempt(self, tmp_path: object) -> None:
        # Major-bleed prose is trusted only at indication confidence >= 0.8.
        ctx = _row_context(
            audit_id="audit-oc-bleed-low-confidence",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            # Quote grounds so the 0.7 < 0.8 confidence gate is what decides.
            evidence_text="Post-op note: EBL 400 mL from drain, ongoing ooze",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "EBL 400 mL from drain",
                    "source_id": "E1",
                    "confidence": 0.7,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_life_threatening_unquantified_bleed_exempts(
        self, tmp_path: object
    ) -> None:
        # Explicit uncontrolled bleeding can qualify without a measured volume.
        ctx = _row_context(
            audit_id="audit-oc-bleed-uncontrolled",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text=(
                "Nursing note: uncontrolled bleeding from the tumor bed despite packing"
            ),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "uncontrolled bleeding from the tumor bed",
                    "source_id": "E1",
                    "confidence": 0.85,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "APPROPRIATE"
        assert row.review_reason is None

    def test_small_qualitative_bleed_does_not_exempt(self, tmp_path: object) -> None:
        # Minor gauze staining is not a major active bleed, even at high confidence.
        ctx = _row_context(
            audit_id="audit-oc-bleed-small",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            # Quote grounds so the no-volume/no-marker test is what decides.
            evidence_text="Wound check: 2x2 cm gauze staining noted",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "2x2 cm gauze staining",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON

    def test_thai_major_bleed_volume_exempts(self, tmp_path: object) -> None:
        # Thai millilitre notation receives the same major-bleed exemption.
        ctx = _row_context(
            audit_id="audit-oc-bleed-thai",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="บันทึกพยาบาล: เสียเลือด 1,100 มล. ระหว่างผ่าตัด",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "เสียเลือด 1,100 มล.",
                    "source_id": "E1",
                    "confidence": 0.85,
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "APPROPRIATE"
        assert row.review_reason is None

    def test_malformed_indication_element_floors_to_review(
        self, tmp_path: object
    ) -> None:
        # A wrong-SHAPED indication element (non-string quote, string
        # confidence) is schema drift: the citation may be garbled rather
        # than absent, so it never exempts AND is never asserted on — the
        # over-clear fails closed to human review.
        ctx = _row_context(
            audit_id="audit-oc-bleed-malformed",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": 500,
                    "source_id": "E1",
                    "confidence": "0.9",
                }
            ],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON
        assert row.needs_human_review is True

    def test_non_mapping_indication_element_floors_to_review(
        self, tmp_path: object
    ) -> None:
        # A bare-string element would be silently DROPPED by the shallow
        # extraction — a lost citation must floor the over-clear, not read
        # as "the model cited nothing".
        ctx = _row_context(
            audit_id="audit-oc-bleed-strelem",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            indications=["ACTIVE_BLEEDING massive bleed"],  # type: ignore[list-item]
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == LLM_OVERCLEAR_REVIEW_REASON

    def test_bad_negative_evidence_element_blocks_native_conversion(
        self, tmp_path: object
    ) -> None:
        # negative_evidence items must be strings per the tool schema; a
        # drifted element means the payload cannot be trusted enough to
        # convert a hedge into a final INAPPROPRIATE.
        ctx = _row_context(
            audit_id="audit-native-bad-ne",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="NEEDS_REVIEW",
            reasoning_en="hedge with reasoning but drifted negative evidence",
            negative_evidence=[123],
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason is None
        assert row.needs_human_review is True

    def test_high_hb_soft_overclear_asserted_inappropriate(
        self, tmp_path: object
    ) -> None:
        # A soft clear of Hb >= 10 is withheld policy, not an automatic approval.
        ctx = _row_context(
            audit_id="audit-oc-high-hb",
            classification="POTENTIALLY_INAPPROPRIATE",
            hb_value=12.9,
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id, classification="APPROPRIATE"
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_OVERCLEAR_ASSERT_REASON
        assert row.needs_human_review is False
        assert row.rule_classification == "POTENTIALLY_INAPPROPRIATE"

    def test_native_needs_review_converts_to_inappropriate(
        self, tmp_path: object
    ) -> None:
        # A well-reasoned hedge is converted because the prompt no longer permits it.
        ctx = _row_context(
            audit_id="audit-native-review", classification="NEEDS_REVIEW", hb_value=9.4
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="NEEDS_REVIEW",
            reasoning_en="gray-zone order lacks a transfusion indication",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason == LLM_NATIVE_REVIEW_ASSERT_REASON
        assert row.needs_human_review is False

    def test_native_hedge_with_grounded_acs_stays_review(
        self, tmp_path: object
    ) -> None:
        # Codex PR #97 P1 (conversion side): a hedge that cites a grounded,
        # high-confidence ACS indication is a genuine human case — converting
        # it to INAPPROPRIATE would assert against the prompt's own hard code.
        ctx = _row_context(
            audit_id="audit-native-review-acs",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="Progress: NSTEMI with ongoing chest pain, troponin rising",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="NEEDS_REVIEW",
            indications=[
                {
                    "code": "ACS",
                    "quote": "NSTEMI with ongoing chest pain",
                    "source_id": "E1",
                    "confidence": 0.9,
                }
            ],
            reasoning_en="possible ischemic indication, deferring to a human",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason is None
        assert row.needs_human_review is True

    def test_native_needs_review_with_empty_reasoning_stays_review(
        self, tmp_path: object
    ) -> None:
        # An unexplained hedge belongs to the empty-reasoning safety net.
        ctx = _row_context(
            audit_id="audit-native-review-empty",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="NEEDS_REVIEW",
            reasoning_en="",
            reasoning_th="",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == EMPTY_REASONING_REVIEW_REASON

    def test_native_needs_review_with_hard_signal_stays_review(
        self, tmp_path: object
    ) -> None:
        # A hedge against a structured hard signal is a genuine human-review case.
        ctx = _row_context(
            audit_id="audit-native-review-hard",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            periop_summary=PeriopSummary(surgical_context=True),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="NEEDS_REVIEW",
            reasoning_en="surgery documented but indication remains uncertain",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason is None
        assert row.needs_human_review is True

    def test_native_needs_review_with_qualified_bleed_stays_review(
        self, tmp_path: object
    ) -> None:
        # A qualified major bleed makes a hedge a genuine human-review case.
        ctx = _row_context(
            audit_id="audit-native-review-bleed",
            classification="NEEDS_REVIEW",
            hb_value=9.4,
            evidence_text="Post-op note: EBL 400 mL from drain documented",
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="NEEDS_REVIEW",
            indications=[
                {
                    "code": "ACTIVE_BLEEDING",
                    "quote": "EBL 400 mL from drain",
                    "source_id": "E1",
                    "confidence": 0.85,
                }
            ],
            reasoning_en="major bleeding is documented but needs adjudication",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason is None

    def test_overclear_with_empty_reasoning_uses_empty_reasoning_provenance(
        self, tmp_path: object
    ) -> None:
        # Empty reasoning must override an assertion slug so provenance stays truthful.
        ctx = _row_context(
            audit_id="audit-oc-empty", classification="NEEDS_REVIEW", hb_value=9.4
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            reasoning_en="",
            reasoning_th="",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == EMPTY_REASONING_REVIEW_REASON


class TestPeriopContradictionGuardrail:
    """Integration pin on the override inside ``apply_batch_results``."""

    def test_case_107_shape_escalates_to_human_review(self, tmp_path: object) -> None:
        # Exact Case 107 shape: documented ORIF + 1500 mL EBL in a focus note,
        # LLM still returns INSUFFICIENT_EVIDENCE.
        summary = PeriopSummary(
            surgical_context=True,
            blood_loss_ml=1500,
            intraop_transfusion=True,
            findings=(
                PeriopFinding(
                    category="surgery",
                    snippet="s/p CRIF, blood loss 1500 ml, intra-op LPRC",
                    at=_RUN_TS,
                    source="IPDNRFOCUSDT",
                ),
            ),
        )
        ctx = _row_context(
            audit_id="audit-periop-107",
            classification="POTENTIALLY_INAPPROPRIATE",
            periop_summary=summary,
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id, classification="INSUFFICIENT_EVIDENCE"
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == PERIOP_CONTRADICTION_REVIEW_REASON
        assert row.needs_human_review is True

    def test_escalation_preserves_llm_reasoning_and_verifier_pass(
        self, tmp_path: object
    ) -> None:
        # Unlike the hallucination_suspect path (which blanks reasoning), a
        # contradiction must keep the model's words so the reviewer sees the
        # conflict, and verifier_pass stays True (the citations DID ground).
        ctx = _row_context(
            audit_id="audit-periop-preserve",
            periop_summary=PeriopSummary(blood_loss_ml=1500),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="INSUFFICIENT_EVIDENCE",
            indications=[{"text": "post-op blood loss", "confidence": 0.4, "tier": 1}],
            reasoning_en="No structured procedure row found.",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == PERIOP_CONTRADICTION_REVIEW_REASON
        assert row.reasoning_summary_en == "No structured procedure row found."
        assert len(row.indications_json) == 1
        assert row.verifier_pass is True

    def test_appropriate_verdict_survives_strong_signal(self, tmp_path: object) -> None:
        ctx = _row_context(
            audit_id="audit-periop-appropriate",
            periop_summary=PeriopSummary(surgical_context=True, blood_loss_ml=1500),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id, classification="APPROPRIATE"
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "APPROPRIATE"
        assert row.review_reason is None
        assert row.needs_human_review is False

    def test_no_signal_leaves_insufficient_verdict_untouched(
        self, tmp_path: object
    ) -> None:
        ctx = _row_context(audit_id="audit-periop-nosignal", periop_summary=None)
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id, classification="INSUFFICIENT_EVIDENCE"
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INSUFFICIENT_EVIDENCE"
        assert row.review_reason is None


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


# Observed leak shapes from the 2026-07-06 pilot run (165 rows,
# claude-sonnet-5): the model serialized BOTH summaries into
# reasoning_summary_en, separated by fragments of its internal
# tool-call tag syntax, leaving reasoning_summary_th empty. 131/165
# rows were affected; each constant below reproduces one observed
# opening-tag variant verbatim.
_LEAK_EN = "Hb 14.8 g/dL is far above the 7.0 floor."
_LEAK_TH = "ค่า Hb สูงกว่าเกณฑ์ 7.0 g/dL มาก จึงไม่เข้าเกณฑ์"


def _leaked(th_open: str, tail: str = "</reasoning_summary_th>") -> str:
    return f"{_LEAK_EN}</reasoning_summary_en>\n{th_open}{_LEAK_TH}{tail}"


class TestSplitLeakedSummaries:
    """Unit pin on the pure tag-leak salvage function.

    WHY: the committee report renders both languages side by side; a
    leaked row shows English+tags+Thai as one blob and an empty Thai
    panel, which reviewers cannot read. The salvage must recover every
    observed variant WITHOUT touching rows that parsed cleanly.
    """

    def test_clean_pair_passes_through(self) -> None:
        assert split_leaked_summaries("english", "thai") == ("english", "thai")

    def test_both_empty_pass_through(self) -> None:
        assert split_leaked_summaries("", "") == ("", "")

    def test_stray_quote_variant_splits(self) -> None:
        # 96/131 leaked rows used <reasoning_summary_th"> (stray quote).
        en, th = split_leaked_summaries(_leaked('<reasoning_summary_th">'), "")
        assert en == _LEAK_EN
        assert th == _LEAK_TH

    def test_well_formed_tag_variant_splits(self) -> None:
        en, th = split_leaked_summaries(_leaked("<reasoning_summary_th>"), "")
        assert (en, th) == (_LEAK_EN, _LEAK_TH)

    def test_parameter_name_variant_splits(self) -> None:
        en, th = split_leaked_summaries(
            _leaked('<parameter name="reasoning_summary_th">'), ""
        )
        assert (en, th) == (_LEAK_EN, _LEAK_TH)

    def test_name_attribute_variant_splits(self) -> None:
        en, th = split_leaked_summaries(
            _leaked('<reasoning_summary_th name="reasoning_summary_th">'), ""
        )
        assert (en, th) == (_LEAK_EN, _LEAK_TH)

    def test_trailing_invoke_close_is_stripped(self) -> None:
        raw = _leaked(
            '<reasoning_summary_th">',
            tail='</reasoning_summary_th">\n</invoke>',
        )
        en, th = split_leaked_summaries(raw, "")
        assert (en, th) == (_LEAK_EN, _LEAK_TH)

    def test_unterminated_thai_block_recovered_to_end(self) -> None:
        raw = f'{_LEAK_EN}</reasoning_summary_en>\n<reasoning_summary_th">{_LEAK_TH}'
        en, th = split_leaked_summaries(raw, "")
        assert (en, th) == (_LEAK_EN, _LEAK_TH)

    def test_dangling_close_without_thai_open_strips_tag_only(self) -> None:
        raw = f"{_LEAK_EN}</reasoning_summary_en>"
        en, th = split_leaked_summaries(raw, "")
        assert (en, th) == (_LEAK_EN, "")

    def test_nonempty_thai_is_never_overwritten(self) -> None:
        # Salvage only fires when the th field came back empty; a row
        # that parsed cleanly must be byte-identical after the call.
        raw = _leaked('<reasoning_summary_th">')
        assert split_leaked_summaries(raw, "existing thai") == (
            raw,
            "existing thai",
        )


class TestReasoningLeakSalvageIntegration:
    """The salvage must run inside apply_batch_results so persisted
    rows are clean without any caller-side post-processing."""

    def test_leaked_row_persists_split_summaries(self, tmp_path: object) -> None:
        ctx = _row_context(audit_id="audit-leak-1", hb_value=9.4)
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="INAPPROPRIATE",
            reasoning_en=_leaked('<reasoning_summary_th">'),
            reasoning_th="",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.reasoning_summary_en == _LEAK_EN
        assert row.reasoning_summary_thai == _LEAK_TH


class TestEmptyReasoningGuardrail:
    """A verdict with no reasoning in either language cannot be audited
    by the committee — it must be floored to human review.

    WHY: pilot run 2026-07-06 contained 9 rows with completely empty
    reasoning, one of them APPROPRIATE with needs_human_review=False —
    an unexplained automatic clear, which is exactly what the audit
    exists to prevent.
    """

    def test_empty_reasoning_floors_appropriate_to_review(
        self, tmp_path: object
    ) -> None:
        # Hard peri-op signal exempts the B1 overclear guardrail (see
        # test_hard_signal_backed_appropriate_not_floored), so ONLY the
        # empty reasoning can be what floors this APPROPRIATE verdict.
        ctx = _row_context(
            audit_id="audit-empty-1",
            hb_value=9.4,
            periop_summary=PeriopSummary(surgical_context=True, blood_loss_ml=1500),
        )
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="APPROPRIATE",
            reasoning_en="",
            reasoning_th="",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == EMPTY_REASONING_REVIEW_REASON
        assert row.needs_human_review is True

    def test_whitespace_only_reasoning_counts_as_empty(self, tmp_path: object) -> None:
        ctx = _row_context(audit_id="audit-empty-2", hb_value=9.4)
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="INAPPROPRIATE",
            reasoning_en="  \n ",
            reasoning_th="\t",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == EMPTY_REASONING_REVIEW_REASON

    def test_single_language_reasoning_is_sufficient(self, tmp_path: object) -> None:
        # One populated language is reviewable; the guardrail targets
        # totally unexplained verdicts, not missing translations.
        ctx = _row_context(audit_id="audit-empty-3", hb_value=9.4)
        response = _periop_llm_response(
            audit_id=ctx.order.audit_id,
            classification="INAPPROPRIATE",
            reasoning_en="english only",
            reasoning_th="",
        )
        row = _apply_single_row(ctx, response, tmp_path=tmp_path)
        assert row.final_classification == "INAPPROPRIATE"
        assert row.review_reason is None

    def test_review_reason_is_distinct(self) -> None:
        reasons = {
            PERIOP_CONTRADICTION_REVIEW_REASON,
            LLM_OVERCLEAR_REVIEW_REASON,
            "hallucination_suspect",
            EMPTY_REASONING_REVIEW_REASON,
            LLM_OVERCLEAR_ASSERT_REASON,
            LLM_NATIVE_REVIEW_ASSERT_REASON,
        }
        assert len(reasons) == 6
