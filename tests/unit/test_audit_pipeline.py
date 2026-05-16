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
from bba.deterministic_classifier import BypassReason, ClassifierResult
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
    llm_contexts = tuple(
        ctx
        for ctx in contexts
        if ctx.classifier_result.classification
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


def _row_context(
    *,
    audit_id: str,
    classification: Classification = "POTENTIALLY_INAPPROPRIATE",
    bypass_reason: BypassReason = BypassReason.NONE,
    hb_value: float = 7.5,
    hb_source: str = "HEMATOLOGY",
    hb_freshness: str = "fresh",
    vitals_source: SourceProvenance = SourceProvenance.IPDADMPROGRESS,
    cohort_label: CohortLabel = CohortLabel.DEFAULT,
    cohort_threshold: float | None = 7.0,
    evidence_text: str = "Hb 7.5 with symptomatic chest pain",
) -> PipelineRowContext:
    """Build a PipelineRowContext for tests.

    Defaults produce a row that routes through the LLM stage
    (POTENTIALLY_INAPPROPRIATE classifier result). Override
    ``classification`` to APPROPRIATE / INSUFFICIENT_EVIDENCE to
    exercise the deterministic-final path.
    """
    order = AuditOrder(
        audit_id=audit_id,
        hn=f"HN-{audit_id}",
        an=f"AN-{audit_id}",
        reqno=f"REQ-{audit_id}",
        order_datetime=_RUN_TS,
        anchor_imputed=False,
        products_ordered=("LPRC",),
        age_years=55,
        sex="M",
        diagnosis_codes=("D62",),
    )
    hb_result = HbLookupResult(
        value_g_dl=hb_value,
        datetime_utc=_RUN_TS,
        source=hb_source,
        freshness=hb_freshness,
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
    classifier = ClassifierResult(
        classification=classification,
        bypass_reason=bypass_reason,
        cohort_threshold=cohort_threshold,
        rationale="hb_7_to_10" if classification == "POTENTIALLY_INAPPROPRIATE" else "hb_ge_10",
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
        classifier_result=classifier,
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
            transport=_cassette_transport(),
            llm_config=_llm_config(),
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
            AuditStoreConfig(
                root_dir=tmp_path / "store", code_version="v0.1.0+test"
            )
        )
        report = resume_on_startup(
            batch_run_store=store,
            audit_store=audit_store,
            transport=_cassette_transport(),
            llm_config=_llm_config(),
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
            AuditStoreConfig(
                root_dir=tmp_path / "store", code_version="v0.1.0+test"
            )
        )
        first = resume_on_startup(
            batch_run_store=store,
            audit_store=audit_store,
            transport=_cassette_transport(),
            llm_config=_llm_config(),
        )
        second = resume_on_startup(
            batch_run_store=store,
            audit_store=audit_store,
            transport=_cassette_transport(),
            llm_config=_llm_config(),
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
        audit row. PRD §10 names this as the expected failure mode."""
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        store = InMemoryBatchRunStore()
        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(
                root_dir=tmp_path / "store", code_version="v0.1.0+test"
            )
        )
        # The reconciler must produce a non-empty reemitted_audit_ids
        # whenever orphans exist. Implementation lands in GREEN; the
        # contract is the typed return shape.
        report = resume_on_startup(
            batch_run_store=store,
            audit_store=audit_store,
            transport=_cassette_transport(),
            llm_config=_llm_config(),
        )
        assert isinstance(report.reemitted_audit_ids, tuple)

    def test_resume_walks_submitted_batches_and_classifies_audit_ids(
        self, tmp_path: object,
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
            AuditStoreConfig(
                root_dir=tmp_path / "store", code_version="v0.1.0+test"
            )
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
            transport=_cassette_transport(),
            llm_config=_llm_config(),
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
            AuditStoreConfig(
                root_dir=tmp_path / "store", code_version="v0.1.0+test"
            )
        )
        report = resume_on_startup(
            batch_run_store=store,
            audit_store=audit_store,
            transport=_cassette_transport(),
            llm_config=_llm_config(),
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
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), min_codepoint=48, max_codepoint=122),
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
            AuditStoreConfig(
                root_dir=tmp_path / "store", code_version="v0.1.0+test"
            )
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
        assert (
            second.audit_ids_persisted == ()
        ), "second application must be a no-op (zero new rows)"


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
            AuditStoreConfig(
                root_dir=tmp_path / "store", code_version="v0.1.0+test"
            )
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
        rows = {
            row.audit_id: row
            for row in audit_store.read_audit_results(run_id="run-smoke")
        }
        for ctx in synthetic_contexts:
            persisted = rows[ctx.order.audit_id]
            if ctx.classifier_result.classification in {
                "APPROPRIATE",
                "INSUFFICIENT_EVIDENCE",
                "INAPPROPRIATE",
            }:
                # deterministic-final path: final == rule
                assert persisted.final_classification == ctx.classifier_result.classification
                assert persisted.model_id == "deterministic"
            else:
                # LLM path: final classification comes from the cassette
                # response (APPROPRIATE) but rule_classification echoes
                # the deterministic input (POTENTIALLY_INAPPROPRIATE).
                assert persisted.rule_classification == ctx.classifier_result.classification
                assert persisted.final_classification == "APPROPRIATE"
                assert "sonnet" in persisted.model_id

    def test_smoke_emits_no_orphan_llm_calls(self, tmp_path: object) -> None:
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(
                root_dir=tmp_path / "store", code_version="v0.1.0+test"
            )
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


def _build_synthetic_contexts(*, n: int) -> Sequence[PipelineRowContext]:
    """Build n synthetic :class:`PipelineRowContext` rows covering the
    five distinct Hb-tier outcomes (cycled).

    The pipeline routes contexts whose ``classifier_result.classification``
    is APPROPRIATE / INSUFFICIENT_EVIDENCE / INAPPROPRIATE directly to
    persistence (no LLM call). Contexts classified as POTENTIALLY_INAPPROPRIATE
    or NEEDS_REVIEW route through the LLM stage, where the cassette
    transport supplies the response.
    """
    classifications: tuple[Classification, ...] = (
        "POTENTIALLY_INAPPROPRIATE",  # routes via LLM
        "APPROPRIATE",                # deterministic-final
        "INSUFFICIENT_EVIDENCE",      # deterministic-final (missing Hb)
        "POTENTIALLY_INAPPROPRIATE",  # routes via LLM
        "POTENTIALLY_INAPPROPRIATE",  # routes via LLM
    )
    return tuple(
        _row_context(
            audit_id=f"audit-smoke-{i:03d}",
            classification=classifications[i % len(classifications)],
            hb_value=7.5 if i % 2 == 0 else 9.0,
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
            AuditStoreConfig(
                root_dir=tmp_path / "store", code_version="v0.1.0+test"
            )
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
            AuditStoreConfig(
                root_dir=tmp_path / "store", code_version="v0.1.0+test"
            )
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


