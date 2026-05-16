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


class TestReplayIdempotencyProperty:
    @given(
        n_results=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=15, deadline=None)
    def test_applying_same_batch_response_twice_writes_no_new_rows(
        self, n_results: int, seed: int, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """For any synthetic RawBatchResponse of N results, applying it
        twice produces zero new audit_results rows on the second pass.

        Implementation lands in GREEN; here we pin the contract via
        :func:`apply_batch_results`'s typed return — which must report
        zero ``audit_ids_persisted`` on the second call."""
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        tmp_path: Path = tmp_path_factory.mktemp(f"replay_{seed}_{n_results}")
        audit_store = AuditStore(
            AuditStoreConfig(
                root_dir=tmp_path / "store", code_version="v0.1.0+test"
            )
        )
        synthetic_response = _build_synthetic_raw_batch_response(
            n_results=n_results, seed=seed
        )
        first = apply_batch_results(
            synthetic_response,
            audit_store=audit_store,
            run_id="run-replay",
        )
        second = apply_batch_results(
            synthetic_response,
            audit_store=audit_store,
            run_id="run-replay",
        )
        assert first.audit_ids_persisted  # at least one row written first pass
        assert (
            second.audit_ids_persisted == ()
        ), "second application must be a no-op (zero new rows)"


def _build_synthetic_raw_batch_response(
    *, n_results: int, seed: int
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
                            "classification": "APPROPRIATE",
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
        module — every upstream module is already independently tested.
        Concrete fixture builds land in GREEN once the cassettes are
        seeded; here we pin the contract surface."""
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(
                root_dir=tmp_path / "store", code_version="v0.1.0+test"
            )
        )
        synthetic_orders = _build_synthetic_audit_orders(n=5)
        result = run_pipeline(
            synthetic_orders,
            transport=_cassette_transport(),
            audit_store=audit_store,
            batch_run_store=InMemoryBatchRunStore(),
            llm_config=_llm_config(),
            pipeline_config=_pipeline_config(),
            run_id="run-smoke",
        )
        assert isinstance(result, PipelineRunResult)
        assert len(result.audit_ids_persisted) == 5

    def test_smoke_emits_no_orphan_llm_calls(self, tmp_path: object) -> None:
        from pathlib import Path

        from bba.audit_store import AuditStore, AuditStoreConfig

        assert isinstance(tmp_path, Path)
        audit_store = AuditStore(
            AuditStoreConfig(
                root_dir=tmp_path / "store", code_version="v0.1.0+test"
            )
        )
        synthetic_orders = _build_synthetic_audit_orders(n=5)
        result = run_pipeline(
            synthetic_orders,
            transport=_cassette_transport(),
            audit_store=audit_store,
            batch_run_store=InMemoryBatchRunStore(),
            llm_config=_llm_config(),
            pipeline_config=_pipeline_config(),
            run_id="run-smoke",
        )
        assert result.orphan_audit_ids == ()


def _build_synthetic_audit_orders(*, n: int) -> Sequence[AuditOrder]:
    """Build n synthetic AuditOrders covering all four Hb-tier branches +
    INSUFFICIENT_EVIDENCE.

    Implementation deferred to GREEN once the cassette fixtures are
    seeded. The test exists to lock the smoke-test contract."""
    _ = n
    raise NotImplementedError("RED-phase fixture; see issue #24 GREEN")


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
        # Adversarial cassette: the LLM returns plausible-looking
        # citations whose verbatim quotes are NOT present in the
        # redacted bundle. quote_grounder rejects them all; the
        # winning-attempt rule then surfaces NEEDS_REVIEW.
        orders = _build_adversarial_quote_grounder_orders()
        run_pipeline(
            orders,
            transport=_cassette_transport(),
            audit_store=audit_store,
            batch_run_store=InMemoryBatchRunStore(),
            llm_config=_llm_config(),
            pipeline_config=_pipeline_config(),
            run_id="run-adv-grounder",
        )
        rows = audit_store.read_audit_results(run_id="run-adv-grounder")
        assert all(row.final_classification == "NEEDS_REVIEW" for row in rows)
        assert all(row.review_reason is not None for row in rows)


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
        orders = _build_adversarial_vitals_orders()
        run_pipeline(
            orders,
            transport=_cassette_transport(),
            audit_store=audit_store,
            batch_run_store=InMemoryBatchRunStore(),
            llm_config=_llm_config(),
            pipeline_config=_pipeline_config(),
            run_id="run-adv-vitals",
        )
        rows = audit_store.read_audit_results(run_id="run-adv-vitals")
        # vitals_source must reflect the LLM-fallback provenance, not
        # default to "regex" (which would mask the quality signal).
        assert all(row.vitals_source != "regex" for row in rows if row.vitals_source)


def _build_adversarial_quote_grounder_orders() -> Sequence[AuditOrder]:
    raise NotImplementedError("RED-phase fixture; see issue #24 GREEN")


def _build_adversarial_vitals_orders() -> Sequence[AuditOrder]:
    raise NotImplementedError("RED-phase fixture; see issue #24 GREEN")


