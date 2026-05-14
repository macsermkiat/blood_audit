"""RED-phase failing tests for issue #19 (bba.audit_store).

Each ``class`` maps to one acceptance criterion in the issue body. Tests assert
contracts (the WHY), not implementation choices — see PRD §"Testing Decisions".

No implementation exists yet; every test MUST fail with ``NotImplementedError``
(behavioral tests) or a model-level rejection (immutability tests) in this
scaffold commit. If a test fails with ``ImportError``/``AttributeError`` instead,
the scaffold is wrong — fix the public surface, not the test.

The acceptance-criterion → test-class map:

* AC ① "round-trip with full audit-row schema"
  → :class:`TestAuditRowRoundTrip`, :class:`TestLlmCallRoundTrip`,
    :class:`TestModelImmutability`
* AC ② "transactional ordering invariant + reconciliation"
  → :class:`TestTransactionalOrderingInvariant`,
    :class:`TestReconciliationFindsOrphanCalls`,
    :class:`TestWriteOrdersPhasesCorrectly`
* AC ③ "idempotent re-run"
  → :class:`TestIdempotentRerun`
* AC ④ "snapshot-view consistency"
  → :class:`TestSnapshotViewConsistency`
* AC ⑤ "cold-storage policy stub"
  → :class:`TestColdStorageMigration`
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from bba.audit_store import (
    AuditRow,
    AuditStore,
    AuditStoreConfig,
    ColdStorageReport,
    LlmCall,
    ReconciliationReport,
    SnapshotView,
    TransactionalOrderingError,
    WriteResult,
    migrate_cold_storage,
)


# =============================================================================
# Fixtures — minimal valid AuditRow / LlmCall builders.
#
# The full PRD §Output schema has ~30 fields. The builders default every field
# to a plausible value so each test only states the fields it actually exercises;
# that keeps the test bodies focused on the property under test (the WHY) rather
# than on filling in 30 unrelated kwargs.
# =============================================================================


def _row(
    *,
    audit_id: str = "audit-001",
    run_id: str = "run-aaa",
    final_classification: str = "APPROPRIATE",
    **overrides: object,
) -> AuditRow:
    base: dict[str, object] = {
        # Identity
        "audit_id": audit_id,
        "run_id": run_id,
        "run_timestamp": datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
        "hn_hash": "hn-sha256-aaa",
        "an_hash": "an-sha256-bbb",
        "reqno": "REQ-12345",
        # Anchor + inputs
        "order_datetime": datetime(2026, 5, 1, 8, 30, 0, tzinfo=UTC),
        "products_ordered": ("LPRC",),
        "hb_value": 6.8,
        "hb_datetime": datetime(2026, 5, 1, 7, 0, 0, tzinfo=UTC),
        "hb_freshness": "fresh_<6h",
        "hb_source": "LABEXM",
        "vitals_sbp": 95.0,
        "vitals_hr": 110.0,
        "vitals_timestamp": datetime(2026, 5, 1, 8, 0, 0, tzinfo=UTC),
        "vitals_source": "IPDADMPROGRESS",
        "prior_rbc_units_24h": 0,
        "prior_rbc_units_7d": 2,
        "cohort_threshold": 7.0,
        "delta_hb_window_results": (
            {"window": "24h", "delta": 1.4, "trigger": False},
        ),
        # Pipeline outputs
        "rule_classification": "APPROPRIATE",
        "final_classification": final_classification,
        "cohort_applied": "general_medical",
        "indications_json": (
            {
                "code": "B1.acute_anemia",
                "quote": "Hb 6.8 with symptomatic tachycardia",
                "source_id": "IPDNRFOCUSDT:42",
                "confidence": 0.92,
            },
        ),
        "negative_evidence_json": (),
        "confidence": 0.91,
        "reasoning_summary_thai": "ผู้ป่วยมีภาวะซีดเฉียบพลันร่วมกับชีพจรเร็ว",
        "reasoning_summary_en": "Acute anemia with tachycardia.",
        "needs_human_review": False,
        "review_reason": None,
        # Reproducibility metadata
        "model_id": "claude-sonnet-4-6-20260201",
        "prompt_hash": "prompt-sha256-ccc",
        "evidence_bundle_hash": "bundle-sha256-ddd",
        "redactor_version": "0.4.1",
        "redactor_model_sha": "redactor-sha256-eee",
        "policy_version": "kcmh-pr17.2-2024",
        "verifier_pass": True,
        "verifier_retries": 0,
        "escalated_to_opus": False,
    }
    base.update(overrides)
    return AuditRow.model_validate(base)


def _call(
    *,
    call_id: str = "call-001",
    audit_id: str = "audit-001",
    run_id: str = "run-aaa",
    request_timestamp: datetime | None = None,
    extended_thinking_blocks: tuple[dict[str, object], ...] | None = (
        {"type": "thinking", "text": "Step 1: check Hb..." * 50},
    ),
    cold_storage_uri: str | None = None,
    **overrides: object,
) -> LlmCall:
    base: dict[str, object] = {
        "call_id": call_id,
        "audit_id": audit_id,
        "run_id": run_id,
        "model_id": "claude-sonnet-4-6-20260201",
        "anthropic_version": "2023-06-01",
        "prompt_cache_id": "cache-aaa",
        "request_json": {"system": "...", "messages": [{"role": "user"}]},
        "response_json": {"id": "msg_01", "stop_reason": "tool_use"},
        "request_timestamp": request_timestamp
        or datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
        "latency_ms": 1200,
        "extended_thinking_blocks": extended_thinking_blocks,
        "cold_storage_uri": cold_storage_uri,
    }
    base.update(overrides)
    return LlmCall.model_validate(base)


@pytest.fixture
def store(tmp_path: Path) -> AuditStore:
    return AuditStore(
        AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
    )


# =============================================================================
# AC ① — Write/read round-trip tested with the full audit-row schema
#
# WHY: PRD §Output schema is the contract between the audit pipeline and every
# downstream consumer (eval_harness, dashboard, report_generator). A round-trip
# regression means six months later we cannot reconstruct what was classified.
# =============================================================================


class TestAuditRowRoundTrip:
    """A written AuditRow must read back byte-for-byte equal."""

    def test_round_trip_preserves_every_field(self, store: AuditStore) -> None:
        row = _row()
        call = _call()
        store.write(row, [call])

        rows = store.read_audit_results()

        assert len(rows) == 1
        assert rows[0] == row

    def test_round_trip_preserves_thai_unicode_in_summary(
        self, store: AuditStore
    ) -> None:
        row = _row(reasoning_summary_thai="ผู้ป่วยมีภาวะเลือดออก")
        store.write(row, [_call()])

        assert store.read_audit_results()[0].reasoning_summary_thai == (
            row.reasoning_summary_thai
        )

    def test_round_trip_preserves_nested_json_fields(self, store: AuditStore) -> None:
        row = _row(
            indications_json=(
                {"code": "B1", "quote": "Hb 6.8", "source_id": "X:1", "confidence": 0.9},
                {"code": "B2", "quote": "tachy", "source_id": "Y:2", "confidence": 0.8},
            ),
            negative_evidence_json=({"code": "no_bleed", "quote": "stable", "source_id": "Z:3"},),
        )
        store.write(row, [_call()])

        rt = store.read_audit_results()[0]
        assert rt.indications_json == row.indications_json
        assert rt.negative_evidence_json == row.negative_evidence_json

    def test_multiple_runs_coexist_in_same_dataset(self, store: AuditStore) -> None:
        store.write(_row(audit_id="a1", run_id="r1"), [_call(call_id="c1", audit_id="a1", run_id="r1")])
        store.write(_row(audit_id="a2", run_id="r2"), [_call(call_id="c2", audit_id="a2", run_id="r2")])

        assert {r.run_id for r in store.read_audit_results()} == {"r1", "r2"}

    def test_read_audit_results_filters_by_run_id(self, store: AuditStore) -> None:
        store.write(_row(audit_id="a1", run_id="r1"), [_call(call_id="c1", audit_id="a1", run_id="r1")])
        store.write(_row(audit_id="a2", run_id="r2"), [_call(call_id="c2", audit_id="a2", run_id="r2")])

        only_r1 = store.read_audit_results(run_id="r1")

        assert {r.audit_id for r in only_r1} == {"a1"}


class TestLlmCallRoundTrip:
    """LlmCall round-trip must preserve extended_thinking_blocks verbatim."""

    def test_call_round_trip_preserves_extended_thinking(self, store: AuditStore) -> None:
        call = _call(
            extended_thinking_blocks=(
                {"type": "thinking", "text": "thought A"},
                {"type": "thinking", "text": "thought B"},
            ),
        )
        store.write(_row(), [call])

        calls = store.read_llm_calls()

        assert len(calls) == 1
        assert calls[0] == call

    def test_multiple_calls_per_audit_id(self, store: AuditStore) -> None:
        c1 = _call(call_id="c1")
        c2 = _call(call_id="c2", model_id="claude-opus-4-7-20260301")
        store.write(_row(), [c1, c2])

        calls = store.read_llm_calls()
        assert {c.call_id for c in calls} == {"c1", "c2"}

    def test_read_llm_calls_filters_by_run_id(self, store: AuditStore) -> None:
        store.write(_row(audit_id="a1", run_id="r1"), [_call(call_id="c1", audit_id="a1", run_id="r1")])
        store.write(_row(audit_id="a2", run_id="r2"), [_call(call_id="c2", audit_id="a2", run_id="r2")])

        only_r1 = store.read_llm_calls(run_id="r1")
        assert {c.call_id for c in only_r1} == {"c1"}


class TestModelImmutability:
    """AuditRow and LlmCall must reject mutation — frozen pydantic.

    WHY: PRD §Output schema says "persisted immutably". A row that can be
    silently mutated in memory after construction breaks the reproducibility
    chain that the entire audit defends.
    """

    def test_audit_row_rejects_attribute_assignment(self) -> None:
        row = _row()
        with pytest.raises(ValidationError):
            row.final_classification = "INAPPROPRIATE"  # type: ignore[misc]

    def test_llm_call_rejects_attribute_assignment(self) -> None:
        call = _call()
        with pytest.raises(ValidationError):
            call.latency_ms = 9999  # type: ignore[misc]

    def test_audit_row_collections_are_tuples_not_lists(self) -> None:
        row = _row()
        assert isinstance(row.products_ordered, tuple)
        assert isinstance(row.indications_json, tuple)
        assert isinstance(row.negative_evidence_json, tuple)
        assert isinstance(row.delta_hb_window_results, tuple)


# =============================================================================
# AC ② — Transactional-ordering invariant: an audit_results row without a
# matching llm_calls row is a bug; test forces the failure mode and confirms
# reconciliation.
#
# WHY: audit_results is the commit marker. If it can exist without the calls
# that produced it, the classification is no longer reproducible from byte 0 —
# the entire reproducibility promise of the system fails open.
# =============================================================================


class TestWriteOrdersPhasesCorrectly:
    """The canonical write() must persist calls before the audit row."""

    def test_canonical_write_leaves_no_invariant_violation(
        self, store: AuditStore
    ) -> None:
        row = _row()
        store.write(row, [_call()])

        store.validate_invariants(row.run_id)  # MUST NOT raise

    def test_canonical_write_leaves_no_orphans(self, store: AuditStore) -> None:
        row = _row()
        store.write(row, [_call()])

        report = store.reconcile(row.run_id)

        assert report.orphan_call_ids == ()
        assert report.orphan_audit_ids == ()


class TestTransactionalOrderingInvariant:
    """audit_results without llm_calls is a bug — validate_invariants raises."""

    def test_audit_result_without_calls_raises(self, store: AuditStore) -> None:
        # Force the bad state: persist phase 2 (audit row) WITHOUT phase 1
        # (calls). In production this only happens via a coding bug; tests use
        # the phase-level seam to construct the failure state without poking
        # the on-disk layout.
        row = _row()
        store._persist_audit_result(row)

        with pytest.raises(TransactionalOrderingError):
            store.validate_invariants(row.run_id)

    def test_invariant_error_names_offending_audit_id(
        self, store: AuditStore
    ) -> None:
        row = _row(audit_id="audit-orphan-42")
        store._persist_audit_result(row)

        with pytest.raises(TransactionalOrderingError, match="audit-orphan-42"):
            store.validate_invariants(row.run_id)

    def test_partial_failure_only_some_audit_results_orphaned(
        self, store: AuditStore
    ) -> None:
        # One full commit, one half-commit. Validate must catch only the half.
        good = _row(audit_id="audit-ok", run_id="r-mix")
        store.write(good, [_call(call_id="c-ok", audit_id="audit-ok", run_id="r-mix")])

        bad = _row(audit_id="audit-bad", run_id="r-mix")
        store._persist_audit_result(bad)

        with pytest.raises(TransactionalOrderingError, match="audit-bad"):
            store.validate_invariants("r-mix")


class TestReconciliationFindsOrphanCalls:
    """llm_calls without audit_results is a crash-after-phase-1 fallout —
    reconcile() catalogues these (not an error condition)."""

    def test_orphan_calls_found_and_named(self, store: AuditStore) -> None:
        # Stage "crashed between phases": phase 1 ran, phase 2 didn't.
        store._persist_llm_calls(
            [_call(call_id="orphan-c1", audit_id="dropped-a1", run_id="r-crash")]
        )

        report = store.reconcile("r-crash")

        assert "orphan-c1" in report.orphan_call_ids
        assert report.orphan_audit_ids == ()

    def test_reconcile_returns_empty_report_when_clean(
        self, store: AuditStore
    ) -> None:
        store.write(
            _row(audit_id="a1", run_id="r1"),
            [_call(call_id="c1", audit_id="a1", run_id="r1")],
        )

        report = store.reconcile("r1")

        assert isinstance(report, ReconciliationReport)
        assert report.orphan_call_ids == ()

    def test_reconcile_isolates_by_run_id(self, store: AuditStore) -> None:
        store._persist_llm_calls(
            [_call(call_id="orphan-a", audit_id="a-a", run_id="r-A")]
        )
        store.write(
            _row(audit_id="a-b", run_id="r-B"),
            [_call(call_id="c-b", audit_id="a-b", run_id="r-B")],
        )

        report_b = store.reconcile("r-B")

        assert "orphan-a" not in report_b.orphan_call_ids


# =============================================================================
# AC ③ — Idempotent re-run: same run_id writes once, returns cached result
# on second call.
#
# WHY: PRD §Implementation Decisions: "Re-trigger is a no-op unless --force".
# A re-run that silently appends a second copy of every row breaks the
# uniqueness assumption of every downstream consumer.
# =============================================================================


class TestIdempotentRerun:
    def test_second_write_returns_skipped_idempotent(self, store: AuditStore) -> None:
        row = _row()
        first = store.write(row, [_call()])
        assert isinstance(first, WriteResult)
        assert first.skipped_idempotent is False

        second = store.write(row, [_call()])

        assert second.skipped_idempotent is True

    def test_idempotent_rerun_does_not_duplicate_rows(self, store: AuditStore) -> None:
        row = _row()
        store.write(row, [_call()])
        store.write(row, [_call()])

        assert len(store.read_audit_results()) == 1
        assert len(store.read_llm_calls()) == 1

    def test_different_run_id_writes_new_row_for_same_audit_id(
        self, store: AuditStore
    ) -> None:
        # Same audit_id, different run_id (e.g., code-version bump → new run)
        # is NOT a no-op: it is a re-derivation and must persist.
        store.write(
            _row(audit_id="a1", run_id="r1"), [_call(call_id="c1", audit_id="a1", run_id="r1")]
        )
        store.write(
            _row(audit_id="a1", run_id="r2"), [_call(call_id="c2", audit_id="a1", run_id="r2")]
        )

        rows = store.read_audit_results()
        assert {r.run_id for r in rows} == {"r1", "r2"}


# =============================================================================
# AC ④ — Snapshot-view consistency: dashboard reads from snapshot N don't
# observe in-flight writes for snapshot N+1.
#
# WHY: PRD §"DuckDB single-writer contention" — a dashboard query that lands
# mid-write would see a partial batch and misreport the monthly summary. The
# snapshot freezes the visible set at materialization time.
# =============================================================================


class TestSnapshotViewConsistency:
    def test_snapshot_freezes_at_materialization(self, store: AuditStore) -> None:
        store.write(
            _row(audit_id="a-existing", run_id="r1"),
            [_call(call_id="c1", audit_id="a-existing", run_id="r1")],
        )

        view = SnapshotView.open(store, as_of=date(2026, 5, 1))

        # Write a NEW row AFTER the snapshot is materialized.
        store.write(
            _row(audit_id="a-new", run_id="r2"),
            [_call(call_id="c2", audit_id="a-new", run_id="r2")],
        )

        visible_ids = {r.audit_id for r in view.read_audit_results()}

        assert "a-existing" in visible_ids
        assert "a-new" not in visible_ids

    def test_reopening_same_day_returns_same_frozen_set(self, store: AuditStore) -> None:
        store.write(_row(audit_id="a1", run_id="r1"), [_call(call_id="c1", audit_id="a1", run_id="r1")])

        view_morning = SnapshotView.open(store, as_of=date(2026, 5, 1))
        store.write(_row(audit_id="a2", run_id="r2"), [_call(call_id="c2", audit_id="a2", run_id="r2")])
        view_afternoon = SnapshotView.open(store, as_of=date(2026, 5, 1))

        # Same as_of must return identical frozen content — re-opening the day's
        # snapshot doesn't re-materialize and pull in newer writes.
        assert {r.audit_id for r in view_morning.read_audit_results()} == {
            r.audit_id for r in view_afternoon.read_audit_results()
        }

    def test_next_day_snapshot_sees_writes_from_previous_day(
        self, store: AuditStore
    ) -> None:
        store.write(_row(audit_id="a1", run_id="r1"), [_call(call_id="c1", audit_id="a1", run_id="r1")])
        SnapshotView.open(store, as_of=date(2026, 5, 1))
        store.write(_row(audit_id="a2", run_id="r2"), [_call(call_id="c2", audit_id="a2", run_id="r2")])

        view_next_day = SnapshotView.open(store, as_of=date(2026, 5, 2))

        assert {r.audit_id for r in view_next_day.read_audit_results()} == {"a1", "a2"}


# =============================================================================
# AC ⑤ — Cold-storage policy stub for extended-thinking blocks (>90 days).
#
# WHY: PRD §10 says Opus extended-thinking blocks move to cold storage after
# 90 days. The stub satisfies the contract for the Phase-1 pipeline; the real
# S3 adapter is a Phase-2 swap behind the same function signature.
# =============================================================================


class TestColdStorageMigration:
    def test_migration_returns_moved_call_ids(self, store: AuditStore) -> None:
        old_ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        store.write(
            _row(audit_id="a1", run_id="r1"),
            [_call(call_id="c-old", audit_id="a1", run_id="r1", request_timestamp=old_ts)],
        )

        report = migrate_cold_storage(store, older_than=datetime(2026, 4, 2, 0, 0, 0, tzinfo=UTC))

        assert isinstance(report, ColdStorageReport)
        assert "c-old" in report.moved_call_ids

    def test_migration_clears_inline_blocks_and_sets_uri(
        self, store: AuditStore
    ) -> None:
        old_ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        store.write(
            _row(audit_id="a1", run_id="r1"),
            [_call(call_id="c-old", audit_id="a1", run_id="r1", request_timestamp=old_ts)],
        )

        migrate_cold_storage(store, older_than=datetime(2026, 4, 2, 0, 0, 0, tzinfo=UTC))

        migrated = next(c for c in store.read_llm_calls() if c.call_id == "c-old")
        assert migrated.extended_thinking_blocks is None
        assert migrated.cold_storage_uri is not None

    def test_migration_skips_recent_calls(self, store: AuditStore) -> None:
        recent = datetime(2026, 5, 1, 0, 0, 0, tzinfo=UTC)
        store.write(
            _row(audit_id="a1", run_id="r1"),
            [_call(call_id="c-recent", audit_id="a1", run_id="r1", request_timestamp=recent)],
        )

        cutoff = recent - timedelta(days=90)
        report = migrate_cold_storage(store, older_than=cutoff)

        assert "c-recent" not in report.moved_call_ids
        kept = next(c for c in store.read_llm_calls() if c.call_id == "c-recent")
        assert kept.extended_thinking_blocks is not None

    def test_migration_is_idempotent(self, store: AuditStore) -> None:
        old_ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        store.write(
            _row(audit_id="a1", run_id="r1"),
            [_call(call_id="c-old", audit_id="a1", run_id="r1", request_timestamp=old_ts)],
        )
        cutoff = datetime(2026, 4, 2, 0, 0, 0, tzinfo=UTC)

        migrate_cold_storage(store, older_than=cutoff)
        second = migrate_cold_storage(store, older_than=cutoff)

        # Already-migrated rows are not re-moved.
        assert second.moved_call_ids == ()
        assert second.bytes_moved == 0
