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


# =============================================================================
# Codex review follow-ups — P1 + P2 (post-GREEN regression guards).
#
# Each class below names the specific failure mode it locks down, so a future
# refactor that re-introduces the bug fails the matching test by name.
# =============================================================================


class TestReadHonorsCommitMarker:
    """Codex P1: a parquet without its commit marker is uncommitted and MUST
    NOT be visible to consumers.

    WHY: if a process crashes between the audit-result parquet write and the
    commit-marker write, leaving the parquet visible would surface an
    uncommitted classification to the dashboard, report generator, and eval
    harness — defeating the entire "marker = commit" contract.
    """

    def test_parquet_without_marker_is_invisible_to_read(
        self, store: AuditStore
    ) -> None:
        # Stage "crashed between phase 2a (parquet write) and phase 2b (mark)".
        store._persist_llm_calls([_call()])
        store._persist_audit_parquet_only(_row())

        assert store.read_audit_results() == ()

    def test_parquet_without_marker_is_invisible_when_filtered_by_run_id(
        self, store: AuditStore
    ) -> None:
        store._persist_llm_calls(
            [_call(call_id="c1", audit_id="a-crashed", run_id="r-crashed")]
        )
        store._persist_audit_parquet_only(
            _row(audit_id="a-crashed", run_id="r-crashed")
        )

        assert store.read_audit_results(run_id="r-crashed") == ()

    def test_committed_row_alongside_uncommitted_parquet_only_returns_committed(
        self, store: AuditStore
    ) -> None:
        # One full commit, one parquet-only (crashed pre-marker). Reader returns
        # only the committed one.
        store.write(
            _row(audit_id="a-ok", run_id="r-mix"),
            [_call(call_id="c-ok", audit_id="a-ok", run_id="r-mix")],
        )
        store._persist_llm_calls(
            [_call(call_id="c-bad", audit_id="a-crashed", run_id="r-mix")]
        )
        store._persist_audit_parquet_only(
            _row(audit_id="a-crashed", run_id="r-mix")
        )

        ids = {r.audit_id for r in store.read_audit_results(run_id="r-mix")}
        assert ids == {"a-ok"}


class TestParquetWriteIsAtomic:
    """Codex P2: parquet records must land via write-then-rename so a crash
    mid-write leaves no half-formed final file.

    WHY: a corrupt final-name parquet would be picked up by ``read_*`` and
    crash the pipeline with an opaque pyarrow error, masking the underlying
    crash. The same atomicity idiom is already used for the commit marker and
    the snapshot view; the record writer must match.

    These tests spy on ``pq.write_table`` to lock down the implementation
    contract (write to ``*.tmp`` then rename) — checking only post-condition
    state would pass vacuously on a direct-write implementation.
    """

    def test_audit_record_writes_via_tmp_path_first(
        self, store: AuditStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from bba.audit_store import store as store_module

        captured: list[Path] = []
        real = store_module.pq.write_table

        def spy(table: object, path: Path, *args: object, **kwargs: object) -> None:
            captured.append(Path(path))
            real(table, path, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(store_module.pq, "write_table", spy)

        store.write(
            _row(audit_id="a-spy"),
            [_call(call_id="c-spy", audit_id="a-spy")],
        )

        assert captured, "pq.write_table was never called"
        for path in captured:
            assert path.suffix == ".tmp", (
                f"pq.write_table wrote directly to {path}; expected a .tmp path"
            )

    def test_audit_write_leaves_no_tmp_residue_on_success(
        self, store: AuditStore
    ) -> None:
        store.write(_row(audit_id="a1"), [_call(audit_id="a1")])

        audit_tmps = list((store.config.root_dir / "audit_results").glob("*.tmp"))
        call_tmps = list((store.config.root_dir / "llm_calls").glob("*.tmp"))

        assert audit_tmps == []
        assert call_tmps == []

    def test_crash_mid_write_leaves_no_final_file(
        self, store: AuditStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from bba.audit_store import store as store_module

        def boom(*_a: object, **_kw: object) -> None:
            raise RuntimeError("simulated crash mid-write")

        monkeypatch.setattr(store_module.pq, "write_table", boom)

        with pytest.raises(RuntimeError, match="simulated crash"):
            store.write(
                _row(audit_id="a-crash"),
                [_call(call_id="c-crash", audit_id="a-crash")],
            )

        # Audit parquet filename now includes a code_version slug — glob the
        # (audit_id, run_id) prefix and confirm no final-name file exists.
        audit_dir = store.config.root_dir / "audit_results"
        call_path = store.config.root_dir / "llm_calls" / "call_c-crash.parquet"

        assert list(audit_dir.glob("audit_a-crash_run-aaa_*.parquet")) == []
        assert not call_path.exists()


class TestWriteRejectsMalformedCalls:
    """Codex P2: ``write(row, calls)`` is the public, canonical path; it must
    not be able to produce the very invariant violation
    :meth:`validate_invariants` is designed to catch.

    Three failure modes are rejected at the boundary, before any disk side
    effect: empty calls, mismatched call.audit_id, mismatched call.run_id.
    """

    def test_write_rejects_empty_calls(self, store: AuditStore) -> None:
        with pytest.raises(ValueError, match="at least one"):
            store.write(_row(), [])

    def test_write_rejects_mismatched_call_audit_id(
        self, store: AuditStore
    ) -> None:
        row = _row(audit_id="a-row")
        bad = _call(call_id="c1", audit_id="a-other", run_id=row.run_id)

        with pytest.raises(ValueError, match="audit_id"):
            store.write(row, [bad])

    def test_write_rejects_mismatched_call_run_id(self, store: AuditStore) -> None:
        row = _row(audit_id="a1", run_id="r-row")
        bad = _call(call_id="c1", audit_id="a1", run_id="r-other")

        with pytest.raises(ValueError, match="run_id"):
            store.write(row, [bad])

    def test_rejection_does_not_persist_anything(self, store: AuditStore) -> None:
        # A failed write must NOT leave the llm_calls partially written —
        # otherwise we re-create the orphan-call state the rejection was meant
        # to prevent.
        with pytest.raises(ValueError):
            store.write(
                _row(audit_id="a1"),
                [_call(call_id="c1", audit_id="OTHER")],
            )

        assert store.read_audit_results() == ()
        assert store.read_llm_calls() == ()


class TestModelsEnforceTzAwareUTC:
    """Codex P2: every persisted timestamp is tz-aware UTC. Naive datetimes
    are rejected at construction; non-UTC aware datetimes are normalized to
    UTC.

    WHY: the store-level invariant "all persisted timestamps are tz-aware UTC"
    is asserted in CONTEXT.md and depended on by downstream comparisons
    (cold-storage cutoff, snapshot rotation). Allowing naive values in
    silently breaks ``request_timestamp < older_than`` comparisons in
    migrate_cold_storage, and lets local-time rows leak into the dashboard.
    """

    def test_audit_row_rejects_naive_run_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            _row(run_timestamp=datetime(2026, 5, 1, 12, 0, 0))  # naive

    def test_audit_row_rejects_naive_order_datetime(self) -> None:
        with pytest.raises(ValidationError):
            _row(order_datetime=datetime(2026, 5, 1, 8, 30, 0))  # naive

    def test_audit_row_normalizes_non_utc_to_utc(self) -> None:
        from zoneinfo import ZoneInfo

        bkk = ZoneInfo("Asia/Bangkok")
        row = _row(run_timestamp=datetime(2026, 5, 1, 12, 0, 0, tzinfo=bkk))

        assert row.run_timestamp == datetime(2026, 5, 1, 5, 0, 0, tzinfo=UTC)
        assert row.run_timestamp.tzinfo is not None
        assert row.run_timestamp.utcoffset() == timedelta(0)

    def test_optional_vitals_timestamp_allows_none(self) -> None:
        row = _row(vitals_timestamp=None, vitals_sbp=None, vitals_hr=None, vitals_source=None)

        assert row.vitals_timestamp is None

    def test_optional_vitals_timestamp_rejects_naive(self) -> None:
        with pytest.raises(ValidationError):
            _row(vitals_timestamp=datetime(2026, 5, 1, 8, 0, 0))  # naive

    def test_llm_call_rejects_naive_request_timestamp(self) -> None:
        with pytest.raises(ValidationError):
            _call(request_timestamp=datetime(2026, 5, 1, 12, 0, 0))  # naive

    def test_llm_call_normalizes_non_utc_request_timestamp(self) -> None:
        from zoneinfo import ZoneInfo

        bkk = ZoneInfo("Asia/Bangkok")
        call = _call(request_timestamp=datetime(2026, 5, 1, 12, 0, 0, tzinfo=bkk))

        assert call.request_timestamp == datetime(2026, 5, 1, 5, 0, 0, tzinfo=UTC)


class TestIdentifiersAreFilesystemSafe:
    """Codex P2 round 6: ``audit_id``, ``run_id``, ``call_id`` are interpolated
    into filenames. Raw upstream strings containing ``/`` or path-traversal
    segments would let writes land outside the intended dataset directory.

    The defense lives at the model boundary: identifiers must match a
    conservative pattern (``[A-Za-z0-9._-]+`` plus a non-empty + non-``./..``
    check) so the persistence layer can compose paths with raw interpolation
    safely.
    """

    @pytest.mark.parametrize("bad", ["", "../etc", "foo/bar", "foo\\bar", ".", ".."])
    def test_audit_row_rejects_unsafe_audit_id(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            _row(audit_id=bad)

    @pytest.mark.parametrize("bad", ["", "../etc", "foo/bar", "foo\\bar", ".", ".."])
    def test_audit_row_rejects_unsafe_run_id(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            _row(run_id=bad)

    @pytest.mark.parametrize("bad", ["", "../etc", "foo/bar", "foo\\bar", ".", ".."])
    def test_llm_call_rejects_unsafe_call_id(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            _call(call_id=bad)

    @pytest.mark.parametrize("bad", ["foo/bar", "../etc"])
    def test_llm_call_rejects_unsafe_audit_id(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            _call(audit_id=bad)

    @pytest.mark.parametrize("bad", ["foo/bar", "../etc"])
    def test_llm_call_rejects_unsafe_run_id(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            _call(run_id=bad)

    @pytest.mark.parametrize("good", ["audit-001", "REQ_123", "abc.def", "x_y-z.1"])
    def test_safe_ids_accepted(self, good: str) -> None:
        row = _row(audit_id=good)
        assert row.audit_id == good


class TestRuntimeDependenciesDeclared:
    """Codex P1 round 7: ``bba.audit_store`` imports :mod:`pydantic` at module
    load. Production installs without dev dependencies (e.g. wheel install,
    ``uv sync --no-dev``) must therefore see ``pydantic`` in
    ``[project].dependencies`` — not just in the dev dependency group.

    Parsing ``pyproject.toml`` here keeps the contract explicit: a future
    refactor that demotes pydantic back into dev-only would fail this test
    before it could ship a broken wheel.
    """

    def test_pydantic_in_runtime_dependencies(self) -> None:
        import tomllib
        from pathlib import Path as _Path

        project_root = _Path(__file__).resolve().parents[2]
        pyproject = tomllib.loads((project_root / "pyproject.toml").read_text())

        deps = pyproject["project"]["dependencies"]

        assert any(d.startswith("pydantic") for d in deps), (
            f"pydantic must be declared in [project].dependencies "
            f"because bba.audit_store imports it at module load; "
            f"got {deps!r}"
        )


class TestIdempotencyMarkerIncludesCodeVersion:
    """Codex P2 round 2: ``AuditStoreConfig.code_version`` docstring promises
    that a code-version bump invalidates the cached completion marker so a
    re-run is forced. The marker must therefore be keyed on ``code_version``
    in addition to ``audit_id`` + ``run_id``.

    WHY: a change to audit-store-layer code (schema bump on the payload
    column, fix to audit_id derivation, etc.) that re-uses an upstream
    ``run_id`` would otherwise silently no-op as "already done" and the
    reviewer dashboard would keep showing the stale classification.
    """

    def test_code_version_bump_forces_rewrite(self, tmp_path: Path) -> None:
        root = tmp_path / "store"
        AuditStore(AuditStoreConfig(root_dir=root, code_version="v1.0.0")).write(
            _row(), [_call()]
        )

        result_v2 = AuditStore(
            AuditStoreConfig(root_dir=root, code_version="v2.0.0")
        ).write(_row(), [_call()])

        assert result_v2.skipped_idempotent is False

    def test_same_code_version_still_idempotent(self, tmp_path: Path) -> None:
        root = tmp_path / "store"
        cfg = AuditStoreConfig(root_dir=root, code_version="v1.0.0")
        AuditStore(cfg).write(_row(), [_call()])

        result = AuditStore(cfg).write(_row(), [_call()])

        assert result.skipped_idempotent is True

    def test_cross_version_reads_remain_visible(self, tmp_path: Path) -> None:
        # Reads MUST stay code_version-agnostic: a v2 process must still be
        # able to inspect what v1 committed (for migration, audit, eval).
        # Idempotency is about whether to write again; visibility of prior
        # committed data is a separate concern.
        root = tmp_path / "store"
        AuditStore(AuditStoreConfig(root_dir=root, code_version="v1.0.0")).write(
            _row(audit_id="a-v1", run_id="r1"),
            [_call(call_id="c-v1", audit_id="a-v1", run_id="r1")],
        )

        v2_store = AuditStore(AuditStoreConfig(root_dir=root, code_version="v2.0.0"))
        rows = v2_store.read_audit_results()

        assert len(rows) == 1
        assert rows[0].audit_id == "a-v1"


class TestNestedJsonImmutability:
    """Codex P2 round 2: ``frozen=True`` + tuple containers only guard the
    outer shell. Each dict inside ``indications_json`` /
    ``negative_evidence_json`` / ``delta_hb_window_results`` must itself be
    immutable, otherwise a cached or shared model can be mutated post-hoc
    and the "persisted immutably" promise breaks.

    WHY: the audit chain must be reconstructible six months later (PRD
    §"Output schema"). A nested-dict mutation between construction and
    persistence would produce an in-memory model whose JSON payload does not
    match what the row claims to contain.
    """

    def test_indications_json_dicts_reject_mutation(self) -> None:
        row = _row()
        with pytest.raises(TypeError):
            row.indications_json[0]["code"] = "MUTATED"  # type: ignore[index]

    def test_negative_evidence_json_dicts_reject_mutation(self) -> None:
        row = _row(
            negative_evidence_json=(
                {"code": "no_bleed", "quote": "stable", "source_id": "X:1"},
            ),
        )
        with pytest.raises(TypeError):
            row.negative_evidence_json[0]["code"] = "MUTATED"  # type: ignore[index]

    def test_delta_hb_window_results_dicts_reject_mutation(self) -> None:
        row = _row()
        with pytest.raises(TypeError):
            row.delta_hb_window_results[0]["window"] = "MUTATED"  # type: ignore[index]

    def test_input_dict_mutation_does_not_leak_into_model(self) -> None:
        """Caller-held reference to the input dict cannot mutate the model.

        The validator must defensively copy each input dict before wrapping,
        so a later mutation on the caller's side leaves the persisted model
        unchanged.
        """
        original = {"code": "A", "quote": "q", "source_id": "s", "confidence": 0.9}
        row = _row(indications_json=(original,))

        original["code"] = "MUTATED"

        assert row.indications_json[0]["code"] == "A"

    def test_immutability_survives_json_round_trip(self, store: AuditStore) -> None:
        # Deep-immutability machinery must not break model_dump_json /
        # read-back equality.
        row = _row()
        store.write(row, [_call()])

        rt = store.read_audit_results()[0]

        assert rt == row
        with pytest.raises(TypeError):
            rt.indications_json[0]["code"] = "MUTATED"  # type: ignore[index]


class TestNestedJsonDeepImmutability:
    """Codex P2 round 3: top-level ``MappingProxyType`` is insufficient when
    JSON payloads contain *nested* dicts or lists. The freeze must recurse
    into nested containers; otherwise the reproducibility contract breaks
    one level deep.
    """

    def test_nested_dict_inside_item_rejects_mutation(self) -> None:
        row = _row(
            indications_json=(
                {
                    "code": "B1",
                    "quote": "q",
                    "source_id": "s",
                    "confidence": 0.9,
                    "metadata": {"source": "labexm", "note": "fresh"},
                },
            ),
        )

        with pytest.raises(TypeError):
            row.indications_json[0]["metadata"]["note"] = "MUTATED"  # type: ignore[index]

    def test_nested_list_inside_item_becomes_tuple(self) -> None:
        row = _row(
            indications_json=(
                {
                    "code": "B1",
                    "quote": "q",
                    "source_id": "s",
                    "confidence": 0.9,
                    "tags": ["acute", "tachycardia"],
                },
            ),
        )

        # Lists in input are frozen to tuples; index assignment on a tuple
        # raises TypeError.
        assert isinstance(row.indications_json[0]["tags"], tuple)
        with pytest.raises(TypeError):
            row.indications_json[0]["tags"][0] = "MUTATED"  # type: ignore[index]

    def test_caller_nested_dict_mutation_does_not_leak(self) -> None:
        nested = {"source": "labexm", "note": "fresh"}
        original = {
            "code": "B1",
            "quote": "q",
            "source_id": "s",
            "confidence": 0.9,
            "metadata": nested,
        }
        row = _row(indications_json=(original,))

        nested["note"] = "MUTATED_AFTER"

        assert row.indications_json[0]["metadata"]["note"] == "fresh"

    def test_deep_freeze_round_trip_preserves_equality(
        self, store: AuditStore
    ) -> None:
        row = _row(
            indications_json=(
                {
                    "code": "B1",
                    "quote": "q",
                    "source_id": "s",
                    "confidence": 0.9,
                    "metadata": {"source": "labexm"},
                    "tags": ["acute", "tachy"],
                },
            ),
        )
        store.write(row, [_call()])

        rt = store.read_audit_results()[0]

        assert rt == row


class TestLlmCallNestedJsonImmutability:
    """Codex P2 round 3: ``request_json``, ``response_json``, and
    ``extended_thinking_blocks`` on :class:`LlmCall` are the per-API-call
    reproducibility record. They need the same defensive deep-copy /
    read-only treatment as :class:`AuditRow`'s JSON fields.
    """

    def test_request_json_rejects_top_level_mutation(self) -> None:
        call = _call(request_json={"system": "x", "messages": []})

        with pytest.raises(TypeError):
            call.request_json["system"] = "MUTATED"  # type: ignore[index]

    def test_request_json_rejects_nested_mutation(self) -> None:
        call = _call(
            request_json={"system": "x", "messages": [{"role": "user", "content": "hi"}]},
        )

        with pytest.raises(TypeError):
            call.request_json["messages"][0]["role"] = "MUTATED"  # type: ignore[index]

    def test_response_json_rejects_mutation(self) -> None:
        call = _call(response_json={"id": "msg_1", "stop_reason": "tool_use"})

        with pytest.raises(TypeError):
            call.response_json["id"] = "MUTATED"  # type: ignore[index]

    def test_extended_thinking_block_rejects_mutation(self) -> None:
        call = _call(
            extended_thinking_blocks=({"type": "thinking", "text": "thought A"},),
        )

        with pytest.raises(TypeError):
            call.extended_thinking_blocks[0]["type"] = "MUTATED"  # type: ignore[index]

    def test_caller_dict_mutation_does_not_leak_into_call(self) -> None:
        original_request = {"system": "A", "messages": [{"role": "user"}]}
        call = _call(request_json=original_request)

        original_request["system"] = "MUTATED"
        original_request["messages"][0]["role"] = "MUTATED"

        assert call.request_json["system"] == "A"
        assert call.request_json["messages"][0]["role"] == "user"

    def test_extended_thinking_none_is_still_allowed(self) -> None:
        # The freeze on tuple-of-dicts must not reject None — purely
        # deterministic short-circuits won't include thinking blocks.
        call = _call(extended_thinking_blocks=None)

        assert call.extended_thinking_blocks is None

    def test_llm_call_round_trip_preserves_equality_with_freeze(
        self, store: AuditStore
    ) -> None:
        call = _call(
            request_json={"system": "x", "messages": [{"role": "user", "content": "hi"}]},
            response_json={"id": "msg_1", "stop_reason": "tool_use", "usage": {"input": 100}},
            extended_thinking_blocks=({"type": "thinking", "text": "step 1"},),
        )
        store.write(_row(), [call])

        rt = store.read_llm_calls()[0]

        assert rt == call


class TestCodeVersionInAuditPaths:
    """Codex P1 round 3: a code-version rerun reusing the same
    ``(audit_id, run_id)`` must not overwrite the previous parquet file.
    The audit-result parquet and its commit marker each carry a
    code-version slug so v1 and v2 land in distinct files, preserving the
    append-only contract under code bumps.

    WHY: if v2 writes to the same path as v1, a crash before marking
    complete leaves v2's uncommitted bytes under v1's marker — readers
    would surface a payload that no run has actually committed. Even on
    success, v1's bytes are gone — the audit chain claim "every
    classification reproducible from frozen hashes" is violated.
    """

    def test_v2_does_not_overwrite_v1_parquet(self, tmp_path: Path) -> None:
        root = tmp_path / "store"
        AuditStore(AuditStoreConfig(root_dir=root, code_version="v1.0.0")).write(
            _row(audit_id="a1", run_id="r1"),
            [_call(call_id="c1", audit_id="a1", run_id="r1")],
        )

        AuditStore(AuditStoreConfig(root_dir=root, code_version="v2.0.0")).write(
            _row(audit_id="a1", run_id="r1", reasoning_summary_en="v2 reasoning"),
            [_call(call_id="c2", audit_id="a1", run_id="r1")],
        )

        # Both versions' parquet files must coexist on disk.
        parquets = sorted(p.name for p in (root / "audit_results").glob("*.parquet"))
        assert len(parquets) == 2

    def test_v2_does_not_overwrite_v1_call_parquet(self, tmp_path: Path) -> None:
        # call_id reuse across code-version reruns must not overwrite the
        # earlier file (PRD §10: every API call's record persisted immutably).
        root = tmp_path / "store"
        AuditStore(AuditStoreConfig(root_dir=root, code_version="v1.0.0")).write(
            _row(audit_id="a1", run_id="r1"),
            [
                _call(
                    call_id="c1",
                    audit_id="a1",
                    run_id="r1",
                    response_json={"version_marker": "v1"},
                )
            ],
        )
        AuditStore(AuditStoreConfig(root_dir=root, code_version="v2.0.0")).write(
            _row(audit_id="a1", run_id="r1"),
            [
                _call(
                    call_id="c1",
                    audit_id="a1",
                    run_id="r1",
                    response_json={"version_marker": "v2"},
                )
            ],
        )

        call_files = sorted((root / "llm_calls").glob("call_c1_*.parquet"))
        assert len(call_files) == 2, (
            "v2 rewrote v1's call file; both code_versions must coexist"
        )

    def test_read_llm_calls_filters_by_code_version(self, tmp_path: Path) -> None:
        root = tmp_path / "store"
        AuditStore(AuditStoreConfig(root_dir=root, code_version="v1.0.0")).write(
            _row(audit_id="a1", run_id="r1"),
            [_call(call_id="c-v1", audit_id="a1", run_id="r1")],
        )
        store_v2 = AuditStore(AuditStoreConfig(root_dir=root, code_version="v2.0.0"))
        store_v2.write(
            _row(audit_id="a1", run_id="r1"),
            [_call(call_id="c-v2", audit_id="a1", run_id="r1")],
        )

        only_v2 = store_v2.read_llm_calls(code_version="v2.0.0")

        assert {c.call_id for c in only_v2} == {"c-v2"}

    def test_read_llm_calls_default_returns_every_version(
        self, tmp_path: Path
    ) -> None:
        # Default behavior (no code_version filter) returns calls from every
        # committed version — symmetric with read_audit_results.
        root = tmp_path / "store"
        AuditStore(AuditStoreConfig(root_dir=root, code_version="v1.0.0")).write(
            _row(audit_id="a1", run_id="r1"),
            [_call(call_id="c-v1", audit_id="a1", run_id="r1")],
        )
        store_v2 = AuditStore(AuditStoreConfig(root_dir=root, code_version="v2.0.0"))
        store_v2.write(
            _row(audit_id="a1", run_id="r1"),
            [_call(call_id="c-v2", audit_id="a1", run_id="r1")],
        )

        ids = {c.call_id for c in store_v2.read_llm_calls()}

        assert ids == {"c-v1", "c-v2"}

    def test_cold_storage_preserves_original_call_slug(
        self, tmp_path: Path
    ) -> None:
        # Migration must rewrite each call at its ORIGINAL path. If the
        # migrator runs under a different code_version, the rewrite must not
        # accidentally land at a new (current-slug) path.
        root = tmp_path / "store"
        old_ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        store_v1 = AuditStore(AuditStoreConfig(root_dir=root, code_version="v1.0.0"))
        store_v1.write(
            _row(audit_id="a1", run_id="r1"),
            [
                _call(
                    call_id="c-old",
                    audit_id="a1",
                    run_id="r1",
                    request_timestamp=old_ts,
                )
            ],
        )

        # Migrator runs under v2.0.0 (different code_version).
        store_v2 = AuditStore(AuditStoreConfig(root_dir=root, code_version="v2.0.0"))
        migrate_cold_storage(
            store_v2,
            older_than=datetime(2026, 4, 2, 0, 0, 0, tzinfo=UTC),
        )

        call_files = sorted((root / "llm_calls").glob("call_c-old_*.parquet"))
        assert len(call_files) == 1, (
            f"migration leaked a second file under the migrator's slug "
            f"instead of rewriting in place: {[p.name for p in call_files]}"
        )

    def test_read_audit_results_filters_by_code_version(
        self, tmp_path: Path
    ) -> None:
        """``read_audit_results`` exposes a ``code_version`` filter symmetric
        with ``read_llm_calls``. WHY: SnapshotView materializes from the
        unfiltered read, and dashboards/eval need to scope to a specific
        committed version to disambiguate cross-version duplicates."""
        root = tmp_path / "store"
        AuditStore(AuditStoreConfig(root_dir=root, code_version="v1.0.0")).write(
            _row(audit_id="a-shared", run_id="r1", reasoning_summary_en="v1"),
            [_call(call_id="c-v1", audit_id="a-shared", run_id="r1")],
        )
        store_v2 = AuditStore(AuditStoreConfig(root_dir=root, code_version="v2.0.0"))
        store_v2.write(
            _row(audit_id="a-shared", run_id="r1", reasoning_summary_en="v2"),
            [_call(call_id="c-v2", audit_id="a-shared", run_id="r1")],
        )

        v1_rows = store_v2.read_audit_results(code_version="v1.0.0")
        v2_rows = store_v2.read_audit_results(code_version="v2.0.0")

        assert {r.reasoning_summary_en for r in v1_rows} == {"v1"}
        assert {r.reasoning_summary_en for r in v2_rows} == {"v2"}

    def test_reconcile_scopes_orphans_by_code_version(
        self, tmp_path: Path
    ) -> None:
        """A v2 orphan call must be detected even when v1 has committed the
        same ``audit_id`` — the call belongs to v2's reproducibility chain,
        not v1's, so it cannot be "covered" by v1's audit row.
        """
        root = tmp_path / "store"

        # v1 fully committed: audit row + call.
        AuditStore(AuditStoreConfig(root_dir=root, code_version="v1.0.0")).write(
            _row(audit_id="a-shared", run_id="r1"),
            [_call(call_id="c-v1", audit_id="a-shared", run_id="r1")],
        )

        # v2 phase 1 only (simulate crash before audit-row commit).
        store_v2 = AuditStore(AuditStoreConfig(root_dir=root, code_version="v2.0.0"))
        store_v2._persist_llm_calls(
            [_call(call_id="c-v2-orphan", audit_id="a-shared", run_id="r1")]
        )

        report = store_v2.reconcile("r1")

        assert "c-v2-orphan" in report.orphan_call_ids

    def test_validate_invariants_scopes_by_code_version(
        self, tmp_path: Path
    ) -> None:
        """A v2 audit row without v2 calls must raise even when v1 has matching
        calls — v1's calls don't reproduce the v2 row."""
        root = tmp_path / "store"

        # v1 fully committed.
        AuditStore(AuditStoreConfig(root_dir=root, code_version="v1.0.0")).write(
            _row(audit_id="a-shared", run_id="r1"),
            [_call(call_id="c-v1", audit_id="a-shared", run_id="r1")],
        )

        # v2 audit row staged without v2 calls (test-only injection).
        store_v2 = AuditStore(AuditStoreConfig(root_dir=root, code_version="v2.0.0"))
        store_v2._persist_audit_result(_row(audit_id="a-shared", run_id="r1"))

        with pytest.raises(TransactionalOrderingError, match="a-shared"):
            store_v2.validate_invariants("r1")

    def test_cold_storage_blob_path_per_version_does_not_overwrite(
        self, tmp_path: Path
    ) -> None:
        """A shared ``call_id`` between v1 and v2 reruns must produce
        distinct cold-storage blobs — one per version. Otherwise the later
        migration overwrites the earlier blob and both rewritten rows point
        at the *same* URI, so the older call is no longer recoverable.
        """
        root = tmp_path / "store"
        old_ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

        AuditStore(AuditStoreConfig(root_dir=root, code_version="v1.0.0")).write(
            _row(audit_id="a1", run_id="r1"),
            [
                _call(
                    call_id="c-shared",
                    audit_id="a1",
                    run_id="r1",
                    request_timestamp=old_ts,
                    extended_thinking_blocks=(
                        {"type": "thinking", "text": "v1 thought"},
                    ),
                )
            ],
        )
        store_v2 = AuditStore(AuditStoreConfig(root_dir=root, code_version="v2.0.0"))
        store_v2.write(
            _row(audit_id="a1", run_id="r1"),
            [
                _call(
                    call_id="c-shared",
                    audit_id="a1",
                    run_id="r1",
                    request_timestamp=old_ts,
                    extended_thinking_blocks=(
                        {"type": "thinking", "text": "v2 thought"},
                    ),
                )
            ],
        )

        migrate_cold_storage(
            store_v2,
            older_than=datetime(2026, 4, 2, 0, 0, 0, tzinfo=UTC),
        )

        cold_blobs = sorted((root / "cold_storage").glob("c-shared*.json"))
        assert len(cold_blobs) == 2, (
            f"v2 migration overwrote v1's cold blob: {[p.name for p in cold_blobs]}"
        )

    def test_cold_storage_uri_resolves_to_correct_version_content(
        self, tmp_path: Path
    ) -> None:
        """After migration, each rewritten call's ``cold_storage_uri`` must
        point at its own version's blob content — not the last writer's."""
        import json as _json

        root = tmp_path / "store"
        old_ts = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

        AuditStore(AuditStoreConfig(root_dir=root, code_version="v1.0.0")).write(
            _row(audit_id="a1", run_id="r1"),
            [
                _call(
                    call_id="c-shared",
                    audit_id="a1",
                    run_id="r1",
                    request_timestamp=old_ts,
                    extended_thinking_blocks=(
                        {"type": "thinking", "text": "v1 thought"},
                    ),
                )
            ],
        )
        store_v2 = AuditStore(AuditStoreConfig(root_dir=root, code_version="v2.0.0"))
        store_v2.write(
            _row(audit_id="a1", run_id="r1"),
            [
                _call(
                    call_id="c-shared",
                    audit_id="a1",
                    run_id="r1",
                    request_timestamp=old_ts,
                    extended_thinking_blocks=(
                        {"type": "thinking", "text": "v2 thought"},
                    ),
                )
            ],
        )
        migrate_cold_storage(
            store_v2,
            older_than=datetime(2026, 4, 2, 0, 0, 0, tzinfo=UTC),
        )

        v1_calls = store_v2.read_llm_calls(code_version="v1.0.0")
        v2_calls = store_v2.read_llm_calls(code_version="v2.0.0")

        v1_blob = _json.loads(Path(v1_calls[0].cold_storage_uri or "").read_text())
        v2_blob = _json.loads(Path(v2_calls[0].cold_storage_uri or "").read_text())

        assert v1_blob[0]["text"] == "v1 thought"
        assert v2_blob[0]["text"] == "v2 thought"

    def test_v2_crash_pre_marker_does_not_leak_under_v1_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "store"
        store_v1 = AuditStore(AuditStoreConfig(root_dir=root, code_version="v1.0.0"))
        store_v1.write(
            _row(audit_id="a1", run_id="r1", reasoning_summary_en="v1 reasoning"),
            [_call(call_id="c1", audit_id="a1", run_id="r1")],
        )

        # Crash v2 between phase 2a (parquet) and phase 2b (marker).
        store_v2 = AuditStore(AuditStoreConfig(root_dir=root, code_version="v2.0.0"))

        original_mark = store_v2._mark_complete

        def crash_before_mark(*_a: object, **_kw: object) -> None:
            raise RuntimeError("simulated crash pre-marker")

        monkeypatch.setattr(store_v2, "_mark_complete", crash_before_mark)
        with pytest.raises(RuntimeError):
            store_v2.write(
                _row(audit_id="a1", run_id="r1", reasoning_summary_en="v2 reasoning"),
                [_call(call_id="c2", audit_id="a1", run_id="r1")],
            )
        monkeypatch.setattr(store_v2, "_mark_complete", original_mark)

        # The v1-committed row must still be readable; the v2 parquet must
        # NOT be surfaced under the v1 marker (no payload mismatch leakage).
        rows = store_v1.read_audit_results()
        assert len(rows) == 1
        assert rows[0].reasoning_summary_en == "v1 reasoning"
