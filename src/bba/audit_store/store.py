"""Append-only Parquet store with transactional ordering.

Phase 1 contract (PRD §10):

1. ``llm_calls`` is written first as a safe append. A crash here leaves an
   *orphan*: a call record without a downstream classification. Reconciliation
   finds these and the operator decides whether to re-derive or quarantine.
2. ``audit_results`` is written last as the *commit marker*. An ``audit_results``
   row whose ``audit_id`` has no matching ``llm_calls`` row is a bug; the
   :meth:`AuditStore.validate_invariants` method raises so the failure mode is
   loud instead of silent.

Idempotency follows ``bba.ingest.RunIdentity`` convention: a per-row marker
``_audit_<audit_id>_<run_id>.complete`` is written atomically (write-then-rename)
after both phases land, and a future :meth:`write` with the same identity
no-ops with ``skipped_idempotent=True``.

The two phase methods (:meth:`_persist_llm_calls`, :meth:`_persist_audit_result`)
are deliberately leading-underscore: they exist on the surface so tests can
inject "crashed between phases" states without poking the on-disk layout, but
production callers go through :meth:`write` which orders the phases correctly.

On-disk layout under ``config.root_dir``::

    audit_results/  audit_<audit_id>_<run_id>.parquet   (one file per row)
    llm_calls/      call_<call_id>.parquet              (one file per call)
    _markers/       _audit_<audit_id>_<run_id>.complete (commit marker)
    _snapshots/     audit_results_<YYYY-MM-DD>.parquet  (frozen daily view)
    cold_storage/   <call_id>.json                      (migrated thinking blocks)

The one-file-per-record layout is deliberately simple: append-only semantics
become file-existence semantics, the commit marker is a separate file (so a
crash mid-Parquet-write leaves an unreferenced data file that the next run
ignores), and run_id-scoped reads use a metadata-only column scan.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bba.audit_store.exceptions import TransactionalOrderingError
from bba.audit_store.models import (
    AuditRow,
    AuditStoreConfig,
    LlmCall,
    ReconciliationReport,
    WriteResult,
)


class AuditStore:
    """Append-only Parquet writer for ``audit_results`` and ``llm_calls``.

    Construct once per process. The store is single-writer; concurrent writers
    are out of scope for Phase 1 (PRD §"DuckDB single-writer contention").
    """

    def __init__(self, config: AuditStoreConfig) -> None:
        self._config = config

    @property
    def config(self) -> AuditStoreConfig:
        return self._config

    # -- Public API -----------------------------------------------------------

    def write(self, row: AuditRow, calls: Sequence[LlmCall]) -> WriteResult:
        """Persist ``calls`` (phase 1) then ``row`` (phase 2, commit marker).

        Returns ``WriteResult(skipped_idempotent=True)`` if a prior write with
        the same ``(audit_id, run_id)`` already committed.
        """
        if self._marker_path(row.audit_id, row.run_id).exists():
            return WriteResult(
                audit_id=row.audit_id,
                run_id=row.run_id,
                llm_calls_written=0,
                skipped_idempotent=True,
            )
        self._persist_llm_calls(calls)
        self._persist_audit_result(row)
        return WriteResult(
            audit_id=row.audit_id,
            run_id=row.run_id,
            llm_calls_written=len(calls),
            skipped_idempotent=False,
        )

    def read_audit_results(
        self, run_id: str | None = None
    ) -> tuple[AuditRow, ...]:
        """Read all audit rows, optionally filtered to a single ``run_id``."""
        if not self._audit_dir.exists():
            return ()
        rows: list[AuditRow] = []
        for path in sorted(self._audit_dir.glob("*.parquet")):
            entry = _read_single_record(path)
            if run_id is not None and entry["run_id"] != run_id:
                continue
            rows.append(AuditRow.model_validate_json(entry["payload"]))
        return tuple(rows)

    def read_llm_calls(self, run_id: str | None = None) -> tuple[LlmCall, ...]:
        """Read all llm-call rows, optionally filtered to a single ``run_id``."""
        if not self._calls_dir.exists():
            return ()
        rows: list[LlmCall] = []
        for path in sorted(self._calls_dir.glob("*.parquet")):
            entry = _read_single_record(path)
            if run_id is not None and entry["run_id"] != run_id:
                continue
            rows.append(LlmCall.model_validate_json(entry["payload"]))
        return tuple(rows)

    def validate_invariants(self, run_id: str) -> None:
        """Raise :class:`TransactionalOrderingError` if any ``audit_results`` row
        for ``run_id`` lacks a matching ``llm_calls`` row.

        The opposite direction (orphan calls) is handled by :meth:`reconcile`;
        that is not an error.
        """
        audit_ids_with_results = {r.audit_id for r in self.read_audit_results(run_id=run_id)}
        audit_ids_with_calls = {c.audit_id for c in self.read_llm_calls(run_id=run_id)}
        offenders = audit_ids_with_results - audit_ids_with_calls
        if offenders:
            offenders_sorted = sorted(offenders)
            raise TransactionalOrderingError(
                f"audit_results without matching llm_calls for run_id={run_id!r}: "
                f"{offenders_sorted}"
            )

    def reconcile(self, run_id: str) -> ReconciliationReport:
        """Find ``llm_calls`` rows that have no matching ``audit_results`` row.

        These are the expected fallout of a crash after phase 1 but before
        phase 2. The report names them so the operator can re-emit or quarantine.
        """
        audit_ids_with_results = {r.audit_id for r in self.read_audit_results(run_id=run_id)}
        calls = self.read_llm_calls(run_id=run_id)
        orphan_call_ids = tuple(
            c.call_id for c in calls if c.audit_id not in audit_ids_with_results
        )
        audit_ids_with_calls = {c.audit_id for c in calls}
        orphan_audit_ids = tuple(sorted(audit_ids_with_results - audit_ids_with_calls))
        return ReconciliationReport(
            run_id=run_id,
            orphan_call_ids=orphan_call_ids,
            orphan_audit_ids=orphan_audit_ids,
        )

    # -- Phase-level seams (test-only; prod callers use :meth:`write`) --------

    def _persist_llm_calls(self, calls: Sequence[LlmCall]) -> None:
        """Phase 1: append ``calls`` to ``llm_calls.parquet``.

        Exposed so tests can stage "crashed after phase 1" states. The leading
        underscore signals that production code MUST go through :meth:`write`,
        which orders the phases correctly.
        """
        self._calls_dir.mkdir(parents=True, exist_ok=True)
        for call in calls:
            self._persist_call_record(call)

    def _persist_audit_result(self, row: AuditRow) -> None:
        """Phase 2: append ``row`` to ``audit_results.parquet`` and drop the
        idempotency marker.

        Exposed so tests can force the "audit_results without llm_calls"
        invariant violation without touching disk layout.
        """
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        path = self._audit_dir / f"audit_{row.audit_id}_{row.run_id}.parquet"
        _write_single_record(
            path,
            {
                "audit_id": row.audit_id,
                "run_id": row.run_id,
                "payload": row.model_dump_json(),
            },
        )
        self._mark_complete(row.audit_id, row.run_id)

    # -- Package-internal helpers --------------------------------------------

    def _persist_call_record(self, call: LlmCall) -> None:
        """Write a single :class:`LlmCall` record to its per-call Parquet file.

        Cold-storage migration also calls this to overwrite a call record with
        the cleared ``extended_thinking_blocks`` + populated ``cold_storage_uri``.
        """
        self._calls_dir.mkdir(parents=True, exist_ok=True)
        path = self._calls_dir / f"call_{call.call_id}.parquet"
        _write_single_record(
            path,
            {
                "call_id": call.call_id,
                "audit_id": call.audit_id,
                "run_id": call.run_id,
                "payload": call.model_dump_json(),
            },
        )

    def _mark_complete(self, audit_id: str, run_id: str) -> None:
        self._markers_dir.mkdir(parents=True, exist_ok=True)
        target = self._marker_path(audit_id, run_id)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text("ok\n", encoding="utf-8")
        tmp.replace(target)

    # -- Paths ---------------------------------------------------------------

    @property
    def _audit_dir(self) -> Path:
        return self._config.root_dir / "audit_results"

    @property
    def _calls_dir(self) -> Path:
        return self._config.root_dir / "llm_calls"

    @property
    def _markers_dir(self) -> Path:
        return self._config.root_dir / "_markers"

    @property
    def snapshots_dir(self) -> Path:
        """Public so :class:`SnapshotView` can locate the daily snapshot files."""
        return self._config.root_dir / "_snapshots"

    @property
    def cold_storage_dir(self) -> Path:
        """Public so :func:`migrate_cold_storage` can spill blocks here."""
        return self._config.root_dir / "cold_storage"

    def _marker_path(self, audit_id: str, run_id: str) -> Path:
        return self._markers_dir / f"_audit_{audit_id}_{run_id}.complete"


def _write_single_record(path: Path, data: dict[str, str]) -> None:
    """Write one record to ``path`` as a one-row Parquet file.

    The columns are stored as Arrow ``string`` to keep the on-disk schema
    stable across pydantic-model changes (the full record lives in ``payload``;
    surface columns are duplicated for predicate-pushdown queries).
    """
    table = pa.table({k: [v] for k, v in data.items()})
    pq.write_table(table, path)


def _read_single_record(path: Path) -> dict[str, str]:
    table = pq.read_table(path)
    record = table.to_pylist()[0]
    return {k: str(v) for k, v in record.items()}
