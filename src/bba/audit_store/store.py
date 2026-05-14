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

        Rejects (without any disk side effect) callers that would produce the
        very invariant violation :meth:`validate_invariants` is designed to
        catch — empty ``calls``, or any call whose ``audit_id``/``run_id``
        does not match ``row``.
        """
        if self._is_already_committed(row.audit_id, row.run_id):
            return WriteResult(
                audit_id=row.audit_id,
                run_id=row.run_id,
                llm_calls_written=0,
                skipped_idempotent=True,
            )

        _validate_calls_match_row(row, calls)

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
        """Read all audit rows, optionally filtered to a single ``run_id``.

        Commit-marker gated: a parquet file whose ``_audit_<audit_id>_<run_id>.complete``
        marker is missing is treated as uncommitted (crashed between phase 2a
        parquet-write and phase 2b mark) and elided from the result.
        """
        if not self._audit_dir.exists():
            return ()
        rows: list[AuditRow] = []
        for path in sorted(self._audit_dir.glob("*.parquet")):
            entry = _read_single_record(path)
            if run_id is not None and entry["run_id"] != run_id:
                continue
            if not self._marker_path(entry["audit_id"], entry["run_id"]).exists():
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
        """Phase 2 (parquet + commit marker): atomically append ``row`` and
        drop the idempotency / commit marker.

        Exposed so tests can force the "audit_results without llm_calls"
        invariant violation without touching disk layout.
        """
        self._persist_audit_parquet_only(row)
        self._mark_complete(row.audit_id, row.run_id)

    def _persist_audit_parquet_only(self, row: AuditRow) -> None:
        """Phase 2a only: write the audit parquet WITHOUT the commit marker.

        Tests use this to stage a "crashed between phase 2a (parquet write)
        and phase 2b (mark complete)" state. Production code MUST go through
        :meth:`_persist_audit_result` or :meth:`write`, which order the
        sub-phases correctly.
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
        """Drop the commit marker, stamped with the current ``code_version``.

        The marker filename is keyed on ``(audit_id, run_id)`` (version-agnostic,
        so cross-version reads stay consistent); the *content* is the writer's
        ``code_version``, consulted by :meth:`_is_already_committed` to decide
        whether a same-key re-run should no-op.
        """
        self._markers_dir.mkdir(parents=True, exist_ok=True)
        target = self._marker_path(audit_id, run_id)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(self._config.code_version + "\n", encoding="utf-8")
        tmp.replace(target)

    def _is_already_committed(self, audit_id: str, run_id: str) -> bool:
        """True iff a commit marker exists AND it was written by this
        ``code_version``.

        A marker stamped with a *different* ``code_version`` does not count
        as committed-for-this-run — the docstring on ``AuditStoreConfig``
        promises that a code-version bump forces a re-run.
        """
        marker = self._marker_path(audit_id, run_id)
        if not marker.exists():
            return False
        return marker.read_text(encoding="utf-8").strip() == self._config.code_version

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
    """Write one record to ``path`` as a one-row Parquet file, atomically.

    The columns are stored as Arrow ``string`` to keep the on-disk schema
    stable across pydantic-model changes (the full record lives in ``payload``;
    surface columns are duplicated for predicate-pushdown queries).

    Atomicity: writes go to a ``.tmp`` sibling first and only ``Path.replace``
    onto the final name if ``pq.write_table`` returns successfully. A crash
    mid-write therefore cannot leave a corrupt final-name file that ``read_*``
    would later try to open.
    """
    table = pa.table({k: [v] for k, v in data.items()})
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def _read_single_record(path: Path) -> dict[str, str]:
    table = pq.read_table(path)
    record = table.to_pylist()[0]
    return {k: str(v) for k, v in record.items()}


def _validate_calls_match_row(row: AuditRow, calls: Sequence[LlmCall]) -> None:
    """Pre-write check: at least one call, and every call shares the row's
    ``audit_id`` and ``run_id``.

    Raised before any disk side effect so a rejected write leaves the store
    clean. PRD §10's transactional-ordering invariant treats an
    ``audit_results`` row without a matching ``llm_calls`` row as a bug; this
    check stops the canonical write API from ever producing that state.
    """
    if not calls:
        raise ValueError(
            f"write() requires at least one llm_call for "
            f"audit_id={row.audit_id!r}; a commit-marker without any backing "
            f"call would violate the transactional-ordering invariant"
        )
    for call in calls:
        if call.audit_id != row.audit_id:
            raise ValueError(
                f"llm_call.audit_id={call.audit_id!r} does not match "
                f"audit_row.audit_id={row.audit_id!r}"
            )
        if call.run_id != row.run_id:
            raise ValueError(
                f"llm_call.run_id={call.run_id!r} does not match "
                f"audit_row.run_id={row.run_id!r}"
            )
