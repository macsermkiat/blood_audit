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

import hashlib
from collections.abc import Iterator, Sequence
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

        Commit-marker gated: each parquet file has a matching marker filename
        derived 1:1 from the parquet's own stem (so the marker is keyed on
        the same ``(audit_id, run_id, code_version_slug)`` triple). A parquet
        without its marker is treated as uncommitted (crashed between phase
        2a and phase 2b) and elided.

        Returns rows from every committed ``code_version`` — cross-version
        reads stay consistent for migration / audit / eval.
        """
        if not self._audit_dir.exists():
            return ()
        rows: list[AuditRow] = []
        for path in sorted(self._audit_dir.glob("*.parquet")):
            entry = _read_single_record(path)
            if run_id is not None and entry["run_id"] != run_id:
                continue
            if not self._marker_for_parquet(path).exists():
                continue
            rows.append(AuditRow.model_validate_json(entry["payload"]))
        return tuple(rows)

    def read_llm_calls(
        self,
        run_id: str | None = None,
        code_version: str | None = None,
    ) -> tuple[LlmCall, ...]:
        """Read all llm-call rows.

        Both filters are optional and AND together when both are supplied.
        ``code_version`` matches the writer's ``AuditStoreConfig.code_version``;
        without it, calls from every committed code-version are returned (the
        default is symmetric with :meth:`read_audit_results`). Pass
        ``code_version`` to reproduce the exact calls that produced an
        audit row at a specific version.
        """
        if not self._calls_dir.exists():
            return ()
        target_slug = (
            _slugify_code_version(code_version) if code_version is not None else None
        )
        rows: list[LlmCall] = []
        for path in sorted(self._calls_dir.glob("*.parquet")):
            entry = _read_single_record(path)
            if run_id is not None and entry["run_id"] != run_id:
                continue
            if target_slug is not None and entry["code_version_slug"] != target_slug:
                continue
            rows.append(LlmCall.model_validate_json(entry["payload"]))
        return tuple(rows)

    def validate_invariants(self, run_id: str) -> None:
        """Raise :class:`TransactionalOrderingError` if any ``audit_results`` row
        for ``run_id`` lacks a matching ``llm_calls`` row *at its own
        code_version*.

        The pairing key is ``(audit_id, code_version_slug)`` — v1's calls
        don't satisfy v2's audit row, even when ``audit_id`` is the same.
        The opposite direction (orphan calls) is handled by :meth:`reconcile`;
        that is not an error.
        """
        audit_pairs = {
            (row.audit_id, slug)
            for row, slug in self._iter_audit_records()
            if row.run_id == run_id
        }
        call_pairs = {
            (call.audit_id, slug)
            for call, slug in self._iter_call_records()
            if call.run_id == run_id
        }
        offenders = audit_pairs - call_pairs
        if offenders:
            offending_ids = sorted({audit_id for audit_id, _slug in offenders})
            raise TransactionalOrderingError(
                f"audit_results without matching llm_calls for run_id={run_id!r}: "
                f"{offending_ids}"
            )

    def reconcile(self, run_id: str) -> ReconciliationReport:
        """Find ``llm_calls`` rows that have no matching ``audit_results`` row
        *at the same ``code_version_slug``*.

        These are the expected fallout of a crash after phase 1 but before
        phase 2. The report names them so the operator can re-emit or
        quarantine. A v1 audit row never satisfies a v2 call: the pairing is
        per-version, since the call belongs to its writer's reproducibility
        chain.
        """
        audit_pairs = {
            (row.audit_id, slug)
            for row, slug in self._iter_audit_records()
            if row.run_id == run_id
        }
        audit_ids_with_results = {audit_id for audit_id, _slug in audit_pairs}

        orphan_call_ids: list[str] = []
        seen_call_pairs: set[tuple[str, str]] = set()
        for call, slug in self._iter_call_records():
            if call.run_id != run_id:
                continue
            seen_call_pairs.add((call.audit_id, slug))
            if (call.audit_id, slug) not in audit_pairs:
                orphan_call_ids.append(call.call_id)

        audit_ids_with_calls = {audit_id for audit_id, _slug in seen_call_pairs}
        orphan_audit_ids = tuple(sorted(audit_ids_with_results - audit_ids_with_calls))

        return ReconciliationReport(
            run_id=run_id,
            orphan_call_ids=tuple(orphan_call_ids),
            orphan_audit_ids=orphan_audit_ids,
        )

    def _iter_audit_records(self) -> Iterator[tuple[AuditRow, str]]:
        """Internal: yield ``(audit_row, code_version_slug)`` pairs for every
        *committed* audit row.

        Marker-gated (same elision rule as :meth:`read_audit_results`):
        uncommitted parquets are not surfaced. Used by :meth:`reconcile` and
        :meth:`validate_invariants` so pairing logic operates on the
        ``(audit_id, slug)`` key rather than just ``audit_id``.
        """
        if not self._audit_dir.exists():
            return
        for path in sorted(self._audit_dir.glob("*.parquet")):
            if not self._marker_for_parquet(path).exists():
                continue
            entry = _read_single_record(path)
            row = AuditRow.model_validate_json(entry["payload"])
            yield row, entry["code_version_slug"]

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

        The parquet filename includes the ``code_version`` slug so a v2
        re-run with the same ``(audit_id, run_id)`` lands in a *new* file
        rather than overwriting v1's bytes — the append-only contract holds
        under version bumps. The slug also lands as a surface column so
        :meth:`reconcile` and :meth:`validate_invariants` can pair audit
        rows with their version-matching calls.
        """
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        path = self._audit_parquet_path(row.audit_id, row.run_id)
        _write_single_record(
            path,
            {
                "audit_id": row.audit_id,
                "run_id": row.run_id,
                "code_version_slug": self._code_version_slug,
                "payload": row.model_dump_json(),
            },
        )

    # -- Package-internal helpers --------------------------------------------

    def _persist_call_record(
        self, call: LlmCall, *, code_version_slug: str | None = None
    ) -> None:
        """Write a single :class:`LlmCall` record to its slug-keyed Parquet file.

        ``code_version_slug`` defaults to the current store's slug. Cold-storage
        migration passes the *original* slug (read off the call's parquet
        surface column) so the rewrite lands at the same path where the call
        was first persisted — call_id reuse across code-version reruns
        therefore never overwrites an earlier file.
        """
        slug = code_version_slug if code_version_slug is not None else self._code_version_slug
        self._calls_dir.mkdir(parents=True, exist_ok=True)
        path = self._calls_dir / f"call_{call.call_id}_{slug}.parquet"
        _write_single_record(
            path,
            {
                "call_id": call.call_id,
                "audit_id": call.audit_id,
                "run_id": call.run_id,
                "code_version_slug": slug,
                "payload": call.model_dump_json(),
            },
        )

    def _iter_call_records(self) -> Iterator[tuple[LlmCall, str]]:
        """Internal: yield ``(call, code_version_slug)`` pairs for every call file.

        Cold-storage migration uses this so it can preserve each call's
        original slug on rewrite (rather than re-stamping with the migrator's
        slug, which would orphan the original file).
        """
        if not self._calls_dir.exists():
            return
        for path in sorted(self._calls_dir.glob("*.parquet")):
            entry = _read_single_record(path)
            call = LlmCall.model_validate_json(entry["payload"])
            yield call, entry["code_version_slug"]

    def _mark_complete(self, audit_id: str, run_id: str) -> None:
        """Drop the commit marker for this row at its slug-keyed path.

        The marker filename mirrors the audit parquet's stem (with an
        underscore prefix and ``.complete`` extension), so the two files
        are paired 1:1 and the marker contents need only signal "ok".
        """
        self._markers_dir.mkdir(parents=True, exist_ok=True)
        target = self._marker_path(audit_id, run_id)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text("ok\n", encoding="utf-8")
        tmp.replace(target)

    def _is_already_committed(self, audit_id: str, run_id: str) -> bool:
        """True iff a commit marker for this exact
        ``(audit_id, run_id, code_version)`` exists.

        A marker written by a *different* ``code_version`` lives at a
        different path and does not count — the docstring on
        :class:`AuditStoreConfig` promises that a code-version bump forces
        a re-run, and the disjoint paths make that structural.
        """
        return self._marker_path(audit_id, run_id).exists()

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

    def _audit_parquet_path(self, audit_id: str, run_id: str) -> Path:
        return self._audit_dir / f"{self._row_stem(audit_id, run_id)}.parquet"

    def _marker_path(self, audit_id: str, run_id: str) -> Path:
        return self._markers_dir / f"_{self._row_stem(audit_id, run_id)}.complete"

    def _marker_for_parquet(self, parquet_path: Path) -> Path:
        """Marker path paired with a given audit parquet.

        The marker filename is the parquet stem with an ``_`` prefix and a
        ``.complete`` extension, so the pairing is structural — no need to
        re-parse ``audit_id``/``run_id``/``code_version`` out of either side.
        """
        return self._markers_dir / f"_{parquet_path.stem}.complete"

    def _row_stem(self, audit_id: str, run_id: str) -> str:
        return f"audit_{audit_id}_{run_id}_{self._code_version_slug}"

    @property
    def _code_version_slug(self) -> str:
        """Stable filesystem-safe slug for ``config.code_version``."""
        return _slugify_code_version(self._config.code_version)


def _slugify_code_version(code_version: str) -> str:
    """Stable filesystem-safe 16-char sha256 prefix of ``code_version``.

    Module-level so :meth:`AuditStore.read_llm_calls` and other consumers can
    derive the slug from a caller-supplied ``code_version`` string without
    instantiating a store at that version.
    """
    return hashlib.sha256(code_version.encode("utf-8")).hexdigest()[:16]


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
