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
"""

from __future__ import annotations

from collections.abc import Sequence

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

    def write(self, row: AuditRow, calls: Sequence[LlmCall]) -> WriteResult:
        """Persist ``calls`` (phase 1) then ``row`` (phase 2, commit marker).

        Returns ``WriteResult(skipped_idempotent=True)`` if a prior write with
        the same ``(audit_id, run_id)`` already committed.
        """
        raise NotImplementedError

    def read_audit_results(
        self, run_id: str | None = None
    ) -> tuple[AuditRow, ...]:
        """Read all audit rows, optionally filtered to a single ``run_id``."""
        raise NotImplementedError

    def read_llm_calls(self, run_id: str | None = None) -> tuple[LlmCall, ...]:
        """Read all llm-call rows, optionally filtered to a single ``run_id``."""
        raise NotImplementedError

    def validate_invariants(self, run_id: str) -> None:
        """Raise :class:`TransactionalOrderingError` if any ``audit_results`` row
        for ``run_id`` lacks a matching ``llm_calls`` row.

        The opposite direction (orphan calls) is handled by :meth:`reconcile`;
        that is not an error.
        """
        raise NotImplementedError

    def reconcile(self, run_id: str) -> ReconciliationReport:
        """Find ``llm_calls`` rows that have no matching ``audit_results`` row.

        These are the expected fallout of a crash after phase 1 but before
        phase 2. The report names them so the operator can re-emit or quarantine.
        """
        raise NotImplementedError

    # -- Phase-level seams (test-only; prod callers use :meth:`write`) --------

    def _persist_llm_calls(self, calls: Sequence[LlmCall]) -> None:
        """Phase 1: append ``calls`` to ``llm_calls.parquet``.

        Exposed so tests can stage "crashed after phase 1" states. The leading
        underscore signals that production code MUST go through :meth:`write`,
        which orders the phases correctly.
        """
        raise NotImplementedError

    def _persist_audit_result(self, row: AuditRow) -> None:
        """Phase 2: append ``row`` to ``audit_results.parquet`` and drop the
        idempotency marker.

        Exposed so tests can force the "audit_results without llm_calls"
        invariant violation without touching disk layout.
        """
        raise NotImplementedError
