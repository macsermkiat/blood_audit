"""Daily-rotated read-only DuckDB snapshot view over ``audit_results``.

PRD §"Split storage": the dashboard reads from a daily snapshot view so an
in-flight batch write does not produce inconsistent reads mid-query. The
snapshot is keyed by date: opening ``SnapshotView.open(store, as_of=today)``
returns a handle over the set of ``audit_results`` rows that existed at the
moment the snapshot file for ``today`` was first materialized; any writes
after that point are invisible to this view until the next day's rotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from bba.audit_store.models import AuditRow
from bba.audit_store.store import AuditStore


@dataclass(frozen=True, slots=True)
class SnapshotView:
    """Read-only handle over a frozen ``audit_results`` snapshot.

    ``SnapshotView.open(store, as_of)`` materializes the snapshot for ``as_of``
    on first call (subsequent opens with the same ``as_of`` return the same
    frozen view). Writes to ``store`` after the materialization point are not
    visible through the returned view.
    """

    store: AuditStore
    as_of: date

    @classmethod
    def open(cls, store: AuditStore, as_of: date) -> SnapshotView:
        """Materialize-if-needed and return a handle to the snapshot for ``as_of``."""
        raise NotImplementedError

    def read_audit_results(self) -> tuple[AuditRow, ...]:
        """Read every audit row visible at this snapshot's materialization point."""
        raise NotImplementedError
