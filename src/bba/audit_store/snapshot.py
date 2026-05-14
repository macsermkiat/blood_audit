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
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from bba.audit_store.models import AuditRow
from bba.audit_store.store import AuditStore, _slugify_code_version


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
        snapshot_path = _snapshot_path(store, as_of)
        if not snapshot_path.exists():
            _materialize(store, snapshot_path)
        return cls(store=store, as_of=as_of)

    def read_audit_results(
        self, code_version: str | None = None
    ) -> tuple[AuditRow, ...]:
        """Read audit rows visible at this snapshot's materialization point.

        ``code_version=None`` returns every committed version (cross-version,
        symmetric with :meth:`AuditStore.read_audit_results`). Pass
        ``code_version`` to disambiguate when a post-bump rerun put a second
        row at the same ``(audit_id, run_id)`` into this snapshot; otherwise
        dashboards would double-count.
        """
        snapshot_path = _snapshot_path(self.store, self.as_of)
        if not snapshot_path.exists():
            return ()
        table = pq.read_table(snapshot_path)
        payloads = table.column("payload").to_pylist()
        slugs = table.column("code_version_slug").to_pylist()
        target_slug = (
            _slugify_code_version(code_version) if code_version is not None else None
        )
        rows: list[AuditRow] = []
        for payload, slug in zip(payloads, slugs, strict=True):
            if target_slug is not None and slug != target_slug:
                continue
            rows.append(AuditRow.model_validate_json(payload))
        return tuple(rows)


def _snapshot_path(store: AuditStore, as_of: date) -> Path:
    return store.snapshots_dir / f"audit_results_{as_of.isoformat()}.parquet"


def _materialize(store: AuditStore, snapshot_path: Path) -> None:
    """Freeze the current ``audit_results`` set into ``snapshot_path``.

    Each row is stored alongside its ``code_version_slug`` so
    :meth:`SnapshotView.read_audit_results` can filter by version on read
    (Codex P2 round 8). Writes atomically (write-then-rename) so a crash
    mid-write cannot leave a half-formed snapshot.
    """
    store.snapshots_dir.mkdir(parents=True, exist_ok=True)
    records = list(store._iter_audit_records())
    payloads = [row.model_dump_json() for row, _slug in records]
    slugs = [slug for _row, slug in records]
    table = pa.table({"code_version_slug": slugs, "payload": payloads})
    tmp = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(snapshot_path)
