"""File-backed :class:`~bba.cli.store_protocol.AuditRunStore` adapter.

Phase 1 ships a deliberately simple implementation: a directory tree
under ``BBA_DATA_DIR/audit_runs/`` with one marker file per completed
run, one Parquet/marker per audited row, and one JSONL file capturing
the ``audit_log`` of ``--force`` overrides.

The simpler-than-Postgres choice is intentional. The CLI's contract on
:class:`~bba.cli.store_protocol.AuditRunStore` is narrow — four methods,
all on a single ``run_id`` key — and the durability requirements are
write-once-then-read-many. A flat file layout satisfies both at zero
operational cost and is trivial to audit by hand. A future
Postgres-backed adapter (covered by :mod:`bba.audit_store`'s extension
ticket) can implement the same Protocol and swap in via
:func:`bba.cli.main._get_audit_run_store` without touching the CLI.

On-disk layout::

    <data_dir>/audit_runs/
        run_<run_id>.complete                 — marker; touched after pipeline
        rows_<run_id>/<audit_id>.row          — one marker per audited row
        audit_log.jsonl                       — appended; one JSON line per event
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path


class FileBackedAuditRunStore:
    """File-backed implementation of the :class:`AuditRunStore` Protocol.

    Construct once per CLI invocation with the base data directory. The
    instance lazily creates subdirectories on first write — the data dir
    itself must already exist (callers should resolve it through
    :func:`bba.cli.main._resolve_data_dir` which raises a typed
    :class:`~bba.cli.exceptions.CliError` when ``BBA_DATA_DIR`` is unset).
    """

    def __init__(self, data_dir: Path, /) -> None:
        self._root = data_dir / "audit_runs"

    # -- AuditRunStore Protocol ----------------------------------------------

    def run_complete(self, run_id: str, /) -> bool:
        """Return ``True`` iff the ``run_<run_id>.complete`` marker exists."""
        return self._marker_path(run_id).is_file()

    def run_count(self, run_id: str, /) -> int:
        """Return the count of per-row markers in ``rows_<run_id>/``."""
        rows_dir = self._rows_dir(run_id)
        if not rows_dir.is_dir():
            return 0
        return sum(1 for _ in rows_dir.iterdir())

    def record_idempotency_override(
        self,
        run_id: str,
        /,
        *,
        reason: str,
    ) -> None:
        """Append a JSON line to ``audit_log.jsonl`` documenting the override."""
        entry: dict[str, object] = {
            "run_id": run_id,
            "idempotency_override": True,
            "reason": reason,
            "ts": datetime.now(tz=UTC).isoformat(),
        }
        self._root.mkdir(parents=True, exist_ok=True)
        with self._audit_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False))
            fh.write("\n")

    def audit_log_entries(
        self, run_id: str, /
    ) -> tuple[Mapping[str, object], ...]:
        """Return all ``audit_log.jsonl`` rows matching ``run_id`` in order."""
        return tuple(self._iter_audit_log(run_id))

    # -- helpers -------------------------------------------------------------

    def mark_run_complete(self, run_id: str, /) -> None:
        """Touch the ``run_<run_id>.complete`` marker (write-then-rename).

        Public-but-not-Protocol: the audit pipeline calls this after a
        successful run; the CLI's idempotency contract depends on the
        marker existing iff the run truly committed every row."""
        self._root.mkdir(parents=True, exist_ok=True)
        target = self._marker_path(run_id)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text("ok\n", encoding="utf-8")
        tmp.replace(target)

    def record_row(self, run_id: str, audit_id: str, /) -> None:
        """Touch a per-row marker under ``rows_<run_id>/<audit_id>.row``.

        Public-but-not-Protocol: lets the audit pipeline declare a row
        committed so :meth:`run_count` reflects real progress."""
        rows_dir = self._rows_dir(run_id)
        rows_dir.mkdir(parents=True, exist_ok=True)
        (rows_dir / f"{audit_id}.row").write_text("ok\n", encoding="utf-8")

    def _marker_path(self, run_id: str) -> Path:
        return self._root / f"run_{run_id}.complete"

    def _rows_dir(self, run_id: str) -> Path:
        return self._root / f"rows_{run_id}"

    @property
    def _audit_log_path(self) -> Path:
        return self._root / "audit_log.jsonl"

    def _iter_audit_log(self, run_id: str) -> Iterator[Mapping[str, object]]:
        path = self._audit_log_path
        if not path.is_file():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                entry = json.loads(stripped)
                if entry.get("run_id") == run_id:
                    yield entry
