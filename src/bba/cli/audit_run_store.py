"""File-backed :class:`~bba.cli.store_protocol.AuditRunStore` adapter.

Phase 1 ships a deliberately simple implementation: a directory tree
under ``BBA_DATA_DIR/audit_runs/`` with one marker file per completed
run, one row marker per audited row, an exclusive ``fcntl.flock`` per
run for the check-then-act sequence, and one JSONL file capturing the
``audit_log`` of ``--force`` overrides.

The simpler-than-Postgres choice is intentional. The CLI's contract on
:class:`~bba.cli.store_protocol.AuditRunStore` is narrow — six methods,
all on a single ``run_id`` key — and the durability requirements are
write-once-then-read-many. A flat file layout satisfies both at zero
operational cost and is trivial to audit by hand. A future
Postgres-backed adapter (covered by :mod:`bba.audit_store`'s extension
ticket) can implement the same Protocol and swap in via
:func:`bba.cli.main._get_audit_run_store` without touching the CLI.

**Platform support.** This adapter relies on :mod:`fcntl` for advisory
file locking and on POSIX-atomic ``O_APPEND`` writes for the audit log.
Both are available on Linux and macOS (the Phase 1 deployment targets);
Windows is out of scope. A future adapter for Windows would either use
:mod:`msvcrt`-based locking or move to the Postgres backend.

On-disk layout::

    <data_dir>/audit_runs/
        run_<run_id>.complete                 — marker; touched after pipeline
        run_<run_id>.lock                     — fcntl.flock target per run
        rows_<run_id>/<audit_id>.row          — one marker per audited row
        audit_log.jsonl                       — appended; one JSON line per event
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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
        """Append a JSON line to ``audit_log.jsonl`` documenting the override.

        Atomic by construction: one ``os.write(2)`` on an
        ``O_APPEND``-opened file descriptor. POSIX guarantees atomic
        appends up to ``PIPE_BUF`` (≥ 4096 bytes); each entry is
        ~150–250 bytes, well under the limit. Concurrent ``--force``
        invocations therefore cannot interleave bytes inside a single
        JSON line."""
        entry: dict[str, object] = {
            "run_id": run_id,
            "idempotency_override": True,
            "reason": reason,
            "ts": datetime.now(tz=UTC).isoformat(),
        }
        line = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
        self._root.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            self._audit_log_path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o644,
        )
        try:
            # ``os.write`` may return a short count on some platforms;
            # loop until the entire JSONL line is written. Each
            # iteration starts a fresh ``write(2)`` syscall, which on
            # POSIX with ``O_APPEND`` is atomic up to ``PIPE_BUF`` — so
            # a short write followed by a concurrent writer's full
            # write cannot interleave bytes inside our line.
            view = memoryview(line)
            while view:
                written = os.write(fd, view)
                if written == 0:
                    raise OSError(
                        f"unexpected zero-byte write to {self._audit_log_path}"
                    )
                view = view[written:]
            # fsync so a power-loss between this call and the next
            # idempotency_override cannot eat the compliance record
            # (PRD §10 "audit_log is the immutable compliance surface").
            os.fsync(fd)
        finally:
            os.close(fd)

    def audit_log_entries(self, run_id: str, /) -> tuple[Mapping[str, object], ...]:
        """Return all ``audit_log.jsonl`` rows matching ``run_id`` in order."""
        return tuple(self._iter_audit_log(run_id))

    @contextmanager
    def acquire_run_lock(self, run_id: str, /) -> Iterator[None]:
        """Exclusive, blocking, ``fcntl.flock``-backed run-level lock.

        Two concurrent ``bba audit`` invocations on the same ``run_id``
        block here so only one passes the check-then-act sequence;
        invocations on *different* ``run_id``'s never contend (the lock
        path is per-run). Crash safety is OS-provided: a crashed
        process releases its kernel-held flock automatically, so a
        retry after a crash does not stall."""
        self._root.mkdir(parents=True, exist_ok=True)
        lock_path = self._lock_path(run_id)
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    # -- helpers -------------------------------------------------------------

    def mark_run_complete(self, run_id: str, /) -> None:
        """Touch the ``run_<run_id>.complete`` marker (write-then-rename).

        The CLI's idempotency contract depends on the marker existing
        iff the run truly committed every row; atomic write-then-rename
        rules out a half-formed marker from a crashed write."""
        self._root.mkdir(parents=True, exist_ok=True)
        target = self._marker_path(run_id)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text("ok\n", encoding="utf-8")
        tmp.replace(target)

    def record_row(self, run_id: str, audit_id: str, /) -> None:
        """Touch a per-row marker under ``rows_<run_id>/<audit_id>.row``.

        Called by the audit pipeline once per committed audit row so
        :meth:`run_count` reflects real progress and stays consistent
        with :meth:`run_complete`."""
        rows_dir = self._rows_dir(run_id)
        rows_dir.mkdir(parents=True, exist_ok=True)
        (rows_dir / f"{audit_id}.row").write_text("ok\n", encoding="utf-8")

    def _marker_path(self, run_id: str) -> Path:
        return self._root / f"run_{run_id}.complete"

    def _rows_dir(self, run_id: str) -> Path:
        return self._root / f"rows_{run_id}"

    def _lock_path(self, run_id: str) -> Path:
        return self._root / f"run_{run_id}.lock"

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
