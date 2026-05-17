"""Protocol surface for the run-level audit store used by the CLI.

The CLI needs six operations on the persistent audit store: ask whether
a run is complete, count how many audit rows belong to a run, record
one row as committed, mark the whole run complete, write a compliance
row when ``--force`` overrides idempotency, read back the audit-log
entries, and acquire an exclusive lock around the check-then-act
sequence so concurrent invocations on the same input cannot
double-execute the pipeline.

This Protocol exists so the CLI can be unit-tested with an in-memory
double without bringing up a Postgres testcontainer — and so the *real*
implementation in :mod:`bba.audit_store` can grow without changing the
CLI surface.

A separate (issue-#19 follow-up) ticket will add a concrete
``AuditRunStore`` adapter to :mod:`bba.audit_store`; the CLI imports
*only* through this Protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable


@runtime_checkable
class AuditRunStore(Protocol):
    """The CLI's view of the audit store.

    The Protocol is :func:`~typing.runtime_checkable` so tests can
    ``isinstance`` against it for fast smoke checks. Positional-only
    ``run_id`` keeps the surface mock-friendly.
    """

    def run_complete(self, run_id: str, /) -> bool:
        """Return ``True`` iff every audit row for ``run_id`` is committed."""

    def run_count(self, run_id: str, /) -> int:
        """Return the number of committed audit rows for ``run_id``."""

    def record_row(self, run_id: str, audit_id: str, /) -> None:
        """Mark one audit row as committed under ``run_id``.

        Called by the audit pipeline for each row it persists. The
        post-condition is that :meth:`run_count` strictly increases by
        one — so a future :meth:`run_complete` that returns ``True``
        agrees with the observed row count."""

    def mark_run_complete(self, run_id: str, /) -> None:
        """Persist that ``run_id`` finished successfully.

        Called by ``bba audit`` after :func:`_run_audit_pipeline` returns
        without raising; the next invocation with the same identity
        triggers the no-op branch. Atomicity is the implementation's
        responsibility (the file-backed adapter uses write-then-rename
        so a crash mid-write cannot leave a half-formed marker)."""

    def record_idempotency_override(
        self,
        run_id: str,
        /,
        *,
        reason: str,
    ) -> None:
        """Append an ``idempotency_override=true`` row to ``audit_log``.

        Called once per ``bba audit --force`` invocation. Persistence is
        synchronous: a call that returns must be durable, so a crash
        between the two ``--force`` invocations cannot lose either
        compliance entry.
        """

    def audit_log_entries(self, run_id: str, /) -> tuple[Mapping[str, object], ...]:
        """Return the immutable audit-log tuple for ``run_id``.

        Order is insertion-time. Returns an empty tuple if there are no
        entries (e.g. on a fresh run).
        """

    def acquire_run_lock(self, run_id: str, /) -> AbstractContextManager[None]:
        """Return a context manager that holds an exclusive run-level lock.

        ``bba audit`` wraps the check-then-act sequence
        (``run_complete`` → run pipeline → ``mark_run_complete``) in
        ``with store.acquire_run_lock(run_id):`` so two concurrent
        invocations on the same input cannot both pass the
        ``run_complete`` guard and double-execute the pipeline.

        The lock is exclusive *per* ``run_id``; concurrent audits of
        *different* inputs run in parallel. The implementation must
        release the lock when the context manager exits, including on
        an exception inside the body."""
