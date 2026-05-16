"""Protocol surface for the run-level audit store used by the CLI.

The CLI only needs four operations on the persistent audit store: ask
whether a run is complete, count how many audit rows belong to a run,
write a compliance row when ``--force`` overrides idempotency, and read
back the audit-log entries for that run (for test/debug introspection).

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
from typing import Protocol, runtime_checkable


@runtime_checkable
class AuditRunStore(Protocol):
    """The CLI's view of the audit store.

    Every method is keyword-only on its run identifier so a test double
    cannot pass a positional surprise. The Protocol is
    :func:`~typing.runtime_checkable` so tests can ``isinstance`` against
    it for fast smoke checks.
    """

    def run_complete(self, run_id: str, /) -> bool:
        """Return ``True`` iff every audit row for ``run_id`` is committed."""

    def run_count(self, run_id: str, /) -> int:
        """Return the number of committed audit rows for ``run_id``."""

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

    def audit_log_entries(
        self, run_id: str, /
    ) -> tuple[Mapping[str, object], ...]:
        """Return the immutable audit-log tuple for ``run_id``.

        Order is insertion-time. Returns an empty tuple if there are no
        entries (e.g. on a fresh run).
        """
