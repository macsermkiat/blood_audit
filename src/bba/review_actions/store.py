"""Postgres-backed review_actions store (RED-phase scaffold).

The GREEN-phase implementation will use psycopg with a connection pool, all
queries parameterized, and rely on DB-level revocations + trigger guards
(defined in ``migrations/``) to enforce the append-only invariant. The
Python layer's role is to:

1. Provide a type-safe, frozen-Pydantic-modeled API.
2. Translate Postgres errors (SQLSTATE on the trigger raise) into the
   well-known :class:`bba.review_actions.AppendOnlyViolationError`.
3. Confirm the schema is at the expected migration head BEFORE the first
   write, raising :class:`bba.review_actions.MigrationStateError` otherwise.

This module's public API is finalized; method bodies raise
``NotImplementedError`` until GREEN lands.
"""

from __future__ import annotations

from bba.review_actions.models import (
    PhiAccessInput,
    PhiAccessLog,
    ReviewAction,
    ReviewActionInput,
    ReviewActionsConfig,
)


class ReviewActionsStore:
    """Postgres-backed store for reviewer decisions and PHI access events.

    Construct once per process; the store owns a connection pool. Methods are
    thread-safe (one connection per call from the pool) so multiple reviewer
    sessions can write concurrently without application-level locking — the
    "no lock contention under multi-reviewer load" invariant from PRD §16.

    Callers MUST run :func:`bba.review_actions.apply_migrations` (or have
    the operator run it out-of-band) before constructing the store. The
    constructor checks the alembic version on first use and raises
    :class:`bba.review_actions.MigrationStateError` against an unmigrated DB.
    """

    def __init__(self, config: ReviewActionsConfig) -> None:
        self._config = config

    @property
    def config(self) -> ReviewActionsConfig:
        return self._config

    def record_action(self, action: ReviewActionInput) -> ReviewAction:
        """Persist one reviewer decision; return the DB-assigned row.

        Append-only: a row written here cannot be UPDATEd or DELETEd. A
        reviewer who wants to revise an earlier decision MUST file a NEW
        action (override path); the dashboard renders the decision timeline
        in chronological order.

        Raises:
            AppendOnlyViolationError: if a trigger-guard fires (only possible
                when a future schema migration accidentally re-enables
                mutation — a defense-in-depth check).
            MigrationStateError: if the DB schema is not at head.
        """
        raise NotImplementedError

    def record_phi_access(self, access: PhiAccessInput) -> PhiAccessLog:
        """Persist one PHI-access event; return the DB-assigned row.

        AC §"PHI-access log completeness": every dashboard read of
        un-redacted text MUST flow through this method BEFORE surfacing the
        text. The dashboard (#26) is the enforcement point; this store
        provides the write path and the integrity checker.
        """
        raise NotImplementedError

    def list_actions(
        self,
        *,
        audit_id: str | None = None,
        reviewer_id: str | None = None,
    ) -> tuple[ReviewAction, ...]:
        """List review actions, optionally filtered.

        Both filters are optional and AND together when both are supplied.
        Results are ordered by ``(created_at, action_id)`` ascending —
        the chronological reviewer-decision timeline for an audit row.
        """
        raise NotImplementedError

    def list_phi_access(
        self,
        *,
        reviewer_id: str | None = None,
        audit_id: str | None = None,
    ) -> tuple[PhiAccessLog, ...]:
        """List PHI-access events, optionally filtered.

        Both filters AND together. Ordered by ``(accessed_at, access_id)``
        ascending; supports the reviewer-activity audit view.
        """
        raise NotImplementedError

    def verify_phi_access_completeness(
        self, *, reviewer_id: str, audit_id: str
    ) -> bool:
        """Return ``True`` iff the reviewer has at least one PHI-access log
        row for the given audit row.

        Used by the dashboard's integrity check to assert that any
        un-redacted-text render was logged. A ``False`` return means either
        the dashboard surfaced text without logging (bug) or the
        ``(reviewer_id, audit_id)`` pair has not yet been accessed (expected).
        Callers know which context applies and react accordingly.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the connection pool. Idempotent — safe to call multiple times."""
        raise NotImplementedError

    def __enter__(self) -> ReviewActionsStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ("ReviewActionsStore",)
