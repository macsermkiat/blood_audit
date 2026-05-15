"""Postgres-backed review_actions store.

Uses psycopg 3 with a thread-safe connection pool (``ConnectionPool``);
each public method checks out one connection for the duration of its call,
so multiple reviewer sessions can write concurrently without
application-level locking. Append-only is enforced at the DB layer
(REVOKE + trigger guard installed by the initial migration); this Python
layer only translates the trigger's SQLSTATE ``P0001`` into a typed
:class:`AppendOnlyViolationError`.

The store assumes the schema is at alembic head. The first call to a
data-mutating method verifies that a single time (lazy check, cached) and
raises :class:`MigrationStateError` against an unmigrated DB.
"""

from __future__ import annotations

import threading

from psycopg import sql
from psycopg.errors import RaiseException
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from bba.review_actions.exceptions import (
    AppendOnlyViolationError,
    MigrationStateError,
)
from bba.review_actions.models import (
    PhiAccessInput,
    PhiAccessLog,
    ReviewAction,
    ReviewActionInput,
    ReviewActionsConfig,
)


_APPEND_ONLY_SQLSTATE = "P0001"
"""The SQLSTATE the trigger guards raise via ``RAISE EXCEPTION ... USING
ERRCODE = 'P0001'``. The trigger message also starts with
``append_only_violation:`` — both are checked so a future migration that
re-uses SQLSTATE P0001 for a different reason doesn't get mis-translated."""


_APPEND_ONLY_MESSAGE_PREFIX = "append_only_violation:"


class ReviewActionsStore:
    """Postgres-backed store for reviewer decisions and PHI access events.

    Construct once per process; the store owns a connection pool. Methods
    are thread-safe — multiple reviewer sessions can write concurrently
    without application-level locking. The connection pool's open call is
    deferred until first use so test fixtures can construct stores against
    a not-yet-migrated DB and have ``record_*`` raise
    :class:`MigrationStateError` (rather than the pool failing at open).
    """

    def __init__(self, config: ReviewActionsConfig) -> None:
        self._config = config
        self._pool: ConnectionPool | None = None
        self._pool_lock = threading.Lock()
        self._migrated_checked = False
        self._closed = False

    @property
    def config(self) -> ReviewActionsConfig:
        return self._config

    # -- Public API -----------------------------------------------------------

    def record_action(self, action: ReviewActionInput) -> ReviewAction:
        """Persist one reviewer decision; return the DB-assigned row."""
        self._ensure_migrated()
        try:
            with self._ensure_pool().connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        sql.SQL(
                            "INSERT INTO review_actions "
                            "(audit_id, reviewer_id, action, "
                            " override_reason, note) "
                            "VALUES (%s, %s, %s, %s, %s) "
                            "RETURNING action_id, audit_id, reviewer_id, "
                            " action, override_reason, note, created_at"
                        ),
                        (
                            action.audit_id,
                            action.reviewer_id,
                            action.action,
                            action.override_reason,
                            action.note,
                        ),
                    )
                    row = cur.fetchone()
                assert row is not None  # RETURNING guarantees one row
                return ReviewAction.model_validate(row)
        except RaiseException as exc:
            raise self._translate_raise(exc) from exc

    def record_phi_access(self, access: PhiAccessInput) -> PhiAccessLog:
        """Persist one PHI-access event; return the DB-assigned row."""
        self._ensure_migrated()
        try:
            with self._ensure_pool().connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        sql.SQL(
                            "INSERT INTO phi_access_log "
                            "(reviewer_id, audit_id, hn_hash, an_hash, "
                            " break_glass_justification) "
                            "VALUES (%s, %s, %s, %s, %s) "
                            "RETURNING access_id, reviewer_id, audit_id, "
                            " hn_hash, an_hash, break_glass_justification, "
                            " accessed_at"
                        ),
                        (
                            access.reviewer_id,
                            access.audit_id,
                            access.hn_hash,
                            access.an_hash,
                            access.break_glass_justification,
                        ),
                    )
                    row = cur.fetchone()
                assert row is not None
                return PhiAccessLog.model_validate(row)
        except RaiseException as exc:
            raise self._translate_raise(exc) from exc

    def list_actions(
        self,
        *,
        audit_id: str | None = None,
        reviewer_id: str | None = None,
    ) -> tuple[ReviewAction, ...]:
        """List review actions, optionally filtered.

        Both filters are optional and AND together. Results are ordered by
        ``(created_at, action_id)`` ascending — the chronological decision
        timeline.
        """
        self._ensure_migrated()
        clauses: list[sql.Composable] = []
        params: list[object] = []
        if audit_id is not None:
            clauses.append(sql.SQL("audit_id = %s"))
            params.append(audit_id)
        if reviewer_id is not None:
            clauses.append(sql.SQL("reviewer_id = %s"))
            params.append(reviewer_id)

        query: sql.Composable = sql.SQL(
            "SELECT action_id, audit_id, reviewer_id, action, "
            " override_reason, note, created_at "
            "FROM review_actions"
        )
        if clauses:
            query = query + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses)
        query = query + sql.SQL(" ORDER BY created_at ASC, action_id ASC")

        with self._ensure_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return tuple(ReviewAction.model_validate(r) for r in rows)

    def list_phi_access(
        self,
        *,
        reviewer_id: str | None = None,
        audit_id: str | None = None,
    ) -> tuple[PhiAccessLog, ...]:
        """List PHI-access events, optionally filtered.

        Both filters AND together. Ordered by ``(accessed_at, access_id)``
        ascending.
        """
        self._ensure_migrated()
        clauses: list[sql.Composable] = []
        params: list[object] = []
        if reviewer_id is not None:
            clauses.append(sql.SQL("reviewer_id = %s"))
            params.append(reviewer_id)
        if audit_id is not None:
            clauses.append(sql.SQL("audit_id = %s"))
            params.append(audit_id)

        query: sql.Composable = sql.SQL(
            "SELECT access_id, reviewer_id, audit_id, hn_hash, an_hash, "
            " break_glass_justification, accessed_at "
            "FROM phi_access_log"
        )
        if clauses:
            query = query + sql.SQL(" WHERE ") + sql.SQL(" AND ").join(clauses)
        query = query + sql.SQL(" ORDER BY accessed_at ASC, access_id ASC")

        with self._ensure_pool().connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return tuple(PhiAccessLog.model_validate(r) for r in rows)

    def verify_phi_access_completeness(
        self, *, reviewer_id: str, audit_id: str
    ) -> bool:
        """Return ``True`` iff the reviewer has at least one PHI-access log
        row for the given audit row."""
        self._ensure_migrated()
        with self._ensure_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM phi_access_log "
                    "WHERE reviewer_id = %s AND audit_id = %s LIMIT 1",
                    (reviewer_id, audit_id),
                )
                return cur.fetchone() is not None

    def close(self) -> None:
        """Close the connection pool. Idempotent."""
        if self._closed:
            return
        self._closed = True
        with self._pool_lock:
            if self._pool is not None:
                self._pool.close()
                self._pool = None

    def __enter__(self) -> ReviewActionsStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    # -- Internal helpers ----------------------------------------------------

    def _ensure_pool(self) -> ConnectionPool:
        if self._closed:
            raise RuntimeError("ReviewActionsStore is closed")
        with self._pool_lock:
            if self._pool is None:
                self._pool = ConnectionPool(
                    conninfo=self._config.dsn,
                    kwargs={"application_name": self._config.app_name},
                    open=True,
                    min_size=1,
                    max_size=8,
                )
            return self._pool

    def _ensure_migrated(self) -> None:
        """Verify the schema is at expected head exactly once per process.

        A missing ``review_actions`` table → :class:`MigrationStateError`.
        Future bumps (head moves forward) are NOT checked here — alembic's
        ``current_revision`` vs ``head_revision`` comparison is the
        dashboard's bootstrap responsibility. This check guards against the
        common "forgot to migrate" mode, not against version drift between
        a running store and a freshly-deployed migration.
        """
        if self._migrated_checked:
            return
        with self._ensure_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = current_schema() "
                    "AND table_name = 'review_actions' LIMIT 1"
                )
                exists = cur.fetchone() is not None
        if not exists:
            raise MigrationStateError(
                "review_actions table not found — run "
                "bba.review_actions.apply_migrations() before writing"
            )
        self._migrated_checked = True

    @staticmethod
    def _translate_raise(exc: RaiseException) -> Exception:
        """Translate a Postgres ``RAISE EXCEPTION`` into the typed exception
        if it matches the trigger-guard SQLSTATE + message contract.

        Anything else is re-raised as-is via ``raise ... from exc`` so the
        caller can introspect.
        """
        sqlstate = getattr(exc, "sqlstate", None)
        message = str(exc.diag.message_primary) if exc.diag else str(exc)
        if (
            sqlstate == _APPEND_ONLY_SQLSTATE
            and _APPEND_ONLY_MESSAGE_PREFIX in message
        ):
            return AppendOnlyViolationError(message)
        return exc


__all__ = ("ReviewActionsStore",)
