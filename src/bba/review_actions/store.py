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

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NoReturn

from psycopg import sql
from psycopg.errors import RaiseException
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from bba.review_actions.exceptions import (
    AppendOnlyViolationError,
    MigrationStateError,
)
from bba.review_actions.migrator import (
    MIGRATIONS_ROOT_DEFAULT,
    current_revision,
    head_revision,
)
from bba.review_actions.models import (
    PhiAccessInput,
    PhiAccessLog,
    ReviewAction,
    ReviewActionInput,
    ReviewActionsConfig,
)


_security_log = logging.getLogger("bba.review_actions.security")
"""Structured logger for security-critical integrity-check failures.

Operators can subscribe to this logger to alert on repeated bypass
attempts (e.g., a deploy that misconfigures the role, an attacker probing
the schema). Each :class:`MigrationStateError` raised by the startup
integrity check emits a ``WARNING``-level record with structured ``extra``
fields naming the failure mode."""


# Postgres ``pg_trigger.tgtype`` bitmask layout:
#   bit 0 (1)  — FOR EACH ROW (else FOR EACH STATEMENT)
#   bit 1 (2)  — BEFORE (else AFTER)
#   bit 2 (4)  — INSERT
#   bit 3 (8)  — DELETE
#   bit 4 (16) — UPDATE
#   bit 5 (32) — TRUNCATE
# Event bits (4/8/16/32) are independent; the migration declares each
# trigger for a single event, so each expected tgtype value has exactly
# one event bit set.

_TGTYPE_BEFORE_UPDATE_ROW = 1 | 2 | 16  # = 19
_TGTYPE_BEFORE_DELETE_ROW = 1 | 2 | 8  # = 11
_TGTYPE_BEFORE_TRUNCATE_STMT = 2 | 32  # = 34 (STATEMENT, no ROW bit)


_REQUIRED_TRIGGER_BINDINGS: tuple[tuple[str, str, int], ...] = (
    ("review_actions_block_update", "review_actions", _TGTYPE_BEFORE_UPDATE_ROW),
    ("review_actions_block_delete", "review_actions", _TGTYPE_BEFORE_DELETE_ROW),
    (
        "review_actions_block_truncate",
        "review_actions",
        _TGTYPE_BEFORE_TRUNCATE_STMT,
    ),
    ("phi_access_log_block_update", "phi_access_log", _TGTYPE_BEFORE_UPDATE_ROW),
    ("phi_access_log_block_delete", "phi_access_log", _TGTYPE_BEFORE_DELETE_ROW),
    (
        "phi_access_log_block_truncate",
        "phi_access_log",
        _TGTYPE_BEFORE_TRUNCATE_STMT,
    ),
)
"""``(trigger_name, table_name, expected_tgtype)`` triples for each
required trigger.

The integrity check verifies each triple (a) exists in the ``public``
schema, (b) is bound to the named table, (c) is bound to the canonical
:data:`_REQUIRED_TRIGGER_FUNCTION` in the ``public`` schema (not some
no-op or same-named function in another schema), (d) is in the
``enabled`` state (``tgenabled = 'O'``), and (e) fires on the right
event + timing (matching ``tgtype``).

A name-and-relname-only check passes when an attacker creates a
same-named trigger in a different schema, swaps the function for one in
a different schema, or changes the event from UPDATE to INSERT — all of
which leave the actual UPDATE/DELETE/TRUNCATE path unguarded."""


_REQUIRED_TRIGGER_FUNCTION = "review_actions_block_mutation"
"""Canonical name of the function bound to every required trigger."""


_PROTECTED_SCHEMA = "public"
"""Schema the migration installs the tables, triggers, and function in.

If the project ever moves to a non-public schema, this constant moves
together with the migration."""


_PROTECTED_TABLES: tuple[str, ...] = ("review_actions", "phi_access_log")


_APPEND_ONLY_SQLSTATE = "P0001"
"""The SQLSTATE the trigger guards raise via ``RAISE EXCEPTION ... USING
ERRCODE = 'P0001'``. The trigger message is the bare token
``append_only_violation`` — both are checked so a future migration that
re-uses SQLSTATE P0001 for a different reason doesn't get mis-translated.

The colon-suffixed form (``append_only_violation:``) was used before
round 2 of codex review, when the trigger message also included the
operation name and table; that form was dropped for info-disclosure
reasons and the matcher was updated accordingly. The migration's
message lives at
``migrations/versions/cfb3b5460004_*.py`` (``RAISE EXCEPTION
'append_only_violation'``)."""


_APPEND_ONLY_MESSAGE_TOKEN = "append_only_violation"


class ReviewActionsStore:
    """Postgres-backed store for reviewer decisions and PHI access events.

    Construct once per process; the store owns a connection pool. Methods
    are thread-safe — multiple reviewer sessions can write concurrently
    without application-level locking. The connection pool's open call is
    deferred until first use so test fixtures can construct stores against
    a not-yet-migrated DB and have ``record_*`` raise
    :class:`MigrationStateError` (rather than the pool failing at open).
    """

    def __init__(
        self,
        config: ReviewActionsConfig,
        *,
        migrations_root: Path = MIGRATIONS_ROOT_DEFAULT,
        require_unprivileged_role: bool = True,
    ) -> None:
        """Construct a store.

        ``migrations_root`` is used by the startup integrity check to compare
        the live DB's alembic revision against the on-disk head. Pass the
        absolute path when constructing from a non-repo-root cwd.

        ``require_unprivileged_role`` defaults to ``True``: the store will
        refuse to operate if its DSN connects as a superuser or as the owner
        of the protected tables (both of which can bypass the trigger guard
        via ``ALTER TABLE ... DISABLE TRIGGER`` or
        ``SET session_replication_role = 'replica'``). Set to ``False`` in
        controlled test contexts where the test connects as the owning role
        on purpose (e.g., to verify a privileged user CAN bypass — a
        regression guard on the privilege model itself).
        """
        self._config = config
        self._migrations_root = migrations_root
        self._require_unprivileged_role = require_unprivileged_role
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

    @contextmanager
    def access_phi(self, access: PhiAccessInput) -> Iterator[PhiAccessLog]:
        """Context manager: write a ``phi_access_log`` row, THEN yield it.

        AC §"PHI-access log completeness" needs the audit trail row to exist
        BEFORE the un-redacted text is surfaced to a human. The dashboard
        (#26) MUST wrap its un-redacted-text retrieval in this context
        manager — the structural pattern is::

            with store.access_phi(PhiAccessInput(...)) as log:
                text = render_unredacted_text_from_audit_store(...)
                return text

        If the body raises, the log row stays committed (write happens at
        ``__enter__``, not on success). This is intentional: an attempted
        access that crashed mid-render is still an access for audit
        purposes. The dashboard never has a path that returns un-redacted
        text without first having logged it.
        """
        log = self.record_phi_access(access)
        try:
            yield log
        finally:
            # No teardown: the log row is committed by record_phi_access().
            pass

    def record_phi_access(self, access: PhiAccessInput) -> PhiAccessLog:
        """Persist one PHI-access event; return the DB-assigned row.

        Callers SHOULD prefer :meth:`access_phi` (context manager) which
        structurally couples the log write to the un-redacted-text
        retrieval. ``record_phi_access`` exists as a primitive for two
        legitimate use cases:

        * **Replay tooling** — a log row was lost; record retroactively
          after reconciliation against application-server logs.
        * **Out-of-band notification** — an audit job records that an
          access happened (e.g., a CSV export ran) without going through
          the dashboard.

        Production dashboards (#26) MUST use :meth:`access_phi`. Linters /
        code review enforce that boundary at the dashboard layer; this
        store cannot tell from a write alone whether the caller is the
        dashboard or the replay tool.

        Audit jobs verify the completeness invariant batch-wise via
        :meth:`verify_phi_access_completeness` against the audit-row
        view-log; a gap there is the canonical "the dashboard bypassed
        access_phi" detector.
        """
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
        """Verify the schema is healthy exactly once per process.

        Three independent checks; each can flag :class:`MigrationStateError`:

        1. **Privilege check** (only if ``require_unprivileged_role=True``):
           the connecting role is not a superuser, does not own the
           protected tables, and is not a member of any role that does.
           All three can bypass the trigger guard via
           ``ALTER TABLE ... DISABLE TRIGGER`` or ``SET ROLE owner`` then
           the same. Runs first so a misconfigured role is rejected before
           the alembic check (which would fail with a confusing
           ``permission denied for alembic_version`` when the role lacks
           SELECT on the version table).

        2. **Alembic head match** — the live DB's current revision set
           equals the on-disk head set. Catches "schema drifted forward
           without redeploy", "forgot to run apply_migrations", and
           multi-head branches that the migrator can't apply
           deterministically.

        3. **Trigger guards intact** — every required trigger exists,
           bound to the right table + right function, and is currently
           ENABLED. Name-only verification is insufficient: a
           ``DISABLE TRIGGER`` keeps the row but neuters the guard, and a
           same-named trigger on the wrong relation would slip past.
        """
        if self._migrated_checked:
            return

        if self._require_unprivileged_role:
            self._check_role_privileges()

        try:
            current = current_revision(
                self._config, migrations_root=self._migrations_root
            )
            head = head_revision(migrations_root=self._migrations_root)
        except FileNotFoundError as exc:
            raise MigrationStateError(f"migrations_root not found: {exc}") from exc
        except RuntimeError as exc:
            # Migrator surfaces multi-head / no-revisions states as
            # RuntimeError; treat as fatal schema-drift signal.
            raise MigrationStateError(f"alembic state error: {exc}") from exc
        if current is None:
            self._raise_security(
                "no_alembic_revisions",
                "no alembic revisions applied — run "
                "bba.review_actions.apply_migrations() before writing",
            )
        if current != head:
            self._raise_security(
                "alembic_revision_drift",
                f"alembic revision drift: live DB at {current!r}, "
                f"on-disk head at {head!r}; run apply_migrations() "
                f"or roll the deployment back",
                current=current,
                head=head,
            )

        self._check_triggers_intact()

        self._migrated_checked = True

    def _check_role_privileges(self) -> None:
        """Reject DSNs whose role can bypass the trigger guard.

        Two bypass surfaces:

        1. **Superuser** — ``rolsuper`` can ``ALTER TABLE ... DISABLE
           TRIGGER`` or ``SET session_replication_role = 'replica'``.
        2. **Membership in the owner role** — any role for which
           ``pg_has_role(current_user, owner, 'MEMBER')`` is true can
           ``SET ROLE owner`` and then ``DISABLE TRIGGER``. ``MEMBER``
           catches both inheriting and non-inheriting members (a
           ``NOINHERIT`` role still has SET-ROLE access via membership);
           ``USAGE`` would miss the non-inheriting case.
        """
        with self._ensure_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT current_user, "
                    " (SELECT rolsuper FROM pg_roles "
                    "  WHERE rolname = current_user)"
                )
                row = cur.fetchone()
                if row is None:
                    self._raise_security(
                        "current_role_introspection_failed",
                        "could not introspect current role",
                    )
                current_user, is_superuser = row

                if is_superuser:
                    self._raise_security(
                        "superuser_role_rejected",
                        f"connecting role {current_user!r} is a superuser; "
                        f"superusers can bypass the trigger guard via "
                        f"DISABLE TRIGGER or session_replication_role. "
                        f"Use an unprivileged role (e.g., "
                        f"review_actions_app) or pass "
                        f"require_unprivileged_role=False explicitly",
                        current_user=current_user,
                    )

                cur.execute(
                    "SELECT DISTINCT tableowner FROM pg_tables "
                    "WHERE tablename = ANY(%s) AND schemaname = %s",
                    (list(_PROTECTED_TABLES), _PROTECTED_SCHEMA),
                )
                owners = [r[0] for r in cur.fetchall()]
                for owner in owners:
                    # 'MEMBER' catches non-inheriting members (NOINHERIT
                    # role can still SET ROLE owner). 'USAGE' alone would
                    # miss those — Codex round 3.
                    cur.execute(
                        "SELECT pg_has_role(%s, %s, 'MEMBER')",
                        (current_user, owner),
                    )
                    has_role = cur.fetchone()
                    if has_role is not None and has_role[0]:
                        self._raise_security(
                            "owner_role_membership_rejected",
                            f"connecting role {current_user!r} has MEMBER "
                            f"access to {owner!r}, which owns the "
                            f"review_actions / phi_access_log tables; "
                            f"MEMBER access enables SET ROLE + DISABLE "
                            f"TRIGGER bypass (inherited or not). Use a "
                            f"role outside the owner's membership chain "
                            f"(e.g., review_actions_app)",
                            current_user=current_user,
                            owner=owner,
                        )

    def _check_triggers_intact(self) -> None:
        """Verify every required trigger exists with the right shape.

        Postgres' ``tgenabled`` values:
          ``'O'`` — origin / enabled (the only acceptable state)
          ``'D'`` — disabled
          ``'R'`` — fires on replica only
          ``'A'`` — fires always

        ``'A'`` would also fire on origin, but the migration installs
        ``'O'``, so anything else signals drift.

        Each required trigger is identified by ``(tgname, relname,
        relnamespace, proname, pronamespace, tgenabled, tgtype)`` — a
        same-named trigger in another schema, or one that fires on the
        wrong event, would slip past a name-only check.
        """
        expected_lookup = {
            (name, table): tgtype for name, table, tgtype in _REQUIRED_TRIGGER_BINDINGS
        }
        with self._ensure_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT t.tgname, "
                    "       c.relname, "
                    "       cn.nspname AS relnsp, "
                    "       p.proname, "
                    "       pn.nspname AS pronsp, "
                    "       t.tgenabled, "
                    "       t.tgtype "
                    "FROM pg_trigger t "
                    "JOIN pg_class c ON t.tgrelid = c.oid "
                    "JOIN pg_namespace cn ON c.relnamespace = cn.oid "
                    "JOIN pg_proc p ON t.tgfoid = p.oid "
                    "JOIN pg_namespace pn ON p.pronamespace = pn.oid "
                    "WHERE NOT t.tgisinternal "
                    "AND t.tgname = ANY(%s) "
                    "AND cn.nspname = %s",
                    (
                        [name for name, _, _ in _REQUIRED_TRIGGER_BINDINGS],
                        _PROTECTED_SCHEMA,
                    ),
                )
                rows = cur.fetchall()

        observed_pairs: set[tuple[str, str]] = set()
        broken: list[str] = []
        for (
            tgname,
            relname,
            relnsp,
            proname,
            pronsp,
            tgenabled,
            tgtype,
        ) in rows:
            # relnsp is already filtered to _PROTECTED_SCHEMA by the query.
            observed_pairs.add((tgname, relname))
            if pronsp != _PROTECTED_SCHEMA:
                broken.append(
                    f"trigger {tgname!r} on {relnsp}.{relname!r} is bound "
                    f"to function {pronsp}.{proname!r}, expected "
                    f"function in schema {_PROTECTED_SCHEMA!r}"
                )
                continue
            if proname != _REQUIRED_TRIGGER_FUNCTION:
                broken.append(
                    f"trigger {tgname!r} on {relname!r} is bound to "
                    f"function {proname!r}, expected "
                    f"{_REQUIRED_TRIGGER_FUNCTION!r}"
                )
                continue
            if tgenabled != "O":
                broken.append(
                    f"trigger {tgname!r} on {relname!r} is not enabled "
                    f"(tgenabled={tgenabled!r}); the guard does not fire"
                )
                continue
            expected_tgtype = expected_lookup.get((tgname, relname))
            if expected_tgtype is not None and tgtype != expected_tgtype:
                broken.append(
                    f"trigger {tgname!r} on {relname!r} has tgtype="
                    f"{tgtype} (expected {expected_tgtype}); event or "
                    f"timing has drifted from the migration"
                )

        missing = sorted(set(expected_lookup.keys()) - observed_pairs)
        if missing:
            self._raise_security(
                "required_triggers_missing",
                f"required trigger guards missing or bound to the wrong "
                f"table/schema: {missing!r}; schema is unsafe to write to",
                missing=missing,
            )
        if broken:
            self._raise_security(
                "trigger_guards_corrupted",
                f"trigger guards are corrupted: {broken}",
                broken=broken,
            )

    def _raise_security(self, event: str, message: str, **fields: object) -> NoReturn:
        """Log a structured security-event record, then raise
        :class:`MigrationStateError`.

        ``event`` is a short stable identifier (``superuser_role_rejected``,
        ``alembic_revision_drift``, etc.) operators can grep / alert on
        without parsing free-form messages. ``fields`` lands in the log
        record's ``extra`` dict for structured-log consumers.
        """
        _security_log.warning(
            "review_actions.security.%s: %s",
            event,
            message,
            extra={
                "review_actions_security_event": event,
                "app_name": self._config.app_name,
                **fields,
            },
        )
        raise MigrationStateError(message)

    @staticmethod
    def _translate_raise(exc: RaiseException) -> Exception:
        """Translate a Postgres ``RAISE EXCEPTION`` into the typed exception
        if it matches the trigger-guard SQLSTATE + message contract.

        Anything else is re-raised as-is via ``raise ... from exc`` so the
        caller can introspect.
        """
        sqlstate = getattr(exc, "sqlstate", None)
        message = str(exc.diag.message_primary) if exc.diag else str(exc)
        if sqlstate == _APPEND_ONLY_SQLSTATE and _APPEND_ONLY_MESSAGE_TOKEN in message:
            return AppendOnlyViolationError(message)
        return exc


__all__ = ("ReviewActionsStore",)
