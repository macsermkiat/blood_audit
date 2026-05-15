"""Exceptions raised by ``bba.review_actions``.

Each exception names a single failure mode the dashboard / orchestrator
needs to react to differently. Catching a bare ``Exception`` would conflate
"the DB rejected your mutation" (data-integrity bug) with "you haven't
migrated yet" (operations error) — both deserve loud, separate handling.
"""

from __future__ import annotations


class AppendOnlyViolationError(Exception):
    """Raised when an UPDATE or DELETE was attempted on an append-only table.

    The schema (see ``migrations/versions/``) REVOKEs UPDATE/DELETE on
    ``review_actions`` and ``phi_access_log``, and installs trigger guards
    that raise on the operation. The store's psycopg layer translates the
    Postgres SQLSTATE into this exception so callers don't need to
    introspect raw DB errors.

    A reviewer who wants to "fix" an earlier decision MUST file a NEW action
    (e.g., the override path) — the original decision stays on the timeline.
    """


class OverrideRequiresReasonError(ValueError):
    """``ReviewActionInput`` with ``action='override'`` must carry
    ``override_reason``.

    Subclass of :class:`ValueError` so callers can catch via
    ``except ValueError`` for generic input-validation handling.
    """


class MigrationStateError(Exception):
    """The database schema is not at the expected alembic head.

    Raised before any data-mutating call so a partially-migrated DB cannot
    silently accept writes against a schema that's missing the trigger
    guards (= the entire point of the append-only invariant).
    """


class PhiAccessLogIncompleteError(Exception):
    """A dashboard surfaced un-redacted text without first writing a
    PHI-access log row.

    AC §"PHI-access log completeness". Raised by the integrity checker
    (``ReviewActionsStore.verify_phi_access_completeness``) when an audit
    row's reviewer view was rendered but the corresponding log row is
    missing — distinguishes "bug in the dashboard" from "bug in the store".
    """


__all__ = (
    "AppendOnlyViolationError",
    "MigrationStateError",
    "OverrideRequiresReasonError",
    "PhiAccessLogIncompleteError",
)
