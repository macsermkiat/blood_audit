"""bba.review_actions — Postgres-backed mutable state + PHI-access log.

See issue #25 for acceptance criteria. PRD §16 Implementation Decisions:

* Append-only Postgres tables for reviewer decisions and PHI-access events.
* DB-level UPDATE/DELETE revocation + trigger guard (defense in depth).
* Every dashboard access of un-redacted text writes a row in
  ``phi_access_log`` BEFORE the text is surfaced.
* Multi-reviewer concurrent writes without lock contention (chosen over
  SQLite for this reason).
* Schema lives in ``<repo_root>/migrations/`` via alembic.

This module is the persistence dependency for #26 (dashboard).
"""

from bba.review_actions.exceptions import (
    AppendOnlyViolationError,
    MigrationStateError,
    OverrideRequiresReasonError,
    PhiAccessLogIncompleteError,
)
from bba.review_actions.migrator import (
    MIGRATIONS_ROOT_DEFAULT,
    apply_migrations,
    current_revision,
    head_revision,
)
from bba.review_actions.models import (
    ACTION_KINDS,
    ActionKind,
    NonEmptyStr,
    PhiAccessInput,
    PhiAccessLog,
    ReviewAction,
    ReviewActionInput,
    ReviewActionsConfig,
    SafeId,
    UTCDatetime,
)
from bba.review_actions.store import ReviewActionsStore


__all__ = [
    "ACTION_KINDS",
    "ActionKind",
    "AppendOnlyViolationError",
    "MIGRATIONS_ROOT_DEFAULT",
    "MigrationStateError",
    "NonEmptyStr",
    "OverrideRequiresReasonError",
    "PhiAccessInput",
    "PhiAccessLog",
    "PhiAccessLogIncompleteError",
    "ReviewAction",
    "ReviewActionInput",
    "ReviewActionsConfig",
    "ReviewActionsStore",
    "SafeId",
    "UTCDatetime",
    "apply_migrations",
    "current_revision",
    "head_revision",
]
