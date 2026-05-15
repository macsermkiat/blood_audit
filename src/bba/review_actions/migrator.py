"""Schema migration entry-point for ``bba.review_actions``.

PRD §16 AC: "Schema migration scripts in ``migrations/`` (alembic or
equivalent)". This module is a thin Python wrapper around alembic; the
migration files themselves live in ``<repo_root>/migrations/versions/``.

The wrapper exists so callers (dashboard bootstrap, test fixtures, CI smoke
tests) can apply migrations programmatically against an arbitrary DSN
without shelling out to the alembic CLI.
"""

from __future__ import annotations

from pathlib import Path

from bba.review_actions.models import ReviewActionsConfig


MIGRATIONS_ROOT_DEFAULT = Path("migrations")
"""Default location of the alembic project root, relative to the repo root.

Test fixtures and the dashboard bootstrap call :func:`apply_migrations` with
the absolute path resolved from ``pyproject.toml``-co-located markers; this
default works for invocations from the repo root.
"""


def apply_migrations(
    config: ReviewActionsConfig,
    *,
    migrations_root: Path = MIGRATIONS_ROOT_DEFAULT,
) -> None:
    """Apply all pending alembic migrations against ``config.dsn``.

    Idempotent: calling on an up-to-date DB is a no-op. The function blocks
    until the schema reaches alembic head. A failure mid-migration leaves
    the DB at the last committed version (alembic transactional-DDL default
    on Postgres).

    Raises:
        FileNotFoundError: ``migrations_root`` does not exist.
        Exception: any error surfaced from alembic / the DB.
    """
    raise NotImplementedError


def current_revision(
    config: ReviewActionsConfig,
    *,
    migrations_root: Path = MIGRATIONS_ROOT_DEFAULT,
) -> str | None:
    """Return the current alembic revision id, or ``None`` if no migrations
    have been applied.

    Used by health-check endpoints in the dashboard and by the store's
    construction-time schema check.
    """
    raise NotImplementedError


def head_revision(
    *,
    migrations_root: Path = MIGRATIONS_ROOT_DEFAULT,
) -> str:
    """Return the alembic ``head`` revision id from the on-disk script
    directory. Pure file-system read — does not touch the DB."""
    raise NotImplementedError


__all__ = (
    "MIGRATIONS_ROOT_DEFAULT",
    "apply_migrations",
    "current_revision",
    "head_revision",
)
