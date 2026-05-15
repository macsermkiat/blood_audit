"""Schema migration entry-point for ``bba.review_actions``.

PRD §16 AC: "Schema migration scripts in ``migrations/`` (alembic or
equivalent)". This module wraps alembic's Python API so callers (dashboard
bootstrap, test fixtures, CI smoke tests) can apply migrations against an
arbitrary DSN without shelling out to the alembic CLI.

The migration files themselves live in ``<repo_root>/migrations/versions/``.
The on-disk config is ``alembic.ini`` at the repo root; this module
constructs an in-memory ``Config`` so the DSN can be injected per-call
without mutating the on-disk ini file.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine

from bba.review_actions.models import ReviewActionsConfig


MIGRATIONS_ROOT_DEFAULT = Path("migrations")
"""Default location of the alembic project root, relative to the repo root."""


def _build_config(
    config: ReviewActionsConfig,
    migrations_root: Path,
) -> Config:
    """Construct an alembic :class:`Config` with the DSN injected.

    ``script_location`` is set to ``migrations_root`` so the wrapper works
    whether the caller passes a relative ``migrations`` (cwd-relative) or
    an absolute path resolved from the repo root.
    """
    if not migrations_root.exists():
        raise FileNotFoundError(
            f"migrations_root does not exist: {migrations_root}"
        )

    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", str(migrations_root))
    alembic_cfg.set_main_option("sqlalchemy.url", config.sqlalchemy_dsn)
    return alembic_cfg


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
    """
    alembic_cfg = _build_config(config, migrations_root)
    command.upgrade(alembic_cfg, "head")


def current_revision(
    config: ReviewActionsConfig,
    *,
    migrations_root: Path = MIGRATIONS_ROOT_DEFAULT,
) -> str | None:
    """Return the current alembic revision id, or ``None`` if no migrations
    have been applied (no ``alembic_version`` table present).

    Used by health-check endpoints in the dashboard and by the store's
    construction-time schema check.
    """
    if not migrations_root.exists():
        raise FileNotFoundError(
            f"migrations_root does not exist: {migrations_root}"
        )

    engine = create_engine(config.sqlalchemy_dsn)
    try:
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            return context.get_current_revision()
    finally:
        engine.dispose()


def head_revision(
    *,
    migrations_root: Path = MIGRATIONS_ROOT_DEFAULT,
) -> str:
    """Return the alembic ``head`` revision id from the on-disk script
    directory. Pure file-system read — does not touch the DB.

    Raises:
        FileNotFoundError: ``migrations_root`` does not exist.
        RuntimeError: no revisions found in ``migrations_root/versions/``.
    """
    if not migrations_root.exists():
        raise FileNotFoundError(
            f"migrations_root does not exist: {migrations_root}"
        )

    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", str(migrations_root))
    script = ScriptDirectory.from_config(alembic_cfg)
    head = script.get_current_head()
    if head is None:
        raise RuntimeError(
            f"no alembic revisions found under {migrations_root}/versions/"
        )
    return head


__all__ = (
    "MIGRATIONS_ROOT_DEFAULT",
    "apply_migrations",
    "current_revision",
    "head_revision",
)
