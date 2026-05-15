"""Alembic environment script.

Invoked by ``alembic upgrade head`` (CLI) and by
``bba.review_actions.apply_migrations`` (Python wrapper). The DSN is read
from the ``sqlalchemy.url`` config option, which the Python wrapper sets
programmatically before calling ``command.upgrade``; the CLI path reads it
from ``alembic.ini``.

No ``target_metadata`` — migrations are raw-SQL via ``op.execute``. The
schema is the source of truth in the version files, not in any SQLAlchemy
ORM model: the audit-row contract and the trigger guards are too
Postgres-specific to round-trip through a generic ORM.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool


config = context.config


target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emits SQL, no DB connection).

    Useful for producing a static SQL script for DBA review before applying
    in a regulated environment.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (against a live DB)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
