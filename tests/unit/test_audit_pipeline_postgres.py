"""Postgres-backed :class:`PostgresBatchRunStore` integration tests.

These tests stand up a real Postgres container via testcontainers-python,
run the alembic migration that ships with the audit_pipeline (creating
the ``batch_runs`` table), and exercise the store's CRUD + the durability
guarantee that the in-memory store cannot provide (Codex review HIGH #3:
"add PostgresBatchRunStore with tests that survive a fresh process/store
instance").

Skipped automatically when Docker is not available — the testcontainers
fixture raises at startup and the suite reports the skip rather than a
spurious failure.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bba.audit_pipeline import (
    BatchRun,
    BatchRunState,
    PostgresBatchRunStore,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def _postgres_container() -> Iterator[object]:
    """Session-scoped Postgres container via testcontainers-python.

    Imported lazily so a missing Docker daemon errors out on first use
    rather than at test-collection time."""
    pytest.importorskip("testcontainers.postgres")
    pytest.importorskip("docker")
    try:
        from testcontainers.postgres import PostgresContainer

        with PostgresContainer("postgres:16-alpine") as pg:
            yield pg
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"Postgres testcontainer unavailable: {exc}")


@pytest.fixture
def fresh_db(_postgres_container: object) -> Iterator[str]:
    """Drop and recreate the public schema before each test, then run
    the audit_pipeline alembic migration so ``batch_runs`` exists."""
    import psycopg
    from alembic import command
    from alembic.config import Config

    dsn = _postgres_container.get_connection_url().replace(  # type: ignore[attr-defined]
        "postgresql+psycopg2://", "postgresql://"
    )

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
            cur.execute("CREATE SCHEMA public")
            cur.execute("GRANT ALL ON SCHEMA public TO PUBLIC")
            for role in ("audit_pipeline_app", "review_actions_app"):
                cur.execute(
                    f"DO $$ BEGIN IF EXISTS ("
                    f"SELECT 1 FROM pg_roles WHERE rolname = '{role}'"
                    f") THEN "
                    f" EXECUTE 'REASSIGN OWNED BY {role} TO CURRENT_USER'; "
                    f" EXECUTE 'DROP OWNED BY {role}'; "
                    f" EXECUTE 'DROP ROLE {role}'; "
                    f"END IF; END $$;"
                )

    alembic_cfg = Config(str(REPO_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    # The project ships psycopg (psycopg3) only — no psycopg2. SQLAlchemy
    # routes a bare ``postgresql://`` URL to the psycopg2 driver by default,
    # so we must opt into psycopg3 explicitly via the +psycopg suffix. This
    # mirrors bba.review_actions.models.ReviewActionsConfig.sqlalchemy_dsn,
    # which is the canonical converter.
    alembic_cfg.set_main_option(
        "sqlalchemy.url",
        dsn.replace("postgresql://", "postgresql+psycopg://", 1),
    )
    command.upgrade(alembic_cfg, "head")
    yield dsn


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, 0, 0, tzinfo=UTC)


def _pending_run() -> BatchRun:
    return BatchRun(
        batch_id="batch-pg-001",
        state=BatchRunState.PENDING,
        run_id="run-pg-001",
        code_version="v0.1.0+pg",
        audit_ids=("audit-001", "audit-002"),
        updated_at=_utc(2026, 5, 16),
    )


class TestPostgresBatchRunStoreCRUD:
    def test_create_then_get_round_trips(self, fresh_db: str) -> None:
        store = PostgresBatchRunStore(fresh_db)
        store.open()
        try:
            run = _pending_run()
            store.create(run)
            fetched = store.get("batch-pg-001")
            assert fetched == run
        finally:
            store.close()

    def test_duplicate_create_raises(self, fresh_db: str) -> None:
        store = PostgresBatchRunStore(fresh_db)
        store.open()
        try:
            store.create(_pending_run())
            with pytest.raises(ValueError):
                store.create(_pending_run())
        finally:
            store.close()

    def test_update_replaces_existing_row(self, fresh_db: str) -> None:
        store = PostgresBatchRunStore(fresh_db)
        store.open()
        try:
            store.create(_pending_run())
            updated = _pending_run().model_copy(
                update={
                    "state": BatchRunState.SUBMITTED,
                    "anthropic_batch_id": "msgbatch_pg_001",
                    "submitted_at": _utc(2026, 5, 16),
                    "updated_at": _utc(2026, 5, 16),
                }
            )
            store.update(updated)
            assert store.get("batch-pg-001").state is BatchRunState.SUBMITTED
        finally:
            store.close()

    def test_list_by_state_filters(self, fresh_db: str) -> None:
        store = PostgresBatchRunStore(fresh_db)
        store.open()
        try:
            store.create(_pending_run())
            submitted = BatchRun(
                batch_id="batch-pg-002",
                state=BatchRunState.SUBMITTED,
                run_id="run-pg-002",
                code_version="v0.1.0+pg",
                audit_ids=("audit-003",),
                anthropic_batch_id="msgbatch_pg_002",
                submitted_at=_utc(2026, 5, 16),
                updated_at=_utc(2026, 5, 16),
            )
            store.create(submitted)
            pending_rows = store.list_by_state(BatchRunState.PENDING)
            assert {r.batch_id for r in pending_rows} == {"batch-pg-001"}
            submitted_rows = store.list_by_state(BatchRunState.SUBMITTED)
            assert {r.batch_id for r in submitted_rows} == {"batch-pg-002"}
        finally:
            store.close()


class TestPostgresBatchRunStoreDurability:
    """The durability guarantee the in-memory store cannot provide.

    Codex review HIGH #3: "add PostgresBatchRunStore with tests that
    survive a fresh process/store instance." A row written by one
    store instance must be readable from a freshly-constructed store
    pointing at the same DSN.
    """

    def test_row_survives_fresh_store_instance(self, fresh_db: str) -> None:
        writer = PostgresBatchRunStore(fresh_db)
        writer.open()
        try:
            writer.create(_pending_run())
        finally:
            writer.close()

        # Fresh store, fresh pool — emulates a process restart.
        reader = PostgresBatchRunStore(fresh_db)
        reader.open()
        try:
            survived = reader.get("batch-pg-001")
        finally:
            reader.close()
        assert survived.batch_id == "batch-pg-001"
        assert survived.state is BatchRunState.PENDING

    def test_db_check_constraint_blocks_pending_with_anthropic_id(
        self, fresh_db: str
    ) -> None:
        """Defence-in-depth: the model layer rejects a PENDING +
        anthropic_batch_id row, AND the DB does the same via CHECK.

        This protects against a misconfigured caller that bypasses the
        pydantic boundary (e.g., raw SQL inserts) — the migration's
        constraint catches it."""
        import psycopg
        from psycopg.errors import CheckViolation

        with psycopg.connect(fresh_db, autocommit=True) as conn:
            with conn.cursor() as cur, pytest.raises(CheckViolation):
                cur.execute(
                    """
                    INSERT INTO batch_runs (
                        batch_id, state, run_id, code_version, audit_ids,
                        anthropic_batch_id, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        "batch-bad",
                        "pending",
                        "run-bad",
                        "v0.1.0+pg",
                        ["audit-001"],
                        "msgbatch_should_be_null",
                        _utc(2026, 5, 16),
                    ),
                )
