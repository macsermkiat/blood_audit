"""``batch_runs`` persistence Protocol + in-memory implementation.

The pipeline's row-level checkpointing requires a durable home for the
:class:`bba.audit_pipeline.BatchRun` rows. Production uses Postgres (the
same choice :mod:`bba.review_actions` made — user constraint #4); tests
use the in-memory implementation defined here so state-machine logic
stays trivially unit-testable.

The Protocol is intentionally narrow (one method per state-machine
operation) so the Postgres implementation can be swapped in without
touching the pipeline orchestrator code.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from bba.audit_pipeline.models import BatchRun, BatchRunState


@runtime_checkable
class BatchRunStore(Protocol):
    """Persistence boundary for :class:`BatchRun` rows.

    Concrete implementations:

    * :class:`InMemoryBatchRunStore` — test-only, no durability.
    * ``PostgresBatchRunStore`` (GREEN) — production. Will use the
      same psycopg-pool pattern :mod:`bba.review_actions` established.

    Every mutating method takes a fully-formed :class:`BatchRun`. The
    state machine is enforced one level up (in
    :func:`bba.audit_pipeline.state_machine.transition`); the store only
    persists the result.
    """

    def create(self, run: BatchRun) -> None:
        """Insert a new row. Raises if ``batch_id`` already exists."""
        ...

    def get(self, batch_id: str) -> BatchRun:
        """Return the current row by ``batch_id``. Raises ``KeyError`` if missing."""
        ...

    def update(self, run: BatchRun) -> None:
        """Replace the existing row identified by ``batch_id``.

        The state transition is validated upstream; this method only
        persists. Raises ``KeyError`` if ``batch_id`` does not exist.
        """
        ...

    def list_by_state(self, state: BatchRunState) -> tuple[BatchRun, ...]:
        """Return every row whose current state matches ``state``.

        Resume-on-startup uses this to scan for SUBMITTED + PARTIAL
        rows that need polling.
        """
        ...

    def list_all(self) -> tuple[BatchRun, ...]:
        """Return every row regardless of state. Used by reconcilers."""
        ...


class InMemoryBatchRunStore:
    """Test-only :class:`BatchRunStore` backed by a dict.

    Single-process, single-threaded; not safe for concurrent writers.
    Use :class:`PostgresBatchRunStore` (GREEN) in production.
    """

    def __init__(self) -> None:
        self._rows: dict[str, BatchRun] = {}

    def create(self, run: BatchRun) -> None:
        if run.batch_id in self._rows:
            raise ValueError(
                f"batch_id {run.batch_id!r} already exists; "
                "use update() to advance an existing row"
            )
        self._rows[run.batch_id] = run

    def get(self, batch_id: str) -> BatchRun:
        if batch_id not in self._rows:
            raise KeyError(batch_id)
        return self._rows[batch_id]

    def update(self, run: BatchRun) -> None:
        if run.batch_id not in self._rows:
            raise KeyError(run.batch_id)
        self._rows[run.batch_id] = run

    def list_by_state(self, state: BatchRunState) -> tuple[BatchRun, ...]:
        return tuple(row for row in self._rows.values() if row.state is state)

    def list_all(self) -> tuple[BatchRun, ...]:
        return tuple(self._rows.values())


# Static check: InMemoryBatchRunStore satisfies the BatchRunStore protocol.
# Catches signature drift at type-check time (mirrors llm_client.cassette).
_PROTOCOL_CHECK: type[BatchRunStore] = InMemoryBatchRunStore


class PostgresBatchRunStore:
    """Postgres-backed :class:`BatchRunStore` for production deployments.

    Survives process restarts (the durability guarantee the in-memory
    store cannot provide). Uses ``psycopg`` v3 with a thread-safe
    connection pool so multiple pipeline workers can write
    concurrently — the state-machine guard upstream of this layer
    rejects illegal transitions, so the DB only sees coherent updates.

    The schema is installed by alembic migration
    ``a1c2e3f4b5d6_audit_pipeline_batch_runs.py``; the table-level
    CHECK constraints mirror :class:`BatchRun`'s pydantic invariants
    for defense in depth (a misconfigured caller bypassing the model
    layer still cannot persist a contradictory row).
    """

    def __init__(self, dsn: str) -> None:
        # Lazy-open the pool so tests can construct a store against a
        # not-yet-migrated DB without the pool failing at __init__.
        from psycopg_pool import ConnectionPool

        self._dsn = dsn
        self._pool = ConnectionPool(dsn, open=False)

    def open(self) -> None:
        """Open the connection pool. Idempotent."""
        self._pool.open()

    def close(self) -> None:
        """Close the connection pool. Idempotent."""
        self._pool.close()

    def create(self, run: BatchRun) -> None:
        from psycopg.errors import UniqueViolation

        try:
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO batch_runs (
                        batch_id, state, run_id, code_version, audit_ids,
                        anthropic_batch_id, submitted_at, updated_at,
                        error_message
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run.batch_id,
                        run.state.value,
                        run.run_id,
                        run.code_version,
                        list(run.audit_ids),
                        run.anthropic_batch_id,
                        run.submitted_at,
                        run.updated_at,
                        run.error_message,
                    ),
                )
        except UniqueViolation as exc:
            raise ValueError(
                f"batch_id {run.batch_id!r} already exists; "
                "use update() to advance an existing row"
            ) from exc

    def get(self, batch_id: str) -> BatchRun:
        from psycopg.rows import dict_row

        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT batch_id, state, run_id, code_version, audit_ids, "
                    "anthropic_batch_id, submitted_at, updated_at, error_message "
                    "FROM batch_runs WHERE batch_id = %s",
                    (batch_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise KeyError(batch_id)
        return self._row_to_batch_run(row)

    def update(self, run: BatchRun) -> None:
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE batch_runs SET
                    state = %s,
                    anthropic_batch_id = %s,
                    submitted_at = %s,
                    updated_at = %s,
                    error_message = %s
                WHERE batch_id = %s
                """,
                (
                    run.state.value,
                    run.anthropic_batch_id,
                    run.submitted_at,
                    run.updated_at,
                    run.error_message,
                    run.batch_id,
                ),
            )
            if cur.rowcount == 0:
                raise KeyError(run.batch_id)

    def list_by_state(self, state: BatchRunState) -> tuple[BatchRun, ...]:
        from psycopg.rows import dict_row

        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT batch_id, state, run_id, code_version, audit_ids, "
                    "anthropic_batch_id, submitted_at, updated_at, error_message "
                    "FROM batch_runs WHERE state = %s "
                    "ORDER BY updated_at",
                    (state.value,),
                )
                rows = cur.fetchall()
        return tuple(self._row_to_batch_run(r) for r in rows)

    def list_all(self) -> tuple[BatchRun, ...]:
        from psycopg.rows import dict_row

        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SELECT batch_id, state, run_id, code_version, audit_ids, "
                    "anthropic_batch_id, submitted_at, updated_at, error_message "
                    "FROM batch_runs ORDER BY updated_at"
                )
                rows = cur.fetchall()
        return tuple(self._row_to_batch_run(r) for r in rows)

    @staticmethod
    def _row_to_batch_run(row: dict[str, object]) -> BatchRun:
        """Translate a Postgres row dict into a :class:`BatchRun`.

        The pydantic model layer re-validates every field (e.g.
        rejects PENDING+anthropic_batch_id, requires error_message
        on FAILED) so a contract drift between the DB and the model
        surfaces at read time."""
        audit_ids_raw = row["audit_ids"]
        if not isinstance(audit_ids_raw, list):
            raise TypeError(
                f"batch_runs.audit_ids must be a list[str]; got {type(audit_ids_raw).__name__}"
            )
        return BatchRun(
            batch_id=str(row["batch_id"]),
            state=BatchRunState(str(row["state"])),
            run_id=str(row["run_id"]),
            code_version=str(row["code_version"]),
            audit_ids=tuple(str(a) for a in audit_ids_raw),
            anthropic_batch_id=str(row["anthropic_batch_id"])
            if row["anthropic_batch_id"] is not None
            else None,
            submitted_at=row["submitted_at"],  # type: ignore[arg-type]
            updated_at=row["updated_at"],  # type: ignore[arg-type]
            error_message=str(row["error_message"])
            if row["error_message"] is not None
            else None,
        )


# Static check: PostgresBatchRunStore also satisfies the protocol.
_POSTGRES_PROTOCOL_CHECK: type[BatchRunStore] = PostgresBatchRunStore


__all__: Sequence[str] = (
    "BatchRunStore",
    "InMemoryBatchRunStore",
    "PostgresBatchRunStore",
)
