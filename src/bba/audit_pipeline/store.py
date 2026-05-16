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


__all__: Sequence[str] = ("BatchRunStore", "InMemoryBatchRunStore")
