"""``batch_runs`` state-machine transitions (issue #24, user constraint #4).

The transition table is the single source of truth for what move is
legal at what state. The pipeline never mutates a :class:`BatchRun` in
place; :func:`transition` returns a NEW :class:`BatchRun` so concurrent
readers (resume reconciler, dashboard polling) can never observe a
half-mutated state.

Terminal states (``COMPLETE``, ``FAILED``) accept no outbound
transitions. Resume-on-startup explicitly skips them.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from types import MappingProxyType

from bba.audit_pipeline.exceptions import BatchStateTransitionError  # noqa: F401  used by GREEN
from bba.audit_pipeline.models import BatchRun, BatchRunState


VALID_TRANSITIONS: Mapping[BatchRunState, frozenset[BatchRunState]] = MappingProxyType(
    {
        BatchRunState.PENDING: frozenset(
            {BatchRunState.SUBMITTED, BatchRunState.FAILED}
        ),
        BatchRunState.SUBMITTED: frozenset(
            {BatchRunState.PARTIAL, BatchRunState.COMPLETE, BatchRunState.FAILED}
        ),
        BatchRunState.PARTIAL: frozenset(
            {BatchRunState.COMPLETE, BatchRunState.FAILED}
        ),
        BatchRunState.COMPLETE: frozenset(),
        BatchRunState.FAILED: frozenset(),
    }
)
"""Map from current state to the set of legal next states.

Empty set = terminal. PENDING never jumps straight to COMPLETE
(must go through SUBMITTED first); the resume reconciler depends on
that ordering to know whether to poll Anthropic.
"""


def transition(
    run: BatchRun,
    *,
    to_state: BatchRunState,
    anthropic_batch_id: str | None = None,
    error_message: str | None = None,
    now: datetime | None = None,
) -> BatchRun:
    """Return a new :class:`BatchRun` advanced to ``to_state``.

    Raises :class:`BatchStateTransitionError` if ``run.state -> to_state``
    is not in :data:`VALID_TRANSITIONS`. The original ``run`` is never
    mutated (pydantic-frozen).

    ``anthropic_batch_id`` must be supplied on the ``PENDING -> SUBMITTED``
    transition; the resulting :class:`BatchRun` carries it through the
    PARTIAL / COMPLETE states unchanged. ``error_message`` must be
    supplied on any transition to ``FAILED``.

    The implementation lives in GREEN (issue #24).
    """
    _ = (run, to_state, anthropic_batch_id, error_message, now)
    raise NotImplementedError("RED-phase scaffold; see issue #24")


def is_terminal(state: BatchRunState) -> bool:
    """Return True iff ``state`` accepts no outbound transitions.

    Pure-data check (reads :data:`VALID_TRANSITIONS`); no I/O. Resume
    uses this to skip rows that have already settled.
    """
    return len(VALID_TRANSITIONS[state]) == 0


def _now_utc() -> datetime:
    """Default ``updated_at`` source — replaced in tests via ``now=`` kwarg."""
    return datetime.now(UTC)


__all__ = [
    "VALID_TRANSITIONS",
    "is_terminal",
    "transition",
]
