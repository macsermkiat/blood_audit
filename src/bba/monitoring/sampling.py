"""Weekly clinical-reviewer sample (PRD §18 / issue #27).

Draws 50–75 random audit rows per ISO week for human reviewer inspection.
Determinism is the load-bearing property: the same ``(week_iso, sample_size,
seed)`` tuple ALWAYS produces the same ``audit_ids`` so a historical sample
can be re-derived from the manifest alone, without RNG-state replay.

Operational, not clinical: this module reads audit rows from
:mod:`bba.audit_store` but never inspects classification logic. The sample
is a uniform-random draw from the row population, NOT a stratified or
prevalence-weighted sample — those are eval-harness concerns (#20).
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.audit_store import AuditRow
from bba.monitoring.models import (
    WEEKLY_REVIEWER_SAMPLE_MAX,
    WEEKLY_REVIEWER_SAMPLE_MIN,
    WeeklyReviewerSample,
)


def draw_weekly_reviewer_sample(
    audit_rows: Sequence[AuditRow],
    *,
    week_iso: str,
    sample_size: int,
    seed: int,
) -> WeeklyReviewerSample:
    """Draw a deterministic weekly reviewer sample.

    ``sample_size`` MUST be in ``[WEEKLY_REVIEWER_SAMPLE_MIN,
    WEEKLY_REVIEWER_SAMPLE_MAX]`` (50–75). The RNG is seeded by
    ``hash((week_iso, sample_size, seed))`` so the same inputs produce the
    same draw across processes, machines, and Python interpreter
    invocations (relies on a hash seed that is NOT the process-level
    ``PYTHONHASHSEED`` — implementation uses an explicit stable hash).

    Raises :class:`ValueError` if ``sample_size`` is out of range or
    exceeds the population.
    """
    raise NotImplementedError


__all__ = ("draw_weekly_reviewer_sample",)
