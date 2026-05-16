"""Top-level cohort assignment entry point (issue #7).

The single public function :func:`assign_cohort` is the deterministic
composition over the predicates in :mod:`bba.cohort_detector.rules`.
Cohort precedence (top wins):

1. ``UNKNOWN`` — ``CohortInputs.procedure_events is None``. Procedure
   data unavailable forbids assigning the cardiac / ortho / default
   cohorts; downstream classifier routes to NEEDS_REVIEW (PRD §5 +
   user constraint: "NEVER silently apply 7.0 when procedure data is
   missing").
2. ``MTP`` — auto-bypass to APPROPRIATE; checked even when other signals
   are present because the RBC-cluster pattern is the most actionable
   safety signal.
3. ``ORTHO_CARDIAC`` — ortho operative event AND cardiac history; takes
   precedence over plain cardiac because it carries the higher 8.0
   threshold.
4. ``CARDIAC_SURGERY`` — recent cardiac operative event; threshold 7.5.
5. ``ESRD_EPO`` — ESRD diagnosis AND dialysis med; threshold 8.0
   (Round 2 fix N1: both signals required).
6. ``HEME_MALIGNANCY_ACTIVE`` — heme malignancy diagnosis AND chemo med
   AND ANC < 500; not threshold-driven (T2-supportive).
7. ``DEFAULT`` — fall-through; threshold 7.0.
"""

from __future__ import annotations

from bba.cohort_detector.models import CohortAssignment, CohortInputs


def assign_cohort(inputs: CohortInputs) -> CohortAssignment:
    """Return the deterministic cohort assignment for ``inputs``.

    Pure function — same input always yields the same
    :class:`CohortAssignment`. The detector does not mutate ``inputs``,
    does not consult any module-global state, and never raises on a
    well-formed :class:`CohortInputs`. Constructing :class:`CohortInputs`
    with invalid types fails at the Pydantic boundary, not here.
    """
    raise NotImplementedError("RED-phase scaffold — implementation pending")


__all__ = ("assign_cohort",)
