"""Top-level deterministic classification entry point (issue #8).

The single public function :func:`classify` is the deterministic composition
over the Hb-tier rules + four bypass pathways. Precedence (top wins) is
specified in PRD §"Implementation Decisions §6":

1. Hb missing                       → ``INSUFFICIENT_EVIDENCE`` (no bypass)
2. Cohort ``MTP``                   → ``APPROPRIATE`` (``bypass_reason=mtp``)
3. Cohort ``UNKNOWN``               → ``NEEDS_REVIEW`` (no bypass; user constraint #9)
4. Bypass: peri-procedural ≤ 6 h    → ``APPROPRIATE`` (``bypass_reason=peri_procedural_6h``)
5. Bypass: delta-Hb trigger fired   → ``APPROPRIATE`` (``bypass_reason=delta_hb``)
6. Hemodilution: Hb < threshold AND ≥ 2 L crystalloid in 4 h
                                    → ``NEEDS_REVIEW`` (``bypass_reason=hemodilution_flagged``)
7. Plain Hb-tier rule:
   - Hb < ``cohort_threshold``       → ``APPROPRIATE``
   - ``cohort_threshold`` ≤ Hb < 10  → ``NEEDS_REVIEW``
   - Hb ≥ 10                         → ``POTENTIALLY_INAPPROPRIATE``

Precedence notes:

* Hb missing precedes every other check — no signal can be interpreted
  without an Hb anchor, including a triggered delta-Hb flag (stale upstream
  state cannot override the structural "no current Hb" fact).
* MTP precedes peri-procedural / delta-Hb so the bypass_reason reflects the
  most clinically load-bearing fact when multiple signals fire.
* UNKNOWN precedes the bypass block because the system explicitly refuses
  to interpret signals without a confirmed cohort (Round 2 fix N1 +
  user constraint #9).
* Hemodilution is checked only inside the ``Hb < threshold`` branch — it is
  scoped to the auto-APPROPRIATE case it would otherwise produce, and never
  promotes a gray-zone or high-Hb result.
"""

from __future__ import annotations

from bba.cohort_detector import CohortLabel
from bba.deterministic_classifier.models import (
    BypassReason,
    ClassifierInputs,
    ClassifierResult,
)

HB_GT_10_THRESHOLD: float = 10.0
"""PRD §6 Hb (g/dL) ceiling above which the deterministic pre-classification
is ``POTENTIALLY_INAPPROPRIATE`` (LLM may override on positive evidence)."""

PERI_PROCEDURAL_WINDOW_HOURS: float = 6.0
"""PRD §6 peri-procedural bypass window (hours). A procedure within this
many hours BEFORE the order anchor auto-bypasses to ``APPROPRIATE``."""

HEMODILUTION_CRYSTALLOID_LITERS: float = 2.0
"""PRD §6 + Round 1 B5: ≥ 2 L of crystalloid in the prior 4 h marks the
Hb as hemodilution-suspect. Triggers ``NEEDS_REVIEW`` rather than
auto-APPROPRIATE even when Hb < cohort_threshold."""


def classify(inputs: ClassifierInputs) -> ClassifierResult:
    """Return the deterministic classification for ``inputs``.

    Pure function — same input always yields the same
    :class:`ClassifierResult`. The classifier does not mutate ``inputs``,
    does not consult module-global state, and never raises on a
    well-formed :class:`ClassifierInputs`. Constructing
    :class:`ClassifierInputs` with invalid types fails at the Pydantic
    boundary, not here.

    See module docstring for precedence ordering.
    """
    hb = inputs.hb_result
    cohort = inputs.cohort_assignment
    threshold = cohort.threshold

    # 1. Hb missing — INSUFFICIENT_EVIDENCE precedes every other check.
    #    A stale upstream delta-Hb flag cannot override the structural
    #    fact that there is no current Hb to anchor any interpretation.
    if hb.value_g_dl is None:
        return ClassifierResult(
            classification="INSUFFICIENT_EVIDENCE",
            bypass_reason=BypassReason.NONE,
            cohort_threshold=threshold,
            rationale="hb_missing",
        )

    # 2. MTP cohort auto-bypass — cluster pattern trumps every Hb tier.
    if cohort.label == CohortLabel.MTP:
        return ClassifierResult(
            classification="APPROPRIATE",
            bypass_reason=BypassReason.MTP,
            cohort_threshold=threshold,
            rationale="bypass_mtp",
        )

    # 3. Cohort UNKNOWN — refuse to default to 7.0 when procedure data
    #    was missing (Round 2 fix N1 + user constraint #9).
    if cohort.label == CohortLabel.UNKNOWN:
        return ClassifierResult(
            classification="NEEDS_REVIEW",
            bypass_reason=BypassReason.NONE,
            cohort_threshold=threshold,
            rationale="cohort_unknown",
        )

    # 4. Peri-procedural bypass — procedure within the 6 h window before
    #    the order anchor. ``<=`` per PRD §6 ("within 6 h" inclusive).
    proximity = inputs.procedure_proximity_hours
    if proximity is not None and proximity <= PERI_PROCEDURAL_WINDOW_HOURS:
        return ClassifierResult(
            classification="APPROPRIATE",
            bypass_reason=BypassReason.PERI_PROCEDURAL_6H,
            cohort_threshold=threshold,
            rationale="bypass_peri_procedural",
        )

    # 5. Delta-Hb bypass — at least one window in the HbLookupResult
    #    fired its tiered threshold (PRD §3 + Round 2 E3).
    if hb.delta_hb_bypass:
        return ClassifierResult(
            classification="APPROPRIATE",
            bypass_reason=BypassReason.DELTA_HB,
            cohort_threshold=threshold,
            rationale="bypass_delta_hb",
        )

    # Non-threshold-driven cohorts that survived steps 1-5 (i.e.,
    # HEME_MALIGNANCY_ACTIVE — MTP and UNKNOWN exited above). The
    # T2-supportive heme cohort defers to the LLM stage for context
    # interpretation; surface as NEEDS_REVIEW with no bypass.
    if threshold is None:
        return ClassifierResult(
            classification="NEEDS_REVIEW",
            bypass_reason=BypassReason.NONE,
            cohort_threshold=None,
            rationale="cohort_non_threshold",
        )

    # 6 + 7. Plain Hb-tier rule with hemodilution carve-out scoped to the
    #        sub-threshold branch.
    hb_value = hb.value_g_dl
    if hb_value < threshold:
        if inputs.crystalloid_liters_prior_4h >= HEMODILUTION_CRYSTALLOID_LITERS:
            return ClassifierResult(
                classification="NEEDS_REVIEW",
                bypass_reason=BypassReason.HEMODILUTION_FLAGGED,
                cohort_threshold=threshold,
                rationale="bypass_hemodilution",
            )
        return ClassifierResult(
            classification="APPROPRIATE",
            bypass_reason=BypassReason.NONE,
            cohort_threshold=threshold,
            rationale="hb_lt_threshold",
        )

    if hb_value < HB_GT_10_THRESHOLD:
        return ClassifierResult(
            classification="NEEDS_REVIEW",
            bypass_reason=BypassReason.NONE,
            cohort_threshold=threshold,
            rationale="hb_7_to_10",
        )

    return ClassifierResult(
        classification="POTENTIALLY_INAPPROPRIATE",
        bypass_reason=BypassReason.NONE,
        cohort_threshold=threshold,
        rationale="hb_ge_10",
    )


__all__ = (
    "HB_GT_10_THRESHOLD",
    "HEMODILUTION_CRYSTALLOID_LITERS",
    "PERI_PROCEDURAL_WINDOW_HOURS",
    "classify",
)
