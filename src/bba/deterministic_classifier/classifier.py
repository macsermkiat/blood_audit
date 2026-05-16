"""Top-level deterministic classification entry point (issue #8).

The single public function :func:`classify` is the deterministic composition
over the Hb-tier rules + four bypass pathways. Precedence (top wins) is
specified in PRD §"Implementation Decisions §6":

1. Hb missing                       → ``INSUFFICIENT_EVIDENCE`` (no bypass)
2. Cohort ``UNKNOWN``               → ``NEEDS_REVIEW`` (no bypass; user constraint #9)
3. Bypass: ``MTP`` cohort           → ``APPROPRIATE`` (``bypass_reason=mtp``)
4. Bypass: peri-procedural ≤ 6 h    → ``APPROPRIATE`` (``bypass_reason=peri_procedural_6h``)
5. Bypass: delta-Hb trigger fired   → ``APPROPRIATE`` (``bypass_reason=delta_hb``)
6. Hemodilution: Hb < threshold AND ≥ 2 L crystalloid in 4 h
                                    → ``NEEDS_REVIEW`` (``bypass_reason=hemodilution_flagged``)
7. Plain Hb-tier rule:
   - Hb < ``cohort_threshold``       → ``APPROPRIATE``
   - ``cohort_threshold`` ≤ Hb < 10  → ``NEEDS_REVIEW`` (the LLM-eligible
                                       gray-zone case; persists as ``NEEDS_REVIEW``
                                       canonically — the LLM pipeline reads
                                       the result and either confirms or
                                       reclassifies)
   - Hb ≥ 10                         → ``POTENTIALLY_INAPPROPRIATE`` (LLM may override)

Note on precedence ordering of the bypass block (rules 3–5): MTP is checked
first because it is a cohort-level safety signal that should not be filtered
out by a parallel peri-procedural or delta-Hb signal; the result is the
same (APPROPRIATE) but the bypass_reason must reflect the most clinically
load-bearing fact. Peri-procedural precedes delta-Hb to match the order
in the PRD scope list.
"""

from __future__ import annotations

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
    raise NotImplementedError


__all__ = (
    "HB_GT_10_THRESHOLD",
    "HEMODILUTION_CRYSTALLOID_LITERS",
    "PERI_PROCEDURAL_WINDOW_HOURS",
    "classify",
)
