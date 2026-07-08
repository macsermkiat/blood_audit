"""Deterministic platelet gate (docs plan §5.1, revised after clinical review).

The single public function :func:`classify_platelet` is the v1 platelet
appropriateness gate. It is intentionally minimal: unlike the RBC
:func:`bba.deterministic_classifier.classify`, it auto-clears NOTHING.

Precedence (top wins):

1. Count missing → mirror the RBC missing-Hb contract:
   * defer flag ON  → ``NEEDS_REVIEW`` (``plt_missing_defer_llm``)
   * defer flag OFF → ``INSUFFICIENT_EVIDENCE`` (``plt_missing``)
2. Count ≥ :data:`PLATELET_REVIEW_CEILING` → ``POTENTIALLY_INAPPROPRIATE``
   (``plt_ge_100``) — routes to LLM/review, NOT terminal. Agrees with the
   ">100k head-bleed: don't give" rule; safe because it is reviewed.
3. Count < :data:`PLATELET_REVIEW_CEILING` (including < 10) → ``NEEDS_REVIEW``
   (``plt_defer_llm``). The LLM reads notes for the indication (prophylaxis /
   LP / CVC / major surgery / active bleeding / dept) AND the exclusion
   populations (dengue-no-bleed, TTP, HIT, ITP, aplastic-no-bleed,
   snakebite-no-bleed), then applies the policy threshold.

Why nothing is auto-cleared (§8/CR-C1): ``APPROPRIATE`` is a terminal
deterministic-final class — no LLM, no human. But the policies WITHHOLD
platelets at very low counts for the exclusion populations above (for TTP/HIT,
transfusion is actively harmful). In a dengue-endemic dataset a
``plt < 10 → APPROPRIATE`` auto-clear would systematically false-clear
inappropriate transfusions, so it was removed. Unlike Hb < 7 (≈no
contraindication population), a low platelet count has 6+ named exclusions.

Out of this pure gate's scope (handled at dispatch, docs plan §5.3 stage 4):
MTP-window suppression — platelet units co-ordered inside an active
massive-transfusion-protocol window are suppressed as independent audit units
upstream (the MTP co-order is a signal, never a second audit row). That needs
cross-order context this single-order gate does not have.
"""

from __future__ import annotations

from bba.platelet_classifier.models import (
    PlateletClassifierInputs,
    PlateletClassifierResult,
)

PLATELET_REVIEW_CEILING: float = 100.0
"""Count (×10³/µL) at or above which a platelet order is
``POTENTIALLY_INAPPROPRIATE`` (routes to review). A SEED pending clinician
sign-off (docs plan §7). OPEN (§5.4): confirm 100 vs an 80-100k band for the
high-bleeding-risk surgical rule — surgical indications permit up to 80-100k,
so the ceiling may need to drop to 80. Encoded as a single named constant so
that ruling changes one line."""


def classify_platelet(
    inputs: PlateletClassifierInputs,
) -> PlateletClassifierResult:
    """Return the deterministic platelet classification for ``inputs``.

    Pure function — same input always yields the same result; no module-global
    state, no mutation, never raises on a well-formed
    :class:`PlateletClassifierInputs`. See the module docstring for precedence.

    Invariant: never returns ``APPROPRIATE`` or ``INAPPROPRIATE`` (v1 does not
    auto-clear or auto-condemn platelets). Present-count verdicts
    (``POTENTIALLY_INAPPROPRIATE`` / ``NEEDS_REVIEW``) always route onward.
    """
    count = inputs.platelet_count

    # 1. Missing count — mirror the RBC missing-Hb contract.
    if count is None:
        if inputs.enable_missing_platelet_defer:
            return PlateletClassifierResult(
                classification="NEEDS_REVIEW",
                review_ceiling=None,
                rationale="plt_missing_defer_llm",
            )
        return PlateletClassifierResult(
            classification="INSUFFICIENT_EVIDENCE",
            review_ceiling=None,
            rationale="plt_missing",
        )

    # 2. At/above the review ceiling — reviewed, not auto-condemned.
    if count >= PLATELET_REVIEW_CEILING:
        return PlateletClassifierResult(
            classification="POTENTIALLY_INAPPROPRIATE",
            review_ceiling=PLATELET_REVIEW_CEILING,
            rationale="plt_ge_100",
        )

    # 3. Below the ceiling (incl. < 10) — defer to the LLM. Nothing is
    #    auto-cleared: the indication and the exclusion populations are all
    #    free-text / context-dependent.
    return PlateletClassifierResult(
        classification="NEEDS_REVIEW",
        review_ceiling=PLATELET_REVIEW_CEILING,
        rationale="plt_defer_llm",
    )


__all__ = ("PLATELET_REVIEW_CEILING", "classify_platelet")
