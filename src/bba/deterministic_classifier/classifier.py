"""Top-level deterministic classification entry point (issue #8).

The single public function :func:`classify` is the deterministic composition
over the Hb-tier rules + five bypass pathways. Precedence (top wins) is
specified in PRD §"Implementation Decisions §6":

1. Hb missing                       → positive-evidence pre-pass
   (``APPROPRIATE`` on MTP / peri-procedural / hard peri-op evidence).
   With the pre-pass flag ON but no hard evidence → ``NEEDS_REVIEW``
   (defer to LLM); with the flag OFF → ``INSUFFICIENT_EVIDENCE``.
2. Hb < 7.0 g/dL                    → ``APPROPRIATE`` (global low-Hb rule)
3. Cohort ``MTP``                   → ``APPROPRIATE`` (``bypass_reason=mtp``)
4. Cohort ``UNKNOWN``               → ``NEEDS_REVIEW`` (no bypass)
5. Bypass: peri-procedural ≤ 6 h    → ``APPROPRIATE`` (``bypass_reason=peri_procedural_6h``)
6. Bypass: pre-op crossmatch ≤ 72 h → ``APPROPRIATE`` (``bypass_reason=pre_op_crossmatch``)
7. Bypass: delta-Hb trigger fired   → ``APPROPRIATE`` (``bypass_reason=delta_hb``)
8. Hemodilution: Hb < threshold AND ≥ 2 L crystalloid in 4 h
                                    → ``NEEDS_REVIEW`` (``bypass_reason=hemodilution_flagged``)
9. Single-low-Hb-no-trend: Hb < threshold AND ``needs_review_single_low_hb``
                                    → ``NEEDS_REVIEW`` (``bypass_reason=none``;
                                       isolated Hb < 8 with no prior 24 h
                                       observation cannot be interpreted as
                                       confirmed anemia without a trend —
                                       PR #52 Codex P1 + hb_lookup contract)
10. Plain Hb-tier rule:
   - Hb < ``cohort_threshold``       → ``APPROPRIATE``
   - ``cohort_threshold`` ≤ Hb < 10  → ``NEEDS_REVIEW``
   - Hb ≥ 10                         → ``POTENTIALLY_INAPPROPRIATE``

Precedence notes:

* Hb missing runs a positive-evidence pre-pass first: hard,
  Hb-independent signals auto-classify ``APPROPRIATE`` just as the
  Hb-present path would (SEED pending clinical sign-off) —
    - active MTP                       → ``bypass_mtp_hb_missing``
    - peri-procedural ≤ 6 h            → ``bypass_peri_procedural_hb_missing``
    - hard peri-op note evidence       → ``bypass_periop_evidence_hb_missing``
      (a charted intra-op transfusion, or EBL ≥ ``PERIOP_MIN_EBL_ML``)
  — and the distinct rationale slugs keep these auditable. Interpreted
  signals (a triggered delta-Hb flag, which needs a current Hb to compute),
  the weaker pre-op crossmatch, and the SOFT "a surgery is documented" /
  "a procedure is upcoming" cues do NOT fire — stale upstream state cannot
  override the structural "no current Hb" fact, and a merely-scheduled
  surgery is exactly what the design refuses to rubber-stamp without an Hb.
  With the flag ON but no hard evidence the case is ``NEEDS_REVIEW``
  (``hb_missing_defer_llm``) so it routes to the LLM — which reads the
  free-text prose and either grounds an ``APPROPRIATE``, floors a genuinely
  ambiguous case to human review, or records an ``INSUFFICIENT_EVIDENCE``
  documentation gap — rather than dead-ending here. With the flag OFF the
  pre-pass is inert and the case is terminal ``INSUFFICIENT_EVIDENCE``.
* Hb < 7.0 g/dL is globally ``APPROPRIATE`` before cohort-specific review
  routes. This includes UNKNOWN / non-threshold cohorts and the safety flags
  that otherwise protect the cohort-threshold branch.
* MTP precedes peri-procedural / delta-Hb so the bypass_reason reflects the
  most clinically load-bearing fact when multiple signals fire.
* Hemodilution is checked only inside the ``Hb < threshold`` branch — it is
  scoped to the cohort-threshold auto-APPROPRIATE case it would otherwise
  produce, and never overrides the global Hb < 7.0 rule.
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

UNIVERSAL_LOW_HB_APPROPRIATE_THRESHOLD: float = 7.0
"""Hb (g/dL) below which RBC/LPRC orders are deterministically
``APPROPRIATE`` before cohort-specific routing."""

PERI_PROCEDURAL_WINDOW_HOURS: float = 6.0
"""PRD §6 peri-procedural bypass window (hours). A procedure within this
many hours BEFORE the order anchor auto-bypasses to ``APPROPRIATE``."""

PRE_OP_CROSSMATCH_WINDOW_HOURS: float = 72.0
"""Pre-op crossmatch bypass window (hours). A procedure within this many
hours AFTER the order anchor is treated as surgical-preparation evidence."""

HEMODILUTION_CRYSTALLOID_LITERS: float = 2.0
"""PRD §6 + Round 1 B5: ≥ 2 L of crystalloid in the prior 4 h marks the
Hb as hemodilution-suspect. Triggers ``NEEDS_REVIEW`` rather than
auto-APPROPRIATE in the cohort-threshold branch, but never overrides the
global Hb < 7.0 rule."""

PERIOP_MIN_EBL_ML: int = 500
"""Estimated-blood-loss floor (mL) that counts as a HARD peri-op signal.

Single source of truth shared by two consumers so they cannot drift:

* the missing-Hb pre-pass below — at or above this, a documented blood
  loss auto-classifies ``APPROPRIATE`` despite the absent Hb (the surgical
  haemorrhage is the indication);
* :data:`bba.audit_pipeline.replay.PERIOP_GUARDRAIL_MIN_EBL_ML` (an alias
  of this value) — the LLM contradiction guardrail's escalation bar.

Sub-500 mL losses are routine and do not, on their own, justify a
transfusion or contradict an "insufficient evidence" verdict."""


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

    # 1. Hb missing — positive-evidence pre-pass (SEED pending clinical
    #    sign-off). Gated behind ``inputs.enable_missing_hb_positive_evidence``,
    #    which defaults to False. When OFF the original PRD spec applies
    #    (missing Hb → INSUFFICIENT_EVIDENCE) and no bypass fires; when ON,
    #    hard Hb-independent signals (active MTP, peri-procedural ≤ 6 h, or a
    #    documented intra-op transfusion / EBL ≥ PERIOP_MIN_EBL_ML)
    #    auto-classify APPROPRIATE exactly as the Hb-present path would,
    #    preserving the canonical MTP → UNKNOWN → peri-procedural ordering.
    #    Everything else with the flag ON defers to the LLM (NEEDS_REVIEW)
    #    rather than dead-ending. Interpreted signals (delta-Hb, which needs
    #    a current Hb), the weaker pre-op crossmatch, and the soft "surgery
    #    documented/upcoming" cues are NOT auto-approved here.
    if hb.value_g_dl is None:
        if inputs.enable_missing_hb_positive_evidence:
            # MTP precedes everything — the cluster pattern is Hb-independent.
            if cohort.label == CohortLabel.MTP:
                return ClassifierResult(
                    classification="APPROPRIATE",
                    bypass_reason=BypassReason.MTP,
                    cohort_threshold=threshold,
                    rationale="bypass_mtp_hb_missing",
                )
            # UNKNOWN must NOT be auto-bypassed by peri-procedural or by
            # peri-op note evidence — mirrors the Hb-present order (UNKNOWN
            # precedes peri-procedural). Missing Hb + unknown context stays
            # the dominant documentation gap; it defers to the LLM below
            # rather than auto-approving.
            if cohort.label != CohortLabel.UNKNOWN:
                proximity = inputs.procedure_proximity_hours
                if proximity is not None and proximity <= PERI_PROCEDURAL_WINDOW_HOURS:
                    return ClassifierResult(
                        classification="APPROPRIATE",
                        bypass_reason=BypassReason.PERI_PROCEDURAL_6H,
                        cohort_threshold=threshold,
                        rationale="bypass_peri_procedural_hb_missing",
                    )
                # HARD peri-op note evidence stands in for the absent Hb: a
                # charted intra-op transfusion, or an estimated blood loss at
                # or above PERIOP_MIN_EBL_ML, is unambiguous surgical-loss
                # documentation. Soft signals (a surgery is merely documented,
                # or an upcoming procedure is scheduled) deliberately do NOT
                # fire here — those are the "surgery exists" cues the design
                # refuses to rubber-stamp on missing Hb, so they fall through
                # to the LLM deferral below.
                if inputs.periop_intraop_transfusion or (
                    inputs.periop_blood_loss_ml is not None
                    and inputs.periop_blood_loss_ml >= PERIOP_MIN_EBL_ML
                ):
                    return ClassifierResult(
                        classification="APPROPRIATE",
                        bypass_reason=BypassReason.PERIOP_EVIDENCE,
                        cohort_threshold=threshold,
                        rationale="bypass_periop_evidence_hb_missing",
                    )
            # Flag on but no HARD Hb-independent positive evidence: route to
            # the LLM rather than dead-ending. A deterministic
            # INSUFFICIENT_EVIDENCE is terminal (never reaches the LLM); the
            # peri-op extractor + free-text prose let the LLM auto-resolve the
            # well-documented majority and floor only the genuinely ambiguous
            # rest. The accuracy invariant holds: the deterministic gate never
            # used a (possibly post-transfusion) Hb here — there is none.
            return ClassifierResult(
                classification="NEEDS_REVIEW",
                bypass_reason=BypassReason.NONE,
                cohort_threshold=threshold,
                rationale="hb_missing_defer_llm",
            )
        # Flag off — preserve the original PRD spec: missing Hb with no
        # operator opt-in is a terminal documentation gap.
        return ClassifierResult(
            classification="INSUFFICIENT_EVIDENCE",
            bypass_reason=BypassReason.NONE,
            cohort_threshold=threshold,
            rationale="hb_missing",
        )

    # 2. Global low-Hb rule — Hb < 7.0 is appropriate before cohort-specific
    #    review routes, including UNKNOWN / non-threshold cohorts and safety
    #    flags such as hemodilution or isolated low-Hb-without-trend.
    if hb.value_g_dl < UNIVERSAL_LOW_HB_APPROPRIATE_THRESHOLD:
        return ClassifierResult(
            classification="APPROPRIATE",
            bypass_reason=BypassReason.NONE,
            cohort_threshold=threshold,
            rationale="hb_lt_7_universal",
        )

    # 3. MTP cohort auto-bypass — cluster pattern trumps every remaining Hb tier.
    if cohort.label == CohortLabel.MTP:
        return ClassifierResult(
            classification="APPROPRIATE",
            bypass_reason=BypassReason.MTP,
            cohort_threshold=threshold,
            rationale="bypass_mtp",
        )

    # 4. Cohort UNKNOWN — for Hb >= 7.0, refuse to default to 7.0 when
    #    procedure data was missing.
    if cohort.label == CohortLabel.UNKNOWN:
        return ClassifierResult(
            classification="NEEDS_REVIEW",
            bypass_reason=BypassReason.NONE,
            cohort_threshold=threshold,
            rationale="cohort_unknown",
        )

    # 5. Peri-procedural bypass — procedure within the 6 h window before
    #    the order anchor. ``<=`` per PRD §6 ("within 6 h" inclusive).
    proximity = inputs.procedure_proximity_hours
    if proximity is not None and proximity <= PERI_PROCEDURAL_WINDOW_HOURS:
        return ClassifierResult(
            classification="APPROPRIATE",
            bypass_reason=BypassReason.PERI_PROCEDURAL_6H,
            cohort_threshold=threshold,
            rationale="bypass_peri_procedural",
        )

    # 6. Pre-op crossmatch bypass — upcoming procedure within 72 h after
    #    the order anchor. This catches blood ordered ahead of surgery.
    upcoming = inputs.upcoming_procedure_hours
    if upcoming is not None and upcoming <= PRE_OP_CROSSMATCH_WINDOW_HOURS:
        return ClassifierResult(
            classification="APPROPRIATE",
            bypass_reason=BypassReason.PRE_OP_CROSSMATCH,
            cohort_threshold=threshold,
            rationale="bypass_pre_op_crossmatch",
        )

    # 7. Delta-Hb bypass — at least one window in the HbLookupResult
    #    fired its tiered threshold (PRD §3 + Round 2 E3).
    if hb.delta_hb_bypass:
        return ClassifierResult(
            classification="APPROPRIATE",
            bypass_reason=BypassReason.DELTA_HB,
            cohort_threshold=threshold,
            rationale="bypass_delta_hb",
        )

    # Non-threshold-driven cohorts that survived the global Hb < 7.0 rule
    # and the bypasses above. In practice this is Hb >= 7.0
    # HEME_MALIGNANCY_ACTIVE; MTP and UNKNOWN exited above. The T2-supportive
    # heme cohort defers to the LLM stage for context interpretation.
    if threshold is None:
        return ClassifierResult(
            classification="NEEDS_REVIEW",
            bypass_reason=BypassReason.NONE,
            cohort_threshold=None,
            rationale="cohort_non_threshold",
        )

    # Plain cohort-threshold rule with two carve-outs scoped to the remaining
    # would-be auto-APPROPRIATE branch (which is now always Hb >= 7.0):
    #   * Hemodilution-flagged (≥ 2 L crystalloid in 4 h)
    #   * Single-low-Hb-no-trend (isolated Hb < 8 with no prior 24 h
    #     observation), for threshold-driven cohorts above the global 7.0 rule.
    hb_value = hb.value_g_dl
    if hb_value < threshold:
        if inputs.crystalloid_liters_prior_4h >= HEMODILUTION_CRYSTALLOID_LITERS:
            return ClassifierResult(
                classification="NEEDS_REVIEW",
                bypass_reason=BypassReason.HEMODILUTION_FLAGGED,
                cohort_threshold=threshold,
                rationale="bypass_hemodilution",
            )
        if hb.needs_review_single_low_hb:
            return ClassifierResult(
                classification="NEEDS_REVIEW",
                bypass_reason=BypassReason.NONE,
                cohort_threshold=threshold,
                rationale="single_low_hb_no_trend",
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
    "PRE_OP_CROSSMATCH_WINDOW_HOURS",
    "UNIVERSAL_LOW_HB_APPROPRIATE_THRESHOLD",
    "classify",
)
