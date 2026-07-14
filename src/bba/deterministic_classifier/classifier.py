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
6. Pre-op crossmatch ≤ 72 h upcoming, Hb ≥ cohort floor
                                    → ``NEEDS_REVIEW`` (``rationale=preop_defer_llm``,
                                       ``bypass_reason=none``); a reservation is not
                                       an indication, so it DEFERS to the LLM instead
                                       of auto-clearing (it does not skip the LLM). A
                                       sub-threshold Hb is NOT deferred — it clears at
                                       step 10 (its Hb is the indication).
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
"""PRD §6 Hb (g/dL) ceiling AT OR ABOVE which the deterministic
pre-classification is ``POTENTIALLY_INAPPROPRIATE`` (the ``hb_ge_10``
branch below; LLM may override on positive evidence). The boundary is
inclusive: Hb exactly 10.0 is the high-Hb side, and RBC task-mode
dispatch (:func:`bba.audit_pipeline.pipeline.rbc_task_mode`) mirrors
this."""

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


def periop_envelope(
    *,
    surgical_context: bool,
    intraop_transfusion: bool,
    procedure_proximity_hours: float | None,
    upcoming_procedure_hours: float | None,
) -> bool:
    """Peri-operative envelope for the returns-ledger exemption (#123).

    ``True`` when a confirmed transfusion sits in a surgical / procedural
    context, so it is exempt from appropriateness judgment (anaesthesia
    frequently does not chart the indication and scoring it would be
    unfair). The envelope fires on ANY of:

    * a surgery is charted (``surgical_context``);
    * a charted intra-operative transfusion (``intraop_transfusion``);
    * a blood-requiring procedure within the peri-procedural window
      (``procedure_proximity_hours <= PERI_PROCEDURAL_WINDOW_HOURS``);
    * a reserve-ahead upcoming procedure within the pre-op crossmatch
      window (``upcoming_procedure_hours <= PRE_OP_CROSSMATCH_WINDOW_HOURS``
      — this catches the pre-op standby case, e.g. TAVI, flagged by its
      procedure rather than an intra-op marker).

    The proximity / upcoming terms are windowed with the SAME bounds the
    classifier applies to these signals for its own peri-procedural and
    pre-op-crossmatch gates (single source of truth), because
    ``procedure_proximity_hours`` / ``upcoming_procedure_hours`` are only
    "no operative event found in the caller's lookback / lookahead window",
    not "nearby": a remote surgery weeks away is neither peri-procedural nor
    a reserve-ahead reservation, and must not exempt an unrelated ward
    transfusion. This is a pure predicate, computed at the flag-gated wiring
    sites and passed to :func:`classify` as ``returns_periop_context`` so the
    classifier stays a pure function of its inputs.
    """
    peri_procedural = (
        procedure_proximity_hours is not None
        and procedure_proximity_hours <= PERI_PROCEDURAL_WINDOW_HOURS
    )
    reserve_ahead = (
        upcoming_procedure_hours is not None
        and upcoming_procedure_hours <= PRE_OP_CROSSMATCH_WINDOW_HOURS
    )
    return bool(
        surgical_context or intraop_transfusion or peri_procedural or reserve_ahead
    )


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

    # A complete all-returned ledger (returns_disposition == "not_transfused")
    # proves that no transfusion occurred — every physical unit reached a
    # non-transfusion terminal (returned or crossmatch-incompatible) — and
    # therefore dominates every clinical/Hb tier. With a guaranteed-complete
    # export an over-dispensed all-returned order also derives "not_transfused"
    # upstream (spec #119 complete-ledger go-live relaxed the earlier NARROW
    # exact-count guard). Structured intra-operative transfusion or major blood
    # loss contradicts the negative disposition; fail loud by leaving the order
    # in the legacy decision chain.
    hard_transfusion_contradiction = inputs.periop_intraop_transfusion or (
        inputs.periop_blood_loss_ml is not None
        and inputs.periop_blood_loss_ml >= PERIOP_MIN_EBL_ML
    )
    if (
        inputs.returns_disposition == "not_transfused"
        and not hard_transfusion_contradiction
    ):
        return ClassifierResult(
            classification="RETURNED_NOT_TRANSFUSED",
            bypass_reason=BypassReason.RETURNED_NOT_TRANSFUSED,
            cohort_threshold=threshold,
            rationale="returned_not_transfused",
        )

    # A confirmed transfusion (a unit not returned, or an explicit
    # transfused-status unit) inside a peri-operative envelope is exempt from
    # appropriateness judgment — anaesthesia frequently does not chart the
    # indication, so scoring it would be unfair. Hb-independent and dominates
    # every clinical tier, exactly like the returned exit above. A confirmed
    # transfusion with NO peri-op context falls through and is judged normally
    # (it can still be POTENTIALLY_INAPPROPRIATE). The envelope is computed at
    # the flag-gated wiring sites (see :func:`periop_envelope`).
    if inputs.returns_disposition == "transfused" and inputs.returns_periop_context:
        return ClassifierResult(
            classification="PERIOP_TRANSFUSION_EXEMPT",
            bypass_reason=BypassReason.PERIOP_TRANSFUSION_EXEMPT,
            cohort_threshold=threshold,
            rationale="periop_transfusion_exempt",
        )

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

    # 6. Pre-op crossmatch — upcoming procedure within 72 h after the order
    #    anchor. This is NOT a clearing bypass: a crossmatch *reservation* is
    #    not a transfusion *indication*, and an upcoming surgery can hide an
    #    active problem the reservation masks (case 68080335 documented
    #    ongoing LGIB only in the nurse note, which neither the AI nor the
    #    human caught). So instead of auto-APPROPRIATE we DEFER to the LLM —
    #    NEEDS_REVIEW routes to the note-reading LLM leg (it is not in the
    #    DETERMINISTIC_FINAL set), which reads the ±72 h note window and
    #    decides on the actual clinical picture. bypass_reason stays NONE
    #    because no clearing bypass fired.
    #
    #    Only defer when the Hb tier would not already clear the order: a
    #    sub-threshold Hb (below the cohort floor) is itself the indication
    #    and must clear (APPROPRIATE, below) regardless of an upcoming
    #    procedure. Deferring it would trap the order — the LLM re-clears the
    #    sub-threshold Hb and the over-clear guardrail (which exempts only
    #    Hb < 7.0) now ASSERTS INAPPROPRIATE (#94), worsening the trap
    #    (Codex P2). Non-threshold cohorts (Hb-independent) fall through to
    #    their own routing below. Hb < 7.0
    #    already returned APPROPRIATE at step 2, so this only gates the
    #    [cohort_floor, ...) gray-zone / high-Hb pre-op cases.
    upcoming = inputs.upcoming_procedure_hours
    if (
        upcoming is not None
        and upcoming <= PRE_OP_CROSSMATCH_WINDOW_HOURS
        and threshold is not None
        and hb.value_g_dl >= threshold
    ):
        return ClassifierResult(
            classification="NEEDS_REVIEW",
            bypass_reason=BypassReason.NONE,
            cohort_threshold=threshold,
            rationale="preop_defer_llm",
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
    "periop_envelope",
)
