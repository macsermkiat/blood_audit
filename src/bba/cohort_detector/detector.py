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
7. ``CARDIOPULMONARY_COMORBIDITY`` — ICD-10 heart-disease diagnosis
   (no surgery/ESRD/heme signal); threshold 8.0. Diagnosis-only, so it is
   checked after the UNKNOWN guard and never masks missing procedure data.
   (Label name retained for persisted-row compatibility; lung-disease
   diagnoses were removed — the cohort is now heart-disease only.)
8. ``DEFAULT`` — fall-through; threshold 7.0.
"""

from __future__ import annotations

from bba.cohort_detector.models import (
    CohortAssignment,
    CohortInputs,
    CohortLabel,
)
from bba.cohort_detector.rules import (
    COHORT_THRESHOLDS,
    detect_mtp_pattern,
    find_cardiac_history_diagnosis,
    find_cardiopulmonary_comorbidity_diagnosis,
    find_chemo_med,
    find_dialysis_med,
    find_esrd_diagnosis,
    find_heme_malignancy_diagnosis,
    find_recent_cardiac_surgery,
    find_recent_ortho_surgery,
    is_neutropenic,
)


def _make(
    label: CohortLabel,
    *,
    evidence_code: str | None = None,
    evidence_name: str | None = None,
) -> CohortAssignment:
    """Build a :class:`CohortAssignment` with the canonical threshold for ``label``.

    Centralizes the ``label -> threshold`` lookup so the precedence
    chain in :func:`assign_cohort` does not embed thresholds inline
    (single source of truth: :data:`COHORT_THRESHOLDS`).
    """
    return CohortAssignment(
        label=label,
        threshold=COHORT_THRESHOLDS[label],
        evidence_code=evidence_code,
        evidence_name=evidence_name,
    )


def assign_cohort(inputs: CohortInputs) -> CohortAssignment:
    """Return the deterministic cohort assignment for ``inputs``.

    Pure function — same input always yields the same
    :class:`CohortAssignment`. The detector does not mutate ``inputs``,
    does not consult any module-global state, and never raises on a
    well-formed :class:`CohortInputs`. Constructing :class:`CohortInputs`
    with invalid types fails at the Pydantic boundary, not here.
    """
    # Procedure data unavailable -> UNKNOWN before any cohort detection
    # runs (PR #51 Codex P2 finding): if the IPTSUMOPRT join was skipped,
    # we cannot rule out a cardiac / ortho context for an apparent MTP,
    # and the conservative route is NEEDS_REVIEW so a human verifies the
    # transfusion. The downstream classifier and dashboard surface the
    # UNKNOWN-with-MTP-pattern case for that human review.
    if inputs.procedure_events is None:
        return _make(CohortLabel.UNKNOWN)

    # MTP override — auto-bypass safety signal trumps every other cohort
    # in the precedence chain, but only once we have confirmed that
    # procedure data was actually delivered (above).
    mtp_match = detect_mtp_pattern(inputs.blood_orders, inputs.order_datetime)
    if mtp_match is not None:
        return _make(CohortLabel.MTP)

    # ORTHO_CARDIAC checked before plain CARDIAC_SURGERY — it carries
    # the stricter 8.0 threshold, so when both signals are present the
    # higher target wins.
    ortho_event = find_recent_ortho_surgery(
        inputs.procedure_events, inputs.order_datetime
    )
    cardiac_history = find_cardiac_history_diagnosis(inputs.diagnosis_codes)
    if ortho_event is not None and cardiac_history is not None:
        return _make(
            CohortLabel.ORTHO_CARDIAC,
            evidence_code=ortho_event.icd9,
            evidence_name=ortho_event.name,
        )

    cardiac_event = find_recent_cardiac_surgery(
        inputs.procedure_events, inputs.order_datetime
    )
    if cardiac_event is not None:
        return _make(
            CohortLabel.CARDIAC_SURGERY,
            evidence_code=cardiac_event.icd9,
            evidence_name=cardiac_event.name,
        )

    esrd_dx = find_esrd_diagnosis(inputs.diagnosis_codes)
    dialysis_med = find_dialysis_med(inputs.med_events, inputs.order_datetime)
    if esrd_dx is not None and dialysis_med is not None:
        return _make(
            CohortLabel.ESRD_EPO,
            evidence_code=esrd_dx,
            evidence_name=dialysis_med.drug,
        )

    heme_dx = find_heme_malignancy_diagnosis(inputs.diagnosis_codes)
    chemo_med = find_chemo_med(inputs.med_events, inputs.order_datetime)
    if (
        heme_dx is not None
        and chemo_med is not None
        and is_neutropenic(inputs.anc_value)
    ):
        return _make(
            CohortLabel.HEME_MALIGNANCY_ACTIVE,
            evidence_code=heme_dx,
            evidence_name=chemo_med.drug,
        )

    # CARDIOPULMONARY_COMORBIDITY — diagnosis-only heart disease, checked
    # AFTER the surgery / ESRD / heme cohorts (each more specific) and BEFORE
    # DEFAULT. It raises the 7.0 default floor to 8.0. Reached only once
    # procedure data was confirmed present (the UNKNOWN guard above), so a
    # diagnosis-based match never masks the missing-procedure-data invariant.
    cardiopulmonary_dx = find_cardiopulmonary_comorbidity_diagnosis(
        inputs.diagnosis_codes
    )
    if cardiopulmonary_dx is not None:
        return _make(
            CohortLabel.CARDIOPULMONARY_COMORBIDITY,
            evidence_code=cardiopulmonary_dx,
        )

    return _make(CohortLabel.DEFAULT)


__all__ = ("assign_cohort",)
