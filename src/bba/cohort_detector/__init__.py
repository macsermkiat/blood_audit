"""bba.cohort_detector — deterministic cohort + Hb-threshold assignment.

See issue #7 for acceptance criteria. PRD §5 (Implementation Decisions)
specifies the per-cohort thresholds, detection signals, and fallback to
``cohort_unknown`` when procedure data is unavailable.

Public function: :func:`assign_cohort` returns a :class:`CohortAssignment`
``(label, threshold, evidence_code, evidence_name)`` consumed directly by
:mod:`bba.deterministic_classifier` (#8). The threshold is numeric — never
an enum — so the classifier compares it against the audit-time Hb without
re-interpretation. ``MTP``, ``HEME_MALIGNANCY_ACTIVE``, and the missing-data
``UNKNOWN`` cohorts carry ``threshold=None`` because they are not
threshold-driven.

This module is the input for #8 (deterministic_classifier) and the
threshold-decision row of every :class:`bba.audit_orders.AuditOrder`.
"""

from bba.cohort_detector.detector import assign_cohort
from bba.cohort_detector.models import (
    BloodOrderEvent,
    CohortAssignment,
    CohortInputs,
    CohortLabel,
    MedEvent,
    OperativeEvent,
)
from bba.cohort_detector.rules import (
    ANC_NEUTROPENIA_THRESHOLD,
    CARDIAC_HISTORY_ICD10_PREFIXES,
    CARDIAC_SURGERY_CODE_PREFIXES,
    CARDIAC_SURGERY_EXCLUDED_CODES,
    CARDIAC_SURGERY_LOOKBACK,
    CARDIAC_SURGERY_THRESHOLD,
    CARDIOPULMONARY_COMORBIDITY_ICD10_PREFIXES,
    CARDIOPULMONARY_COMORBIDITY_THRESHOLD,
    CHEMO_LOOKBACK,
    CHEMO_MED_KEYWORDS,
    COHORT_THRESHOLDS,
    DEFAULT_THRESHOLD,
    DIALYSIS_LOOKBACK,
    DIALYSIS_MED_KEYWORDS,
    ESRD_EPO_THRESHOLD,
    ESRD_ICD10_CODES,
    HEME_MALIGNANCY_ICD10_PREFIXES,
    MTP_RBC_UNIT_THRESHOLD,
    MTP_TIME_WINDOW,
    ORTHO_CARDIAC_THRESHOLD,
    ORTHO_SURGERY_CODE_PREFIXES,
    detect_mtp_pattern,
    find_cardiac_history_diagnosis,
    find_cardiopulmonary_comorbidity_diagnosis,
    find_chemo_med,
    find_dialysis_med,
    find_esrd_diagnosis,
    find_heme_malignancy_diagnosis,
    find_recent_cardiac_surgery,
    find_recent_ortho_surgery,
    is_cardiac_surgery_code,
    is_chemo_med,
    is_dialysis_med,
    is_neutropenic,
    is_ortho_surgery_code,
    normalize_icd9,
)

__all__ = [
    "ANC_NEUTROPENIA_THRESHOLD",
    "BloodOrderEvent",
    "CARDIAC_HISTORY_ICD10_PREFIXES",
    "CARDIAC_SURGERY_CODE_PREFIXES",
    "CARDIAC_SURGERY_EXCLUDED_CODES",
    "CARDIAC_SURGERY_LOOKBACK",
    "CARDIAC_SURGERY_THRESHOLD",
    "CARDIOPULMONARY_COMORBIDITY_ICD10_PREFIXES",
    "CARDIOPULMONARY_COMORBIDITY_THRESHOLD",
    "CHEMO_LOOKBACK",
    "CHEMO_MED_KEYWORDS",
    "COHORT_THRESHOLDS",
    "CohortAssignment",
    "CohortInputs",
    "CohortLabel",
    "DEFAULT_THRESHOLD",
    "DIALYSIS_LOOKBACK",
    "DIALYSIS_MED_KEYWORDS",
    "ESRD_EPO_THRESHOLD",
    "ESRD_ICD10_CODES",
    "HEME_MALIGNANCY_ICD10_PREFIXES",
    "MTP_RBC_UNIT_THRESHOLD",
    "MTP_TIME_WINDOW",
    "MedEvent",
    "ORTHO_CARDIAC_THRESHOLD",
    "ORTHO_SURGERY_CODE_PREFIXES",
    "OperativeEvent",
    "assign_cohort",
    "detect_mtp_pattern",
    "find_cardiac_history_diagnosis",
    "find_cardiopulmonary_comorbidity_diagnosis",
    "find_chemo_med",
    "find_dialysis_med",
    "find_esrd_diagnosis",
    "find_heme_malignancy_diagnosis",
    "find_recent_cardiac_surgery",
    "find_recent_ortho_surgery",
    "is_cardiac_surgery_code",
    "is_chemo_med",
    "is_dialysis_med",
    "is_neutropenic",
    "is_ortho_surgery_code",
    "normalize_icd9",
]
