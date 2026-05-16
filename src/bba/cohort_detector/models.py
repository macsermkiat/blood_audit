"""Pydantic v2 + dataclass models for the cohort_detector module.

All public types are immutable. The detector is a pure function from a
:class:`CohortInputs` snapshot to a :class:`CohortAssignment`; downstream
:mod:`bba.deterministic_classifier` (#8) consumes ``label`` + ``threshold``
directly without further interpretation.

Public-surface invariants (issue #7 acceptance criteria):

* ``CohortAssignment.threshold`` is numeric (``float``) — never an enum
  member — when the cohort has a hard Hb threshold. ``MTP``,
  ``HEME_MALIGNANCY_ACTIVE``, and the missing-data ``UNKNOWN`` cohort
  carry ``threshold=None`` because they are not threshold-driven cohorts:
  MTP auto-bypasses to APPROPRIATE, heme is T2-supportive (not a hard
  number), and UNKNOWN routes to NEEDS_REVIEW. Tests assert this contract
  explicitly so a refactor cannot quietly downgrade the type.

* ``CohortInputs.procedure_events`` distinguishes "data unavailable"
  (``None``) from "patient has no operative events" (``()`` empty tuple).
  The first triggers ``CohortLabel.UNKNOWN``; the second is a clean
  no-cardiac/no-ortho signal. PRD §5 + Round 2 fix N1 forbid silently
  applying the default 7.0 threshold when procedure data is missing.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel, ConfigDict


class CohortLabel(StrEnum):
    """The seven cohort outcomes assignable by :func:`assign_cohort`.

    Values are ``snake_case`` and intended for direct logging and downstream
    classifier dispatch. Adding a new label must be paired with a threshold
    decision in :data:`COHORT_THRESHOLDS` (see :mod:`bba.cohort_detector.rules`).
    """

    CARDIAC_SURGERY = "cardiac_surgery"
    ORTHO_CARDIAC = "ortho_cardiac"
    ESRD_EPO = "esrd_epo"
    MTP = "mtp"
    HEME_MALIGNANCY_ACTIVE = "heme_malignancy_active"
    DEFAULT = "default"
    UNKNOWN = "cohort_unknown"


class OperativeEvent(BaseModel):
    """One row from the joined IPTSUMOPRT / ICD9CM tables for a patient.

    ``icd9`` is the ICD-9-CM Vol 3 procedure code in dot-stripped form
    (e.g., ``"3601"`` for "Single vessel PTCA"). The orchestrator strips
    any decimal point during the join so this module's prefix matchers
    can stay format-stable.

    ``or_flag`` is True iff IPTSUMOPRT.Orflag == 1 (operating-room procedure).
    The cohort allow-lists require ``or_flag`` to gate non-OR cardiac items
    like 894 (cardiac stress test) and 3796 (pacemaker pulse generator).

    ``operative_datetime`` MUST be tz-aware UTC; naive datetimes are
    rejected by :class:`AwareDatetime`. ``name`` is the procedure
    description from the ICD9CM dictionary, carried so the
    :class:`CohortAssignment.evidence_name` field can render without a
    second join at decision time.
    """

    model_config = ConfigDict(frozen=True)

    icd9: str
    or_flag: bool
    operative_datetime: AwareDatetime
    name: str | None = None


class MedEvent(BaseModel):
    """One MED-table row: drug name plus tz-aware administration timestamp.

    ``drug`` is the raw HOSxP DRUG string. The cohort detector applies
    case-insensitive substring matching against keyword lists
    (:data:`DIALYSIS_MED_KEYWORDS`, :data:`CHEMO_MED_KEYWORDS`); brand /
    generic / route variants are matched without per-row normalization
    upstream.
    """

    model_config = ConfigDict(frozen=True)

    drug: str
    timestamp: AwareDatetime


class BloodOrderEvent(BaseModel):
    """One historical RBC order used for MTP temporal-cluster detection.

    ``rbc_units`` is the count of RBC units on this single order (the
    granular unit, not a transfusion-batch sum). ``co_ordered_with_ffp``
    and ``co_ordered_with_platelets`` flag the parallel-component
    ordering pattern that is the second arm of the MTP rule (≥4 RBC units
    in 1 h **OR** RBC + FFP + platelets co-ordered in the same window).
    """

    model_config = ConfigDict(frozen=True)

    timestamp: AwareDatetime
    rbc_units: int
    co_ordered_with_ffp: bool = False
    co_ordered_with_platelets: bool = False


class CohortInputs(BaseModel):
    """Joined per-(audit_id) inputs the deterministic detector consumes.

    ``procedure_events`` semantics — and the reason it is ``... | None``
    rather than always-tuple — see :meth:`module-level docstring`.

    ``order_datetime`` is the audit anchor (matches
    :attr:`bba.audit_orders.AuditOrder.order_datetime`); the cardiac-surgery
    30-day lookback and MTP 1-h cluster windows resolve relative to it.
    ``anc_value`` is the absolute neutrophil count at order time (cells/uL),
    used by the heme-malignancy rule.
    """

    model_config = ConfigDict(frozen=True)

    audit_id: str
    hn: str
    an: str | None
    order_datetime: AwareDatetime
    procedure_events: tuple[OperativeEvent, ...] | None
    diagnosis_codes: tuple[str, ...]
    med_events: tuple[MedEvent, ...]
    blood_orders: tuple[BloodOrderEvent, ...]
    anc_value: int | None


class CohortAssignment(BaseModel):
    """Output of :func:`bba.cohort_detector.assign_cohort`.

    ``threshold`` is numeric for the threshold-driven cohorts (cardiac:
    7.5; ortho-cardiac and ESRD-EPO: 8.0; default: 7.0) and ``None`` for
    the non-threshold cohorts (``MTP``, ``HEME_MALIGNANCY_ACTIVE``,
    ``UNKNOWN``). The downstream classifier interprets the ``label`` and
    ``threshold`` together — never the threshold alone.

    ``evidence_code`` and ``evidence_name`` carry the single triggering
    fact (e.g., the matched procedure ICD-9 code + name, the matched
    ICD-10 diagnosis code, or the dialysis drug name). For ``DEFAULT``
    and ``UNKNOWN`` both are ``None`` — there is no positive triggering
    evidence to surface.
    """

    model_config = ConfigDict(frozen=True)

    label: CohortLabel
    threshold: float | None
    evidence_code: str | None
    evidence_name: str | None


__all__: Sequence[str] = (
    "BloodOrderEvent",
    "CohortAssignment",
    "CohortInputs",
    "CohortLabel",
    "MedEvent",
    "OperativeEvent",
)
