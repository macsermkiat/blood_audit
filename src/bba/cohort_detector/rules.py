"""Per-cohort predicate functions + allow-list constants.

Constants are exported so the test suite (and downstream modules) can
parametrize against the canonical seeds. Every allow-list in this module
is a Phase-1 SEED requiring clinical sign-off before production; the
issue body documents the seed sources (ICD-9-CM Vol 3 procedure-code
ranges, ICD-10 chapter prefixes for cardiac history / ESRD / heme
malignancy).

Predicates return either the matching evidence (an :class:`OperativeEvent`,
:class:`MedEvent`, :class:`BloodOrderEvent`, or matched ICD-10 code
string) or ``None``. The composer in :mod:`bba.cohort_detector.detector`
turns matches into :class:`CohortAssignment` records; predicates here do
not concern themselves with cohort precedence.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from bba.cohort_detector.models import (
    BloodOrderEvent,
    CohortLabel,
    MedEvent,
    OperativeEvent,
)

# =============================================================================
# Cohort thresholds
# =============================================================================

DEFAULT_THRESHOLD: float = 7.0
"""PRD §5 fall-through Hb threshold (g/dL) for the ``DEFAULT`` cohort."""

CARDIAC_SURGERY_THRESHOLD: float = 7.5
"""Hb threshold (g/dL) for the ``cardiac_surgery`` cohort (PRD §5 + Round 1)."""

ORTHO_SURGERY_THRESHOLD: float = 8.0
"""Hb threshold (g/dL) for the ``ortho_surgery`` cohort.

Chula orthopedic-surgery guideline (``policy_2``/``policy_3``): "ผู้ป่วยที่จะ
ผ่าตัดเกี่ยวกับระบบกระดูก: 8 g/dL". An orthopedic operation raises the
transfusion floor to 8.0 on its own — a cardiac-history diagnosis is NOT
required (unlike the deprecated ``ortho_cardiac`` cohort). Ranked above
``cardiac_surgery`` (7.5) so the higher floor wins when a patient carries
both surgical contexts."""

ORTHO_CARDIAC_THRESHOLD: float = 8.0
"""Hb threshold (g/dL) for the DEPRECATED ``ortho_cardiac`` cohort (PRD §5).

No longer emitted by :func:`bba.cohort_detector.assign_cohort` — the
orthopedic-surgery guideline applies the 8.0 floor to an ortho operation
alone (see :data:`ORTHO_SURGERY_THRESHOLD`), which subsumes the fused
"ortho + cardiac history" cohort. Retained (with its threshold) so
persisted audit-store rows written before the split still resolve."""

ESRD_EPO_THRESHOLD: float = 7.0
"""Restrictive Hb floor (g/dL) for the ``esrd_epo`` cohort.

Chronic EPO-managed anemia is too vague to justify a permissive floor.
Patients who also have heart disease or an orthopedic/cardiac-surgery
context retain the applicable higher floor through cohort precedence."""

CARDIOPULMONARY_COMORBIDITY_THRESHOLD: float = 8.0
"""Hb threshold (g/dL) for the ``cardiopulmonary_comorbidity`` cohort.

Restrictive-transfusion practice raises the trigger from the 7.0 default to
8.0 for patients carrying a heart-disease comorbidity (clinician rule:
"> 8 with heart disease, > 7 without"). Distinct from the surgery-based
``cardiac_surgery`` cohort (7.5) — this cohort is diagnosis-driven, not
procedure-driven.

NOTE: the ``cardiopulmonary_comorbidity`` label name is retained for
backward compatibility with persisted rows, but lung-disease diagnoses were
removed from the trigger set (see
:data:`CARDIOPULMONARY_COMORBIDITY_ICD10_PREFIXES`) — the cohort is now
heart-disease only."""

COHORT_THRESHOLDS: dict[CohortLabel, float | None] = {
    CohortLabel.CARDIAC_SURGERY: CARDIAC_SURGERY_THRESHOLD,
    CohortLabel.ORTHO_SURGERY: ORTHO_SURGERY_THRESHOLD,
    CohortLabel.ORTHO_CARDIAC: ORTHO_CARDIAC_THRESHOLD,
    CohortLabel.ESRD_EPO: ESRD_EPO_THRESHOLD,
    CohortLabel.CARDIOPULMONARY_COMORBIDITY: CARDIOPULMONARY_COMORBIDITY_THRESHOLD,
    CohortLabel.MTP: None,
    CohortLabel.HEME_MALIGNANCY_ACTIVE: None,
    CohortLabel.DEFAULT: DEFAULT_THRESHOLD,
    CohortLabel.UNKNOWN: None,
}
"""Single source of truth for the ``label → threshold`` mapping.

Every :class:`CohortLabel` MUST appear as a key. Adding a new label
without adding the corresponding threshold (or explicit ``None``) is a
test failure: :class:`TestThresholdMapTotality` enforces it.
"""

# =============================================================================
# Procedure-code allow-lists (ICD-9-CM Vol 3, dot-stripped)
# =============================================================================

CARDIAC_SURGERY_CODE_PREFIXES: tuple[str, ...] = ("36", "38", "39")
"""Two-digit ICD-9-CM Vol 3 prefixes seeded from issue #7:

* ``36xx`` — PTCA, coronary stents, coronary bypass grafts
* ``38xx`` — aortic / large-vessel resection with replacement
* ``39xx`` — aorta-iliac-femoral bypass

Pre-clinical sign-off; production deployment requires clinical review.
The :data:`CARDIAC_SURGERY_EXCLUDED_CODES` set carves out non-OR cardiac
items that share these prefixes.
"""

CARDIAC_SURGERY_EXCLUDED_CODES: frozenset[str] = frozenset({"894", "3796"})
"""Defense-in-depth carve-out: codes that are non-OR cardiac and must NOT
trigger the cardiac-surgery cohort even if they slip past the
:data:`CARDIAC_SURGERY_CODE_PREFIXES` test (e.g., a future widening of
the prefix list).

* ``894`` — cardiac stress test (a diagnostic, not an operation).
* ``3796`` — pacemaker pulse generator implant; non-OR cardiac procedure.

Per the issue body: "EXCLUDE non-OR cardiac items like 894 and 3796
unless clinical confirms".
"""

ORTHO_SURGERY_CODE_PREFIXES: tuple[str, ...] = ("78", "79", "81")
"""Two-digit ICD-9-CM Vol 3 prefixes seeded from issue #7:

* ``78xx`` — fracture / bone-graft fixation
* ``79xx`` — fracture reduction (closed / open)
* ``81xx`` — joint procedures (hip / knee replacement, arthrodesis)
"""

# =============================================================================
# Diagnosis-code allow-lists (ICD-10)
# =============================================================================

CARDIAC_HISTORY_ICD10_PREFIXES: frozenset[str] = frozenset(
    {"I20", "I21", "I22", "I23", "I24", "I25", "I50"}
)
"""ICD-10 ischemic-heart-disease + heart-failure codes used as the
"cardiac history" half of the ortho_cardiac rule (PRD §5)."""

ESRD_ICD10_CODES: frozenset[str] = frozenset({"N18.5", "N18.6"})
"""End-stage renal disease (CKD stage 5 + dialysis-dependent CKD).
Round 2 fix N1 requires both an ESRD diagnosis AND a dialysis-med
signal — neither alone triggers ``esrd_epo``."""

HEME_MALIGNANCY_ICD10_PREFIXES: tuple[str, ...] = ("C8", "C9")
"""ICD-10 hematological-malignancy chapter prefixes:

* ``C8x`` — Hodgkin / non-Hodgkin lymphoma
* ``C9x`` — leukemia, multiple myeloma, related plasma-cell neoplasms
"""

CARDIOPULMONARY_COMORBIDITY_ICD10_PREFIXES: frozenset[str] = frozenset(
    {
        # Heart disease
        "I11",  # hypertensive heart disease
        "I13",  # hypertensive heart + chronic kidney disease
        "I20",  # angina pectoris
        "I21",  # acute myocardial infarction
        "I22",  # subsequent MI
        "I23",  # MI complications
        "I24",  # other acute ischemic heart disease
        "I25",  # chronic ischemic heart disease
        "I42",  # cardiomyopathy
        "I50",  # heart failure
    }
)
"""ICD-10 heart-disease comorbidity prefixes for the 8.0 g/dL cardiopulmonary
floor.

SEED list, frozen before scoring and pending clinical sign-off (mirrors the
freezing policy of every other allow-list here). Sourced from the ICD-10
ischemic-heart / heart-failure / cardiomyopathy categories (I2x, I11/I13/I42/
I50), NOT chosen by which pilot cases it flips. Essential hypertension (I10)
without heart involvement is deliberately excluded.

Lung-disease (J-code) diagnoses were removed from this cohort: chronic
respiratory disease alone no longer raises the floor. The label name still
reads ``cardiopulmonary_comorbidity`` for backward compatibility with
persisted rows, but the trigger is heart-disease only."""

# =============================================================================
# Medication keyword lists (substring, case-insensitive matching)
# =============================================================================

DIALYSIS_MED_KEYWORDS: tuple[str, ...] = (
    "heparin",
    "sevelamer",
    "cinacalcet",
)
"""Dialysis-context medication keywords. ``heparin`` alone is non-specific
(used in many contexts), but the rule requires it to co-occur with an
ESRD diagnosis (Round 2 fix N1), so the substring match is safe inside
the composed predicate."""

CHEMO_MED_KEYWORDS: tuple[str, ...] = (
    "doxorubicin",
    "cyclophosphamide",
    "vincristine",
    "rituximab",
    "cisplatin",
    "etoposide",
    "cytarabine",
)
"""Seed list of cytotoxic / monoclonal chemotherapeutics. Pre-clinical
sign-off; the canonical list lives with the heme-onc service."""

# =============================================================================
# Numeric thresholds and time windows
# =============================================================================

ANC_NEUTROPENIA_THRESHOLD: int = 500
"""Absolute neutrophil count cutoff (cells/uL) for "active heme
malignancy" per PRD §5 + Round 2 N3. Strict less-than: ANC == 500 does
NOT trigger the heme cohort."""

MTP_RBC_UNIT_THRESHOLD: int = 4
"""Minimum RBC units in a 1-h window for the MTP temporal-cluster rule.
Strict at-or-above: 3 units does NOT trigger; 4 units DOES."""

MTP_TIME_WINDOW: timedelta = timedelta(hours=1)
"""Cluster window for the MTP RBC-unit count. Boundary semantics tested
in :class:`TestMtpBoundary`: just-under 1 h triggers, just-over does not."""

CARDIAC_SURGERY_LOOKBACK: timedelta = timedelta(days=30)
"""Backward window from ``order_datetime`` over which a cardiac-surgery
operative event still counts as "recent". Boundary semantics tested in
:class:`TestCardiacSurgeryLookback`."""

CHEMO_LOOKBACK: timedelta = timedelta(days=30)
"""Backward window for "recent chemo meds" per the issue body. A chemo
agent administered more than 30 days before the audit anchor does NOT
count as active and must not contribute to the heme-malignancy cohort
(otherwise a long-since-completed regimen would over-trigger T2-supportive
classification)."""

DIALYSIS_LOOKBACK: timedelta = timedelta(days=14)
"""Backward window for "active dialysis meds". Two weeks matches the
typical refill cadence for the seed drugs (sevelamer / cinacalcet) and
is conservative enough that intermittent inpatient HD heparin still
counts. ESRD-EPO additionally gates on the ICD-10 N18.5/.6 diagnosis
(Round 2 N1), so a non-ESRD patient on heparin for DVT prophylaxis
is not at risk of misclassification."""

# =============================================================================
# Predicates — each returns the matching evidence (or None)
# =============================================================================


def normalize_icd9(code: str) -> str:
    """Return ``code`` stripped of decimal points and surrounding whitespace.

    The ICD-9-CM Vol 3 prefix matchers operate on dot-stripped form
    (``"36.01"`` and ``"3601"`` both normalize to ``"3601"``).
    """
    return code.strip().replace(".", "")


def is_cardiac_surgery_code(code: str, or_flag: bool) -> bool:
    """True iff ``code`` is a cardiac-surgery operative procedure.

    Three gates, applied in order:

    1. ``or_flag`` must be True (ORFLAG="1" in the ICD9CM dictionary for
       this code; see :class:`OperativeEvent`).
    2. ``code`` (dot-stripped) must NOT be in
       :data:`CARDIAC_SURGERY_EXCLUDED_CODES`.
    3. ``code`` (dot-stripped) must start with one of the prefixes in
       :data:`CARDIAC_SURGERY_CODE_PREFIXES`.

    All three conditions must hold; missing any one returns False.
    """
    if not or_flag:
        return False
    normalized = normalize_icd9(code)
    if normalized in CARDIAC_SURGERY_EXCLUDED_CODES:
        return False
    return normalized.startswith(CARDIAC_SURGERY_CODE_PREFIXES)


def is_ortho_surgery_code(code: str, or_flag: bool) -> bool:
    """True iff ``code`` is an orthopedic operative procedure.

    Same three-gate structure as :func:`is_cardiac_surgery_code` but with
    :data:`ORTHO_SURGERY_CODE_PREFIXES` and no separate exclusion set —
    the orthopedic seed is conservative enough at Phase 1.
    """
    if not or_flag:
        return False
    normalized = normalize_icd9(code)
    return normalized.startswith(ORTHO_SURGERY_CODE_PREFIXES)


def _drug_matches_keywords(drug: str, keywords: Sequence[str]) -> bool:
    """Case-insensitive substring match against a keyword list."""
    haystack = drug.lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def is_dialysis_med(drug: str) -> bool:
    """True iff ``drug`` (case-insensitive) contains any keyword in
    :data:`DIALYSIS_MED_KEYWORDS`."""
    return _drug_matches_keywords(drug, DIALYSIS_MED_KEYWORDS)


def is_chemo_med(drug: str) -> bool:
    """True iff ``drug`` (case-insensitive) contains any keyword in
    :data:`CHEMO_MED_KEYWORDS`."""
    return _drug_matches_keywords(drug, CHEMO_MED_KEYWORDS)


def is_neutropenic(anc: int | None) -> bool:
    """True iff ``anc`` is non-None AND strictly less than
    :data:`ANC_NEUTROPENIA_THRESHOLD`. Missing ANC (``None``) is NOT
    neutropenic — Round 2 N3 requires positive evidence, not absence."""
    if anc is None:
        return False
    return anc < ANC_NEUTROPENIA_THRESHOLD


def _icd10_code_matches_prefix(code: str, prefix: str) -> bool:
    """ICD-10 boundary-safe prefix match.

    Mirrors :func:`bba.audit_orders.rules._code_matches_prefix`. The
    rule is structured around the 3-char category boundary:

    * 1- or 2-char partial-chapter prefixes (``"O"``, ``"C8"``) match
      the prefix followed by digits (``"C8"`` matches ``"C81"`` and
      ``"C83.30"``).
    * 3-char category prefixes (``"D55"``, ``"I50"``, ``"N18"``) match
      the bare code or the code followed by ``"."`` and a subcategory.
      Digit continuation past the 3-char boundary is forbidden:
      ``"D550"`` is NOT ``"D55"`` — it's a different category.
    * Explicit subcategory prefixes (``"N18.5"``) match the bare code
      or further subdivisions following a digit (``"N18.50"``). The
      "3-char boundary" rule does not apply once the prefix has crossed
      the dot.
    """
    if not code.startswith(prefix):
        return False
    if len(code) == len(prefix):
        return True
    next_char = code[len(prefix)]
    if next_char == ".":
        return True
    if not next_char.isdigit():
        return False
    # Digit continuation: forbidden only when crossing the 3-char
    # category boundary (no dot yet AND prefix length already >= 3).
    if "." not in prefix and len(prefix) >= 3:
        return False
    return True


def _first_match(codes: Sequence[str], prefixes: Sequence[str]) -> str | None:
    for code in codes:
        for prefix in prefixes:
            if _icd10_code_matches_prefix(code, prefix):
                return code
    return None


def find_recent_cardiac_surgery(
    events: Sequence[OperativeEvent], anchor: datetime
) -> OperativeEvent | None:
    """Return the most-recent cardiac-surgery event within
    :data:`CARDIAC_SURGERY_LOOKBACK` of ``anchor``, or None.

    ``anchor`` is the audit ``order_datetime`` (tz-aware UTC). Events
    after the anchor are ignored — surgery cannot be retroactively
    pre-anchor.
    """
    cutoff = anchor - CARDIAC_SURGERY_LOOKBACK
    candidates = [
        ev
        for ev in events
        if is_cardiac_surgery_code(ev.icd9, ev.or_flag)
        and cutoff <= ev.operative_datetime <= anchor
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda ev: ev.operative_datetime)


def find_recent_ortho_surgery(
    events: Sequence[OperativeEvent], anchor: datetime
) -> OperativeEvent | None:
    """Mirror of :func:`find_recent_cardiac_surgery` for orthopedic codes.

    Uses the same lookback window. (PRD §5 does not separately tighten
    the ortho window; surgical recovery timelines are similar.)
    """
    cutoff = anchor - CARDIAC_SURGERY_LOOKBACK
    candidates = [
        ev
        for ev in events
        if is_ortho_surgery_code(ev.icd9, ev.or_flag)
        and cutoff <= ev.operative_datetime <= anchor
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda ev: ev.operative_datetime)


def find_cardiac_history_diagnosis(
    diagnosis_codes: Sequence[str],
) -> str | None:
    """Return the first ICD-10 code in ``diagnosis_codes`` matching
    :data:`CARDIAC_HISTORY_ICD10_PREFIXES` under the boundary rules of
    :func:`bba.audit_orders.rules._code_matches_prefix`, or None.
    """
    return _first_match(diagnosis_codes, sorted(CARDIAC_HISTORY_ICD10_PREFIXES))


def find_esrd_diagnosis(diagnosis_codes: Sequence[str]) -> str | None:
    """Return the first ICD-10 code in ``diagnosis_codes`` that is in
    :data:`ESRD_ICD10_CODES`, or None.

    Exact subcategory match (boundary-safe via the same rule used by
    :func:`bba.audit_orders.rules._code_matches_prefix`): ``"N18.5"``
    matches the bare code or ``"N18.5"`` + further subdivisions.
    """
    return _first_match(diagnosis_codes, sorted(ESRD_ICD10_CODES))


def find_dialysis_med(meds: Sequence[MedEvent], anchor: datetime) -> MedEvent | None:
    """Return the most-recent :class:`MedEvent` matching
    :func:`is_dialysis_med` whose timestamp is within
    :data:`DIALYSIS_LOOKBACK` of ``anchor``, or None.

    Stale medication history (e.g., a single sevelamer dose from a
    year-ago admission) does NOT count as "active dialysis"; passing
    the anchor here keeps that scoping inside the rule rather than
    forcing every caller to pre-filter (and risk silent inconsistency).
    """
    cutoff = anchor - DIALYSIS_LOOKBACK
    candidates = [
        med
        for med in meds
        if is_dialysis_med(med.drug) and cutoff <= med.timestamp <= anchor
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda m: m.timestamp)


def find_chemo_med(meds: Sequence[MedEvent], anchor: datetime) -> MedEvent | None:
    """Return the most-recent :class:`MedEvent` matching
    :func:`is_chemo_med` whose timestamp is within :data:`CHEMO_LOOKBACK`
    of ``anchor``, or None.

    The issue body says "recent chemo meds" — the recency check lives
    here so that a long-completed regimen does not falsely trigger
    HEME_MALIGNANCY_ACTIVE for a patient whose disease is now in
    remission.
    """
    cutoff = anchor - CHEMO_LOOKBACK
    candidates = [
        med
        for med in meds
        if is_chemo_med(med.drug) and cutoff <= med.timestamp <= anchor
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda m: m.timestamp)


def find_heme_malignancy_diagnosis(
    diagnosis_codes: Sequence[str],
) -> str | None:
    """Return the first ICD-10 code in ``diagnosis_codes`` matching
    :data:`HEME_MALIGNANCY_ICD10_PREFIXES`, or None."""
    return _first_match(diagnosis_codes, HEME_MALIGNANCY_ICD10_PREFIXES)


def find_cardiopulmonary_comorbidity_diagnosis(
    diagnosis_codes: Sequence[str],
) -> str | None:
    """Return the first ICD-10 code in ``diagnosis_codes`` matching
    :data:`CARDIOPULMONARY_COMORBIDITY_ICD10_PREFIXES`, or None.

    Uses the same boundary-safe prefix rule as the other diagnosis
    predicates, so ``"I25"`` matches ``"I25.10"`` but ``"I250"`` (a
    different category continuation) does not.
    """
    return _first_match(
        diagnosis_codes, sorted(CARDIOPULMONARY_COMORBIDITY_ICD10_PREFIXES)
    )


def detect_mtp_pattern(
    orders: Sequence[BloodOrderEvent], anchor: datetime
) -> BloodOrderEvent | None:
    """Return the order that triggered MTP detection, or None.

    Two arms, either of which fires the rule:

    * Cluster: total ``rbc_units`` across all orders within
      ``[anchor - MTP_TIME_WINDOW, anchor]`` sums to >=
      :data:`MTP_RBC_UNIT_THRESHOLD`.
    * Co-order: the orders within the window collectively show both
      FFP and platelets co-ordered alongside RBC. Either both flags
      on a single order or split across separate orders within the
      same window counts — the underlying pattern is "MTP activation
      in a 1-h window", not "single multi-component order".

    The triggering order returned is the latest order in the window
    (deterministic tiebreak; the caller uses it for the
    ``CohortAssignment.evidence_*`` fields, none of which depend on
    which specific order fired the rule).

    The window is closed-closed: an order at exactly ``anchor - 1h`` is
    included; one strictly outside is not. Orders after the anchor are
    ignored.
    """
    cutoff = anchor - MTP_TIME_WINDOW
    in_window = [order for order in orders if cutoff <= order.timestamp <= anchor]
    if not in_window:
        return None
    has_ffp = any(order.co_ordered_with_ffp for order in in_window)
    has_plt = any(order.co_ordered_with_platelets for order in in_window)
    if has_ffp and has_plt:
        return max(in_window, key=lambda o: o.timestamp)
    total_units = sum(order.rbc_units for order in in_window)
    if total_units >= MTP_RBC_UNIT_THRESHOLD:
        return max(in_window, key=lambda o: o.timestamp)
    return None


__all__: Sequence[str] = (
    "ANC_NEUTROPENIA_THRESHOLD",
    "CARDIAC_HISTORY_ICD10_PREFIXES",
    "CARDIAC_SURGERY_CODE_PREFIXES",
    "CARDIAC_SURGERY_EXCLUDED_CODES",
    "CARDIAC_SURGERY_LOOKBACK",
    "CARDIAC_SURGERY_THRESHOLD",
    "CARDIOPULMONARY_COMORBIDITY_ICD10_PREFIXES",
    "CARDIOPULMONARY_COMORBIDITY_THRESHOLD",
    "CHEMO_MED_KEYWORDS",
    "COHORT_THRESHOLDS",
    "DEFAULT_THRESHOLD",
    "DIALYSIS_MED_KEYWORDS",
    "ESRD_EPO_THRESHOLD",
    "ESRD_ICD10_CODES",
    "HEME_MALIGNANCY_ICD10_PREFIXES",
    "MTP_RBC_UNIT_THRESHOLD",
    "MTP_TIME_WINDOW",
    "ORTHO_CARDIAC_THRESHOLD",
    "ORTHO_SURGERY_CODE_PREFIXES",
    "ORTHO_SURGERY_THRESHOLD",
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
)
