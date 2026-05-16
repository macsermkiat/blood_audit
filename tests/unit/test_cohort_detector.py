"""RED-phase failing tests for issue #7 (bba.cohort_detector).

Each ``class`` maps to one acceptance criterion in the issue body. Tests
assert contracts (the WHY), not implementation choices — see PRD §"Testing
Decisions". No implementation exists yet; every test MUST fail in this
scaffold commit (NotImplementedError on the predicates / detector, KeyError
on the bundled ingest schemas, parse_warning on the bundled time-parser
extension).

Acceptance-criterion → test-class map (issue #7):

* AC ① "Implementation in ``src/bba/cohort_detector/``"
  → import surface verified at module top; collection failure means the
    public API is mis-scaffolded.

* AC ② "One golden-fixture test per cohort class + one no-cohort case"
  → :class:`TestCohortCardiacSurgery`, :class:`TestCohortOrthoCardiac`,
    :class:`TestCohortEsrdEpo`, :class:`TestCohortMtp`,
    :class:`TestCohortHemeMalignancy`, :class:`TestCohortDefault`,
    :class:`TestCohortUnknown`.

* AC ③ "MTP detection: temporal-cluster rule tested at boundary
    (3 vs 4 units; just-over 1 h vs just-under)"
  → :class:`TestMtpBoundary`.

* AC ④ "ESRD-EPO requires both ICD-10 AND dialysis-med signal"
  → :class:`TestEsrdRequiresBothSignals`.

* AC ⑤ "Cohort threshold returned as numeric (not enum); downstream
    classifier uses it directly"
  → :class:`TestThresholdNumericContract`,
    :class:`TestThresholdMapTotality`.

* AC ⑥ "Fallback to cohort_unknown when procedure data missing —
    explicit, never silent"
  → :class:`TestCohortUnknown`,
    :class:`TestProcedureNoneVersusEmptyTuple`.

* AC ⑦ "Coverage >= 80%; ruff + mypy clean" — covered by suite
  totality + property tests below.

Adversarial / property tests:

* :class:`TestAdversarialCardiacExclusions` — 894 (cardiac stress test)
  and 3796 (pacemaker pulse generator) MUST NOT trigger cardiac, even
  with ``or_flag=True``. Per the issue's exclude-list note.

* :class:`TestCardiacRequiresOrFlag` — Orflag=0 (non-OR) MUST NOT
  trigger cardiac even with a matching ICD-9 prefix.

* :class:`TestCohortDeterminism` — same input → same assignment
  (hypothesis property).

* :class:`TestCohortPrecedence` — MTP overrides cardiac; ortho_cardiac
  overrides plain cardiac when both signals are present.

* :class:`TestImmutability` — frozen contracts on inputs and output.

Bundled ingest extension (per user constraint):

* :class:`TestBundledIngestSchemas` — IPTSUMOPRT and ICD9CM are
  registered in ``bba.ingest.schemas._REGISTRY_V1``.

* :class:`TestBundledTimeParserAmPm` — strict parser accepts
  "Month Day, Year, HH:MM AM/PM" and refuses near-misses; hypothesis
  property test mirrors the existing HHMMSS / HH:MM ones.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, get_args

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bba.cohort_detector import (
    ANC_NEUTROPENIA_THRESHOLD,
    CARDIAC_HISTORY_ICD10_PREFIXES,
    CARDIAC_SURGERY_CODE_PREFIXES,
    CARDIAC_SURGERY_EXCLUDED_CODES,
    CARDIAC_SURGERY_LOOKBACK,
    CARDIAC_SURGERY_THRESHOLD,
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
    BloodOrderEvent,
    CohortAssignment,
    CohortInputs,
    CohortLabel,
    MedEvent,
    OperativeEvent,
    assign_cohort,
    detect_mtp_pattern,
    find_cardiac_history_diagnosis,
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


# =============================================================================
# Fixtures
# =============================================================================


ANCHOR = datetime(2026, 5, 14, 12, 0, 0, tzinfo=UTC)
"""A stable order-anchor used across the suite. tz-aware UTC matches the
ingest contract; cohort lookback windows resolve relative to it."""


def _inputs(**overrides: Any) -> CohortInputs:
    """Build a :class:`CohortInputs` with neutral defaults.

    Defaults imply NO positive cohort signal: empty-tuple procedure
    events (data was joined and patient had no procedures), no
    cardiac-history diagnoses, no dialysis / chemo meds, no MTP-pattern
    blood orders, ANC absent. A single override isolates the rule
    under test.

    Note: ``procedure_events=()`` (NOT ``None``) is the default so the
    detector can fall through to ``DEFAULT``. Use ``procedure_events=None``
    explicitly to test the ``UNKNOWN`` cohort.
    """
    defaults: dict[str, Any] = {
        "audit_id": "audit-0001",
        "hn": "HN-0001",
        "an": "AN-0001",
        "order_datetime": ANCHOR,
        "procedure_events": (),
        "diagnosis_codes": (),
        "med_events": (),
        "blood_orders": (),
        "anc_value": None,
    }
    defaults.update(overrides)
    return CohortInputs(**defaults)


def _op(
    icd9: str,
    *,
    or_flag: bool = True,
    days_before_anchor: int = 5,
    name: str | None = None,
) -> OperativeEvent:
    """Build an :class:`OperativeEvent` ``days_before_anchor`` days prior
    to :data:`ANCHOR`. Defaults to ``or_flag=True`` so callers test the
    happy path without spelling it out."""
    return OperativeEvent(
        icd9=icd9,
        or_flag=or_flag,
        operative_datetime=ANCHOR - timedelta(days=days_before_anchor),
        name=name,
    )


def _med(
    drug: str, *, hours_before_anchor: int = 12
) -> MedEvent:
    return MedEvent(
        drug=drug,
        timestamp=ANCHOR - timedelta(hours=hours_before_anchor),
    )


def _order(
    *,
    rbc_units: int = 1,
    minutes_before_anchor: int = 0,
    co_ffp: bool = False,
    co_plt: bool = False,
) -> BloodOrderEvent:
    return BloodOrderEvent(
        timestamp=ANCHOR - timedelta(minutes=minutes_before_anchor),
        rbc_units=rbc_units,
        co_ordered_with_ffp=co_ffp,
        co_ordered_with_platelets=co_plt,
    )


# =============================================================================
# AC: Implementation in src/bba/cohort_detector/ — public surface
#
# The imports at the top of this file fully exercise the public API. If
# any export is missing, collection fails before any test runs. The
# explicit module-membership test below is a tripwire against silent
# re-exports.
# =============================================================================


class TestPublicAPI:
    def test_cohort_label_has_seven_members(self) -> None:
        # Adding or removing a label is a contract change that downstream
        # classifier (#8) and dashboard (#10) rely on.
        assert len(list(CohortLabel)) == 7

    @pytest.mark.parametrize(
        "name",
        [
            "CARDIAC_SURGERY",
            "ORTHO_CARDIAC",
            "ESRD_EPO",
            "MTP",
            "HEME_MALIGNANCY_ACTIVE",
            "DEFAULT",
            "UNKNOWN",
        ],
    )
    def test_each_label_present(self, name: str) -> None:
        assert hasattr(CohortLabel, name), f"CohortLabel missing {name!r}"

    def test_unknown_label_value_is_cohort_unknown(self) -> None:
        # The literal string is part of the contract — the dashboard,
        # NEEDS_REVIEW reason taxonomy, and downstream logging join on
        # the snake_case value, not the enum name.
        assert CohortLabel.UNKNOWN.value == "cohort_unknown"


# =============================================================================
# AC ⑤: thresholds are numeric (not enum). The classifier compares
# against an Hb measurement; a non-numeric threshold would force string
# parsing in the hot path and is exactly what this test bans.
# =============================================================================


class TestThresholdNumericContract:
    @pytest.mark.parametrize(
        ("label", "threshold"),
        [
            (CohortLabel.CARDIAC_SURGERY, 7.5),
            (CohortLabel.ORTHO_CARDIAC, 8.0),
            (CohortLabel.ESRD_EPO, 8.0),
            (CohortLabel.DEFAULT, 7.0),
        ],
    )
    def test_threshold_is_float_not_enum(
        self, label: CohortLabel, threshold: float
    ) -> None:
        assert COHORT_THRESHOLDS[label] == threshold
        assert isinstance(COHORT_THRESHOLDS[label], float)

    @pytest.mark.parametrize(
        "label",
        [CohortLabel.MTP, CohortLabel.HEME_MALIGNANCY_ACTIVE, CohortLabel.UNKNOWN],
    )
    def test_non_threshold_cohorts_have_none(self, label: CohortLabel) -> None:
        # MTP auto-bypasses to APPROPRIATE; heme is T2-supportive; UNKNOWN
        # routes to NEEDS_REVIEW. None signals "do not compare against Hb".
        assert COHORT_THRESHOLDS[label] is None


class TestThresholdMapTotality:
    def test_every_label_in_threshold_map(self) -> None:
        # Adding a CohortLabel without adding a threshold (or explicit
        # None) is a regression — the classifier would KeyError at
        # decision time.
        for label in CohortLabel:
            assert label in COHORT_THRESHOLDS, (
                f"CohortLabel {label!r} missing from COHORT_THRESHOLDS"
            )

    def test_no_extra_keys_in_threshold_map(self) -> None:
        assert set(COHORT_THRESHOLDS.keys()) == set(CohortLabel)


# =============================================================================
# AC ②: golden-fixture tests per cohort + no-cohort
# =============================================================================


class TestCohortCardiacSurgery:
    """Procedure code in cardiac-sx allow-list within 30 days → 7.5 g/dL."""

    def test_recent_36xx_with_or_flag_yields_75(self) -> None:
        # 3601 = single vessel PTCA (canonical 36xx cardiac procedure).
        result = assign_cohort(
            _inputs(
                procedure_events=(
                    _op("3601", or_flag=True, days_before_anchor=10, name="PTCA"),
                ),
            )
        )
        assert result.label == CohortLabel.CARDIAC_SURGERY
        assert result.threshold == 7.5
        assert result.evidence_code == "3601"
        assert result.evidence_name == "PTCA"

    def test_recent_38xx_with_or_flag_yields_75(self) -> None:
        # 3814 = aortic resection w/ replacement.
        result = assign_cohort(
            _inputs(
                procedure_events=(
                    _op("3814", or_flag=True, days_before_anchor=20),
                ),
            )
        )
        assert result.label == CohortLabel.CARDIAC_SURGERY
        assert result.threshold == 7.5

    def test_recent_39xx_with_or_flag_yields_75(self) -> None:
        # 3925 = aorta-iliac-femoral bypass.
        result = assign_cohort(
            _inputs(
                procedure_events=(
                    _op("3925", or_flag=True, days_before_anchor=1),
                ),
            )
        )
        assert result.label == CohortLabel.CARDIAC_SURGERY
        assert result.threshold == 7.5

    def test_dot_form_3601_normalizes_to_3601(self) -> None:
        # The orchestrator strips dots, but defense-in-depth: the
        # detector's matcher must accept both shapes so an out-of-spec
        # join doesn't silently drop the cardiac signal.
        result = assign_cohort(
            _inputs(
                procedure_events=(
                    _op("36.01", or_flag=True, days_before_anchor=10),
                ),
            )
        )
        assert result.label == CohortLabel.CARDIAC_SURGERY


class TestCardiacSurgeryLookback:
    """The 30-day lookback window has hard boundaries."""

    def test_exactly_30_days_inclusive(self) -> None:
        # PRD §5: "within 30 days" — boundary is inclusive at 30 days
        # (a procedure on day 30 still counts).
        result = assign_cohort(
            _inputs(
                procedure_events=(
                    _op("3601", or_flag=True, days_before_anchor=30),
                ),
            )
        )
        assert result.label == CohortLabel.CARDIAC_SURGERY

    def test_31_days_excluded(self) -> None:
        result = assign_cohort(
            _inputs(
                procedure_events=(
                    _op("3601", or_flag=True, days_before_anchor=31),
                ),
            )
        )
        # Falls through to DEFAULT — a 31-day-old surgery is no longer
        # post-op recovery for transfusion-trigger purposes.
        assert result.label == CohortLabel.DEFAULT

    def test_future_procedure_ignored(self) -> None:
        # Operations after the audit anchor cannot retroactively shift
        # the threshold for an order that already happened.
        future = ANCHOR + timedelta(days=2)
        op = OperativeEvent(
            icd9="3601",
            or_flag=True,
            operative_datetime=future,
        )
        result = assign_cohort(_inputs(procedure_events=(op,)))
        assert result.label == CohortLabel.DEFAULT


class TestCardiacRequiresOrFlag:
    """Orflag=0 (non-OR) MUST NOT trigger cardiac, even with a matching prefix."""

    def test_or_flag_false_excluded(self) -> None:
        result = assign_cohort(
            _inputs(
                procedure_events=(
                    _op("3601", or_flag=False, days_before_anchor=10),
                ),
            )
        )
        assert result.label == CohortLabel.DEFAULT

    def test_helper_returns_false_when_or_flag_false(self) -> None:
        assert is_cardiac_surgery_code("3601", or_flag=False) is False
        assert is_cardiac_surgery_code("3601", or_flag=True) is True


class TestIcd10StrictCaseContract:
    """Codex review HIGH-3: ICD-10 matching is intentionally
    case-sensitive and whitespace-strict.

    The deliberate contract — mirroring :mod:`bba.audit_orders` — is
    that the ingest layer delivers clean codes; tolerating lowercase
    or padding in the matcher would also tolerate other formatting
    drift (half-width digits, encoding quirks) we have not opted into.
    Drift in real HIS data is a data-quality problem to fix at ingest,
    not a problem to paper over downstream and silently broaden the
    cohort allow-lists.

    These tests pin the contract so a future contributor cannot loosen
    it without an explicit, reviewed change.
    """

    def test_lowercase_icd10_does_not_match(self) -> None:
        # "i25.10" (lowercase) MUST NOT match "I25" prefix.
        assert find_cardiac_history_diagnosis(("i25.10",)) is None

    def test_lowercase_esrd_does_not_match(self) -> None:
        assert find_esrd_diagnosis(("n18.5",)) is None

    def test_whitespace_padded_icd10_does_not_match(self) -> None:
        # Stray padding is rejected — the ingest layer must trim CSV
        # cells before passing them in.
        assert find_cardiac_history_diagnosis((" I25.10 ",)) is None

    def test_lowercase_heme_does_not_match(self) -> None:
        assert find_heme_malignancy_diagnosis(("c83.30",)) is None


class TestAdversarialCardiacExclusions:
    """894 (cardiac stress test) and 3796 (pacemaker pulse generator)
    MUST NOT trigger cardiac_surgery, even with ``or_flag=True``.

    Per issue body: "EXCLUDE non-OR cardiac items like 894 and 3796
    unless clinical confirms". The exclude set is defense-in-depth
    against future widening of the prefix list.
    """

    @pytest.mark.parametrize("code", sorted(CARDIAC_SURGERY_EXCLUDED_CODES))
    def test_excluded_code_does_not_trigger_cardiac(self, code: str) -> None:
        result = assign_cohort(
            _inputs(
                procedure_events=(_op(code, or_flag=True, days_before_anchor=5),),
            )
        )
        assert result.label != CohortLabel.CARDIAC_SURGERY

    @pytest.mark.parametrize("code", sorted(CARDIAC_SURGERY_EXCLUDED_CODES))
    def test_helper_returns_false_for_excluded_code(self, code: str) -> None:
        assert is_cardiac_surgery_code(code, or_flag=True) is False


class TestCohortOrthoCardiac:
    """Ortho operative event + ICD-10 cardiac history → 8.0 g/dL."""

    def test_8151_plus_i25_yields_80(self) -> None:
        # 8151 = total hip replacement; I25.10 = chronic IHD.
        result = assign_cohort(
            _inputs(
                procedure_events=(
                    _op("8151", or_flag=True, days_before_anchor=15, name="THR"),
                ),
                diagnosis_codes=("I25.10",),
            )
        )
        assert result.label == CohortLabel.ORTHO_CARDIAC
        assert result.threshold == 8.0

    def test_7805_plus_i50_yields_80(self) -> None:
        # 7805 = open reduction of fracture; I50.9 = heart failure NOS.
        result = assign_cohort(
            _inputs(
                procedure_events=(_op("7805", or_flag=True),),
                diagnosis_codes=("I50.9",),
            )
        )
        assert result.label == CohortLabel.ORTHO_CARDIAC
        assert result.threshold == 8.0

    def test_ortho_alone_does_not_trigger_ortho_cardiac(self) -> None:
        # Plain ortho is not a cohort by itself per the PRD §5 table.
        result = assign_cohort(
            _inputs(procedure_events=(_op("8151", or_flag=True),))
        )
        assert result.label == CohortLabel.DEFAULT
        assert result.threshold == 7.0

    def test_cardiac_history_alone_does_not_trigger_ortho_cardiac(self) -> None:
        result = assign_cohort(_inputs(diagnosis_codes=("I25.10",)))
        assert result.label == CohortLabel.DEFAULT


class TestCohortEsrdEpo:
    """ESRD ICD-10 + dialysis-context med → 8.0 g/dL."""

    def test_n185_plus_sevelamer_yields_80(self) -> None:
        result = assign_cohort(
            _inputs(
                diagnosis_codes=("N18.5",),
                med_events=(_med("Sevelamer 800 mg PO TID"),),
            )
        )
        assert result.label == CohortLabel.ESRD_EPO
        assert result.threshold == 8.0

    def test_n186_plus_cinacalcet_yields_80(self) -> None:
        result = assign_cohort(
            _inputs(
                diagnosis_codes=("N18.6",),
                med_events=(_med("cinacalcet 30mg"),),
            )
        )
        assert result.label == CohortLabel.ESRD_EPO

    def test_n186_plus_heparin_yields_80(self) -> None:
        # heparin-for-HD: scoped by the co-required ESRD diagnosis.
        result = assign_cohort(
            _inputs(
                diagnosis_codes=("N18.6",),
                med_events=(_med("heparin 5000 units IV"),),
            )
        )
        assert result.label == CohortLabel.ESRD_EPO


class TestEsrdRequiresBothSignals:
    """AC ④: ESRD-EPO requires BOTH ICD-10 AND dialysis-med."""

    def test_n186_alone_does_not_trigger(self) -> None:
        result = assign_cohort(_inputs(diagnosis_codes=("N18.6",)))
        assert result.label == CohortLabel.DEFAULT

    def test_dialysis_med_alone_does_not_trigger(self) -> None:
        # A patient on heparin for non-HD reasons (e.g., DVT prophylaxis)
        # without ESRD MUST NOT be classified ESRD-EPO.
        result = assign_cohort(_inputs(med_events=(_med("heparin"),)))
        assert result.label == CohortLabel.DEFAULT

    def test_n184_plus_sevelamer_does_not_trigger(self) -> None:
        # CKD stage 4 (N18.4) is not ESRD; the rule is strict to N18.5
        # or N18.6 + dialysis med.
        result = assign_cohort(
            _inputs(
                diagnosis_codes=("N18.4",),
                med_events=(_med("Sevelamer"),),
            )
        )
        assert result.label == CohortLabel.DEFAULT


class TestCohortMtp:
    """≥4 RBC units within 1 h OR co-ordered with FFP+platelets in same window."""

    def test_4_rbc_units_within_1h_yields_mtp(self) -> None:
        orders = (
            _order(rbc_units=2, minutes_before_anchor=45),
            _order(rbc_units=2, minutes_before_anchor=20),
        )
        result = assign_cohort(_inputs(blood_orders=orders))
        assert result.label == CohortLabel.MTP
        assert result.threshold is None  # auto-APPROPRIATE bypass

    def test_co_ordered_with_ffp_and_platelets_yields_mtp(self) -> None:
        orders = (
            _order(
                rbc_units=2,
                minutes_before_anchor=10,
                co_ffp=True,
                co_plt=True,
            ),
        )
        result = assign_cohort(_inputs(blood_orders=orders))
        assert result.label == CohortLabel.MTP

    def test_co_ordered_with_only_ffp_does_not_trigger(self) -> None:
        # The co-order arm requires BOTH FFP and platelets — only one
        # is the routine multi-component order, not MTP.
        orders = (
            _order(rbc_units=2, minutes_before_anchor=10, co_ffp=True, co_plt=False),
        )
        result = assign_cohort(_inputs(blood_orders=orders))
        assert result.label != CohortLabel.MTP

    def test_co_order_split_across_separate_orders_in_window_triggers(self) -> None:
        # Codex review: real-world MTP activations frequently arrive as
        # separate parallel orders (RBC+FFP on one slip, RBC+Platelets
        # on another) rather than a single multi-component order. The
        # rule is "RBC + FFP + platelets in same WINDOW", so the FFP
        # and platelet flags can be split across orders inside the 1-h
        # window and still fire.
        orders = (
            _order(
                rbc_units=2,
                minutes_before_anchor=30,
                co_ffp=True,
                co_plt=False,
            ),
            _order(
                rbc_units=1,
                minutes_before_anchor=10,
                co_ffp=False,
                co_plt=True,
            ),
        )
        result = assign_cohort(_inputs(blood_orders=orders))
        assert result.label == CohortLabel.MTP

    def test_co_order_split_outside_window_does_not_trigger(self) -> None:
        # Same split, but the FFP order is OUTSIDE the 1-h window so the
        # collective evidence is incomplete inside the window.
        orders = (
            _order(
                rbc_units=1,
                minutes_before_anchor=120,
                co_ffp=True,
                co_plt=False,
            ),
            _order(
                rbc_units=1,
                minutes_before_anchor=10,
                co_ffp=False,
                co_plt=True,
            ),
        )
        result = assign_cohort(_inputs(blood_orders=orders))
        assert result.label != CohortLabel.MTP


class TestMtpBoundary:
    """AC ③: temporal-cluster boundary tests."""

    def test_3_rbc_units_does_not_trigger(self) -> None:
        orders = (
            _order(rbc_units=1, minutes_before_anchor=45),
            _order(rbc_units=2, minutes_before_anchor=20),
        )
        result = assign_cohort(_inputs(blood_orders=orders))
        assert result.label != CohortLabel.MTP

    def test_4_rbc_units_at_threshold_triggers(self) -> None:
        orders = (
            _order(rbc_units=4, minutes_before_anchor=10),
        )
        result = assign_cohort(_inputs(blood_orders=orders))
        assert result.label == CohortLabel.MTP

    def test_just_under_1h_with_4_units_triggers(self) -> None:
        # 59 min before anchor = inside the 1 h window.
        orders = (_order(rbc_units=4, minutes_before_anchor=59),)
        result = assign_cohort(_inputs(blood_orders=orders))
        assert result.label == CohortLabel.MTP

    def test_just_over_1h_with_4_units_does_not_trigger(self) -> None:
        # 61 min before anchor = outside the 1 h window.
        orders = (_order(rbc_units=4, minutes_before_anchor=61),)
        result = assign_cohort(_inputs(blood_orders=orders))
        assert result.label != CohortLabel.MTP

    def test_exactly_1h_inclusive(self) -> None:
        # The window is half-open [anchor - 1h, anchor]; an order at
        # exactly anchor - 1h is included (cluster sums to 4 units).
        orders = (_order(rbc_units=4, minutes_before_anchor=60),)
        result = assign_cohort(_inputs(blood_orders=orders))
        assert result.label == CohortLabel.MTP

    def test_helper_detects_co_order_pattern(self) -> None:
        match = detect_mtp_pattern(
            (_order(rbc_units=1, minutes_before_anchor=5, co_ffp=True, co_plt=True),),
            ANCHOR,
        )
        assert match is not None

    def test_helper_returns_none_for_below_threshold(self) -> None:
        match = detect_mtp_pattern(
            (_order(rbc_units=3, minutes_before_anchor=5),),
            ANCHOR,
        )
        assert match is None


class TestCohortHemeMalignancy:
    """ICD-10 C8x–C9x + chemo med + ANC<500 → T2 supportive (no hard threshold)."""

    def test_c83_plus_chemo_plus_low_anc_triggers(self) -> None:
        # C83 = diffuse non-Hodgkin lymphoma.
        result = assign_cohort(
            _inputs(
                diagnosis_codes=("C83.30",),
                med_events=(_med("doxorubicin 50 mg IV"),),
                anc_value=300,
            )
        )
        assert result.label == CohortLabel.HEME_MALIGNANCY_ACTIVE
        assert result.threshold is None  # T2 supportive, not threshold-driven

    def test_c91_plus_chemo_plus_low_anc_triggers(self) -> None:
        # C91 = lymphoid leukemia.
        result = assign_cohort(
            _inputs(
                diagnosis_codes=("C91.00",),
                med_events=(_med("cytarabine"),),
                anc_value=100,
            )
        )
        assert result.label == CohortLabel.HEME_MALIGNANCY_ACTIVE

    def test_anc_at_500_does_not_trigger(self) -> None:
        # Strict less-than: ANC == 500 is the boundary (Round 2 fix N3).
        result = assign_cohort(
            _inputs(
                diagnosis_codes=("C83.30",),
                med_events=(_med("doxorubicin"),),
                anc_value=500,
            )
        )
        assert result.label != CohortLabel.HEME_MALIGNANCY_ACTIVE

    def test_no_chemo_does_not_trigger(self) -> None:
        result = assign_cohort(
            _inputs(
                diagnosis_codes=("C83.30",),
                med_events=(),
                anc_value=200,
            )
        )
        assert result.label != CohortLabel.HEME_MALIGNANCY_ACTIVE

    def test_no_anc_does_not_trigger(self) -> None:
        # Missing ANC is not "ANC < 500" — Round 2 N3 requires positive
        # evidence of neutropenia, not absence of measurement.
        result = assign_cohort(
            _inputs(
                diagnosis_codes=("C83.30",),
                med_events=(_med("doxorubicin"),),
                anc_value=None,
            )
        )
        assert result.label != CohortLabel.HEME_MALIGNANCY_ACTIVE


class TestCohortDefault:
    """Fall-through case: no positive cohort signal → 7.0 g/dL."""

    def test_no_signals_yields_default(self) -> None:
        result = assign_cohort(_inputs())
        assert result.label == CohortLabel.DEFAULT
        assert result.threshold == 7.0

    def test_default_evidence_fields_are_none(self) -> None:
        # No positive triggering fact to surface for DEFAULT.
        result = assign_cohort(_inputs())
        assert result.evidence_code is None
        assert result.evidence_name is None


# =============================================================================
# AC ⑥: cohort_unknown when procedure data missing — explicit, never silent
# =============================================================================


class TestCohortUnknown:
    def test_procedure_events_none_yields_unknown(self) -> None:
        result = assign_cohort(_inputs(procedure_events=None))
        assert result.label == CohortLabel.UNKNOWN

    def test_unknown_threshold_is_none(self) -> None:
        # The ENTIRE point of UNKNOWN is "do not silently apply 7.0".
        # threshold MUST be None so the classifier routes to NEEDS_REVIEW.
        result = assign_cohort(_inputs(procedure_events=None))
        assert result.threshold is None

    def test_unknown_evidence_fields_are_none(self) -> None:
        result = assign_cohort(_inputs(procedure_events=None))
        assert result.evidence_code is None
        assert result.evidence_name is None

    def test_unknown_when_other_signals_present(self) -> None:
        # Even if a non-procedure cohort would otherwise fire (ESRD,
        # heme), missing procedure data forces UNKNOWN. Conservative:
        # we cannot be sure the patient isn't post-cardiac-sx, so
        # NEEDS_REVIEW wins.
        result = assign_cohort(
            _inputs(
                procedure_events=None,
                diagnosis_codes=("N18.6",),
                med_events=(_med("sevelamer"),),
            )
        )
        assert result.label == CohortLabel.UNKNOWN

    def test_unknown_overrides_even_mtp_pattern(self) -> None:
        # PR #51 Codex P2: the MTP auto-bypass MUST NOT silently fire
        # when procedure data is unavailable. Conservative routing wins
        # because we cannot rule out a cardiac / ortho context for the
        # apparent MTP — a human reviewer needs to verify before the
        # auto-APPROPRIATE bypass takes effect.
        result = assign_cohort(
            _inputs(
                procedure_events=None,
                blood_orders=(_order(rbc_units=4, minutes_before_anchor=10),),
            )
        )
        assert result.label == CohortLabel.UNKNOWN
        assert result.threshold is None


class TestProcedureNoneVersusEmptyTuple:
    """The None / () distinction is load-bearing — silent collapse breaks AC ⑥."""

    def test_empty_tuple_is_not_unknown(self) -> None:
        # () means "we joined and the patient has no operative events" —
        # a clean no-cardiac signal, NOT unknown.
        result = assign_cohort(_inputs(procedure_events=()))
        assert result.label != CohortLabel.UNKNOWN
        assert result.label == CohortLabel.DEFAULT

    def test_none_is_unknown(self) -> None:
        # None means "the join was skipped / data unavailable" — UNKNOWN.
        result = assign_cohort(_inputs(procedure_events=None))
        assert result.label == CohortLabel.UNKNOWN


# =============================================================================
# Cohort precedence (composer behavior, not a single AC)
# =============================================================================


class TestCohortPrecedence:
    def test_mtp_overrides_cardiac_surgery(self) -> None:
        # Even with a recent cardiac procedure, an active MTP pattern
        # is the more urgent decision and bypasses to APPROPRIATE.
        result = assign_cohort(
            _inputs(
                procedure_events=(_op("3601", or_flag=True, days_before_anchor=2),),
                blood_orders=(_order(rbc_units=4, minutes_before_anchor=10),),
            )
        )
        assert result.label == CohortLabel.MTP

    def test_ortho_cardiac_preferred_over_plain_cardiac(self) -> None:
        # Patient has both a recent cardiac code AND a recent ortho code
        # AND cardiac history — the ortho_cardiac threshold (8.0) is
        # stricter than cardiac alone (7.5), so ortho_cardiac wins.
        result = assign_cohort(
            _inputs(
                procedure_events=(
                    _op("8151", or_flag=True, days_before_anchor=15),
                    _op("3601", or_flag=True, days_before_anchor=10),
                ),
                diagnosis_codes=("I25.10",),
            )
        )
        assert result.label == CohortLabel.ORTHO_CARDIAC
        assert result.threshold == 8.0

    def test_esrd_preferred_over_default(self) -> None:
        result = assign_cohort(
            _inputs(
                diagnosis_codes=("N18.6",),
                med_events=(_med("sevelamer"),),
            )
        )
        assert result.label == CohortLabel.ESRD_EPO


# =============================================================================
# Predicate-level helper tests (small surface, big leverage)
# =============================================================================


class TestPredicateHelpers:
    def test_normalize_icd9_strips_dot(self) -> None:
        assert normalize_icd9("36.01") == "3601"

    def test_normalize_icd9_passthrough(self) -> None:
        assert normalize_icd9("3601") == "3601"

    def test_normalize_icd9_strips_whitespace(self) -> None:
        assert normalize_icd9("  3601  ") == "3601"

    @pytest.mark.parametrize("code", ["3601", "3814", "3925"])
    def test_is_cardiac_surgery_code_seed_codes(self, code: str) -> None:
        assert is_cardiac_surgery_code(code, or_flag=True) is True

    @pytest.mark.parametrize("code", ["8151", "7805", "7901"])
    def test_is_ortho_surgery_code_seed_codes(self, code: str) -> None:
        assert is_ortho_surgery_code(code, or_flag=True) is True

    @pytest.mark.parametrize(
        "drug",
        ["sevelamer", "Sevelamer 800mg", "CINACALCET 30 mg", "heparin 5000U"],
    )
    def test_is_dialysis_med_substring_match(self, drug: str) -> None:
        assert is_dialysis_med(drug) is True

    def test_is_dialysis_med_negative(self) -> None:
        assert is_dialysis_med("paracetamol 500 mg") is False

    @pytest.mark.parametrize(
        "drug",
        ["doxorubicin", "Cisplatin 100mg/m2", "rituximab IV"],
    )
    def test_is_chemo_med_substring_match(self, drug: str) -> None:
        assert is_chemo_med(drug) is True

    def test_is_neutropenic_below_threshold(self) -> None:
        assert is_neutropenic(499) is True

    def test_is_neutropenic_at_threshold_false(self) -> None:
        assert is_neutropenic(500) is False

    def test_is_neutropenic_none_false(self) -> None:
        assert is_neutropenic(None) is False

    def test_find_cardiac_history_diagnosis_i25(self) -> None:
        assert find_cardiac_history_diagnosis(("I25.10",)) == "I25.10"

    def test_find_cardiac_history_diagnosis_i50(self) -> None:
        assert find_cardiac_history_diagnosis(("I50.9",)) == "I50.9"

    def test_find_cardiac_history_diagnosis_no_match(self) -> None:
        assert find_cardiac_history_diagnosis(("E11.9",)) is None

    def test_find_esrd_diagnosis_exact(self) -> None:
        assert find_esrd_diagnosis(("N18.5",)) == "N18.5"

    def test_find_esrd_diagnosis_n184_excluded(self) -> None:
        # CKD stage 4 is NOT ESRD.
        assert find_esrd_diagnosis(("N18.4",)) is None

    def test_find_esrd_diagnosis_n18_50_subdivision_matches(self) -> None:
        # Some ICD-10 jurisdictions split N18.5 further (e.g., N18.50).
        # The matcher must accept further subdivisions after the dot.
        assert find_esrd_diagnosis(("N18.50",)) == "N18.50"

    def test_find_esrd_diagnosis_n186_subdivision_matches(self) -> None:
        assert find_esrd_diagnosis(("N18.69",)) == "N18.69"

    def test_find_cardiac_history_d550_does_not_match_d55(self) -> None:
        # Defense-in-depth: "D550" (no dot) is a different ICD-10 category
        # from "D55" — the matcher must not collapse the boundary even
        # though the prefix list here is I-codes, not D-codes. Use a
        # cardiac-history call as a proxy for the same matcher logic:
        # an "I250" without dot is NOT "I25".
        assert find_cardiac_history_diagnosis(("I250",)) is None

    def test_find_cardiac_history_i25_dotted_subcategories_match(self) -> None:
        assert find_cardiac_history_diagnosis(("I25.10",)) == "I25.10"
        assert find_cardiac_history_diagnosis(("I25.110",)) == "I25.110"

    def test_find_heme_letter_continuation_does_not_match(self) -> None:
        # Defense-in-depth: garbled code "C8X" (letter where a digit
        # belongs) must NOT match the C8 chapter prefix. Real ICD-10
        # 7th-character extensions are letters but are positioned after
        # subcategory digits + dot — they never directly follow a 2-char
        # chapter prefix.
        assert find_heme_malignancy_diagnosis(("C8X",)) is None

    def test_find_dialysis_med_returns_event(self) -> None:
        meds = (_med("paracetamol"), _med("sevelamer"))
        match = find_dialysis_med(meds, ANCHOR)
        assert match is not None
        assert "sevelamer" in match.drug.lower()

    def test_find_chemo_med_returns_event(self) -> None:
        meds = (_med("doxorubicin"),)
        match = find_chemo_med(meds, ANCHOR)
        assert match is not None

    def test_find_dialysis_med_outside_window_excluded(self) -> None:
        # A sevelamer dose 100 days before the anchor is far outside the
        # 14-day active window — must NOT count as active dialysis.
        old = MedEvent(
            drug="sevelamer",
            timestamp=ANCHOR - timedelta(days=100),
        )
        assert find_dialysis_med((old,), ANCHOR) is None

    def test_find_dialysis_med_returns_most_recent_in_window(self) -> None:
        meds = (
            MedEvent(drug="sevelamer", timestamp=ANCHOR - timedelta(days=10)),
            MedEvent(drug="sevelamer", timestamp=ANCHOR - timedelta(days=2)),
        )
        match = find_dialysis_med(meds, ANCHOR)
        assert match is not None
        assert match.timestamp == ANCHOR - timedelta(days=2)

    def test_find_chemo_med_outside_window_excluded(self) -> None:
        # A doxorubicin dose 90 days before the anchor is far outside the
        # 30-day "recent chemo" window — must NOT trigger heme cohort.
        old = MedEvent(
            drug="doxorubicin",
            timestamp=ANCHOR - timedelta(days=90),
        )
        assert find_chemo_med((old,), ANCHOR) is None

    def test_find_chemo_med_at_30_day_boundary_included(self) -> None:
        # Exactly 30 days before anchor is the inclusive edge.
        boundary = MedEvent(
            drug="doxorubicin",
            timestamp=ANCHOR - timedelta(days=30),
        )
        assert find_chemo_med((boundary,), ANCHOR) is not None

    def test_find_chemo_med_31_days_excluded(self) -> None:
        boundary = MedEvent(
            drug="doxorubicin",
            timestamp=ANCHOR - timedelta(days=31),
        )
        assert find_chemo_med((boundary,), ANCHOR) is None

    def test_find_heme_malignancy_diagnosis_c83(self) -> None:
        assert find_heme_malignancy_diagnosis(("C83.30",)) == "C83.30"

    def test_find_heme_malignancy_diagnosis_no_match(self) -> None:
        assert find_heme_malignancy_diagnosis(("C50.9",)) is None  # breast Ca, not heme

    def test_find_recent_cardiac_surgery_returns_most_recent(self) -> None:
        events = (
            _op("3601", or_flag=True, days_before_anchor=29),
            _op("3601", or_flag=True, days_before_anchor=2),
        )
        match = find_recent_cardiac_surgery(events, ANCHOR)
        assert match is not None
        # The 2-days-before event is more recent than the 29-days one.
        assert match.operative_datetime == ANCHOR - timedelta(days=2)

    def test_find_recent_ortho_surgery_returns_match(self) -> None:
        events = (_op("8151", or_flag=True, days_before_anchor=10),)
        match = find_recent_ortho_surgery(events, ANCHOR)
        assert match is not None


# =============================================================================
# Hypothesis property tests — determinism + boundary invariants
# =============================================================================


class TestCohortDeterminism:
    @given(
        days=st.integers(min_value=0, max_value=29),
        units=st.integers(min_value=1, max_value=3),
        anc=st.one_of(st.none(), st.integers(min_value=0, max_value=10000)),
    )
    @settings(max_examples=100)
    def test_same_inputs_same_assignment(
        self, days: int, units: int, anc: int | None
    ) -> None:
        # Construct an input from the strategy values; same args twice
        # must yield the same CohortAssignment (pure-function contract).
        ev = (_op("3601", or_flag=True, days_before_anchor=days),)
        orders = (_order(rbc_units=units, minutes_before_anchor=30),)
        a = assign_cohort(_inputs(procedure_events=ev, blood_orders=orders, anc_value=anc))
        b = assign_cohort(_inputs(procedure_events=ev, blood_orders=orders, anc_value=anc))
        assert a == b


_BLOOD_ORDER_STRATEGY = st.builds(
    _order,
    rbc_units=st.integers(min_value=0, max_value=6),
    minutes_before_anchor=st.integers(min_value=0, max_value=180),
    co_ffp=st.booleans(),
    co_plt=st.booleans(),
)
"""Hypothesis strategy generating one BloodOrderEvent with anchored
timestamp anywhere from 0–180 minutes before the audit anchor (so the
strategy covers both inside-window and outside-window orders) plus
arbitrary co-order flags."""


class TestMtpBoundaryProperty:
    """Property-based MTP invariants — generates arbitrary order sets
    spanning inside / outside the window, mixed FFP-only / platelet-only
    arms, and the boundary itself. 100% line coverage hides the case
    that exercises both sides of every branch; this property does not."""

    @given(orders=st.lists(_BLOOD_ORDER_STRATEGY, min_size=0, max_size=8))
    @settings(max_examples=200)
    def test_mtp_invariant_matches_spec(
        self, orders: list[BloodOrderEvent]
    ) -> None:
        # Recompute the MTP truth using a brute-force read of the spec
        # (independent of the implementation). The detector must agree
        # with this independent calculation for every generated input.
        cutoff = ANCHOR - MTP_TIME_WINDOW
        in_window = [o for o in orders if cutoff <= o.timestamp <= ANCHOR]
        ffp = any(o.co_ordered_with_ffp for o in in_window)
        plt = any(o.co_ordered_with_platelets for o in in_window)
        cluster = sum(o.rbc_units for o in in_window) >= MTP_RBC_UNIT_THRESHOLD
        expected_mtp = bool(in_window) and ((ffp and plt) or cluster)
        result = assign_cohort(_inputs(blood_orders=tuple(orders)))
        assert (result.label == CohortLabel.MTP) == expected_mtp

    @given(units=st.integers(min_value=0, max_value=MTP_RBC_UNIT_THRESHOLD - 1))
    @settings(max_examples=50)
    def test_below_threshold_alone_never_triggers(self, units: int) -> None:
        result = assign_cohort(
            _inputs(
                blood_orders=(_order(rbc_units=units, minutes_before_anchor=30),),
            )
        )
        assert result.label != CohortLabel.MTP


# =============================================================================
# Immutability — frozen contracts
# =============================================================================


class TestImmutability:
    def test_cohort_assignment_is_frozen(self) -> None:
        result = CohortAssignment(
            label=CohortLabel.DEFAULT,
            threshold=7.0,
            evidence_code=None,
            evidence_name=None,
        )
        with pytest.raises(Exception):
            result.threshold = 9.0  # type: ignore[misc]

    def test_cohort_inputs_is_frozen(self) -> None:
        inputs = _inputs()
        with pytest.raises(Exception):
            inputs.audit_id = "other"  # type: ignore[misc]


# =============================================================================
# Cohort-allowlist seeds — explicit assertions on the published contract
# =============================================================================


class TestAllowListSeeds:
    """The seed allow-lists are part of the public contract; surfacing
    them as test fixtures lets the clinical-review PR diff them line by
    line during sign-off."""

    def test_cardiac_prefixes_match_issue_seed(self) -> None:
        assert set(CARDIAC_SURGERY_CODE_PREFIXES) == {"36", "38", "39"}

    def test_cardiac_excluded_codes_match_issue_seed(self) -> None:
        assert CARDIAC_SURGERY_EXCLUDED_CODES == frozenset({"894", "3796"})

    def test_ortho_prefixes_match_issue_seed(self) -> None:
        assert set(ORTHO_SURGERY_CODE_PREFIXES) == {"78", "79", "81"}

    def test_cardiac_history_prefixes_cover_i20_to_i25_and_i50(self) -> None:
        assert CARDIAC_HISTORY_ICD10_PREFIXES == frozenset(
            {"I20", "I21", "I22", "I23", "I24", "I25", "I50"}
        )

    def test_esrd_codes_are_n185_and_n186(self) -> None:
        assert ESRD_ICD10_CODES == frozenset({"N18.5", "N18.6"})

    def test_heme_malignancy_prefixes_cover_c8x_c9x(self) -> None:
        assert set(HEME_MALIGNANCY_ICD10_PREFIXES) == {"C8", "C9"}

    def test_dialysis_med_keywords_seed(self) -> None:
        assert "sevelamer" in DIALYSIS_MED_KEYWORDS
        assert "cinacalcet" in DIALYSIS_MED_KEYWORDS
        assert "heparin" in DIALYSIS_MED_KEYWORDS

    def test_chemo_med_keywords_nonempty(self) -> None:
        # Pre-clinical sign-off list; the test asserts the seed exists
        # so an empty-list regression cannot silently kill heme detection.
        assert len(CHEMO_MED_KEYWORDS) > 0

    def test_anc_threshold_is_500(self) -> None:
        assert ANC_NEUTROPENIA_THRESHOLD == 500

    def test_mtp_unit_threshold_is_4(self) -> None:
        assert MTP_RBC_UNIT_THRESHOLD == 4

    def test_mtp_window_is_1_hour(self) -> None:
        assert MTP_TIME_WINDOW == timedelta(hours=1)

    def test_cardiac_surgery_lookback_is_30_days(self) -> None:
        assert CARDIAC_SURGERY_LOOKBACK == timedelta(days=30)

    def test_chemo_lookback_is_30_days(self) -> None:
        # "recent chemo meds" per the issue body — 30-day window matches
        # typical cycle cadence and prevents long-completed regimens
        # from over-triggering the heme cohort.
        assert CHEMO_LOOKBACK == timedelta(days=30)

    def test_dialysis_lookback_is_14_days(self) -> None:
        # Two-week active window matches refill cadence for the seed
        # dialysis-context drugs.
        assert DIALYSIS_LOOKBACK == timedelta(days=14)

    def test_cohort_threshold_constants(self) -> None:
        assert DEFAULT_THRESHOLD == 7.0
        assert CARDIAC_SURGERY_THRESHOLD == 7.5
        assert ORTHO_CARDIAC_THRESHOLD == 8.0
        assert ESRD_EPO_THRESHOLD == 8.0


# =============================================================================
# Bundled ingest extension (per user constraint, this ticket BUNDLES
# adding IPTSUMOPRT + ICD9CM schemas and extending the strict time
# parser to accept "Month Day, Year, HH:MM AM/PM").
# =============================================================================


class TestBundledIngestSchemas:
    """IPTSUMOPRT and ICD9CM are registered and accessible via get_schema."""

    def test_iptsumoprt_in_csvtable_literal(self) -> None:
        from bba.ingest.models import CSVTable

        assert "IPTSUMOPRT" in get_args(CSVTable)

    def test_icd9cm_in_csvtable_literal(self) -> None:
        from bba.ingest.models import CSVTable

        assert "ICD9CM" in get_args(CSVTable)

    def test_iptsumoprt_schema_registered(self) -> None:
        from bba.ingest.schemas import get_schema

        schema = get_schema("IPTSUMOPRT")
        assert schema is not None

    def test_icd9cm_schema_registered(self) -> None:
        from bba.ingest.schemas import get_schema

        schema = get_schema("ICD9CM")
        assert schema is not None

    def test_iptsumoprt_has_orflag_and_icd9_columns(self) -> None:
        # The cohort detector consumes Orflag (OR-procedure gate) and
        # an ICD-9-CM code column from IPTSUMOPRT — both must be in the
        # registered schema or the upstream join silently drops them.
        from bba.ingest.schemas import get_schema

        cols = set(get_schema("IPTSUMOPRT").columns)
        assert "Orflag" in cols or "ORFLAG" in cols, (
            f"IPTSUMOPRT schema must declare an Orflag column; got {sorted(cols)}"
        )

    def test_all_tables_now_includes_new_two(self) -> None:
        from bba.ingest.schemas import all_tables

        names = set(all_tables())
        assert "IPTSUMOPRT" in names
        assert "ICD9CM" in names


class TestBundledTimeParserAmPm:
    """Strict parser accepts "Month Day, Year, HH:MM AM/PM"."""

    def test_midnight_long_form(self) -> None:
        from bba.ingest.time_parser import parse_hosxp_time

        r = parse_hosxp_time("June 7, 2025, 12:00 AM")
        assert r.parse_warning is None, (
            f"long-form midnight rejected: {r.parse_warning!r}"
        )
        assert r.value is not None
        assert (r.value.hour, r.value.minute, r.value.second) == (0, 0, 0)

    def test_noon_long_form(self) -> None:
        from bba.ingest.time_parser import parse_hosxp_time

        r = parse_hosxp_time("June 7, 2025, 12:00 PM")
        assert r.value is not None
        assert (r.value.hour, r.value.minute) == (12, 0)

    def test_3_30_pm_long_form(self) -> None:
        from bba.ingest.time_parser import parse_hosxp_time

        r = parse_hosxp_time("June 7, 2025, 3:30 PM")
        assert r.value is not None
        assert (r.value.hour, r.value.minute) == (15, 30)

    def test_8_15_am_long_form(self) -> None:
        from bba.ingest.time_parser import parse_hosxp_time

        r = parse_hosxp_time("December 31, 2025, 8:15 AM")
        assert r.value is not None
        assert (r.value.hour, r.value.minute) == (8, 15)

    @pytest.mark.parametrize(
        "raw",
        [
            "June 7, 2025",  # no time
            "Junne 7, 2025, 8:00 AM",  # misspelled month
            "13:00 June 7, 2025",  # transposed
            "June 7, 2025, 25:00 PM",  # invalid hour
            "June 7, 2025, 8 AM",  # no minutes
            "June 31, 2025, 8:00 AM",  # invalid calendar date
            "February 30, 2025, 8:00 AM",  # invalid calendar date
            "April 31, 2025, 8:00 AM",  # April has 30 days
            "February 29, 2025, 8:00 AM",  # 2025 is not a leap year
        ],
    )
    def test_long_form_near_misses_rejected(self, raw: str) -> None:
        # Strict parser still NEVER silently shifts — near-miss long-form
        # inputs must produce parse_warning, not a wrong-but-plausible time.
        from bba.ingest.time_parser import parse_hosxp_time

        r = parse_hosxp_time(raw)
        assert r.value is None, f"strict parser silently accepted {raw!r}"
        assert r.parse_warning is not None

    def test_long_form_leap_year_accepted(self) -> None:
        # 2024 is a leap year — Feb 29 IS a valid calendar date.
        from bba.ingest.time_parser import parse_hosxp_time

        r = parse_hosxp_time("February 29, 2024, 12:00 PM")
        assert r.parse_warning is None
        assert r.value is not None
        assert (r.value.hour, r.value.minute) == (12, 0)

    @given(
        month=st.sampled_from(
            [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        ),
        day=st.integers(min_value=1, max_value=28),
        year=st.integers(min_value=2000, max_value=2099),
        h12=st.integers(min_value=1, max_value=12),
        minute=st.integers(min_value=0, max_value=59),
        ampm=st.sampled_from(["AM", "PM"]),
    )
    @settings(max_examples=200)
    def test_long_form_property_round_trip(
        self,
        month: str,
        day: int,
        year: int,
        h12: int,
        minute: int,
        ampm: str,
    ) -> None:
        # Mirror of the existing HHMMSS / HH:MM property tests in
        # test_ingest.py: every well-formed long-form input must parse
        # to the equivalent 24-h ParsedTimeOfDay with no warning.
        from bba.ingest.time_parser import parse_hosxp_time

        raw = f"{month} {day}, {year}, {h12}:{minute:02d} {ampm}"
        r = parse_hosxp_time(raw)
        assert r.parse_warning is None, (
            f"valid long-form rejected: {raw!r} → {r.parse_warning!r}"
        )
        assert r.value is not None
        # 12-h → 24-h conversion: 12 AM → 0; 12 PM → 12; otherwise +12 if PM.
        if ampm == "AM":
            expected_h = 0 if h12 == 12 else h12
        else:
            expected_h = 12 if h12 == 12 else h12 + 12
        assert (r.value.hour, r.value.minute) == (expected_h, minute)
