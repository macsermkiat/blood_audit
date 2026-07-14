"""Unit tests for the returns-ledger pre-flight validation report (ticket #125).

The pre-flight is a READ-ONLY validation report that gates flipping
``RETURNS_LEDGER_ENABLED`` on. It never changes pipeline behaviour, so these
tests pin the small, pure decision functions it is built from:

* ``is_reissue`` — the reissue / partial-coverage counter.
* ``hard_transfusion_contradiction`` / ``is_screened_returned_not_transfused`` —
  the screened-set predicate, cross-checked against the REAL
  :func:`bba.deterministic_classifier.classify` so it cannot silently drift from
  the #122 terminal it is validating.
* ``nonreturned_unit_count`` — the invariant re-derived straight from the raw
  ledger rows (so the invariant check cannot false-pass off the same counters
  ``summarize_returns`` produced).
* ``administration_recall_conflicts`` — the ให้เลือด administration-note recall
  harness over the screened orders.
* ``recommendation`` — the deterministic go / hold gate.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

from bba.cohort_detector import CohortAssignment, CohortLabel
from bba.deterministic_classifier import ClassifierInputs, classify
from bba.hb_lookup import HbLookupResult
from bba.returns_ledger import ReturnsSummary
from bba.vitals_extractor import VitalsNote


def _load_preflight() -> ModuleType:
    pilot_dir = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
    if str(pilot_dir) not in sys.path:
        sys.path.insert(0, str(pilot_dir))
    spec = importlib.util.spec_from_file_location(
        "pilot_preflight_returns_validation_test",
        pilot_dir / "preflight_returns_validation.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's frozen @dataclass field annotations
    # (strings under `from __future__ import annotations`) resolve via
    # sys.modules[cls.__module__] instead of crashing on a None module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PF = _load_preflight()


def _summary(
    disposition: str, *, units: int = 2, ordered: int | None = 2
) -> ReturnsSummary:
    """Build a ReturnsSummary whose derived disposition is ``disposition``."""
    if disposition == "not_transfused":
        return ReturnsSummary(
            units_total=units,
            units_returned=units,
            ordered_unit_amount=ordered,
            ledger_complete=True,
        )
    if disposition == "transfused":
        return ReturnsSummary(
            units_total=units,
            units_returned=units - 1,
            units_transfused=1,
            ordered_unit_amount=ordered,
            ledger_complete=True,
        )
    # inconclusive
    return ReturnsSummary(
        units_total=units, units_returned=units, ledger_complete=False
    )


def _note(text: str, *, at: datetime | None = None) -> VitalsNote:
    return VitalsNote(
        timestamp=at or datetime(2026, 7, 13, tzinfo=UTC),
        text=text,
        source="IPDADMPROGRESS",
    )


# --- is_reissue --------------------------------------------------------------


def test_is_reissue_true_when_units_exceed_ordered() -> None:
    s = ReturnsSummary(
        units_total=6, units_returned=6, ordered_unit_amount=4, ledger_complete=True
    )
    assert PF.is_reissue(s) is True


def test_is_reissue_false_when_units_equal_ordered() -> None:
    assert PF.is_reissue(_summary("not_transfused", units=2, ordered=2)) is False


def test_is_reissue_false_when_incomplete() -> None:
    # An incomplete ledger is never screened, so it is not a reissue offender.
    assert PF.is_reissue(_summary("inconclusive")) is False


def test_is_reissue_false_when_ordered_unknown() -> None:
    s = ReturnsSummary(
        units_total=2, units_returned=2, ordered_unit_amount=None, ledger_complete=False
    )
    assert PF.is_reissue(s) is False


# --- relaxed over-dispense guard (complete-ledger go-live) --------------------


def test_over_dispensed_all_returned_now_screens_not_transfused() -> None:
    # With a guaranteed-complete ledger an over-dispensed all-returned order
    # genuinely had ALL its units returned, so it screens not_transfused — the
    # earlier NARROW exact-count guard is relaxed (spec #119 complete-ledger
    # go-live). It is still surfaced as a reissue for the sign-off.
    s = ReturnsSummary(
        units_total=3, units_returned=3, ordered_unit_amount=2, ledger_complete=True
    )
    assert s.disposition == "not_transfused"
    assert PF.is_reissue(s) is True


# --- hard_transfusion_contradiction / screened predicate ---------------------


@pytest.mark.parametrize(
    ("intraop", "ebl", "expected"),
    [
        (False, None, False),
        (True, None, True),
        (False, 499, False),
        (False, 500, True),
        (False, 1300, True),
    ],
)
def test_hard_transfusion_contradiction(intraop, ebl, expected) -> None:
    assert (
        PF.hard_transfusion_contradiction(
            intraop_transfusion=intraop, blood_loss_ml=ebl
        )
        is expected
    )


def test_screened_true_for_clean_not_transfused() -> None:
    assert (
        PF.is_screened_returned_not_transfused(
            _summary("not_transfused"), intraop_transfusion=False, blood_loss_ml=None
        )
        is True
    )


def test_screened_false_when_hard_contradiction() -> None:
    assert (
        PF.is_screened_returned_not_transfused(
            _summary("not_transfused"), intraop_transfusion=False, blood_loss_ml=600
        )
        is False
    )


@pytest.mark.parametrize("disposition", ["transfused", "inconclusive"])
def test_screened_false_for_non_returned_disposition(disposition) -> None:
    assert (
        PF.is_screened_returned_not_transfused(
            _summary(disposition), intraop_transfusion=False, blood_loss_ml=None
        )
        is False
    )


def _neutral_inputs(
    summary: ReturnsSummary, *, intraop: bool, ebl: int | None
) -> ClassifierInputs:
    """Real ClassifierInputs with neutral Hb/cohort so only the returns exit varies."""
    return ClassifierInputs(
        audit_id="preflight-crosscheck",
        hb_result=HbLookupResult(
            value_g_dl=None,
            datetime_utc=None,
            source=None,
            freshness="missing",
            delta_hb_bypass=False,
            delta_hb_windows=(),
            needs_review_single_low_hb=False,
        ),
        cohort_assignment=CohortAssignment(
            label=CohortLabel.DEFAULT,
            threshold=7.0,
            evidence_code=None,
            evidence_name=None,
        ),
        order_datetime=datetime(2026, 7, 13, tzinfo=UTC),
        procedure_proximity_hours=None,
        crystalloid_liters_prior_4h=0.0,
        periop_blood_loss_ml=ebl,
        periop_intraop_transfusion=intraop,
        returns_disposition=summary.disposition,
        returns_periop_context=False,
    )


@pytest.mark.parametrize(
    "disposition", ["not_transfused", "transfused", "inconclusive"]
)
@pytest.mark.parametrize("intraop", [False, True])
@pytest.mark.parametrize("ebl", [None, 499, 500])
def test_screened_predicate_matches_real_classifier(disposition, intraop, ebl) -> None:
    """The pre-flight's screened predicate must equal the real #122 terminal.

    If classifier.py's returned-exit contradiction guard ever changes, this
    cross-check fails loudly rather than letting the validation report screen a
    different set than the pipeline it is meant to validate.
    """
    summary = _summary(disposition)
    predicted = PF.is_screened_returned_not_transfused(
        summary, intraop_transfusion=intraop, blood_loss_ml=ebl
    )
    actual = (
        classify(_neutral_inputs(summary, intraop=intraop, ebl=ebl)).classification
        == "RETURNED_NOT_TRANSFUSED"
    )
    assert predicted is actual


# --- nonreturned_unit_count (invariant, re-derived from raw rows) ------------


def test_nonreturned_unit_count_all_returned() -> None:
    rows = [{"UNITSTAT": "3"}, {"UNITSTAT": "3"}]
    assert PF.nonreturned_unit_count(rows) == 0


def test_nonreturned_unit_count_flags_transfused_and_dispensed() -> None:
    rows = [{"UNITSTAT": "3"}, {"UNITSTAT": "5"}, {"UNITSTAT": "2"}]
    assert PF.nonreturned_unit_count(rows) == 2


def test_nonreturned_unit_count_handles_blank_and_missing() -> None:
    rows = [{"UNITSTAT": ""}, {"OTHER": "x"}, {"UNITSTAT": " 3 "}]
    # blank and missing are non-returned; " 3 " is returned after strip.
    assert PF.nonreturned_unit_count(rows) == 2


def test_nonreturned_unit_count_treats_incompatible_as_non_transfusion() -> None:
    # Unitstat=7 (crossmatch-incompatible) was never given, so it does NOT count
    # as evidence of a transfusion: an all-returned-or-incompatible order has zero
    # non-returned units and can screen not_transfused (spec #119 "fix status 7").
    rows = [{"UNITSTAT": "3"}, {"UNITSTAT": "7"}]
    assert PF.nonreturned_unit_count(rows) == 0


def test_nonreturned_unit_count_collapses_lifecycle_rows() -> None:
    # A unit dispensed then returned is ONE physical unit (shared full key
    # DNRNO/SEQNO/BDTYPE); collapsed to its terminal returned status it is not a
    # non-returned unit, so the dispense row does not inflate the invariant count.
    rows = [
        {"UNITSTAT": "2", "DNRNO": "U1", "SEQNO": "0", "BDTYPE": "LDPRC2"},
        {"UNITSTAT": "3", "DNRNO": "U1", "SEQNO": "0", "BDTYPE": "LDPRC2"},
    ]
    assert PF.nonreturned_unit_count(rows) == 0


def test_nonreturned_unit_count_fails_closed_on_partial_key() -> None:
    # A dispense + return sharing only DNRNO (no SEQNO/BDTYPE) must NOT collapse:
    # the dispensed unit stays counted so the invariant catches a possible
    # transfusion rather than silently clearing it.
    rows = [
        {"UNITSTAT": "2", "DNRNO": "U1"},
        {"UNITSTAT": "3", "DNRNO": "U1"},
    ]
    assert PF.nonreturned_unit_count(rows) == 1


def test_invariant_holds_for_summarize_not_transfused() -> None:
    """A summarize_returns not_transfused disposition implies zero non-returned units.

    Ties the raw-row invariant to summarize_returns: an all-'3' ledger yields
    not_transfused AND zero non-returned units, so a screened order can never
    contain a non-returned unit.
    """
    from bba.returns_ledger import summarize_returns

    rows = [{"UNITSTAT": "3"}, {"UNITSTAT": "3"}]
    summary = summarize_returns(rows, ["2"])
    assert summary.disposition == "not_transfused"
    assert PF.nonreturned_unit_count(rows) == 0


# --- administration recall harness -------------------------------------------


def test_recall_flags_affirmative_administration_note() -> None:
    conflicts = PF.administration_recall_conflicts(
        {"R1": (_note("ดูแลให้เลือด LPRC 1 unit iv drip in 4 hr at 12.20 น."),)}
    )
    assert len(conflicts) == 1
    assert conflicts[0].reqno == "R1"
    assert "gave_blood" in conflicts[0].categories
    assert conflicts[0].snippets  # a snippet was captured for review


def test_recall_ignores_planning_and_reservation_notes() -> None:
    conflicts = PF.administration_recall_conflicts(
        {
            "R2": (_note("plan จองเลือด LPRC 2 unit เตรียมผ่าตัด"),),
            "R3": (_note("ยังไม่ได้ให้เลือด"),),
        }
    )
    assert conflicts == ()


# --- windowed recall (temporal attribution) ----------------------------------


def test_parse_ledger_date_iso_format() -> None:
    # The complete production export uses ISO datetimes with fractional seconds.
    from datetime import date as _date

    assert PF.parse_ledger_date("2025-03-31 15:29:00.000") == _date(2025, 3, 31)
    assert PF.parse_ledger_date("2025-07-09 00:00:00.000") == _date(2025, 7, 9)
    assert PF.parse_ledger_date("2025-03-31") == _date(2025, 3, 31)


def test_parse_ledger_date_us_long_format_still_accepted() -> None:
    # The earlier partial export's US long datetime remains parseable.
    from datetime import date as _date

    assert PF.parse_ledger_date("March 31, 2025, 3:29 PM") == _date(2025, 3, 31)
    assert PF.parse_ledger_date("July 9, 2025, 12:00 AM") == _date(2025, 7, 9)


def test_parse_ledger_date_returns_none_on_garbage_or_blank() -> None:
    # Fail SAFE: an unparseable date leaves the order unwindowable -> caller
    # keeps the full admission notes rather than dropping a possible marker.
    assert PF.parse_ledger_date("") is None
    assert PF.parse_ledger_date("not a date") is None
    assert PF.parse_ledger_date(None) is None


def test_recall_window_pads_min_and_max() -> None:
    from datetime import date as _date

    window = PF.recall_window(
        [_date(2025, 8, 23), None, _date(2025, 8, 24)], pad_days=2
    )
    assert window == (_date(2025, 8, 21), _date(2025, 8, 26))


def test_recall_window_none_when_no_parseable_date() -> None:
    assert PF.recall_window([None, None]) is None


def test_notes_in_window_filters_by_date() -> None:
    from datetime import date as _date

    inside = _note("ให้เลือด LPRC 1 unit", at=datetime(2025, 8, 24, tzinfo=UTC))
    outside = _note("ให้เลือด LPRC 1 unit", at=datetime(2025, 9, 10, tzinfo=UTC))
    kept = PF.notes_in_window(
        (inside, outside), (_date(2025, 8, 21), _date(2025, 8, 26))
    )
    assert kept == (inside,)


def test_notes_in_window_none_keeps_all() -> None:
    a = _note("x", at=datetime(2025, 1, 1, tzinfo=UTC))
    b = _note("y", at=datetime(2030, 1, 1, tzinfo=UTC))
    assert PF.notes_in_window((a, b), None) == (a, b)


def test_explaining_sibling_matches_not_returned_unit_in_window() -> None:
    from datetime import date as _date

    sibs = (
        PF.SiblingUnit("O1", _date(2025, 4, 2), is_returned=False, status="2"),
        PF.SiblingUnit("O2", _date(2025, 4, 2), is_returned=True, status="3"),
    )
    match = PF.explaining_sibling((_date(2025, 3, 29), _date(2025, 4, 2)), sibs)
    assert match is not None and match.reqno == "O1"


def test_explaining_sibling_ignores_returned_or_out_of_window() -> None:
    from datetime import date as _date

    window = (_date(2025, 3, 29), _date(2025, 4, 2))
    # returned sibling (a second standby) is not an administration source
    assert (
        PF.explaining_sibling(
            window, (PF.SiblingUnit("O", _date(2025, 4, 1), True, "3"),)
        )
        is None
    )
    # not-returned but dispensed outside the window
    assert (
        PF.explaining_sibling(
            window, (PF.SiblingUnit("O", _date(2025, 4, 20), False, "2"),)
        )
        is None
    )
    # no window -> no attribution
    assert (
        PF.explaining_sibling(
            None, (PF.SiblingUnit("O", _date(2025, 4, 1), False, "2"),)
        )
        is None
    )


def test_explaining_sibling_matches_on_give_date_when_dispensed_earlier() -> None:
    # A sibling transfused (GIVEDATE) inside the window explains the note even if
    # it was dispensed before the window opened (spec #119 GIVEDATE anchor).
    from datetime import date as _date

    window = (_date(2025, 4, 1), _date(2025, 4, 5))
    sib = PF.SiblingUnit(
        "O1",
        _date(2025, 3, 20),
        is_returned=False,
        status="5",
        give_date=_date(2025, 4, 3),
    )
    match = PF.explaining_sibling(window, (sib,))
    assert match is not None and match.reqno == "O1"


def test_recall_conflict_note_dates_are_json_serializable() -> None:
    # RecallConflict.note_dates holds datetime.date objects; _write_artifact
    # json.dumps(asdict(result), ...) must not crash on them. Guard the exact
    # encoder options _write_artifact uses (default=str) so the machine-readable
    # HOLD/GO artifact is always written.
    import json
    from dataclasses import asdict
    from datetime import date as _date

    conflict = PF.RecallConflict(
        reqno="R1",
        categories=("gave_blood",),
        snippets=("x",),
        note_dates=(_date(2025, 4, 2),),
    )
    dumped = json.dumps(asdict(conflict), ensure_ascii=False, default=str)
    assert "2025-04-02" in dumped


def test_recall_conflict_records_marker_note_dates() -> None:
    # Attribution is note-specific, so the conflict must carry the date(s) of the
    # notes that actually flagged — not just the padded order window.
    conflicts = PF.administration_recall_conflicts(
        {
            "R1": (
                _note("nothing here", at=datetime(2025, 4, 1, tzinfo=UTC)),
                _note(
                    "ให้เลือด LPRC 1 unit iv drip in 4 hr",
                    at=datetime(2025, 4, 5, tzinfo=UTC),
                ),
            )
        }
    )
    from datetime import date as _date

    assert len(conflicts) == 1
    assert conflicts[0].note_dates == (_date(2025, 4, 5),)


def test_sibling_units_scope_to_admission_and_dedup() -> None:
    # A sibling REQNO can appear under >1 admission in the complete export, and
    # its rows may be lifecycle rows of one physical unit. _sibling_units_for must
    # (a) drop rows of a foreign AN and (b) collapse (DNRNO,SEQNO,BDTYPE) rows to
    # one physical unit at its terminal status.
    reqnos_by_an = {"ANx": {"OWN", "SIB"}}
    trans_by_reqno = {
        "SIB": [
            # this admission: dispensed then returned -> one returned unit
            {
                "AN": "ANx",
                "DNRNO": "D1",
                "SEQNO": "0",
                "BDTYPE": "LDPRC2",
                "UNITSTAT": "2",
                "PAYDATE": "2025-04-01 00:00:00.000",
            },
            {
                "AN": "ANx",
                "DNRNO": "D1",
                "SEQNO": "0",
                "BDTYPE": "LDPRC2",
                "UNITSTAT": "3",
                "PAYDATE": "2025-04-01 00:00:00.000",
            },
            # a DIFFERENT admission's dispensed unit must be excluded
            {
                "AN": "ANy",
                "DNRNO": "D9",
                "SEQNO": "0",
                "BDTYPE": "LDPRC2",
                "UNITSTAT": "2",
                "PAYDATE": "2025-04-02 00:00:00.000",
            },
        ]
    }
    units = PF._sibling_units_for("ANx", "OWN", reqnos_by_an, trans_by_reqno)
    assert len(units) == 1  # foreign-AN row excluded; lifecycle rows collapsed
    assert units[0].is_returned is True  # terminal returned -> not a source
    assert units[0].status == "3"


def test_sibling_incompatible_unit_is_not_a_source() -> None:
    # A crossmatch-incompatible (Unitstat=7) sibling was never given, so it must
    # not be treated as a transfusion source that suppresses a recall conflict.
    reqnos_by_an = {"ANx": {"OWN", "SIB"}}
    trans_by_reqno = {
        "SIB": [
            {
                "AN": "ANx",
                "DNRNO": "D1",
                "SEQNO": "0",
                "BDTYPE": "LDPRC2",
                "UNITSTAT": "7",
                "PAYDATE": "2025-04-01 00:00:00.000",
            }
        ]
    }
    units = PF._sibling_units_for("ANx", "OWN", reqnos_by_an, trans_by_reqno)
    assert len(units) == 1
    assert units[0].is_returned is True  # incompatible -> not a transfusion source


def test_explaining_sibling_picks_earliest_deterministically() -> None:
    from datetime import date as _date

    window = (_date(2025, 4, 1), _date(2025, 4, 5))
    sibs = (
        PF.SiblingUnit("LATE", _date(2025, 4, 4), False, "2"),
        PF.SiblingUnit("EARLY", _date(2025, 4, 2), False, "5"),
    )
    match = PF.explaining_sibling(window, sibs)
    assert match is not None and match.reqno == "EARLY"


def test_windowed_recall_drops_out_of_window_administration() -> None:
    # A real administration note 16 days after the return window is a different
    # transfusion in the same admission and must not flag the returned order.
    from datetime import date as _date

    late = _note(
        "ให้เลือด LPRC 1 unit iv drip in 4 hr", at=datetime(2025, 7, 26, tzinfo=UTC)
    )
    window = PF.recall_window([_date(2025, 7, 9), _date(2025, 7, 10)])
    windowed = {"R": PF.notes_in_window((late,), window)}
    assert PF.administration_recall_conflicts(windowed) == ()
    # …but admission-wide (no window) still sees it.
    assert len(PF.administration_recall_conflicts({"R": (late,)})) == 1


# --- recommendation gate -----------------------------------------------------


def test_recommendation_go_when_all_clean() -> None:
    assert (
        PF.recommendation(
            screened_count=41,
            notes_available=True,
            recall_conflicts=0,
            invariant_violations=0,
        )
        == "GO"
    )


def test_recommendation_go_even_with_reissue_on_complete_ledger() -> None:
    # A reissue / over-dispense is no longer a gate trigger: on a complete ledger
    # an over-dispensed all-returned order is safe to screen, and any transfusion
    # it might hide is caught by the recall check (spec #119 complete-ledger).
    assert (
        PF.recommendation(
            screened_count=41,
            notes_available=True,
            recall_conflicts=0,
            invariant_violations=0,
        )
        == "GO"
    )


def test_recommendation_hold_when_recall_conflict() -> None:
    assert (
        PF.recommendation(
            screened_count=41,
            notes_available=True,
            recall_conflicts=3,
            invariant_violations=0,
        )
        == "HOLD"
    )


def test_recommendation_hold_when_invariant_violated() -> None:
    assert (
        PF.recommendation(
            screened_count=41,
            notes_available=True,
            recall_conflicts=0,
            invariant_violations=1,
        )
        == "HOLD"
    )


def test_recommendation_hold_when_nothing_screened() -> None:
    # An empty / misjoined ledger screens nothing; absence of findings must NOT
    # read as GO (Codex critical).
    assert (
        PF.recommendation(
            screened_count=0,
            notes_available=True,
            recall_conflicts=0,
            invariant_violations=0,
        )
        == "HOLD"
    )


def test_recommendation_hold_when_notes_missing() -> None:
    # No note sources -> recall is vacuous, not clean (Codex high).
    assert (
        PF.recommendation(
            screened_count=41,
            notes_available=False,
            recall_conflicts=0,
            invariant_violations=0,
        )
        == "HOLD"
    )
