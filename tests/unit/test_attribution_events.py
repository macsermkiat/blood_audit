"""Attribution per issue event (blood_audit #276, plan 2.3 and PR 8).

One REQNO can hold several issue events, each a separate decision to send
blood. The blood bank issues later units against the first open REQNO, so a
later event is often caused by a fresh order from another doctor (case 21).
Crediting it to the REQNO's doctor is the #276 error: the doctor who placed
the first order would carry the next doctor's decision. These tests pin the
switch: each event counts once, for the doctor of its linked order.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from bba.attribution import (
    UNATTRIBUTED_DOCTOR_ID,
    DoctorRecord,
    EventVerdict,
    OrderLabValue,
    build_department_scorecards,
    build_doctor_scorecards,
    build_rankings,
    load_order_labs,
)

_OWN, _FRESH_2, _FRESH_3, _FRESH_4 = "R21", "F2", "F3", "F4"
_REQNO_TO_DOCTOR = {
    _OWN: "doc-own",
    _FRESH_2: "doc-2",
    _FRESH_3: "doc-3",
    _FRESH_4: "doc-4",
}
_REGISTRY = {
    code: DoctorRecord(dct=code, display_name=code, deptlct="DEPT", deptname="Dept")
    for code in _REQNO_TO_DOCTOR.values()
}


def _event(audit_id: str, linked: str, verdict: str) -> EventVerdict:
    return EventVerdict(
        audit_id=audit_id,
        reqno=_OWN,
        linked_reqno=linked,
        component="red_cell",
        verdict=verdict,
    )


def _case_21() -> tuple[EventVerdict, ...]:
    """Case-21-shaped REQNO: event 1 all returned under its own order; events
    2-4 each caused by a fresh order from a different doctor."""
    return (
        _event("ev1", _OWN, "RETURNED_NOT_TRANSFUSED"),
        _event("ev2", _FRESH_2, "INAPPROPRIATE"),
        _event("ev3", _FRESH_3, "APPROPRIATE"),
        _event("ev4", _FRESH_4, "PERIOP_TRANSFUSION_EXEMPT"),
    )


def _labs(hb: dict[str, float]) -> dict[str, OrderLabValue]:
    return {
        audit_id: OrderLabValue(
            audit_id=audit_id, reqno=_OWN, component="red_cell", hb_value_g_dl=value
        )
        for audit_id, value in hb.items()
    }


def test_later_events_are_credited_to_the_linked_order_doctor() -> None:
    cards = {
        c.physician_id: c
        for c in build_doctor_scorecards(_case_21(), _REQNO_TO_DOCTOR, _REGISTRY)
    }

    assert cards["doc-2"].inappropriate_count == 1
    assert cards["doc-3"].appropriate_count == 1
    assert cards["doc-4"].periop_transfusion_exempt_count == 1
    assert cards["doc-own"].inappropriate_count == 0
    assert cards["doc-own"].appropriate_count == 0


def test_returned_event_is_attributed_but_not_scored() -> None:
    result = build_rankings(
        verdicts=_case_21(),
        reqno_to_doctor=_REQNO_TO_DOCTOR,
        dct_registry=_REGISTRY,
        order_labs=_labs({"ev2": 8.5, "ev3": 7.6}),
        min_orders=1,
    )
    cards = {
        c.physician_id: c
        for c in build_doctor_scorecards(_case_21(), _REQNO_TO_DOCTOR, _REGISTRY)
    }

    assert cards["doc-own"].returned_not_transfused_count == 1
    assert cards["doc-own"].total_orders == 0
    assert result.totals.returned_not_transfused == 1
    assert result.totals.total == 2


def test_standing_order_event_stays_with_the_reqno_doctor() -> None:
    events = (
        _event("ev1", _OWN, "APPROPRIATE"),
        _event("ev2", _OWN, "INAPPROPRIATE"),
    )

    (card,) = build_doctor_scorecards(events, _REQNO_TO_DOCTOR, _REGISTRY)

    assert card.physician_id == "doc-own"
    assert card.total_orders == 2


def test_linked_order_without_a_doctor_is_unattributed_not_the_reqno_doctor() -> None:
    events = (_event("ev2", "F-unknown", "INAPPROPRIATE"),)

    (card,) = build_doctor_scorecards(events, _REQNO_TO_DOCTOR, _REGISTRY)

    assert card.physician_id == UNATTRIBUTED_DOCTOR_ID


def test_trigger_mean_uses_each_event_hb() -> None:
    result = build_rankings(
        verdicts=_case_21(),
        reqno_to_doctor={_OWN: "doc-own", _FRESH_2: "doc-x", _FRESH_3: "doc-x"},
        dct_registry=_REGISTRY,
        order_labs=_labs({"ev1": 11.3, "ev2": 8.5, "ev3": 7.6, "ev4": 8.6}),
        min_orders=1,
    )
    rows = {r.group_id: r for r in result.doctors.rows}

    assert rows["doc-x"].mean_hb == pytest.approx((8.5 + 7.6) / 2)
    assert rows["doc-x"].hb_order_n == 2


def test_a_repeated_audit_id_fails_loud() -> None:
    events = (*_case_21(), _event("ev2", _FRESH_2, "INAPPROPRIATE"))

    with pytest.raises(ValueError, match="ev2"):
        build_rankings(
            verdicts=events, reqno_to_doctor=_REQNO_TO_DOCTOR, dct_registry=_REGISTRY
        )


def test_load_order_labs_keeps_two_events_of_one_reqno(tmp_path: Path) -> None:
    path = tmp_path / "report.csv"
    columns = [
        "audit_id",
        "reqno",
        "component",
        "hb_value_g_dl",
        "hb_freshness",
        "platelet_count_k_ul",
        "platelet_freshness",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for audit_id, hb in (("ev2", "8.5"), ("ev3", "7.6")):
            writer.writerow(
                {
                    "audit_id": audit_id,
                    "reqno": _OWN,
                    "component": "red_cell",
                    "hb_value_g_dl": hb,
                    "hb_freshness": "fresh",
                    "platelet_count_k_ul": "",
                    "platelet_freshness": "missing",
                }
            )

    labs = load_order_labs(path)

    assert labs["ev2"].hb_value_g_dl == 8.5
    assert labs["ev3"].hb_value_g_dl == 7.6
    assert {lab.reqno for lab in labs.values()} == {_OWN}


def test_equal_triggers_on_several_events_do_not_trip_the_range_invariant() -> None:
    # One doctor with three events at the same Hb: the float mean of
    # 11.7, 11.7, 11.7 is 11.699999999999998, below the minimum, and the
    # GroupLabStats validator rejected the whole January ranking.
    events = tuple(_event(f"ev{i}", _OWN, "INAPPROPRIATE") for i in range(3))

    result = build_rankings(
        verdicts=events,
        reqno_to_doctor=_REQNO_TO_DOCTOR,
        dct_registry=_REGISTRY,
        order_labs=_labs({f"ev{i}": 11.7 for i in range(3)}),
        min_orders=1,
    )

    assert result.doctors.rows[0].mean_hb == pytest.approx(11.7)


def test_departments_follow_the_linked_order_doctor() -> None:
    registry = {
        "doc-own": DoctorRecord(
            dct="doc-own", display_name="o", deptlct="OWN", deptname="Own"
        ),
        "doc-2": DoctorRecord(
            dct="doc-2", display_name="t", deptlct="OTHER", deptname="Other"
        ),
    }
    events = (
        _event("ev1", _OWN, "APPROPRIATE"),
        _event("ev2", _FRESH_2, "INAPPROPRIATE"),
    )

    cards = {
        c.ward_id: c
        for c in build_department_scorecards(events, _REQNO_TO_DOCTOR, registry)
    }

    assert cards["OWN"].appropriate_count == 1
    assert cards["OWN"].inappropriate_count == 0
    assert cards["OTHER"].inappropriate_count == 1
