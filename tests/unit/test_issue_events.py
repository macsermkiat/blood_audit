"""Unit tests for issue-event clustering (blood_audit #276, plan PR 1).

The audit judges one transfusion decision per issue event: the physical units a
REQNO issued close together, anchored on the first unit's issue time. Getting
the clusters wrong either merges two decisions under one Hb (the #276 error) or
splits one decision into redundant judgments, and the first unit of each event
fixes the event's audit_id, so its order is part of the contract.

Fixtures are synthetic: invented REQNOs, admissions, donor numbers and a
fictional year, shaped like the January 2025 gold cases.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from bba.issue_events import (
    IssuedUnit,
    IssueEvent,
    cluster_events,
    units_from_ledger,
)

AN = "AN-TEST-1"
BANGKOK = ZoneInfo("Asia/Bangkok")
OWN_ORDER = datetime(2030, 3, 1, 8, 0, tzinfo=BANGKOK)


def at(hours: float) -> datetime:
    return OWN_ORDER + timedelta(hours=hours)


def _date(moment: datetime | None) -> str:
    return moment.strftime("%Y-%m-%d 00:00:00.000") if moment else ""


def _time(moment: datetime | None) -> str:
    # The export stores times as integer-like text with leading zeros dropped.
    return str(int(moment.strftime("%H%M%S"))) if moment else ""


def row(
    dnrno: str,
    *,
    seqno: str = "1",
    bdtype: str = "LDPRC",
    unitstat: str = "2",
    an: str = AN,
    pay: datetime | None = None,
    rtn: datetime | None = None,
    give: datetime | None = None,
) -> dict[str, str]:
    return {
        "REQNO": "R-TEST-1",
        "AN": an,
        "DNRNO": dnrno,
        "SEQNO": seqno,
        "BDTYPE": bdtype,
        "UNITSTAT": unitstat,
        "PAYDATE": _date(pay),
        "PAYTIME": _time(pay),
        "RTNDATE": _date(rtn),
        "RTNTIME": _time(rtn),
        "GIVEDATE": _date(give),
        "GIVETIME": _time(give),
    }


def events_for(
    rows: list[dict[str, str]], order_times: tuple[datetime, ...] = (OWN_ORDER,)
) -> tuple[IssueEvent, ...]:
    return cluster_events(units_from_ledger(rows, AN).issued, order_times)


def test_case_21_shape_each_separate_issue_is_its_own_event() -> None:
    # Case 21: four units issued a day or more apart under one REQNO are four
    # transfusion decisions, each judged at its own issue time. The first unit
    # came back unused, so that event alone is all-returned.
    rows = [
        row("D1", unitstat="3", pay=at(4.6), rtn=at(6.0)),
        row("D2", pay=at(27.0)),
        row("D3", pay=at(47.4)),
        row("D4", pay=at(80.0)),
    ]
    orders = (OWN_ORDER, at(22.8), at(44.0), at(78.0))

    events = events_for(rows, orders)

    assert [e.issue_datetime for e in events] == [at(4.6), at(27.0), at(47.4), at(80.0)]
    assert [len(e.units) for e in events] == [1, 1, 1, 1]
    assert [e.all_returned for e in events] == [True, False, False, False]


def test_units_pulled_together_form_one_event_per_pull() -> None:
    # 68000024 shape: single units and pairs pulled within minutes. A pair is
    # one decision, so the REQNO yields 1, 2, 1, 2 units per event.
    pays = (49.0, 62.4, 62.45, 74.3, 96.3, 96.35)
    rows = [row(f"D{i}", pay=at(h)) for i, h in enumerate(pays)]

    events = events_for(rows)

    assert [len(e.units) for e in events] == [1, 2, 1, 2]
    assert [e.issue_datetime for e in events] == [
        at(49.0),
        at(62.4),
        at(74.3),
        at(96.3),
    ]


@pytest.mark.parametrize(
    ("gap", "expected_events"),
    [
        (timedelta(hours=3, minutes=59), 1),
        (timedelta(hours=4), 1),
        (timedelta(hours=4, minutes=1), 2),
    ],
)
def test_four_hour_gap_is_inclusive(gap: timedelta, expected_events: int) -> None:
    # Plan 1.2: a unit joins the current event when its gap to the previous
    # unit is at most 4 h; past that a new Hb approaches even odds.
    rows = [row("D1", pay=at(1.0)), row("D2", pay=at(1.0) + gap)]

    assert len(events_for(rows)) == expected_events


def test_gap_is_measured_from_the_previous_unit_not_the_event_start() -> None:
    # Units 3 h apart chain into one event even though the last is 6 h after
    # the first: each gap carries no new decision.
    rows = [row("D1", pay=at(1.0)), row("D2", pay=at(4.0)), row("D3", pay=at(7.0))]

    events = events_for(rows)

    assert len(events) == 1
    assert len(events[0].units) == 3


def test_gap_is_configurable() -> None:
    rows = [row("D1", pay=at(1.0)), row("D2", pay=at(6.0))]
    units = units_from_ledger(rows, AN).issued

    assert len(cluster_events(units, (OWN_ORDER,))) == 2
    assert len(cluster_events(units, (OWN_ORDER,), gap=timedelta(hours=6))) == 1


@pytest.mark.parametrize(
    ("order_offset_h", "expected_events"),
    [
        (-1.0, 1),  # placed before the first unit
        (2.5, 1),  # exactly at the first unit's issue: the interval is open there
        (3.0, 2),  # between the two units
        (3.5, 2),  # exactly at the second unit's issue: the interval is closed there
        (4.0, 1),  # after the second unit
    ],
)
def test_order_placed_between_two_units_starts_a_new_event(
    order_offset_h: float, expected_events: int
) -> None:
    # Plan 1.2: an order placed in (previous unit issue, this unit issue] is a
    # new decision even when the units are well inside the 4 h gap.
    rows = [row("D1", pay=at(2.5)), row("D2", pay=at(3.5))]

    assert len(events_for(rows, (OWN_ORDER, at(order_offset_h)))) == expected_events


def test_unit_never_issued_is_reserved_not_issued_and_forms_no_event() -> None:
    # A crossmatch hold returned without leaving the bank has no PAYDATE: it is
    # not a transfusion decision and never counts as returned-after-issue.
    rows = [row("D1", pay=at(2.0)), row("D2", unitstat="3")]

    ledger = units_from_ledger(rows, AN)
    events = cluster_events(ledger.issued, (OWN_ORDER,))

    assert ledger.units_reserved_not_issued == 1
    assert [u.dnrno for u in ledger.issued] == ["D1"]
    assert len(events) == 1
    assert events[0].all_returned is False


def test_request_with_no_issued_unit_has_no_events() -> None:
    # A REQNO whose units never left the bank (no_issue_record) yields no event.
    ledger = units_from_ledger([row("D1", unitstat="3"), row("D2", unitstat="7")], AN)

    assert ledger.issued == ()
    assert ledger.units_reserved_not_issued == 2
    assert cluster_events(ledger.issued, (OWN_ORDER,)) == ()


def test_rows_on_a_foreign_admission_are_dropped() -> None:
    # A REQNO can recur on another admission; a foreign returned unit must not
    # join this order's events or make them read as all-returned.
    rows = [
        row("D1", pay=at(2.0)),
        row("D9", unitstat="3", an="AN-OTHER", pay=at(2.1), rtn=at(3.0)),
    ]

    events = events_for(rows)

    assert [[u.dnrno for u in e.units] for e in events] == [["D1"]]
    assert events[0].all_returned is False


def test_lifecycle_rows_of_one_unit_collapse_to_its_terminal_state() -> None:
    # A dispense row and a later return row of the same physical unit are one
    # unit: issued at the earliest PAY time, terminal status returned.
    rows = [
        row("D1", seqno="4", pay=at(3.0)),
        row("D1", seqno="4", unitstat="3", pay=at(3.2), rtn=at(5.0)),
    ]

    ledger = units_from_ledger(rows, AN)

    assert ledger.issued == (
        IssuedUnit(
            dnrno="D1",
            seqno="4",
            bdtype="LDPRC",
            status="3",
            issue_at=at(3.0),
            returned_at=at(5.0),
            given_at=None,
        ),
    )


def test_given_time_is_recorded_without_changing_the_issue_anchor() -> None:
    ledger = units_from_ledger([row("D1", unitstat="5", pay=at(3.0), give=at(3.5))], AN)

    (unit,) = ledger.issued
    assert unit.issue_at == at(3.0)
    assert unit.given_at == at(3.5)
    assert cluster_events(ledger.issued, (OWN_ORDER,))[0].issue_datetime == at(3.0)


def test_stray_return_date_on_a_dispensed_unit_is_not_a_return() -> None:
    # Plan 1.3: returned keys on terminal status 3, never on RTNDATE alone.
    events = events_for([row("D1", unitstat="2", pay=at(2.0), rtn=at(4.0))])

    assert events[0].all_returned is False
    assert events[0].units_returned == 0


@pytest.mark.parametrize(
    ("statuses", "all_returned", "units_returned"),
    [
        (("3", "3"), True, 2),
        # Crossmatch-incompatible was never given, but it is not a return:
        # issue_units_returned must agree with the returns-ledger count.
        (("3", "7"), True, 1),
        (("3", "2"), False, 1),  # mixed: judged as a transfusion
        (("5", "2"), False, 0),
    ],
)
def test_event_is_all_returned_only_when_no_unit_was_transfused(
    statuses: tuple[str, str], all_returned: bool, units_returned: int
) -> None:
    rows = [
        row(f"D{i}", unitstat=status, pay=at(2.0 + i * 0.1))
        for i, status in enumerate(statuses)
    ]

    (event,) = events_for(rows)

    assert event.all_returned is all_returned
    assert event.units_returned == units_returned


def test_identical_issue_times_order_by_donor_sequence_and_product() -> None:
    # The first unit fixes the event's audit_id, so ties on issue time break on
    # (DNRNO, SEQNO, BDTYPE) whatever order the ledger rows arrive in.
    same = at(2.0)
    rows = [
        row("B", seqno="1", pay=same),
        row("A", seqno="2", pay=same),
        row("A", seqno="1", bdtype="SDRF2", pay=same),
        row("A", seqno="1", bdtype="SDRF", pay=same),
    ]

    expected = [
        ("A", "1", "SDRF"),
        ("A", "1", "SDRF2"),
        ("A", "2", "LDPRC"),
        ("B", "1", "LDPRC"),
    ]

    ledger = units_from_ledger(rows, AN)
    (event,) = cluster_events(ledger.issued, (OWN_ORDER,))

    assert [(u.dnrno, u.seqno, u.bdtype) for u in ledger.issued] == expected
    assert [(u.dnrno, u.seqno, u.bdtype) for u in event.units] == expected


def test_dispensed_unit_without_a_readable_issue_time_fails_loud() -> None:
    # A dispensed or transfused unit did leave the bank. Counting it as never
    # issued would drop a transfusion from the audit without a trace.
    dispensed = row("D1", unitstat="2", pay=at(2.0))
    dispensed["PAYTIME"] = ""

    with pytest.raises(ValueError, match="status 2"):
        units_from_ledger([dispensed], AN)


def test_ledger_needs_the_order_admission() -> None:
    # Without an AN the ledger rows cannot be scoped, and foreign units could
    # leak in, so the call refuses instead of reading them all.
    with pytest.raises(ValueError, match="AN"):
        units_from_ledger([row("D1", pay=at(2.0))], "")


def _unit(dnrno: str, issue_at: datetime) -> IssuedUnit:
    return IssuedUnit(
        dnrno=dnrno,
        seqno="1",
        bdtype="LDPRC",
        status="2",
        issue_at=issue_at,
        returned_at=None,
        given_at=None,
    )


def test_event_cannot_be_built_out_of_issue_order_or_empty() -> None:
    with pytest.raises(ValidationError):
        IssueEvent(units=(_unit("D2", at(3.0)), _unit("D1", at(2.0))))
    with pytest.raises(ValidationError):
        IssueEvent(units=())
