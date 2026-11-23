"""Unit tests for the per-event returns summary (blood_audit #276, plan PR 5).

The audit judges each issue event on its own units. A REQNO-level summary
calls a REQNO transfused as soon as one unit was kept, which hides an event
whose units all came back: that event must screen RETURNED_NOT_TRANSFUSED and
be excluded from scoring, while its sibling events are still judged. The
REQNO totals are kept unchanged for the report and the go-live evidence.

Fixtures are synthetic: invented REQNOs, admissions, donor numbers and a
fictional year, shaped like the January 2025 gold cases.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from itertools import product
from zoneinfo import ZoneInfo

import pytest

from bba.issue_events import IssueEvent, cluster_events, units_from_ledger
from bba.returns_ledger import (
    ReturnsSummary,
    rows_for_admission,
    summarize_event_returns,
    summarize_returns,
)

AN = "AN-TEST-1"
BANGKOK = ZoneInfo("Asia/Bangkok")
OWN_ORDER = datetime(2030, 3, 1, 8, 0, tzinfo=BANGKOK)


def at(hours: float) -> datetime:
    return OWN_ORDER + timedelta(hours=hours)


def row(
    dnrno: str,
    unitstat: str,
    pay: datetime | None,
    *,
    an: str = AN,
    rtn: datetime | None = None,
) -> dict[str, str]:
    def day(moment: datetime | None) -> str:
        return moment.strftime("%Y-%m-%d 00:00:00.000") if moment else ""

    def clock(moment: datetime | None) -> str:
        return str(int(moment.strftime("%H%M%S"))) if moment else ""

    return {
        "REQNO": "R-TEST-1",
        "AN": an,
        "DNRNO": dnrno,
        "SEQNO": "1",
        "BDTYPE": "LDPRC",
        "UNITSTAT": unitstat,
        "PAYDATE": day(pay),
        "PAYTIME": clock(pay),
        "RTNDATE": day(rtn),
        "RTNTIME": clock(rtn),
    }


def events_of(rows: list[dict[str, str]]) -> tuple[IssueEvent, ...]:
    return cluster_events(units_from_ledger(rows, AN).issued, [OWN_ORDER])


def event_summaries(
    rows: list[dict[str, str]], unitamt_lines: list[str]
) -> tuple[ReturnsSummary, list[ReturnsSummary]]:
    """The REQNO summary and one summary per event, as PR 6 will call them."""
    scoped = rows_for_admission(rows, AN)
    reqno = summarize_returns(scoped, unitamt_lines)
    return reqno, [
        summarize_event_returns([u.status for u in e.units]) for e in events_of(scoped)
    ]


# Case 21 shape: the first unit came back unused, then three units issued a
# day apart were kept. The REQNO reads "transfused"; event 1 did not happen.
CASE_21 = [
    row("D1", "3", at(1), rtn=at(3)),
    row("D2", "2", at(26)),
    row("D3", "2", at(50)),
    row("D4", "2", at(74)),
]


def test_all_returned_event_is_not_transfused_while_its_reqno_is() -> None:
    reqno, events = event_summaries(CASE_21, ["4"])

    assert reqno.disposition == "transfused"
    assert [e.disposition for e in events] == [
        "not_transfused",
        "transfused",
        "transfused",
        "transfused",
    ]


def test_mixed_event_is_transfused_and_shows_its_returned_units() -> None:
    # 68000242 shape: six units issued together, three kept and three returned.
    # The kept units make it a transfusion to judge; the returned count is a
    # fact the row and the prompt carry.
    rows = [row(f"D{i}", s, at(1)) for i, s in enumerate("222333")]

    _, [event] = event_summaries(rows, ["6"])

    assert event.disposition == "transfused"
    assert (event.units_total, event.units_returned) == (6, 3)


def test_reqno_totals_are_unchanged_and_equal_the_sum_of_its_events() -> None:
    reqno, events = event_summaries(CASE_21, ["4"])

    assert reqno == ReturnsSummary(
        units_total=4,
        units_returned=1,
        units_transfused=0,
        units_incompat=0,
        ordered_unit_amount=4,
        ledger_complete=True,
    )
    for field in (
        "units_total",
        "units_returned",
        "units_transfused",
        "units_incompat",
    ):
        assert sum(getattr(e, field) for e in events) == getattr(reqno, field)


def test_incompatible_unit_counts_as_not_given_but_not_as_returned() -> None:
    # Status 7 (crossmatch incompatible) left the bank and was never given.
    rows = [row("D1", "3", at(1)), row("D2", "7", at(1))]

    _, [event] = event_summaries(rows, ["2"])

    assert event.disposition == "not_transfused"
    assert (event.units_returned, event.units_incompat) == (1, 1)


def test_stray_return_date_on_a_dispensed_unit_is_not_a_return() -> None:
    rows = [row("D1", "2", at(1), rtn=at(2))]

    _, [event] = event_summaries(rows, ["1"])

    assert event.disposition == "transfused"
    assert event.units_returned == 0


def test_reserved_not_issued_unit_is_neither_issued_nor_returned_on_the_event() -> None:
    # A crossmatch hold returned without leaving the bank has no PAYDATE. It
    # stays in the REQNO totals but belongs to no event.
    rows = [row("D1", "2", at(1)), row("D2", "3", None)]

    reqno, [event] = event_summaries(rows, ["2"])

    assert (reqno.units_total, reqno.units_returned) == (2, 1)
    assert (event.units_total, event.units_returned) == (1, 0)
    assert event.disposition == "transfused"


def test_all_returned_event_is_not_transfused_when_its_reqno_issued_short() -> None:
    # Owner decision 1.3: two units ordered, one issued and returned. The
    # REQNO still fails closed; the event is judged on the units it issued.
    rows = [row("D1", "3", at(1))]

    reqno, [event] = event_summaries(rows, ["2"])

    assert reqno.disposition == "inconclusive"
    assert event.disposition == "not_transfused"


def test_foreign_admission_units_are_in_neither_summary() -> None:
    rows = [row("D1", "2", at(1)), row("D9", "3", at(1), an="AN-OTHER")]

    reqno, [event] = event_summaries(rows, ["1"])

    assert reqno.units_total == event.units_total == 1
    assert reqno.units_returned == event.units_returned == 0


def test_an_event_without_units_is_rejected() -> None:
    with pytest.raises(ValueError, match="no units"):
        summarize_event_returns([])


@pytest.mark.parametrize("ordered", ["2", "3"])
@pytest.mark.parametrize(
    "statuses", list(product(["1", "2", "3", "4", "5", "7", ""], repeat=2))
)
def test_event_screens_not_transfused_exactly_when_all_its_units_returned(
    statuses: tuple[str, str], ordered: str
) -> None:
    # The #249 cutoff skips an all-returned event as a prior transfusion; the
    # classifier must agree on which events those are, also when the REQNO
    # issued fewer units than ordered.
    rows = [row(f"D{i}", s, at(1)) for i, s in enumerate(statuses)]

    _, [event_summary] = event_summaries(rows, [ordered])
    [event] = events_of(rows)

    expected = "not_transfused" if event.all_returned else "transfused"
    assert event_summary.disposition == expected
