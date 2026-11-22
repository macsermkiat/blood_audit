"""Unit tests for issue-event order linkage (blood_audit #276, plan PR 2).

Each later issue of blood is judged against the order that caused it and
credited to the doctor who placed that order. The blood bank books later
units on the first open REQNO, so the REQNO alone names the wrong order for
every event after the first: an order placed between two issues is the fresh
decision, and with no such order the event is more blood on the standing one.

Fixtures are synthetic: invented REQNOs and admissions in a fictional year,
shaped like the January 2025 gold cases.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from bba.audit_orders import BloodOrderInput
from bba.ingest.models import ParsedTimeOfDay
from bba.issue_events import (
    CandidateOrder,
    IssuedUnit,
    IssueEvent,
    candidate_order,
    link_events,
)

AN = "AN-TEST-1"
BANGKOK = ZoneInfo("Asia/Bangkok")
OWN_PLACED = datetime(2030, 3, 1, 8, 0, tzinfo=BANGKOK)


def at(hours: float) -> datetime:
    return OWN_PLACED + timedelta(hours=hours)


def order(
    reqno: str,
    placed_h: float,
    *,
    status: str = "1",
    an: str = AN,
    component: str = "red_cell",
) -> CandidateOrder:
    return CandidateOrder.model_validate(
        {
            "reqno": reqno,
            "an": an,
            "component": component,
            "status": status,
            "placed_at": at(placed_h),
        }
    )


OWN = order("R-OWN", 0, status="4")


def event(*issue_h: float) -> IssueEvent:
    return IssueEvent(
        units=tuple(
            IssuedUnit(
                dnrno=f"D{i}",
                seqno="1",
                bdtype="LDPRC",
                status="2",
                issue_at=at(h),
                returned_at=None,
                given_at=None,
            )
            for i, h in enumerate(issue_h)
        )
    )


def bdvst_row(
    reqno: str,
    *,
    status: str = "4",
    an: str | None = AN,
    hour: int | None = 8,
    visit_hour: int | None = 8,
    products: tuple[str, ...] = ("LDPRC",),
) -> BloodOrderInput:
    day = date(2030, 3, 1)
    return BloodOrderInput(
        hn="HN-TEST-1",
        an=an,
        reqno=reqno,
        bdvstst=status,
        reqtype="P",
        canceldate=None,
        req_date=day if hour is not None else None,
        req_time=time_of_day(hour),
        bdvst_date=day if visit_hour is not None else None,
        bdvst_time=time_of_day(visit_hour),
        products=products,
        diagnosis_codes=(),
    )


def time_of_day(hour: int | None) -> ParsedTimeOfDay | None:
    return None if hour is None else ParsedTimeOfDay(hour=hour, minute=0, second=0)


def event_after(own: CandidateOrder, hours: float) -> IssueEvent:
    return event((own.placed_at - OWN_PLACED).total_seconds() / 3600 + hours)


def summary(
    own: CandidateOrder,
    events: tuple[IssueEvent, ...],
    candidates: tuple[CandidateOrder, ...],
) -> list[tuple[str, str]]:
    return [
        (link.tag, link.linked.reqno) for link in link_events(own, events, candidates)
    ]


def test_case_21_shape_links_own_then_three_fresh_orders() -> None:
    # Case 21: the own order, a status-1 order before the first issue, then a
    # fresh status-1 order by another doctor before each of the three later
    # issues, each a day or more apart.
    candidates = (
        OWN,
        order("R-PRE", 2.0),
        order("R-F2", 30.0),
        order("R-F3", 55.0),
        order("R-F4", 80.0),
    )
    events = (event(4.6), event(31.0), event(56.5), event(81.0))

    links = link_events(OWN, events, candidates)

    assert [(link.tag, link.linked.reqno) for link in links] == [
        ("own", "R-OWN"),
        ("fresh_order", "R-F2"),
        ("fresh_order", "R-F3"),
        ("fresh_order", "R-F4"),
    ]
    assert [link.event for link in links] == list(events)
    # Event 1 stays on the order the blood bank fulfilled even though a later
    # order sits before its issue; the #276 rule would have picked that one.
    assert links[0].latest_before == order("R-PRE", 2.0)
    assert not any(link.order_after_issue for link in links)


def test_later_event_with_no_new_order_is_standing_on_the_own_order() -> None:
    # The only other order predates event 1, so event 2 is more blood on the
    # standing order, while the #276 rule would hand it to that other order.
    other = order("R-EARLY", 1.0, status="4")
    links = link_events(OWN, (event(3.0), event(20.0)), (OWN, other))

    assert [(link.tag, link.linked.reqno) for link in links] == [
        ("own", "R-OWN"),
        ("standing_order", "R-OWN"),
    ]
    assert [link.latest_before for link in links] == [other, other]


@pytest.mark.parametrize("status", ["6", "7", "11", "12"])
def test_refused_and_cancelled_orders_are_never_candidates(status: str) -> None:
    assert candidate_order(bdvst_row("R-X", status=status)) is None


@pytest.mark.parametrize("status", ["1", "4", "5"])
def test_pending_and_issued_orders_are_candidates(status: str) -> None:
    candidate = candidate_order(bdvst_row("R-X", status=status))

    assert candidate is not None
    assert (candidate.reqno, candidate.status) == ("R-X", status)


def test_cancelled_order_between_issues_leaves_the_event_standing() -> None:
    rows = (bdvst_row("R-OWN", status="4"), bdvst_row("R-CXL", status="7", hour=14))
    candidates = tuple(c for row in rows if (c := candidate_order(row)) is not None)
    own = candidates[0]

    links = link_events(own, (event_after(own, 2), event_after(own, 20)), candidates)

    assert [(link.tag, link.linked.reqno) for link in links] == [
        ("own", "R-OWN"),
        ("standing_order", "R-OWN"),
    ]


def test_own_order_placed_after_the_issue_links_anyway_and_is_flagged() -> None:
    own = order("R-OWN", 5.0, status="4")

    (link,) = link_events(own, (event(2.0),), (own,))

    assert (link.tag, link.linked.reqno) == ("own", "R-OWN")
    assert link.order_after_issue
    assert link.latest_before is None


def test_another_issued_request_can_be_the_fresh_order() -> None:
    other = order("R-OTHER", 10.0, status="4")

    assert summary(OWN, (event(2.0), event(12.0)), (OWN, other)) == [
        ("own", "R-OWN"),
        ("fresh_order", "R-OTHER"),
    ]


def test_fresh_window_is_open_at_the_previous_event_and_closed_at_this_one() -> None:
    events = (event(2.0), event(12.0))

    at_previous_issue = order("R-AT-PREV", 2.0)
    assert summary(OWN, events, (OWN, at_previous_issue))[1] == (
        "standing_order",
        "R-OWN",
    )
    at_this_issue = order("R-AT-THIS", 12.0)
    assert summary(OWN, events, (OWN, at_this_issue))[1] == (
        "fresh_order",
        "R-AT-THIS",
    )
    after_this_issue = order("R-AFTER", 12.5)
    assert summary(OWN, events, (OWN, after_this_issue))[1] == (
        "standing_order",
        "R-OWN",
    )


def test_latest_before_includes_an_order_placed_at_the_issue_time() -> None:
    same_time = order("R-SAME", 2.0)

    (link,) = link_events(OWN, (event(2.0),), (OWN, same_time))

    assert link.latest_before == same_time


def test_latest_of_several_fresh_orders_wins() -> None:
    first, last = order("R-F-A", 5.0), order("R-F-B", 9.0)

    assert summary(OWN, (event(2.0), event(12.0)), (OWN, last, first))[1] == (
        "fresh_order",
        "R-F-B",
    )


def test_fresh_orders_placed_at_the_same_time_break_on_reqno() -> None:
    tied = (order("R-F-B", 9.0), order("R-F-A", 9.0))

    assert summary(OWN, (event(2.0), event(12.0)), (OWN, *tied))[1] == (
        "fresh_order",
        "R-F-B",
    )


def test_orders_on_another_admission_or_family_are_ignored() -> None:
    candidates = (
        OWN,
        order("R-OTHER-AN", 10.0, an="AN-TEST-2"),
        order("R-PLT", 10.0, component="platelet"),
    )

    links = link_events(OWN, (event(2.0), event(12.0)), candidates)

    assert [(link.tag, link.linked.reqno) for link in links] == [
        ("own", "R-OWN"),
        ("standing_order", "R-OWN"),
    ]
    assert [link.latest_before for link in links] == [OWN, OWN]


def test_no_events_link_to_nothing() -> None:
    assert link_events(OWN, (), (OWN,)) == ()


def test_events_out_of_issue_order_fail_loud() -> None:
    with pytest.raises(ValueError, match="issue order"):
        link_events(OWN, (event(12.0), event(2.0)), (OWN,))


def test_candidate_time_is_the_request_time_else_the_visit_time() -> None:
    requested = candidate_order(bdvst_row("R-REQ", hour=9, visit_hour=11))
    visit_only = candidate_order(bdvst_row("R-VISIT", hour=None, visit_hour=11))

    assert requested is not None and visit_only is not None
    assert requested.placed_at == datetime(2030, 3, 1, 9, tzinfo=BANGKOK)
    assert visit_only.placed_at == datetime(2030, 3, 1, 11, tzinfo=BANGKOK)


@pytest.mark.parametrize(
    ("products", "component"),
    [
        (("LDPRC",), "red_cell"),
        (("LDPRC", "LPPC"), "red_cell"),
        (("LPPC",), "platelet"),
    ],
)
def test_candidate_family_follows_the_audit_order_family(
    products: tuple[str, ...], component: str
) -> None:
    candidate = candidate_order(bdvst_row("R-X", products=products))

    assert candidate is not None
    assert candidate.component == component


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(bdvst_row("R-X", products=("FFP",)), id="plasma only"),
        pytest.param(bdvst_row("R-X", products=()), id="no product lines"),
        pytest.param(bdvst_row("R-X", an=None), id="no admission"),
        pytest.param(
            bdvst_row("R-X", hour=None, visit_hour=None), id="no placement time"
        ),
    ],
)
def test_orders_that_cannot_be_placed_on_the_admission_are_not_candidates(
    row: BloodOrderInput,
) -> None:
    assert candidate_order(row) is None
