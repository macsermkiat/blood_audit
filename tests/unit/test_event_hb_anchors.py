"""Unit tests for issue-event Hb anchors and the #249 cutoff (plan PR 4).

An issue event is judged on the labs at the moment its blood left the bank,
whatever order it is linked to. When that order was placed a day or more
before the issue (blood reserved ahead), the LLM bundle widens its Hb window
past the issue; the gap is measured from the linked order, so a fresh order
placed just before a late issue does not widen it.

The 24 h Hb minimum must not reach back past a transfusion that already
corrected the low (#249). The cutoff is the latest earlier red-cell event on
the admission within 24 h that actually transfused a unit: an event whose
units all came back (#264 option A) gave no blood and is skipped.

Fixtures are synthetic: invented HNs, REQNOs, admissions, donor numbers and a
fictional year, shaped like the January 2025 gold cases 68000460 and 68001587.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from bba.audit_orders import AuditOrdersConfig, BloodOrderInput, build_audit_orders
from bba.hb_lookup import prior_transfusion_cutoff, resolve_event_anchors
from bba.ingest.models import ParsedTimeOfDay
from bba.issue_events import (
    AuditEvent,
    IssuedUnit,
    LedgerUnits,
    build_audit_events,
    candidate_order,
)

HN = "HN-TEST-1"
AN = "AN-TEST-1"
OTHER_AN = "AN-TEST-2"
BANGKOK = ZoneInfo("Asia/Bangkok")
ORIGIN = datetime(2030, 3, 1, 8, 0, tzinfo=BANGKOK)
RED_CELL = ("LDPRC",)
PLATELET = ("LDPPC",)


def at(hours: float) -> datetime:
    return ORIGIN + timedelta(hours=hours)


def order(
    reqno: str,
    placed_h: float = 0,
    *,
    an: str = AN,
    products: tuple[str, ...] = RED_CELL,
    status: str = "4",
) -> BloodOrderInput:
    placed = at(placed_h)
    return BloodOrderInput(
        hn=HN,
        an=an,
        reqno=reqno,
        bdvstst=status,
        reqtype="P",
        canceldate=None,
        req_date=placed.date(),
        req_time=ParsedTimeOfDay(hour=placed.hour, minute=placed.minute, second=0),
        bdvst_date=None,
        bdvst_time=None,
        products=products,
        diagnosis_codes=(),
    )


def unit(
    dnrno: str, issue_h: float, status: str = "2", bdtype: str = "LDPRC2"
) -> IssuedUnit:
    return IssuedUnit(
        dnrno=dnrno,
        seqno="1",
        bdtype=bdtype,
        status=status,
        issue_at=at(issue_h),
        returned_at=None,
        given_at=None,
    )


def audit_events(
    audited: dict[BloodOrderInput, Sequence[IssuedUnit]],
    *,
    others: Sequence[BloodOrderInput] = (),
) -> dict[tuple[str, int], AuditEvent]:
    """Audit records keyed by (REQNO, event_seq)."""
    filtered = build_audit_orders(list(audited), AuditOrdersConfig(code_version="t"))
    ledgers = {
        o.reqno: LedgerUnits(
            issued=tuple(sorted(units, key=lambda u: u.issue_order)),
            units_reserved_not_issued=0,
        )
        for o, units in audited.items()
    }
    candidates = [
        c for o in (*audited, *others) if (c := candidate_order(o)) is not None
    ]
    built = build_audit_events(
        filtered, ledgers, candidates, usetypes={}, operations={}
    )
    return {(e.order.reqno, e.event_seq): e for e in built.events}


def cutoff_of(
    events: dict[tuple[str, int], AuditEvent], reqno: str, seq: int
) -> tuple[str, int] | None:
    got = prior_transfusion_cutoff(events.values(), events[(reqno, seq)])
    return None if got is None else (got.order.reqno, got.event_seq)


# --- resolve_event_anchors -------------------------------------------------


def test_labs_and_evidence_are_read_at_the_issue_not_the_order() -> None:
    events = audit_events({order("R-OWN"): [unit("D-1", 4.6)]})

    anchor = resolve_event_anchors(events[("R-OWN", 1)])

    assert anchor.anchor_utc == at(4.6)
    assert anchor.reason == "issue_datetime"
    assert anchor.gap_hours == pytest.approx(4.6)


@pytest.mark.parametrize(
    ("issue_h", "reason"),
    [
        (24 - 1 / 60, "issue_datetime"),
        (24.0, "issue_reanchor"),
        (49.0, "issue_reanchor"),
    ],
)
def test_hb_window_widens_once_the_order_is_a_day_old_at_issue(
    issue_h: float, reason: str
) -> None:
    events = audit_events({order("R-OWN"): [unit("D-1", issue_h)]})

    anchor = resolve_event_anchors(events[("R-OWN", 1)])

    assert anchor.anchor_utc == at(issue_h)
    assert anchor.reason == reason
    assert anchor.gap_hours == pytest.approx(issue_h)


def test_order_gap_is_measured_from_the_linked_fresh_order() -> None:
    # Case 21 event 2: 27 h after the REQNO, 4.2 h after the fresh order.
    events = audit_events(
        {order("R-OWN"): [unit("D-1", 4.6, "3"), unit("D-2", 27.0)]},
        others=[order("R-FRESH", 22.8, status="1")],
    )
    later = events[("R-OWN", 2)]
    assert later.linked_reqno == "R-FRESH"

    anchor = resolve_event_anchors(later)

    assert anchor.anchor_utc == at(27.0)
    assert anchor.reason == "issue_datetime"
    assert anchor.gap_hours == pytest.approx(4.2)


def test_standing_order_event_measures_its_gap_from_the_reqno() -> None:
    events = audit_events({order("R-OWN"): [unit("D-1", 2.0), unit("D-2", 30.0)]})
    later = events[("R-OWN", 2)]
    assert later.linkage_rule == "standing_order"

    anchor = resolve_event_anchors(later)

    assert (anchor.anchor_utc, anchor.reason) == (at(30.0), "issue_reanchor")
    assert anchor.gap_hours == pytest.approx(30.0)


# --- prior_transfusion_cutoff ----------------------------------------------


def test_all_returned_previous_event_is_not_a_transfusion() -> None:
    # 68000460 shape: event 1 came back unused; event 2 has no cutoff.
    events = audit_events({order("R-OWN"): [unit("D-1", 4.7, "3"), unit("D-2", 26.4)]})

    assert cutoff_of(events, "R-OWN", 2) is None


@pytest.mark.parametrize("returned_status", ["3", "7"])
def test_cutoff_skips_the_returned_event_for_the_earlier_transfused_one(
    returned_status: str,
) -> None:
    events = audit_events(
        {
            order("R-OWN"): [
                unit("D-1", 5.0),
                unit("D-2", 15.0, returned_status),
                unit("D-3", 25.0),
            ]
        }
    )

    assert cutoff_of(events, "R-OWN", 3) == ("R-OWN", 1)


def test_returned_prior_does_not_promote_a_transfusion_older_than_24h() -> None:
    # 68001587 shape: event 2 came back, event 1 is 71 h before event 3.
    events = audit_events(
        {
            order("R-OWN"): [
                unit("D-1", 5.4),
                unit("D-2", 59.1, "3"),
                unit("D-3", 76.5),
            ]
        }
    )

    assert cutoff_of(events, "R-OWN", 3) is None
    assert cutoff_of(events, "R-OWN", 1) is None


def test_latest_transfusion_within_24h_is_the_cutoff() -> None:
    events = audit_events(
        {order("R-OWN"): [unit("D-1", 1.0), unit("D-2", 6.0), unit("D-3", 12.0)]}
    )

    assert cutoff_of(events, "R-OWN", 3) == ("R-OWN", 2)


def test_partly_returned_event_still_transfused() -> None:
    events = audit_events(
        {
            order("R-OWN"): [
                unit("D-1", 1.0, "3"),
                unit("D-2", 1.0, "5"),
                unit("D-3", 10.0),
            ]
        }
    )
    assert events[("R-OWN", 1)].issue_units == 2

    assert cutoff_of(events, "R-OWN", 2) == ("R-OWN", 1)


def test_transfusion_under_another_reqno_on_the_admission_counts() -> None:
    events = audit_events(
        {
            order("R-A"): [unit("D-1", 1.0)],
            order("R-B", 8.0): [unit("D-2", 10.0)],
        }
    )

    assert cutoff_of(events, "R-B", 1) == ("R-A", 1)
    assert cutoff_of(events, "R-A", 1) is None


def test_transfusion_on_another_admission_is_ignored() -> None:
    events = audit_events(
        {
            order("R-A", an=OTHER_AN): [unit("D-1", 1.0)],
            order("R-B", 8.0): [unit("D-2", 10.0)],
        }
    )

    assert cutoff_of(events, "R-B", 1) is None


def test_platelet_transfusion_does_not_cut_the_hb_window() -> None:
    events = audit_events(
        {
            order("R-PLT", products=PLATELET): [unit("D-1", 1.0, bdtype="LDPPC5")],
            order("R-RBC", 8.0): [unit("D-2", 10.0)],
        }
    )

    assert cutoff_of(events, "R-RBC", 1) is None


def test_platelet_event_has_no_hb_cutoff() -> None:
    events = audit_events(
        {
            order("R-RBC"): [unit("D-1", 1.0)],
            order("R-PLT", 8.0, products=PLATELET): [
                unit("D-2", 10.0, bdtype="LDPPC5")
            ],
        }
    )

    assert cutoff_of(events, "R-PLT", 1) is None


@pytest.mark.parametrize(
    ("gap_h", "expected"),
    [
        (24.0, ("R-A", 1)),
        (24 + 1 / 60, None),
        (0.0, None),
    ],
)
def test_cutoff_window_is_24h_inclusive_and_excludes_the_same_instant(
    gap_h: float, expected: tuple[str, int] | None
) -> None:
    events = audit_events(
        {
            order("R-A"): [unit("D-1", 1.0)],
            order("R-B", 0.5): [unit("D-2", 1.0 + gap_h)],
        }
    )

    assert cutoff_of(events, "R-B", 1) == expected


def test_simultaneous_transfusions_break_the_tie_on_reqno() -> None:
    events = audit_events(
        {
            order("R-A"): [unit("D-1", 1.0)],
            order("R-B"): [unit("D-2", 1.0)],
            order("R-C", 2.0): [unit("D-3", 5.0)],
        }
    )

    assert cutoff_of(events, "R-C", 1) == ("R-B", 1)
