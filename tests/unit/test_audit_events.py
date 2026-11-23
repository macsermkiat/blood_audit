"""Unit tests for per-event audit records (blood_audit #276, plan PR 3).

Each issue of blood becomes its own audit record: its own audit_id (so the
audit store keeps one judgment per event), the order it is judged against,
and whether the order's declared surgical use exempts it. A REQNO that is
excluded, or that issued nothing, keeps one REQNO-level record under its
legacy audit_id so the v2 listing and joins survive.

Fixtures are synthetic: invented HNs, REQNOs, admissions, donor numbers and a
fictional year, shaped like the January 2025 gold cases.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import TypeAdapter

from bba.audit_orders import (
    AuditOrdersConfig,
    BloodOrderInput,
    build_audit_id,
    build_audit_orders,
    build_event_audit_id,
)
from bba.audit_store.models import SafeId
from bba.ingest.models import ParsedTimeOfDay
from bba.issue_events import (
    AuditEvents,
    IssuedUnit,
    LedgerUnits,
    build_audit_events,
    candidate_order,
)

HN = "HN-TEST-1"
AN = "AN-TEST-1"
BANGKOK = ZoneInfo("Asia/Bangkok")
DAY = date(2030, 3, 1)
OWN_PLACED = datetime(2030, 3, 1, 8, 0, tzinfo=BANGKOK)


def at(hours: float) -> datetime:
    return OWN_PLACED + timedelta(hours=hours)


def bdvst(
    reqno: str,
    placed_h: float = 0,
    *,
    status: str = "4",
    an: str | None = AN,
    products: tuple[str, ...] = ("LDPRC",),
    diagnosis: tuple[str, ...] = (),
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
        diagnosis_codes=diagnosis,
    )


def unit(
    dnrno: str,
    issue_h: float,
    *,
    status: str = "2",
    bdtype: str = "LDPRC2",
    seqno: str = "1",
) -> IssuedUnit:
    return IssuedUnit(
        dnrno=dnrno,
        seqno=seqno,
        bdtype=bdtype,
        status=status,
        issue_at=at(issue_h),
        returned_at=None,
        given_at=None,
    )


def ledger(*units: IssuedUnit, reserved: int = 0) -> LedgerUnits:
    return LedgerUnits(
        issued=tuple(sorted(units, key=lambda u: u.issue_order)),
        units_reserved_not_issued=reserved,
    )


def build(
    audited: Sequence[BloodOrderInput],
    ledgers: Mapping[str, LedgerUnits],
    *,
    others: Sequence[BloodOrderInput] = (),
    usetypes: Mapping[str, str | None] | None = None,
    operations: Mapping[str, Sequence[datetime]] | None = None,
) -> AuditEvents:
    """Run audit_orders over ``audited`` and build events against every order."""
    orders = build_audit_orders(audited, AuditOrdersConfig(code_version="test"))
    candidates = [
        c for r in (*audited, *others) if (c := candidate_order(r)) is not None
    ]
    return build_audit_events(
        orders,
        ledgers,
        candidates,
        usetypes=usetypes or {},
        operations=operations or {},
    )


# Case 21 shape: four separate issues under R-OWN, each later one after a
# fresh pending order placed in the gap. Event 1 has two units pulled at the
# same minute, so the donor number decides which unit fixes its audit_id.
CASE_21_FRESH = (bdvst("R-F2", 20, status="1"), bdvst("R-F3", 40, status="1"))
CASE_21_LEDGER = ledger(
    unit("D-B", 4.6, status="3"),
    unit("D-A", 4.6, status="3"),
    unit("D-C", 27.0),
    unit("D-D", 47.4),
    unit("D-E", 80.0),
    reserved=1,
)


def case_21() -> AuditEvents:
    return build(
        [bdvst("R-OWN")],
        {"R-OWN": CASE_21_LEDGER},
        others=CASE_21_FRESH,
    )


def test_each_issue_event_is_its_own_audit_record() -> None:
    result = case_21()

    assert [e.event_seq for e in result.events] == [1, 2, 3, 4]
    assert {e.events_in_reqno for e in result.events} == {4}
    assert result.excluded == ()
    assert [e.issue_units for e in result.events] == [2, 1, 1, 1]
    first = result.events[0]
    assert first.issue_datetime == at(4.6)
    assert first.issue_units_returned == 2
    assert first.issue_unit_statuses == ("3", "3")
    assert first.units_reserved_not_issued == 1


def test_event_audit_ids_are_disjoint_within_a_reqno() -> None:
    ids = [e.audit_id for e in case_21().events]

    assert len(set(ids)) == 4
    assert build_audit_id(HN, "R-OWN") not in ids


def test_event_audit_id_is_keyed_on_the_first_unit_after_the_tie_break() -> None:
    first = case_21().events[0]

    # D-A and D-B left the bank at the same minute; D-A sorts first.
    assert first.audit_id == build_event_audit_id(HN, "R-OWN", ("D-A", "1", "LDPRC2"))


def test_event_audit_id_formula_is_fixed() -> None:
    # Signed runs bind audit_ids (plan 4.1): the formula itself is the contract.
    payload = b"HN-TEST-1:R-OWN:D-A:1:LDPRC2"

    assert (
        build_event_audit_id(HN, "R-OWN", ("D-A", "1", "LDPRC2"))
        == (hashlib.sha256(payload).hexdigest()[:32])
    )


def test_every_event_carries_the_legacy_order_id_as_its_parent() -> None:
    parents = {e.parent_audit_id for e in case_21().events}

    assert parents == {build_audit_id(HN, "R-OWN")}


def test_event_audit_ids_are_stable_across_runs() -> None:
    assert [e.audit_id for e in case_21().events] == [
        e.audit_id for e in case_21().events
    ]


def test_event_audit_id_satisfies_the_audit_store_safe_id() -> None:
    adapter = TypeAdapter(SafeId)

    for event in case_21().events:
        assert adapter.validate_python(event.audit_id) == event.audit_id


@pytest.mark.parametrize(
    "other",
    [
        ("HN-TEST-2", "R-OWN", ("D-A", "1", "LDPRC2")),
        (HN, "R-OTHER", ("D-A", "1", "LDPRC2")),
        (HN, "R-OWN", ("D-Z", "1", "LDPRC2")),
        (HN, "R-OWN", ("D-A", "2", "LDPRC2")),
        (HN, "R-OWN", ("D-A", "1", "LDPRC2I")),
    ],
)
def test_event_audit_id_changes_with_every_identity_part(
    other: tuple[str, str, tuple[str, str, str]],
) -> None:
    base = build_event_audit_id(HN, "R-OWN", ("D-A", "1", "LDPRC2"))

    assert build_event_audit_id(*other) != base


def test_events_link_to_the_order_that_caused_them() -> None:
    events = case_21().events

    assert [e.linkage_rule for e in events] == [
        "own",
        "fresh_order",
        "fresh_order",
        "standing_order",
    ]
    assert [e.linked_reqno for e in events] == ["R-OWN", "R-F2", "R-F3", "R-OWN"]
    assert [e.linked_order_status for e in events] == ["4", "1", "1", "4"]
    assert events[1].linked_order_datetime == at(20)
    assert events[1].order_to_issue_hours == pytest.approx(7.0)
    assert [e.linked_reqno_latest_before for e in events] == [
        "R-OWN",
        "R-F2",
        "R-F3",
        "R-F3",
    ]


def test_reqno_with_no_issued_unit_is_one_no_issue_record() -> None:
    result = build([bdvst("R-NONE")], {"R-NONE": ledger(reserved=2)})

    assert result.events == ()
    (record,) = result.excluded
    assert record.reason == "no_issue_record"
    assert record.audit_id == record.parent_audit_id == build_audit_id(HN, "R-NONE")
    assert record.event_seq is None
    assert record.events_in_reqno == 0
    assert record.units_reserved_not_issued == 2


def test_reqno_excluded_by_audit_orders_keeps_one_record_with_its_measured_events() -> (
    None
):
    excluded = bdvst("R-THAL", diagnosis=("D561",))
    result = build(
        [excluded],
        {"R-THAL": ledger(unit("D-1", 1), unit("D-2", 30))},
    )

    assert result.events == ()
    (record,) = result.excluded
    assert (record.reason, record.detail) == ("hemoglobinopathy", "D561")
    assert record.audit_id == record.parent_audit_id == build_audit_id(HN, "R-THAL")
    assert record.event_seq is None
    assert record.events_in_reqno == 2


def test_excluded_reqno_that_cannot_be_linked_counts_no_events() -> None:
    # No admission: its ledger cannot be scoped, so no event can be measured.
    result = build([bdvst("R-OPD", an=None)], {})

    (record,) = result.excluded
    assert (record.reason, record.events_in_reqno) == ("no_an", 0)


def test_every_audited_reqno_is_either_events_or_one_excluded_record() -> None:
    audited = [
        bdvst("R-OWN"),
        bdvst("R-NONE", 1),
        bdvst("R-THAL", 2, diagnosis=("D561",)),
    ]
    result = build(
        audited,
        {
            "R-OWN": CASE_21_LEDGER,
            "R-NONE": ledger(),
            "R-THAL": ledger(unit("D-9", 3)),
        },
        others=CASE_21_FRESH,
    )

    event_reqnos = {e.order.reqno for e in result.events}
    excluded_reqnos = [r.reqno for r in result.excluded]
    assert event_reqnos == {"R-OWN"}
    assert sorted(excluded_reqnos) == ["R-NONE", "R-THAL"]


@pytest.mark.parametrize(
    ("component", "products", "bdtype", "mismatch"),
    [
        ("red_cell", ("LDPRC",), "LDPRC2I", False),
        ("red_cell", ("LPRC",), "SDRF2", False),
        ("red_cell", ("LDPRC",), "LDPPC5", True),
        ("red_cell", ("LDPRC",), "FFP", True),
        ("platelet", ("LDPPC",), "SDPFI4", False),
        ("platelet", ("LDPPC",), "LDPPC5I", False),
        ("platelet", ("LDPPC",), "LDPRC4", True),
    ],
)
def test_unit_from_another_product_family_is_flagged_and_kept(
    component: str, products: tuple[str, ...], bdtype: str, mismatch: bool
) -> None:
    ordered = bdvst("R-OWN", products=products)
    family_unit = "LDPRC2" if component == "red_cell" else "LDPPC5"
    result = build(
        [ordered],
        {
            "R-OWN": ledger(
                unit("D-1", 1, bdtype=family_unit), unit("D-2", 1, bdtype=bdtype)
            )
        },
    )

    (event,) = result.events
    assert event.order.component == component
    assert event.issue_units == 2
    assert event.unit_family_mismatch is mismatch


# Declared use (owner decision 8.15 (c), amended 2026-11-22). R-OWN issues at
# 1 h; a fresh order R-F placed at 10 h causes the issue at 20 h. The periop
# window of that issue runs from 14 h to 92 h.
FRESH_PLACED_H = 10.0
FRESH_ISSUE_H = 20.0


def declared(
    *,
    own_usetype: str | None = "1",
    fresh_usetype: str | None = "2",
    operations_h: Sequence[float] = (),
    fresh_placed_h: float = FRESH_PLACED_H,
    operations_an: str = AN,
) -> AuditEvents:
    return build(
        [bdvst("R-OWN")],
        {"R-OWN": ledger(unit("D-1", 1), unit("D-2", FRESH_ISSUE_H))},
        others=[bdvst("R-F", fresh_placed_h, status="1")],
        usetypes={"R-OWN": own_usetype, "R-F": fresh_usetype},
        operations={operations_an: [at(h) for h in operations_h]},
    )


@pytest.mark.parametrize(
    ("usetype", "applies"),
    [("2", True), ("3", True), ("1", False), ("5", False), (None, False)],
)
def test_first_event_applies_its_own_declared_surgical_use(
    usetype: str | None, applies: bool
) -> None:
    first = declared(own_usetype=usetype).events[0]

    assert first.linkage_rule == "own"
    assert first.linked_usetype == usetype
    assert first.declared_use_applies is applies


def test_first_event_needs_no_operation_record() -> None:
    first = declared(own_usetype="2", operations_h=()).events[0]

    assert first.declared_use_applies is True


def test_missing_usetype_means_no_declaration() -> None:
    result = build([bdvst("R-OWN")], {"R-OWN": ledger(unit("D-1", 1))})

    (event,) = result.events
    assert (event.linked_usetype, event.declared_use_applies) == (None, False)


def test_standing_order_event_never_inherits_the_declaration() -> None:
    result = build(
        [bdvst("R-OWN")],
        {"R-OWN": ledger(unit("D-1", 1), unit("D-2", 30))},
        usetypes={"R-OWN": "2"},
        operations={AN: [at(31)]},
    )

    first, later = result.events
    assert later.linkage_rule == "standing_order"
    assert first.declared_use_applies is True
    assert later.declared_use_applies is False


def test_fresh_declaring_order_with_an_operation_in_the_window_is_exempt() -> None:
    later = declared(fresh_usetype="3", operations_h=(30,)).events[1]

    assert later.linkage_rule == "fresh_order"
    assert later.linked_usetype == "3"
    assert later.declared_use_applies is True


@pytest.mark.parametrize("usetype", ["2", "3"])
def test_fresh_declaring_order_without_an_operation_is_judged_on_hb(
    usetype: str,
) -> None:
    later = declared(fresh_usetype=usetype, operations_h=()).events[1]

    assert later.declared_use_applies is False


def test_fresh_ward_order_with_an_operation_is_not_exempt() -> None:
    later = declared(fresh_usetype="1", operations_h=(30,)).events[1]

    assert later.declared_use_applies is False


@pytest.mark.parametrize(
    ("operation_h", "applies"),
    [
        (FRESH_ISSUE_H - 6, True),
        (FRESH_ISSUE_H + 72, True),
        (FRESH_ISSUE_H - 6 - 1 / 3600, False),
        (FRESH_ISSUE_H + 72 + 1 / 3600, False),
    ],
)
def test_operation_window_runs_from_6h_before_to_72h_after_the_issue(
    operation_h: float, applies: bool
) -> None:
    later = declared(operations_h=(operation_h,)).events[1]

    assert later.declared_use_applies is applies


def test_operation_must_follow_the_fresh_order() -> None:
    # Placed at 18 h, the order comes after an operation at 15 h that is
    # inside the issue's window: that operation did not follow the order.
    later = declared(fresh_placed_h=18, operations_h=(15,)).events[1]

    assert later.linkage_rule == "fresh_order"
    assert later.declared_use_applies is False


def test_operation_on_another_admission_does_not_count() -> None:
    later = declared(operations_h=(30,), operations_an="AN-TEST-2").events[1]

    assert later.declared_use_applies is False


def test_included_order_without_its_ledger_fails_loud() -> None:
    with pytest.raises(ValueError, match="ledger"):
        build([bdvst("R-OWN")], {})


def test_excluded_candidate_without_its_ledger_fails_loud() -> None:
    with pytest.raises(ValueError, match="ledger"):
        build([bdvst("R-THAL", diagnosis=("D561",))], {})


def test_naive_operation_times_fail_loud() -> None:
    with pytest.raises(ValueError, match="timezone"):
        build(
            [bdvst("R-OWN")],
            {"R-OWN": ledger(unit("D-1", 1))},
            operations={AN: [datetime(2030, 3, 2, 8, 0)]},
        )


def test_included_order_missing_from_the_candidates_fails_loud() -> None:
    orders = build_audit_orders([bdvst("R-OWN")], AuditOrdersConfig(code_version="t"))

    with pytest.raises(ValueError, match="candidate"):
        build_audit_events(orders, {"R-OWN": ledger()}, [], usetypes={}, operations={})


def test_candidate_that_disagrees_with_its_audit_order_fails_loud() -> None:
    orders = build_audit_orders([bdvst("R-OWN")], AuditOrdersConfig(code_version="t"))
    moved = candidate_order(bdvst("R-OWN", 3))
    assert moved is not None

    with pytest.raises(ValueError, match="disagrees"):
        build_audit_events(
            orders, {"R-OWN": ledger()}, [moved], usetypes={}, operations={}
        )


def test_two_candidates_for_one_reqno_fail_loud() -> None:
    orders = build_audit_orders([bdvst("R-OWN")], AuditOrdersConfig(code_version="t"))
    candidates = [candidate_order(bdvst("R-OWN")), candidate_order(bdvst("R-OWN", 3))]

    with pytest.raises(ValueError, match="candidates"):
        build_audit_events(
            orders,
            {"R-OWN": ledger()},
            [c for c in candidates if c is not None],
            usetypes={},
            operations={},
        )


def test_colliding_event_audit_ids_fail_loud() -> None:
    # Two keyless units (a malformed ledger line each) open two events of one
    # REQNO; the same first-unit key would give both events one audit_id.
    keyless = ledger(unit("", 1, seqno=""), unit("", 30, seqno=""))

    with pytest.raises(ValueError, match="audit_id"):
        build([bdvst("R-OWN")], {"R-OWN": keyless})
