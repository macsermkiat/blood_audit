"""Contract tests for ``bba.hb_lookup.resolve_lab_anchor``.

Clinician review 2026-11-09: the Hb or platelet count an order is judged on
is the latest one before the blood was dispensed (วันจ่าย), not before it was
requested (วันจอง). A draw taken between request and dispense is what the
doctor saw when the blood left the bank, so it must decide the order.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bba.hb_lookup import (
    AnchorCandidate,
    HbObservation,
    resolve_hb_with_fallback,
    resolve_lab_anchor,
)

UTC = timezone.utc
REQUEST = datetime(2025, 4, 23, 5, 56, tzinfo=UTC)


def _candidate(delta: timedelta, reason: str) -> AnchorCandidate:
    return AnchorCandidate(anchor_utc=REQUEST + delta, display="disp", reason=reason)


def test_dispense_after_request_becomes_the_lab_anchor() -> None:
    dispense = _candidate(timedelta(hours=2, minutes=18), "issue_datetime")
    anchor = resolve_lab_anchor(order_datetime=REQUEST, candidates=[dispense])
    assert anchor == dispense


def test_no_dispense_record_keeps_the_request_time() -> None:
    bank = _candidate(timedelta(minutes=4), "blood_bank_visit_fallback")
    anchor = resolve_lab_anchor(order_datetime=REQUEST, candidates=[bank])
    assert anchor.anchor_utc == REQUEST
    assert anchor.reason == "order_datetime"


def test_dispense_recorded_before_request_is_ignored() -> None:
    stale = _candidate(timedelta(hours=-3), "issue_datetime")
    anchor = resolve_lab_anchor(order_datetime=REQUEST, candidates=[stale])
    assert anchor.anchor_utc == REQUEST


def test_hb_drawn_between_request_and_dispense_decides_the_order() -> None:
    at_request = HbObservation(
        value_g_dl=8.2,
        datetime_utc=REQUEST - timedelta(minutes=3),
        source="HEMATOLOGY",
        item_no=1,
    )
    before_dispense = HbObservation(
        value_g_dl=6.8,
        datetime_utc=REQUEST + timedelta(hours=1),
        source="HEMATOLOGY",
        item_no=2,
    )
    dispense = _candidate(timedelta(hours=2), "issue_datetime")
    lab_anchor = resolve_lab_anchor(order_datetime=REQUEST, candidates=[dispense])

    hb, _, _ = resolve_hb_with_fallback(
        observations=[at_request, before_dispense],
        order_datetime=lab_anchor.anchor_utc,
        candidates=[dispense],
    )

    assert hb.value_g_dl == 6.8
