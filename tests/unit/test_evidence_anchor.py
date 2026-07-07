"""Contract tests for ``bba.hb_lookup.resolve_evidence_anchor``.

The evidence anchor is the point all per-source evidence windows (Hb
lookback, notes, CBC, meds, vitals) are computed relative to. For most
orders it is the order REQTIME. But blood reserved for elective surgery is
crossmatched days before it is issued/transfused (a "type & crossmatch,
hold" order); for those, anchoring the evidence on the reservation date
misses the entire transfusion context — the op-day Hb drop and operative
notes that justify (or not) the transfusion.

These tests pin the policy: re-anchor onto the *issue* datetime (PICK /
USE) when it lands a configurable threshold (default 24h) or more after
the order, and only the issue datetime — the blood-bank *visit* timestamp
tracks the reservation, not the transfusion, so it must never re-anchor.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bba.hb_lookup import AnchorCandidate, resolve_evidence_anchor

UTC = timezone.utc
# A reservation REQTIME. Issue/transfusion lands days later for the
# elective pre-reserved cohort this feature targets.
ORDER = datetime(2025, 1, 14, 19, 9, 31, tzinfo=UTC)


def _candidate(
    *, delta: timedelta, reason: str, display: str = "disp"
) -> AnchorCandidate:
    return AnchorCandidate(anchor_utc=ORDER + delta, display=display, reason=reason)


class TestReanchorsOnLateIssue:
    """An issue datetime >= threshold after the order moves the anchor."""

    def test_issue_5_days_later_reanchors(self) -> None:
        issue = _candidate(
            delta=timedelta(days=5, hours=12),
            reason="issue_datetime",
            display="2025-01-20 08:00",
        )
        ev = resolve_evidence_anchor(order_datetime=ORDER, candidates=[issue])
        assert ev.reason == "transfusion_reanchor"
        assert ev.anchor_utc == issue.anchor_utc
        assert ev.gap_hours == 132.0
        assert ev.display == "2025-01-20 08:00"

    def test_exactly_threshold_reanchors(self) -> None:
        issue = _candidate(delta=timedelta(hours=24), reason="issue_datetime")
        ev = resolve_evidence_anchor(order_datetime=ORDER, candidates=[issue])
        assert ev.reason == "transfusion_reanchor"
        assert ev.gap_hours == 24.0


class TestKeepsOrderAnchor:
    """Same-day and near-order issues keep the order anchor unchanged."""

    def test_issue_under_threshold_keeps_order(self) -> None:
        issue = _candidate(
            delta=timedelta(hours=23, minutes=59), reason="issue_datetime"
        )
        ev = resolve_evidence_anchor(order_datetime=ORDER, candidates=[issue])
        assert ev.reason == "order_datetime"
        assert ev.anchor_utc == ORDER
        assert ev.gap_hours == 0.0
        assert ev.display == ""

    def test_no_candidates_keeps_order(self) -> None:
        ev = resolve_evidence_anchor(order_datetime=ORDER, candidates=[])
        assert ev.reason == "order_datetime"
        assert ev.anchor_utc == ORDER

    def test_issue_before_order_keeps_order(self) -> None:
        # A negative gap (issue recorded before the order) must never
        # re-anchor backwards onto a stale window.
        issue = _candidate(delta=timedelta(hours=-30), reason="issue_datetime")
        ev = resolve_evidence_anchor(order_datetime=ORDER, candidates=[issue])
        assert ev.reason == "order_datetime"
        assert ev.anchor_utc == ORDER


class TestOnlyIssueDatetimeReanchors:
    """The blood-bank visit tracks the reservation, never the transfusion."""

    def test_blood_bank_fallback_does_not_reanchor(self) -> None:
        # Even a far-future blood-bank visit timestamp must not re-anchor:
        # only PICK/USE (issue_datetime) marks the transfusion event.
        bank = _candidate(delta=timedelta(days=5), reason="blood_bank_visit_fallback")
        ev = resolve_evidence_anchor(order_datetime=ORDER, candidates=[bank])
        assert ev.reason == "order_datetime"
        assert ev.anchor_utc == ORDER

    def test_issue_preferred_over_blood_bank(self) -> None:
        issue = _candidate(
            delta=timedelta(days=5), reason="issue_datetime", display="issue"
        )
        bank = _candidate(
            delta=timedelta(days=6), reason="blood_bank_visit_fallback", display="bank"
        )
        ev = resolve_evidence_anchor(order_datetime=ORDER, candidates=[issue, bank])
        assert ev.reason == "transfusion_reanchor"
        assert ev.display == "issue"


class TestCustomThreshold:
    """The re-anchor threshold is configurable."""

    def test_custom_threshold_respected(self) -> None:
        issue = _candidate(delta=timedelta(hours=30), reason="issue_datetime")
        ev = resolve_evidence_anchor(
            order_datetime=ORDER,
            candidates=[issue],
            threshold=timedelta(hours=48),
        )
        assert ev.reason == "order_datetime"
