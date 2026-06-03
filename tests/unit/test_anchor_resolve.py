"""Contract tests for ``bba.hb_lookup.resolve_hb_with_fallback``.

The resolver is the single anchor-resolution policy shared by the
deterministic report and the LLM gate (see
``docs/handoff-hb-anchor-unification.md``). These tests pin the contract
the divergence bug violated, not the implementation: order-time wins when
it has a value, the fallback ladder is tried issue-before-blood-bank, a
pre-order candidate is never used (it could surface a stale Hb the order
window correctly excluded), and an all-miss stays missing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bba.hb_lookup import AnchorCandidate, resolve_hb_with_fallback
from bba.hb_lookup.models import HbObservation

UTC = timezone.utc
# Case 7's order REQTIME, in UTC. Hb draws landed minutes *after* it.
ORDER = datetime(2025, 10, 27, 14, 43, 37, tzinfo=UTC)


def _obs(
    *, delta: timedelta, value: float, source: str = "HEMATOLOGY", item_no: int = 1
) -> HbObservation:
    return HbObservation(
        value_g_dl=value,
        datetime_utc=ORDER + delta,
        source=source,  # type: ignore[arg-type]
        item_no=item_no,
    )


def _candidate(
    *, delta: timedelta, reason: str, display: str = "disp"
) -> AnchorCandidate:
    return AnchorCandidate(anchor_utc=ORDER + delta, display=display, reason=reason)


class TestOrderTimeWins:
    """A non-missing order-time Hb short-circuits the fallback ladder."""

    def test_order_time_hit_ignores_candidates(self) -> None:
        obs = [_obs(delta=timedelta(hours=-1), value=9.0)]
        candidate = _candidate(
            delta=timedelta(minutes=5), reason="blood_bank_visit_fallback"
        )
        hb, display, reason = resolve_hb_with_fallback(
            observations=obs, order_datetime=ORDER, candidates=[candidate]
        )
        assert hb.value_g_dl == 9.0
        assert reason == "order_datetime"
        assert display == ""


class TestFallbackLadderOrder:
    """When order-time misses, issue_datetime is tried before blood-bank."""

    def test_issue_candidate_wins_over_blood_bank(self) -> None:
        # Both candidates resolve to a value, but to *different* draws.
        # Issue (earlier anchor) sees only the 10.0; blood-bank (later
        # anchor) would see the 8.0. Issue must win -> 10.0.
        obs = [
            _obs(delta=timedelta(minutes=3), value=10.0, item_no=1),
            _obs(delta=timedelta(minutes=6), value=8.0, item_no=2),
        ]
        candidates = [
            _candidate(delta=timedelta(minutes=4), reason="issue_datetime"),
            _candidate(delta=timedelta(minutes=10), reason="blood_bank_visit_fallback"),
        ]
        hb, display, reason = resolve_hb_with_fallback(
            observations=obs, order_datetime=ORDER, candidates=candidates
        )
        assert hb.value_g_dl == 10.0
        assert reason == "issue_datetime"


class TestCase7BloodBankFallback:
    """REQNO 68066907: order-time misses, blood-bank fallback finds 10.0."""

    def test_blood_bank_fallback_surfaces_post_order_hb(self) -> None:
        obs = [_obs(delta=timedelta(minutes=3), value=10.0)]
        candidate = _candidate(
            delta=timedelta(minutes=4),
            reason="blood_bank_visit_fallback",
            display="2025-10-27 14:47:27",
        )
        hb, display, reason = resolve_hb_with_fallback(
            observations=obs, order_datetime=ORDER, candidates=[candidate]
        )
        assert hb.value_g_dl == 10.0
        assert reason == "blood_bank_visit_fallback"
        assert display == "2025-10-27 14:47:27"


class TestPreOrderCandidateSkipped:
    """A candidate before the order must not surface a stale, out-of-window Hb."""

    def test_pre_order_candidate_is_skipped(self) -> None:
        # The only draw is 8 days before the order -> outside the order's
        # 7-day window, so order-time correctly misses. A pre-order
        # candidate 2 days back would have a window that *includes* that
        # stale draw; the >= order_datetime guard must skip it.
        obs = [_obs(delta=timedelta(days=-8), value=9.0)]
        candidate = _candidate(delta=timedelta(days=-2), reason="issue_datetime")
        hb, display, reason = resolve_hb_with_fallback(
            observations=obs, order_datetime=ORDER, candidates=[candidate]
        )
        assert hb.value_g_dl is None
        assert reason == "order_datetime"
        assert display == ""


class TestAllMiss:
    """No observations -> missing result, regardless of candidates."""

    def test_all_miss_returns_missing(self) -> None:
        candidate = _candidate(delta=timedelta(minutes=5), reason="issue_datetime")
        hb, display, reason = resolve_hb_with_fallback(
            observations=[], order_datetime=ORDER, candidates=[candidate]
        )
        assert hb.value_g_dl is None
        assert hb.freshness == "missing"
        assert reason == "order_datetime"
        assert display == ""
