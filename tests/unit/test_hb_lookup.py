"""RED-phase failing tests for issue #5 (bba.hb_lookup).

Each ``class`` maps to one acceptance criterion in the issue body. Tests
assert contracts (the WHY), not implementation choices — per PRD
§"Testing Decisions".

No implementation exists yet; every test MUST fail in this scaffold commit
(``NotImplementedError`` from the entry-point stubs).

Constants chosen for human-readable fixtures:

* Anchor: 2026-05-15 12:00:00 UTC. Bounded math from this point gives
  whole-hour offsets without daylight-saving artefacts (Asia/Bangkok has
  no DST, but UTC arithmetic is simpler).
* Hb values picked to land cleanly inside [2.0, 25.0] and to hit the
  delta-Hb thresholds exactly at the boundary.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from bba.hb_lookup.lookup import lookup_hb
from bba.hb_lookup.models import (
    DeltaHbWindow,
    HbLookupResult,
    HbObservation,
)
from bba.hb_lookup.parse import parse_hb_value


ANCHOR = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)


def _obs(
    *,
    offset_hours: float,
    value: float,
    source: str = "HEMATOLOGY",
    item_no: int = 1,
) -> HbObservation:
    """Construct an HbObservation at ``ANCHOR - offset_hours``.

    Positive ``offset_hours`` means earlier than the anchor (the usual case
    for "look back" tests). Negative would be after the anchor.
    """
    return HbObservation(
        value_g_dl=value,
        datetime_utc=ANCHOR - timedelta(hours=offset_hours),
        source=source,  # type: ignore[arg-type]
        item_no=item_no,
    )


# =============================================================================
# AC: Numeric-range validation rejects out-of-bound and non-numeric RESULT strings
# =============================================================================


class TestParseHbValueHappyPath:
    """Allow-list: numeric strings in [2.0, 25.0] parse cleanly to a float."""

    def test_typical_value(self) -> None:
        assert parse_hb_value("13.5") == 13.5

    def test_integer_string(self) -> None:
        assert parse_hb_value("10") == 10.0

    def test_lower_bound_inclusive(self) -> None:
        # 2.0 g/dL is the lower analytic-validity bound; included.
        assert parse_hb_value("2.0") == 2.0

    def test_upper_bound_inclusive(self) -> None:
        # 25.0 g/dL is the upper analytic-validity bound; included.
        assert parse_hb_value("25.0") == 25.0

    def test_whitespace_tolerated(self) -> None:
        # HOSxP exports sometimes pad numeric columns with spaces.
        assert parse_hb_value("  13.5  ") == 13.5


class TestParseHbValueRejection:
    """Reject everything that is not an in-range numeric — return ``None``."""

    def test_below_lower_bound(self) -> None:
        # 1.9 < 2.0; non-physiological, almost certainly a transcription error.
        assert parse_hb_value("1.9") is None

    def test_above_upper_bound(self) -> None:
        # 25.1 > 25.0; non-physiological even for polycythaemia.
        assert parse_hb_value("25.1") is None

    def test_zero(self) -> None:
        # The "no result entered" sentinel some HOSxP exporters use.
        assert parse_hb_value("0") is None

    def test_negative(self) -> None:
        assert parse_hb_value("-1") is None

    def test_non_numeric(self) -> None:
        assert parse_hb_value("abc") is None

    def test_unit_suffix_rejected(self) -> None:
        # "13.5 g/dL" with unit means upstream contamination; refuse.
        assert parse_hb_value("13.5 g/dL") is None

    def test_empty_string(self) -> None:
        assert parse_hb_value("") is None

    def test_whitespace_only(self) -> None:
        assert parse_hb_value("   ") is None

    def test_none_input(self) -> None:
        assert parse_hb_value(None) is None

    def test_nan_rejected(self) -> None:
        # float("nan") parses but is not a valid clinical value — reject.
        assert parse_hb_value("nan") is None
        assert parse_hb_value("NaN") is None

    def test_infinity_rejected(self) -> None:
        # float("inf") parses but is non-physiological.
        assert parse_hb_value("inf") is None
        assert parse_hb_value("-inf") is None


class TestParseHbValueProperty:
    """Property invariants — encode the WHY of numeric validation."""

    @given(value=st.floats(min_value=2.0, max_value=25.0, allow_nan=False))
    @settings(max_examples=200)
    def test_any_in_range_value_round_trips(self, value: float) -> None:
        # Every value in [2.0, 25.0] must be accepted; reject is a regression.
        parsed = parse_hb_value(str(value))
        assert parsed is not None, f"in-range value {value!r} rejected"
        assert parsed == pytest.approx(value)

    @given(
        value=st.floats(allow_nan=False, allow_infinity=False).filter(
            lambda v: v < 2.0 or v > 25.0
        )
    )
    @settings(max_examples=200)
    def test_any_out_of_range_value_is_rejected(self, value: float) -> None:
        # The whole point of the range check: catch transcription errors
        # before they reach the deterministic classifier.
        assert parse_hb_value(str(value)) is None, (
            f"out-of-range value {value!r} silently accepted"
        )

    @given(raw=st.text(max_size=20).filter(lambda s: not _looks_like_number(s)))
    @settings(max_examples=300)
    def test_non_numeric_strings_always_return_none(self, raw: str) -> None:
        # Strict-loud: anything that is not a recognisable number is None.
        assert parse_hb_value(raw) is None, (
            f"non-numeric string {raw!r} silently parsed to a value"
        )


def _looks_like_number(s: str) -> bool:
    """Crude check used only to filter hypothesis strategies."""
    try:
        float(s.strip())
        return True
    except (ValueError, TypeError):
        return False


# =============================================================================
# AC: Freshness tier boundary tests (<24h / 24-72h / 72h-7d / >7d missing)
# =============================================================================


class TestFreshnessTiers:
    """Freshness is measured from the order anchor backwards; the tier
    drives the deterministic classifier's evidence-window decision."""

    def test_under_24h_is_fresh(self) -> None:
        result = lookup_hb(
            observations=[_obs(offset_hours=12, value=10.0)],
            anchor_utc=ANCHOR,
        )
        assert result.freshness == "fresh"

    def test_exact_24h_is_stale_24_72h(self) -> None:
        # Boundary semantics: tier is [24h, 72h) — 24h exactly belongs to
        # the stale tier, not to "fresh".
        result = lookup_hb(
            observations=[_obs(offset_hours=24, value=10.0)],
            anchor_utc=ANCHOR,
        )
        assert result.freshness == "stale_24_72h"

    def test_inside_24_72h_window_is_stale_24_72h(self) -> None:
        result = lookup_hb(
            observations=[_obs(offset_hours=48, value=10.0)],
            anchor_utc=ANCHOR,
        )
        assert result.freshness == "stale_24_72h"

    def test_exact_72h_is_stale_3_7d(self) -> None:
        result = lookup_hb(
            observations=[_obs(offset_hours=72, value=10.0)],
            anchor_utc=ANCHOR,
        )
        assert result.freshness == "stale_3_7d"

    def test_inside_3_7d_window_is_stale_3_7d(self) -> None:
        # 5 days = 120 hours, within [72h, 7d) = [72h, 168h).
        result = lookup_hb(
            observations=[_obs(offset_hours=120, value=10.0)],
            anchor_utc=ANCHOR,
        )
        assert result.freshness == "stale_3_7d"

    def test_exact_7d_is_missing(self) -> None:
        # 7d = 168h is the upper bound — not included in stale_3_7d.
        result = lookup_hb(
            observations=[_obs(offset_hours=168, value=10.0)],
            anchor_utc=ANCHOR,
        )
        assert result.freshness == "missing"

    def test_beyond_7d_is_missing(self) -> None:
        result = lookup_hb(
            observations=[_obs(offset_hours=200, value=10.0)],
            anchor_utc=ANCHOR,
        )
        assert result.freshness == "missing"

    def test_no_observations_is_missing(self) -> None:
        result = lookup_hb(observations=[], anchor_utc=ANCHOR)
        assert result.freshness == "missing"
        assert result.value_g_dl is None
        assert result.datetime_utc is None
        assert result.source is None
        assert result.delta_hb_bypass is False

    def test_observations_after_anchor_are_ignored(self) -> None:
        # The lookup is strictly BEFORE the anchor — a result reported
        # after the order is unavailable to the ordering clinician.
        result = lookup_hb(
            observations=[_obs(offset_hours=-3, value=10.0)],
            anchor_utc=ANCHOR,
        )
        assert result.freshness == "missing"


# =============================================================================
# AC: LABEXM source preference — HEMATOLOGY beats POCT when both in window
# =============================================================================


class TestSourcePreference:
    """LABEXM 290095 (HEMATOLOGY) is the canonical Hb; 500001 (POCT) is the
    fallback when no HEMATOLOGY exists in the 7d lookback."""

    def test_hematology_preferred_when_both_present(self) -> None:
        # Even when POCT is more recent, prefer HEMATOLOGY.
        result = lookup_hb(
            observations=[
                _obs(offset_hours=2, value=8.0, source="POCT", item_no=1),
                _obs(offset_hours=6, value=10.0, source="HEMATOLOGY", item_no=2),
            ],
            anchor_utc=ANCHOR,
        )
        assert result.source == "HEMATOLOGY"
        assert result.value_g_dl == 10.0

    def test_poct_used_when_no_hematology(self) -> None:
        result = lookup_hb(
            observations=[
                _obs(offset_hours=2, value=8.0, source="POCT", item_no=1),
            ],
            anchor_utc=ANCHOR,
        )
        assert result.source == "POCT"
        assert result.value_g_dl == 8.0

    def test_hematology_outside_window_falls_back_to_poct(self) -> None:
        # HEMATOLOGY beyond 7d does NOT count for preference — only
        # observations in the 7d lookback are eligible.
        result = lookup_hb(
            observations=[
                _obs(offset_hours=200, value=12.0, source="HEMATOLOGY", item_no=1),
                _obs(offset_hours=2, value=8.0, source="POCT", item_no=2),
            ],
            anchor_utc=ANCHOR,
        )
        assert result.source == "POCT"
        assert result.value_g_dl == 8.0

    def test_most_recent_hematology_wins_among_hematology(self) -> None:
        result = lookup_hb(
            observations=[
                _obs(offset_hours=10, value=8.5, source="HEMATOLOGY", item_no=1),
                _obs(offset_hours=2, value=10.0, source="HEMATOLOGY", item_no=2),
            ],
            anchor_utc=ANCHOR,
        )
        assert result.source == "HEMATOLOGY"
        assert result.value_g_dl == 10.0


# =============================================================================
# AC: Multi-Hb tie-breaking (same datetime → highest ITEMNO wins)
# =============================================================================


class TestTieBreaking:
    """Two Hb observations with identical datetime + source are resolved by
    the Lab row identifier ``item_no``. Higher wins — the later-inserted
    row is the corrected / amended one in HOSxP semantics."""

    def test_same_datetime_higher_item_no_wins(self) -> None:
        result = lookup_hb(
            observations=[
                _obs(offset_hours=2, value=8.0, source="HEMATOLOGY", item_no=1),
                _obs(offset_hours=2, value=10.0, source="HEMATOLOGY", item_no=42),
            ],
            anchor_utc=ANCHOR,
        )
        # item_no=42 wins → value is 10.0, not 8.0.
        assert result.value_g_dl == 10.0

    def test_tied_peak_prior_is_deterministic_under_reordering(self) -> None:
        # Two priors share the highest Hb value in the 6h window. The
        # "peak prior" reported in the delta-Hb window's audit fields must
        # not depend on caller's input order; the convention is to prefer
        # the more recent of the tied pair (and finally the highest
        # item_no), matching _select_current.
        forward = lookup_hb(
            observations=[
                _obs(offset_hours=0, value=8.0, item_no=3),
                _obs(offset_hours=3, value=12.0, item_no=2),
                _obs(offset_hours=5, value=12.0, item_no=1),
            ],
            anchor_utc=ANCHOR,
        )
        reverse = lookup_hb(
            observations=[
                _obs(offset_hours=5, value=12.0, item_no=1),
                _obs(offset_hours=3, value=12.0, item_no=2),
                _obs(offset_hours=0, value=8.0, item_no=3),
            ],
            anchor_utc=ANCHOR,
        )
        # Whole-window equality is the strongest cross-ordering contract.
        assert forward.delta_hb_windows == reverse.delta_hb_windows
        # And the chosen peak prior is the more recent of the tied pair.
        win_6h = _window(forward, hours=6)
        assert win_6h.prior_value_g_dl == 12.0
        assert win_6h.prior_datetime_utc == ANCHOR - timedelta(hours=3)

    def test_same_datetime_tie_break_is_deterministic_under_reordering(self) -> None:
        # The contract must not depend on input ordering.
        forward = lookup_hb(
            observations=[
                _obs(offset_hours=2, value=8.0, source="HEMATOLOGY", item_no=1),
                _obs(offset_hours=2, value=10.0, source="HEMATOLOGY", item_no=42),
            ],
            anchor_utc=ANCHOR,
        )
        reverse = lookup_hb(
            observations=[
                _obs(offset_hours=2, value=10.0, source="HEMATOLOGY", item_no=42),
                _obs(offset_hours=2, value=8.0, source="HEMATOLOGY", item_no=1),
            ],
            anchor_utc=ANCHOR,
        )
        assert forward.value_g_dl == reverse.value_g_dl == 10.0
        assert forward.datetime_utc == reverse.datetime_utc


# =============================================================================
# AC: Tiered delta-Hb trigger — ≥1.5/6h, ≥2/12h, ≥2.5/24h
# =============================================================================


class TestDeltaHb6hWindow:
    """6-hour window: drop ≥ 1.5 g/dL → triggered."""

    def test_drop_above_15_in_6h_triggers(self) -> None:
        # Current 8.0 at t=0; prior 9.6 at t-3h → drop = 1.6 ≥ 1.5.
        result = lookup_hb(
            observations=[
                _obs(offset_hours=0, value=8.0, item_no=2),
                _obs(offset_hours=3, value=9.6, item_no=1),
            ],
            anchor_utc=ANCHOR,
        )
        assert result.delta_hb_bypass is True
        win_6h = _window(result, hours=6)
        assert win_6h.triggered is True
        assert win_6h.drop_g_dl == pytest.approx(1.6)

    def test_drop_exactly_15_in_6h_triggers(self) -> None:
        # Threshold is ≥, so 1.5 exactly is "triggered".
        result = lookup_hb(
            observations=[
                _obs(offset_hours=0, value=8.0, item_no=2),
                _obs(offset_hours=3, value=9.5, item_no=1),
            ],
            anchor_utc=ANCHOR,
        )
        assert _window(result, hours=6).triggered is True
        assert result.delta_hb_bypass is True

    def test_drop_below_15_in_6h_does_not_trigger(self) -> None:
        # Drop = 1.4 < 1.5 → not triggered in 6h, and 1.4 < 2.0 → also not
        # triggered in 12h, and 1.4 < 2.5 → not in 24h. delta_hb_bypass is
        # False.
        result = lookup_hb(
            observations=[
                _obs(offset_hours=0, value=8.0, item_no=2),
                _obs(offset_hours=3, value=9.4, item_no=1),
            ],
            anchor_utc=ANCHOR,
        )
        assert _window(result, hours=6).triggered is False
        assert result.delta_hb_bypass is False


class TestDeltaHb12hWindow:
    """12-hour window: drop ≥ 2.0 g/dL → triggered."""

    def test_drop_exactly_20_in_12h_triggers(self) -> None:
        # Prior at t-8h: outside the 6h window, inside the 12h window.
        # Drop 2.0 ≥ 2.0 → 12h triggers; 6h doesn't (no prior in 6h).
        result = lookup_hb(
            observations=[
                _obs(offset_hours=0, value=8.0, item_no=2),
                _obs(offset_hours=8, value=10.0, item_no=1),
            ],
            anchor_utc=ANCHOR,
        )
        assert _window(result, hours=12).triggered is True
        assert _window(result, hours=6).triggered is False
        assert result.delta_hb_bypass is True

    def test_drop_below_20_in_12h_does_not_trigger(self) -> None:
        # Drop = 1.9 in 8h. Outside 6h (so 6h is not triggered for absence
        # of prior). In 12h, drop 1.9 < 2.0 → not triggered. In 24h, drop
        # 1.9 < 2.5 → not triggered.
        result = lookup_hb(
            observations=[
                _obs(offset_hours=0, value=8.0, item_no=2),
                _obs(offset_hours=8, value=9.9, item_no=1),
            ],
            anchor_utc=ANCHOR,
        )
        assert _window(result, hours=12).triggered is False
        assert result.delta_hb_bypass is False


class TestDeltaHb24hWindow:
    """24-hour window: drop ≥ 2.5 g/dL → triggered."""

    def test_drop_exactly_25_in_24h_triggers(self) -> None:
        result = lookup_hb(
            observations=[
                _obs(offset_hours=0, value=8.0, item_no=2),
                _obs(offset_hours=20, value=10.5, item_no=1),
            ],
            anchor_utc=ANCHOR,
        )
        assert _window(result, hours=24).triggered is True
        assert result.delta_hb_bypass is True

    def test_drop_below_25_in_24h_does_not_trigger(self) -> None:
        result = lookup_hb(
            observations=[
                _obs(offset_hours=0, value=8.0, item_no=2),
                _obs(offset_hours=20, value=10.4, item_no=1),
            ],
            anchor_utc=ANCHOR,
        )
        assert _window(result, hours=24).triggered is False
        assert result.delta_hb_bypass is False

    def test_drop_outside_24h_window_does_not_trigger(self) -> None:
        # A 30h-old prior is outside the 24h window — even a huge drop does
        # not earn the bypass, because the rule is about acuity.
        result = lookup_hb(
            observations=[
                _obs(offset_hours=0, value=8.0, item_no=2),
                _obs(offset_hours=30, value=15.0, item_no=1),
            ],
            anchor_utc=ANCHOR,
        )
        assert _window(result, hours=24).triggered is False
        assert result.delta_hb_bypass is False


class TestDeltaHbWindowsShape:
    """Output contract: always three windows, in 6h/12h/24h order."""

    def test_windows_count_is_three(self) -> None:
        result = lookup_hb(
            observations=[_obs(offset_hours=0, value=10.0)],
            anchor_utc=ANCHOR,
        )
        assert len(result.delta_hb_windows) == 3

    def test_windows_have_canonical_hours_in_order(self) -> None:
        result = lookup_hb(
            observations=[_obs(offset_hours=0, value=10.0)],
            anchor_utc=ANCHOR,
        )
        hours = tuple(w.window_hours for w in result.delta_hb_windows)
        assert hours == (6, 12, 24)

    def test_windows_have_canonical_thresholds(self) -> None:
        result = lookup_hb(
            observations=[_obs(offset_hours=0, value=10.0)],
            anchor_utc=ANCHOR,
        )
        thresholds = tuple(w.threshold_g_dl for w in result.delta_hb_windows)
        assert thresholds == (1.5, 2.0, 2.5)

    def test_bypass_true_iff_any_window_triggered(self) -> None:
        # Bigger 24h drop triggers all three windows when prior is in 6h.
        result = lookup_hb(
            observations=[
                _obs(offset_hours=0, value=8.0, item_no=2),
                _obs(offset_hours=2, value=12.0, item_no=1),
            ],
            anchor_utc=ANCHOR,
        )
        assert any(w.triggered for w in result.delta_hb_windows) is True
        assert result.delta_hb_bypass is True


# =============================================================================
# AC: Single-Hb (no prior in window) flagged for NEEDS_REVIEW if Hb < 8
# =============================================================================


class TestSingleLowHbReview:
    """A single Hb below 8 with no prior trend earns ``needs_review_single_low_hb``
    so the deterministic classifier routes it to manual review rather than
    silently treating an isolated low value as a confirmed anaemia signal."""

    def test_single_hb_below_8_flags_review(self) -> None:
        result = lookup_hb(
            observations=[_obs(offset_hours=2, value=7.5)],
            anchor_utc=ANCHOR,
        )
        assert result.needs_review_single_low_hb is True

    def test_single_hb_exactly_8_does_not_flag(self) -> None:
        # The threshold is strictly < 8.0 — 8.0 itself is not low enough
        # alone to trigger the review flag.
        result = lookup_hb(
            observations=[_obs(offset_hours=2, value=8.0)],
            anchor_utc=ANCHOR,
        )
        assert result.needs_review_single_low_hb is False

    def test_single_hb_above_8_does_not_flag(self) -> None:
        result = lookup_hb(
            observations=[_obs(offset_hours=2, value=11.0)],
            anchor_utc=ANCHOR,
        )
        assert result.needs_review_single_low_hb is False

    def test_low_hb_with_prior_in_24h_does_not_flag(self) -> None:
        # Two Hbs → there IS a trend → delta-Hb path applies; the
        # single-Hb safety net is not needed.
        result = lookup_hb(
            observations=[
                _obs(offset_hours=2, value=7.5, item_no=2),
                _obs(offset_hours=20, value=8.0, item_no=1),
            ],
            anchor_utc=ANCHOR,
        )
        assert result.needs_review_single_low_hb is False

    def test_low_hb_with_prior_beyond_24h_still_flags(self) -> None:
        # A prior > 24h is irrelevant for the single-Hb-low review trigger
        # because the trend signal (delta-Hb) cannot apply.
        result = lookup_hb(
            observations=[
                _obs(offset_hours=2, value=7.5, item_no=2),
                _obs(offset_hours=30, value=8.0, item_no=1),
            ],
            anchor_utc=ANCHOR,
        )
        assert result.needs_review_single_low_hb is True

    def test_missing_does_not_flag(self) -> None:
        # No Hb at all is a different problem than "one low Hb with no trend".
        result = lookup_hb(observations=[], anchor_utc=ANCHOR)
        assert result.needs_review_single_low_hb is False


# =============================================================================
# AC: Pydantic v2 outputs are immutable — frozen models from the public API
# =============================================================================


class TestPublicOutputsAreImmutable:
    def test_result_is_frozen(self) -> None:
        result = lookup_hb(
            observations=[_obs(offset_hours=2, value=10.0)],
            anchor_utc=ANCHOR,
        )
        assert isinstance(result, HbLookupResult)
        with pytest.raises(ValidationError):
            result.value_g_dl = 99.0  # type: ignore[misc]

    def test_result_windows_is_a_tuple(self) -> None:
        # The list-vs-tuple distinction is the same point as Codex P2 on
        # IngestResult: ``frozen`` blocks reassignment, not nested mutation.
        result = lookup_hb(
            observations=[_obs(offset_hours=2, value=10.0)],
            anchor_utc=ANCHOR,
        )
        assert isinstance(result.delta_hb_windows, tuple)
        with pytest.raises(AttributeError):
            result.delta_hb_windows.append(  # type: ignore[attr-defined]
                DeltaHbWindow(
                    window_hours=99,
                    threshold_g_dl=0.0,
                    prior_value_g_dl=None,
                    prior_datetime_utc=None,
                    drop_g_dl=None,
                    triggered=False,
                )
            )


class TestHbObservationInvariants:
    """The type itself enforces analytic validity — constructing an out-of-range
    or naive-datetime observation raises rather than letting a malformed input
    leak into the lookup."""

    def test_value_below_2_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HbObservation(
                value_g_dl=1.9,
                datetime_utc=ANCHOR,
                source="HEMATOLOGY",
                item_no=1,
            )

    def test_value_above_25_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HbObservation(
                value_g_dl=25.1,
                datetime_utc=ANCHOR,
                source="HEMATOLOGY",
                item_no=1,
            )

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HbObservation(
                value_g_dl=10.0,
                datetime_utc=datetime(2026, 5, 15, 12, 0, 0),  # no tzinfo
                source="HEMATOLOGY",
                item_no=1,
            )

    def test_non_utc_tz_aware_datetime_rejected(self) -> None:
        # Per the project's tz contract (see RowTimestamp): the persisted
        # timestamp is UTC. A Bangkok-aware datetime carries the right
        # instant in tzinfo but would, if accepted here, propagate as
        # `datetime_utc` in the public lookup output — silently leaking a
        # local time into downstream classifiers.
        bangkok = ZoneInfo("Asia/Bangkok")
        with pytest.raises(ValidationError):
            HbObservation(
                value_g_dl=10.0,
                datetime_utc=datetime(2026, 5, 15, 19, 0, 0, tzinfo=bangkok),
                source="HEMATOLOGY",
                item_no=1,
            )

    def test_unknown_source_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HbObservation(
                value_g_dl=10.0,
                datetime_utc=ANCHOR,
                source="VENOUS_GAS",  # type: ignore[arg-type]
                item_no=1,
            )


# =============================================================================
# Helpers
# =============================================================================


def _window(result: HbLookupResult, *, hours: int) -> DeltaHbWindow:
    """Find the DeltaHbWindow for ``hours`` in the result tuple."""
    matches = [w for w in result.delta_hb_windows if w.window_hours == hours]
    assert len(matches) == 1, (
        f"expected exactly one {hours}h window, got {len(matches)}: "
        f"{[w.window_hours for w in result.delta_hb_windows]}"
    )
    return matches[0]
