"""Contract tests for :mod:`bba.platelet_lookup` value parsing + model.

The platelet ``RESULT`` column is messier than Hb: the tests below pin the
four real shapes seen in the raw Lab feed (plain, comma-thousands, ``<N``
censoring, ``--`` missing) so a regression in the parser is caught as a
routing bug, not a silent mis-count. Clinical stakes: a censored ``<2`` that
collapsed to "missing" would defer a documented critical count; a comma value
that failed to parse would drop a real thrombocytosis reading.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from bba.platelet_lookup import (
    MAX_PLATELET,
    MIN_PLATELET,
    PLATELET_LABEXM,
    PlateletObservation,
    parse_platelet_count,
)


class TestParsePlateletCountRealShapes:
    """The four ``RESULT`` shapes observed in the raw feed."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("362", 362.0),
            ("10", 10.0),
            ("159", 159.0),
            (" 450 ", 450.0),  # HOSxP pads numeric columns
        ],
    )
    def test_plain_numeric(self, raw: str, expected: float) -> None:
        assert parse_platelet_count(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("1,117", 1117.0), ("1,030", 1030.0), ("1,079", 1079.0)],
    )
    def test_comma_grouped_thousands(self, raw: str, expected: float) -> None:
        # The comma is a thousands separator (Thai locale decimals use "."),
        # so it must strip to a real >1000 value, not fail to parse.
        assert parse_platelet_count(raw) == expected

    def test_left_censored_below_detection_keeps_the_bound(self) -> None:
        # "<2" is a REAL critically-low count, not missing data. It must stay a
        # numeric value (the bound) so it routes as a low count, not deferred
        # as absent.
        assert parse_platelet_count("<2") == 2.0

    def test_right_censored_keeps_the_bound(self) -> None:
        assert parse_platelet_count(">1000") == 1000.0

    @pytest.mark.parametrize("raw", ["--", "", "   ", None, "clotted", "N/A"])
    def test_missing_and_nonnumeric_are_none(self, raw: str | None) -> None:
        assert parse_platelet_count(raw) is None


class TestParsePlateletCountRange:
    """Out-of-range values are transcription errors, rejected loud."""

    def test_boundaries_inclusive(self) -> None:
        assert parse_platelet_count(str(MIN_PLATELET)) == MIN_PLATELET
        assert parse_platelet_count(str(MAX_PLATELET)) == MAX_PLATELET

    @pytest.mark.parametrize("raw", ["0", "-5", "3001", "99999", "nan", "inf"])
    def test_out_of_range_rejected(self, raw: str) -> None:
        assert parse_platelet_count(raw) is None

    def test_hb_range_would_reject_normal_platelet(self) -> None:
        # Guards the "cannot reuse hb parser" rationale: a normal platelet
        # count (159) is far outside the Hb window [2, 25], so hb_lookup's
        # parser could never serve platelets. Here it parses fine.
        assert parse_platelet_count("159") == 159.0


class TestPlateletObservation:
    """The validated observation model mirrors HbObservation's invariants."""

    def _dt(self) -> datetime:
        return datetime(2025, 5, 12, 3, 0, tzinfo=UTC)

    def test_valid_construction(self) -> None:
        obs = PlateletObservation(
            value_k_ul=362.0,
            datetime_utc=self._dt(),
            source="HEMATOLOGY",
            item_no=1,
        )
        assert obs.value_k_ul == 362.0

    def test_out_of_range_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PlateletObservation(
                value_k_ul=5000.0,
                datetime_utc=self._dt(),
                source="HEMATOLOGY",
                item_no=1,
            )

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PlateletObservation(
                value_k_ul=100.0,
                datetime_utc=datetime(2025, 5, 12, 3, 0),  # noqa: DTZ001 naive on purpose
                source="HEMATOLOGY",
                item_no=1,
            )

    def test_non_utc_aware_datetime_rejected(self) -> None:
        bangkok = timezone(timedelta(hours=7))
        with pytest.raises(ValidationError):
            PlateletObservation(
                value_k_ul=100.0,
                datetime_utc=datetime(2025, 5, 12, 10, 0, tzinfo=bangkok),
                source="HEMATOLOGY",
                item_no=1,
            )

    def test_frozen(self) -> None:
        obs = PlateletObservation(
            value_k_ul=100.0,
            datetime_utc=self._dt(),
            source="HEMATOLOGY",
            item_no=1,
        )
        with pytest.raises(ValidationError):
            obs.value_k_ul = 200.0  # type: ignore[misc]


class TestLabConfig:
    def test_labexm_is_platelet_count_code(self) -> None:
        assert PLATELET_LABEXM == "290078"


class TestLookupPlatelet:
    """Most-recent-count selection + freshness (mirrors hb_lookup contract)."""

    def _obs(self, value: float, hours_before_anchor: float, item_no: int = 1):
        from bba.platelet_lookup import PlateletObservation

        anchor = datetime(2025, 5, 12, 12, 0, tzinfo=UTC)
        return PlateletObservation(
            value_k_ul=value,
            datetime_utc=anchor - timedelta(hours=hours_before_anchor),
            source="HEMATOLOGY",
            item_no=item_no,
        )

    @property
    def _anchor(self) -> datetime:
        return datetime(2025, 5, 12, 12, 0, tzinfo=UTC)

    def test_no_observations_is_missing(self) -> None:
        from bba.platelet_lookup import lookup_platelet

        result = lookup_platelet(observations=[], anchor_utc=self._anchor)
        assert result.freshness == "missing"
        assert result.value_k_ul is None
        assert result.datetime_utc is None
        assert result.source is None

    def test_selects_most_recent_before_anchor(self) -> None:
        from bba.platelet_lookup import lookup_platelet

        result = lookup_platelet(
            observations=[
                self._obs(200.0, hours_before_anchor=48.0),
                self._obs(80.0, hours_before_anchor=2.0),
                self._obs(300.0, hours_before_anchor=120.0),
            ],
            anchor_utc=self._anchor,
        )
        assert result.value_k_ul == 80.0
        assert result.freshness == "fresh"

    def test_ignores_observations_after_anchor(self) -> None:
        from bba.platelet_lookup import lookup_platelet

        # A count drawn AFTER the order anchor (negative hours-before) must not
        # be selected — the deterministic gate is backward-only.
        result = lookup_platelet(
            observations=[
                self._obs(50.0, hours_before_anchor=-3.0),
                self._obs(150.0, hours_before_anchor=5.0),
            ],
            anchor_utc=self._anchor,
        )
        assert result.value_k_ul == 150.0

    def test_ignores_observations_beyond_lookback(self) -> None:
        from bba.platelet_lookup import lookup_platelet

        result = lookup_platelet(
            observations=[self._obs(90.0, hours_before_anchor=24 * 8)],
            anchor_utc=self._anchor,
        )
        assert result.freshness == "missing"
        assert result.value_k_ul is None

    def test_same_datetime_tie_breaks_on_highest_item_no(self) -> None:
        from bba.platelet_lookup import lookup_platelet

        # Amended Lab row (higher item_no) at the same instant wins.
        result = lookup_platelet(
            observations=[
                self._obs(100.0, hours_before_anchor=6.0, item_no=1),
                self._obs(60.0, hours_before_anchor=6.0, item_no=2),
            ],
            anchor_utc=self._anchor,
        )
        assert result.value_k_ul == 60.0

    @pytest.mark.parametrize(
        ("hours", "expected"),
        [
            (1.0, "fresh"),
            (23.9, "fresh"),
            (24.0, "stale_24_72h"),
            (71.9, "stale_24_72h"),
            (72.0, "stale_3_7d"),
            (167.0, "stale_3_7d"),
        ],
    )
    def test_freshness_tiers(self, hours: float, expected: str) -> None:
        from bba.platelet_lookup import lookup_platelet

        result = lookup_platelet(
            observations=[self._obs(100.0, hours_before_anchor=hours)],
            anchor_utc=self._anchor,
        )
        assert result.freshness == expected
