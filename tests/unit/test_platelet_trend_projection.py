"""The "expected to drop below 10,000 /uL within 24 hours" clause is arithmetic.

Clinician ruling 2026-09-21 (issue #237): the clause means a STRAIGHT-LINE
projection only. Take the last two platelet counts before the order and extend
the line 24 h past the latest count; the clause holds iff the projected value
is below 10,000 /uL. On the first real-data run the model cleared 19 orders on
this clause while the count was flat or rising, so code computes the number and
the guardrail floors a clear the number does not support.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from bba.platelet_guardrail import (
    PLATELET_TREND_REVIEW_REASON,
    PlateletHardSignals,
    platelet_trend_unsupported,
)
from bba.platelet_lookup import PlateletObservation, project_24h

_ANCHOR = datetime(2026, 9, 21, 4, 0, tzinfo=UTC)
_MARROW_ONLY = PlateletHardSignals(prophylactic_marrow_failure=True)


def _obs(hours_before: float, value: float, item_no: int = 1) -> PlateletObservation:
    return PlateletObservation(
        value_k_ul=value,
        datetime_utc=_ANCHOR - timedelta(hours=hours_before),
        source="HEMATOLOGY",
        item_no=item_no,
    )


class TestProjection:
    def test_falling_count_projects_24h_past_the_latest_draw(self) -> None:
        # 30 -> 18 over 24 h is -0.5 /h; 24 h past the latest draw gives 6.
        # The horizon starts at the latest COUNT, not at the order time.
        assert project_24h(
            observations=[_obs(26, 30.0), _obs(2, 18.0)], anchor_utc=_ANCHOR
        ) == pytest.approx(6.0)

    def test_rising_count_projects_upward(self) -> None:
        # The motivating over-clear: a rising count cannot be "expected to fall".
        assert project_24h(
            observations=[_obs(26, 12.0), _obs(2, 15.0)], anchor_utc=_ANCHOR
        ) == pytest.approx(18.0)

    def test_flat_count_projects_itself(self) -> None:
        assert project_24h(
            observations=[_obs(26, 14.0), _obs(2, 14.0)], anchor_utc=_ANCHOR
        ) == pytest.approx(14.0)

    def test_uses_only_the_last_two_draws(self) -> None:
        # An older steep fall must not leak into the line.
        observations = [_obs(60, 90.0), _obs(26, 14.0), _obs(2, 14.0)]
        assert project_24h(
            observations=observations, anchor_utc=_ANCHOR
        ) == pytest.approx(14.0)

    def test_a_projection_below_zero_is_reported_as_zero(self) -> None:
        # A count cannot be negative; the rendered evidence line must not say so.
        assert (
            project_24h(
                observations=[_obs(14, 40.0), _obs(2, 10.0)], anchor_utc=_ANCHOR
            )
            == 0.0
        )

    def test_draws_a_few_minutes_apart_have_no_projection(self) -> None:
        # A recheck 20 min later (30 -> 28) is assay noise, not a trend: its
        # slope would project 0 and read as a supported fall at 28,000 /uL.
        observations = [_obs(2 + 20 / 60, 30.0), _obs(2, 28.0)]
        assert project_24h(observations=observations, anchor_utc=_ANCHOR) is None

    def test_draws_exactly_the_minimum_gap_apart_are_usable(self) -> None:
        assert project_24h(
            observations=[_obs(8, 21.0), _obs(2, 18.0)], anchor_utc=_ANCHOR
        ) == pytest.approx(6.0)

    def test_single_draw_has_no_projection(self) -> None:
        assert project_24h(observations=[_obs(2, 12.0)], anchor_utc=_ANCHOR) is None

    def test_no_draws_has_no_projection(self) -> None:
        assert project_24h(observations=[], anchor_utc=_ANCHOR) is None

    def test_post_order_draw_is_ignored(self) -> None:
        # A count taken after the order is response data, not decision evidence.
        observations = [_obs(26, 30.0), _obs(2, 18.0), _obs(-3, 5.0)]
        assert project_24h(
            observations=observations, anchor_utc=_ANCHOR
        ) == pytest.approx(6.0)

    def test_draw_at_exactly_seven_days_is_outside_the_window(self) -> None:
        # Same STRICT 7-day lower bound as lookup_platelet, so the projection
        # never reads a count the deterministic gate cannot see.
        assert (
            project_24h(
                observations=[_obs(168, 30.0), _obs(2, 18.0)], anchor_utc=_ANCHOR
            )
            is None
        )

    def test_draws_more_than_72h_apart_have_no_projection(self) -> None:
        # A line through counts 4 days apart says nothing about the next 24 h.
        assert (
            project_24h(
                observations=[_obs(100, 30.0), _obs(2, 18.0)], anchor_utc=_ANCHOR
            )
            is None
        )

    def test_draws_exactly_72h_apart_are_usable(self) -> None:
        assert project_24h(
            observations=[_obs(74, 42.0), _obs(2, 18.0)], anchor_utc=_ANCHOR
        ) == pytest.approx(10.0)

    def test_same_timestamp_keeps_the_corrected_row(self) -> None:
        # Two Lab rows at one timestamp are a correction, not two draws: the
        # higher item_no wins (lookup_platelet's tie-break) and the slope is
        # taken against the previous distinct draw.
        observations = [
            _obs(26, 30.0),
            _obs(2, 25.0, item_no=5),
            _obs(2, 18.0, item_no=9),
        ]
        assert project_24h(
            observations=observations, anchor_utc=_ANCHOR
        ) == pytest.approx(6.0)

    def test_only_same_timestamp_rows_have_no_projection(self) -> None:
        observations = [_obs(2, 25.0, item_no=5), _obs(2, 18.0, item_no=9)]
        assert project_24h(observations=observations, anchor_utc=_ANCHOR) is None


class TestTrendGuardrail:
    def test_flat_count_clear_on_the_clause_alone_is_floored(self) -> None:
        assert platelet_trend_unsupported("APPROPRIATE", _MARROW_ONLY, 14.0, 14.0)

    def test_projection_at_the_threshold_is_not_below_it(self) -> None:
        assert platelet_trend_unsupported("APPROPRIATE", _MARROW_ONLY, 14.0, 10.0)

    def test_missing_projection_is_floored_not_guessed(self) -> None:
        # Never-guess: with fewer than two draws nothing supports "expected to
        # fall", so a human decides.
        assert platelet_trend_unsupported("APPROPRIATE", _MARROW_ONLY, 14.0, None)

    def test_supported_projection_lets_the_clear_stand(self) -> None:
        assert not platelet_trend_unsupported("APPROPRIATE", _MARROW_ONLY, 14.0, 6.0)

    def test_count_below_10_needs_no_projection(self) -> None:
        # Indication 5's first arm (count < 10,000) stands on the count itself.
        assert not platelet_trend_unsupported("APPROPRIATE", _MARROW_ONLY, 9.0, None)

    @pytest.mark.parametrize(
        "signals",
        [
            PlateletHardSignals(prophylactic_marrow_failure=True, active_bleeding=True),
            PlateletHardSignals(
                prophylactic_marrow_failure=True, procedure_indication=True
            ),
            PlateletHardSignals(
                prophylactic_marrow_failure=True, intracranial_bleed_indication=True
            ),
            PlateletHardSignals(active_bleeding=True),
        ],
    )
    def test_another_indication_carries_the_clear(
        self, signals: PlateletHardSignals
    ) -> None:
        # Bleeding / procedure / intracranial thresholds sit above 10,000, so
        # the trend is not what those clears rest on.
        assert not platelet_trend_unsupported("APPROPRIATE", signals, 14.0, 14.0)

    @pytest.mark.parametrize(
        "final", ["INAPPROPRIATE", "NEEDS_REVIEW", "INSUFFICIENT_EVIDENCE"]
    )
    def test_only_a_clear_is_floored(self, final: str) -> None:
        assert not platelet_trend_unsupported(final, _MARROW_ONLY, 14.0, 14.0)

    def test_missing_trigger_count_is_left_to_the_other_gates(self) -> None:
        assert not platelet_trend_unsupported("APPROPRIATE", _MARROW_ONLY, None, None)

    def test_review_reason_slug(self) -> None:
        assert PLATELET_TREND_REVIEW_REASON == "platelet_trend_unsupported"
