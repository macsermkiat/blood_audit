"""``trigger_draws`` names the exact Hb draws the guardrail judges.

Review of e94166e (2026-11-10): the LLM leg flagged the latest and lowest Hb
of any source with no prior-transfusion cutoff, while ``lookup_hb`` (which the
replay guardrail reads) prefers HEMATOLOGY and applies the cutoff. The model
then cited a value the guardrail did not see, and its APPROPRIATE was asserted
INAPPROPRIATE. Flags must come from the same selection as ``lookup_hb``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bba.hb_lookup import HbObservation, lookup_hb, trigger_draws

ANCHOR = datetime(2025, 4, 23, 8, 0, tzinfo=timezone.utc)


def _obs(hours: float, value: float, source: str, item_no: int) -> HbObservation:
    return HbObservation(
        value_g_dl=value,
        datetime_utc=ANCHOR + timedelta(hours=hours),
        source=source,  # type: ignore[arg-type]
        item_no=item_no,
    )


def test_flags_follow_the_hematology_preference_of_lookup_hb() -> None:
    hematology = _obs(-10, 7.8, "HEMATOLOGY", 1)
    poct = _obs(-2, 6.9, "POCT", 2)
    draws = trigger_draws(observations=[hematology, poct], anchor_utc=ANCHOR)
    result = lookup_hb(observations=[hematology, poct], anchor_utc=ANCHOR)

    assert draws.current == hematology
    assert draws.min_24h == hematology
    assert result.value_g_dl == draws.current.value_g_dl
    assert result.min_24h_g_dl == draws.min_24h.value_g_dl


def test_24h_minimum_respects_the_prior_transfusion_cutoff() -> None:
    corrected_low = _obs(-2, 7.4, "HEMATOLOGY", 1)
    after_transfusion = _obs(-1, 8.4, "HEMATOLOGY", 2)
    cutoff = ANCHOR - timedelta(hours=1, minutes=30)
    draws = trigger_draws(
        observations=[corrected_low, after_transfusion],
        anchor_utc=ANCHOR,
        not_before_utc=cutoff,
    )
    result = lookup_hb(
        observations=[corrected_low, after_transfusion],
        anchor_utc=ANCHOR,
        not_before_utc=cutoff,
    )

    assert draws.min_24h == after_transfusion
    assert result.min_24h_g_dl == 8.4


def test_no_draw_in_the_lookback_gives_no_flags() -> None:
    draws = trigger_draws(
        observations=[_obs(-200, 7.0, "HEMATOLOGY", 1)], anchor_utc=ANCHOR
    )
    assert draws.current is None
    assert draws.min_24h is None
    assert draws.min_48h is None
