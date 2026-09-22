"""SUB_THRESHOLD_HB is decided by the lowest Hb in the 24 hours before the order
(issue #249; clinician ruling 2026-09-22, window chosen over 72 h).

68011290: closest Hb 8.1 at the 8.0 floor, 7.9 twelve hours earlier. The rule
that the 24 h minimum decides lived only inside an evidence block, the prompt
said "order-time Hb", and the replay over-clear guardrail checked the closest
value, so a correct citation of the 7.9 was asserted INAPPROPRIATE. The three
layers now agree: the lookup carries the minimum, the guardrail accepts it, and
the prompt states it. A low that an intervening transfusion has already
corrected does not count: the minimum is taken from the last transfusion in the
window onward.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

from bba.hb_lookup.lookup import lookup_hb
from bba.hb_lookup.models import HbLookupResult, HbObservation
from bba.prompt_builder.system_prompt import system_prompt_for

ANCHOR = datetime(2026, 5, 15, 12, 0, 0, tzinfo=timezone.utc)


def _obs(
    hours_before: float, value: float, source: str = "HEMATOLOGY"
) -> HbObservation:
    return HbObservation(
        value_g_dl=value,
        datetime_utc=ANCHOR - timedelta(hours=hours_before),
        source=source,  # type: ignore[arg-type]
        item_no=int(hours_before * 10),
    )


class TestLookupCarriesTheMinimum:
    def test_lowest_value_in_the_window_is_reported(self) -> None:
        # 68011290: closest 8.1, 7.9 twelve hours earlier.
        result = lookup_hb(
            observations=[_obs(12, 7.9), _obs(1, 8.1)], anchor_utc=ANCHOR
        )
        assert result.value_g_dl == 8.1
        assert result.min_24h_g_dl == 7.9

    def test_a_low_older_than_24_hours_does_not_count(self) -> None:
        # 68013076: 6.9 two days before an order placed at 9.4 (72 h would have
        # cleared it on a recovered value).
        result = lookup_hb(
            observations=[_obs(53, 6.9), _obs(2, 9.4)], anchor_utc=ANCHOR
        )
        assert result.min_24h_g_dl == 9.4

    def test_a_low_after_the_anchor_does_not_count(self) -> None:
        result = lookup_hb(
            observations=[_obs(-3, 6.0), _obs(2, 9.4)], anchor_utc=ANCHOR
        )
        assert result.min_24h_g_dl == 9.4

    def test_a_low_before_an_intervening_transfusion_does_not_count(self) -> None:
        # Transfused 10 h before the order: the 6.8 from before that transfusion
        # was already corrected; only values from the transfusion onward count.
        result = lookup_hb(
            observations=[_obs(14, 6.8), _obs(6, 8.3), _obs(1, 8.1)],
            anchor_utc=ANCHOR,
            not_before_utc=ANCHOR - timedelta(hours=10),
        )
        assert result.min_24h_g_dl == 8.1

    def test_hematology_values_are_preferred_over_poct_like_the_current_value(
        self,
    ) -> None:
        result = lookup_hb(
            observations=[_obs(5, 6.5, source="POCT"), _obs(2, 8.4)], anchor_utc=ANCHOR
        )
        assert result.min_24h_g_dl == 8.4

    def test_missing_result_has_no_minimum(self) -> None:
        assert lookup_hb(observations=[], anchor_utc=ANCHOR).min_24h_g_dl is None

    def test_existing_constructors_need_no_minimum(self) -> None:
        # Every sentinel / test fixture builds HbLookupResult without it.
        result = HbLookupResult(
            value_g_dl=None,
            datetime_utc=None,
            source=None,
            freshness="missing",
            delta_hb_bypass=False,
            delta_hb_windows=(),
            needs_review_single_low_hb=False,
        )
        assert result.min_24h_g_dl is None


class TestPromptStatesTheRule:
    @pytest.mark.parametrize("mode", ["HB_7_10_REVIEW", "HB_GT_10_OVERRIDE"])
    def test_sub_threshold_is_the_24h_minimum_since_the_last_transfusion(
        self, mode: str
    ) -> None:
        prompt = system_prompt_for(task_mode=mode, cohort_threshold=7.0)
        assert re.search(r"lowest Hb[^.]*24 hours before the order", prompt)
        assert "transfusion" in prompt.split("SUB_THRESHOLD_HB", 1)[1][:400]
        assert "order-time Hb" not in prompt

    def test_override_template_no_longer_forbids_it(self) -> None:
        # Hb 6.8 twelve hours earlier, latest 10.2: routed to the override
        # template, which used to say SUB_THRESHOLD_HB "cannot apply here".
        prompt = system_prompt_for(task_mode="HB_GT_10_OVERRIDE", cohort_threshold=7.0)
        assert "cannot apply here" not in prompt


class TestPilotTransfusionCutoff:
    def test_previous_rbc_order_in_the_window_bounds_the_minimum(self) -> None:
        import importlib.util
        import sys
        from pathlib import Path

        pilot = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
        if str(pilot) not in sys.path:
            sys.path.insert(0, str(pilot))
        spec = importlib.util.spec_from_file_location(
            "pilot_run_llm_leg_min24", pilot / "run_llm_leg.py"
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)

        earlier = ANCHOR - timedelta(hours=10)
        older = ANCHOR - timedelta(hours=40)
        by_an = {"AN1": [older, earlier, ANCHOR]}

        # The most recent RBC order strictly before this one, inside 24 h.
        assert mod._last_rbc_order_before(by_an, "AN1", ANCHOR) == earlier
        # An order more than 24 h back is outside the window: no cutoff.
        assert (
            mod._last_rbc_order_before({"AN1": [older, ANCHOR]}, "AN1", ANCHOR) is None
        )
        assert mod._last_rbc_order_before({}, "AN1", ANCHOR) is None
