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

        # Clinician review 2026-11-09: the map holds dispense times and the
        # Hb is read at this order's own dispense. The order itself (at the
        # anchor) never counts; a sibling counts only once its blood was
        # dispensed before this one's, not merely ordered before it.
        own_dispense = ANCHOR + timedelta(hours=4)
        sibling_dispensed_later = ANCHOR + timedelta(hours=6)
        sibling_dispensed_earlier = ANCHOR + timedelta(hours=2)
        assert (
            mod._last_rbc_order_before(
                {"AN1": [own_dispense, sibling_dispensed_later]}, "AN1", own_dispense
            )
            is None
        )
        assert (
            mod._last_rbc_order_before(
                {"AN1": [own_dispense, sibling_dispensed_earlier]},
                "AN1",
                own_dispense,
            )
            == sibling_dispensed_earlier
        )


def _pilot_llm_leg():
    import importlib.util
    import sys
    from pathlib import Path

    pilot = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
    if str(pilot) not in sys.path:
        sys.path.insert(0, str(pilot))
    spec = importlib.util.spec_from_file_location(
        "pilot_run_llm_leg_trigger", pilot / "run_llm_leg.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestEvidenceSeesWhatTheGuardrailJudges:
    """Third review of the dispense change (2026-11-10): the model must be
    shown, and flagged on, every Hb draw the replay guardrail reads."""

    def test_bundle_reaches_a_poct_draw_between_order_and_dispense(self) -> None:
        # Order at -3 h, dispense at 0. HEMATOLOGY 9.1 at -30 h is the
        # current value, but the 24 h minimum is a POCT 7.4 at -1 h. The
        # bundle bound must reach the dispense time, not stop at the
        # order or the older HEMATOLOGY draw.
        mod = _pilot_llm_leg()
        order = ANCHOR - timedelta(hours=3)
        hematology_draw = ANCHOR - timedelta(hours=30)
        assert mod._hb_bundle_upper(ANCHOR, hematology_draw, order) == ANCHOR

    def test_bundle_window_unchanged_when_nothing_is_after_it(self) -> None:
        mod = _pilot_llm_leg()
        assert mod._hb_bundle_upper(ANCHOR, ANCHOR - timedelta(hours=2), ANCHOR) is None

    def test_fallback_hb_is_flagged_at_the_anchor_that_supplied_it(self) -> None:
        from bba.hb_lookup import (
            AnchorCandidate,
            resolve_hb_with_fallback,
            trigger_draws,
        )

        mod = _pilot_llm_leg()
        bank = AnchorCandidate(
            anchor_utc=ANCHOR + timedelta(hours=3),
            display="bank",
            reason="blood_bank_visit_fallback",
        )
        later = HbObservation(
            value_g_dl=6.8,
            datetime_utc=ANCHOR + timedelta(hours=2),
            source="HEMATOLOGY",
            item_no=7,
        )
        hb, display, reason = resolve_hb_with_fallback(
            observations=[later], order_datetime=ANCHOR, candidates=[bank]
        )
        moment = mod._hb_trigger_anchor(ANCHOR, reason, display, [bank])
        draws = trigger_draws(observations=[later], anchor_utc=moment)

        assert moment == bank.anchor_utc
        assert draws.current == later
        assert hb.value_g_dl == draws.current.value_g_dl


class TestUndispensedSiblingIsNoTransfusion:
    """Issue #264, clinician sign-off 2026-11-11 (option A): an RBC order with
    no dispense record is no evidence that blood was given, so it must not
    cut off the Hb another order on the same admission is judged on."""

    def test_only_dispensed_rbc_orders_enter_the_cutoff_map(self) -> None:
        from types import SimpleNamespace

        from bba.hb_lookup import AnchorCandidate

        mod = _pilot_llm_leg()
        # B ordered 09:00, dispensed 12:00. A ordered 10:00, never dispensed.
        b_order = ANCHOR
        b_dispense = ANCHOR + timedelta(hours=3)
        a_order = ANCHOR + timedelta(hours=1)
        orders = [
            SimpleNamespace(
                component="rbc", an="AN1", reqno="B", order_datetime=b_order
            ),
            SimpleNamespace(
                component="rbc", an="AN1", reqno="A", order_datetime=a_order
            ),
            SimpleNamespace(
                component="platelet", an="AN1", reqno="P", order_datetime=a_order
            ),
        ]
        candidates = {
            "B": [
                AnchorCandidate(
                    anchor_utc=b_dispense, display="12:00", reason="issue_datetime"
                )
            ],
            "P": [
                AnchorCandidate(
                    anchor_utc=a_order, display="10:00", reason="issue_datetime"
                )
            ],
        }

        by_an = mod._rbc_dispense_times_by_an(orders, candidates)

        assert by_an == {"AN1": [b_dispense]}
        # So B's Hb at 09:30 is no longer cut off by A's order time.
        assert mod._last_rbc_order_before(by_an, "AN1", b_dispense) is None
