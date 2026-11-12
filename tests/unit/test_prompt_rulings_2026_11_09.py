"""Clinician review of 2026-11-09, platelet prompt rulings.

1. Active bleeding always justifies platelets ("ถ้า active bleeding ให้เสมอ
   bypass ไปได้เลย"). The old indication 7 required a count at or below
   50,000 /uL, and no indication could clear a count at or above 100,000, so a
   bleeding patient at 120,000 was condemned. Owner decision: grounded active
   bleeding is APPROPRIATE at any count. The TTP / HIT / ITP / snakebite lines
   keep their own "life-threatening bleeding only" wording, and an intracranial
   bleed stays under indication 6.
2. A central venous catheter is a procedure ("CVC คือ central venous catheter =
   procedure"). The notes name it many ways; the prompt must list them so the
   model does not miss a C-line or DLC insertion.

These tests pin the prompt wording, because the model follows the text.
"""

from __future__ import annotations

from bba.prompt_builder.system_prompt import platelet_system_prompt


def _platelet() -> str:
    return platelet_system_prompt().replace("\xa0", " ")


class TestActiveBleedingClearsAtAnyCount:
    def test_indication_states_any_count(self) -> None:
        prompt = _platelet()
        assert "Clinically significant active bleeding" in prompt
        assert "APPROPRIATE at any platelet count" in prompt

    def test_old_50k_bleeding_cutoff_is_gone(self) -> None:
        prompt = _platelet()
        assert "hard signal for what qualifies): count ≤50,000" not in prompt

    def test_100k_ceiling_names_bleeding_as_the_exception(self) -> None:
        prompt = _platelet()
        assert "Active bleeding (indication 7) has no count threshold" in prompt

    def test_bleeding_exception_does_not_reach_withhold_populations(self) -> None:
        prompt = _platelet()
        assert "withhold unless life-threatening bleeding is present" in prompt
        assert "transfuse only if life-threatening bleed AND count ≤50,000" in prompt

    def test_intracranial_bleed_is_not_routed_through_indication_7(self) -> None:
        prompt = _platelet()
        assert "An intracranial bleed is indication 6, not indication 7" in prompt


class TestCentralVenousCatheterIsAProcedure:
    def test_catheter_aliases_are_named(self) -> None:
        prompt = _platelet()
        for alias in ("CVC", "central line", "C-line", "DLC", "Perm cath"):
            assert alias in prompt, alias

    def test_any_central_venous_catheter_counts(self) -> None:
        assert "any central venous catheter insertion" in _platelet()


class TestLabTimingIsTheDispenseTime:
    """Labs are read at dispense time (same review), so the prompt must define
    'before the order' for labs the same way, or the model judges on a
    different Hb from the one the guardrail checks."""

    def test_rbc_prompts_define_lab_timing(self) -> None:
        from bba.prompt_builder.system_prompt import system_prompt_for

        for mode in ("HB_7_10_REVIEW", "HB_GT_10_OVERRIDE"):
            prompt = system_prompt_for(task_mode=mode, cohort_threshold=7.0)
            assert "LAB TIMING" in prompt
            assert "the dispense time" in prompt
            # Reserve-ahead orders flag Hb against 'transfusion'; the definition
            # must cover that label too (second review of e94166e).
            assert "the transfusion time when they say 'transfusion'" in prompt

    def test_platelet_prompt_defines_lab_timing(self) -> None:
        prompt = _platelet()
        assert "LAB TIMING" in prompt
        assert "pre-order counts" not in prompt


class TestPlateletWordingFollowUps:
    def test_bleeding_still_needs_grounding(self) -> None:
        prompt = _platelet()
        assert "Active bleeding (indication 7) has no count threshold" in prompt
        assert "must still be grounded in the notes" in prompt

    def test_terminal_line_drops_indication_7_threshold(self) -> None:
        prompt = _platelet()
        assert "no indication 4–6 or 8 at or below its threshold" in prompt


class TestProcedureChargesGroundProcedureIndications:
    """The platelet leg now receives INCPT procedure charges. The prompt says
    indications must be grounded in the notes, so it must also say a charge
    counts, or the model sees a billed CVC insertion and still ignores it."""

    def test_charge_record_grounds_indications_1_to_3(self) -> None:
        prompt = _platelet()
        assert "PROCEDURE RECORDS" in prompt
        assert "grounds indications 1–3 as a note would" in prompt

    def test_double_lumen_catheter_charge_is_named_as_a_cvc(self) -> None:
        assert "Double lumen catheter (CVP)" in _platelet()
