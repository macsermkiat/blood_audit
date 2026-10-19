"""Owner ruling of 2026-09-27: a value exactly AT an indication threshold qualifies.

Review case REQNO 68004713 had Hb 7.0 against the default 7.0 floor. The
deterministic rule and the prompt both treated the floor as "not
sub-threshold", so the order went to the gray zone and the LLM called it
INAPPROPRIATE. The owner ruled that every indication cutoff is inclusive
("<="): the RBC universal 7.0 rule, every cohort floor, and every platelet
indication cutoff (10,000 / 20,000 / 50,000 / 80,000 /uL), including the
"expected to fall within 24 h" clauses. Follow-up ruling 2026-09-28: the
platelet 100,000 /uL cutoff (high-risk surgery, acute intracranial bleed)
stays STRICT (<100,000); a count exactly at 100,000 does not qualify. These tests pin the prompt wording
that carries the ruling, because the model follows the text, not the code.
"""

from __future__ import annotations

import re

import pytest

from bba.prompt_builder.system_prompt import platelet_system_prompt, system_prompt_for

_PLATELET_CUTOFFS = ("10,000", "20,000", "50,000", "80,000")


def _platelet() -> str:
    # The prompt separates some counts from "/μL" with a no-break space;
    # normalise it so the wording checks are about the words, not the spacing.
    return platelet_system_prompt().replace("\xa0", " ")


class TestRbcFloorIsInclusive:
    @pytest.mark.parametrize("mode", ["HB_7_10_REVIEW", "HB_GT_10_OVERRIDE"])
    def test_sub_threshold_is_at_or_below_the_floor(self, mode: str) -> None:
        prompt = system_prompt_for(task_mode=mode, cohort_threshold=8.0)
        assert "at or below the cohort floor of 8.0 g/dL" in prompt
        assert "exactly at 8.0 g/dL qualifies" in prompt
        assert "NOT sub-threshold" not in prompt
        assert "strictly below the cohort floor" not in prompt

    def test_gt_10_override_applies_sub_threshold_at_the_floor(self) -> None:
        prompt = system_prompt_for(task_mode="HB_GT_10_OVERRIDE", cohort_threshold=7.0)
        assert "at or below the 7.0 g/dL floor, SUB_THRESHOLD_HB applies" in prompt
        assert "strictly below the 7.0" not in prompt

    def test_gray_zone_restrictive_rule_starts_above_the_floor(self) -> None:
        prompt = system_prompt_for(task_mode="HB_7_10_REVIEW", cohort_threshold=7.0)
        assert "Above the 7.0 g/dL floor, transfusion is APPROPRIATE only" in prompt
        assert "At or above the 7.0" not in prompt
        assert "an Hb above 7.0 g/dL is INAPPROPRIATE" in prompt
        assert "at or below the cohort threshold of 7.0 g/dL" in prompt

    def test_high_hb_routing_boundary_is_unchanged(self) -> None:
        # Out of scope for the ruling: Hb >= 10 is the routing gate.
        prompt = system_prompt_for(task_mode="HB_GT_10_OVERRIDE", cohort_threshold=7.0)
        assert "at or above 10 g/dL" in prompt


class TestPlateletCutoffsAreInclusive:
    def test_positive_indications_say_at_or_below(self) -> None:
        prompt = _platelet()
        assert "a count exactly at a ≤ cutoff qualifies" in prompt
        assert "count is below the stated threshold" not in prompt

    @pytest.mark.parametrize("cutoff", _PLATELET_CUTOFFS)
    def test_every_cutoff_is_written_inclusive(self, cutoff: str) -> None:
        prompt = _platelet()
        assert f"≤{cutoff} /μL" in prompt
        assert f"<{cutoff} /μL" not in prompt

    def test_hard_signal_shorthand_is_inclusive(self) -> None:
        prompt = _platelet()
        assert "LP ≤50–80k" in prompt
        assert "major surgery ≤80k (<100k high risk)" in prompt
        assert not re.search(r"(?<!\()<\d+(–\d+)?k\b", prompt)

    def test_expected_fall_clauses_are_inclusive(self) -> None:
        prompt = _platelet()
        assert "drop below 10,000" not in prompt
        assert "expected <10,000" not in prompt
        assert prompt.count("to 10,000 /μL or below within 24 hours") >= 4

    def test_exclusion_override_low_count_is_inclusive(self) -> None:
        prompt = _platelet()
        assert "a count at or below 10,000 /μL does NOT by itself" in prompt

    def test_100k_cutoff_stays_strict(self) -> None:
        # Follow-up ruling 2026-09-28: only the 100,000 /uL cutoff stays
        # strict. Plt 100,000 with an acute bleed or high-risk surgery is not
        # below the cutoff, so it does not qualify.
        prompt = _platelet()
        assert "<100,000 /μL if the patient or procedure" in prompt
        assert "<100,000 /μL when the bleed is acute" in prompt
        assert "≤100,000" not in prompt
        assert "except the 100,000 /μL cutoffs" in prompt

    def test_head_bleed_exclusion_above_100k_is_unchanged(self) -> None:
        # Unchanged: exclusion B (>100,000). A count exactly at 100,000 meets
        # neither the strict <100,000 indication nor this exclusion.
        assert "platelet count >100,000 /μL" in _platelet()

    def test_local_trigger_example_is_quoted_verbatim(self) -> None:
        # The standing-order example is chart text under audit, not policy.
        assert "'if plt < 20,000 give platelets'" in _platelet()


class TestPlateletToolSchemaIsInclusive:
    """The tool-schema field descriptions restate the cutoffs; they must not
    contradict the prompt, or the model gets two different rules."""

    @staticmethod
    def _descriptions() -> str:
        from bba.llm_client.transport import _PLATELET_TOOL_INPUT_SCHEMA

        props = _PLATELET_TOOL_INPUT_SCHEMA["properties"]
        return "\n".join(str(v.get("description", "")) for v in props.values())

    def test_no_strict_less_than_cutoff_remains(self) -> None:
        text = self._descriptions()
        # The only strict cutoff left is 100,000 (ruling 2026-09-28).
        assert not re.search(r"<(?!100,000|100k)\d", text), text

    def test_cutoffs_are_written_inclusive(self) -> None:
        text = self._descriptions()
        for fragment in (
            "LP <=50-80k /uL",
            "major surgery <=80k /uL (<100k /uL high risk)",
            "count <=10,000 /uL",
            "<100,000 /uL when acute",
            "<=50,000 /uL when stable",
            "<=20,000 /uL during the ATG course",
            "to 10,000 /uL or below within 24 hours",
        ):
            assert fragment in text, fragment
        assert not re.search(r"(?<!at or )below its", text)
        assert "sits below" not in text
