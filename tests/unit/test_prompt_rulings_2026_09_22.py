"""Clinician rulings of 2026-09-22 on the prompt gaps the dev-test consortium found.

A four-reviewer consortium over 31 answers (PR #247) named six points on which
the prompt was silent or ambiguous, so the model was inventing the rule. The
clinician ruled on each; these tests pin the wording that carries each ruling,
because a missing sentence here silently reverts to model guesswork.
"""

from __future__ import annotations

import re

from bba.prompt_builder.system_prompt import platelet_system_prompt, system_prompt_for


def _platelet() -> str:
    return platelet_system_prompt()


def _rbc() -> str:
    return system_prompt_for(task_mode="HB_7_10_REVIEW", cohort_threshold=7.0)


class TestIntracranialBleedStaysAcuteForOneWeek:
    def test_one_week_window_is_stated(self) -> None:
        # 68000711: acute-on-chronic SDH on CT four days before a platelet
        # order at 37,000 was judged with "how long is acute?" unanswered.
        prompt = _platelet()
        assert "7 days" in prompt or "one week" in prompt.lower()

    def test_after_the_week_the_stable_rule_applies_unless_new_evidence(self) -> None:
        prompt = _platelet()
        assert re.search(r"after (7 days|one week)", prompt, re.IGNORECASE)
        assert "new imaging" in prompt.lower()


class TestActiveBleedingIsANumberedIndication:
    def test_active_bleeding_has_its_own_threshold(self) -> None:
        # Before: active bleeding appeared only in the hard-signal list, and
        # indications 4 and 5 were framed "without active bleeding", so the
        # model had to invent the count at which bleeding qualifies.
        prompt = _platelet()
        assert re.search(
            r"7\. .*active bleeding.*<50,000", prompt, re.IGNORECASE | re.DOTALL
        )


class TestProcedureList:
    def test_added_procedures_share_the_50k_threshold(self) -> None:
        prompt = _platelet().lower()
        for procedure in ("thoracocentesis", "arthrocentesis", "dental extraction"):
            assert procedure in prompt

    def test_bone_marrow_biopsy_grounds_no_transfusion(self) -> None:
        prompt = _platelet().lower()
        assert "bone marrow" in prompt
        assert re.search(
            r"bone marrow[^.]*(no threshold|any count|does not ground)", prompt
        )

    def test_unlisted_procedures_go_to_review(self) -> None:
        prompt = _platelet()
        assert re.search(r"[Aa]ny other invasive procedure[^.]*NEEDS_REVIEW", prompt)

    def test_surgery_done_within_the_previous_day_still_counts(self) -> None:
        # 68053394: post-op day 1 from an arthrotomy at 59,000 was judged
        # INAPPROPRIATE because the surgery was not "planned".
        prompt = _platelet()
        assert re.search(r"performed within the previous 24 h", prompt)


class TestIcdCodesAloneNeverGround:
    def test_platelet_prompt_says_so(self) -> None:
        prompt = _platelet()
        assert re.search(
            r"ICD[^.]*(alone|by itself)[^.]*(never|not)[^.]*ground",
            prompt,
            re.IGNORECASE,
        )

    def test_rbc_prompt_says_so(self) -> None:
        prompt = _rbc()
        assert re.search(
            r"ICD[^.]*(alone|by itself)[^.]*(never|not)[^.]*ground",
            prompt,
            re.IGNORECASE,
        )


class TestRbcSubThresholdUsesTheTwentyFourHourMinimum:
    def test_definition_names_the_lowest_hb_in_the_24_hours_before_the_order(
        self,
    ) -> None:
        # 68011290: closest Hb 8.1 (at the floor) vs 7.9 twelve hours earlier;
        # the rule deciding which counts lived only inside an evidence block.
        prompt = _rbc()
        assert re.search(r"lowest Hb[^.]*24 hours before the order", prompt)
        assert "order-time Hb" not in prompt

    def test_platelet_prompt_is_not_touched_by_the_rbc_rule(self) -> None:
        assert "lowest Hb" not in _platelet()


class TestNoContradictionsAfterTheRulings:
    def test_high_hb_override_prompt_does_not_forbid_the_24h_minimum(self) -> None:
        # Codex on PR #248: Hb 6.8 twelve hours before the order, latest 10.2:
        # routing sends it to the override prompt, which still said
        # SUB_THRESHOLD_HB "cannot apply here".
        prompt = system_prompt_for(task_mode="HB_GT_10_OVERRIDE", cohort_threshold=7.0)
        assert "cannot apply here" not in prompt
        assert re.search(r"lowest Hb[^.]*24 hours before the order", prompt)

    def test_terminal_line_allows_review_for_an_unlisted_procedure(self) -> None:
        # Codex on PR #248: indication 2 sends an unlisted procedure to
        # NEEDS_REVIEW, but the closing rule reserved NEEDS_REVIEW for
        # conflicting evidence only.
        prompt = _platelet()
        assert re.search(
            r"Reserve NEEDS_REVIEW for[^.]*procedure the list above does not name",
            prompt,
        )
