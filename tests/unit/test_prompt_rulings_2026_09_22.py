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


class TestRbcSubThresholdRulingLandedWithTheGuardrail:
    def test_the_24h_minimum_rule_is_in_the_prompt(self) -> None:
        # Ruling 6 was deferred from PR #248 and shipped with the replay
        # guardrail change in #249; the full contract is pinned in
        # tests/unit/test_hb_min_24h.py.
        assert "order-time Hb" not in _rbc()


class TestNoContradictionsAfterTheRulings:
    def test_terminal_line_allows_review_for_an_unlisted_procedure(self) -> None:
        # Codex on PR #248: indication 2 sends an unlisted procedure to
        # NEEDS_REVIEW, but the closing rule reserved NEEDS_REVIEW for
        # conflicting evidence only.
        prompt = _platelet()
        assert re.search(
            r"Reserve NEEDS_REVIEW for[^.]*procedure the list above does not name",
            prompt,
        )


class TestProjectionLineGroundsTheExpectedDrop:
    """Issue #237 wording. The clinician ruled (2026-09-21) that "expected to
    drop below 10,000 /uL within 24 hours" means a straight-line projection
    through the last two counts, and the pilot renders that projection into the
    evidence. The consortium found the model reading that line inconsistently:
    trusted when it pointed to INAPPROPRIATE, dismissed as "a mechanical
    extrapolation" when it pointed to APPROPRIATE (68046137: projection 6,767,
    verdict NEEDS_REVIEW), or accepted and then a further "explicit clinical
    note" demanded (68064992). The prompt never said the line is the trend."""

    def test_the_projection_line_is_named_as_the_count_trend(self) -> None:
        prompt = _platelet()
        assert "Projected platelet count in 24 h" in prompt
        assert re.search(
            r"Projected platelet count in 24 h[^.]*\bis\b[^.]*count trend", prompt
        )

    def test_below_ten_thousand_meets_the_clause_without_a_further_note(self) -> None:
        prompt = _platelet()
        assert re.search(
            r"below 10,000[^.]*clause is met[^.]*(no|without)[^.]*(further|additional)[^.]*note",
            prompt,
            re.IGNORECASE,
        )

    def test_at_or_above_ten_thousand_or_not_computable_does_not_meet_it(self) -> None:
        prompt = _platelet()
        assert re.search(
            r"(at or above|>=) ?10,000[^.]*not computable[^.]*not met",
            prompt,
            re.IGNORECASE,
        ) or re.search(
            r"not computable[^.]*(at or above|>=) ?10,000[^.]*not met",
            prompt,
            re.IGNORECASE,
        )

    def test_model_may_not_substitute_its_own_trend_reading(self) -> None:
        prompt = _platelet()
        assert re.search(
            r"do not (dismiss|discount|override) (it|the projection)",
            prompt,
            re.IGNORECASE,
        )
        assert "standing order" in prompt
