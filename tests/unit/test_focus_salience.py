"""Indication-salience ranking of IPDNRFOCUSDT notes under the 5+5 cap.

Measured 2026-09-22 over all 39,749 IPD orders: 72.7% exceed the 5-before /
5-after cap and closest-first keeps only 33.6% of nurse notes whose RESPONSE
column records an observed transfusion indication (bleeding, pallor, Hb with a
value, shock...). On 8.1% of orders the LLM saw none of them. Ranking hits
first, closest-first within each tier, keeps 91.8%.

The ACTION column is templated care-plan text ("Observe signs bleeding
เช่น...") that matches on nearly every surgical patient, and RESPONSE often
records the negative ("no bleed"), so the hit test reads the labelled Response
segment only and skips negated matches. These tests pin both rules and the
ordering contract they feed.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
if str(PILOT_DIR) not in sys.path:
    sys.path.insert(0, str(PILOT_DIR))

from _periop_notes import vitals_notes_for  # noqa: E402
from bba.evidence_bundle_builder.models import FocusNote  # noqa: E402
from bba.evidence_bundle_builder.ranking import (  # noqa: E402
    focus_indication_hit,
    split_focus_notes_5_5,
)

ANCHOR = datetime(2025, 3, 1, 12, 0, tzinfo=UTC)
_ORDER_UTC = datetime(2025, 1, 3, 10, 0, tzinfo=UTC)


def _note(
    offset_hours: float, text: str = "Action: turn q2h\nResponse: comfortable"
) -> FocusNote:
    return FocusNote(timestamp=ANCHOR + timedelta(hours=offset_hours), text=text)


class TestFocusIndicationHit:
    def test_observed_bleeding_in_response_is_a_hit(self) -> None:
        assert focus_indication_hit(
            "Action: observe wound\nResponse: active bleeding at drain"
        )

    def test_templated_action_plan_alone_is_not_a_hit(self) -> None:
        # The ACTION column is a care plan, not an observation: this text sits
        # on nearly every post-op patient regardless of what happened.
        assert not focus_indication_hit(
            "Action: Observe signs bleeding เช่น แผลผ่าตัด\nResponse: rested well"
        )

    def test_negated_observation_is_not_a_hit(self) -> None:
        assert not focus_indication_hit(
            "Action: observe wound\nResponse: แผลผ่าตัด no bleed"
        )
        assert not focus_indication_hit("Action: monitor\nResponse: ไม่มีเลือดออก")
        assert not focus_indication_hit(
            "Action: monitor\nResponse: no sign of septic shock"
        )

    def test_negation_does_not_reach_across_a_clause_boundary(self) -> None:
        # "no pain" must not suppress the bleeding observed in the next clause.
        assert focus_indication_hit("Action: x\nResponse: no pain, bleeding at wound")
        assert focus_indication_hit("Action: x\nResponse: ไม่ปวด แต่ มีเลือดออก")

    def test_hb_counts_only_with_a_value(self) -> None:
        assert focus_indication_hit("Action: lab\nResponse: Hb 6.8 g/dL reported")
        assert not focus_indication_hit("Action: lab\nResponse: follow Hb result")

    def test_thai_pallor_and_gi_bleed_terms_are_hits(self) -> None:
        assert focus_indication_hit("Action: assess\nResponse: ผู้ป่วยซีด อ่อนเพลีย")
        assert focus_indication_hit("Action: assess\nResponse: ถ่ายดำ 2 ครั้ง")

    def test_unlabelled_text_falls_back_to_whole_note(self) -> None:
        # Callers that do not label the join still get the lexical rule,
        # without the ACTION exclusion, rather than silently never ranking.
        assert focus_indication_hit("massive bleeding per vagina")
        assert not focus_indication_hit("pain score 3")

    def test_is_case_insensitive(self) -> None:
        assert focus_indication_hit("Action: x\nResponse: BLEEDING from NG tube")


class TestSplitPrefersIndicationHits:
    def test_far_hit_survives_cap_over_closer_routine_notes(self) -> None:
        routine = tuple(_note(-h) for h in (1, 2, 3, 4, 5, 6))
        hit = _note(-20, "Action: assess\nResponse: melena 300 ml")
        out = split_focus_notes_5_5(notes=routine + (hit,), anchor=ANCHOR)
        assert len(out) == 5
        assert hit in out

    def test_hits_are_emitted_before_routine_notes_on_each_side(self) -> None:
        # Emission order is what the LLM reads first and what truncation drops
        # last, so the indication note must lead its side.
        before_hit = _note(-8, "Action: assess\nResponse: hematemesis")
        after_hit = _note(8, "Action: assess\nResponse: Hct 21 %")
        notes = (_note(-1), _note(-2), before_hit, _note(1), _note(2), after_hit)
        out = split_focus_notes_5_5(notes=notes, anchor=ANCHOR)
        assert out[0] == before_hit
        assert out[3] == after_hit

    def test_within_a_tier_order_stays_closest_first(self) -> None:
        hits = tuple(_note(-h, f"Action: a\nResponse: bleeding {h}") for h in (7, 3, 5))
        out = split_focus_notes_5_5(notes=hits, anchor=ANCHOR)
        assert [n.timestamp for n in out] == [
            ANCHOR - timedelta(hours=h) for h in (3, 5, 7)
        ]

    def test_no_hits_keeps_legacy_closest_first_selection(self) -> None:
        notes = tuple(_note(-h) for h in (1, 2, 3, 4, 5, 6, 7, 8))
        out = split_focus_notes_5_5(notes=notes, anchor=ANCHOR)
        assert sorted((ANCHOR - n.timestamp).total_seconds() / 3600 for n in out) == [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ]

    def test_selection_is_invariant_under_input_shuffle(self) -> None:
        notes = (
            _note(-1),
            _note(-9, "Action: a\nResponse: bleeding"),
            _note(-2),
            _note(-3),
            _note(-4),
            _note(-5),
            _note(-6),
        )
        forward = split_focus_notes_5_5(notes=notes, anchor=ANCHOR)
        reversed_ = split_focus_notes_5_5(notes=tuple(reversed(notes)), anchor=ANCHOR)
        assert forward == reversed_


class TestPilotFocusJoinIsLabelled:
    def test_action_and_response_are_labelled_so_the_builder_can_split_them(
        self,
    ) -> None:
        focus = {
            "AN": "AN1",
            "PROGRESSDATE": "2025-01-03 00:00:00.000",
            "PROGRESSTIME": "141500",
            "ACTION": "Observe signs bleeding",
            "RESPONSE": "no bleed",
        }
        (note,) = vitals_notes_for([], [focus], "AN1", _ORDER_UTC)
        assert note.text == "Action: Observe signs bleeding\nResponse: no bleed"
        assert not focus_indication_hit(note.text)

    def test_missing_column_is_omitted_not_labelled_empty(self) -> None:
        focus = {
            "AN": "AN1",
            "PROGRESSDATE": "2025-01-03 00:00:00.000",
            "PROGRESSTIME": "141500",
            "ACTION": "observe neuro signs",
        }
        (note,) = vitals_notes_for([], [focus], "AN1", _ORDER_UTC)
        assert note.text == "Action: observe neuro signs"
