"""Tests for the affirmative-only administration evidence scan (issue #107).

The scan can confirm administration when a charted marker exists, but it can
never deny administration. In particular, an empty summary means UNKNOWN — it
is not a representation of "not transfused".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bba.vitals_extractor import (
    AdministrationFinding,
    AdministrationSummary,
    VitalsNote,
    administration_citation_has_negative_context,
    administration_citation_supports_red_cell,
    scan_administration,
)

ANCHOR = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)


def _note(source: str, minutes: int, text: str) -> VitalsNote:
    """Build a :class:`VitalsNote` ``minutes`` from :data:`ANCHOR`."""
    return VitalsNote(
        source=source,  # type: ignore[arg-type]  # narrowed by VitalsNote
        timestamp=ANCHOR + timedelta(minutes=minutes),
        text=text,
    )


class TestAffirmativeMarkers:
    @pytest.mark.parametrize(
        ("text", "category", "marker"),
        [
            ("ให้เลือดแล้ว", "gave_blood", "ให้เลือด"),
            ("ให้ LPRC 2 unit", "gave_blood", "ให้ LPRC"),
            ("PRC 2 units transfused", "unit_count", "PRC 2 units"),
            ("ได้รับ 2 ยูนิต LPRC", "unit_count", "2 ยูนิต LPRC"),
            ("post-transfusion no reaction", "post_transfusion", "post-transfusion"),
            ("หลังให้เลือด อาการคงที่", "post_transfusion", "หลังให้เลือด"),
            ("transfusion reaction absent", "post_transfusion", "transfusion reaction"),
        ],
    )
    def test_positive_marker_has_category_and_provenance(
        self, text: str, category: str, marker: str
    ) -> None:
        note = _note("IPDNRFOCUSDT", 30, text)
        summary = scan_administration([note])
        finding = next(f for f in summary.findings if f.category == category)
        assert summary.has_affirmative_marker is True
        assert marker in finding.snippet
        assert finding.at == note.timestamp
        assert finding.source == "IPDNRFOCUSDT"


class TestNegativeContextGuard:
    @pytest.mark.parametrize(
        "text",
        [
            "จะให้เลือด",
            "เตรียมให้เลือด",
            "แผนให้ FFP",
            "plan to give FFP 2 units",
            "ส่ง LPRC ไป cath lab",
            "G/M LPRC 2 unit",
            "จอง LPRC 2 u",
            "ไม่ได้ให้เลือด",
            "ไม่ให้เลือด",
            "ยังไม่ได้ให้ LPRC",
            "งดให้เลือด",
            "no PRC 2 units transfused",
            "LPRC not given",
            "no history of transfusion reaction",
            "ประวัติ transfusion reaction",
            "pre-transfusion check completed",
            "ไม่ได้รับ LPRC 2 units",
            "LPRC 2 units not received",
            "ปฏิเสธการให้เลือด",
            "patient refused PRC 2 units",
        ],
    )
    def test_guarded_line_does_not_count(self, text: str) -> None:
        summary = scan_administration([_note("IPDADMPROGRESS", -30, text)])
        assert summary.has_affirmative_marker is False
        assert summary.findings == ()

    def test_guard_is_scoped_to_one_line(self) -> None:
        note = _note(
            "IPDNRFOCUSDT",
            10,
            "แผนให้ FFP 2 units\nให้เลือดแล้ว ผู้ป่วยอาการคงที่",
        )
        summary = scan_administration([note])
        assert summary.has_affirmative_marker is True
        gave = next(f for f in summary.findings if f.category == "gave_blood")
        assert "ให้เลือดแล้ว" in gave.snippet

    def test_case_68026306_dispatch_citation_is_negative_context(self) -> None:
        quote = "ดูแลประสามงานธนาคารเลือด ส่ง LPRC 4 unit ไป cath lab"
        source = f"Nursing record: {quote}"

        assert administration_citation_has_negative_context(source, quote) is True

    def test_genuine_administration_citation_is_not_negative_context(self) -> None:
        quote = "ได้รับ LPRC 1 unit at OR"
        source = f"Nursing record: {quote} ผู้ป่วยอาการคงที่"

        assert administration_citation_has_negative_context(source, quote) is False

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            ("ดูแลให้ได้รับ Cryo 10 unit and FFP 2 unit", False),
            ("Blood loss in OR 400 ml", False),
            ("ได้รับ LPRC 1 unit at OR", True),
            ("no complication ขณะได้รับเลือด", True),
            ("หลังให้เลือด ผู้ป่วยอาการคงที่", True),
        ],
    )
    def test_red_cell_citation_requirement(self, line: str, expected: bool) -> None:
        source = f"Nursing record: {line}"

        assert administration_citation_supports_red_cell(source, line) is expected

    def test_transplant_does_not_trigger_word_bounded_plan_guard(self) -> None:
        summary = scan_administration(
            [_note("IPDADMPROGRESS", -5, "transplant service: ให้เลือดแล้ว")]
        )
        assert summary.has_affirmative_marker is True
        assert any(f.category == "gave_blood" for f in summary.findings)

    @pytest.mark.parametrize(
        "text",
        [
            "post transfusion no fever",
            "no transfusion reaction",
        ],
    )
    def test_post_transfusion_reaction_checks_are_not_suppressed(
        self, text: str
    ) -> None:
        summary = scan_administration([_note("IPDNRFOCUSDT", 5, text)])
        assert summary.has_affirmative_marker is True
        assert any(f.category == "post_transfusion" for f in summary.findings)


class TestIssue117ResidualCitations:
    """Snippet-level pins for the three reserve-ahead residuals (#117).

    Each REJECT case is the exact narrative line that falsely confirmed
    administration through the extractor or a grounded LLM citation; each
    PRESERVE case is a genuine red-cell administration that MUST still confirm.
    The PRESERVE cases encode WHY the new guards are narrow: the fix rejects
    counselling / dispatch / crossmatch / pre-transfusion / history-arrow text
    WITHOUT a blanket "ได้ LPRC" rejection that would erase real transfusions.
    """

    # -- 68046079: education FFP note + transport-to-OR --------------------
    def test_case_68046079_counselling_line_is_not_a_marker(self) -> None:
        line = (
            "A : - แจ้งให้ผู้ป่วยทราบถึงความจำเป็นในการให้เลือด วิธีการให้เลือด "
            "อาการและภาวะแทรกซ้อนที่อาจจะเกิดขึ้นได้ขณะและหลังให้เลือดตามความจำเป็น"
        )
        assert scan_administration([_note("IPDNRFOCUSDT", 0, line)]).is_empty is True

    def test_case_68046079_complication_watch_line_is_not_a_marker(self) -> None:
        line = (
            "- Observe complication ขณะให้เลือด ได้แก่ มีผื่นคัน ไข้สูง หนาวสั่น "
            "ถ้ามีอาการแพ้เลือด ให้หยุดการให้เลือรายงานแพทย์ทันที"
        )
        assert scan_administration([_note("IPDNRFOCUSDT", 5, line)]).is_empty is True

    def test_case_68046079_transport_to_or_is_not_a_marker(self) -> None:
        line = "- ให้นำ LPRC 2 unit , FFP 2 unit IV to OR"
        assert scan_administration([_note("IPDNRFOCUSDT", 10, line)]).is_empty is True
        assert (
            administration_citation_has_negative_context(f"Nursing: {line}", line)
            is True
        )

    def test_case_68046079_pre_transfusion_vitals_is_negative_context(self) -> None:
        line = (
            "- ดูเเลเจาะ lab INR PT PTT พรุ่งนี้เช้า + ติดตามผล R:  at 00.15น. "
            "V/Sก่อนให้เลือด BT = 37.7  C  PR = 109  bpm"
        )
        quote = "at 00.15น. V/Sก่อนให้เลือด BT = 37.7  C"
        assert administration_citation_has_negative_context(line, quote) is True

    # -- 68080335: reversed crossmatch shorthand ---------------------------
    def test_case_68080335_reversed_crossmatch_is_not_a_marker(self) -> None:
        line = "* Notify ผล Lab ดูเเล M/G LPRC 2Unit ได้เเล้วให้เลย"
        assert scan_administration([_note("IPDNRFOCUSDT", 0, line)]).is_empty is True
        assert (
            administration_citation_has_negative_context(f"Nursing: {line}", line)
            is True
        )

    # -- 68055153: retrospective history-summary arrow ---------------------
    def test_case_68055153_history_arrow_is_negative_context(self) -> None:
        line = "- 2/9/68 Hb=7.6 ,Hct=23.3 ,Plt= 354000, PTT=36 --> LPRC 1 unit"
        quote = "2/9/68 Hb=7.6 ,Hct=23.3 ,Plt= 354000, PTT=36 --> LPRC 1 unit"
        assert administration_citation_has_negative_context(line, quote) is True

    # -- PRESERVE: genuine administrations must still confirm --------------
    @pytest.mark.parametrize(
        "line",
        [
            "ให้เลือด LPRC 2 unit v drip in 3 hr at 15.00 น.",
        ],
    )
    def test_genuine_administration_still_marks(self, line: str) -> None:
        assert scan_administration(
            [_note("IPDNRFOCUSDT", 0, line)]
        ).has_affirmative_marker

    @pytest.mark.parametrize(
        "quote",
        [
            # near-identical in FORM to 68080335's rejected history citation,
            # but a current intra-HD / at-OR administration: must NOT be guarded.
            "ได้ LPRC 1 unit iv intra HD",
            "ได้รับ LPRC 1 unit at OR",
        ],
    )
    def test_genuine_administration_citation_survives(self, quote: str) -> None:
        source = f"Nursing record: {quote} ผู้ป่วยอาการคงที่"
        assert administration_citation_has_negative_context(source, quote) is False

    def test_reversed_crossmatch_guard_does_not_match_milligrams(self) -> None:
        # \bM/G\b requires the slash, so "mg" (milligrams) never trips it.
        quote = "Gentamicin 400 mg iv drip in 1 hr"
        source = f"Nursing record: {quote}"
        assert administration_citation_has_negative_context(source, quote) is False

    # The complication cue is anchored on the prospective "ขณะ(ให้|ได้รับ)"
    # instruction form: a COMPLETED "no complication post-transfusion" note
    # that carries its own affirmative marker must still confirm.
    def test_completed_post_transfusion_complication_note_still_marks(self) -> None:
        line = (
            "Observe complication post-transfusion: none, ได้รับเลือด LPRC 1 unit เรียบร้อย"
        )
        assert scan_administration(
            [_note("IPDNRFOCUSDT", 0, line)]
        ).has_affirmative_marker

    # The history-arrow cue requires a leading DD/MM/YY date: a bare "-->" used
    # as a live "therefore" connective must NOT be treated as negative context.
    def test_undated_arrow_connective_is_not_negative_context(self) -> None:
        quote = "anemia --> เลือด LPRC 1 unit ให้แล้วเสร็จสิ้น 15.00 น."
        source = f"Progress: {quote}"
        assert administration_citation_has_negative_context(source, quote) is False

    # "นำ ... ไปให้ผู้ป่วย" (carry to give TO the patient) is administration,
    # not dispatch: the ไป/to guard is word-bounded so "ไปให้" does not trip it.
    def test_carry_to_give_to_patient_still_marks(self) -> None:
        line = "นำ LPRC 1 unit ไปให้ผู้ป่วย ได้รับเลือดครบถ้วน"
        assert scan_administration(
            [_note("IPDNRFOCUSDT", 0, line)]
        ).has_affirmative_marker

    # Codex #118 P2: the counselling cue requires ผู้ป่วย, so "แจ้งแพทย์ทราบ"
    # (notified the doctor) next to a genuine administration must still confirm.
    def test_notify_doctor_beside_administration_still_confirms(self) -> None:
        quote = "แจ้งแพทย์ทราบ ได้รับ LPRC 1 unit at OR"
        source = f"Nursing record: {quote}"
        assert administration_citation_has_negative_context(source, quote) is False

    # Codex #118 P2: a date mid-sentence (not at line start) is a live "-->"
    # connective, not a dated history entry, so it must still confirm.
    def test_midline_dated_arrow_is_not_negative_context(self) -> None:
        quote = "anemia on 12/07/68 --> เลือด LPRC 1 unit ให้แล้วเสร็จสิ้น"
        source = f"Progress: {quote}"
        assert administration_citation_has_negative_context(source, quote) is False


class TestNonMarkers:
    @pytest.mark.parametrize(
        "text",
        [
            "EBL 1500 ml",
            "intra-op blood loss 1500 ml",
            "crossmatch for OR",
            "เลือด 2 unit",
            # A bare or order-restating component+count is the audited order
            # itself, not administration (Codex round 2 on PR #112).
            "LPRC 2 units",
            "order LPRC 2 units",
            "แพทย์ order PRC 2 units",
            # Non-red-cell products given do not confirm the reserved red
            # cells were given (Codex round 5 on PR #112).
            "ให้ FFP 2 units",
            "ให้ SDP 1 unit",
            "platelets 6 units given",
            # An administered verb bound to something OTHER than the
            # component-count span must not count (Codex round 6 on PR #112).
            "order PRC 2 units; patient received Lasix after that dose today",
            "order PRC 2 units. patient received Lasix after that dose today",
            "not yet received PRC 2 units",
        ],
    )
    def test_non_marker_does_not_count(self, text: str) -> None:
        summary = scan_administration([_note("IPDNRFOCUSDT", 20, text)])
        assert summary.has_affirmative_marker is False
        assert summary.is_empty is True


class TestAggregation:
    def test_empty_input_means_unknown_not_non_administration(self) -> None:
        summary = scan_administration([])
        # UNKNOWN: no affirmative marker was found. This does not mean that the
        # patient was not transfused, because the hospital never records status 5.
        assert summary.has_affirmative_marker is False
        assert summary.is_empty is True
        assert summary.findings == ()

    def test_marker_free_input_means_unknown_not_non_administration(self) -> None:
        summary = scan_administration(
            [_note("IPDADMPROGRESS", 0, "patient comfortable")]
        )
        # Still UNKNOWN, never evidence of non-administration.
        assert summary.has_affirmative_marker is False
        assert summary.is_empty is True

    def test_one_finding_per_category_and_earliest_wins(self) -> None:
        early = _note("IPDADMPROGRESS", -20, "ให้เลือดแล้ว")
        late = _note("IPDNRFOCUSDT", 20, "ให้ PRC")
        summary = scan_administration([late, early])
        gave = [f for f in summary.findings if f.category == "gave_blood"]
        assert len(gave) == 1
        assert gave[0].at == early.timestamp

    def test_deterministic_across_input_shuffle(self) -> None:
        notes = [
            _note("IPDNRFOCUSDT", 30, "post-transfusion stable"),
            _note("IPDADMPROGRESS", -30, "PRC 2 units transfused"),
            _note("IPDNRFOCUSDT", 0, "ให้เลือดแล้ว"),
        ]
        forward = scan_administration(notes)
        reverse = scan_administration(list(reversed(notes)))
        assert forward == reverse


class TestFactOnlyContract:
    def test_summary_fields_are_exactly_the_fact_set(self) -> None:
        assert set(AdministrationSummary.model_fields) == {
            "has_affirmative_marker",
            "findings",
        }

    def test_finding_fields_are_exactly_the_fact_set(self) -> None:
        assert set(AdministrationFinding.model_fields) == {
            "category",
            "snippet",
            "at",
            "source",
        }


class TestImmutability:
    def test_summary_is_frozen(self) -> None:
        summary = scan_administration([_note("IPDNRFOCUSDT", 0, "ให้เลือด")])
        with pytest.raises((TypeError, AttributeError, ValueError)):
            summary.has_affirmative_marker = False  # type: ignore[misc]

    def test_finding_is_frozen(self) -> None:
        finding = AdministrationFinding(
            category="gave_blood",
            snippet="ให้เลือด",
            at=ANCHOR,
            source="IPDNRFOCUSDT",
        )
        with pytest.raises((TypeError, AttributeError, ValueError)):
            finding.category = "unit_count"  # type: ignore[misc]
