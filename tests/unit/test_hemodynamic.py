"""RED-phase failing tests for issue #76 (bba.vitals_extractor.hemodynamic).

The hemodynamic scan is a SEPARATE pass from the single-note ``extract_vitals``
selection (issue #6). It scans EVERY in-window note for the lowest measured MAP
(the nadir) and any vasopressor mention, returning a PHI-free, fact-only
:class:`HemodynamicSummary`. This is the evidence channel that was starved in
Case 2 / REQNO 68012352: the source notes said ``ABP = 84/49 MAP 56`` and
``on NE / Levophed`` but none of it reached the LLM.

Each ``class`` maps to one acceptance behaviour. Tests assert the WHY (nadir is
the worst value across the window, targets are not measurements, ambiguous
abbreviations like NAD are never drugs), not the regex shape.

BINDING GUARDRAIL under test: the summary carries only facts. There is no
"refractory"/"escalating"/appropriateness field; hemodynamic instability is one
supporting factor, never a standalone verdict. These tests must not introduce
any such field.

No implementation exists yet; the module-level imports double as the public-API
surface check.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bba.vitals_extractor import (
    MAP_MAX,
    MAP_MIN,
    HemodynamicSummary,
    VasopressorMention,
    VitalsNote,
    is_map_valid,
    scan_hemodynamics,
)

ANCHOR = datetime(2026, 5, 14, 12, 0, 0, tzinfo=timezone.utc)


def _note(
    source: str,
    minutes: int,
    text: str,
    *,
    base: datetime = ANCHOR,
) -> VitalsNote:
    """Build a :class:`VitalsNote` ``minutes`` after ``base`` (negative = before)."""
    return VitalsNote(
        source=source,  # type: ignore[arg-type]  # narrowed by VitalsNote
        timestamp=base + timedelta(minutes=minutes),
        text=text,
    )


# =============================================================================
# AC: MAP sanity bounds filter OCR noise without second-guessing clinicians.
# =============================================================================
class TestMapBounds:
    def test_bounds_are_inclusive(self) -> None:
        assert is_map_valid(MAP_MIN) is True
        assert is_map_valid(MAP_MAX) is True

    def test_out_of_range_rejected(self) -> None:
        assert is_map_valid(MAP_MIN - 1) is False
        assert is_map_valid(MAP_MAX + 1) is False

    def test_septic_shock_nadir_is_in_range(self) -> None:
        # A MAP of 56 (Case 2) is a genuine clinical outlier the auditor must
        # see, not noise. The bounds exist to drop misparses, not real lows.
        assert is_map_valid(56) is True


# =============================================================================
# AC: MAP nadir = the LOWEST measured MAP across the whole window, never the
# most-recent value. Septic shock is defined by the worst point, not the latest.
# =============================================================================
class TestMapNadir:
    def test_nadir_is_minimum_not_most_recent(self) -> None:
        notes = [
            _note("IPDADMPROGRESS", -120, "ABP = 85/51 MAP 65"),
            _note("IPDNRFOCUSDT", -60, "ABP = 84/49 MAP 56"),  # the nadir
            _note("IPDADMPROGRESS", -30, "MAP 72"),  # most recent, higher
        ]
        summary = scan_hemodynamics(notes)
        assert summary.map_nadir == 56

    def test_nadir_provenance_points_at_the_worst_note(self) -> None:
        worst = _note("IPDNRFOCUSDT", -60, "ABP = 84/49 MAP 56")
        notes = [
            _note("IPDADMPROGRESS", -120, "MAP 65"),
            worst,
            _note("IPDADMPROGRESS", -30, "MAP 72"),
        ]
        summary = scan_hemodynamics(notes)
        assert summary.map_nadir_at == worst.timestamp
        assert summary.map_nadir_source == "IPDNRFOCUSDT"

    def test_abp_notation_does_not_block_map_detection(self) -> None:
        # The legacy BP regex is anchored on \bBP and cannot see ABP; the
        # diastolic 49 must never be misread as the MAP either.
        summary = scan_hemodynamics([_note("IPDADMPROGRESS", -30, "ABP = 84/49 MAP 56")])
        assert summary.map_nadir == 56

    def test_target_map_is_not_a_measurement(self) -> None:
        # "keep MAP >/= 65" is an order/goal, not a recorded value. Counting it
        # would invent a measurement the clinician never charted.
        summary = scan_hemodynamics([_note("IPDADMPROGRESS", -30, "keep MAP >/= 65")])
        assert summary.map_nadir is None

    def test_target_keyword_without_operator_is_not_a_measurement(self) -> None:
        summary = scan_hemodynamics([_note("IPDADMPROGRESS", -30, "goal MAP 65")])
        assert summary.map_nadir is None

    def test_measurement_and_target_in_same_note_keeps_only_measurement(self) -> None:
        # Case 2 shape: a charted nadir plus the resuscitation target.
        summary = scan_hemodynamics(
            [_note("IPDNRFOCUSDT", -45, "ABP = 84/49 MAP 56, keep MAP >/= 65")]
        )
        assert summary.map_nadir == 56

    def test_out_of_bound_map_never_becomes_the_nadir(self) -> None:
        # "MAP 8" is a misparse; the true nadir is the in-range 56, not 8.
        notes = [
            _note("IPDADMPROGRESS", -60, "MAP 56"),
            _note("IPDADMPROGRESS", -30, "MAP 8"),
        ]
        summary = scan_hemodynamics(notes)
        assert summary.map_nadir == 56


# =============================================================================
# AC: Real KCMH charting writes MAP as the PARENTHESISED value after the
# arterial pressure -- "ABP 77/44 (56)" -- not the labelled "MAP 56" form the
# issue body idealised. The pilot run against the encrypted bundle proved the
# labelled regex matched nothing on real data (Case 2 / REQNO 68012352), so the
# parenthesised form must be recognised too. A physiological guard keeps the
# parenthesised number only when it sits between diastolic and systolic, so an
# unrelated bracketed integer can never be fabricated into a MAP.
# =============================================================================
class TestAbpParenthesizedMap:
    def test_parenthesized_map_with_space(self) -> None:
        # Verbatim shape from the encrypted bundle (REQNO 68012352 nurse note).
        summary = scan_hemodynamics(
            [_note("IPDNRFOCUSDT", -30, "ABP 77/44 (56) mmHg., HR 92 bpm")]
        )
        assert summary.map_nadir == 56

    def test_parenthesized_map_without_space(self) -> None:
        summary = scan_hemodynamics(
            [_note("IPDNRFOCUSDT", -30, "ABP = 120/58(83) mmHg")]
        )
        assert summary.map_nadir == 83

    def test_parenthesized_value_outside_bp_range_is_not_a_map(self) -> None:
        # 140 is within the MAP sanity bounds (30-180) but ABOVE systolic 70, so
        # it cannot physiologically be a MAP -- the guard must reject it rather
        # than invent a hypertensive reading from a stray bracketed number.
        summary = scan_hemodynamics([_note("IPDADMPROGRESS", -30, "ABP 70/50 (140)")])
        assert summary.map_nadir is None

    def test_labeled_and_parenthesized_forms_coexist(self) -> None:
        # Mixed charting across the window: the nadir is still the lowest MAP,
        # regardless of which notation each note used.
        notes = [
            _note("IPDADMPROGRESS", -120, "ABP = 85/51 MAP 65"),
            _note("IPDNRFOCUSDT", -60, "ABP 77/44 (56) mmHg"),  # the nadir
            _note("IPDNRFOCUSDT", -30, "ABP = 120/58(83) mmHg"),
        ]
        summary = scan_hemodynamics(notes)
        assert summary.map_nadir == 56

    def test_real_thai_note_yields_map_and_vasopressor(self) -> None:
        # The exact starved-evidence shape: a Thai-prose nurse note carrying the
        # parenthesised ABP MAP AND "Levophed" in the same line.
        text = "D: ผู้ป่วย on A-line Lt. Radial monitor ABP 77/44 (56) mmHg., EKG SR, HR 92 bpm, on Levophed"
        summary = scan_hemodynamics([_note("IPDNRFOCUSDT", -30, text)])
        assert summary.map_nadir == 56
        assert {v.agent for v in summary.vasopressors} == {"norepinephrine"}


# =============================================================================
# AC: Vasopressor detection by unambiguous name. Ambiguous clinical
# abbreviations (NAD = no acute distress, Na = sodium) are NEVER drugs.
# =============================================================================
class TestVasopressorDetection:
    def test_levophed_maps_to_norepinephrine(self) -> None:
        summary = scan_hemodynamics([_note("IPDNRFOCUSDT", -30, "on NE / Levophed")])
        agents = {v.agent for v in summary.vasopressors}
        assert agents == {"norepinephrine"}

    def test_same_agent_named_twice_is_one_mention(self) -> None:
        summary = scan_hemodynamics(
            [_note("IPDNRFOCUSDT", -30, "Levophed running, titrate Levophed, NE on")]
        )
        norepi = [v for v in summary.vasopressors if v.agent == "norepinephrine"]
        assert len(norepi) == 1

    def test_distinct_agents_are_separate_mentions(self) -> None:
        summary = scan_hemodynamics(
            [_note("IPDADMPROGRESS", -30, "on norepinephrine and vasopressin")]
        )
        agents = {v.agent for v in summary.vasopressors}
        assert agents == {"norepinephrine", "vasopressin"}

    def test_dose_is_captured_when_present(self) -> None:
        summary = scan_hemodynamics(
            [_note("IPDADMPROGRESS", -30, "norepinephrine 0.1 mcg/kg/min")]
        )
        norepi = next(v for v in summary.vasopressors if v.agent == "norepinephrine")
        assert norepi.dose is not None
        assert "0.1" in norepi.dose

    def test_dose_is_optional(self) -> None:
        summary = scan_hemodynamics([_note("IPDNRFOCUSDT", -30, "on Levophed")])
        norepi = next(v for v in summary.vasopressors if v.agent == "norepinephrine")
        assert norepi.dose is None

    def test_ambiguous_abbreviations_are_not_vasopressors(self) -> None:
        # NAD = "no acute distress", Na = sodium, N/A = not applicable. Treating
        # any of these as norepinephrine would fabricate vasopressor support and
        # corrupt the hemodynamic evidence. This is a clinical-safety guardrail.
        summary = scan_hemodynamics(
            [_note("IPDADMPROGRESS", -30, "GA: NAD, Na 140, urine output N/A")]
        )
        assert summary.vasopressors == ()

    def test_vasopressor_provenance_is_recorded(self) -> None:
        note = _note("IPDNRFOCUSDT", -45, "on Levophed")
        summary = scan_hemodynamics([note])
        mention = summary.vasopressors[0]
        assert mention.at == note.timestamp
        assert mention.source == "IPDNRFOCUSDT"


# =============================================================================
# AC: Thai-language prose is handled (real KCMH notes mix Thai with English
# drug names and vital tokens).
# =============================================================================
class TestThaiProse:
    def test_thai_note_yields_map_and_vasopressor(self) -> None:
        text = "ผู้ป่วยความดันต่ำ ABP = 84/49 MAP 56 on Levophed keep MAP >/= 65"
        summary = scan_hemodynamics([_note("IPDNRFOCUSDT", -30, text)])
        assert summary.map_nadir == 56
        assert {v.agent for v in summary.vasopressors} == {"norepinephrine"}


# =============================================================================
# AC: Absence is null, not zero or empty-with-flags. The scan is total.
# =============================================================================
class TestEmptyAndNullable:
    def test_empty_note_list_is_empty_summary(self) -> None:
        summary = scan_hemodynamics([])
        assert isinstance(summary, HemodynamicSummary)
        assert summary.is_empty is True
        assert summary.map_nadir is None
        assert summary.vasopressors == ()

    def test_notes_without_hemodynamics_are_empty(self) -> None:
        summary = scan_hemodynamics(
            [_note("IPDADMPROGRESS", -30, "patient comfortable, tolerating diet")]
        )
        assert summary.is_empty is True

    def test_is_empty_false_when_anything_present(self) -> None:
        only_map = scan_hemodynamics([_note("IPDADMPROGRESS", -30, "MAP 56")])
        only_pressor = scan_hemodynamics([_note("IPDNRFOCUSDT", -30, "on Levophed")])
        assert only_map.is_empty is False
        assert only_pressor.is_empty is False


# =============================================================================
# AC: HemodynamicSummary is immutable (frozen) — audit-chain reproducibility.
# =============================================================================
class TestImmutability:
    def test_summary_is_frozen(self) -> None:
        summary = scan_hemodynamics([_note("IPDADMPROGRESS", -30, "MAP 56")])
        with pytest.raises((TypeError, AttributeError, ValueError)):
            summary.map_nadir = 99  # type: ignore[misc]

    def test_mention_is_frozen(self) -> None:
        mention = VasopressorMention(
            agent="norepinephrine",
            dose=None,
            at=ANCHOR,
            source="IPDNRFOCUSDT",
        )
        with pytest.raises((TypeError, AttributeError, ValueError)):
            mention.agent = "dopamine"  # type: ignore[misc]
