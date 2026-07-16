"""Clinical-note operation matching tests for ticket T3."""

from __future__ import annotations

from bba.preop_reservation.models import CandidateOperation, MsbosRow
from bba.preop_reservation.note_operation import resolve_operation_from_notes


def _candidate(
    operation: str, *, msbos: str = "G/M", recommended_units: int = 2
) -> CandidateOperation:
    return CandidateOperation(
        operation=operation,
        msbos=msbos,
        recommended_units=recommended_units,
    )


def test_single_full_operation_phrase_resolves_its_recommendation() -> None:
    resolved = resolve_operation_from_notes(
        candidates=[
            _candidate("Craniotomy for tumor", recommended_units=4),
            _candidate("Craniotomy for aneurysm", recommended_units=2),
        ],
        note_texts=["Plan: craniotomy for tumor tomorrow."],
    )

    assert resolved == MsbosRow(msbos="G/M", recommended_units=4), (
        "one full candidate phrase in the shipped notes should recover its exact "
        "MSBOS recommendation"
    )


def test_zero_operation_matches_refuses_to_guess() -> None:
    resolved = resolve_operation_from_notes(
        candidates=[_candidate("Craniotomy for tumor")],
        note_texts=["Plan: neurologic observation."],
    )

    assert resolved is None, "unmatched prose must not be treated as an operation"


def test_candidate_phrase_split_across_two_notes_does_not_match() -> None:
    # Matching is per-note: a candidate whose tokens are spread across SEPARATE
    # notes must not resolve. A future refactor that concatenated notes would
    # falsely join "Craniotomy" + "tumor" and resolve the wrong recommendation.
    resolved = resolve_operation_from_notes(
        candidates=[
            _candidate("Craniotomy tumor", recommended_units=4),
            _candidate("Craniotomy aneurysm", recommended_units=2),
        ],
        note_texts=["Craniotomy", "tumor"],
    )

    assert resolved is None, (
        "a candidate phrase split across two separate notes must not resolve "
        "(no cross-note concatenation)"
    )


def test_generic_alias_subsumed_by_specific_name_does_not_false_resolve() -> None:
    # Real reference shape (ICD-9 8151): "THA" (G/M 1) is a sub-phrase of
    # "THA, revision THA" (G/M 2). A note about the revision case must NOT
    # resolve to the primary-THA (lower-unit) row via the generic name — 81/120
    # conflicting codes carry such a subsumption pair.
    candidates = [
        _candidate("THA", recommended_units=1),
        _candidate("THA, revision THA", recommended_units=2),
    ]
    assert (
        resolve_operation_from_notes(
            candidates=candidates,
            note_texts=["Plan: revision THA next week."],
        )
        is None
    ), "a substring-generic candidate must not resolve a more-specific note"
    assert (
        resolve_operation_from_notes(
            candidates=candidates, note_texts=["THA scheduled."]
        )
        is None
    ), "a bare generic mention is ambiguous with the specific variant"


def test_specific_operation_phrase_still_resolves_over_its_generic_alias() -> None:
    # The longer, specific name still resolves when its FULL phrase is present.
    candidates = [
        _candidate("THA", recommended_units=1),
        _candidate("THA, revision THA", recommended_units=2),
    ]
    assert resolve_operation_from_notes(
        candidates=candidates, note_texts=["Booked for THA, revision THA."]
    ) == MsbosRow(msbos="G/M", recommended_units=2), (
        "the specific operation phrase must still resolve its own recommendation"
    )


def test_two_distinct_recommendations_refuse_to_guess() -> None:
    resolved = resolve_operation_from_notes(
        candidates=[
            _candidate("Craniotomy for tumor", recommended_units=4),
            _candidate("Craniotomy for aneurysm", recommended_units=2),
        ],
        note_texts=["Discussed craniotomy for tumor and craniotomy for aneurysm."],
    )

    assert resolved is None, (
        "matching two clinically different recommendations is still ambiguous"
    )


def test_duplicate_names_with_same_recommendation_resolve_once() -> None:
    resolved = resolve_operation_from_notes(
        candidates=[
            _candidate("Fullhouse FESS", msbos="G/M", recommended_units=0),
            _candidate("Limited FESS", msbos="G/M", recommended_units=0),
        ],
        note_texts=["Fullhouse FESS / Limited FESS documented."],
    )

    assert resolved == MsbosRow(msbos="G/M", recommended_units=0), (
        "raw operation-name duplicates must collapse by recommendation identity"
    )


def test_matching_is_deterministic_across_note_and_candidate_reordering() -> None:
    candidates = [
        _candidate("Craniotomy for tumor", recommended_units=4),
        _candidate("Craniotomy for aneurysm", recommended_units=2),
    ]
    notes = ["No aneurysm operation planned.", "Craniotomy for tumor scheduled."]

    forward = resolve_operation_from_notes(candidates=candidates, note_texts=notes)
    reversed_inputs = resolve_operation_from_notes(
        candidates=list(reversed(candidates)), note_texts=list(reversed(notes))
    )

    assert forward == reversed_inputs == MsbosRow(msbos="G/M", recommended_units=4), (
        "input ordering must not alter a unique matched recommendation"
    )


def test_candidate_token_embedded_in_larger_word_does_not_match() -> None:
    resolved = resolve_operation_from_notes(
        candidates=[_candidate("TUR")],
        note_texts=["The patient returns for follow-up."],
    )

    assert resolved is None, "whole-token matching must not find TUR inside returns"


def test_thai_phrase_matches_across_whitespace_and_punctuation_boundaries() -> None:
    resolved = resolve_operation_from_notes(
        candidates=[_candidate("ผ่าตัด หัวใจ")],
        note_texts=["วางแผน: ผ่าตัด,หัวใจ พรุ่งนี้"],
    )

    assert resolved == MsbosRow(msbos="G/M", recommended_units=2), (
        "Unicode-aware punctuation normalization should preserve Thai token matches"
    )


def test_thai_phrase_embedded_without_space_does_not_match() -> None:
    resolved = resolve_operation_from_notes(
        candidates=[_candidate("ผ่าตัด หัวใจ")],
        note_texts=["วางแผนผ่าตัดหัวใจพรุ่งนี้"],
    )

    assert resolved is None, (
        "the frozen high-precision matcher intentionally rejects Thai text with no "
        "token boundary around the full phrase"
    )
