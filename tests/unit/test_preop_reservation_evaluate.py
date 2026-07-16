"""Pure MSBOS reservation judgment tests for ticket #163."""

from __future__ import annotations

import pytest

from bba.preop_reservation import (
    ReservationDecision,
    evaluate_reservation,
    evaluate_reservation_with_notes,
)
from bba.preop_reservation.reference import _reference_from_rows


_HASH = "b" * 64
_ROWS = [
    {"icd9_code_nodot": "1000", "msbos": "none", "recommended_units": ""},
    {"icd9_code_nodot": "2000", "msbos": "G/M", "recommended_units": "2"},
    {"icd9_code_nodot": "3000", "msbos": "T/S", "recommended_units": "1"},
    {
        "icd9_code_nodot": "4000",
        "operation": "Operation Alpha",
        "msbos": "G/M",
        "recommended_units": "1",
    },
    {
        "icd9_code_nodot": "4000",
        "operation": "Operation Beta",
        "msbos": "G/M",
        "recommended_units": "2",
    },
]


@pytest.mark.parametrize(
    ("code", "reserved", "reason", "is_over", "recommended"),
    [
        ("1000", 1, "over_none", True, 0),
        ("1000", 0, "within_recommendation", False, 0),
        ("2000", 3, "over_gm_excess", True, 2),
        ("2000", 2, "within_recommendation", False, 2),
        ("2000", 1, "within_recommendation", False, 2),
        # T2 (#164): T/S recommended but units crossmatched -> over; zero units
        # is a compliant screen-only reservation.
        ("3000", 8, "over_type_and_screen_crossmatched", True, 1),
        ("3000", 1, "over_type_and_screen_crossmatched", True, 1),
        ("3000", 0, "type_and_screen_screen_only", False, 1),
        ("9999", 2, "unresolved_code", False, 0),
        ("4000", 2, "ambiguous_code", False, 0),
        ("", 2, "no_planned_op", False, 0),
    ],
)
def test_evaluate_reservation_rules_table(
    code: str,
    reserved: int,
    reason: str,
    is_over: bool,
    recommended: int,
) -> None:
    reference = _reference_from_rows(_ROWS, content_hash=_HASH)

    decision = evaluate_reservation(
        reserved_units=reserved,
        planned_icd9_nodot=code,
        reference=reference,
    )

    assert decision.reason == reason
    assert decision.is_over is is_over
    assert decision.recommended_units == recommended
    assert decision.reserved_units == reserved
    assert decision.reference_hash == _HASH


def test_evaluate_reservation_is_deterministic_across_reference_input_order() -> None:
    forward = _reference_from_rows(_ROWS, content_hash=_HASH)
    reverse = _reference_from_rows(reversed(_ROWS), content_hash=_HASH)

    first = evaluate_reservation(
        reserved_units=3,
        planned_icd9_nodot=" 2000 ",
        reference=forward,
    )
    second = evaluate_reservation(
        reserved_units=3,
        planned_icd9_nodot="2000",
        reference=reverse,
    )

    assert first == second


def test_reservation_decision_dump_includes_in_run_note_resolution_signal() -> None:
    decision = ReservationDecision(
        reason="within_recommendation",
        reference_hash=_HASH,
        note_resolved=True,
    )

    assert decision.model_dump()["note_resolved"] is True, (
        "T5 needs the explicit in-run note-resolution signal in model dumps; no "
        "production ReservationDecision byte serialization exists"
    )


def test_ambiguous_code_with_one_note_match_applies_normal_over_rules() -> None:
    reference = _reference_from_rows(_ROWS, content_hash=_HASH)

    decision = evaluate_reservation_with_notes(
        reserved_units=3,
        planned_icd9_nodot="4000",
        reference=reference,
        note_texts=["Consent completed for Operation Beta."],
    )

    assert decision.reason == "over_gm_excess", (
        "a uniquely note-resolved operation must re-enter the frozen MSBOS rule table"
    )
    assert decision.is_over is True
    assert decision.recommended_units == 2
    assert decision.note_resolved is True, (
        "the decision must distinguish note recovery from direct code resolution"
    )


@pytest.mark.parametrize(
    "note_texts",
    [
        ["No operation named."],
        ["Operation Alpha and Operation Beta were both discussed."],
    ],
    ids=["zero-matches", "multiple-distinct-recommendations"],
)
def test_ambiguous_code_without_unique_recommendation_needs_review(
    note_texts: list[str],
) -> None:
    reference = _reference_from_rows(_ROWS, content_hash=_HASH)

    decision = evaluate_reservation_with_notes(
        reserved_units=3,
        planned_icd9_nodot="4000",
        reference=reference,
        note_texts=note_texts,
    )

    assert decision.reason == "operation_unresolved", (
        "zero or multiple note matches must route to review instead of guessing"
    )
    assert decision.is_over is False
    assert decision.note_resolved is False


@pytest.mark.parametrize("code", ["", "9999", "2000"])
def test_notes_evaluator_is_identical_for_non_ambiguous_inputs(code: str) -> None:
    reference = _reference_from_rows(_ROWS, content_hash=_HASH)

    plain = evaluate_reservation(
        reserved_units=3, planned_icd9_nodot=code, reference=reference
    )
    with_notes = evaluate_reservation_with_notes(
        reserved_units=3,
        planned_icd9_nodot=code,
        reference=reference,
        note_texts=["Operation Beta"],
    )

    assert with_notes == plain, (
        "notes may affect only conflicting-code ambiguity, never established behavior"
    )
