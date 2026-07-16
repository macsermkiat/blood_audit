"""Pure MSBOS reservation judgment tests for ticket #163."""

from __future__ import annotations

import pytest

from bba.preop_reservation import evaluate_reservation
from bba.preop_reservation.reference import _reference_from_rows


_HASH = "b" * 64
_ROWS = [
    {"icd9_code_nodot": "1000", "msbos": "none", "recommended_units": ""},
    {"icd9_code_nodot": "2000", "msbos": "G/M", "recommended_units": "2"},
    {"icd9_code_nodot": "3000", "msbos": "T/S", "recommended_units": "1"},
    {"icd9_code_nodot": "4000", "msbos": "G/M", "recommended_units": "1"},
    {"icd9_code_nodot": "4000", "msbos": "G/M", "recommended_units": "2"},
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
