"""MSBOS reference loading and validation tests for ticket #163."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from importlib import resources

import pytest

from bba.preop_reservation import MsbosRow, load_msbos_reference
from bba.preop_reservation.reference import (
    MSBOS_REFERENCE_FILENAME,
    MsbosReferenceError,
    _reference_from_rows,
    parse_recommended_units,
)


def test_vendored_reference_loads_with_stable_sha256_and_bom_handling() -> None:
    first = load_msbos_reference()
    second = load_msbos_reference()

    assert first is second
    assert re.fullmatch(r"[0-9a-f]{64}", first.content_hash)
    assert first.content_hash == second.content_hash
    reference_path = resources.files("bba.preop_reservation").joinpath(
        "data", MSBOS_REFERENCE_FILENAME
    )
    with reference_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert reader.fieldnames[0] == "icd9_code"
        assert "icd9_code_nodot" in reader.fieldnames


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", 0), ("2", 2), ("1-2", 2)],
)
def test_parse_recommended_units(raw: str, expected: int) -> None:
    assert parse_recommended_units(raw) == expected


def test_parse_recommended_units_rejects_non_numeric_value() -> None:
    with pytest.raises(MsbosReferenceError, match="recommended_units"):
        parse_recommended_units("many")


@pytest.mark.parametrize(
    "row",
    [
        {"icd9_code_nodot": "1234", "msbos": "INVALID", "recommended_units": "2"},
        {"icd9_code_nodot": "1234", "msbos": "G/M", "recommended_units": "bad"},
    ],
)
def test_malformed_schedule_row_is_rejected(row: dict[str, str]) -> None:
    with pytest.raises(MsbosReferenceError, match="row 2"):
        _reference_from_rows([row], content_hash="a" * 64)


def test_resolve_unique_ambiguous_and_absent_codes_data_driven() -> None:
    reference_path = resources.files("bba.preop_reservation").joinpath(
        "data", MSBOS_REFERENCE_FILENAME
    )
    with reference_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    recommendations: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for row in rows:
        code = row["icd9_code_nodot"].strip()
        if code:
            recommendations[code].add(
                (
                    row["msbos"].strip(),
                    parse_recommended_units(row["recommended_units"]),
                )
            )
    unique_code = next(
        code for code, values in recommendations.items() if len(values) == 1
    )
    ambiguous_code = next(
        code for code, values in recommendations.items() if len(values) > 1
    )

    reference = load_msbos_reference()

    assert isinstance(reference.resolve(unique_code), MsbosRow)
    assert reference.resolve(ambiguous_code) == "ambiguous"
    assert reference.resolve("not-in-reference") is None


def test_candidates_for_conflicting_code_are_named_sorted_and_absent_is_empty() -> None:
    reference = load_msbos_reference()

    candidates = reference.candidates_for("0124")

    assert candidates, "a real conflicting code must expose its raw named rows"
    assert {candidate.operation for candidate in candidates} >= {
        "Craniofacial resection",
        "Craniotomy (aneurysm)",
    }, "candidate names are the note-disambiguation vocabulary"
    assert candidates == tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.operation,
                candidate.msbos,
                candidate.recommended_units,
            ),
        )
    ), "candidate order must be deterministic regardless of CSV row order"
    assert reference.candidates_for("not-in-reference") == (), (
        "an absent code has no candidate operation names"
    )


def test_groups_for_returns_sorted_unique_groups_and_absent_is_empty() -> None:
    reference = load_msbos_reference()

    groups = reference.groups_for("0124")

    assert groups == tuple(sorted(set(groups)))
    assert groups == ("Head-Neck", "Oto", "Rhino", "ศัลยกรรมระบบประสาท")
    assert reference.groups_for("not-in-reference") == ()


def test_groups_for_optional_column_absence_is_empty() -> None:
    reference = _reference_from_rows(
        [
            {
                "icd9_code_nodot": "1234",
                "msbos": "G/M",
                "recommended_units": "2",
                "operation": "Operation A",
            }
        ],
        content_hash="d" * 64,
    )

    assert reference.groups_for("1234") == (), (
        "procedure_group remains optional so RBC-only test seams stay unchanged"
    )


def test_multiple_operation_names_with_one_recommendation_still_resolve() -> None:
    reference = load_msbos_reference()

    candidates = reference.candidates_for("194")

    assert {candidate.operation for candidate in candidates} == {
        "EAC surgery",
        "Myringotomy + PE tube",
    }, "the regression code must genuinely carry multiple raw operation names"
    assert reference.resolve("194") == MsbosRow(msbos="none", recommended_units=0), (
        "MsbosRow identity must remain recommendation-only, never operation-name based"
    )


def test_candidate_index_does_not_change_reference_identity_or_hash() -> None:
    shared = {
        "icd9_code_nodot": "1234",
        "msbos": "G/M",
        "recommended_units": "2",
    }
    first = _reference_from_rows(
        [{**shared, "operation": "Operation A"}], content_hash="c" * 64
    )
    second = _reference_from_rows(
        [{**shared, "operation": "Operation B"}], content_hash="c" * 64
    )

    assert first == second, (
        "the auxiliary candidate-name index must not alter reference equality"
    )
    assert first.content_hash == second.content_hash == "c" * 64, (
        "the supplied raw-byte hash remains the reference content identity"
    )
    assert "Operation A" not in repr(first), (
        "the auxiliary candidate-name index must not alter reference repr"
    )


def test_group_index_does_not_change_rbc_reference_identity_repr_or_resolution() -> (
    None
):
    shared = {
        "icd9_code_nodot": "1234",
        "msbos": "G/M",
        "recommended_units": "2",
        "operation": "Operation A",
    }
    first = _reference_from_rows(
        [{**shared, "procedure_group": "Head-Neck"}], content_hash="e" * 64
    )
    second = _reference_from_rows(
        [{**shared, "procedure_group": "C Spine"}], content_hash="e" * 64
    )

    assert first == second
    assert repr(first) == repr(second)
    assert first.content_hash == second.content_hash == "e" * 64
    assert (
        first.resolve("1234")
        == second.resolve("1234")
        == MsbosRow(msbos="G/M", recommended_units=2)
    )
