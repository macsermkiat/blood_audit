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
