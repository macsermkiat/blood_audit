"""Tests for the BDVSTDT.USETYPE declared-use vocabulary."""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from bba.declared_use import (
    DECLARED_SURGICAL_LABELS,
    USETYPE_LABELS,
    DeclaredUse,
    collapse_usetype,
    label_for,
)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("1", "ward"),
        ("2", "surgery"),
        ("3", "type_screen"),
        ("4", "day_care"),
        ("5", "unknown"),
        ("junk", "unknown"),
        (" 2 ", "surgery"),
        ("", "unknown"),
    ],
)
def test_label_for_maps_known_codes_and_defaults_unknown(
    code: str, expected: str
) -> None:
    # Act
    result = label_for(code)

    # Assert
    assert result == expected


def test_usetype_labels_is_read_only() -> None:
    # Act and assert
    with pytest.raises(TypeError):
        USETYPE_LABELS["5"] = "unknown"  # type: ignore[index]


def test_declared_surgical_labels_is_the_locked_frozenset() -> None:
    # Assert
    assert isinstance(DECLARED_SURGICAL_LABELS, frozenset)
    assert DECLARED_SURGICAL_LABELS == frozenset({"surgery", "type_screen"})


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["2"], "2"),
        (["2", "2"], "2"),
        (["", " ", "2"], "2"),
        ([], None),
        (["", ""], None),
    ],
)
def test_collapse_usetype_returns_one_distinct_non_blank_code(
    values: list[str], expected: str | None
) -> None:
    # Act
    result = collapse_usetype(values)

    # Assert
    assert result == expected


def test_collapse_usetype_mixed_codes_warns_and_returns_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Act
    with caplog.at_level(logging.WARNING, logger="bba.declared_use"):
        result = collapse_usetype(["1", "2"])

    # Assert
    assert result is None
    assert caplog.records
    assert "1" in caplog.records[0].message
    assert "2" in caplog.records[0].message


def test_declared_use_is_frozen() -> None:
    # Arrange
    declared_use = DeclaredUse(code="2", label="surgery")

    # Act and assert
    with pytest.raises(ValidationError):
        declared_use.code = "1"


@pytest.mark.parametrize(
    ("code", "expected_label"),
    [("2", "surgery"), ("5", "unknown")],
)
def test_declared_use_from_code_binds_the_matching_label(
    code: str, expected_label: str
) -> None:
    # Act
    declared_use = DeclaredUse.from_code(code)

    # Assert
    assert declared_use.code == code
    assert declared_use.label == expected_label


def test_declared_use_rejects_code_label_mismatch() -> None:
    # A surgical code must never be representable as a non-surgical label:
    # #149 routes on .label for the pre-op deferral, so an inconsistent pair
    # like code="2" (surgery) with label="ward" would silently mis-route a
    # surgical order. The model binds label to code and fails loud instead.
    with pytest.raises(ValidationError):
        DeclaredUse(code="2", label="ward")
