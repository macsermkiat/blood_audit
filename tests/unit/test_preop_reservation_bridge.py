"""OPRTACT->ICD9 bridge reference tests for ticket #197.

Synthetic fixtures drive the build script's pure parse/collapse helpers and
the loader's ``_bridge_from_rows`` construction seam (mirroring the MSBOS
reference loader's seam); one smoke test loads the real packaged CSV.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

import pytest

from bba.preop_reservation.bridge import (
    BridgeReferenceError,
    OprtactBridge,
    _bridge_from_rows,
    load_oprtact_bridge,
)
from bba.preop_reservation.reference import load_msbos_reference


def _load_build_script() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "build_oprtact_icd9_bridge.py"
    )
    spec = importlib.util.spec_from_file_location("build_oprtact_bridge_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the dataclass decorator can resolve the script's
    # postponed annotations via sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BUILD = _load_build_script()


def _raw_row(
    oprtact: str = "P0001",
    first_choice: str = "3725::Biopsy of heart (score=1.000)",
    human: str = "0",
    second: str = "",
    third: str = "",
) -> dict[str, str]:
    return {
        "OPRTACT": oprtact,
        "First Choice": first_choice,
        "Second choice": second,
        "Third choice": third,
        "Human suggestion": human,
    }


def _row(
    oprtact: str = "P0001",
    *,
    icd9: str = "3725",
    icd9_nodot: str | None = None,
    score: str = "1.000",
    human_index: str = "0",
    human_agreed: str = "true",
    human_icd9: str = "",
    name: str = "Biopsy of heart",
) -> dict[str, str]:
    return {
        "oprtact": oprtact,
        "icd9": icd9,
        "icd9_nodot": icd9_nodot if icd9_nodot is not None else icd9.replace(".", ""),
        "score": score,
        "human_index": human_index,
        "human_agreed": human_agreed,
        "human_icd9": human_icd9,
        "name": name,
    }


def _bridge(rows: Sequence[Mapping[str, str]]) -> OprtactBridge:
    return _bridge_from_rows(rows, content_hash="a" * 64)


# --- build script: First Choice cell grammar --------------------------------


def test_parse_first_choice_canonical_cell() -> None:
    icd9, name, score = _BUILD.parse_first_choice("3725::Biopsy of heart (score=1.000)")

    assert icd9 == "3725"
    assert name == "Biopsy of heart"
    assert score == "1.000"


def test_parse_first_choice_name_containing_parens() -> None:
    icd9, name, score = _BUILD.parse_first_choice(
        "343::Excision (partial) of mediastinum (score=0.98)"
    )

    assert icd9 == "343"
    assert name == "Excision (partial) of mediastinum"
    assert score == "0.98"


@pytest.mark.parametrize(
    "cell",
    [
        "3725 Biopsy of heart (score=1.000)",  # missing ::
        "3725::Biopsy of heart",  # missing score suffix
        "::Biopsy of heart (score=1.000)",  # empty icd9
        "3725::Biopsy of heart (score=high)",  # non-float score
    ],
)
def test_parse_first_choice_malformed_cell_fails_loud(cell: str) -> None:
    with pytest.raises(_BUILD.BridgeBuildError):
        _BUILD.parse_first_choice(cell)


# --- build script: row rules -------------------------------------------------


def test_build_rows_excludes_blank_oprtact_without_aborting() -> None:
    rows, counts = _BUILD.build_rows([_raw_row(oprtact=""), _raw_row(oprtact="P0002")])

    assert [r["oprtact"] for r in rows] == ["P0002"]
    assert counts.blank_key_skipped == 1


def test_build_rows_skips_blank_first_choice() -> None:
    rows, counts = _BUILD.build_rows(
        [_raw_row(oprtact="P0001", first_choice=""), _raw_row(oprtact="P0002")]
    )

    assert [r["oprtact"] for r in rows] == ["P0002"]
    assert counts.eligible == 1


def test_build_rows_identical_duplicate_collapses_to_first_occurrence() -> None:
    first = _raw_row(
        oprtact="PA101",
        first_choice="411::Coronary arteriography (score=0.879)",
        human="1",
    )
    second = _raw_row(
        oprtact="PA101",
        first_choice="411::Different label, same choice (score=0.879)",
        human="0",
    )

    rows, counts = _BUILD.build_rows([first, second])

    assert len(rows) == 1
    assert rows[0]["name"] == "Coronary arteriography"
    assert rows[0]["human_index"] == "1"
    assert rows[0]["human_agreed"] == "false"
    assert counts.collapsed_duplicates == 1


@pytest.mark.parametrize(
    "second_choice_cell",
    [
        "412::Coronary arteriography (score=0.879)",  # icd9 conflict
        "411::Coronary arteriography (score=0.880)",  # raw score conflict
    ],
)
def test_build_rows_genuine_duplicate_conflict_fails_loud(
    second_choice_cell: str,
) -> None:
    first = _raw_row(
        oprtact="PA101", first_choice="411::Coronary arteriography (score=0.879)"
    )
    second = _raw_row(oprtact="PA101", first_choice=second_choice_cell)

    with pytest.raises(_BUILD.BridgeBuildError, match="PA101"):
        _BUILD.build_rows([first, second])


def test_build_rows_dotted_icd9_normalizes_nodot() -> None:
    rows, _ = _BUILD.build_rows(
        [_raw_row(first_choice="06.02::Synthetic dotted op (score=0.5)")]
    )

    assert rows[0]["icd9"] == "06.02"
    assert rows[0]["icd9_nodot"] == "0602"


def test_canonical_nodot_restores_category_leading_zero() -> None:
    # The raw export stripped the 0X category zero: '309' must become '0309'
    # because the ICD-9-CM name matches 03.09, not 30.9.
    names = {
        "0309": "Other exploration and decompression of spinal canal",
        "309": "Other operations on larynx",
    }

    assert (
        _BUILD.canonical_nodot(
            "309", "Other exploration and decompression of spinal canal", names
        )
        == "0309"
    )


def test_canonical_nodot_restores_category_00_double_zero() -> None:
    # Both category zeros stripped: '11' -> '0011' (00.11), not '011' (01.1).
    # The single-pad candidate names a different entry, so only the 4-char form
    # matches the operation name.
    names = {
        "011": "Diagnostic procedures on skull, brain, and cerebral meninges",
        "0011": "Infusion of drotrecogin alfa (activated)",
    }

    assert (
        _BUILD.canonical_nodot("11", "Infusion of drotrecogin alfa (activated)", names)
        == "0011"
    )


def test_canonical_nodot_leaves_legitimate_three_digit_code() -> None:
    # 55.4 -> '554' is a real 3-digit code; the padded '0554' does not match the
    # name, so it must be left untouched.
    names = {"554": "Partial nephrectomy"}

    assert _BUILD.canonical_nodot("554", "Partial nephrectomy", names) == "554"


def test_canonical_nodot_no_map_is_identity() -> None:
    assert _BUILD.canonical_nodot("309", "anything", {}) == "309"


def test_build_rows_restores_leading_zero_with_icd9cm_map() -> None:
    names = {"0309": "Other exploration and decompression of spinal canal"}
    rows, counts = _BUILD.build_rows(
        [
            _raw_row(
                oprtact="P0243",
                first_choice=(
                    "309::Other exploration and decompression of "
                    "spinal canal (score=1.000)"
                ),
            )
        ],
        names,
    )

    assert rows[0]["icd9"] == "0309"
    assert rows[0]["icd9_nodot"] == "0309"
    assert counts.leading_zero_restored == 1


@pytest.mark.parametrize(
    ("human", "agreed"),
    [("0", "true"), ("1", "false"), ("", "false"), ("4", "false")],
)
def test_build_rows_human_agreed_derivation(human: str, agreed: str) -> None:
    rows, _ = _BUILD.build_rows([_raw_row(human=human)])

    assert rows[0]["human_index"] == human
    assert rows[0]["human_agreed"] == agreed


@pytest.mark.parametrize(
    ("human", "expected_human_icd9"),
    [
        ("0", ""),  # agreement: no separate human code
        ("1", "4573"),  # selects the Second choice
        ("2", "7935"),  # selects the Third choice
        ("3", ""),  # out-of-range index points at nothing
        ("", ""),  # no selection
    ],
)
def test_build_rows_human_icd9_selection(human: str, expected_human_icd9: str) -> None:
    rows, _ = _BUILD.build_rows(
        [
            _raw_row(
                human=human,
                second="4573::Open synthetic repair (score=0.80)",
                third="7935::Other synthetic repair (score=0.60)",
            )
        ]
    )

    assert rows[0]["human_icd9"] == expected_human_icd9


def test_build_rows_blank_selected_cell_is_no_selection() -> None:
    rows, _ = _BUILD.build_rows([_raw_row(human="1", second="")])

    assert rows[0]["human_icd9"] == ""


def test_build_rows_malformed_selected_cell_fails_loud() -> None:
    with pytest.raises(_BUILD.BridgeBuildError):
        _BUILD.build_rows([_raw_row(human="1", second="4573 no separator")])


def test_build_rows_output_sorted_by_oprtact() -> None:
    rows, _ = _BUILD.build_rows(
        [_raw_row(oprtact="P0900"), _raw_row(oprtact="AS001"), _raw_row(oprtact="MD01")]
    )

    assert [r["oprtact"] for r in rows] == ["AS001", "MD01", "P0900"]


# --- loader: construction seam ----------------------------------------------


def test_bridge_from_rows_happy_path() -> None:
    bridge = _bridge([_row("P0001"), _row("P0002", icd9="0602", score="0.5")])

    entry = bridge.get("P0001")
    assert entry is not None
    assert entry.icd9 == "3725"
    assert entry.icd9_nodot == "3725"
    assert entry.score == pytest.approx(1.0)
    assert entry.human_index == "0"
    assert entry.human_agreed is True
    assert entry.name == "Biopsy of heart"
    assert bridge.get(" P0002 ") is not None
    assert bridge.get("MISSING") is None
    assert len(bridge) == 2
    assert bridge.content_hash == "a" * 64


def test_bridge_from_rows_mapping_is_immutable() -> None:
    bridge = _bridge([_row("P0001")])

    with pytest.raises(TypeError):
        bridge._entries["P0002"] = bridge._entries["P0001"]  # type: ignore[index]


def test_bridge_from_rows_missing_column_rejected() -> None:
    row = _row("P0001")
    del row["score"]

    with pytest.raises(BridgeReferenceError, match="missing columns"):
        _bridge([row])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("oprtact", "", "blank oprtact"),
        ("icd9", "", "blank icd9"),
        ("icd9_nodot", "9999", "icd9_nodot"),
        ("score", "abc", "score"),
        ("score", "inf", "score"),
        ("score", "nan", "score"),
        ("human_agreed", "True", "human_agreed"),
        ("human_agreed", "1", "human_agreed"),
        ("human_agreed", "", "human_agreed"),
    ],
)
def test_bridge_from_rows_malformed_field_rejected(
    field: str, value: str, message: str
) -> None:
    row = _row("P0001")
    row[field] = value

    with pytest.raises(BridgeReferenceError, match=message):
        _bridge([row])


def test_bridge_from_rows_duplicate_key_rejected() -> None:
    with pytest.raises(BridgeReferenceError, match="duplicate"):
        _bridge([_row("P0001"), _row("P0001", score="0.9")])


# --- packaged-file smoke ------------------------------------------------------

# The 53 distinct INCPT sentinel source codes observed in the frozen 2026-07
# pilot cohort (/tmp/bba_mini report, msbos_resolved_icd9 "INCPT:<code>").
_PILOT_SOURCE_CODES = (
    "AS056",
    "AS058",
    "CC011",
    "EY619",
    "L0021",
    "L0071",
    "L0083",
    "L0177",
    "MD529",
    "MD530",
    "P0001",
    "P0067",
    "P0204",
    "P0214",
    "P0234",
    "P0580",
    "P0594",
    "P0597",
    "P0614",
    "P0621",
    "P0624",
    "P0635",
    "P0636",
    "P0694",
    "P0752",
    "P0763",
    "P0839",
    "P0846",
    "P0899",
    "P0937",
    "P0939",
    "P0941",
    "P0955",
    "P0962",
    "P0978",
    "P0990",
    "P1077",
    "P1112",
    "P1173",
    "P1177",
    "P1178",
    "P1247",
    "P1266",
    "P1273",
    "P1301",
    "P1306",
    "P1332",
    "P1345",
    "P1817",
    "P1933",
    "P2008",
    "P2231",
    "SU062",
)


def test_packaged_bridge_smoke() -> None:
    bridge = load_oprtact_bridge()

    assert len(bridge) == 6433
    packaged = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "bba"
        / "preop_reservation"
        / "data"
        / "oprtact_icd9_bridge.csv"
    )
    assert bridge.content_hash == hashlib.sha256(packaged.read_bytes()).hexdigest()

    missing = [c for c in _PILOT_SOURCE_CODES if bridge.get(c) is None]
    assert missing == []

    reference = load_msbos_reference()
    msbos_hits = {
        entry.icd9_nodot
        for entry in (bridge.get(c) for c in _PILOT_SOURCE_CODES)
        if entry is not None and reference.resolve(entry.icd9_nodot) is not None
    }
    # 18, not 17: the category leading-zero fix restored 0302 (P0204,
    # "Reopening of laminectomy site"), which now resolves in the MSBOS ref.
    assert len(msbos_hits) == 18


def test_packaged_bridge_first_choice_spot_checks() -> None:
    bridge = load_oprtact_bridge()

    p0937 = bridge.get("P0937")
    assert p0937 is not None and p0937.icd9 == "554" and p0937.human_agreed is True
    assert p0937.human_icd9 == ""
    p1247 = bridge.get("P1247")
    assert p1247 is not None and p1247.icd9 == "8151" and p1247.human_agreed is True
    # First Choice, NOT the Human-selected codes (315 / 3809) — which are
    # carried separately for the disagreement guard.
    p0580 = bridge.get("P0580")
    assert p0580 is not None and p0580.icd9 == "343"
    assert p0580.human_icd9 == "315"
    p0624 = bridge.get("P0624")
    assert p0624 is not None and p0624.icd9 == "3800"
    assert p0624.human_icd9 == "3809"
    # The P0752 class (First misses MSBOS, Human hits): the guard's data.
    p0752 = bridge.get("P0752")
    assert p0752 is not None and p0752.icd9 == "1733"
    assert p0752.human_icd9 == "4573"


def test_load_oprtact_bridge_is_cached() -> None:
    assert load_oprtact_bridge() is load_oprtact_bridge()
