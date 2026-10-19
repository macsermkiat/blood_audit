"""The MSBOS supplement adds surgeon-confirmed codes to existing operations only.

Why this matters: the hospital OPBloodLimit workbook is an immutable export. When
a surgeon maps an uncovered ICD-9 code onto one of its operations, that mapping
must extend the vendored reference without ever changing a tariff -- a
supplement row that names a missing operation, or disagrees with the row's MSBOS
recommendation, has to abort the build rather than be applied quietly.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "build_opbloodlimit_reference.py"
DATA = REPO / "src" / "bba" / "preop_reservation" / "data"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "build_opbloodlimit_reference", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load()


def _op(
    sheet: str, group: str, operation: str, msbos: str, units: str, codes: list[str]
):
    return {
        "specialty": sheet,
        "sheet": sheet,
        "procedure_group": group,
        "operation": operation,
        "msbos": msbos,
        "msbos_meaning": "",
        "recommended_units": units,
        "codes": codes,
    }


def _row(**overrides: str) -> dict[str, str]:
    base = {
        "sheet": "Ortho",
        "procedure_group": "Amputation",
        "operation": "Amputation",
        "icd9_code": "84.17",
        "msbos": "T/S",
        "recommended_units": "",
    }
    return {**base, **overrides}


OPS = [
    _op("Ortho", "Amputation", "Amputation", "T/S", "1-2", ["84.00", "84.10"]),
    _op("Ortho", "TL spine", "3 level ขึ้นไป", "G/M", "1", ["81.04"]),
    _op("Ortho", "Metastasis spine", "3 level ขึ้นไป", "G/M", "1", ["81.04"]),
    _op("OB-Gyn", "Hysteroscopy", "Lesion of uterus", "T/S", "1", ["68.29"]),
    _op("OB-Gyn", "Myomectomy", "Lesion of uterus", "T/S", "1", ["68.29"]),
]


def test_supplement_code_lands_on_the_named_operation_without_mutating_input() -> None:
    new_ops, added = builder.apply_supplement(OPS, [_row()])

    assert added == 1
    assert new_ops[0]["codes"] == ["84.00", "84.10", "84.17"]
    assert OPS[0]["codes"] == ["84.00", "84.10"], "input ops must not be mutated"
    assert [o["codes"] for o in new_ops[1:]] == [o["codes"] for o in OPS[1:]]


def test_ts_units_mismatch_is_tolerated_because_the_loader_ignores_ts_units() -> None:
    # The surgeon wrote "-" for a Type & Screen row; the reference says "1-2".
    # The loader normalises T/S units to 0, so this is not a tariff change.
    _, added = builder.apply_supplement(OPS, [_row(recommended_units="-")])
    assert added == 1


def test_gm_units_mismatch_aborts_so_a_tariff_cannot_change_quietly() -> None:
    row = _row(
        procedure_group="TL spine",
        operation="3 level ขึ้นไป",
        icd9_code="84.51",
        msbos="G/M",
        recommended_units="2",
    )
    with pytest.raises(builder.SupplementError, match="units '2' disagree"):
        builder.apply_supplement(OPS, [row])


def test_msbos_token_mismatch_aborts() -> None:
    with pytest.raises(builder.SupplementError, match="msbos 'G/M' disagrees"):
        builder.apply_supplement(OPS, [_row(msbos="G/M")])


def test_unknown_operation_aborts_instead_of_creating_a_row() -> None:
    with pytest.raises(builder.SupplementError, match="no operation matches"):
        builder.apply_supplement(OPS, [_row(operation="Amputation (typo)")])


def test_same_operation_name_in_two_groups_requires_the_group() -> None:
    row = _row(
        procedure_group="",
        operation="3 level ขึ้นไป",
        icd9_code="84.51",
        msbos="G/M",
        recommended_units="1",
    )
    # Both groups agree on (G/M, 1): a blank group means "every group" and adds to each.
    new_ops, added = builder.apply_supplement(OPS, [row])
    assert added == 2
    assert new_ops[1]["codes"] == ["81.04", "84.51"]
    assert new_ops[2]["codes"] == ["81.04", "84.51"]


def test_blank_group_with_disagreeing_recommendations_aborts() -> None:
    ops = [*OPS, _op("Ortho", "Pediatric", "3 level ขึ้นไป", "T/S", "", [])]
    row = _row(
        procedure_group="",
        operation="3 level ขึ้นไป",
        icd9_code="84.51",
        msbos="G/M",
        recommended_units="1",
    )
    with pytest.raises(builder.SupplementError, match="matched groups disagree"):
        builder.apply_supplement(ops, [row])


def test_code_already_on_the_operation_aborts_as_a_stale_supplement() -> None:
    with pytest.raises(builder.SupplementError, match="already listed"):
        builder.apply_supplement(OPS, [_row(icd9_code="84.10")])


def test_leading_zero_is_restored_on_the_supplement_code() -> None:
    row = _row(procedure_group="Amputation", icd9_code="4.73")
    new_ops, _ = builder.apply_supplement(OPS, [row])
    assert new_ops[0]["codes"][-1] == "04.73"


def test_vendored_reference_contains_every_vendored_supplement_row() -> None:
    """The two vendored artifacts must be built from each other, not edited apart."""
    supplement = builder.load_supplement(DATA / "OPBloodLimit_supplement.csv")
    with (DATA / "OPBloodLimit_by_icd9.csv").open(
        encoding="utf-8-sig", newline=""
    ) as f:
        reference = list(csv.DictReader(f))
    assert supplement, "vendored supplement is empty"
    for row in supplement:
        code = builder.fmt_icd(row["icd9_code"]).replace(".", "")
        hits = [
            r
            for r in reference
            if r["icd9_code_nodot"] == code
            and r["sheet"] == row["sheet"]
            and r["operation"] == row["operation"]
            and (
                not row["procedure_group"]
                or r["procedure_group"] == row["procedure_group"]
            )
        ]
        assert hits, f"supplement row not in vendored reference: {row}"
        assert {h["msbos"] for h in hits} == {builder.norm_msbos(row["msbos"])}
