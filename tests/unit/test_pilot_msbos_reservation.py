"""Focused pilot seam and byte-parity tests for ticket #162."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

from bba import feature_flags
from bba.attribution.outputs import RANKING_CSV_COLUMNS
from bba.cohort_detector import OperativeEvent


PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"


def _load_pilot_module(filename: str, module_name: str) -> ModuleType:
    if str(PILOT_DIR) not in sys.path:
        sys.path.insert(0, str(PILOT_DIR))
    spec = importlib.util.spec_from_file_location(module_name, PILOT_DIR / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_run_llm_leg(module_name: str) -> ModuleType:
    return _load_pilot_module("run_llm_leg.py", module_name)


def _load_run_pipeline(module_name: str) -> ModuleType:
    return _load_pilot_module("run_pipeline.py", module_name)


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("1", True), ("0", False), ("anything-else", False)],
)
def test_run_llm_leg_msbos_env_override(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", env_value)

    module = _load_run_llm_leg(f"pilot_run_llm_leg_msbos_{env_value}")

    assert module.MSBOS_RESERVATION_PILOT_ENABLED is expected
    assert ("+msbos" in module.CODE_VERSION) is expected


def test_run_llm_leg_msbos_unset_uses_library_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BBA_PILOT_MSBOS_RESERVATION", raising=False)

    module = _load_run_llm_leg("pilot_run_llm_leg_msbos_default")

    assert module.MSBOS_RESERVATION_PILOT_ENABLED is False
    assert "+msbos" not in module.CODE_VERSION
    assert (
        module.MSBOS_RESERVATION_PILOT_ENABLED
        is feature_flags.MSBOS_RESERVATION_ENABLED
    )


@pytest.mark.parametrize("env_value", [None, "0"])
def test_msbos_flag_off_serialized_schemas_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
) -> None:
    if env_value is None:
        monkeypatch.delenv("BBA_PILOT_MSBOS_RESERVATION", raising=False)
    else:
        monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", env_value)
    llm_module = _load_run_llm_leg(
        f"pilot_run_llm_leg_msbos_schema_{env_value or 'default'}"
    )
    pipeline_module = _load_run_pipeline(
        f"pilot_run_pipeline_msbos_schema_{env_value or 'default'}"
    )

    serialized_columns = (
        tuple(pipeline_module.REPORT_FIELDNAMES)
        + tuple(pipeline_module.RETURNS_LEDGER_FIELDNAMES)
        + tuple(pipeline_module.DECLARED_USETYPE_FIELDNAMES)
        + RANKING_CSV_COLUMNS
    )
    assert llm_module.MSBOS_RESERVATION_PILOT_ENABLED is False
    assert "+msbos" not in llm_module.CODE_VERSION
    assert not any("over_reservation" in name for name in serialized_columns)


def test_msbos_deterministic_final_vocabulary_is_inert_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "0")
    module = _load_run_llm_leg("pilot_run_llm_leg_msbos_vocab")

    assert "PREOP_OVER_RESERVATION" in module.DETERMINISTIC_FINAL


@pytest.mark.parametrize("env_value", [None, "0"])
def test_msbos_flag_off_never_loads_reference_or_reservations(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
) -> None:
    if env_value is None:
        monkeypatch.delenv("BBA_PILOT_MSBOS_RESERVATION", raising=False)
    else:
        monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", env_value)
    module = _load_run_llm_leg(
        f"pilot_run_llm_leg_msbos_no_load_{env_value or 'default'}"
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("flag-off MSBOS loader/accessor was invoked")

    monkeypatch.setattr(module, "load_msbos_reference", forbidden)
    monkeypatch.setattr(module, "reserved_units_by_component", forbidden)
    monkeypatch.setattr(module, "_read_csv", lambda _filename: [])
    monkeypatch.setattr(module, "_read_optional_csv", lambda _filename: [])
    monkeypatch.setattr(
        module, "_read_preferred_optional_csv", lambda _preferred, _fallback: []
    )
    monkeypatch.setattr(module, "load_bdvsttrans_rows", lambda _bundle: [])

    built = module._build_inputs()

    assert built[-2] is None
    assert built[-1] == {}


def test_planned_op_icd9_uses_nearest_upcoming_and_detects_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "0")
    module = _load_run_llm_leg("pilot_run_llm_leg_msbos_planned_op")
    order_datetime = datetime(2026, 7, 16, 4, 0, tzinfo=UTC)

    def event(code: str, offset_hours: int) -> OperativeEvent:
        return OperativeEvent(
            icd9=code,
            or_flag=True,
            operative_datetime=order_datetime + timedelta(hours=offset_hours),
        )

    assert (
        module._planned_op_icd9(
            [event("past", -1), event("later", 12), event("nearest", 4)],
            order_datetime,
        )
        == "nearest"
    )
    assert (
        module._planned_op_icd9(
            [event("1000", 4), event("2000", 4), event("later", 12)],
            order_datetime,
        )
        == "\x00AMBIG"
    )
    assert module._planned_op_icd9([event("past", -1)], order_datetime) == ""
