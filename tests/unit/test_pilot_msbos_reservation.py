"""Focused pilot seam and byte-parity tests for ticket #162."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from bba import feature_flags
from bba.attribution.outputs import RANKING_CSV_COLUMNS


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


def test_run_llm_leg_msbos_unset_uses_library_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BBA_PILOT_MSBOS_RESERVATION", raising=False)

    module = _load_run_llm_leg("pilot_run_llm_leg_msbos_default")

    assert module.MSBOS_RESERVATION_PILOT_ENABLED is False
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
    assert not any("over_reservation" in name for name in serialized_columns)
