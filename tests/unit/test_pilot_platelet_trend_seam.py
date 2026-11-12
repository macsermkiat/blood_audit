"""Pilot seam for the platelet count-trend floor (``BBA_PILOT_PLATELET_TREND``).

Issue #237 ships behind a default-off library flag until the sandbox result is
read. The seam changes both the prompt evidence (projection line) and the
verdicts (floor), so it must carry its own code identity: the audit store is
idempotent on (run_id, audit_id, code_version) and would otherwise keep the
pre-feature rows.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from bba import feature_flags

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"


def _load_llm_leg(module_name: str) -> ModuleType:
    if str(PILOT_DIR) not in sys.path:
        sys.path.insert(0, str(PILOT_DIR))
    spec = importlib.util.spec_from_file_location(
        module_name, PILOT_DIR / "run_llm_leg.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_unset_env_keeps_the_seam_off_and_the_code_identity_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BBA_PILOT_PLATELET_TREND", raising=False)

    module = _load_llm_leg("pilot_run_llm_leg_plt_trend_unset")

    assert module.PLATELET_TREND_PILOT_ENABLED is False
    assert "+plttrend" not in module.CODE_VERSION


def test_env_one_enables_and_gets_its_own_code_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BBA_PILOT_PLATELET_TREND", "1")

    module = _load_llm_leg("pilot_run_llm_leg_plt_trend_on")

    assert module.PLATELET_TREND_PILOT_ENABLED is True
    assert module.CODE_VERSION.endswith("+plttrend")
    # Importing the pilot leg must not flip the live-pipeline flag.
    assert feature_flags.PLATELET_TREND_GUARDRAIL_ENABLED is False


def test_env_zero_forces_off_after_a_library_go_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BBA_PILOT_PLATELET_TREND", "0")
    monkeypatch.setattr(feature_flags, "PLATELET_TREND_GUARDRAIL_ENABLED", True)

    module = _load_llm_leg("pilot_run_llm_leg_plt_trend_forced_off")

    assert module.PLATELET_TREND_PILOT_ENABLED is False
