"""Pilot seam for the platelet LLM leg (``BBA_PILOT_PLATELET_LLM``).

The library flag ``feature_flags.PLATELET_LLM_ENABLED`` stays default-OFF: the
platelet LLM leg has no clinician sign-off for the live pipeline. The pilot
needs an operator switch so a sandbox run can submit platelet cases WITHOUT
editing library source, and an unset env var must keep today's behaviour
(no platelet submissions, so no unplanned API spend).
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


def test_unset_env_follows_live_library_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The library flag is read at call time, so a go-live flip (or a test
    # patching it) takes effect without a pilot env var.
    monkeypatch.delenv("BBA_PILOT_PLATELET_LLM", raising=False)
    module = _load_llm_leg("pilot_run_llm_leg_plt_llm_live")

    monkeypatch.setattr(feature_flags, "PLATELET_LLM_ENABLED", True)

    assert module._platelet_llm_enabled() is True


def test_env_zero_forces_off_even_when_library_flag_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Operator escape hatch: "0" must block platelet spend after a go-live.
    monkeypatch.setenv("BBA_PILOT_PLATELET_LLM", "0")
    monkeypatch.setattr(feature_flags, "PLATELET_LLM_ENABLED", True)
    module = _load_llm_leg("pilot_run_llm_leg_plt_llm_forced_off")

    assert module._platelet_llm_enabled() is False


def test_unset_env_follows_library_default_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Unset must mean "no platelet LLM submissions": the run is user-gated spend.
    monkeypatch.delenv("BBA_PILOT_PLATELET_LLM", raising=False)

    module = _load_llm_leg("pilot_run_llm_leg_plt_llm_unset")

    assert feature_flags.PLATELET_LLM_ENABLED is False
    assert module._platelet_llm_enabled() is False


def test_env_one_enables_without_touching_library_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BBA_PILOT_PLATELET_LLM", "1")

    module = _load_llm_leg("pilot_run_llm_leg_plt_llm_on")

    assert module._platelet_llm_enabled() is True
    # Importing the pilot leg must not flip the live-pipeline flag.
    assert feature_flags.PLATELET_LLM_ENABLED is False


@pytest.mark.parametrize("value", ["0", "", "true", "yes"])
def test_any_other_value_forces_off(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    # Same contract as the other pilot seams: only the literal "1" enables.
    monkeypatch.setenv("BBA_PILOT_PLATELET_LLM", value)

    module = _load_llm_leg(f"pilot_run_llm_leg_plt_llm_off_{value or 'empty'}")

    assert module._platelet_llm_enabled() is False
