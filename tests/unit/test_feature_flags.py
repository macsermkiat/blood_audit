"""The returns-ledger flag must default OFF and be part of the flag contract.

Spec #119 ships the whole returns-ledger behavior behind a default-off flag so a
run reproduces prior results exactly and enabling the feature is a deliberate act.
"""

from __future__ import annotations

from bba import feature_flags


def test_returns_ledger_flag_defaults_off() -> None:
    assert feature_flags.RETURNS_LEDGER_ENABLED is False


def test_returns_ledger_flag_is_exported() -> None:
    assert "RETURNS_LEDGER_ENABLED" in feature_flags.__all__


def test_existing_flags_still_default_off() -> None:
    # Guard against an accidental default flip when adding the new flag.
    assert feature_flags.PLATELET_LLM_ENABLED is False
    assert feature_flags.RESERVE_AHEAD_ROUTER_ENABLED is False
