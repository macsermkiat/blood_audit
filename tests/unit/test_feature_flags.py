"""The returns-ledger flag contract.

Spec #119 shipped the whole returns-ledger behavior behind a flag so a run could
reproduce prior results exactly while the feature was validated. The NARROW
go-live decision (validated by the #125 pre-flight + deterministic-leg smoke)
enables it by default; the pre-feature behavior is now the monkeypatched
exception. The other feature flags remain default-off.
"""

from __future__ import annotations

from bba import feature_flags


def test_returns_ledger_flag_defaults_on() -> None:
    # Go-live (spec #119 NARROW): the returns-ledger disposition router is the
    # default. A run reproduces the pre-feature behavior only by monkeypatching
    # this flag back to False.
    assert feature_flags.RETURNS_LEDGER_ENABLED is True


def test_returns_ledger_flag_is_exported() -> None:
    assert "RETURNS_LEDGER_ENABLED" in feature_flags.__all__


def test_existing_flags_still_default_off() -> None:
    # The other flags remain default-off; guard against an accidental flip.
    assert feature_flags.PLATELET_LLM_ENABLED is False
    assert feature_flags.RESERVE_AHEAD_ROUTER_ENABLED is False
    assert feature_flags.DECLARED_USETYPE_ENABLED is False


def test_declared_usetype_flag_is_exported() -> None:
    assert "DECLARED_USETYPE_ENABLED" in feature_flags.__all__
