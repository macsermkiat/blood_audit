"""Contract tests for :mod:`bba.platelet_guardrail` (docs plan §5.3 stage 5).

The guardrail is what makes the "ADD platelet hard signals" ruling safe. The
clinical stakes pinned here:

* An LLM clear of a sub-ceiling count with NO grounded indication must be
  floored (the TTP/HIT/dengue exclusion trap) — :class:`TestFloorsUngroundedClear`.
* A clear WITH a grounded positive indication is allowed through
  — :class:`TestAllowsGroundedClear`.
* A bare low count is never itself an exemption (§8/CR-C2) — the signals object,
  not the count, gates the exemption.
"""

from __future__ import annotations

import pytest

from bba.platelet_guardrail import (
    PLATELET_OVERCLEAR_REVIEW_REASON,
    PlateletHardSignals,
    platelet_overclear_suspect,
)

_NO_SIGNALS = PlateletHardSignals()


class TestFloorsUngroundedClear:
    """An LLM APPROPRIATE over a withheld verdict with no signal is suspect."""

    @pytest.mark.parametrize("rule", ["NEEDS_REVIEW", "INSUFFICIENT_EVIDENCE"])
    def test_ungrounded_clear_is_suspect(self, rule: str) -> None:
        assert platelet_overclear_suspect("APPROPRIATE", rule, _NO_SIGNALS) is True

    def test_bare_low_count_is_not_an_exemption(self) -> None:
        # CR-C2: there is no count input here — a low count alone can never
        # exempt. Only a grounded PlateletHardSignals field does.
        assert platelet_overclear_suspect(
            "APPROPRIATE", "NEEDS_REVIEW", PlateletHardSignals()
        )


class TestAllowsGroundedClear:
    """A grounded positive indication exempts the clear (not suspect)."""

    @pytest.mark.parametrize(
        "signals",
        [
            PlateletHardSignals(active_bleeding=True),
            PlateletHardSignals(procedure_indication=True),
            PlateletHardSignals(prophylactic_marrow_failure=True),
            PlateletHardSignals(active_bleeding=True, procedure_indication=True),
        ],
    )
    def test_grounded_clear_is_allowed(self, signals: PlateletHardSignals) -> None:
        assert (
            platelet_overclear_suspect("APPROPRIATE", "NEEDS_REVIEW", signals) is False
        )


class TestGuardrailScope:
    """The guardrail only fires on an LLM APPROPRIATE over a withheld verdict."""

    @pytest.mark.parametrize(
        "final", ["NEEDS_REVIEW", "POTENTIALLY_INAPPROPRIATE", "INSUFFICIENT_EVIDENCE"]
    )
    def test_non_appropriate_final_never_suspect(self, final: str) -> None:
        assert platelet_overclear_suspect(final, "NEEDS_REVIEW", _NO_SIGNALS) is False

    @pytest.mark.parametrize("rule", ["APPROPRIATE", "POTENTIALLY_INAPPROPRIATE"])
    def test_non_withholding_rule_never_suspect(self, rule: str) -> None:
        # If the deterministic leg did not withhold (it already cleared, or is
        # a high-count POTENTIALLY_INAPPROPRIATE handled elsewhere), this
        # gray-zone guardrail is out of scope.
        assert platelet_overclear_suspect("APPROPRIATE", rule, _NO_SIGNALS) is False


class TestHardSignalsModel:
    def test_defaults_are_all_false(self) -> None:
        s = PlateletHardSignals()
        assert not s.any_signal()

    def test_any_signal_true_when_one_set(self) -> None:
        assert PlateletHardSignals(active_bleeding=True).any_signal()

    def test_frozen(self) -> None:
        from pydantic import ValidationError

        s = PlateletHardSignals()
        with pytest.raises(ValidationError):
            s.active_bleeding = True  # type: ignore[misc]


def test_review_reason_slug() -> None:
    assert PLATELET_OVERCLEAR_REVIEW_REASON == "platelet_llm_overclear_suspect"
