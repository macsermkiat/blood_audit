"""Contract tests for :mod:`bba.platelet_classifier` (docs plan §5.1).

The clinical contract, not the implementation, is pinned here:

* §8/CR-C1 — v1 auto-clears NOTHING. The gate must NEVER emit ``APPROPRIATE``
  (that would false-clear the exclusion populations: dengue-no-bleed, TTP,
  HIT, ITP, aplastic). :class:`TestNeverAutoClears` +
  :class:`TestClassifierInvariantsProperty`.
* §8/CR-M2 — no present-count verdict is ever deterministic-final; every count
  routes onward to the LLM/review. :class:`TestNeverDeterministicFinal`.
* §5.1 gate boundaries at the review ceiling and the missing-count contract.
"""

from __future__ import annotations

from typing import get_args

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bba.audit_pipeline.pipeline import _DETERMINISTIC_FINAL_CLASSIFICATIONS
from bba.audit_store import Classification
from bba.platelet_classifier import (
    PLATELET_REVIEW_CEILING,
    PlateletClassifierInputs,
    PlateletClassifierResult,
    classify_platelet,
)


def _inputs(count: float | None, *, defer: bool = False) -> PlateletClassifierInputs:
    return PlateletClassifierInputs(
        audit_id="a1",
        platelet_count=count,
        enable_missing_platelet_defer=defer,
    )


class TestGateBoundaries:
    """The §5.1 count gate around the review ceiling (100 ×10³/µL)."""

    @pytest.mark.parametrize("count", [100.0, 150.0, 450.0, 999.0, 1117.0])
    def test_at_or_above_ceiling_is_potentially_inappropriate(
        self, count: float
    ) -> None:
        result = classify_platelet(_inputs(count))
        assert result.classification == "POTENTIALLY_INAPPROPRIATE"
        assert result.rationale == "plt_ge_100"
        assert result.review_ceiling == PLATELET_REVIEW_CEILING

    @pytest.mark.parametrize("count", [99.9, 50.0, 10.0, 9.0, 2.0, 1.0])
    def test_below_ceiling_defers_to_llm(self, count: float) -> None:
        result = classify_platelet(_inputs(count))
        assert result.classification == "NEEDS_REVIEW"
        assert result.rationale == "plt_defer_llm"

    def test_very_low_count_is_not_auto_cleared(self) -> None:
        # The removed plt<10→APPROPRIATE defect: a count of 5 must route to
        # review, NOT clear — a dengue-no-bleed / TTP / HIT patient at plt<10
        # is exactly the population the policy withholds platelets from.
        result = classify_platelet(_inputs(5.0))
        assert result.classification == "NEEDS_REVIEW"
        assert result.rationale != "plt_lt_10"  # no such slug exists


class TestMissingCountContract:
    """Missing count mirrors the RBC missing-Hb opt-in contract."""

    def test_flag_off_is_terminal_insufficient_evidence(self) -> None:
        result = classify_platelet(_inputs(None, defer=False))
        assert result.classification == "INSUFFICIENT_EVIDENCE"
        assert result.rationale == "plt_missing"

    def test_flag_on_defers_to_llm(self) -> None:
        result = classify_platelet(_inputs(None, defer=True))
        assert result.classification == "NEEDS_REVIEW"
        assert result.rationale == "plt_missing_defer_llm"


class TestNeverAutoClears:
    """§8/CR-C1: the deterministic platelet gate never auto-clears."""

    @pytest.mark.parametrize(
        "count", [None, 1.0, 5.0, 9.9, 50.0, 99.9, 100.0, 500.0, 3000.0]
    )
    @pytest.mark.parametrize("defer", [True, False])
    def test_never_appropriate_or_inappropriate(
        self, count: float | None, defer: bool
    ) -> None:
        result = classify_platelet(_inputs(count, defer=defer))
        assert result.classification not in {"APPROPRIATE", "INAPPROPRIATE"}


class TestNeverDeterministicFinal:
    """§8/CR-M2: a PRESENT count never yields a deterministic-final verdict."""

    @pytest.mark.parametrize(
        "count", [1.0, 5.0, 9.9, 10.0, 50.0, 99.9, 100.0, 150.0, 3000.0]
    )
    def test_present_count_routes_onward(self, count: float) -> None:
        result = classify_platelet(_inputs(count))
        assert result.classification not in _DETERMINISTIC_FINAL_CLASSIFICATIONS

    def test_only_missing_count_flag_off_is_terminal(self) -> None:
        # The single terminal path is the no-data documentation gap, and it is
        # INSUFFICIENT_EVIDENCE (not a clear) — mirrors RBC exactly.
        terminal = classify_platelet(_inputs(None, defer=False))
        assert terminal.classification in _DETERMINISTIC_FINAL_CLASSIFICATIONS
        assert terminal.classification == "INSUFFICIENT_EVIDENCE"


class TestClassifierInvariantsProperty:
    """Property tests over the full input space (hypothesis)."""

    @given(
        count=st.one_of(st.none(), st.floats(min_value=0.0, max_value=5000.0)),
        defer=st.booleans(),
    )
    def test_result_classification_is_canonical(
        self, count: float | None, defer: bool
    ) -> None:
        result = classify_platelet(_inputs(count, defer=defer))
        assert result.classification in get_args(Classification)

    @given(
        count=st.one_of(st.none(), st.floats(min_value=0.0, max_value=5000.0)),
        defer=st.booleans(),
    )
    def test_never_clears_over_input_space(
        self, count: float | None, defer: bool
    ) -> None:
        result = classify_platelet(_inputs(count, defer=defer))
        assert result.classification not in {"APPROPRIATE", "INAPPROPRIATE"}

    @given(count=st.floats(min_value=0.0001, max_value=5000.0), defer=st.booleans())
    def test_present_count_never_final_over_input_space(
        self, count: float, defer: bool
    ) -> None:
        result = classify_platelet(_inputs(count, defer=defer))
        assert result.classification not in _DETERMINISTIC_FINAL_CLASSIFICATIONS


class TestResultModel:
    def test_frozen(self) -> None:
        from pydantic import ValidationError

        result = classify_platelet(_inputs(100.0))
        with pytest.raises(ValidationError):
            result.classification = "APPROPRIATE"  # type: ignore[misc]

    def test_result_type(self) -> None:
        assert isinstance(classify_platelet(_inputs(100.0)), PlateletClassifierResult)
