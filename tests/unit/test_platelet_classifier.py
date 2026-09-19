"""Contract tests for :mod:`bba.platelet_classifier` (docs plan §5.1).

The clinical contract, not the implementation, is pinned here:

* §8/CR-C1 — v1 auto-clears NOTHING. The gate must NEVER emit ``APPROPRIATE``
  (that would false-clear the exclusion populations: dengue-no-bleed, TTP,
  HIT, ITP, aplastic). :class:`TestNeverAutoClears` +
  :class:`TestClassifierInvariantsProperty`.
* §8/CR-M2 — no present-count verdict is ever deterministic-final; every count
  routes onward to the LLM/review. :class:`TestNeverDeterministicFinal`.
* §5.1 gate boundaries at the review ceiling and the missing-count contract.
* Cohort-gated prophylaxis auto-clear (flag ``PLATELET_PROPHYLAXIS_AUTOCLEAR_ENABLED``,
  2026-09-19): the ONLY ``APPROPRIATE`` path, and only when every structured
  condition holds. :class:`TestProphylaxisAutoclear`. Every pre-existing test
  runs with the flag off and still pins v1 behaviour.
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


class TestProphylaxisAutoclear:
    """The medicine policy's "chemo / HSCT, no bleeding: 10,000/µL" row on
    structured codes only. WHY: hematology orders sit almost entirely below
    100k, so without this rule every one of them costs an LLM call; but the
    CR-C1 populations must still never be cleared, so each exclusion code is
    pinned to block the clear."""

    @staticmethod
    def _in(
        count: float | None = 5.0,
        codes: tuple[str, ...] = ("C920", "Z511"),
        freshness: str | None = "fresh",
        *,
        flag: bool = True,
    ) -> PlateletClassifierInputs:
        return PlateletClassifierInputs(
            audit_id="a1",
            platelet_count=count,
            diagnosis_codes=codes,
            platelet_freshness=freshness,
            enable_prophylaxis_autoclear=flag,
        )

    def test_clears_fresh_low_count_on_chemo_admission(self) -> None:
        result = classify_platelet(self._in())
        assert result.classification == "APPROPRIATE"
        assert result.rationale == "plt_lt_10_heme_prophylaxis"
        assert result.classification in _DETERMINISTIC_FINAL_CLASSIFICATIONS

    def test_flag_off_defers_even_when_every_condition_holds(self) -> None:
        result = classify_platelet(self._in(flag=False))
        assert result.classification == "NEEDS_REVIEW"
        assert result.rationale == "plt_defer_llm"

    @pytest.mark.parametrize("count", [10.0, 10.1, 50.0, 99.9])
    def test_count_at_or_above_threshold_defers(self, count: float) -> None:
        assert classify_platelet(self._in(count=count)).classification == "NEEDS_REVIEW"

    @pytest.mark.parametrize(
        "freshness", ["stale_24_72h", "stale_3_7d", "missing", None]
    )
    def test_stale_count_is_never_cleared(self, freshness: str | None) -> None:
        # A 3-day-old count of 5 says nothing about today's count.
        assert (
            classify_platelet(self._in(freshness=freshness)).classification
            == "NEEDS_REVIEW"
        )

    def test_no_indication_code_defers(self) -> None:
        # Low count on a non-heme admission (e.g. pneumonia): indication is
        # unknown, the LLM must read the notes.
        assert (
            classify_platelet(self._in(codes=("J150", "N179"))).classification
            == "NEEDS_REVIEW"
        )
        assert classify_platelet(self._in(codes=())).classification == "NEEDS_REVIEW"

    @pytest.mark.parametrize(
        "code",
        [
            "D693",
            "M311",
            "D758",
            "D610",
            "D612",
            "D613",
            "D618",
            "D619",
            "A90",
            "A91",
            "A97",
            "T630",
        ],
    )
    def test_each_withhold_population_blocks_the_clear(self, code: str) -> None:
        # The CR-C1 failure mode: a heme patient who ALSO carries ITP / TTP /
        # HIT / aplastic / dengue / snakebite must not be auto-cleared.
        result = classify_platelet(self._in(codes=("C920", code)))
        assert result.classification == "NEEDS_REVIEW"
        assert result.rationale == "plt_defer_llm"

    def test_drug_induced_aplastic_anaemia_is_an_indication_not_an_exclusion(
        self,
    ) -> None:
        # D61.1 codes chemotherapy pancytopenia in this dataset (65 of 71
        # D61.x admissions in the hematology sandbox). Treating it as
        # "aplastic anaemia" would withhold exactly where the policy transfuses.
        assert (
            classify_platelet(self._in(codes=("D611",))).classification == "APPROPRIATE"
        )
        assert (
            classify_platelet(self._in(codes=("D613",))).classification
            == "NEEDS_REVIEW"
        )

    @pytest.mark.parametrize("codes", [("c92.0",), ("C92.0", "z51.1"), (" C920 ",)])
    def test_code_matching_is_case_and_dot_insensitive(
        self, codes: tuple[str, ...]
    ) -> None:
        assert classify_platelet(self._in(codes=codes)).classification == "APPROPRIATE"

    def test_never_inappropriate_and_appropriate_only_when_qualifying(self) -> None:
        from bba.platelet_classifier import qualifies_for_prophylaxis_autoclear

        for inputs in (
            self._in(),
            self._in(count=12.0),
            self._in(codes=("C920", "A91")),
            self._in(freshness="stale_24_72h"),
            self._in(flag=False),
            self._in(count=None),
            self._in(count=150.0),
        ):
            result = classify_platelet(inputs)
            assert result.classification != "INAPPROPRIATE"
            assert (
                result.classification == "APPROPRIATE"
            ) == qualifies_for_prophylaxis_autoclear(inputs)

    @given(
        count=st.one_of(st.none(), st.floats(min_value=0.0, max_value=5000.0)),
        fresh=st.sampled_from(["fresh", "stale_24_72h", "stale_3_7d", "missing"]),
        codes=st.lists(
            st.sampled_from(
                ["C920", "Z511", "D611", "J150", "D693", "M311", "A91", "D613", "T630"]
            ),
            max_size=4,
        ),
    )
    def test_property_appropriate_iff_all_structured_conditions(
        self, count: float | None, fresh: str, codes: list[str]
    ) -> None:
        from bba.platelet_classifier import (
            PLATELET_PROPHYLAXIS_EXCLUSION_PREFIXES,
            PLATELET_PROPHYLAXIS_INDICATION_PREFIXES,
        )

        result = classify_platelet(
            self._in(count=count, codes=tuple(codes), freshness=fresh)
        )
        expected = (
            count is not None
            and count < 10.0
            and fresh == "fresh"
            and any(
                c.startswith(PLATELET_PROPHYLAXIS_INDICATION_PREFIXES) for c in codes
            )
            and not any(
                c.startswith(PLATELET_PROPHYLAXIS_EXCLUSION_PREFIXES) for c in codes
            )
        )
        assert (result.classification == "APPROPRIATE") == expected
        assert result.classification != "INAPPROPRIATE"
