"""Contract tests for the peri-op-fix verification harness.

The harness must (a) collapse verdicts into the human's 3 buckets, (b) split
the confusion matrix by mechanism so a headline can't hide which leg moved a
case, and (c) surface the two honest costs — regressions and the LLM-volume
delta. Cases 47 (68062324) and 100 (68069089) — the motivating LLM
over-clears the human labelled ``ไม่สมเหตุสมผล`` (inappropriate) — anchor the
before/after scenario: before the fix the LLM cleared them (wrong); after it
they are no longer force-cleared.
"""

from __future__ import annotations

import pytest

from bba.verification import (
    CaseVerdict,
    bucket_of,
    build_matrix,
    compare_runs,
    confusion_by_mechanism,
    find_regressions,
)


class TestBucketOf:
    @pytest.mark.parametrize(
        "label,bucket",
        [
            ("APPROPRIATE", "appropriate"),
            ("INAPPROPRIATE", "inappropriate"),
            ("NEEDS_REVIEW", "unresolved"),
            ("INSUFFICIENT_EVIDENCE", "unresolved"),
            ("POTENTIALLY_INAPPROPRIATE", "unresolved"),
            ("PREOP_RESERVATION_UNCONFIRMED", "unresolved"),
            ("PREOP_OVER_RESERVATION", "inappropriate"),
            ("RETURNED_NOT_TRANSFUSED", "excluded"),
            ("PERIOP_TRANSFUSION_EXEMPT", "excluded"),
        ],
    )
    def test_maps_every_classification(self, label: str, bucket: str) -> None:
        assert bucket_of(label) == bucket

    def test_unknown_label_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="unknown classification"):
            bucket_of("DEFINITELY_NOT_A_LABEL")


class TestBuildMatrix:
    def test_full_grid_always_emitted(self) -> None:
        # 9 cells even for a single-case run — no absent-vs-zero ambiguity.
        labels = {"r1": "APPROPRIATE"}
        verdicts = {
            "r1": CaseVerdict(reqno="r1", classification="APPROPRIATE", mechanism="llm")
        }
        matrix = build_matrix(labels, verdicts)
        assert len(matrix.cells) == 9
        assert matrix.count("appropriate", "appropriate") == 1
        assert matrix.count("inappropriate", "inappropriate") == 0

    def test_only_intersection_is_scored(self) -> None:
        # A label with no verdict (run covered a subset) is not counted, so a
        # partial run never looks complete.
        labels = {"r1": "APPROPRIATE", "r2": "INAPPROPRIATE"}
        verdicts = {
            "r1": CaseVerdict(
                reqno="r1", classification="APPROPRIATE", mechanism="deterministic"
            )
        }
        matrix = build_matrix(labels, verdicts)
        assert matrix.total == 1

    def test_returned_not_transfused_is_excluded_from_matrix(self) -> None:
        labels = {"r1": "INAPPROPRIATE"}
        verdicts = {
            "r1": CaseVerdict(
                reqno="r1",
                classification="RETURNED_NOT_TRANSFUSED",
                mechanism="deterministic",
            )
        }
        assert build_matrix(labels, verdicts).total == 0

    def test_periop_transfusion_exempt_is_excluded_from_matrix(self) -> None:
        labels = {"r1": "INAPPROPRIATE"}
        verdicts = {
            "r1": CaseVerdict(
                reqno="r1",
                classification="PERIOP_TRANSFUSION_EXEMPT",
                mechanism="deterministic",
            )
        }
        assert build_matrix(labels, verdicts).total == 0

    def test_scope_splits_by_mechanism(self) -> None:
        labels = {"d": "APPROPRIATE", "l": "INAPPROPRIATE"}
        verdicts = {
            "d": CaseVerdict(
                reqno="d", classification="APPROPRIATE", mechanism="deterministic"
            ),
            "l": CaseVerdict(reqno="l", classification="NEEDS_REVIEW", mechanism="llm"),
        }
        det = build_matrix(labels, verdicts, scope="deterministic")
        llm = build_matrix(labels, verdicts, scope="llm")
        assert det.total == 1 and llm.total == 1
        assert det.count("appropriate", "appropriate") == 1
        assert llm.count("inappropriate", "unresolved") == 1

    def test_accuracy_is_diagonal_fraction(self) -> None:
        labels = {"a": "APPROPRIATE", "b": "INAPPROPRIATE", "c": "APPROPRIATE"}
        verdicts = {
            "a": CaseVerdict(reqno="a", classification="APPROPRIATE", mechanism="llm"),
            "b": CaseVerdict(
                reqno="b", classification="INAPPROPRIATE", mechanism="llm"
            ),
            "c": CaseVerdict(reqno="c", classification="NEEDS_REVIEW", mechanism="llm"),
        }
        matrix = build_matrix(labels, verdicts)
        assert matrix.correct == 2
        assert matrix.accuracy == pytest.approx(2 / 3)


class TestConfusionByMechanism:
    def test_returns_det_llm_all_in_order(self) -> None:
        labels = {"x": "APPROPRIATE"}
        verdicts = {
            "x": CaseVerdict(reqno="x", classification="APPROPRIATE", mechanism="llm")
        }
        det, llm, allm = confusion_by_mechanism(labels, verdicts)
        assert (det.scope, llm.scope, allm.scope) == ("deterministic", "llm", "all")
        assert allm.total == 1 and det.total == 0 and llm.total == 1


# Cases 47 (68062324) and 100 (68069089): human said inappropriate; the LLM
# over-cleared both. "before" = the pilot run that shipped (LLM APPROPRIATE);
# "after" = the fixed run where the guardrail / prompt no longer force-clears.
_C47 = "68062324"
_C100 = "68069089"
_HUMAN = {_C47: "INAPPROPRIATE", _C100: "INAPPROPRIATE"}
_BEFORE = {
    _C47: CaseVerdict(reqno=_C47, classification="APPROPRIATE", mechanism="llm"),
    _C100: CaseVerdict(reqno=_C100, classification="APPROPRIATE", mechanism="llm"),
}
_AFTER = {
    # Case 47: guardrail floored the LLM over-clear to review.
    _C47: CaseVerdict(reqno=_C47, classification="NEEDS_REVIEW", mechanism="llm"),
    # Case 100: prompt recalibration got the LLM to call it inappropriate.
    _C100: CaseVerdict(reqno=_C100, classification="INAPPROPRIATE", mechanism="llm"),
}


class TestMotivatingCasesFixture:
    def test_before_run_misclassifies_both_as_appropriate(self) -> None:
        matrix = build_matrix(_HUMAN, _BEFORE, scope="llm")
        # Human inappropriate, pipeline appropriate — the dangerous off-diagonal.
        assert matrix.count("inappropriate", "appropriate") == 2
        assert matrix.correct == 0

    def test_after_run_no_longer_force_clears(self) -> None:
        matrix = build_matrix(_HUMAN, _AFTER, scope="llm")
        assert matrix.count("inappropriate", "appropriate") == 0
        # Case 100 now correct (inappropriate); Case 47 moved to unresolved —
        # "no longer force-cleared", tracked separately from "correct".
        assert matrix.count("inappropriate", "inappropriate") == 1
        assert matrix.count("inappropriate", "unresolved") == 1


class TestCompareRuns:
    def test_llm_volume_delta_counts_llm_routed_cases(self) -> None:
        labels = {"a": "APPROPRIATE", "b": "INAPPROPRIATE"}
        before = {
            "a": CaseVerdict(
                reqno="a", classification="APPROPRIATE", mechanism="deterministic"
            ),
            "b": CaseVerdict(
                reqno="b", classification="APPROPRIATE", mechanism="deterministic"
            ),
        }
        after = {
            # 'a' deferred to the LLM by the pre-op fix.
            "a": CaseVerdict(reqno="a", classification="NEEDS_REVIEW", mechanism="llm"),
            "b": CaseVerdict(
                reqno="b", classification="APPROPRIATE", mechanism="deterministic"
            ),
        }
        comparison = compare_runs(labels, before, after)
        assert comparison.llm_volume_before == 0
        assert comparison.llm_volume_after == 1
        assert comparison.llm_volume_delta == 1

    def test_llm_volume_excludes_excluded_truth_or_prediction(self) -> None:
        labels = {
            "truth-excluded": "RETURNED_NOT_TRANSFUSED",
            "prediction-excluded": "APPROPRIATE",
            "scored": "APPROPRIATE",
        }
        verdicts = {
            "truth-excluded": CaseVerdict(
                reqno="truth-excluded",
                classification="APPROPRIATE",
                mechanism="llm",
            ),
            "prediction-excluded": CaseVerdict(
                reqno="prediction-excluded",
                classification="RETURNED_NOT_TRANSFUSED",
                mechanism="llm",
            ),
            "scored": CaseVerdict(
                reqno="scored", classification="NEEDS_REVIEW", mechanism="llm"
            ),
        }
        comparison = compare_runs(labels, verdicts, verdicts)
        assert comparison.llm_volume_before == 1
        assert comparison.llm_volume_after == 1

    def test_regression_is_a_correct_appropriate_flipped_away(self) -> None:
        labels = {"good": "APPROPRIATE"}
        before = {
            "good": CaseVerdict(
                reqno="good", classification="APPROPRIATE", mechanism="deterministic"
            )
        }
        after = {
            "good": CaseVerdict(
                reqno="good", classification="NEEDS_REVIEW", mechanism="llm"
            )
        }
        comparison = compare_runs(labels, before, after)
        assert comparison.regressions == ("good",)

    def test_fixing_a_wrong_clear_is_not_a_regression(self) -> None:
        # Human said inappropriate; before wrong (appropriate), after review.
        # This is a FIX, not a regression — regressions only track
        # truly-appropriate orders that stop being cleared.
        comparison = compare_runs(_HUMAN, _BEFORE, _AFTER)
        assert comparison.regressions == ()

    def test_matrix_accessor_fetches_scoped_run(self) -> None:
        comparison = compare_runs(_HUMAN, _BEFORE, _AFTER)
        before_llm = comparison.matrix("before", "llm")
        after_llm = comparison.matrix("after", "llm")
        assert before_llm.count("inappropriate", "appropriate") == 2
        assert after_llm.count("inappropriate", "appropriate") == 0

    def test_matrix_accessor_rejects_unknown_scope(self) -> None:
        comparison = compare_runs(_HUMAN, _BEFORE, _AFTER)
        with pytest.raises(KeyError):
            comparison.matrix("before", "nonexistent")  # type: ignore[arg-type]


class TestFindRegressions:
    def test_only_appropriate_truth_can_regress(self) -> None:
        # A human-inappropriate case flipping around is never a regression.
        labels = {"x": "INAPPROPRIATE"}
        before = {
            "x": CaseVerdict(reqno="x", classification="INAPPROPRIATE", mechanism="llm")
        }
        after = {
            "x": CaseVerdict(reqno="x", classification="NEEDS_REVIEW", mechanism="llm")
        }
        assert find_regressions(labels, before, after) == ()

    def test_requires_case_in_both_runs(self) -> None:
        labels = {"x": "APPROPRIATE"}
        before = {
            "x": CaseVerdict(
                reqno="x", classification="APPROPRIATE", mechanism="deterministic"
            )
        }
        after: dict[str, CaseVerdict] = {}
        assert find_regressions(labels, before, after) == ()
