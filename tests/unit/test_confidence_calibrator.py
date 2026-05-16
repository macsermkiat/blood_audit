"""RED-phase failing tests for issue #23 (bba.confidence_calibrator).

Each ``class`` maps to one acceptance criterion in the issue body. Tests
assert contracts (the WHY), not implementation choices — see PRD
§"Testing Decisions".

No implementation exists yet; every public function in
``bba.confidence_calibrator`` raises ``NotImplementedError`` in this
RED-phase scaffold. The module-level imports double as the public-API
surface check: if any re-export goes missing, collection fails before
any test runs.

Acceptance-criterion → test-class map (issue #23 body):

* AC ① "Implementation in ``src/bba/confidence_calibrator/``" → implicit
  by the module-level imports at the top of this file.
* AC ② "Isotonic fit math verified against scikit-learn reference" →
  :class:`TestIsotonicSklearnReference`.
* AC ③ "ECE computation tested against published example" →
  :class:`TestEcePublishedExample`.
* AC ④ "Agreement-based confidence: deterministic seed control for
  3× shuffle" → :class:`TestShuffleSeedsDeterministic`,
  :class:`TestAgreementConfidence`.
* AC ⑤ "Calibration plot generator (reliability diagram) to
  docs/eval/" → :class:`TestReliabilityDiagram`.
* AC ⑥ "Coverage ≥ 70%; ruff + mypy clean" → structural; coverage is
  enforced by the ralph-loop promise gate, ruff + mypy by CI.

Cross-cutting:

* :class:`TestModelImmutability` — Pydantic / dataclass ``frozen=True``.
* :class:`TestPropertyBased` — :mod:`hypothesis` property tests (the
  "deep module" check from the ralph-loop promise gate).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from bba.confidence_calibrator import (
    DEFAULT_AGREEMENT_RUNS,
    DEFAULT_N_BINS,
    ECE_RECAL_HOLDOUT_SIZE,
    REVIEW_CONFIDENCE_THRESHOLD,
    AgreementResult,
    BinStats,
    CalibratorNotFittedError,
    ConfidenceCalibratorError,
    EceResult,
    InvalidCalibrationDataError,
    IsotonicCalibrator,
    IsotonicFit,
    agreement_confidence,
    compute_ece,
    generate_reliability_diagram,
    pav_fit,
    shuffle_seeds,
)


# =============================================================================
# Reference fixtures (hardcoded sklearn / Guo et al. 2017 outputs).
#
# We do NOT import scikit-learn at runtime — the audit container has no sklearn
# dependency. These vectors were computed offline once against
# sklearn.isotonic.IsotonicRegression(out_of_bounds='clip') and are pinned here
# so a runtime PAV drift is caught immediately.
# =============================================================================

SKLEARN_REF_SCORES: Final = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
SKLEARN_REF_LABELS: Final = (0, 0, 0, 1, 0, 1, 1, 1, 1, 1)

# PAV on the above (manual derivation, matches sklearn IsotonicRegression):
#   block A: x in {0.1, 0.2, 0.3}, pooled y = 0.0    (weight 3)
#   block B: x in {0.4, 0.5},      pooled y = 0.5    (weight 2)
#   block C: x in {0.6..1.0},      pooled y = 1.0    (weight 5)
#
# Calibrated prediction at training x-values:
SKLEARN_REF_PRED: Final = {
    0.1: 0.0,
    0.2: 0.0,
    0.3: 0.0,
    0.4: 0.5,
    0.5: 0.5,
    0.6: 1.0,
    0.7: 1.0,
    0.8: 1.0,
    0.9: 1.0,
    1.0: 1.0,
}
# Linear interpolation in the (0.5, 0.5) → (0.6, 1.0) interval:
#   predict(0.55) = 0.5 + (0.55 - 0.5) / (0.6 - 0.5) * (1.0 - 0.5) = 0.75
SKLEARN_REF_INTERP_X: Final = 0.55
SKLEARN_REF_INTERP_Y: Final = 0.75


# Guo et al. 2017, "On Calibration of Modern Neural Networks", ICML.
# Worked 2-bin example (bin edges [0, 0.5, 1.0]):
#
#   bin 0: p=0.1 → label 0, p=0.4 → label 1
#     mean_conf = 0.25, accuracy = 0.5, gap = 0.25, weight = 2/4
#   bin 1: p=0.7 → label 1, p=0.9 → label 0
#     mean_conf = 0.8,  accuracy = 0.5, gap = 0.3,  weight = 2/4
#
#   ECE = (2/4) * 0.25 + (2/4) * 0.3 = 0.275
ECE_REF_PROBS: Final = (0.1, 0.4, 0.7, 0.9)
ECE_REF_LABELS: Final = (0, 1, 1, 0)
ECE_REF_N_BINS: Final = 2
ECE_REF_VALUE: Final = 0.275


# =============================================================================
# AC: Implementation in src/bba/confidence_calibrator/.
# Module-surface lock: any missing re-export breaks collection above.
# =============================================================================


class TestPublicSurface:
    """The full :mod:`bba.confidence_calibrator` import names are stable."""

    def test_default_constants_have_expected_values(self) -> None:
        # The 0.7 review threshold is the deployed gate (PRD §14); the
        # 200-row holdout is the monthly recalibration sample. Drift here
        # would silently change the audit pipeline's NEEDS_REVIEW rate.
        assert REVIEW_CONFIDENCE_THRESHOLD == 0.7
        assert ECE_RECAL_HOLDOUT_SIZE == 200
        assert DEFAULT_N_BINS == 10
        assert DEFAULT_AGREEMENT_RUNS == 3


# =============================================================================
# AC ②: Isotonic fit math verified against scikit-learn reference.
# =============================================================================


class TestIsotonicSklearnReference:
    """PAV fit and predict match the pinned sklearn IsotonicRegression output.

    Why: PRD §14 anchors the calibrated confidence to sklearn's behavior
    so the operator's intuition ("this is just isotonic regression") is
    correct. A drift here would silently shift the 0.7 review threshold's
    operating point.
    """

    def test_predict_matches_sklearn_at_training_points(self) -> None:
        cal = IsotonicCalibrator()
        cal.fit(SKLEARN_REF_SCORES, SKLEARN_REF_LABELS)
        predictions = cal.predict(tuple(SKLEARN_REF_PRED.keys()))
        for actual, expected in zip(predictions, SKLEARN_REF_PRED.values()):
            assert actual == pytest.approx(expected, abs=1e-9)

    def test_predict_linear_interpolates_between_blocks(self) -> None:
        cal = IsotonicCalibrator()
        cal.fit(SKLEARN_REF_SCORES, SKLEARN_REF_LABELS)
        (y,) = cal.predict([SKLEARN_REF_INTERP_X])
        assert y == pytest.approx(SKLEARN_REF_INTERP_Y, abs=1e-9)

    def test_pav_fit_returns_normalized_curve(self) -> None:
        # Functional ``pav_fit`` is the offline equivalence target for the
        # sklearn reference test. Strict-monotone X, non-decreasing Y in
        # [0, 1] is the invariant the audit pipeline relies on.
        fit = pav_fit(SKLEARN_REF_SCORES, SKLEARN_REF_LABELS)
        assert isinstance(fit, IsotonicFit)
        assert fit.n_training == len(SKLEARN_REF_SCORES)
        xs = fit.x_thresholds
        ys = fit.y_values
        assert len(xs) == len(ys)
        assert len(xs) >= 2
        assert all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))
        assert all(ys[i] <= ys[i + 1] for i in range(len(ys) - 1))
        assert all(0.0 <= y <= 1.0 for y in ys)


class TestIsotonicPredictClipsOutOfRange:
    """Inputs below ``min(x_thresholds)`` clip down, above clip up.

    Why: a future raw LLM confidence outside the calibration range must
    NOT extrapolate — extrapolating an isotonic curve is undefined and
    would silently produce probabilities outside [0, 1].
    """

    def test_predict_below_range_clips_to_lowest_y(self) -> None:
        cal = IsotonicCalibrator()
        cal.fit(SKLEARN_REF_SCORES, SKLEARN_REF_LABELS)
        (y,) = cal.predict([-0.5])
        assert y == pytest.approx(0.0, abs=1e-9)

    def test_predict_above_range_clips_to_highest_y(self) -> None:
        cal = IsotonicCalibrator()
        cal.fit(SKLEARN_REF_SCORES, SKLEARN_REF_LABELS)
        (y,) = cal.predict([1.5])
        assert y == pytest.approx(1.0, abs=1e-9)


class TestIsotonicCalibratorState:
    """Lifecycle: predict before fit fails loud; refit overwrites."""

    def test_predict_before_fit_raises(self) -> None:
        cal = IsotonicCalibrator()
        assert cal.is_fitted is False
        with pytest.raises(CalibratorNotFittedError):
            cal.predict([0.5])

    def test_fit_result_before_fit_raises(self) -> None:
        cal = IsotonicCalibrator()
        with pytest.raises(CalibratorNotFittedError):
            _ = cal.fit_result

    def test_fit_marks_is_fitted_true(self) -> None:
        cal = IsotonicCalibrator()
        cal.fit(SKLEARN_REF_SCORES, SKLEARN_REF_LABELS)
        assert cal.is_fitted is True

    def test_refit_overwrites_prior_curve(self) -> None:
        # Monthly recalibration semantics (PRD §14): the second fit must
        # produce a different curve when trained on different labels.
        cal = IsotonicCalibrator()
        cal.fit(SKLEARN_REF_SCORES, SKLEARN_REF_LABELS)
        first = cal.fit_result
        # Reversed labels → reversed monotone curve (after pooling).
        cal.fit(SKLEARN_REF_SCORES, tuple(1 - lab for lab in SKLEARN_REF_LABELS))
        second = cal.fit_result
        assert second != first


class TestIsotonicInputValidation:
    """Structural-contract validation; surfaces bad calibration jobs early."""

    def test_length_mismatch_raises(self) -> None:
        cal = IsotonicCalibrator()
        with pytest.raises(InvalidCalibrationDataError):
            cal.fit([0.1, 0.2, 0.3], [0, 1])

    def test_empty_input_raises(self) -> None:
        cal = IsotonicCalibrator()
        with pytest.raises(InvalidCalibrationDataError):
            cal.fit([], [])

    def test_label_not_zero_or_one_raises(self) -> None:
        cal = IsotonicCalibrator()
        with pytest.raises(InvalidCalibrationDataError):
            cal.fit([0.1, 0.2], [0, 2])

    def test_score_out_of_range_raises(self) -> None:
        cal = IsotonicCalibrator()
        with pytest.raises(InvalidCalibrationDataError):
            cal.fit([0.1, 1.5], [0, 1])


# =============================================================================
# AC ③: ECE computation tested against published example.
# =============================================================================


class TestEcePublishedExample:
    """Guo et al. 2017 worked 2-bin example matches eq. (3).

    Why: a wrong ECE silently breaks the monthly recalibration gate; the
    transfusion committee would not see drift even when calibration has
    degraded.
    """

    def test_published_two_bin_example(self) -> None:
        result = compute_ece(ECE_REF_PROBS, ECE_REF_LABELS, n_bins=ECE_REF_N_BINS)
        assert isinstance(result, EceResult)
        assert result.ece == pytest.approx(ECE_REF_VALUE, abs=1e-9)
        assert result.n_samples == len(ECE_REF_PROBS)
        assert result.n_bins == ECE_REF_N_BINS

    def test_per_bin_breakdown_matches_published(self) -> None:
        result = compute_ece(ECE_REF_PROBS, ECE_REF_LABELS, n_bins=ECE_REF_N_BINS)
        assert len(result.bins) == ECE_REF_N_BINS
        b0, b1 = result.bins
        assert b0.count == 2
        assert b0.mean_confidence == pytest.approx(0.25, abs=1e-9)
        assert b0.accuracy == pytest.approx(0.5, abs=1e-9)
        assert b1.count == 2
        assert b1.mean_confidence == pytest.approx(0.8, abs=1e-9)
        assert b1.accuracy == pytest.approx(0.5, abs=1e-9)


class TestEceEdgeCases:
    """Empty bins, perfect calibration, boundary inputs."""

    def test_perfect_calibration_is_zero(self) -> None:
        # Every sample at p=0.5 with half labelled 1, half labelled 0:
        # mean_conf == accuracy in the single populated bin → ECE = 0.
        result = compute_ece([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0], n_bins=2)
        assert result.ece == pytest.approx(0.0, abs=1e-12)

    def test_empty_bins_contribute_zero_weight(self) -> None:
        # All probabilities cluster in one bin; the other bin is empty.
        # ECE must remain finite (no divide-by-zero) and the empty bin
        # still appears in the result.
        result = compute_ece([0.05, 0.10], [0, 1], n_bins=2)
        assert len(result.bins) == 2
        assert result.bins[1].count == 0
        assert 0.0 <= result.ece <= 1.0

    def test_probability_at_one_lands_in_last_bin(self) -> None:
        # The final bin is closed on the right; p=1.0 must not overflow.
        result = compute_ece([1.0, 0.0], [1, 0], n_bins=10)
        # Two distinct singleton bins: first (count 1, conf 0, acc 0)
        # and last (count 1, conf 1, acc 1). Both have zero gap →
        # ECE = 0.
        assert result.ece == pytest.approx(0.0, abs=1e-12)


class TestEceInputValidation:
    """Structural-contract validation for the ECE entry point."""

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(InvalidCalibrationDataError):
            compute_ece([0.1, 0.2], [0])

    def test_empty_raises(self) -> None:
        with pytest.raises(InvalidCalibrationDataError):
            compute_ece([], [])

    def test_n_bins_zero_raises(self) -> None:
        with pytest.raises(InvalidCalibrationDataError):
            compute_ece([0.5], [1], n_bins=0)

    def test_prob_out_of_range_raises(self) -> None:
        with pytest.raises(InvalidCalibrationDataError):
            compute_ece([1.2], [1])

    def test_label_not_binary_raises(self) -> None:
        with pytest.raises(InvalidCalibrationDataError):
            compute_ece([0.5], [3])


# =============================================================================
# AC ④: Agreement-based confidence with deterministic 3× shuffle seeds.
# =============================================================================


class TestShuffleSeedsDeterministic:
    """Same ``base_seed`` + ``n_runs`` produces the exact same seed tuple.

    Why: PRD §14 "deterministic seed control" makes the audit row
    fully reproducible — re-running the monthly job on the same input
    must produce the same agreement verdict.
    """

    def test_same_inputs_same_output(self) -> None:
        a = shuffle_seeds(42, 3)
        b = shuffle_seeds(42, 3)
        assert a == b

    def test_returns_exactly_n_runs_seeds(self) -> None:
        seeds = shuffle_seeds(42, DEFAULT_AGREEMENT_RUNS)
        assert isinstance(seeds, tuple)
        assert len(seeds) == DEFAULT_AGREEMENT_RUNS

    def test_different_base_seeds_produce_different_outputs(self) -> None:
        # The mixing function must spread small base_seed differences;
        # base_seed + i would correlate the 3 shuffles too tightly.
        a = shuffle_seeds(42, 3)
        b = shuffle_seeds(43, 3)
        assert a != b

    def test_seeds_within_run_are_distinct(self) -> None:
        # All 3 shufflings must use different seeds so the three
        # few-shot orderings actually differ.
        seeds = shuffle_seeds(42, 3)
        assert len(set(seeds)) == 3

    def test_negative_base_seed_raises(self) -> None:
        with pytest.raises(InvalidCalibrationDataError):
            shuffle_seeds(-1, 3)

    def test_zero_n_runs_raises(self) -> None:
        with pytest.raises(InvalidCalibrationDataError):
            shuffle_seeds(42, 0)


class TestAgreementConfidence:
    """Vote tabulation: count / total with first-seen tie-breaking."""

    def test_unanimous_three_returns_one_point_zero(self) -> None:
        result = agreement_confidence(
            ["APPROPRIATE", "APPROPRIATE", "APPROPRIATE"],
        )
        assert result.majority == "APPROPRIATE"
        assert result.agreement_count == 3
        assert result.confidence == pytest.approx(1.0, abs=1e-9)

    def test_two_vs_one_returns_two_thirds(self) -> None:
        result = agreement_confidence(
            ["APPROPRIATE", "INAPPROPRIATE", "APPROPRIATE"],
        )
        assert result.majority == "APPROPRIATE"
        assert result.agreement_count == 2
        assert result.confidence == pytest.approx(2.0 / 3.0, abs=1e-9)

    def test_three_way_tie_resolves_to_first_seen(self) -> None:
        result = agreement_confidence(
            ["INAPPROPRIATE", "APPROPRIATE", "NEEDS_REVIEW"],
        )
        # First-seen tie-breaking: INAPPROPRIATE wins. The agreement
        # count is 1 (the count of the majority class).
        assert result.majority == "INAPPROPRIATE"
        assert result.agreement_count == 1
        assert result.confidence == pytest.approx(1.0 / 3.0, abs=1e-9)

    def test_classifications_preserved_in_order(self) -> None:
        result = agreement_confidence(
            ["A", "B", "A"],
        )
        assert result.classifications == ("A", "B", "A")

    def test_empty_raises(self) -> None:
        with pytest.raises(InvalidCalibrationDataError):
            agreement_confidence([])


# =============================================================================
# AC ⑤: Calibration plot generator (reliability diagram) to docs/eval/.
# =============================================================================


class TestReliabilityDiagram:
    """SVG reliability diagram is written and contains the required parts.

    Why: the diagram is the transfusion committee's monthly visual
    drift check (PRD §14). Wrong bin counts, missing diagonal, or a
    silently-skipped write would hide degraded calibration.
    """

    def test_creates_svg_file_at_target_path(self, tmp_path: Path) -> None:
        out = tmp_path / "docs" / "eval" / "reliability.svg"
        returned = generate_reliability_diagram(
            ECE_REF_PROBS, ECE_REF_LABELS, out, n_bins=ECE_REF_N_BINS,
        )
        assert returned == out
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<svg" in content
        assert "</svg>" in content

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "a" / "b" / "c" / "reliability.svg"
        generate_reliability_diagram(
            ECE_REF_PROBS, ECE_REF_LABELS, out, n_bins=ECE_REF_N_BINS,
        )
        assert out.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        out = tmp_path / "reliability.svg"
        out.write_text("stale", encoding="utf-8")
        generate_reliability_diagram(
            ECE_REF_PROBS, ECE_REF_LABELS, out, n_bins=ECE_REF_N_BINS,
        )
        assert "stale" not in out.read_text(encoding="utf-8")

    def test_includes_diagonal_reference_line(self, tmp_path: Path) -> None:
        # The y=x diagonal is the "perfect calibration" reference; without
        # it the diagram cannot communicate drift visually.
        out = tmp_path / "reliability.svg"
        generate_reliability_diagram(
            ECE_REF_PROBS, ECE_REF_LABELS, out, n_bins=ECE_REF_N_BINS,
        )
        content = out.read_text(encoding="utf-8")
        # Convention: the diagonal is tagged so it is locatable from the
        # SVG without re-parsing geometry.
        assert "reliability-diagonal" in content

    def test_renders_one_marker_per_non_empty_bin(self, tmp_path: Path) -> None:
        # The reference inputs put 2 samples in each of the 2 bins, so
        # both bins are non-empty and both emit a marker.
        out = tmp_path / "reliability.svg"
        generate_reliability_diagram(
            ECE_REF_PROBS, ECE_REF_LABELS, out, n_bins=ECE_REF_N_BINS,
        )
        content = out.read_text(encoding="utf-8")
        assert content.count("reliability-bin") == ECE_REF_N_BINS

    def test_empty_bins_are_skipped(self, tmp_path: Path) -> None:
        # Both probs fall in bin 0; bin 1 is empty. The empty bin must
        # NOT produce a marker — a marker at accuracy=0 would falsely
        # show zero accuracy for a bin that had no samples and would
        # mislead the transfusion committee's drift review.
        out = tmp_path / "reliability.svg"
        generate_reliability_diagram(
            [0.05, 0.10], [0, 1], out, n_bins=2,
        )
        content = out.read_text(encoding="utf-8")
        assert content.count("reliability-bin") == 1

    def test_title_is_xml_escaped(self, tmp_path: Path) -> None:
        # Caller-supplied title may contain ``&`` / ``<`` / ``>``; the
        # renderer must escape them so the SVG stays well-formed and
        # cannot be markup-injected.
        out = tmp_path / "reliability.svg"
        generate_reliability_diagram(
            ECE_REF_PROBS, ECE_REF_LABELS, out, n_bins=ECE_REF_N_BINS,
            title="Drift & calibration <check>",
        )
        content = out.read_text(encoding="utf-8")
        assert "&amp;" in content
        assert "&lt;check&gt;" in content
        # The raw injected ``<check>`` tag must not appear unescaped.
        assert "<check>" not in content

    def test_invalid_inputs_raise_before_writing(self, tmp_path: Path) -> None:
        out = tmp_path / "should_not_exist.svg"
        with pytest.raises(InvalidCalibrationDataError):
            generate_reliability_diagram([], [], out)
        # The file must NOT have been created — a half-written diagram
        # would mislead the operator who sees a fresh mtime.
        assert not out.exists()


# =============================================================================
# Cross-cutting: model immutability.
# =============================================================================


class TestModelImmutability:
    """Pydantic models and the IsotonicFit dataclass are frozen.

    Why: an audit row's calibrated confidence must be reconstructible
    from frozen inputs six months later (PRD §"reproducibility = we
    have the original answer"). Silent mutation of a model after the
    fact would break that contract.
    """

    def test_bin_stats_is_frozen(self) -> None:
        bs = BinStats(
            bin_lower=0.0, bin_upper=0.5, count=2,
            mean_confidence=0.25, accuracy=0.5,
        )
        with pytest.raises(ValidationError):
            bs.count = 99  # type: ignore[misc]

    def test_ece_result_is_frozen(self) -> None:
        r = EceResult(ece=0.0, n_samples=0, n_bins=1, bins=())
        with pytest.raises(ValidationError):
            r.ece = 0.5  # type: ignore[misc]

    def test_agreement_result_is_frozen(self) -> None:
        a = AgreementResult(
            classifications=("A",), majority="A",
            agreement_count=1, confidence=1.0,
        )
        with pytest.raises(ValidationError):
            a.confidence = 0.5  # type: ignore[misc]

    def test_isotonic_fit_is_frozen(self) -> None:
        fit = IsotonicFit(
            x_thresholds=(0.1, 0.5),
            y_values=(0.0, 1.0),
            n_training=2,
        )
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            fit.n_training = 99  # type: ignore[misc]


class TestExceptionHierarchy:
    """All public errors descend from :class:`ConfidenceCalibratorError`.

    Why: callers in the audit pipeline catch the base class to route
    a failed calibration job to a single error-handling path. A new
    error type that bypasses the base would silently escape that path.
    """

    def test_calibrator_not_fitted_descends(self) -> None:
        assert issubclass(CalibratorNotFittedError, ConfidenceCalibratorError)

    def test_invalid_calibration_data_descends(self) -> None:
        assert issubclass(InvalidCalibrationDataError, ConfidenceCalibratorError)


# =============================================================================
# Hypothesis property tests — the "deep module" check from the promise gate.
# =============================================================================


class TestPropertyBased:
    """Invariants that must hold across random valid inputs."""

    @settings(max_examples=50, deadline=None)
    @given(
        scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=2,
            max_size=50,
            unique=True,
        ),
        labels_seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_isotonic_predict_is_monotone(
        self,
        scores: list[float],
        labels_seed: int,
    ) -> None:
        # Derive labels deterministically from the seed so the property
        # is reproducible: bit i of the seed → label for sorted position i.
        sorted_scores = sorted(scores)
        labels = [(labels_seed >> (i % 31)) & 1 for i in range(len(sorted_scores))]
        cal = IsotonicCalibrator()
        cal.fit(sorted_scores, labels)
        # Predict at the training points; output must be non-decreasing.
        preds = cal.predict(sorted_scores)
        for i in range(len(preds) - 1):
            assert preds[i] <= preds[i + 1] + 1e-9

    @settings(max_examples=50, deadline=None)
    @given(
        scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=2,
            max_size=50,
            unique=True,
        ),
        labels_seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_isotonic_predict_in_unit_interval(
        self,
        scores: list[float],
        labels_seed: int,
    ) -> None:
        sorted_scores = sorted(scores)
        labels = [(labels_seed >> (i % 31)) & 1 for i in range(len(sorted_scores))]
        cal = IsotonicCalibrator()
        cal.fit(sorted_scores, labels)
        preds = cal.predict(sorted_scores)
        for p in preds:
            assert 0.0 - 1e-9 <= p <= 1.0 + 1e-9

    @settings(max_examples=50, deadline=None)
    @given(
        probs=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            min_size=1,
            max_size=50,
        ),
        labels_seed=st.integers(min_value=0, max_value=2**31 - 1),
        n_bins=st.integers(min_value=1, max_value=20),
    )
    def test_ece_is_in_unit_interval(
        self,
        probs: list[float],
        labels_seed: int,
        n_bins: int,
    ) -> None:
        labels = [(labels_seed >> (i % 31)) & 1 for i in range(len(probs))]
        result = compute_ece(probs, labels, n_bins=n_bins)
        assert 0.0 <= result.ece <= 1.0
        assert result.n_samples == len(probs)
        assert result.n_bins == n_bins
        assert len(result.bins) == n_bins

    @settings(max_examples=30, deadline=None)
    @given(
        base_seed=st.integers(min_value=0, max_value=2**31 - 1),
        n_runs=st.integers(min_value=1, max_value=10),
    )
    def test_shuffle_seeds_deterministic_property(
        self,
        base_seed: int,
        n_runs: int,
    ) -> None:
        # The cornerstone of audit reproducibility: same inputs, same
        # output, every time, forever.
        a = shuffle_seeds(base_seed, n_runs)
        b = shuffle_seeds(base_seed, n_runs)
        assert a == b
        assert len(a) == n_runs
