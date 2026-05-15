"""RED-phase failing tests for issue #20 (bba.eval_harness).

Each ``class`` maps to one acceptance criterion in the issue body. Tests
assert contracts (the WHY), not implementation choices — PRD §"Testing
Decisions". In this scaffold commit:

* Every behavioral test fails with ``NotImplementedError`` — the functions
  exist (so imports resolve) but raise ``NotImplementedError`` until the
  GREEN phase wires them up.
* :class:`TestModelImmutability` and :class:`TestPublicSurface` PASS in
  RED on purpose: they are regression guards on the declared interface
  contract (``frozen=True``, public re-exports, threshold-constant value),
  not behavioral assertions. The audit_store precedent (commit ``bc45f82``)
  established this convention: "model-immutability tests pass because
  frozen=True is part of the interface scaffold itself".
* The top-of-file import block doubles as a public-surface check: a
  missing re-export fails collection before any test runs.

The acceptance-criterion → test-class map:

* AC ① "module exists in src/bba/eval_harness/"
  → public-import surface check (the imports at the top of this file)
* AC ② "stratified-with-enrichment produces target n per stratum"
  → :class:`TestStratifiedSampling`
* AC ③ "Horvitz-Thompson reweighting against known references"
  → :class:`TestHorvitzThompsonReweighting`
* AC ④ "Wilson CI against scipy reference"
  → :class:`TestWilsonCI`, :class:`TestNormalQuantile`
* AC ⑤ "κ + Gwet's AC1 + PABAK against published example tables"
  → :class:`TestCohenKappa`, :class:`TestGwetAC1`, :class:`TestPABAK`,
    :class:`TestAgreementCombined`, :class:`TestKappaPrevalenceParadox`
* AC ⑥ "cluster-robust SE on synthetic clustered data"
  → :class:`TestClusterRobustSE`
* AC ⑦ "LOMO-CV; auto-fallback to blocked temporal if ≥ 12 months"
  → :class:`TestLOMOCV`, :class:`TestBlockedTemporalSplit`,
    :class:`TestSplitStrategyAutoSelect`, :class:`TestTemporalCVSplits`
* AC ⑧ "hierarchical FDR multiple-comparisons correction"
  → :class:`TestBonferroniCorrection`, :class:`TestBHFDRCorrection`,
    :class:`TestHierarchicalCorrection`
* PRD §11 "verifier-as-classifier confusion-matrix consumer from #11"
  → :class:`TestVerifierAsClassifier`
* PRD §11 "outcome-anchored falsification on labeled subset"
  → :class:`TestOutcomeAnchoredFalsification`
* Property tests (hypothesis) — "deep" module requirement
  → :class:`TestPropertyWilsonContainsPoint`,
    :class:`TestPropertyBonferroniMonotonic`,
    :class:`TestPropertyHorvitzThompsonUniform`
* Pure-function / determinism contract
  → :class:`TestPureFunctionContract`
"""

from __future__ import annotations

import io
import sys
from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from bba.eval_harness import (
    DEFAULT_INAPPROPRIATE_ENRICHMENT_PER_STRATUM,
    DEFAULT_STRATUM_TARGETS,
    LOMO_DATASET_THRESHOLD_MONTHS,
    AgreementResult,
    AuditCase,
    ClassifierMetrics,
    CorrectedTest,
    EmptyInputError,
    FalsificationOutcome,
    FalsificationResult,
    HierarchicalCorrectionResult,
    HorvitzThompsonEstimate,
    SamplingTargets,
    ShapeMismatchError,
    SplitStrategy,
    StratifiedSample,
    Stratum,
    StratumDraw,
    TemporalSplit,
    WilsonInterval,
    agreement_with_metrics,
    benjamini_hochberg_correction,
    blocked_temporal_split,
    bonferroni_correction,
    cluster_robust_proportion_ci,
    cohen_kappa,
    dataset_month_span,
    evaluate_confusion_matrix,
    gwet_ac1,
    hierarchical_correction,
    horvitz_thompson_prevalence,
    lomo_cv_splits,
    normal_quantile,
    outcome_anchored_falsification,
    pabak,
    select_split_strategy,
    stratified_with_enrichment,
    temporal_cv_splits,
    wilson_ci,
)
from bba.quote_grounder import ConfusionMatrix


# =============================================================================
# Fixtures — minimal AuditCase builder and per-stratum population factories.
#
# The harness operates on a thin projection of AuditRow; full audit rows have
# ~30 fields that the harness never reads. Building small AuditCases here
# keeps every test focused on the contract under test, not on filling 30
# unrelated kwargs.
# =============================================================================


def _case(
    *,
    audit_id: str,
    stratum: Stratum = Stratum.HB_7_10,
    pred: str = "APPROPRIATE",
    gold: str | None = None,
    physician: str = "MD-001",
    ward: str = "W-IM-3",
    when: datetime | None = None,
) -> AuditCase:
    return AuditCase(
        audit_id=audit_id,
        stratum=stratum,
        pred_classification=pred,  # type: ignore[arg-type]
        gold_classification=gold,  # type: ignore[arg-type]
        order_datetime=when or datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        physician_id=physician,
        ward=ward,
    )


def _stratum_population(
    stratum: Stratum,
    *,
    n_total: int,
    n_positives: int,
    when: datetime | None = None,
) -> list[AuditCase]:
    """A synthetic population of ``n_total`` cases in ``stratum``, with
    ``n_positives`` predicted INAPPROPRIATE and the rest APPROPRIATE."""
    cases: list[AuditCase] = []
    for i in range(n_positives):
        cases.append(
            _case(
                audit_id=f"{stratum.value}-pos-{i:04d}",
                stratum=stratum,
                pred="INAPPROPRIATE",
                when=when,
            )
        )
    for i in range(n_total - n_positives):
        cases.append(
            _case(
                audit_id=f"{stratum.value}-neg-{i:04d}",
                stratum=stratum,
                pred="APPROPRIATE",
                when=when,
            )
        )
    return cases


def _default_targets() -> SamplingTargets:
    return SamplingTargets(
        per_stratum=DEFAULT_STRATUM_TARGETS,
        inappropriate_enrichment_per_stratum=DEFAULT_INAPPROPRIATE_ENRICHMENT_PER_STRATUM,
    )


# =============================================================================
# AC ② Stratified-with-enrichment sampling.
#
# The PRD User Story #25 numbers (Hb<7: 140, Hb 7-10: 350, Hb>10: 260,
# INSUFFICIENT: 100, adversarial: 80, cohort-exception: 140) define the
# per-stratum targets; the per-stratum INAPPROPRIATE-positive enrichment is
# 138 (PRD §11). These are the report's load-bearing sampling promises and
# must be testable without subjective interpretation.
# =============================================================================


class TestStratifiedSampling:
    """stratified_with_enrichment honors per-stratum targets and enrichment."""

    def _full_population(self) -> list[AuditCase]:
        # Build a population large enough that every stratum's target +
        # enrichment can be satisfied. Per-stratum positive counts are chosen
        # to exceed the enrichment target so the "enough positives" branch
        # is exercised here; an undersized stratum is tested separately.
        population: list[AuditCase] = []
        sizes = {
            Stratum.HB_LT_7: (1_500, 400),
            Stratum.HB_7_10: (4_000, 500),
            Stratum.HB_GT_10: (3_000, 300),
            Stratum.INSUFFICIENT: (800, 200),
            Stratum.ADVERSARIAL: (400, 200),
            Stratum.COHORT_EXCEPTION: (700, 250),
        }
        for stratum, (n_total, n_pos) in sizes.items():
            population.extend(
                _stratum_population(
                    stratum, n_total=n_total, n_positives=n_pos
                )
            )
        return population

    def test_each_stratum_meets_target(self) -> None:
        # The headline contract: every stratum must hit its declared target
        # exactly. Slipping by even one would make the per-stratum ±0.05
        # Wilson CI claim untrue at the boundary.
        sample = stratified_with_enrichment(
            self._full_population(), _default_targets(), rng_seed=42
        )
        by_stratum = {d.stratum: d for d in sample.draws}
        for stratum, target in DEFAULT_STRATUM_TARGETS.items():
            assert stratum in by_stratum, f"stratum {stratum} missing from sample"
            assert len(by_stratum[stratum].cases) == target

    def test_inappropriate_enrichment_target_met(self) -> None:
        # PRD §11: "~138 INAPPROPRIATE-positives per stratum." The sampler
        # oversamples positives toward 138, capped by both the stratum
        # target and the available positive pool. For the standard PRD
        # targets, only the adversarial stratum (target=80) is below the
        # 138 enrichment cap; every other stratum must hit ≥ 138 positives.
        sample = stratified_with_enrichment(
            self._full_population(), _default_targets(), rng_seed=42
        )
        by_stratum = {d.stratum: d for d in sample.draws}
        for stratum, target in DEFAULT_STRATUM_TARGETS.items():
            expected_min = min(
                DEFAULT_INAPPROPRIATE_ENRICHMENT_PER_STRATUM, target
            )
            assert by_stratum[stratum].drawn_positives >= expected_min, (
                f"stratum {stratum} drew "
                f"{by_stratum[stratum].drawn_positives} positives "
                f"(expected ≥ {expected_min})"
            )

    def test_deterministic_with_seed(self) -> None:
        # PRD §"Reproducibility = 'we have the original answer'" — the
        # sampler must yield identical draws given the same seed and inputs.
        pop = self._full_population()
        a = stratified_with_enrichment(pop, _default_targets(), rng_seed=7)
        b = stratified_with_enrichment(pop, _default_targets(), rng_seed=7)
        a_ids = tuple(
            tuple(c.audit_id for c in d.cases) for d in a.draws
        )
        b_ids = tuple(
            tuple(c.audit_id for c in d.cases) for d in b.draws
        )
        assert a_ids == b_ids

    def test_different_seed_yields_different_draw(self) -> None:
        # Determinism without RNG variation would mean the seed is ignored
        # — a silent regression. Two different seeds must produce at least
        # one differing case across the (sizeable) sample.
        pop = self._full_population()
        a = stratified_with_enrichment(pop, _default_targets(), rng_seed=1)
        b = stratified_with_enrichment(pop, _default_targets(), rng_seed=2)
        a_all = {c.audit_id for d in a.draws for c in d.cases}
        b_all = {c.audit_id for d in b.draws for c in d.cases}
        assert a_all != b_all

    def test_each_case_appears_at_most_once(self) -> None:
        # No double-dipping. Without-replacement sampling within a stratum
        # is implicit; the test guards the contract.
        sample = stratified_with_enrichment(
            self._full_population(), _default_targets(), rng_seed=42
        )
        all_ids = [c.audit_id for d in sample.draws for c in d.cases]
        assert len(all_ids) == len(set(all_ids))

    def test_inclusion_probabilities_recorded(self) -> None:
        # The HT reweighter (AC #3) consumes per-stratum inclusion
        # probabilities. The sampler must emit them — positive- and base-
        # rate inclusion probs differ within a stratum because of enrichment.
        sample = stratified_with_enrichment(
            self._full_population(), _default_targets(), rng_seed=42
        )
        for draw in sample.draws:
            # base-rate prob == (target - drawn_positives) / (pop_size - pop_positives)
            non_pos_pop = draw.population_size - draw.population_positives
            non_pos_drawn = len(draw.cases) - draw.drawn_positives
            expected_base = non_pos_drawn / non_pos_pop
            assert draw.base_inclusion_probability == pytest.approx(
                expected_base, rel=0, abs=1e-12
            )
            # positive-rate prob == drawn_positives / population_positives
            expected_pos = draw.drawn_positives / draw.population_positives
            assert draw.positive_inclusion_probability == pytest.approx(
                expected_pos, rel=0, abs=1e-12
            )

    def test_population_smaller_than_target_returns_all(self) -> None:
        # When a stratum's population is smaller than its target, the sampler
        # returns the entire stratum — NOT an error. The report can then
        # cite "n=population_size" with the correct Wilson CI width rather
        # than failing the whole run.
        thin_pop = _stratum_population(
            Stratum.ADVERSARIAL, n_total=30, n_positives=10
        )
        targets = SamplingTargets(
            per_stratum={Stratum.ADVERSARIAL: 80},
            inappropriate_enrichment_per_stratum=5,
        )
        sample = stratified_with_enrichment(thin_pop, targets, rng_seed=0)
        assert len(sample.draws) == 1
        assert len(sample.draws[0].cases) == 30

    def test_enrichment_clipped_when_few_positives(self) -> None:
        # If the stratum has fewer positives than the enrichment target,
        # take all positives and fill the remainder from negatives — do not
        # raise unless the *total* target also cannot be met.
        pop = _stratum_population(
            Stratum.COHORT_EXCEPTION, n_total=500, n_positives=40
        )
        targets = SamplingTargets(
            per_stratum={Stratum.COHORT_EXCEPTION: 140},
            inappropriate_enrichment_per_stratum=138,
        )
        sample = stratified_with_enrichment(pop, targets, rng_seed=0)
        draw = sample.draws[0]
        assert draw.drawn_positives == 40  # all positives, not 138
        assert len(draw.cases) == 140

    def test_scarce_negatives_draws_extra_positives(self) -> None:
        # Codex P0: if negatives are scarcer than (target - enrichment),
        # the sampler must draw extra positives to hit the stratum target —
        # NOT raise. Example: target=140, enrichment=138, pop_pos=500,
        # pop_neg=1. Need at least 139 positives + 1 negative.
        pop = _stratum_population(
            Stratum.COHORT_EXCEPTION, n_total=501, n_positives=500
        )
        targets = SamplingTargets(
            per_stratum={Stratum.COHORT_EXCEPTION: 140},
            inappropriate_enrichment_per_stratum=138,
        )
        sample = stratified_with_enrichment(pop, targets, rng_seed=0)
        draw = sample.draws[0]
        assert len(draw.cases) == 140
        assert draw.drawn_positives == 139  # not 138, to leave room for 1 neg
        assert draw.drawn_positives + (140 - draw.drawn_positives) == 140


# =============================================================================
# AC ③ Horvitz-Thompson reweighting against known references.
# =============================================================================


class TestHorvitzThompsonReweighting:
    """horvitz_thompson_prevalence is unbiased under known sampling design."""

    def _two_stratum_sample(
        self,
        *,
        stratum_a_pop: int,
        stratum_a_pos_pop: int,
        stratum_a_drawn: int,
        stratum_a_drawn_pos: int,
        stratum_b_pop: int,
        stratum_b_pos_pop: int,
        stratum_b_drawn: int,
        stratum_b_drawn_pos: int,
    ) -> StratifiedSample:
        # Hand-rolled StratifiedSample with deterministic inclusion probs and
        # known per-stratum positives. Used to compute the HT estimator by
        # hand and compare to the function's output.
        def draw(
            stratum: Stratum, pop: int, pos_pop: int, drawn: int, drawn_pos: int
        ) -> StratumDraw:
            cases = (
                tuple(
                    _case(
                        audit_id=f"{stratum.value}-p-{i}",
                        stratum=stratum,
                        pred="INAPPROPRIATE",
                    )
                    for i in range(drawn_pos)
                )
                + tuple(
                    _case(
                        audit_id=f"{stratum.value}-n-{i}",
                        stratum=stratum,
                        pred="APPROPRIATE",
                    )
                    for i in range(drawn - drawn_pos)
                )
            )
            return StratumDraw(
                stratum=stratum,
                cases=cases,
                population_size=pop,
                population_positives=pos_pop,
                drawn_positives=drawn_pos,
                base_inclusion_probability=(drawn - drawn_pos) / (pop - pos_pop),
                positive_inclusion_probability=drawn_pos / pos_pop,
            )

        return StratifiedSample(
            draws=(
                draw(
                    Stratum.HB_LT_7,
                    stratum_a_pop,
                    stratum_a_pos_pop,
                    stratum_a_drawn,
                    stratum_a_drawn_pos,
                ),
                draw(
                    Stratum.HB_GT_10,
                    stratum_b_pop,
                    stratum_b_pos_pop,
                    stratum_b_drawn,
                    stratum_b_drawn_pos,
                ),
            )
        )

    def test_known_population_prevalence_recovered(self) -> None:
        # Hand-computed reference: two strata, no enrichment within a stratum
        # (i.e., the positive- and base-rate inclusion probabilities are
        # equal). The HT estimator should recover the population prevalence
        # exactly: (200 + 30) / (1000 + 1000) = 0.115.
        sample = self._two_stratum_sample(
            stratum_a_pop=1000,
            stratum_a_pos_pop=200,
            stratum_a_drawn=100,
            stratum_a_drawn_pos=20,  # 20/200 = 0.10 == 80/800
            stratum_b_pop=1000,
            stratum_b_pos_pop=30,
            stratum_b_drawn=100,
            stratum_b_drawn_pos=3,  # 3/30 = 0.10 == 97/970
        )

        def is_pos(c: AuditCase) -> bool:
            return c.pred_classification == "INAPPROPRIATE"

        est = horvitz_thompson_prevalence(sample, is_pos)
        assert est.prevalence == pytest.approx(0.115, abs=1e-9)

    def test_enrichment_undone_by_reweighting(self) -> None:
        # Adversarial case: the sample is enriched (positive inclusion prob
        # higher than base-rate prob), so the *sample* prevalence is biased
        # high. The HT estimator must reweight back to the true population
        # prevalence.
        # Single stratum: 1000 cases, 100 positives → true prevalence 0.10.
        # Enriched sample: draw 50 positives (prob 50/100 = 0.50) and 50
        # negatives (prob 50/900 ≈ 0.05556). Sample prevalence = 0.50;
        # HT-reweighted prevalence should equal the true 0.10.
        positives = tuple(
            _case(audit_id=f"p{i}", pred="INAPPROPRIATE") for i in range(50)
        )
        negatives = tuple(
            _case(audit_id=f"n{i}", pred="APPROPRIATE") for i in range(50)
        )
        sample = StratifiedSample(
            draws=(
                StratumDraw(
                    stratum=Stratum.HB_LT_7,
                    cases=positives + negatives,
                    population_size=1000,
                    population_positives=100,
                    drawn_positives=50,
                    base_inclusion_probability=50 / 900,
                    positive_inclusion_probability=50 / 100,
                ),
            )
        )

        def is_pos(c: AuditCase) -> bool:
            return c.pred_classification == "INAPPROPRIATE"

        est = horvitz_thompson_prevalence(sample, is_pos)
        # HT: sum(y_i/pi_i) = 50 / 0.5 = 100; / N = 100/1000 = 0.10.
        assert est.prevalence == pytest.approx(0.10, abs=1e-9)

    def test_zero_indicator_yields_zero_prevalence(self) -> None:
        sample = self._two_stratum_sample(
            stratum_a_pop=100,
            stratum_a_pos_pop=10,
            stratum_a_drawn=20,
            stratum_a_drawn_pos=5,
            stratum_b_pop=100,
            stratum_b_pos_pop=10,
            stratum_b_drawn=20,
            stratum_b_drawn_pos=5,
        )
        est = horvitz_thompson_prevalence(sample, lambda _c: False)
        assert est.prevalence == pytest.approx(0.0, abs=1e-12)

    def test_variance_and_se_consistent(self) -> None:
        sample = self._two_stratum_sample(
            stratum_a_pop=1000,
            stratum_a_pos_pop=100,
            stratum_a_drawn=50,
            stratum_a_drawn_pos=10,
            stratum_b_pop=500,
            stratum_b_pos_pop=20,
            stratum_b_drawn=30,
            stratum_b_drawn_pos=5,
        )

        def is_pos(c: AuditCase) -> bool:
            return c.pred_classification == "INAPPROPRIATE"

        est = horvitz_thompson_prevalence(sample, is_pos)
        assert est.variance >= 0
        assert est.se == pytest.approx(est.variance**0.5, abs=1e-12)

    def test_variance_against_hand_computed_reference(self) -> None:
        # Codex P1: variance must be pinned to a known reference, not just
        # "≥ 0". Hand-derived under the canonical Sarndal SRS-with-
        # replacement approximation (eq 3.4.5):
        #   Var(p̂) = sum_h n_h * Var(w_i within h) / N²
        # Single stratum, pop=1000, pop_pos=100, drawn 10 pos + 10 neg.
        # pi_pos = 10/100 = 0.10  → 10 cases with w = 1/0.10 = 10.0
        # pi_neg = 10/900 ≈ 0.01111 → 10 cases with w = 0 (indicator=0)
        # mean(w) = 5.0; Var(w) sample = ((10*25)+(10*25))/19 = 500/19
        #                                = 26.31578947...
        # Stratum contribution: 20 * (500/19) / 1000² = 10000/19 / 1e6
        #                       = 526.31578947e-6
        positives = tuple(
            _case(audit_id=f"p{i}", pred="INAPPROPRIATE") for i in range(10)
        )
        negatives = tuple(
            _case(audit_id=f"n{i}", pred="APPROPRIATE") for i in range(10)
        )
        sample = StratifiedSample(
            draws=(
                StratumDraw(
                    stratum=Stratum.HB_LT_7,
                    cases=positives + negatives,
                    population_size=1000,
                    population_positives=100,
                    drawn_positives=10,
                    base_inclusion_probability=10 / 900,
                    positive_inclusion_probability=10 / 100,
                ),
            )
        )

        def is_pos(c: AuditCase) -> bool:
            return c.pred_classification == "INAPPROPRIATE"

        est = horvitz_thompson_prevalence(sample, is_pos)
        expected_var = (20 * (500 / 19)) / (1000.0 * 1000.0)
        assert est.variance == pytest.approx(expected_var, rel=1e-9)
        assert est.se == pytest.approx(expected_var**0.5, rel=1e-9)

    def test_drawn_case_with_zero_pi_raises(self) -> None:
        # Codex P0: a drawn case with non-positive inclusion probability
        # is a sampling-design bug — the harness must raise, not silently
        # skip and bias HT downward.
        sample = StratifiedSample(
            draws=(
                StratumDraw(
                    stratum=Stratum.HB_LT_7,
                    cases=(_case(audit_id="bad", pred="INAPPROPRIATE"),),
                    population_size=100,
                    population_positives=10,
                    drawn_positives=1,
                    base_inclusion_probability=0.1,
                    positive_inclusion_probability=0.0,  # invariant violation
                ),
            )
        )

        def is_pos(c: AuditCase) -> bool:
            return c.pred_classification == "INAPPROPRIATE"

        with pytest.raises(ValueError):
            horvitz_thompson_prevalence(sample, is_pos)

    def test_prevalence_bounded_in_unit_interval(self) -> None:
        # Even pathologically enriched samples should not produce a HT
        # prevalence > 1 — the reweighting must respect the natural ceiling.
        sample = self._two_stratum_sample(
            stratum_a_pop=200,
            stratum_a_pos_pop=50,
            stratum_a_drawn=50,
            stratum_a_drawn_pos=50,
            stratum_b_pop=200,
            stratum_b_pos_pop=50,
            stratum_b_drawn=50,
            stratum_b_drawn_pos=50,
        )

        def is_pos(c: AuditCase) -> bool:
            return c.pred_classification == "INAPPROPRIATE"

        est = horvitz_thompson_prevalence(sample, is_pos)
        assert 0.0 <= est.prevalence <= 1.0


# =============================================================================
# AC ④ Wilson CI against scipy reference.
#
# Reference values are scipy.stats.binomtest(...).proportion_ci(method='wilson')
# computed offline at scipy 1.13. Hardcoded here so the test does NOT take a
# runtime scipy dependency; the publication script may use scipy to double-
# check, but the harness never does.
# =============================================================================


class TestWilsonCI:
    """wilson_ci matches scipy's proportion_ci method='wilson' at 95%."""

    def test_balanced_midpoint(self) -> None:
        # Wilson CI for (50, 100, 0.95) derived from the canonical formula
        # with z = Phi^{-1}(0.975) at full precision. Matches scipy's
        # binomtest(50, 100).proportion_ci(method='wilson') to ~1e-7.
        ci = wilson_ci(50, 100, confidence=0.95)
        assert ci.point == pytest.approx(0.5, abs=1e-12)
        assert ci.lower == pytest.approx(0.40383153, abs=1e-6)
        assert ci.upper == pytest.approx(0.59616847, abs=1e-6)
        assert ci.confidence == 0.95

    def test_zero_successes(self) -> None:
        # Wilson CI for (0, 100, 0.95). Lower bound stays at 0 (boundary
        # behavior — the regulator-visible failure mode for the normal-
        # approximation interval is exactly this case).
        ci = wilson_ci(0, 100, confidence=0.95)
        assert ci.lower == pytest.approx(0.0, abs=1e-12)
        assert ci.upper == pytest.approx(0.03699350, abs=1e-6)

    def test_all_successes(self) -> None:
        # Symmetric boundary case: (100, 100, 0.95) → upper saturates at 1.
        ci = wilson_ci(100, 100, confidence=0.95)
        assert ci.lower == pytest.approx(0.96300650, abs=1e-6)
        assert ci.upper == pytest.approx(1.0, abs=1e-12)

    def test_small_sample(self) -> None:
        # Wilson CI for (2, 10, 0.95) — small-sample shape where Wilson and
        # the normal approximation diverge most noticeably.
        ci = wilson_ci(2, 10, confidence=0.95)
        assert ci.lower == pytest.approx(0.05668215, abs=1e-6)
        assert ci.upper == pytest.approx(0.50983753, abs=1e-6)

    def test_99pct_confidence_wider_than_95pct(self) -> None:
        # The confidence argument must be honored. 99% must be wider than 95%.
        ci_95 = wilson_ci(50, 100, confidence=0.95)
        ci_99 = wilson_ci(50, 100, confidence=0.99)
        assert (ci_99.upper - ci_99.lower) > (ci_95.upper - ci_95.lower)

    def test_zero_trials_raises(self) -> None:
        # No defined center for zero trials. EmptyInputError, not silent NaN.
        with pytest.raises(EmptyInputError):
            wilson_ci(0, 0)

    def test_successes_exceed_trials_raises(self) -> None:
        with pytest.raises(ValueError):
            wilson_ci(11, 10)

    def test_negative_successes_raises(self) -> None:
        with pytest.raises(ValueError):
            wilson_ci(-1, 10)


class TestNormalQuantile:
    """normal_quantile is the inverse-CDF of the standard normal."""

    def test_z_0975_matches_published(self) -> None:
        # z_{0.975} = 1.959963984540054... (the value implied by 95% Wilson)
        assert normal_quantile(0.975) == pytest.approx(1.959964, abs=1e-5)

    def test_z_05_is_zero(self) -> None:
        assert normal_quantile(0.5) == pytest.approx(0.0, abs=1e-9)

    def test_symmetric_around_half(self) -> None:
        # Standard normal is symmetric: Phi^{-1}(p) = -Phi^{-1}(1-p)
        for p in (0.05, 0.1, 0.25, 0.4):
            assert normal_quantile(p) == pytest.approx(
                -normal_quantile(1 - p), abs=1e-8
            )


# =============================================================================
# AC ⑤ Cohen's κ + Gwet's AC1 + PABAK.
#
# Reference values are computed from PUBLISHED worked examples. Sources
# documented inline so a future code reviewer can re-derive each number.
# =============================================================================


class TestCohenKappa:
    """cohen_kappa matches Cohen (1960) and edge cases."""

    def test_published_balanced_example(self) -> None:
        # Worked example: 50 items, two raters.
        # Confusion: A=Y/B=Y: 20, A=Y/B=N: 5, A=N/B=Y: 10, A=N/B=N: 15.
        # p_o = (20+15)/50 = 0.70
        # p_e = (25/50)*(30/50) + (25/50)*(20/50) = 0.50
        # κ = (0.70 - 0.50) / (1 - 0.50) = 0.40
        a = (["Y"] * 25) + (["N"] * 25)
        b = (["Y"] * 20 + ["N"] * 5) + (["Y"] * 10 + ["N"] * 15)
        assert cohen_kappa(a, b) == pytest.approx(0.40, abs=1e-9)

    def test_perfect_agreement_is_one(self) -> None:
        labels = ["A", "B", "C", "A", "B", "C", "A"]
        assert cohen_kappa(labels, labels) == pytest.approx(1.0, abs=1e-12)

    def test_complete_disagreement_binary_is_minus_one(self) -> None:
        # Balanced marginals + zero observed agreement: κ = -1.
        a = ["Y", "Y", "N", "N", "Y", "Y", "N", "N"]
        b = ["N", "N", "Y", "Y", "N", "N", "Y", "Y"]
        assert cohen_kappa(a, b) == pytest.approx(-1.0, abs=1e-9)

    def test_random_labels_kappa_near_zero(self) -> None:
        # Independent random raters with same marginals → κ ≈ 0.
        # Constructed so the two raters agree at exactly the chance rate.
        # A: YYNN, B: YNYN over 4 items → p_o = 0.5, p_e = 0.5 → κ = 0.
        a = ["Y", "Y", "N", "N"]
        b = ["Y", "N", "Y", "N"]
        assert cohen_kappa(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_unequal_length_raises_shape_mismatch(self) -> None:
        with pytest.raises(ShapeMismatchError):
            cohen_kappa(["Y", "N"], ["Y"])

    def test_empty_inputs_raise(self) -> None:
        with pytest.raises(EmptyInputError):
            cohen_kappa([], [])


class TestGwetAC1:
    """gwet_ac1 matches Gwet (2008) and the kappa-prevalence-paradox case."""

    def test_high_agreement_high_prevalence_example(self) -> None:
        # 50 items, binary: both Y 40, both N 5, A=Y/B=N 3, A=N/B=Y 2.
        # Hand-derivation in docstring:
        #   p_o = 45/50 = 0.90
        #   marginal pi_Y = (43/50 + 42/50)/2 = 0.85
        #   p_e_ac1 (binary) = 2*pi_Y*(1-pi_Y) = 2*0.85*0.15 = 0.255
        #   AC1 = (0.90 - 0.255)/(1 - 0.255) = 0.645/0.745 ≈ 0.86577
        a = (["Y"] * 40 + ["Y"] * 3 + ["N"] * 2 + ["N"] * 5)
        b = (["Y"] * 40 + ["N"] * 3 + ["Y"] * 2 + ["N"] * 5)
        assert gwet_ac1(a, b) == pytest.approx(0.645 / 0.745, abs=1e-9)

    def test_perfect_agreement_is_one(self) -> None:
        labels = ["Y"] * 10 + ["N"] * 10
        assert gwet_ac1(labels, labels) == pytest.approx(1.0, abs=1e-9)

    def test_unequal_length_raises_shape_mismatch(self) -> None:
        with pytest.raises(ShapeMismatchError):
            gwet_ac1(["Y", "N"], ["Y"])

    def test_empty_inputs_raise(self) -> None:
        with pytest.raises(EmptyInputError):
            gwet_ac1([], [])


class TestPABAK:
    """pabak matches the Byrt (1993) analytic formula."""

    def test_binary_formula(self) -> None:
        # Binary: PABAK = 2 * p_o - 1.
        # 10 items, 7 agreements → PABAK = 0.4. Built by-hand below:
        # a: Y Y Y Y Y N N N N N
        # b: Y Y N N Y N N Y N N  (agreements at indices 0,1,4,5,6,8,9 = 7)
        a = ["Y", "Y", "Y", "Y", "Y", "N", "N", "N", "N", "N"]
        b = ["Y", "Y", "N", "N", "Y", "N", "N", "Y", "N", "N"]
        assert sum(1 for x, y in zip(a, b, strict=True) if x == y) == 7
        assert pabak(a, b) == pytest.approx(2 * 0.7 - 1, abs=1e-9)

    def test_three_category(self) -> None:
        # k-category: PABAK = (k*p_o - 1)/(k-1).
        # 9 items over {A,B,C}; 6 agreements → p_o = 6/9 = 2/3.
        # PABAK = (3*(2/3) - 1)/2 = (2 - 1)/2 = 0.5
        a = ["A", "A", "B", "B", "C", "C", "A", "B", "C"]
        b = ["A", "A", "B", "B", "C", "C", "B", "C", "A"]  # 6 agreements
        assert pabak(a, b) == pytest.approx(0.5, abs=1e-9)

    def test_perfect_agreement_is_one(self) -> None:
        labels = ["Y"] * 5 + ["N"] * 5
        assert pabak(labels, labels) == pytest.approx(1.0, abs=1e-9)

    def test_complete_disagreement_binary_is_minus_one(self) -> None:
        a = ["Y"] * 5 + ["N"] * 5
        b = ["N"] * 5 + ["Y"] * 5
        assert pabak(a, b) == pytest.approx(-1.0, abs=1e-9)

    def test_unequal_length_raises_shape_mismatch(self) -> None:
        with pytest.raises(ShapeMismatchError):
            pabak(["Y"], ["Y", "N"])


class TestAgreementCombined:
    """agreement_with_metrics returns κ + AC1 + PABAK in one bundle."""

    def test_bundle_matches_individual_functions(self) -> None:
        # 50 items, the same fixture used in the published-example tests.
        a = (["Y"] * 40 + ["Y"] * 3 + ["N"] * 2 + ["N"] * 5)
        b = (["Y"] * 40 + ["N"] * 3 + ["Y"] * 2 + ["N"] * 5)
        bundle = agreement_with_metrics(a, b)
        assert isinstance(bundle, AgreementResult)
        assert bundle.cohen_kappa == pytest.approx(cohen_kappa(a, b), abs=1e-12)
        assert bundle.gwet_ac1 == pytest.approx(gwet_ac1(a, b), abs=1e-12)
        assert bundle.pabak == pytest.approx(pabak(a, b), abs=1e-12)
        assert bundle.n_items == 50
        assert bundle.observed_agreement == pytest.approx(0.90, abs=1e-9)


class TestKappaPrevalenceParadox:
    """High-prevalence stratum: AC1 > κ (the PRD §11 motivation for AC1)."""

    def test_high_prevalence_kappa_lower_than_ac1(self) -> None:
        # PRD §11 explicitly cites the Hb>10 majority-class stratum as the
        # case where κ deflates and AC1 / PABAK stay informative. Reproduce
        # the inequality with a published-style fixture.
        a = (["Y"] * 40 + ["Y"] * 3 + ["N"] * 2 + ["N"] * 5)
        b = (["Y"] * 40 + ["N"] * 3 + ["Y"] * 2 + ["N"] * 5)
        k = cohen_kappa(a, b)
        ac1 = gwet_ac1(a, b)
        pabak_v = pabak(a, b)
        # Both AC1 and PABAK should be materially above κ in this scenario.
        assert ac1 > k + 0.10
        assert pabak_v > k + 0.10


# =============================================================================
# AC ⑥ Cluster-robust SE on synthetic clustered data.
# =============================================================================


class TestClusterRobustSE:
    """cluster_robust_proportion_ci matches hand-computed CR0 estimator."""

    def test_perfect_intracluster_correlation(self) -> None:
        # 4 clusters x 5 obs = 20 obs total. Half clusters all-1, half all-0.
        # Hand-derivation:
        #   p̂ = 10/20 = 0.5
        #   naive SE = sqrt(p̂(1-p̂)/n) = sqrt(0.5*0.5/20) ≈ 0.111803
        #   For each cluster: U_g = sum_i (y_i - p̂) = 5 * (1-0.5) or 5*(-0.5)
        #                        = ±2.5; U_g² = 6.25; sum = 25
        #   CR0 variance = sum_g U_g² / n² = 25/400 = 0.0625; SE = 0.25.
        indicators = [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5
        cluster_ids = ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5
        est = cluster_robust_proportion_ci(
            indicators, cluster_ids, confidence=0.95
        )
        assert est.n_clusters == 4
        assert est.n_obs == 20
        assert est.point == pytest.approx(0.5, abs=1e-12)
        assert est.cluster_robust_se == pytest.approx(0.25, abs=1e-9)
        assert est.naive_se == pytest.approx(
            (0.5 * 0.5 / 20) ** 0.5, abs=1e-9
        )

    def test_singletons_collapse_to_naive(self) -> None:
        # When every observation is its own cluster, the cluster-robust SE
        # collapses to the binomial SE (the no-clustering baseline).
        indicators = [True, False, True, True, False, False, True, False]
        cluster_ids = [str(i) for i in range(len(indicators))]
        est = cluster_robust_proportion_ci(indicators, cluster_ids)
        assert est.cluster_robust_se == pytest.approx(est.naive_se, abs=1e-9)

    def test_ci_wider_than_point_with_clustering(self) -> None:
        # The Wald CI must include the point estimate and span > 0 when
        # there is observed variation.
        indicators = [True] * 5 + [False] * 5 + [True] * 5 + [False] * 5
        cluster_ids = ["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5
        est = cluster_robust_proportion_ci(indicators, cluster_ids)
        assert est.lower < est.point < est.upper

    def test_unequal_lengths_raise(self) -> None:
        with pytest.raises(ShapeMismatchError):
            cluster_robust_proportion_ci([True, False], ["A"])

    def test_empty_input_raises(self) -> None:
        with pytest.raises(EmptyInputError):
            cluster_robust_proportion_ci([], [])

    def test_single_cluster_raises(self) -> None:
        # Codex P0: CR0 with only 1 cluster yields a degenerate zero-width
        # CI (residuals cancel to exactly zero). The harness must refuse
        # rather than silently report overconfidence — the regulator-visible
        # failure mode of cluster-robust SE on under-clustered data.
        indicators = [True, False, True, False]
        cluster_ids = ["only-cluster"] * 4
        with pytest.raises(ValueError):
            cluster_robust_proportion_ci(indicators, cluster_ids)

    def test_cluster_robust_se_at_least_naive_under_positive_icc(self) -> None:
        # Positive intra-cluster correlation → cluster-robust SE ≥ naive SE.
        # Use 6 clusters with strongly correlated outcomes within cluster.
        indicators = (
            [True] * 4 + [False] * 4 + [True] * 4 + [False] * 4 + [True] * 4 + [False] * 4
        )
        cluster_ids = (
            ["c1"] * 4 + ["c2"] * 4 + ["c3"] * 4 + ["c4"] * 4 + ["c5"] * 4 + ["c6"] * 4
        )
        est = cluster_robust_proportion_ci(indicators, cluster_ids)
        assert est.cluster_robust_se >= est.naive_se


# =============================================================================
# AC ⑦ Temporal splits: LOMO-CV + blocked + auto-selection.
# =============================================================================


def _monthly_population(months: list[tuple[int, int]]) -> list[AuditCase]:
    """Build a population with N cases in each (year, month) in ``months``."""
    pop: list[AuditCase] = []
    for i, (y, m) in enumerate(months):
        for j in range(5):
            pop.append(
                _case(
                    audit_id=f"y{y}-m{m:02d}-{j}",
                    when=datetime(y, m, 1, 12, 0, 0, tzinfo=UTC),
                    physician=f"MD-{i}",
                )
            )
    return pop


class TestLOMOCV:
    """lomo_cv_splits: one holdout per calendar month."""

    def test_one_split_per_month(self) -> None:
        # 6 months → 6 splits.
        cases = _monthly_population(
            [(2026, m) for m in range(1, 7)]
        )
        splits = lomo_cv_splits(cases)
        assert len(splits) == 6

    def test_holdout_is_one_month(self) -> None:
        cases = _monthly_population([(2026, 1), (2026, 2), (2026, 3)])
        splits = lomo_cv_splits(cases)
        # Each split's holdout audit_ids must all belong to one month.
        for split in splits:
            months_in_holdout = {
                aid.split("-m")[1].split("-")[0] for aid in split.holdout_audit_ids
            }
            assert len(months_in_holdout) == 1

    def test_train_excludes_holdout(self) -> None:
        cases = _monthly_population([(2026, 1), (2026, 2), (2026, 3)])
        splits = lomo_cv_splits(cases)
        for split in splits:
            assert (
                set(split.train_audit_ids) & set(split.holdout_audit_ids)
            ) == set()
            assert (
                len(split.train_audit_ids) + len(split.holdout_audit_ids)
                == len(cases)
            )

    def test_every_case_appears_in_some_holdout(self) -> None:
        cases = _monthly_population([(2026, 1), (2026, 2), (2026, 3)])
        splits = lomo_cv_splits(cases)
        held_out = {
            aid for split in splits for aid in split.holdout_audit_ids
        }
        assert held_out == {c.audit_id for c in cases}

    def test_holdout_label_is_month_tag(self) -> None:
        # Use a fixed format that the report writer can render directly.
        cases = _monthly_population([(2026, 1), (2026, 2)])
        labels = sorted(s.holdout_label for s in lomo_cv_splits(cases))
        assert labels == ["2026-01", "2026-02"]


class TestBlockedTemporalSplit:
    """blocked_temporal_split: contiguous chronological blocks."""

    def test_block_count(self) -> None:
        cases = _monthly_population([(2026, m) for m in range(1, 13)])
        splits = blocked_temporal_split(cases, n_blocks=4)
        assert len(splits) == 4

    def test_disjoint_holdouts(self) -> None:
        cases = _monthly_population([(2026, m) for m in range(1, 13)])
        splits = blocked_temporal_split(cases, n_blocks=4)
        seen: set[str] = set()
        for split in splits:
            for aid in split.holdout_audit_ids:
                assert aid not in seen, "holdouts must be disjoint"
                seen.add(aid)
        assert seen == {c.audit_id for c in cases}

    def test_blocks_are_chronological(self) -> None:
        # Block i's holdout dates must be ≤ block i+1's holdout dates.
        cases = _monthly_population([(2026, m) for m in range(1, 13)])
        splits = blocked_temporal_split(cases, n_blocks=4)
        by_id = {c.audit_id: c.order_datetime for c in cases}
        max_per_block = [
            max(by_id[aid] for aid in s.holdout_audit_ids) for s in splits
        ]
        assert max_per_block == sorted(max_per_block)


class TestSplitStrategyAutoSelect:
    """select_split_strategy picks LOMO if < 12 months, blocked otherwise."""

    def test_three_months_picks_lomo(self) -> None:
        cases = _monthly_population([(2026, 1), (2026, 2), (2026, 3)])
        assert select_split_strategy(cases) == "lomo"

    def test_twelve_months_picks_blocked(self) -> None:
        # Boundary: exactly 12 months → blocked (the threshold is >=, not >,
        # so the design choice is one-sided and report-stable).
        cases = _monthly_population([(2026, m) for m in range(1, 13)])
        assert select_split_strategy(cases) == "blocked"

    def test_eleven_months_picks_lomo(self) -> None:
        cases = _monthly_population([(2026, m) for m in range(1, 12)])
        assert select_split_strategy(cases) == "lomo"

    def test_threshold_constant_drives_decision(self) -> None:
        # The threshold itself is exposed for downstream auditing; the
        # behavior must follow its value, not a hardcoded literal here.
        assert LOMO_DATASET_THRESHOLD_MONTHS == 12

    def test_sparse_calendar_span_uses_range_not_density(self) -> None:
        # Codex P0: span is calendar range, not distinct-month count. A
        # dataset with only Jan + Dec of the same year SPANS 12 months and
        # must auto-select blocked, even though only 2 months have data.
        cases = _monthly_population([(2026, 1), (2026, 12)])
        assert dataset_month_span(cases) == 12
        assert select_split_strategy(cases) == "blocked"


class TestTemporalCVSplits:
    """temporal_cv_splits: auto dispatch by strategy."""

    def test_short_dataset_dispatches_to_lomo(self) -> None:
        cases = _monthly_population([(2026, 1), (2026, 2)])
        auto = temporal_cv_splits(cases)
        manual = lomo_cv_splits(cases)
        assert len(auto) == len(manual)
        assert sorted(s.holdout_label for s in auto) == sorted(
            s.holdout_label for s in manual
        )

    def test_long_dataset_dispatches_to_blocked(self) -> None:
        cases = _monthly_population([(2026, m) for m in range(1, 13)])
        auto = temporal_cv_splits(cases)
        assert len(auto) == 4  # default n_blocks


class TestDatasetMonthSpan:
    """dataset_month_span is the number of distinct calendar months."""

    def test_span_of_three_months(self) -> None:
        cases = _monthly_population([(2026, 1), (2026, 2), (2026, 3)])
        assert dataset_month_span(cases) == 3

    def test_span_ignores_within_month_volume(self) -> None:
        # 100 cases all in Jan 2026 still spans 1 month.
        cases = [
            _case(
                audit_id=f"x-{i}",
                when=datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC),
            )
            for i in range(100)
        ]
        assert dataset_month_span(cases) == 1

    def test_empty_raises_empty_input(self) -> None:
        with pytest.raises(EmptyInputError):
            dataset_month_span([])


# =============================================================================
# AC ⑧ Multiple-comparisons correction: Bonferroni + BH-FDR hierarchical.
# =============================================================================


class TestBonferroniCorrection:
    """bonferroni_correction multiplies by k and caps at 1."""

    def test_known_three_test_example(self) -> None:
        # k=3, raw=[0.01, 0.02, 0.10]. Adjusted = [0.03, 0.06, 0.30].
        # Reject at α=0.05: only the first.
        out = bonferroni_correction(
            {"a": 0.01, "b": 0.02, "c": 0.10}, alpha=0.05
        )
        by_name = {t.name: t for t in out}
        assert by_name["a"].adjusted_p == pytest.approx(0.03, abs=1e-12)
        assert by_name["b"].adjusted_p == pytest.approx(0.06, abs=1e-12)
        assert by_name["c"].adjusted_p == pytest.approx(0.30, abs=1e-12)
        assert by_name["a"].rejected is True
        assert by_name["b"].rejected is False
        assert by_name["c"].rejected is False

    def test_adjusted_capped_at_one(self) -> None:
        # Large raw p × large k → cap at 1.0 (not 1.5).
        out = bonferroni_correction({"a": 0.5, "b": 0.5, "c": 0.5})
        assert all(t.adjusted_p == pytest.approx(1.0, abs=1e-12) for t in out)

    def test_empty_input_returns_empty_tuple(self) -> None:
        assert bonferroni_correction({}) == ()

    def test_preserves_input_order(self) -> None:
        # Dict iteration order is insertion order in Python 3.7+; the report
        # writer relies on it.
        names = tuple(t.name for t in bonferroni_correction(
            {"c": 0.01, "a": 0.02, "b": 0.03}
        ))
        assert names == ("c", "a", "b")


class TestBHFDRCorrection:
    """benjamini_hochberg_correction matches the BH (1995) worked example."""

    def test_known_ten_test_example(self) -> None:
        # BH (1995) worked example. Sorted p:
        # [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216].
        # At alpha=0.05, reject the first 2 (q-values 0.01 and 0.04).
        raw = {
            "h01": 0.001,
            "h02": 0.008,
            "h03": 0.039,
            "h04": 0.041,
            "h05": 0.042,
            "h06": 0.060,
            "h07": 0.074,
            "h08": 0.205,
            "h09": 0.212,
            "h10": 0.216,
        }
        out = benjamini_hochberg_correction(raw, alpha=0.05)
        by_name = {t.name: t for t in out}
        # q_(1) = 0.001 * 10/1 = 0.01
        assert by_name["h01"].adjusted_p == pytest.approx(0.01, abs=1e-9)
        # q_(2) = 0.008 * 10/2 = 0.04
        assert by_name["h02"].adjusted_p == pytest.approx(0.04, abs=1e-9)
        # q_(3) = min over j>=3 of m*p_(j)/j = min(0.13, 0.1025, 0.084, ...) = 0.084
        assert by_name["h03"].adjusted_p == pytest.approx(0.084, abs=1e-9)
        assert by_name["h01"].rejected is True
        assert by_name["h02"].rejected is True
        assert by_name["h03"].rejected is False

    def test_bh_more_powerful_than_bonferroni(self) -> None:
        # BH is more powerful: with the BH worked example, BH rejects 2
        # while Bonferroni rejects only 1 at α=0.05 (0.01*10 = 0.1 > 0.05
        # for the second hypothesis).
        raw = {
            "h01": 0.001,
            "h02": 0.008,
            "h03": 0.039,
            "h04": 0.041,
            "h05": 0.042,
            "h06": 0.060,
            "h07": 0.074,
            "h08": 0.205,
            "h09": 0.212,
            "h10": 0.216,
        }
        bh = benjamini_hochberg_correction(raw, alpha=0.05)
        bf = bonferroni_correction(raw, alpha=0.05)
        assert sum(t.rejected for t in bh) > sum(t.rejected for t in bf)

    def test_empty_input_returns_empty_tuple(self) -> None:
        assert benjamini_hochberg_correction({}) == ()

    def test_adjusted_capped_at_one(self) -> None:
        out = benjamini_hochberg_correction(
            {"a": 0.5, "b": 0.6, "c": 0.7, "d": 0.9}
        )
        assert all(0 <= t.adjusted_p <= 1.0 for t in out)


class TestHierarchicalCorrection:
    """hierarchical_correction: Bonferroni on primary, BH on exploratory."""

    def test_two_families_corrected_independently(self) -> None:
        primary = {"p1": 0.01, "p2": 0.04}  # Bonferroni: 0.02, 0.08
        exploratory = {
            "e1": 0.001,
            "e2": 0.008,
            "e3": 0.039,
            "e4": 0.041,
            "e5": 0.042,
            "e6": 0.060,
            "e7": 0.074,
            "e8": 0.205,
            "e9": 0.212,
            "e10": 0.216,
        }
        result = hierarchical_correction(primary, exploratory, alpha=0.05)
        primary_by_name = {t.name: t for t in result.primary}
        explor_by_name = {t.name: t for t in result.exploratory}
        # Primary: Bonferroni multiplies by k=2.
        assert primary_by_name["p1"].adjusted_p == pytest.approx(
            0.02, abs=1e-12
        )
        assert primary_by_name["p2"].adjusted_p == pytest.approx(
            0.08, abs=1e-12
        )
        assert primary_by_name["p1"].rejected is True
        assert primary_by_name["p2"].rejected is False
        # Exploratory: BH against k=10 — same as TestBHFDRCorrection.
        assert explor_by_name["e1"].adjusted_p == pytest.approx(
            0.01, abs=1e-9
        )
        assert explor_by_name["e2"].adjusted_p == pytest.approx(
            0.04, abs=1e-9
        )

    def test_families_do_not_pool(self) -> None:
        # If the families pooled, the primary family's "k" would be 12,
        # producing adjusted_p = 0.12 on p1 (not 0.02). Test that they don't.
        primary = {"p1": 0.01}
        exploratory = {f"e{i}": 0.5 for i in range(1, 12)}  # 11 exploratory
        result = hierarchical_correction(primary, exploratory)
        assert result.primary[0].adjusted_p == pytest.approx(0.01, abs=1e-12)


# =============================================================================
# Verifier-as-classifier confusion-matrix consumer (PRD §11).
# =============================================================================


class TestVerifierAsClassifier:
    """evaluate_confusion_matrix turns a 2x2 cm into metrics with Wilson CIs."""

    def test_perfect_classifier(self) -> None:
        cm = ConfusionMatrix(
            true_positive=100, true_negative=100, false_positive=0, false_negative=0
        )
        metrics = evaluate_confusion_matrix(cm)
        assert metrics.sensitivity.point == pytest.approx(1.0, abs=1e-12)
        assert metrics.specificity.point == pytest.approx(1.0, abs=1e-12)
        assert metrics.ppv.point == pytest.approx(1.0, abs=1e-12)
        assert metrics.npv.point == pytest.approx(1.0, abs=1e-12)
        assert metrics.f1 == pytest.approx(1.0, abs=1e-12)

    def test_balanced_classifier_against_known_metrics(self) -> None:
        # TP=80, FN=20, FP=10, TN=90.
        # sensitivity = 80/100 = 0.80; specificity = 90/100 = 0.90
        # ppv = 80/90 = 0.8889; npv = 90/110 = 0.8182
        # accuracy = (80+90)/200 = 0.85
        # F1 = 2*0.8889*0.80 / (0.8889+0.80) = 1.4222/1.6889 = 0.8421
        cm = ConfusionMatrix(
            true_positive=80, true_negative=90, false_positive=10, false_negative=20
        )
        m = evaluate_confusion_matrix(cm)
        assert m.sensitivity.point == pytest.approx(0.80, abs=1e-9)
        assert m.specificity.point == pytest.approx(0.90, abs=1e-9)
        assert m.ppv.point == pytest.approx(80 / 90, abs=1e-9)
        assert m.npv.point == pytest.approx(90 / 110, abs=1e-9)
        assert m.accuracy.point == pytest.approx(170 / 200, abs=1e-9)
        assert m.f1 == pytest.approx(
            2 * (80 / 90) * 0.80 / ((80 / 90) + 0.80), abs=1e-9
        )

    def test_all_zero_matrix_no_zero_division(self) -> None:
        # Empty 2x2 must not raise — every rate is 0 with degenerate CI.
        cm = ConfusionMatrix(
            true_positive=0, true_negative=0, false_positive=0, false_negative=0
        )
        m = evaluate_confusion_matrix(cm)
        assert m.f1 == 0.0
        for rate in (m.sensitivity, m.specificity, m.ppv, m.npv, m.accuracy):
            assert rate.point == 0.0
            assert rate.lower == 0.0
            assert rate.upper == 0.0

    def test_no_positives_collapses_sensitivity_and_ppv(self) -> None:
        # TP+FN = 0 → sensitivity collapsed; TP+FP = 0 → ppv collapsed.
        cm = ConfusionMatrix(
            true_positive=0, true_negative=50, false_positive=0, false_negative=0
        )
        m = evaluate_confusion_matrix(cm)
        assert m.sensitivity.point == 0.0
        assert m.ppv.point == 0.0
        # Specificity remains computable: 50/50 = 1.0.
        assert m.specificity.point == pytest.approx(1.0, abs=1e-12)


# =============================================================================
# Outcome-anchored falsification on labeled subset (PRD §11).
# =============================================================================


class TestOutcomeAnchoredFalsification:
    """outcome_anchored_falsification computes support / contradiction rates."""

    def test_clear_support_case(self) -> None:
        # 10 cases: all INAPPROPRIATE predictions; 8 supported (no further
        # transfusion), 2 contradicted (further transfusion).
        preds = ["INAPPROPRIATE"] * 10
        outs = (
            [FalsificationOutcome.NO_FURTHER_TRANSFUSION] * 8
            + [FalsificationOutcome.FURTHER_TRANSFUSION_24H] * 2
        )
        result = outcome_anchored_falsification(preds, outs)
        assert result.n_labeled == 10
        assert result.n_inappropriate_pred == 10
        assert result.n_supported == 8
        assert result.n_contradicted == 2
        assert result.support_rate.point == pytest.approx(0.8, abs=1e-12)
        assert result.contradiction_rate.point == pytest.approx(0.2, abs=1e-12)

    def test_non_inappropriate_predictions_ignored(self) -> None:
        # APPROPRIATE predictions are NOT falsification targets — the analysis
        # only grades the pipeline's positive (INAPPROPRIATE) calls.
        preds = ["APPROPRIATE", "INAPPROPRIATE", "APPROPRIATE", "INAPPROPRIATE"]
        outs = [
            FalsificationOutcome.FURTHER_TRANSFUSION_24H,
            FalsificationOutcome.NO_FURTHER_TRANSFUSION,
            FalsificationOutcome.DEATH_FROM_ANEMIA_30D,
            FalsificationOutcome.NO_FURTHER_TRANSFUSION,
        ]
        result = outcome_anchored_falsification(preds, outs)
        # Only 2 INAPPROPRIATE predictions count; both supported.
        assert result.n_inappropriate_pred == 2
        assert result.n_supported == 2
        assert result.n_contradicted == 0

    def test_death_counted_as_contradiction(self) -> None:
        preds = ["INAPPROPRIATE"]
        outs = [FalsificationOutcome.DEATH_FROM_ANEMIA_30D]
        result = outcome_anchored_falsification(preds, outs)
        assert result.n_contradicted == 1
        assert result.n_supported == 0

    def test_unequal_lengths_raise(self) -> None:
        with pytest.raises(ShapeMismatchError):
            outcome_anchored_falsification(
                ["INAPPROPRIATE"], []
            )

    def test_empty_input_raises(self) -> None:
        with pytest.raises(EmptyInputError):
            outcome_anchored_falsification([], [])

    def test_zero_inappropriate_predictions_collapses_rates(self) -> None:
        # No positive calls → rates collapse to [0,0] but the analysis does
        # not crash. Matches the classifier.evaluate_confusion_matrix policy.
        preds = ["APPROPRIATE"] * 5
        outs = [FalsificationOutcome.NO_FURTHER_TRANSFUSION] * 5
        result = outcome_anchored_falsification(preds, outs)
        assert result.n_inappropriate_pred == 0
        assert result.support_rate.point == 0.0
        assert result.contradiction_rate.point == 0.0


# =============================================================================
# Hypothesis property tests — "deep" module requirement (PRD §"Testing
# Decisions": "Property-based tests where the property is invariant").
# =============================================================================


class TestPropertyWilsonContainsPoint:
    """For any (successes, trials, confidence), Wilson L ≤ point ≤ Wilson U."""

    @given(
        successes=st.integers(min_value=0, max_value=500),
        trials_offset=st.integers(min_value=0, max_value=500),
        confidence=st.sampled_from([0.80, 0.90, 0.95, 0.99]),
    )
    @settings(max_examples=50, deadline=None)
    def test_wilson_interval_contains_point(
        self, successes: int, trials_offset: int, confidence: float
    ) -> None:
        trials = successes + trials_offset + 1  # trials ≥ 1, successes ≤ trials
        ci = wilson_ci(successes, trials, confidence=confidence)
        assert ci.lower <= ci.point <= ci.upper

    @given(
        successes=st.integers(min_value=0, max_value=200),
        trials_offset=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_wilson_bounds_in_unit_interval(
        self, successes: int, trials_offset: int
    ) -> None:
        trials = successes + trials_offset + 1
        ci = wilson_ci(successes, trials)
        assert 0.0 <= ci.lower <= ci.upper <= 1.0


class TestPropertyBonferroniMonotonic:
    """Bonferroni adjusted_p is monotonic non-decreasing in family size."""

    @given(
        p=st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        small_k=st.integers(min_value=1, max_value=10),
        extra=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=30, deadline=None)
    def test_more_tests_higher_or_equal_adjusted(
        self, p: float, small_k: int, extra: int
    ) -> None:
        # Adding tests can only raise the adjusted p (or leave it capped).
        small = {f"t{i}": (p if i == 0 else 0.5) for i in range(small_k)}
        large = {f"t{i}": (p if i == 0 else 0.5) for i in range(small_k + extra)}
        small_out = {t.name: t for t in bonferroni_correction(small)}
        large_out = {t.name: t for t in bonferroni_correction(large)}
        assert (
            large_out["t0"].adjusted_p >= small_out["t0"].adjusted_p - 1e-12
        )


class TestPropertyHorvitzThompsonUniform:
    """Under uniform inclusion probability, HT collapses to the sample mean."""

    @given(
        pop_size=st.integers(min_value=20, max_value=200),
        positive_count=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=20, deadline=None)
    def test_ht_equals_sample_mean_under_uniform_inclusion(
        self, pop_size: int, positive_count: int
    ) -> None:
        # When all inclusion probs are equal, the HT estimator reduces to the
        # ordinary sample proportion times (pop/sample) / pop = sample
        # proportion. Construct that case exactly.
        positive_count = min(positive_count, pop_size // 2)
        # Draw n = pop_size / 4 from a single stratum, with no enrichment.
        drawn = pop_size // 4
        drawn_pos = positive_count // 4 if positive_count >= 4 else 0
        # Skip degenerate (no positives drawn → trivially zero)
        if drawn == 0:
            return
        cases = (
            tuple(
                _case(audit_id=f"p{i}", pred="INAPPROPRIATE")
                for i in range(drawn_pos)
            )
            + tuple(
                _case(audit_id=f"n{i}", pred="APPROPRIATE")
                for i in range(drawn - drawn_pos)
            )
        )
        # uniform inclusion: base prob == positive prob == drawn/pop_size
        uniform_p = drawn / pop_size
        sample = StratifiedSample(
            draws=(
                StratumDraw(
                    stratum=Stratum.HB_7_10,
                    cases=cases,
                    population_size=pop_size,
                    population_positives=max(1, positive_count),
                    drawn_positives=drawn_pos,
                    base_inclusion_probability=uniform_p,
                    positive_inclusion_probability=uniform_p,
                ),
            )
        )

        def is_pos(c: AuditCase) -> bool:
            return c.pred_classification == "INAPPROPRIATE"

        est = horvitz_thompson_prevalence(sample, is_pos)
        sample_prop = drawn_pos / drawn
        # HT collapses to sample_prop under uniform pi.
        assert est.prevalence == pytest.approx(sample_prop, abs=1e-9)


# =============================================================================
# Pure-function / determinism contract.
#
# The harness is a graded measurement layer; logging or printing to stdout
# during measurement is a contract violation (a future operator running the
# pipeline silently would otherwise see "evaluation noise" land in their
# audit log).
# =============================================================================


class TestPureFunctionContract:
    """No stdout/stderr writes; deterministic outputs given fixed inputs."""

    def test_no_stdout_or_stderr_writes(self) -> None:
        buf_out, buf_err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = buf_out, buf_err
            wilson_ci(50, 100)
            cohen_kappa(["Y", "N"], ["Y", "N"])
            gwet_ac1(["Y", "N"], ["Y", "N"])
            pabak(["Y", "N"], ["Y", "N"])
            bonferroni_correction({"a": 0.01, "b": 0.5})
            benjamini_hochberg_correction({"a": 0.01, "b": 0.5})
            cluster_robust_proportion_ci([True, False], ["a", "b"])
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        assert buf_out.getvalue() == ""
        assert buf_err.getvalue() == ""

    def test_determinism_same_input_same_output(self) -> None:
        a = wilson_ci(50, 100)
        b = wilson_ci(50, 100)
        assert a == b


# =============================================================================
# Model-level immutability — pydantic frozen=True is part of the contract.
#
# Models flow into the eval-harness report writer (#28) verbatim; mutating a
# field after construction would corrupt the persisted report.
# =============================================================================


class TestModelImmutability:
    """All public eval-harness models are frozen and reject post-hoc mutation."""

    def test_wilson_interval_is_frozen(self) -> None:
        ci = WilsonInterval(point=0.5, lower=0.4, upper=0.6, confidence=0.95)
        with pytest.raises(ValidationError):
            ci.point = 0.99  # type: ignore[misc]

    def test_temporal_split_is_frozen(self) -> None:
        split = TemporalSplit(
            train_audit_ids=("a", "b"),
            holdout_audit_ids=("c",),
            holdout_label="2026-01",
        )
        with pytest.raises(ValidationError):
            split.holdout_label = "2026-02"  # type: ignore[misc]

    def test_corrected_test_is_frozen(self) -> None:
        t = CorrectedTest(
            name="h1", raw_p=0.01, adjusted_p=0.03, rejected=True
        )
        with pytest.raises(ValidationError):
            t.rejected = False  # type: ignore[misc]

    def test_classifier_metrics_is_frozen(self) -> None:
        rate = WilsonInterval(point=1.0, lower=1.0, upper=1.0, confidence=0.95)
        m = ClassifierMetrics(
            accuracy=rate,
            sensitivity=rate,
            specificity=rate,
            ppv=rate,
            npv=rate,
            f1=1.0,
        )
        with pytest.raises(ValidationError):
            m.f1 = 0.0  # type: ignore[misc]

    def test_falsification_result_is_frozen(self) -> None:
        rate = WilsonInterval(point=0.8, lower=0.6, upper=0.9, confidence=0.95)
        r = FalsificationResult(
            n_labeled=10,
            n_inappropriate_pred=10,
            n_supported=8,
            n_contradicted=2,
            support_rate=rate,
            contradiction_rate=rate,
        )
        with pytest.raises(ValidationError):
            r.n_supported = 5  # type: ignore[misc]

    def test_stratum_draw_is_frozen(self) -> None:
        draw = StratumDraw(
            stratum=Stratum.HB_7_10,
            cases=(),
            population_size=100,
            population_positives=10,
            drawn_positives=5,
            base_inclusion_probability=0.5,
            positive_inclusion_probability=0.5,
        )
        with pytest.raises(ValidationError):
            draw.population_size = 200  # type: ignore[misc]

    def test_hierarchical_correction_result_is_frozen(self) -> None:
        r = HierarchicalCorrectionResult(
            primary=(),
            exploratory=(),
            alpha=0.05,
        )
        with pytest.raises(ValidationError):
            r.alpha = 0.01  # type: ignore[misc]


# =============================================================================
# Re-export surface — collection-time guard against missing public symbols.
#
# Re-exports tested implicitly by the top-of-file imports; collecting this
# file at all means every name above resolves. The class below exists so a
# regression that drops a public symbol fails with a clear test name (rather
# than collection-only ImportError that hides under -q output).
# =============================================================================


class TestPublicSurface:
    """Public surface includes every symbol the report writer (#28) reads."""

    def test_split_strategy_literal_values(self) -> None:
        # SplitStrategy is a literal alias — exercise its two values at the
        # boundary so type-only drift surfaces here, not in mypy alone.
        valid: SplitStrategy
        valid = "lomo"
        assert valid in ("lomo", "blocked")
        valid = "blocked"
        assert valid in ("lomo", "blocked")

    def test_horvitz_thompson_estimate_constructor(self) -> None:
        # The model must accept the documented fields.
        est = HorvitzThompsonEstimate(prevalence=0.1, variance=0.001, se=0.0316)
        assert est.prevalence == 0.1

    def test_sampling_targets_uses_mapping(self) -> None:
        # SamplingTargets accepts a Mapping[Stratum, int] — exercise it.
        targets = SamplingTargets(
            per_stratum={Stratum.HB_LT_7: 140},
            inappropriate_enrichment_per_stratum=138,
        )
        assert targets.per_stratum[Stratum.HB_LT_7] == 140
