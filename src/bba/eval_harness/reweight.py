"""Horvitz-Thompson prevalence reweighting (PRD §11, User Story #28).

Population-level prevalence cannot be read off a stratified-with-enrichment
sample directly: the enrichment step makes the sample non-representative on
purpose. The HT estimator weights each sampled case by the inverse of its
inclusion probability, summed across strata, divided by the known population
size. The variance estimator is the standard HT formula for independent
strata under with-replacement sampling — a slight over-estimate vs the
without-replacement truth, conservative for the publication report.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from bba.eval_harness.models import (
    AuditCase,
    HorvitzThompsonEstimate,
    StratifiedSample,
)


def _case_inclusion_probability(
    case: AuditCase, *, base_pi: float, positive_pi: float
) -> float:
    return positive_pi if case.pred_classification == "INAPPROPRIATE" else base_pi


def horvitz_thompson_prevalence(
    sample: StratifiedSample,
    indicator: Callable[[AuditCase], bool],
) -> HorvitzThompsonEstimate:
    """HT estimate of population prevalence under ``indicator``."""
    total_population = sum(d.population_size for d in sample.draws)
    if total_population == 0:
        return HorvitzThompsonEstimate(prevalence=0.0, variance=0.0, se=0.0)

    weighted_sum = 0.0
    variance_sum = 0.0
    for draw in sample.draws:
        # Per-stratum HT contribution and variance term. Treat the stratum
        # as an independent SRS for the variance estimator (Sarndal et al.
        # 1992, eq 3.4.5); inflates SE slightly vs without-replacement.
        stratum_weighted = 0.0
        stratum_weighted_sq = 0.0
        for case in draw.cases:
            pi = _case_inclusion_probability(
                case,
                base_pi=draw.base_inclusion_probability,
                positive_pi=draw.positive_inclusion_probability,
            )
            if pi <= 0.0:
                # A case with zero inclusion prob should not appear in the
                # sample; if it does (e.g., a population with zero positives
                # somehow yielded a positive case), skip contribution rather
                # than blow up — the report's other strata still apply.
                continue
            y = 1.0 if indicator(case) else 0.0
            weight = y / pi
            stratum_weighted += weight
            stratum_weighted_sq += weight * weight
        weighted_sum += stratum_weighted
        # Stratum variance contribution: sum_i (1 - pi_i) * (y_i/pi_i)²
        # simplified under the SRS-with-replacement approximation to:
        #   N_h² * Var(weighted_sum_within_stratum) / n_h
        # Use the canonical sample variance of the within-stratum HT weights.
        n_h = len(draw.cases)
        if n_h > 1:
            mean_w = stratum_weighted / n_h
            sq_dev = sum(
                (
                    (
                        (1.0 if indicator(c) else 0.0)
                        / _case_inclusion_probability(
                            c,
                            base_pi=draw.base_inclusion_probability,
                            positive_pi=draw.positive_inclusion_probability,
                        )
                    )
                    - mean_w
                )
                ** 2
                for c in draw.cases
                if _case_inclusion_probability(
                    c,
                    base_pi=draw.base_inclusion_probability,
                    positive_pi=draw.positive_inclusion_probability,
                )
                > 0.0
            )
            var_w = sq_dev / (n_h - 1)
            variance_sum += (n_h * var_w) / (total_population * total_population)

    prevalence = weighted_sum / total_population
    # Bound the point estimate in [0, 1] — degenerate samples could push it
    # slightly past 1 under finite-precision arithmetic.
    prevalence = max(0.0, min(1.0, prevalence))
    variance = max(0.0, variance_sum)
    se = math.sqrt(variance)
    return HorvitzThompsonEstimate(
        prevalence=prevalence, variance=variance, se=se
    )
