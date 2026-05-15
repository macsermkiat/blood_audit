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
                # A drawn case cannot have zero inclusion probability — the
                # sampling design is contradicting itself. Silent skip would
                # bias HT downward without an audit trail (codex P0).
                raise ValueError(
                    f"horvitz_thompson_prevalence: case {case.audit_id} in "
                    f"stratum {draw.stratum} has inclusion probability "
                    f"{pi!r} — non-positive pi on a drawn case is a "
                    f"sampling-design bug"
                )
            y = 1.0 if indicator(case) else 0.0
            weight = y / pi
            stratum_weighted += weight
            stratum_weighted_sq += weight * weight
        weighted_sum += stratum_weighted
        # Stratum variance contribution: sum_i (1 - pi_i) * (y_i/pi_i)²
        # under the SRS-with-replacement approximation (Sarndal et al. 1992
        # eq 3.4.5) simplifies to N_h * Var(w_i) / N², where Var(w_i) is the
        # sample variance of the within-stratum HT weights. The first-pass
        # raise above guarantees every drawn case has pi > 0, so the
        # variance pass can use stratum_weighted / stratum_weighted_sq
        # directly without a per-case pi guard.
        n_h = len(draw.cases)
        if n_h > 1:
            mean_w = stratum_weighted / n_h
            # Sample variance via the algebraic identity:
            #   sum((w_i - mean)²) = sum(w_i²) - n * mean²
            sq_dev = stratum_weighted_sq - n_h * mean_w * mean_w
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
