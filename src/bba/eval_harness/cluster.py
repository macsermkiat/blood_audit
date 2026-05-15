"""Cluster-robust standard errors (PRD §11, User Story #31).

Audited orders are not i.i.d.: a single attending physician produces many
orders in a given month, and a single ward concentrates patient
populations. The naive binomial SE understates uncertainty by roughly
1.5-2x in this design (PRD §"Anti-hallucination is layered, not single-
shot."), which would inflate the per-stratum sensitivity claims that the
publication report cites.

This module implements the CR0 cluster-robust sandwich variance estimator
for a binomial proportion. ``cluster_id`` is typically a concatenation of
``physician_id`` and ``ward`` — the harness treats their cross as the
effective cluster (the same physician on two wards is not the same cluster).

Reference (manual): Liang & Zeger (1986); Cameron & Miller (2015) survey.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from bba.eval_harness.exceptions import EmptyInputError, ShapeMismatchError
from bba.eval_harness.intervals import normal_quantile
from bba.eval_harness.models import ClusterRobustEstimate


def cluster_robust_proportion_ci(
    indicators: Sequence[bool],
    cluster_ids: Sequence[str],
    *,
    confidence: float = 0.95,
) -> ClusterRobustEstimate:
    """Cluster-robust Wald CI for a proportion."""
    if len(indicators) != len(cluster_ids):
        raise ShapeMismatchError(
            f"cluster_robust_proportion_ci: indicators ({len(indicators)}) and "
            f"cluster_ids ({len(cluster_ids)}) disagree"
        )
    if not indicators:
        raise EmptyInputError(
            "cluster_robust_proportion_ci: indicators must be non-empty"
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError(
            f"cluster_robust_proportion_ci: confidence must be in (0,1), got {confidence!r}"
        )

    n = len(indicators)
    successes = sum(1 for x in indicators if x)
    p_hat = successes / n

    # CR0 sandwich for the mean of a binary indicator: Var(p̂) =
    # (1/n²) * sum_g (sum_{i in g} (y_i - p̂))². For singletons, the cluster
    # sum collapses to (y_i - p̂); summing across i collapses to n*p̂(1-p̂),
    # which gives the naive binomial variance — the property the tests
    # assert structurally.
    cluster_sums: dict[str, float] = defaultdict(float)
    for ind, cid in zip(indicators, cluster_ids, strict=True):
        cluster_sums[cid] += (1.0 if ind else 0.0) - p_hat
    if len(cluster_sums) < 2:
        # CR0 with a single cluster yields zero variance and a degenerate
        # zero-width CI — silently overconfident (codex P0). Surface the
        # design violation so the caller can choose between collecting
        # multi-cluster data or falling back to a documented naive SE.
        raise ValueError(
            "cluster_robust_proportion_ci: at least 2 clusters are required; "
            f"got {len(cluster_sums)}"
        )
    cr_variance = sum(u * u for u in cluster_sums.values()) / (n * n)
    cr_se = math.sqrt(cr_variance)
    naive_se = math.sqrt(p_hat * (1.0 - p_hat) / n)

    z = normal_quantile(0.5 + confidence / 2.0)
    margin = z * cr_se
    lower = max(0.0, p_hat - margin)
    upper = min(1.0, p_hat + margin)

    return ClusterRobustEstimate(
        point=p_hat,
        cluster_robust_se=cr_se,
        naive_se=naive_se,
        n_clusters=len(cluster_sums),
        n_obs=n,
        lower=lower,
        upper=upper,
        confidence=confidence,
    )
