"""Cluster-robust standard errors (PRD §11, User Story #31).

Audited orders are not i.i.d.: a single attending physician produces many
orders in a given month, and a single ward concentrates patient
populations. The naive binomial SE understates uncertainty by roughly
1.5-2x in this design (PRD §"Anti-hallucination is layered, not single-
shot."), which would inflate the per-stratum sensitivity claims that the
publication report cites.

This module implements the CR0 cluster-robust sandwich variance estimator
for a binomial proportion. ``cluster_id`` is the concatenation of
``physician_id`` and ``ward`` — the harness treats their cross as the
effective cluster (the same physician on two wards is not the same cluster).

Reference (manual): Liang & Zeger (1986); Cameron & Miller (2015) survey.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.eval_harness.models import ClusterRobustEstimate


def cluster_robust_proportion_ci(
    indicators: Sequence[bool],
    cluster_ids: Sequence[str],
    *,
    confidence: float = 0.95,
) -> ClusterRobustEstimate:
    """Cluster-robust Wald CI for a proportion.

    Parameters
    ----------
    indicators
        Per-observation indicator (``True`` ↦ "event"). The estimated
        proportion is the simple mean of these.
    cluster_ids
        Per-observation cluster label. Observations sharing a label are
        treated as one cluster for the sandwich variance.
    confidence
        Two-sided Wald coverage (default 0.95).

    Returns
    -------
    ClusterRobustEstimate
        ``cluster_robust_se`` is the CR0 sandwich SE; ``naive_se`` is the
        binomial SE under the i.i.d. assumption (provided for comparison —
        the report cites the ratio as the "design effect").

    Raises
    ------
    ShapeMismatchError
        ``indicators`` and ``cluster_ids`` disagree in length.
    EmptyInputError
        Either input is empty.
    """
    raise NotImplementedError("eval_harness.cluster: RED phase, see issue #20")
