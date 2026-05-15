"""Wilson score confidence intervals for binomial proportions (PRD §11).

The Wilson form is the publication-grade interval for the per-stratum
sensitivity targets: bounded in [0, 1], near-symmetric near 0.5, and
shrinking smoothly at the boundary. Normal-approximation intervals leak
out of [0, 1] near the boundary, which would be a regulator-visible defect
on strata where natural prevalence is very high or very low (e.g., Hb<7).

Reference (manual): Wilson (1927) "Probable Inference, the Law of
Succession, and Statistical Inference." Validation in tests is against
``scipy.stats.binomtest(...).proportion_ci(method='wilson')`` reference
values hardcoded into the test (scipy is not a runtime dependency).
"""

from __future__ import annotations

from bba.eval_harness.models import WilsonInterval


def wilson_ci(
    successes: int, trials: int, *, confidence: float = 0.95
) -> WilsonInterval:
    """Wilson score confidence interval for a binomial proportion.

    Raises :class:`bba.eval_harness.exceptions.EmptyInputError` for
    ``trials == 0`` (a CI on zero observations has no defined center).
    ``successes`` must be in ``[0, trials]``; out-of-range raises
    :class:`ValueError`. ``confidence`` is the two-sided coverage probability
    (default 0.95).
    """
    raise NotImplementedError("eval_harness.intervals: RED phase, see issue #20")


def normal_quantile(p: float) -> float:
    """Inverse-CDF of the standard normal at probability ``p``.

    Uses the Acklam approximation; sufficient precision (≤ 1.15e-9 absolute
    error) for the CI calculation. Kept public so the cluster-robust SE Wald
    interval can reuse it without importing scipy.
    """
    raise NotImplementedError("eval_harness.intervals: RED phase, see issue #20")
