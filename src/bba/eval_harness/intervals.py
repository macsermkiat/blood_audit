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

import math

from bba.eval_harness.exceptions import EmptyInputError
from bba.eval_harness.models import WilsonInterval

# Acklam (2003) coefficients for inverse-normal-CDF approximation.
# Absolute error ≤ 1.15e-9 over the full real line — overkill for our
# Wilson-CI confidence levels (typically 0.80 / 0.90 / 0.95 / 0.99), but
# zero runtime cost compared to a coarser scheme.
_ACK_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACK_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACK_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACK_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_ACK_PLOW = 0.02425
_ACK_PHIGH = 1 - _ACK_PLOW


def normal_quantile(p: float) -> float:
    """Inverse-CDF of the standard normal at probability ``p``.

    Uses the Acklam (2003) approximation; absolute error ≤ 1.15e-9.
    Raises ``ValueError`` for ``p`` outside the open interval (0, 1).
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"normal_quantile: p must be in (0, 1), got {p!r}")
    if p < _ACK_PLOW:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (
                (((_ACK_C[0] * q + _ACK_C[1]) * q + _ACK_C[2]) * q + _ACK_C[3]) * q
                + _ACK_C[4]
            )
            * q
            + _ACK_C[5]
        ) / ((((_ACK_D[0] * q + _ACK_D[1]) * q + _ACK_D[2]) * q + _ACK_D[3]) * q + 1.0)
    if p > _ACK_PHIGH:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (
                (((_ACK_C[0] * q + _ACK_C[1]) * q + _ACK_C[2]) * q + _ACK_C[3]) * q
                + _ACK_C[4]
            )
            * q
            + _ACK_C[5]
        ) / ((((_ACK_D[0] * q + _ACK_D[1]) * q + _ACK_D[2]) * q + _ACK_D[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (
        (
            (
                (((_ACK_A[0] * r + _ACK_A[1]) * r + _ACK_A[2]) * r + _ACK_A[3]) * r
                + _ACK_A[4]
            )
            * r
            + _ACK_A[5]
        )
        * q
        / (
            (
                (((_ACK_B[0] * r + _ACK_B[1]) * r + _ACK_B[2]) * r + _ACK_B[3]) * r
                + _ACK_B[4]
            )
            * r
            + 1.0
        )
    )


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
    if trials == 0:
        raise EmptyInputError(
            "wilson_ci: trials must be > 0 (no defined center on zero obs)"
        )
    if successes < 0 or successes > trials:
        raise ValueError(f"wilson_ci: successes ({successes}) must be in [0, {trials}]")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"wilson_ci: confidence must be in (0, 1), got {confidence!r}")

    z = normal_quantile(0.5 + confidence / 2.0)
    n = float(trials)
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return WilsonInterval(point=p_hat, lower=lower, upper=upper, confidence=confidence)
