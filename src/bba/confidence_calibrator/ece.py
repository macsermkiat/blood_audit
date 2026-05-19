"""Expected Calibration Error (ECE) for monthly recalibration checks.

PRD §14: "Monthly recalibration check via ECE on held-out 200". ECE is
the weighted sum of bin-level gaps between average predicted confidence
and observed accuracy. Bins are equal-width over ``[0.0, 1.0]``; the
final bin is closed on the right so ``prob = 1.0`` lands in the last bin
rather than overflowing.

Reference: Guo, Pleiss, Sun, Weinberger (2017) "On Calibration of Modern
Neural Networks", ICML; eq. (3).

The hardcoded reference example in the test suite uses the worked four-
sample illustration from the same paper so the ECE math is anchored to
a published value.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.confidence_calibrator.exceptions import InvalidCalibrationDataError
from bba.confidence_calibrator.models import (
    DEFAULT_N_BINS,
    BinStats,
    EceResult,
)


def _validate_ece_inputs(
    probs: Sequence[float],
    labels: Sequence[int],
    n_bins: int,
) -> None:
    """Structural-contract validation for :func:`compute_ece`."""
    if n_bins < 1:
        raise InvalidCalibrationDataError(
            f"n_bins must be >= 1; got {n_bins}",
        )
    if len(probs) != len(labels):
        raise InvalidCalibrationDataError(
            f"probs and labels length mismatch: {len(probs)} vs {len(labels)}",
        )
    if len(probs) == 0:
        raise InvalidCalibrationDataError(
            "probs must be non-empty; ECE needs at least one prediction",
        )
    for p in probs:
        if not (0.0 <= p <= 1.0):
            raise InvalidCalibrationDataError(
                f"prob {p!r} outside [0.0, 1.0]; calibrated confidence "
                "is a probability",
            )
    for lab in labels:
        if lab not in (0, 1):
            raise InvalidCalibrationDataError(
                f"label {lab!r} not in {{0, 1}}; gold labels are binary",
            )


def compute_ece(
    probs: Sequence[float],
    labels: Sequence[int],
    n_bins: int = DEFAULT_N_BINS,
) -> EceResult:
    """Compute Expected Calibration Error over equal-width bins.

    ``probs`` are predicted P(positive) in ``[0.0, 1.0]``; ``labels``
    are gold outcomes in ``{0, 1}``. Length mismatch, out-of-range
    inputs, ``n_bins < 1``, or empty ``probs`` raises
    :class:`InvalidCalibrationDataError`.

    Empty bins contribute zero weight to ECE — matches the Guo et al.
    convention and is asserted by ``test_empty_bins_contribute_zero_weight``.
    """
    _validate_ece_inputs(probs, labels, n_bins)

    n = len(probs)
    bin_edges = tuple(i / n_bins for i in range(n_bins + 1))

    # Per-bin running sums so we never materialize the per-bin point
    # lists (matters for the monthly 200-row holdout job).
    counts = [0] * n_bins
    sum_conf = [0.0] * n_bins
    sum_label = [0] * n_bins

    for p, lab in zip(probs, labels, strict=True):
        # Half-open bins ``[edges[i], edges[i+1])`` with the final bin
        # closed on the right so prob == 1.0 lands in the last bin.
        bin_idx = min(int(p * n_bins), n_bins - 1)
        counts[bin_idx] += 1
        sum_conf[bin_idx] += p
        sum_label[bin_idx] += lab

    bins: list[BinStats] = []
    weighted_gap = 0.0
    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        c = counts[i]
        if c == 0:
            bins.append(
                BinStats(
                    bin_lower=lower,
                    bin_upper=upper,
                    count=0,
                    mean_confidence=0.0,
                    accuracy=0.0,
                ),
            )
            continue
        mean_conf = sum_conf[i] / c
        accuracy = sum_label[i] / c
        gap = abs(mean_conf - accuracy)
        weight = c / n
        weighted_gap += weight * gap
        bins.append(
            BinStats(
                bin_lower=lower,
                bin_upper=upper,
                count=c,
                mean_confidence=mean_conf,
                accuracy=accuracy,
            ),
        )

    # Floating-point drift can push the ECE one ulp outside [0, 1]; clip
    # so the Pydantic ``ge=0, le=1`` constraint never spuriously rejects
    # a mathematically valid result.
    weighted_gap = max(0.0, min(1.0, weighted_gap))

    return EceResult(
        ece=weighted_gap,
        n_samples=n,
        n_bins=n_bins,
        bins=tuple(bins),
    )
