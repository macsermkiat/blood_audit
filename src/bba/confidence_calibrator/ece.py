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
from bba.confidence_calibrator.models import DEFAULT_N_BINS, EceResult


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
    convention and is asserted by :class:`bba.confidence_calibrator.tests`.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #23")
