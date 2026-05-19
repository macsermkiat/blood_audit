"""Pool-adjacent-violators (PAV) isotonic regression for confidence.

PRD §14: raw LLM confidence is rewritten through isotonic regression fit
on a held-out calibration split, so the deployed 0.7 ``NEEDS_REVIEW``
threshold reflects empirical P(correct) rather than the model's
self-reported number.

Public surface:

* :class:`IsotonicCalibrator` — stateful wrapper around an
  :class:`IsotonicFit`. ``fit`` runs PAV; ``predict`` interpolates and
  clips. ``predict`` before ``fit`` raises
  :class:`CalibratorNotFittedError` to surface a missing calibration
  step rather than silently returning the identity mapping.
* :func:`pav_fit` — pure functional PAV exposed for offline reference
  checks against scikit-learn (no scikit-learn runtime dep).

Implementation is pure-Python (stdlib :mod:`bisect`) so the audit
container has no scikit-learn runtime dependency; the fit-vs-sklearn
equivalence is verified by hardcoded reference vectors in the test
suite, mirroring :mod:`bba.eval_harness.intervals` which references
scipy without importing it.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

from bba.confidence_calibrator.exceptions import (
    CalibratorNotFittedError,
    InvalidCalibrationDataError,
)
from bba.confidence_calibrator.models import IsotonicFit


def _validate_fit_inputs(
    scores: Sequence[float],
    labels: Sequence[int],
) -> None:
    """Structural-contract validation for ``fit`` / :func:`pav_fit`."""
    if len(scores) != len(labels):
        raise InvalidCalibrationDataError(
            f"scores and labels length mismatch: {len(scores)} vs {len(labels)}",
        )
    if len(scores) == 0:
        raise InvalidCalibrationDataError(
            "scores must be non-empty; isotonic regression needs at least "
            "one training pair",
        )
    for s in scores:
        if not (0.0 <= s <= 1.0):
            raise InvalidCalibrationDataError(
                f"score {s!r} outside [0.0, 1.0]; raw LLM confidence is a "
                "probability and must be bounded",
            )
    for lab in labels:
        if lab not in (0, 1):
            raise InvalidCalibrationDataError(
                f"label {lab!r} not in {{0, 1}}; gold-standard labels are binary",
            )


def pav_fit(
    scores: Sequence[float],
    labels: Sequence[int],
) -> IsotonicFit:
    """Pure functional PAV fit, exposed for offline reference checks.

    Returned :class:`IsotonicFit` is normalized so ``x_thresholds`` is
    strictly increasing and ``y_values`` is non-decreasing in
    ``[0.0, 1.0]``. Used both by :class:`IsotonicCalibrator.fit` and
    by the sklearn-reference equivalence test.

    Duplicate ``scores`` are merged by weighted-average before PAV runs
    (matches the sklearn ``IsotonicRegression`` convention).
    """
    _validate_fit_inputs(scores, labels)

    # Sort by score so PAV scans monotonically. Stable sort by (x, y).
    pairs = sorted(zip(scores, labels, strict=True), key=lambda pair: pair[0])

    # Collapse duplicate-x by weighted mean so each input x is one point.
    grouped: list[tuple[float, float, int]] = []  # (x, mean_y, weight)
    for x, y in pairs:
        if grouped and grouped[-1][0] == x:
            xx, ssy, ww = grouped[-1]
            new_w = ww + 1
            new_y = (ssy * ww + y) / new_w
            grouped[-1] = (xx, new_y, new_w)
        else:
            grouped.append((x, float(y), 1))

    # Single-pass stack-based PAV. Each stack entry is one PAV block:
    # (left_x, right_x, mean_y, weight).
    stack: list[tuple[float, float, float, int]] = []
    for x, mean_y, weight in grouped:
        stack.append((x, x, mean_y, weight))
        while len(stack) >= 2 and stack[-2][2] > stack[-1][2]:
            l1, _, y1, w1 = stack[-2]
            _, r2, y2, w2 = stack[-1]
            merged_y = (y1 * w1 + y2 * w2) / (w1 + w2)
            merged_w = w1 + w2
            stack.pop()
            stack.pop()
            stack.append((l1, r2, merged_y, merged_w))

    # Materialize the piecewise-constant curve. Each block contributes
    # two threshold points (left_x, mean_y), (right_x, mean_y) unless the
    # block has a single x value, in which case it contributes one.
    xs: list[float] = []
    ys: list[float] = []
    for left_x, right_x, mean_y, _ in stack:
        clipped_y = max(0.0, min(1.0, mean_y))
        if not xs or left_x > xs[-1]:
            xs.append(left_x)
            ys.append(clipped_y)
        if right_x > xs[-1]:
            xs.append(right_x)
            ys.append(clipped_y)

    return IsotonicFit(
        x_thresholds=tuple(xs),
        y_values=tuple(ys),
        n_training=len(scores),
    )


def _predict_one(
    x: float,
    x_thresholds: tuple[float, ...],
    y_values: tuple[float, ...],
) -> float:
    """Single-point predict with clipping outside the fitted range."""
    if x <= x_thresholds[0]:
        return y_values[0]
    if x >= x_thresholds[-1]:
        return y_values[-1]
    # Strictly inside the fitted range. ``bisect_right`` returns the
    # first index where ``x_thresholds[i] > x``; we want the interval
    # ``[x_thresholds[i-1], x_thresholds[i]]`` for linear interpolation.
    idx = bisect_right(x_thresholds, x)
    x_lo = x_thresholds[idx - 1]
    x_hi = x_thresholds[idx]
    y_lo = y_values[idx - 1]
    y_hi = y_values[idx]
    if x_hi == x_lo:
        return y_lo
    t = (x - x_lo) / (x_hi - x_lo)
    return y_lo + t * (y_hi - y_lo)


class IsotonicCalibrator:
    """Stateful isotonic-regression calibrator.

    The instance starts unfitted; ``fit`` populates :attr:`fit_result`
    via the PAV algorithm. Successive ``fit`` calls overwrite the prior
    curve (monthly recalibration semantics, PRD §14).
    """

    def __init__(self) -> None:
        self._fit: IsotonicFit | None = None

    @property
    def is_fitted(self) -> bool:
        """``True`` when :meth:`fit` has produced a stored curve."""
        return self._fit is not None

    @property
    def fit_result(self) -> IsotonicFit:
        """The stored :class:`IsotonicFit`.

        Raises :class:`CalibratorNotFittedError` when ``fit`` has not
        been called yet.
        """
        if self._fit is None:
            raise CalibratorNotFittedError(
                "IsotonicCalibrator.fit_result accessed before fit; "
                "PRD §14 requires isotonic fit on the training split.",
            )
        return self._fit

    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
    ) -> IsotonicFit:
        """Fit the PAV curve on ``(scores, labels)`` pairs.

        ``scores`` are raw LLM confidences in ``[0.0, 1.0]``; ``labels``
        are gold-standard outcomes in ``{0, 1}``. Length mismatch,
        out-of-range labels, or empty input raises
        :class:`InvalidCalibrationDataError`. Successive calls overwrite
        the prior curve.
        """
        self._fit = pav_fit(scores, labels)
        return self._fit

    def predict(self, scores: Sequence[float]) -> tuple[float, ...]:
        """Map each input ``score`` to a calibrated probability.

        Calls outside the fitted ``x_thresholds`` range are clipped to
        the boundary ``y_values``. Calling before :meth:`fit` raises
        :class:`CalibratorNotFittedError`.
        """
        if self._fit is None:
            raise CalibratorNotFittedError(
                "IsotonicCalibrator.predict called before fit; "
                "PRD §14 requires isotonic fit on the training split.",
            )
        xs = self._fit.x_thresholds
        ys = self._fit.y_values
        return tuple(_predict_one(float(s), xs, ys) for s in scores)
