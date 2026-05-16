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

Implementation is pure-numpy so the audit container has no scikit-learn
runtime dependency; the fit-vs-sklearn equivalence is verified offline
and locked in by hardcoded reference vectors in the test suite (mirrors
:mod:`bba.eval_harness.intervals` which references scipy without
importing it).
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.confidence_calibrator.exceptions import (
    CalibratorNotFittedError,
    InvalidCalibrationDataError,
)
from bba.confidence_calibrator.models import IsotonicFit


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
                "IsotonicCalibrator.predict called before fit; PRD §14 "
                "requires isotonic fit on the training split before "
                "predict can be called.",
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
        :class:`InvalidCalibrationDataError`.
        """
        raise NotImplementedError("RED-phase scaffold; see issue #23")

    def predict(self, scores: Sequence[float]) -> tuple[float, ...]:
        """Map each input ``score`` to a calibrated probability.

        Calls outside the fitted ``x_thresholds`` range are clipped to
        the boundary ``y_values``. Calling before :meth:`fit` raises
        :class:`CalibratorNotFittedError`.
        """
        raise NotImplementedError("RED-phase scaffold; see issue #23")


def pav_fit(
    scores: Sequence[float],
    labels: Sequence[int],
) -> IsotonicFit:
    """Pure functional PAV fit, exposed for offline reference checks.

    Returned :class:`IsotonicFit` is normalized so ``x_thresholds`` is
    strictly increasing and ``y_values`` is non-decreasing in
    ``[0.0, 1.0]``. Used both by :class:`IsotonicCalibrator.fit` and
    by the sklearn-reference equivalence test.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #23")
