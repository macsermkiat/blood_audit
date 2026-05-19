"""Custom exceptions for :mod:`bba.confidence_calibrator`.

PRD §14 specifies isotonic-regression calibration of raw LLM confidence and
an agreement-based fallback (Sonnet x 3 reshuffled few-shot, confidence =
fraction agreeing). These exceptions distinguish contract violations
(raised) from routing-style outcomes that the audit pipeline persists with
the row (returned on result types). Mirrors the :mod:`bba.llm_client`
convention.
"""

from __future__ import annotations


class ConfidenceCalibratorError(Exception):
    """Base class for any failure inside :mod:`bba.confidence_calibrator`."""


class CalibratorNotFittedError(ConfidenceCalibratorError):
    """:meth:`IsotonicCalibrator.predict` invoked before :meth:`fit`.

    PRD §14: "Isotonic regression fit on training-split predictions vs
    gold-standard labels". A predict-before-fit call would silently
    return the identity mapping or zero — both would mask a missing
    calibration step in production. We raise so the audit pipeline
    fails loudly instead.
    """


class InvalidCalibrationDataError(ConfidenceCalibratorError):
    """Calibration inputs violate a structural contract.

    Examples: scores / labels length mismatch, labels outside ``{0, 1}``,
    scores outside ``[0.0, 1.0]``, empty input, ``n_bins < 1`` for ECE.
    Raised before any math runs so a malformed monthly recalibration job
    aborts at the boundary rather than persisting a corrupt curve.
    """
