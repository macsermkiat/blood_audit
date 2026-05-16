"""bba.confidence_calibrator — isotonic + agreement-based confidence.

See issue #23 for acceptance criteria. PRD §14 (Implementation Decisions)
defines the contract:

* Isotonic regression fit on (raw_LLM_confidence, gold_label) pairs so
  the deployed 0.7 ``NEEDS_REVIEW`` threshold reflects empirical
  P(correct) rather than the LLM's self-reported number.
* Optional agreement-based confidence: Sonnet x 3 with reshuffled
  few-shot ordering, confidence = fraction of runs agreeing on the
  majority classification.
* Monthly recalibration check via ECE on a held-out set (default 200);
  a reliability diagram is rendered to ``docs/eval/`` so the
  transfusion committee can see drift visually.

The module is pure-Python + numpy at runtime (no scikit-learn /
matplotlib runtime deps); equivalence with sklearn's PAV is locked in
by hardcoded reference vectors in the test suite, and the reliability
diagram is rendered as SVG.
"""

from bba.confidence_calibrator.agreement import (
    agreement_confidence,
    shuffle_seeds,
)
from bba.confidence_calibrator.ece import compute_ece
from bba.confidence_calibrator.exceptions import (
    CalibratorNotFittedError,
    ConfidenceCalibratorError,
    InvalidCalibrationDataError,
)
from bba.confidence_calibrator.isotonic import IsotonicCalibrator, pav_fit
from bba.confidence_calibrator.models import (
    DEFAULT_AGREEMENT_RUNS,
    DEFAULT_N_BINS,
    ECE_RECAL_HOLDOUT_SIZE,
    REVIEW_CONFIDENCE_THRESHOLD,
    AgreementResult,
    BinStats,
    EceResult,
    IsotonicFit,
)
from bba.confidence_calibrator.reliability import generate_reliability_diagram

__all__ = [
    "DEFAULT_AGREEMENT_RUNS",
    "DEFAULT_N_BINS",
    "ECE_RECAL_HOLDOUT_SIZE",
    "REVIEW_CONFIDENCE_THRESHOLD",
    "AgreementResult",
    "BinStats",
    "CalibratorNotFittedError",
    "ConfidenceCalibratorError",
    "EceResult",
    "InvalidCalibrationDataError",
    "IsotonicCalibrator",
    "IsotonicFit",
    "agreement_confidence",
    "compute_ece",
    "generate_reliability_diagram",
    "pav_fit",
    "shuffle_seeds",
]
