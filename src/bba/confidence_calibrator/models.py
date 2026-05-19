"""Pydantic v2 + dataclass models for :mod:`bba.confidence_calibrator`.

All public types are immutable (Pydantic ``frozen=True`` / frozen dataclass)
so a calibrator fit, an ECE report, or an agreement verdict cannot be
mutated after the fact — mirrors :mod:`bba.vitals_extractor` and
:mod:`bba.llm_client`.

PRD §14 defines the two confidence paths the audit pipeline persists:

* Isotonic regression on training-split predictions vs. gold labels.
* Agreement-based confidence: Sonnet x 3 with reshuffled few-shot,
  confidence = fraction of runs agreeing on the majority classification.

Both paths produce a single ``confidence`` in ``[0.0, 1.0]`` consumed by
the 0.7 ``NEEDS_REVIEW`` gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_N_BINS = 10
"""Default reliability-diagram and ECE bin count (Guo et al. 2017)."""

DEFAULT_AGREEMENT_RUNS = 3
"""Default number of reshuffled few-shot runs for agreement-based confidence.

PRD §14 / user-story #40: "Sonnet x 3 with reshuffled few-shot, confidence
= fraction agreeing".
"""

ECE_RECAL_HOLDOUT_SIZE = 200
"""Held-out size for the monthly ECE recalibration check (PRD §14)."""

REVIEW_CONFIDENCE_THRESHOLD = 0.7
"""Calibrated confidence below this routes to ``NEEDS_REVIEW`` (PRD §14)."""


@dataclass(frozen=True, slots=True)
class IsotonicFit:
    """Pool-adjacent-violators (PAV) fitted isotonic curve.

    ``x_thresholds`` is the sorted, deduplicated set of training input
    scores; ``y_values`` is the same length and is monotonically
    non-decreasing in ``[0.0, 1.0]``. ``predict`` interpolates linearly
    between adjacent thresholds and clips outside the fitted range.

    ``n_training`` is the count of (score, label) pairs used to fit;
    monthly recalibration jobs compare counts to detect a stale or empty
    refit input.
    """

    x_thresholds: tuple[float, ...]
    y_values: tuple[float, ...]
    n_training: int


class BinStats(BaseModel):
    """Per-bin reliability statistics used by ECE + reliability diagram.

    ``bin_lower`` / ``bin_upper`` define the half-open bin
    ``[lower, upper)`` (the final bin is closed on the right). ``count``
    is the number of predictions falling in the bin; ``mean_confidence``
    is the average predicted probability inside the bin;
    ``accuracy`` is the fraction of labels equal to 1.

    Empty bins (``count == 0``) carry ``mean_confidence = 0.0`` and
    ``accuracy = 0.0`` and contribute zero weight to ECE — matches the
    published Guo et al. 2017 convention.
    """

    model_config = ConfigDict(frozen=True)

    bin_lower: float
    bin_upper: float
    count: int = Field(ge=0)
    mean_confidence: float
    accuracy: float


class EceResult(BaseModel):
    """Expected Calibration Error decomposed by bin.

    ``ece`` is the weighted sum of ``|accuracy - mean_confidence|`` over
    bins, weights ``count / n_samples``. ``bins`` is exactly ``n_bins``
    entries even if some are empty so the reliability diagram can render
    every interval. ``ece`` is in ``[0.0, 1.0]``.
    """

    model_config = ConfigDict(frozen=True)

    ece: float = Field(ge=0.0, le=1.0)
    n_samples: int = Field(ge=0)
    n_bins: int = Field(ge=1)
    bins: tuple[BinStats, ...]


class AgreementResult(BaseModel):
    """Outcome of an agreement-based confidence vote.

    ``classifications`` is the ordered list of label strings produced by
    each shuffled run (``DEFAULT_AGREEMENT_RUNS`` by default).
    ``majority`` is the classification with the highest count;
    ``agreement_count`` is its count; ``confidence`` is
    ``agreement_count / len(classifications)``.

    Tie-breaking is deterministic: when two classifications tie at the
    top count, the first-seen value in ``classifications`` wins so the
    same inputs always produce the same verdict (PRD §14 deterministic
    seed control).
    """

    model_config = ConfigDict(frozen=True)

    classifications: tuple[str, ...]
    majority: str
    agreement_count: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
