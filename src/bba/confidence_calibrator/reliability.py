"""Reliability-diagram (calibration plot) SVG renderer.

PRD §14 acceptance criterion: "Calibration plot generator (reliability
diagram) to docs/eval/". The audit container has no matplotlib
dependency (deployment is on-prem at KCMH with offline HF / pythainlp
caches) so we emit SVG directly — it renders in browsers, GitHub
markdown, and the reviewer dashboard without a binary build step.

The diagram shows:

* Per-bin accuracy bars at the bin midpoint.
* A diagonal y=x reference line (perfect calibration).
* An ECE annotation in the title block.

Bars sized to bin counts are out of scope for this module; the audit
PRD only asks for a reliability plot, not a confidence histogram.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from bba.confidence_calibrator.exceptions import InvalidCalibrationDataError
from bba.confidence_calibrator.models import DEFAULT_N_BINS


def generate_reliability_diagram(
    probs: Sequence[float],
    labels: Sequence[int],
    out_path: Path,
    n_bins: int = DEFAULT_N_BINS,
    title: str = "Reliability Diagram",
) -> Path:
    """Render a reliability diagram as SVG to ``out_path`` and return it.

    Creates the parent directory if missing. Overwrites an existing file
    at the same path (monthly recalibration jobs write the same canonical
    filename).

    Same validation as :func:`compute_ece` — length mismatch, out-of-range
    inputs, empty ``probs``, ``n_bins < 1`` raise
    :class:`InvalidCalibrationDataError`.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #23")
