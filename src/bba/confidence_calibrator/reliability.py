"""Reliability-diagram (calibration plot) SVG renderer.

PRD §14 acceptance criterion: "Calibration plot generator (reliability
diagram) to docs/eval/". The audit container has no matplotlib
dependency (deployment is on-prem at KCMH with offline HF / pythainlp
caches) so we emit SVG directly — it renders in browsers, GitHub
markdown, and the reviewer dashboard without a binary build step.

The diagram shows:

* Per-bin accuracy markers at the bin midpoint (one marker per bin,
  including empty bins, so a missing marker means a regression).
* A diagonal y=x reference line (perfect calibration).
* The ECE value in the diagram title.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from bba.confidence_calibrator.ece import compute_ece
from bba.confidence_calibrator.models import DEFAULT_N_BINS

_SVG_WIDTH = 400
_SVG_HEIGHT = 400
_SVG_MARGIN = 40


def _to_svg_xy(x: float, y: float) -> tuple[float, float]:
    """Map data (x, y) in ``[0, 1]^2`` to SVG coordinates.

    SVG y-axis grows downward, so accuracy ``1.0`` maps to the top of
    the plot area.
    """
    plot_w = _SVG_WIDTH - 2 * _SVG_MARGIN
    plot_h = _SVG_HEIGHT - 2 * _SVG_MARGIN
    sx = _SVG_MARGIN + x * plot_w
    sy = _SVG_HEIGHT - _SVG_MARGIN - y * plot_h
    return sx, sy


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
    :class:`InvalidCalibrationDataError`. Validation happens BEFORE any
    filesystem I/O so a rejected job never leaves a stale or empty file
    on disk.
    """
    ece_result = compute_ece(probs, labels, n_bins=n_bins)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{_SVG_WIDTH}" height="{_SVG_HEIGHT}" '
        f'viewBox="0 0 {_SVG_WIDTH} {_SVG_HEIGHT}">',
    )
    parts.append(f"<title>{title} (ECE={ece_result.ece:.4f})</title>")
    parts.append(
        f'<rect x="0" y="0" width="{_SVG_WIDTH}" height="{_SVG_HEIGHT}" '
        'fill="white"/>',
    )

    plot_w = _SVG_WIDTH - 2 * _SVG_MARGIN
    plot_h = _SVG_HEIGHT - 2 * _SVG_MARGIN
    parts.append(
        f'<rect x="{_SVG_MARGIN}" y="{_SVG_MARGIN}" '
        f'width="{plot_w}" height="{plot_h}" '
        'fill="none" stroke="black"/>',
    )

    # Diagonal y=x reference (perfect calibration).
    x0, y0 = _to_svg_xy(0.0, 0.0)
    x1, y1 = _to_svg_xy(1.0, 1.0)
    parts.append(
        f'<line class="reliability-diagonal" '
        f'x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" '
        'stroke="gray" stroke-dasharray="4,4"/>',
    )

    # One marker per bin (including empty bins).
    for bin_stats in ece_result.bins:
        midpoint = (bin_stats.bin_lower + bin_stats.bin_upper) / 2.0
        cx, cy = _to_svg_xy(midpoint, bin_stats.accuracy)
        parts.append(
            f'<circle class="reliability-bin" '
            f'cx="{cx}" cy="{cy}" r="4" fill="steelblue"/>',
        )

    parts.append("</svg>")

    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path
