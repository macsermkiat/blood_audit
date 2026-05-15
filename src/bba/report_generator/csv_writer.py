"""CSV writer for monthly report sections (issue #28).

Each section's CSV is a *self-contained* file: column header, data rows,
and the reproducibility footer stamped on every data row. Stamping per row
(rather than only in a trailing footer line) means a downstream consumer
can grep / filter / join the CSV without losing the reproducibility chain.

Output uses ``\\n`` line endings (Unix) and UTF-8 encoding without a BOM:
the file is consumed by Python tooling and the dashboard's web view, not
opened in Excel-on-Windows; forcing CRLF would produce noisy diffs in the
golden snapshot tests.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from bba.report_generator.exceptions import FooterStampError
from bba.report_generator.models import (
    ReportFooter,
    ReportSection,
    SectionName,
    SectionRow,
)


CSV_NEWLINE = "\n"
"""Line terminator for every CSV emitted by this module. Locked to ``\\n``
so the golden-snapshot tests are byte-stable across platforms.
"""

CSV_ENCODING = "utf-8"
"""Encoding for every CSV emitted by this module. No BOM."""


_FOOTER_COLUMNS: tuple[str, ...] = (
    "policy_version",
    "model_id",
    "redactor_version",
)
"""The three footer columns appended after every section's data columns.
Order matches :class:`ReportFooter` field order so the writer never has to
reach into the model's introspection API to discover the column order.
"""


def section_filename(name: SectionName) -> str:
    """Return the canonical CSV filename for ``name``.

    Exposed so the orchestrator and the test fixtures agree on the filename
    without re-encoding the convention in two places.
    """
    return f"{name}.csv"


def _data_columns(section_name: SectionName) -> tuple[str, ...]:
    """Return the data columns for ``section_name`` in canonical order.

    Each tuple here is the column order asserted by the golden-snapshot
    tests for that section. Reordering a column is a spec-level change
    (the schema doc in ``docs/report-schema.md`` references these names);
    the explicit mapping makes the change visible in code review.
    """
    columns_by_section: dict[SectionName, tuple[str, ...]] = {
        "hospital_trend": (
            "month",
            "total_orders",
            "appropriate",
            "inappropriate",
            "needs_review",
            "insufficient_evidence",
            "inappropriate_rate",
        ),
        "ward_scorecard": (
            "ward_id",
            "total_orders",
            "appropriate",
            "inappropriate",
            "needs_review",
            "insufficient_evidence",
            "inappropriate_rate",
        ),
        "physician_own_view": (
            "physician_id",
            "own_total",
            "own_inappropriate_rate",
            "peer_median_inappropriate_rate",
            "peer_p25_inappropriate_rate",
            "peer_p75_inappropriate_rate",
        ),
        "indication_distribution": (
            "indication_code",
            "total_orders",
            "share",
        ),
        "cohort_exception": (
            "cohort_applied",
            "total_orders",
            "inappropriate",
            "inappropriate_rate",
        ),
        "pipeline_health": (
            "total_orders",
            "classified_orders",
            "needs_review_count",
            "needs_review_rate",
        ),
    }
    return columns_by_section[section_name]


def _format_cell(value: object) -> str:
    """Render a single Python value into its CSV cell representation.

    Floats: a fixed-precision representation matters for byte-stable
    snapshots. ``0.5`` and ``0.50000000000000004`` must both render as
    ``0.5``; we format with ``%g`` semantics by stripping trailing zeros
    after a fixed-precision render. ``True``/``False`` would otherwise
    render as ``True``/``False`` (capitalised) which is awkward for
    downstream consumers; render as lowercase to match JSON.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        formatted = f"{value:.6f}"
        # Strip trailing zeros but preserve at least one digit after the
        # decimal point so a float renders as "0.0", not "0" — that
        # preserves the type signal in the CSV (a downstream consumer
        # can keep rate columns distinct from count columns).
        if "." in formatted:
            formatted = formatted.rstrip("0")
            if formatted.endswith("."):
                formatted += "0"
        return formatted
    return str(value)


def _row_to_cells(row: SectionRow, columns: tuple[str, ...]) -> list[str]:
    """Extract ``columns`` from ``row`` and format each cell."""
    return [_format_cell(getattr(row, c)) for c in columns]


def _footer_cells(footer: ReportFooter) -> list[str]:
    """Render the footer fields in canonical order."""
    return [
        footer.policy_version,
        footer.model_id,
        footer.redactor_version,
    ]


def _validate_footer(footer: ReportFooter) -> None:
    """Raise :class:`FooterStampError` if any footer field is empty.

    The Pydantic ``min_length=1`` constraint already rejects empty fields
    at construction, but a re-validation here guards against a writer
    being invoked with a partially constructed footer (e.g., mid-refactor
    where the field validator was bypassed).
    """
    for col in _FOOTER_COLUMNS:
        if not getattr(footer, col):
            raise FooterStampError(
                f"footer field {col!r} is empty; every CSV row must carry "
                "policy_version, model_id, redactor_version"
            )


def _render_csv_text(section: ReportSection) -> str:
    """Return the section's CSV body as a string.

    Implemented separately from :func:`write_section_csv` so a future
    in-memory consumer (e.g., the dashboard preview endpoint) does not
    need to round-trip through the filesystem.
    """
    _validate_footer(section.footer)
    data_cols = _data_columns(section.name)
    header = list(data_cols) + list(_FOOTER_COLUMNS)
    footer_cells = _footer_cells(section.footer)
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator=CSV_NEWLINE)
    writer.writerow(header)
    for row in section.rows:
        writer.writerow(_row_to_cells(row, data_cols) + footer_cells)
    return buf.getvalue()


def write_section_csv(section: ReportSection, output_dir: Path) -> Path:
    """Write ``section`` to ``output_dir / f"{section.name}.csv"`` and return
    the resulting :class:`Path`.

    The output path is derived deterministically from ``section.name`` so a
    second call with the same section overwrites the first (idempotent
    re-run is a project-wide contract; see ``bba.audit_store``).

    Raises :class:`FileNotFoundError` if ``output_dir`` does not exist; the
    caller (the orchestrator) is responsible for creating the directory.
    Raises :class:`FooterStampError` if the section's footer has any empty
    field.
    """
    if not output_dir.exists():
        raise FileNotFoundError(
            f"output_dir {output_dir} does not exist; "
            "the caller must create it before invoking write_section_csv"
        )
    out_path = output_dir / section_filename(section.name)
    text = _render_csv_text(section)
    out_path.write_text(text, encoding=CSV_ENCODING, newline="")
    return out_path
