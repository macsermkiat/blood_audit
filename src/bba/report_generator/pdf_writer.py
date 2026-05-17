"""PDF renderer for the monthly report (issue #28).

One PDF per monthly run: a cover page (month + footer), then one page per
section. The renderer takes the already-aggregated :class:`ReportSection`
list so it has no opinion about how the numbers were computed — only how
they are laid out on the page.

PDF rendering is the most fragile part of this module (font handling,
binary-stability across reportlab patch releases). The "PDF rendering
tested with a sample run" acceptance criterion (issue #28) is satisfied
by asserting the bytes start with the PDF magic header and that the file
is non-trivial in size; byte-identical PDF output is not promised because
reportlab embeds a generation timestamp.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors

from bba.report_generator.csv_writer import _data_columns
from bba.report_generator.models import (
    ReportFooter,
    ReportSection,
    SectionName,
)


PDF_MAGIC = b"%PDF-"
"""The four-byte magic header every PDF file starts with. Used by the
acceptance test instead of byte-identical comparison because reportlab
embeds a non-deterministic generation timestamp.
"""


_SECTION_TITLES: dict[SectionName, str] = {
    "hospital_trend": "Hospital-wide trend",
    "ward_scorecard": "Per-ward scorecard",
    "physician_own_view": "Per-physician own-view",
    "indication_distribution": "Indication-distribution breakdown",
    "cohort_exception": "Cohort-exception breakdown",
    "pipeline_health": "Pipeline-health summary",
}


def _format_cell_value(value: object) -> str:
    """Render a Python value into its PDF-table cell representation.

    Mirrors the CSV's float convention ("0.5", "0.0") so the PDF and CSV
    render the same numeric values consistently. The implementation is a
    copy of :func:`bba.report_generator.csv_writer._format_cell` minus
    the bool branch; PDF doesn't carry bool cells in any section schema.
    """
    if isinstance(value, float):
        formatted = f"{value:.6f}"
        if "." in formatted:
            formatted = formatted.rstrip("0")
            if formatted.endswith("."):
                formatted += "0"
        return formatted
    return str(value)


def _section_table(section: ReportSection) -> Table:
    columns = _data_columns(section.name)
    header_row = [str(c) for c in columns]
    data_rows = [
        [_format_cell_value(getattr(row, c)) for c in columns] for row in section.rows
    ]
    table_data = (
        [header_row, *data_rows]
        if data_rows
        else [header_row, ["(no data)"] + [""] * (len(header_row) - 1)]
    )
    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


_FOOTER_PDF_FIELDS: tuple[str, ...] = (
    "policy_version",
    "model_id",
    "redactor_version",
    "redactor_model_sha",
    "prompt_hash",
    "evidence_bundle_hash",
)
"""Footer fields rendered on every PDF page. Matches the CSV writer's
``_FOOTER_COLUMNS`` order so the printed page and the persisted CSV
carry the same reproducibility identifiers in the same order.
"""

_FOOTER_PDF_LINE_1: tuple[str, ...] = (
    "policy_version",
    "model_id",
    "redactor_version",
)
_FOOTER_PDF_LINE_2: tuple[str, ...] = (
    "redactor_model_sha",
    "prompt_hash",
    "evidence_bundle_hash",
)
"""Six-field footer split across two printed lines.

The single-line layout at 7pt would overflow the 540pt printable width
of a LETTER page for realistic SHA-bearing field values (~700pt rendered
length). Wrapping by-half keeps every field visible while preserving
the persisted CSV's identifier order — the policy / model / redactor
trio sits above the three content-hash fields.
"""


def _footer_line(footer: ReportFooter, fields: Sequence[str], sep: str) -> str:
    """Render one footer line as ``"name: value | name: value | ..."``."""
    return sep.join(f"{field}: {getattr(footer, field)}" for field in fields)


def _footer_paragraph(footer: ReportFooter) -> Paragraph:
    styles = getSampleStyleSheet()
    line1 = _footer_line(footer, _FOOTER_PDF_LINE_1, " &nbsp; | &nbsp; ")
    line2 = _footer_line(footer, _FOOTER_PDF_LINE_2, " &nbsp; | &nbsp; ")
    return Paragraph(f"{line1}<br/>{line2}", styles["Italic"])


def render_report_pdf(
    sections: Sequence[ReportSection],
    footer: ReportFooter,
    month_label: str,
    output_path: Path,
) -> Path:
    """Render ``sections`` to a PDF at ``output_path`` and return the path.

    ``month_label`` is a human-readable string ("May 2026") used on the
    cover page. The full six-field reproducibility footer
    (``policy_version`` / ``model_id`` / ``redactor_version`` /
    ``redactor_model_sha`` / ``prompt_hash`` / ``evidence_bundle_hash``)
    appears on every page across two lines (single-line layout would
    overflow the printable width for realistic SHA values) so a
    printed-then-detached page still carries the complete
    reproducibility chain.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_footer_line_1 = _footer_line(footer, _FOOTER_PDF_LINE_1, " | ")
    page_footer_line_2 = _footer_line(footer, _FOOTER_PDF_LINE_2, " | ")

    def _on_page(canvas, doc) -> None:  # type: ignore[no-untyped-def]
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        # Line 2 (the hashes) sits closer to the page edge; line 1 sits
        # above it. Both are inside the 0.5-inch bottom margin reserved
        # by ``bottomMargin`` on ``SimpleDocTemplate``.
        canvas.drawString(0.5 * inch, 0.50 * inch, page_footer_line_1)
        canvas.drawString(0.5 * inch, 0.36 * inch, page_footer_line_2)
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=f"KCMH RBC Transfusion Audit — {month_label}",
    )
    styles = getSampleStyleSheet()
    story: list[object] = [
        Paragraph("KCMH RBC Transfusion Audit", styles["Title"]),
        Paragraph(f"Monthly report — {month_label}", styles["Heading2"]),
        Spacer(1, 0.2 * inch),
        _footer_paragraph(footer),
        PageBreak(),
    ]
    for section in sections:
        title = _SECTION_TITLES.get(section.name, section.name)
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Spacer(1, 0.15 * inch))
        story.append(_section_table(section))
        story.append(PageBreak())
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)
    return output_path
