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
    SectionRow,
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
        [_format_cell_value(getattr(row, c)) for c in columns]
        for row in section.rows
    ]
    table_data = [header_row, *data_rows] if data_rows else [header_row, ["(no data)"] + [""] * (len(header_row) - 1)]
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


def _footer_paragraph(footer: ReportFooter) -> Paragraph:
    styles = getSampleStyleSheet()
    text = (
        f"policy_version: {footer.policy_version} &nbsp; | &nbsp; "
        f"model_id: {footer.model_id} &nbsp; | &nbsp; "
        f"redactor_version: {footer.redactor_version}"
    )
    return Paragraph(text, styles["Italic"])


def render_report_pdf(
    sections: Sequence[ReportSection],
    footer: ReportFooter,
    month_label: str,
    output_path: Path,
) -> Path:
    """Render ``sections`` to a PDF at ``output_path`` and return the path.

    ``month_label`` is a human-readable string ("May 2026") used on the
    cover page. The footer (policy_version / model_id / redactor_version)
    appears on every page so a printed-then-detached page still carries
    the reproducibility chain.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_page(canvas, doc) -> None:  # type: ignore[no-untyped-def]
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(
            0.5 * inch,
            0.4 * inch,
            (
                f"policy_version: {footer.policy_version} | "
                f"model_id: {footer.model_id} | "
                f"redactor_version: {footer.redactor_version}"
            ),
        )
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
        Paragraph(f"KCMH RBC Transfusion Audit", styles["Title"]),
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
