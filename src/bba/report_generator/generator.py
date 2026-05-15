"""Top-level orchestrator for the monthly report (issue #28).

:func:`generate_monthly_report` is the only entry point the monthly CLI
(``bba report``) and the dashboard's "download" handler need to call. It
filters by month, aggregates each section, writes CSVs, renders the PDF,
and returns the resulting :class:`ReportArtifacts`.
"""

from __future__ import annotations

from bba.report_generator.aggregate import (
    aggregate_cohort_exception,
    aggregate_hospital_trend,
    aggregate_indication_distribution,
    aggregate_physician_own_view,
    aggregate_pipeline_health,
    aggregate_ward_scorecard,
    filter_rows_for_month,
)
from bba.report_generator.csv_writer import section_filename, write_section_csv
from bba.report_generator.exceptions import EmptyInputError
from bba.report_generator.models import (
    ReportArtifacts,
    ReportInputs,
    ReportSection,
    SECTION_NAMES,
)
from bba.report_generator.pdf_writer import render_report_pdf


_MONTH_NAMES: tuple[str, ...] = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _month_label(month: object) -> str:
    """Render ``month`` as "May 2026". Pulled out so the orchestrator
    test can stub a fixed label without re-deriving the formatter."""
    # ``month`` is :data:`FirstOfMonth` (a ``date``); the Pydantic validator
    # has already enforced day=1, so name + year is unambiguous.
    return f"{_MONTH_NAMES[month.month - 1]} {month.year}"  # type: ignore[attr-defined]


def generate_monthly_report(inputs: ReportInputs) -> ReportArtifacts:
    """Produce the six section CSVs and the PDF for ``inputs.month``.

    Side effects: writes seven files into ``inputs.output_dir`` (six CSVs
    + one PDF). The directory is created if it does not exist. The
    returned :class:`ReportArtifacts` has absolute paths so downstream
    consumers (email distribution, SSO portal upload) do not need to
    re-resolve relative paths.

    Raises :class:`EmptyInputError` if no row falls inside the month;
    see the exception's docstring for why this is not silent.
    """
    output_dir = inputs.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    in_month_rows = filter_rows_for_month(inputs.rows, inputs.month)
    if not in_month_rows:
        raise EmptyInputError(
            f"no MonthlyReportRow falls inside {inputs.month.isoformat()}; "
            "ingest must have already loaded the month — investigate before "
            "shipping an empty report"
        )

    sections: tuple[ReportSection, ...] = (
        ReportSection(
            name="hospital_trend",
            rows=aggregate_hospital_trend(in_month_rows, inputs.month),
            footer=inputs.footer,
        ),
        ReportSection(
            name="ward_scorecard",
            rows=aggregate_ward_scorecard(in_month_rows),
            footer=inputs.footer,
        ),
        ReportSection(
            name="physician_own_view",
            rows=aggregate_physician_own_view(
                in_month_rows, inputs.physician_ids_for_own_view
            ),
            footer=inputs.footer,
        ),
        ReportSection(
            name="indication_distribution",
            rows=aggregate_indication_distribution(in_month_rows),
            footer=inputs.footer,
        ),
        ReportSection(
            name="cohort_exception",
            rows=aggregate_cohort_exception(in_month_rows),
            footer=inputs.footer,
        ),
        ReportSection(
            name="pipeline_health",
            rows=aggregate_pipeline_health(in_month_rows),
            footer=inputs.footer,
        ),
    )

    csv_paths = {
        section.name: write_section_csv(section, output_dir)
        for section in sections
    }

    pdf_path = output_dir / f"report_{inputs.month.isoformat()}.pdf"
    render_report_pdf(
        sections=sections,
        footer=inputs.footer,
        month_label=_month_label(inputs.month),
        output_path=pdf_path,
    )

    # Sanity check: every section in SECTION_NAMES produced a CSV. If a
    # future refactor adds a section to the Literal but forgets the
    # corresponding aggregator, the artifacts contract still holds.
    assert set(csv_paths.keys()) == set(SECTION_NAMES), (
        f"csv_paths missing sections: {set(SECTION_NAMES) - set(csv_paths.keys())}"
    )

    return ReportArtifacts(
        csv_paths=csv_paths,
        pdf_path=pdf_path,
        footer=inputs.footer,
    )


__all__ = ["generate_monthly_report", "section_filename"]
