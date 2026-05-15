"""Top-level orchestrator for the monthly report (issue #28).

:func:`generate_monthly_report` is the only entry point the monthly CLI
(``bba report``) and the dashboard's "download" handler need to call. It
filters by month, aggregates each section, writes CSVs, renders the PDF,
and returns the resulting :class:`ReportArtifacts`.

The per-physician own-view CSVs are *structurally* separated from the
committee-wide section CSVs: each physician's rate lives in its own file
(``physician_own_view_<physician_id>.csv``) so the "own-data only"
property from PRD user story #10 is enforced at the artifact level —
not by trusting a downstream caller to filter rows before distribution.
"""

from __future__ import annotations

from pathlib import Path

from bba.report_generator.aggregate import (
    aggregate_cohort_exception,
    aggregate_hospital_trend,
    aggregate_indication_distribution,
    aggregate_physician_own_view,
    aggregate_pipeline_health,
    aggregate_ward_scorecard,
    filter_rows_for_month,
)
from bba.report_generator.csv_writer import (
    physician_own_view_filename,
    write_section_csv,
)
from bba.report_generator.exceptions import EmptyInputError
from bba.report_generator.models import (
    PhysicianOwnViewRow,
    ReportArtifacts,
    ReportFooter,
    ReportInputs,
    ReportSection,
    SECTION_NAMES,
    SectionName,
)
from bba.report_generator.pdf_writer import render_report_pdf


_COMMITTEE_SECTION_NAMES: tuple[SectionName, ...] = tuple(
    n for n in SECTION_NAMES if n != "physician_own_view"
)
"""The five sections that ship as committee-wide CSVs.

``physician_own_view`` is excluded from the committee CSV set because
its data is per-physician; a single CSV containing every physician's
rate would break the own-data-only contract on distribution. The PDF
*does* still include a physician-own-view table for the committee's
internal review.
"""


_MONTH_NAMES: tuple[str, ...] = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def _month_label(month: object) -> str:
    """Render ``month`` as ``"May 2026"``.

    ``month`` is :data:`FirstOfMonth` (a ``date``); the Pydantic validator
    has already enforced day=1, so the month name + year is unambiguous.
    """
    return f"{_MONTH_NAMES[month.month - 1]} {month.year}"  # type: ignore[attr-defined]


def generate_monthly_report(inputs: ReportInputs) -> ReportArtifacts:
    """Produce the five committee CSVs, the per-physician own-view CSVs,
    and the PDF for ``inputs.month``.

    Side effects: writes ``5 + N + 1`` files into ``inputs.output_dir``
    where ``N = len(inputs.physician_ids_for_own_view)``. The directory
    is created if it does not exist. The returned
    :class:`ReportArtifacts` carries absolute paths so downstream
    consumers (email distribution, SSO portal upload) do not need to
    re-resolve relative paths.

    Raises :class:`EmptyInputError` if no row falls inside the month
    (Asia/Bangkok local); see the exception's docstring for why this
    is not silent.
    """
    output_dir = inputs.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    in_month_rows = filter_rows_for_month(inputs.rows, inputs.month)
    if not in_month_rows:
        raise EmptyInputError(
            f"no MonthlyReportRow falls inside {inputs.month.isoformat()} "
            "(Asia/Bangkok local); ingest must have already loaded the "
            "month — investigate before shipping an empty report"
        )

    physician_view_rows = aggregate_physician_own_view(
        in_month_rows, inputs.physician_ids_for_own_view
    )

    pdf_sections: tuple[ReportSection, ...] = (
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
            rows=physician_view_rows,
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

    committee_section_set: set[SectionName] = set(_COMMITTEE_SECTION_NAMES)
    csv_paths: dict[SectionName, Path] = {
        section.name: write_section_csv(section, output_dir)
        for section in pdf_sections
        if section.name in committee_section_set
    }

    physician_csv_paths = _write_physician_own_view_csvs(
        physician_view_rows, inputs.footer, output_dir
    )

    pdf_path = output_dir / f"report_{inputs.month.isoformat()}.pdf"
    render_report_pdf(
        sections=pdf_sections,
        footer=inputs.footer,
        month_label=_month_label(inputs.month),
        output_path=pdf_path,
    )

    # Sanity check: every committee section produced a CSV. If a future
    # refactor adds a committee section to SECTION_NAMES but forgets the
    # aggregator wiring, the artifacts contract still holds.
    assert set(csv_paths.keys()) == committee_section_set, (
        f"csv_paths missing committee sections: "
        f"{committee_section_set - set(csv_paths.keys())}"
    )

    return ReportArtifacts(
        csv_paths=csv_paths,
        physician_own_view_csv_paths=physician_csv_paths,
        pdf_path=pdf_path,
        footer=inputs.footer,
    )


def _write_physician_own_view_csvs(
    rows: tuple[PhysicianOwnViewRow, ...],
    footer: ReportFooter,
    output_dir: Path,
) -> dict[str, Path]:
    """Write one CSV per physician own-view row and return a mapping.

    Each file holds exactly one data row (the physician's own numbers
    plus aggregate peer quartiles). Distributing the file to that
    physician therefore never exposes another physician's rate.
    """
    paths: dict[str, Path] = {}
    for row in rows:
        section = ReportSection(
            name="physician_own_view",
            rows=(row,),
            footer=footer,
        )
        path = write_section_csv(
            section,
            output_dir,
            filename_override=physician_own_view_filename(row.physician_id),
        )
        paths[row.physician_id] = path
    return paths


__all__ = ["generate_monthly_report"]
