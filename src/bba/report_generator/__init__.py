"""bba.report_generator — monthly CSV + PDF aggregation.

See issue #28 for acceptance criteria. Implementation Decisions §19 in the
PRD scopes the six sections (hospital-wide trend, per-ward scorecard,
per-physician own-view, indication-distribution breakdown, cohort-exception
breakdown, pipeline-health summary) and the reproducibility footer
(``policy_version``, ``model_id``, ``redactor_version``).

This module is a "thin" module per the PRD's deep/thin split: most of the
clinical logic lives upstream (``bba.audit_orders``,
``bba.deterministic_classifier``, ``bba.audit_pipeline``). The
report-generator's job is to project the month's audited orders into six
committee-readable views with reproducibility metadata stamped on every
artifact.
"""

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
    CSV_ENCODING,
    CSV_NEWLINE,
    section_filename,
    write_section_csv,
)
from bba.report_generator.exceptions import (
    EmptyInputError,
    FooterStampError,
    ReportGenerationError,
)
from bba.report_generator.generator import generate_monthly_report
from bba.report_generator.models import (
    SECTION_NAMES,
    Classification,
    CohortExceptionRow,
    FirstOfMonth,
    HospitalTrendRow,
    IndicationDistributionRow,
    MonthlyReportRow,
    PhysicianOwnViewRow,
    PipelineHealthRow,
    ReportArtifacts,
    ReportFooter,
    ReportInputs,
    ReportSection,
    SectionName,
    SectionRow,
    UTCDatetime,
    WardScorecardRow,
)
from bba.report_generator.pdf_writer import PDF_MAGIC, render_report_pdf

__all__ = [
    "CSV_ENCODING",
    "CSV_NEWLINE",
    "Classification",
    "CohortExceptionRow",
    "EmptyInputError",
    "FirstOfMonth",
    "FooterStampError",
    "HospitalTrendRow",
    "IndicationDistributionRow",
    "MonthlyReportRow",
    "PDF_MAGIC",
    "PhysicianOwnViewRow",
    "PipelineHealthRow",
    "ReportArtifacts",
    "ReportFooter",
    "ReportGenerationError",
    "ReportInputs",
    "ReportSection",
    "SECTION_NAMES",
    "SectionName",
    "SectionRow",
    "UTCDatetime",
    "WardScorecardRow",
    "aggregate_cohort_exception",
    "aggregate_hospital_trend",
    "aggregate_indication_distribution",
    "aggregate_physician_own_view",
    "aggregate_pipeline_health",
    "aggregate_ward_scorecard",
    "filter_rows_for_month",
    "generate_monthly_report",
    "render_report_pdf",
    "section_filename",
    "write_section_csv",
]
