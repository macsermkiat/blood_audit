"""Frozen pydantic models for monthly report aggregation (issue #28).

Inputs and outputs are deliberately decoupled from :class:`bba.audit_store.AuditRow`.
The audit store persists the *technical* row (hashes, model_id, evidence
bundle hash, ...); the report generator consumes a *report-shaped* row that
includes the ingest-side metadata the audit-store schema does not carry
(``ward_id``, ``physician_id``). The upstream layer is responsible for the
join.

All persisted artifacts (CSV + PDF) carry a :class:`ReportFooter` with
``policy_version``, ``model_id``, ``redactor_version`` so a downstream
auditor can reconstruct any line from the same inputs + the same code
(PRD §"Output schema").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


# =============================================================================
# Shared aliases
# =============================================================================


Classification = Literal[
    "APPROPRIATE",
    "INAPPROPRIATE",
    "NEEDS_REVIEW",
    "INSUFFICIENT_EVIDENCE",
]
"""Mirrors :data:`bba.audit_store.models.Classification`. Duplicated here so
the report-generator's input contract does not transitively import the
storage layer's runtime dependencies (pyarrow, duckdb)."""


SectionName = Literal[
    "hospital_trend",
    "ward_scorecard",
    "physician_own_view",
    "indication_distribution",
    "cohort_exception",
    "pipeline_health",
]
"""The six report sections required by issue #28 scope.

The order here is the order they appear in the PDF and the order their CSV
files are referenced from the manifest. Hospital-wide trend leads because
the committee's reading flow is "overall first, then drill into wards".
"""


SECTION_NAMES: tuple[SectionName, ...] = (
    "hospital_trend",
    "ward_scorecard",
    "physician_own_view",
    "indication_distribution",
    "cohort_exception",
    "pipeline_health",
)
"""Canonical tuple of section names. Use this instead of ``get_args`` so the
ordering is explicit and downstream code does not silently reorder if the
Literal expansion order changes in a future Python version.
"""


def _ensure_utc(dt: datetime) -> datetime:
    """Reject naive datetimes; normalize aware non-UTC to UTC.

    Mirrors the audit-store / evidence-bundle convention: naive timestamps
    would later cross a month-boundary comparison silently, bucketing an
    order into the wrong report. Reject at construction.
    """
    if dt.tzinfo is None:
        raise ValueError(
            "datetime must be tz-aware; naive datetimes are forbidden in "
            "report_generator (month-boundary comparisons require UTC)"
        )
    return dt.astimezone(UTC)


UTCDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]
"""A ``datetime`` constrained to tz-aware UTC at validation time."""


def _ensure_first_of_month(d: date) -> date:
    """Reject any ``date`` that is not the first of its month.

    The report is monthly. A row tagged with ``month=date(2026, 5, 15)``
    would be ambiguous: is that "month containing 2026-05-15" or
    "rolling 30-day window ending 2026-05-15"? Forcing first-of-month
    eliminates the ambiguity.
    """
    if d.day != 1:
        raise ValueError(
            f"month must be the first of the month (got {d.isoformat()})"
        )
    return d


FirstOfMonth = Annotated[date, AfterValidator(_ensure_first_of_month)]
"""A ``date`` constrained to the first day of a calendar month."""


# =============================================================================
# Footer (stamped on every section)
# =============================================================================


class ReportFooter(BaseModel):
    """Reproducibility footer stamped on every CSV row and the PDF.

    Per PRD §"Output schema": ``policy_version``, ``model_id``,
    ``redactor_version`` are the three identifiers needed to re-derive any
    report line from the source AuditRows. They are required (not optional)
    on every artifact; the empty string is rejected because "we forgot to
    stamp" is the failure mode this footer exists to prevent.
    """

    model_config = ConfigDict(frozen=True)

    policy_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    redactor_version: str = Field(min_length=1)


# =============================================================================
# Input row (the report-generator's input contract)
# =============================================================================


class MonthlyReportRow(BaseModel):
    """One audited RBC order, projected into the report-shaped contract.

    The upstream layer (a dashboard query or batch join in the monthly CLI)
    joins :class:`bba.audit_store.AuditRow` with the ingest-side metadata
    (``ward_id``, ``physician_id``) and produces a tuple of these. Keeping
    the input shape report-specific isolates the generator from changes to
    the audit-store schema.

    ``indication_codes`` is a tuple (not a list) so the model is hashable
    and structurally immutable.
    """

    model_config = ConfigDict(frozen=True)

    audit_id: str = Field(min_length=1)
    an_hash: str = Field(min_length=1)
    hn_hash: str = Field(min_length=1)
    order_datetime: UTCDatetime
    ward_id: str = Field(min_length=1)
    physician_id: str = Field(min_length=1)
    final_classification: Classification
    cohort_applied: str = Field(min_length=1)
    indication_codes: tuple[str, ...]
    needs_human_review: bool


# =============================================================================
# Section row schemas (one per report section)
# =============================================================================


class HospitalTrendRow(BaseModel):
    """One row in the hospital-wide trend section.

    A single month produces one row; a multi-month trend would produce N.
    The Phase-1 monthly run emits one row per call, but the schema is
    multi-row to allow the committee's quarterly review to concatenate
    three monthly CSVs without a schema change.
    """

    model_config = ConfigDict(frozen=True)

    month: FirstOfMonth
    total_orders: int = Field(ge=0)
    appropriate: int = Field(ge=0)
    inappropriate: int = Field(ge=0)
    needs_review: int = Field(ge=0)
    insufficient_evidence: int = Field(ge=0)
    inappropriate_rate: float = Field(ge=0.0, le=1.0)


class WardScorecardRow(BaseModel):
    """One row in the per-ward scorecard section."""

    model_config = ConfigDict(frozen=True)

    ward_id: str = Field(min_length=1)
    total_orders: int = Field(ge=0)
    appropriate: int = Field(ge=0)
    inappropriate: int = Field(ge=0)
    needs_review: int = Field(ge=0)
    insufficient_evidence: int = Field(ge=0)
    inappropriate_rate: float = Field(ge=0.0, le=1.0)


class PhysicianOwnViewRow(BaseModel):
    """Per-physician own-view scorecard row.

    The "own-data only" property (PRD user story #10) is enforced at the
    schema level: the row carries the physician's own absolute numbers
    plus *aggregate* peer benchmarks. There is no field on this row that
    would let a recipient reconstruct another physician's individual
    orders or rate; the peer benchmarks are quartile statistics over the
    set of all physicians in the month.
    """

    model_config = ConfigDict(frozen=True)

    physician_id: str = Field(min_length=1)
    own_total: int = Field(ge=0)
    own_inappropriate_rate: float = Field(ge=0.0, le=1.0)
    peer_median_inappropriate_rate: float = Field(ge=0.0, le=1.0)
    peer_p25_inappropriate_rate: float = Field(ge=0.0, le=1.0)
    peer_p75_inappropriate_rate: float = Field(ge=0.0, le=1.0)


class IndicationDistributionRow(BaseModel):
    """One row in the indication-distribution breakdown section."""

    model_config = ConfigDict(frozen=True)

    indication_code: str = Field(min_length=1)
    total_orders: int = Field(ge=1)
    share: float = Field(ge=0.0, le=1.0)


class CohortExceptionRow(BaseModel):
    """One row in the cohort-exception breakdown section.

    "Cohort exception" follows PRD §"Cohort thresholds": cardiac-surgery,
    ortho+cardiac, and similar cohorts have non-default Hb thresholds,
    and this section breaks the month's totals down by which cohort
    threshold was applied. ``cohort_applied="default"`` is a valid row.
    """

    model_config = ConfigDict(frozen=True)

    cohort_applied: str = Field(min_length=1)
    total_orders: int = Field(ge=0)
    inappropriate: int = Field(ge=0)
    inappropriate_rate: float = Field(ge=0.0, le=1.0)


class PipelineHealthRow(BaseModel):
    """One row in the pipeline-health summary section.

    Tracks the operational health of the monthly run: how many orders the
    pipeline classified end-to-end vs. how many fell to ``NEEDS_REVIEW``
    or ``INSUFFICIENT_EVIDENCE``. A spike in ``needs_review_rate`` is the
    signal the committee uses to flag silent regression in the LLM stack.
    """

    model_config = ConfigDict(frozen=True)

    total_orders: int = Field(ge=0)
    classified_orders: int = Field(ge=0)
    needs_review_count: int = Field(ge=0)
    needs_review_rate: float = Field(ge=0.0, le=1.0)


# =============================================================================
# Inputs / Outputs
# =============================================================================


class ReportInputs(BaseModel):
    """The full input contract for :func:`generate_monthly_report`.

    ``physician_ids_for_own_view`` is an explicit list so the caller (the
    monthly CLI) controls which physicians get rendered. Empty tuple is a
    valid value: the committee report may omit the physician section
    entirely and individual physician reports get generated separately.
    """

    model_config = ConfigDict(frozen=True)

    month: FirstOfMonth
    rows: tuple[MonthlyReportRow, ...]
    footer: ReportFooter
    output_dir: Path
    physician_ids_for_own_view: tuple[str, ...] = ()


class ReportArtifacts(BaseModel):
    """The output of :func:`generate_monthly_report`.

    Every section gets a CSV path; the PDF is a single file containing all
    six sections plus the footer. ``csv_paths`` is keyed by
    :data:`SectionName` so the caller can pull a specific section without
    string-indexing.
    """

    model_config = ConfigDict(frozen=True)

    csv_paths: Mapping[SectionName, Path]
    pdf_path: Path
    footer: ReportFooter


# =============================================================================
# Section envelope (passed to the CSV and PDF writers)
# =============================================================================


SectionRow = (
    HospitalTrendRow
    | WardScorecardRow
    | PhysicianOwnViewRow
    | IndicationDistributionRow
    | CohortExceptionRow
    | PipelineHealthRow
)
"""Union of every section's row type. Use in writer signatures so the writer
accepts a heterogeneous payload but each *call* still passes one row type."""


class ReportSection(BaseModel):
    """A complete section: name + rows + footer.

    The writer layer takes a ``ReportSection`` and emits one CSV per
    section. Carrying the footer per section (rather than only at the
    document level) means a partial extract of a single section's CSV
    still has the reproducibility metadata on every row.
    """

    model_config = ConfigDict(frozen=True)

    name: SectionName
    rows: Sequence[SectionRow]
    footer: ReportFooter
