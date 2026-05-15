"""Section-level aggregation functions (issue #28).

Each function takes the month's full set of :class:`MonthlyReportRow` and
returns the rows for one report section. The CSV writer and PDF renderer
operate on these aggregated outputs; they do not touch the raw input rows.

This separation is what makes the snapshot-based template tests possible:
golden CSVs assert byte-identical output for a fixed input, and the only
way a refactor of the aggregation logic can change the bytes is if the
clinically-meaningful numbers change.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.report_generator.models import (
    CohortExceptionRow,
    FirstOfMonth,
    HospitalTrendRow,
    IndicationDistributionRow,
    MonthlyReportRow,
    PhysicianOwnViewRow,
    PipelineHealthRow,
    WardScorecardRow,
)


def filter_rows_for_month(
    rows: Sequence[MonthlyReportRow], month: FirstOfMonth
) -> tuple[MonthlyReportRow, ...]:
    """Return only the rows whose ``order_datetime`` falls inside ``month``.

    Boundaries are UTC, half-open ``[month, next_month)``. An order timestamped
    at exactly ``next_month`` 00:00 UTC belongs to the *next* month, not this
    one. The :data:`UTCDatetime` validator guarantees the comparison is
    well-defined.
    """
    raise NotImplementedError


def aggregate_hospital_trend(
    rows: Sequence[MonthlyReportRow], month: FirstOfMonth
) -> tuple[HospitalTrendRow, ...]:
    """Return one :class:`HospitalTrendRow` for ``month``.

    Phase-1 monthly run emits a single-row tuple; the schema is multi-row
    so a quarterly concatenation produces a valid CSV without restructuring.
    """
    raise NotImplementedError


def aggregate_ward_scorecard(
    rows: Sequence[MonthlyReportRow],
) -> tuple[WardScorecardRow, ...]:
    """Return one :class:`WardScorecardRow` per distinct ``ward_id``.

    Output is sorted by ``ward_id`` ascending so the CSV is byte-stable
    across runs (the input order from upstream is not guaranteed).
    """
    raise NotImplementedError


def aggregate_physician_own_view(
    rows: Sequence[MonthlyReportRow],
    physician_ids: Sequence[str],
) -> tuple[PhysicianOwnViewRow, ...]:
    """Return one :class:`PhysicianOwnViewRow` per ``physician_id`` in
    ``physician_ids``.

    Peer benchmarks (median / p25 / p75 of inappropriate_rate) are computed
    over *every* physician with at least one order in ``rows``, not only the
    physicians in ``physician_ids`` — otherwise a small ``physician_ids``
    subset would mis-report the population distribution.

    The "own-data only" property (PRD user story #10) is structural: the
    returned row carries only the physician's own absolute numbers plus
    aggregate peer quartiles. Individual peer rows are not exposed.
    """
    raise NotImplementedError


def aggregate_indication_distribution(
    rows: Sequence[MonthlyReportRow],
) -> tuple[IndicationDistributionRow, ...]:
    """Return one :class:`IndicationDistributionRow` per distinct
    ``indication_code`` appearing in ``rows[i].indication_codes``.

    An order with N indication codes contributes 1 to each code's count
    (multi-indication orders are not split). Codes that appear zero times
    are omitted (the row schema requires ``total_orders >= 1``). Output is
    sorted by ``total_orders`` descending, then ``indication_code``
    ascending, for byte-stable CSVs.
    """
    raise NotImplementedError


def aggregate_cohort_exception(
    rows: Sequence[MonthlyReportRow],
) -> tuple[CohortExceptionRow, ...]:
    """Return one :class:`CohortExceptionRow` per distinct ``cohort_applied``.

    Includes the ``"default"`` cohort. Output sorted by ``cohort_applied``
    ascending for byte-stability.
    """
    raise NotImplementedError


def aggregate_pipeline_health(
    rows: Sequence[MonthlyReportRow],
) -> tuple[PipelineHealthRow, ...]:
    """Return a single :class:`PipelineHealthRow` summarising the month's
    operational health.

    ``classified_orders`` counts rows whose ``final_classification`` is
    ``APPROPRIATE`` or ``INAPPROPRIATE`` (i.e., the pipeline reached a
    confident terminal label). ``needs_review_count`` counts rows whose
    ``final_classification`` is ``NEEDS_REVIEW`` *or* whose
    ``needs_human_review`` flag is set (some appropriate/inappropriate
    classifications are still flagged for human spot-check by policy).
    """
    raise NotImplementedError
