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

from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime, timedelta

from bba.report_generator.models import (
    Classification,
    CohortExceptionRow,
    FirstOfMonth,
    HospitalTrendRow,
    IndicationDistributionRow,
    MonthlyReportRow,
    PhysicianOwnViewRow,
    PipelineHealthRow,
    WardScorecardRow,
)


def _next_month_start(month: FirstOfMonth) -> datetime:
    """Return the UTC timestamp of the first instant of ``month + 1 month``.

    Used as the half-open upper bound of the month filter. Computed by
    adding 32 days and snapping back to day-1 (rather than using
    ``relativedelta``) so the report-generator has no third-party
    date-arithmetic dependency.
    """
    next_anywhere = month.replace(day=28) + timedelta(days=10)
    next_first = next_anywhere.replace(day=1)
    return datetime(next_first.year, next_first.month, 1, tzinfo=UTC)


def _month_start_utc(month: FirstOfMonth) -> datetime:
    return datetime(month.year, month.month, 1, tzinfo=UTC)


def filter_rows_for_month(
    rows: Sequence[MonthlyReportRow], month: FirstOfMonth
) -> tuple[MonthlyReportRow, ...]:
    """Return only the rows whose ``order_datetime`` falls inside ``month``.

    Boundaries are UTC, half-open ``[month, next_month)``. An order
    timestamped at exactly ``next_month`` 00:00 UTC belongs to the *next*
    month, not this one. The :data:`UTCDatetime` validator guarantees the
    comparison is well-defined.
    """
    lower = _month_start_utc(month)
    upper = _next_month_start(month)
    return tuple(r for r in rows if lower <= r.order_datetime < upper)


def _count_classifications(
    rows: Iterable[MonthlyReportRow],
) -> dict[Classification, int]:
    """Return a ``Classification -> count`` mapping with every label
    initialised to zero.

    Zero-initialising every label means downstream consumers can index by
    label without a ``KeyError`` and the row schema's ``ge=0`` constraint
    is always satisfied even on an empty input.
    """
    counts: dict[Classification, int] = {
        "APPROPRIATE": 0,
        "INAPPROPRIATE": 0,
        "NEEDS_REVIEW": 0,
        "INSUFFICIENT_EVIDENCE": 0,
    }
    for row in rows:
        counts[row.final_classification] += 1
    return counts


def _rate(numerator: int, denominator: int) -> float:
    """Return ``numerator / denominator``, or ``0.0`` if denominator is zero.

    A zero-volume ward / cohort / month has rate=0 rather than undefined;
    the schema constrains ``inappropriate_rate`` to ``[0,1]`` so NaN would
    fail validation. Zero is the sensible mass-default.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def aggregate_hospital_trend(
    rows: Sequence[MonthlyReportRow], month: FirstOfMonth
) -> tuple[HospitalTrendRow, ...]:
    """Return one :class:`HospitalTrendRow` for ``month``.

    Phase-1 monthly run emits a single-row tuple; the schema is multi-row
    so a quarterly concatenation produces a valid CSV without restructuring.
    """
    if not rows:
        return ()
    counts = _count_classifications(rows)
    total = len(rows)
    return (
        HospitalTrendRow(
            month=month,
            total_orders=total,
            appropriate=counts["APPROPRIATE"],
            inappropriate=counts["INAPPROPRIATE"],
            needs_review=counts["NEEDS_REVIEW"],
            insufficient_evidence=counts["INSUFFICIENT_EVIDENCE"],
            inappropriate_rate=_rate(counts["INAPPROPRIATE"], total),
        ),
    )


def aggregate_ward_scorecard(
    rows: Sequence[MonthlyReportRow],
) -> tuple[WardScorecardRow, ...]:
    """Return one :class:`WardScorecardRow` per distinct ``ward_id``.

    Output is sorted by ``ward_id`` ascending so the CSV is byte-stable
    across runs (the input order from upstream is not guaranteed).
    """
    by_ward: dict[str, list[MonthlyReportRow]] = {}
    for row in rows:
        by_ward.setdefault(row.ward_id, []).append(row)
    out: list[WardScorecardRow] = []
    for ward_id in sorted(by_ward):
        ward_rows = by_ward[ward_id]
        counts = _count_classifications(ward_rows)
        total = len(ward_rows)
        out.append(
            WardScorecardRow(
                ward_id=ward_id,
                total_orders=total,
                appropriate=counts["APPROPRIATE"],
                inappropriate=counts["INAPPROPRIATE"],
                needs_review=counts["NEEDS_REVIEW"],
                insufficient_evidence=counts["INSUFFICIENT_EVIDENCE"],
                inappropriate_rate=_rate(counts["INAPPROPRIATE"], total),
            )
        )
    return tuple(out)


def _quantile(values: Sequence[float], q: float) -> float:
    """Return the linear-interpolation quantile of ``values`` at ``q``.

    Matches ``statistics.quantiles(..., n=4)`` semantics for q in
    ``{0.25, 0.5, 0.75}`` but stays in pure floats so the output is
    deterministic across Python patch releases. ``values`` must be
    non-empty; the caller guards that.
    """
    if not values:
        raise ValueError("_quantile requires at least one value")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


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
    by_physician: dict[str, list[MonthlyReportRow]] = {}
    for row in rows:
        by_physician.setdefault(row.physician_id, []).append(row)

    peer_rates: list[float] = []
    for pid, prows in by_physician.items():
        counts = _count_classifications(prows)
        peer_rates.append(_rate(counts["INAPPROPRIATE"], len(prows)))

    median = _quantile(peer_rates, 0.5) if peer_rates else 0.0
    p25 = _quantile(peer_rates, 0.25) if peer_rates else 0.0
    p75 = _quantile(peer_rates, 0.75) if peer_rates else 0.0

    out: list[PhysicianOwnViewRow] = []
    for pid in physician_ids:
        own_rows = by_physician.get(pid, [])
        counts = _count_classifications(own_rows)
        own_total = len(own_rows)
        out.append(
            PhysicianOwnViewRow(
                physician_id=pid,
                own_total=own_total,
                own_inappropriate_rate=_rate(counts["INAPPROPRIATE"], own_total),
                peer_median_inappropriate_rate=median,
                peer_p25_inappropriate_rate=p25,
                peer_p75_inappropriate_rate=p75,
            )
        )
    return tuple(out)


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
    counter: Counter[str] = Counter()
    for row in rows:
        for code in row.indication_codes:
            counter[code] += 1
    total_orders = len(rows)
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple(
        IndicationDistributionRow(
            indication_code=code,
            total_orders=n,
            share=_rate(n, total_orders),
        )
        for code, n in items
    )


def aggregate_cohort_exception(
    rows: Sequence[MonthlyReportRow],
) -> tuple[CohortExceptionRow, ...]:
    """Return one :class:`CohortExceptionRow` per distinct ``cohort_applied``.

    Includes the ``"default"`` cohort. Output sorted by ``cohort_applied``
    ascending for byte-stability.
    """
    by_cohort: dict[str, list[MonthlyReportRow]] = {}
    for row in rows:
        by_cohort.setdefault(row.cohort_applied, []).append(row)
    out: list[CohortExceptionRow] = []
    for cohort in sorted(by_cohort):
        crows = by_cohort[cohort]
        counts = _count_classifications(crows)
        out.append(
            CohortExceptionRow(
                cohort_applied=cohort,
                total_orders=len(crows),
                inappropriate=counts["INAPPROPRIATE"],
                inappropriate_rate=_rate(counts["INAPPROPRIATE"], len(crows)),
            )
        )
    return tuple(out)


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
    if not rows:
        return (
            PipelineHealthRow(
                total_orders=0,
                classified_orders=0,
                needs_review_count=0,
                needs_review_rate=0.0,
            ),
        )
    total = len(rows)
    classified = sum(
        1
        for r in rows
        if r.final_classification in ("APPROPRIATE", "INAPPROPRIATE")
    )
    needs_review = sum(
        1
        for r in rows
        if r.final_classification == "NEEDS_REVIEW" or r.needs_human_review
    )
    return (
        PipelineHealthRow(
            total_orders=total,
            classified_orders=classified,
            needs_review_count=needs_review,
            needs_review_rate=_rate(needs_review, total),
        ),
    )
