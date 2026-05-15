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
from zoneinfo import ZoneInfo

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


REPORT_TZ = ZoneInfo("Asia/Bangkok")
"""Local timezone for the monthly report bucket.

PRD §"Tz-aware throughout": datetimes stored UTC, rendered Asia/Bangkok.
User story #19: "I want all datetimes stored as UTC in Parquet and
rendered as Asia/Bangkok in the dashboard, so that month-boundary orders
bucket correctly." A monthly report that filtered by UTC month would
mis-bucket orders placed in the last 7 hours of the local month (those
land in the *next* UTC month) — the report month is a hospital business
month, not a UTC month.
"""


def _next_month_first_of(month: FirstOfMonth) -> datetime:
    """Return the first instant of the month *after* ``month`` (local TZ).

    Computed by adding 32 days and snapping back to day-1 (rather than
    using ``relativedelta``) so the report-generator has no third-party
    date-arithmetic dependency. The returned ``datetime`` is **naive**
    on purpose — the caller attaches :data:`REPORT_TZ`.
    """
    next_anywhere = month.replace(day=28) + timedelta(days=10)
    next_first = next_anywhere.replace(day=1)
    return datetime(next_first.year, next_first.month, 1)


def _month_bounds_utc(month: FirstOfMonth) -> tuple[datetime, datetime]:
    """Return the half-open UTC bounds ``[lower, upper)`` for ``month``
    interpreted in Asia/Bangkok local time.

    A row whose ``order_datetime`` (already UTC by validation) lies in
    ``[lower, upper)`` is in the local-month bucket. The conversion via
    :data:`REPORT_TZ` is the place where the PRD's "stored UTC, rendered
    Asia/Bangkok" rule is enforced for monthly bucketing.
    """
    lower_local = datetime(month.year, month.month, 1, tzinfo=REPORT_TZ)
    upper_local = _next_month_first_of(month).replace(tzinfo=REPORT_TZ)
    return lower_local.astimezone(UTC), upper_local.astimezone(UTC)


def filter_rows_for_month(
    rows: Sequence[MonthlyReportRow], month: FirstOfMonth
) -> tuple[MonthlyReportRow, ...]:
    """Return only the rows whose ``order_datetime`` falls inside
    ``month`` *in Asia/Bangkok local time*.

    Boundaries are half-open ``[month_local_start, next_month_local_start)``,
    converted to UTC for the comparison. PRD §"Tz-aware throughout":
    datetimes are stored UTC and rendered Asia/Bangkok; the monthly
    report bucket is the hospital business month, not the UTC month.

    The :data:`UTCDatetime` validator on :class:`MonthlyReportRow`
    guarantees ``order_datetime`` is tz-aware so the comparison is
    well-defined.
    """
    lower, upper = _month_bounds_utc(month)
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
        # Deduplicate within one order's indication tuple: each *distinct*
        # code contributes 1 per order, never N for a code that repeats
        # in the input. The schema contract is "one row per distinct
        # indication code, one contribution per (order, distinct code)";
        # a duplicate from an upstream join would otherwise inflate
        # ``total_orders`` past ``len(rows)`` and could push ``share``
        # above 1.0.
        for code in set(row.indication_codes):
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

    See :class:`PipelineHealthRow` for the bucket definitions. The
    ``INSUFFICIENT_EVIDENCE`` bucket is split out from the review
    bucket so the committee can distinguish "LLM is uncertain"
    (``NEEDS_REVIEW``) from "documentation absent" (``INSUFFICIENT_EVIDENCE``)
    — PRD §"Documentation absence ≠ INAPPROPRIATE".
    """
    total = len(rows)
    if not total:
        return (
            PipelineHealthRow(
                total_orders=0,
                classified_orders=0,
                needs_review_count=0,
                needs_review_rate=0.0,
                insufficient_evidence_count=0,
                insufficient_evidence_rate=0.0,
            ),
        )
    classified = sum(
        1
        for r in rows
        if r.final_classification in ("APPROPRIATE", "INAPPROPRIATE")
    )
    # PipelineHealthRow's contract: needs_review_count counts NEEDS_REVIEW
    # rows *plus* APPROPRIATE/INAPPROPRIATE rows whose needs_human_review
    # flag is set (committee spot-check policy). It must NOT also pick up
    # INSUFFICIENT_EVIDENCE rows — those are documentation-absence, not
    # LLM-uncertainty, and live in their own bucket. Without this guard
    # an upstream that sets needs_human_review on an INSUFFICIENT_EVIDENCE
    # row would double-count it (review + insufficient) and a
    # documentation-absence spike would masquerade as an LLM-review spike.
    needs_review = sum(
        1
        for r in rows
        if r.final_classification == "NEEDS_REVIEW"
        or (
            r.final_classification in ("APPROPRIATE", "INAPPROPRIATE")
            and r.needs_human_review
        )
    )
    insufficient = sum(
        1 for r in rows if r.final_classification == "INSUFFICIENT_EVIDENCE"
    )
    return (
        PipelineHealthRow(
            total_orders=total,
            classified_orders=classified,
            needs_review_count=needs_review,
            needs_review_rate=_rate(needs_review, total),
            insufficient_evidence_count=insufficient,
            insufficient_evidence_rate=_rate(insufficient, total),
        ),
    )
