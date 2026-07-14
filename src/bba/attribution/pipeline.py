"""End-to-end assembly: verdicts + attribution → :class:`RankingResult`.

This is the single function the pilot runner (and, next build, the
full-cohort runner) calls. Swapping the verdict source is the ONLY
change between the two builds — everything below the ``verdicts``
mapping is verdict-source-agnostic.
"""

from __future__ import annotations

from collections.abc import Mapping

from bba.attribution.lab_stats import (
    OrderLabValue,
    aggregate_department_lab_stats,
    aggregate_doctor_lab_stats,
)
from bba.attribution.models import (
    Bucket,
    BucketTotals,
    DoctorRecord,
    RankingResult,
    RankingTable,
)
from bba.attribution.ranking import (
    DEFAULT_MIN_ORDERS,
    DEFAULT_TOP_N,
    rank_department_scorecards,
    rank_doctor_scorecards,
)
from bba.attribution.scorecards import (
    build_department_scorecards,
    build_doctor_scorecards,
)


def _bucket_totals(verdicts: Mapping[str, str]) -> BucketTotals:
    """Collapse raw verdicts into the 3-bucket totals used for
    reconciliation (the 300 human labels must land on 162/32/106)."""
    appropriate = sum(1 for c in verdicts.values() if c == "APPROPRIATE")
    inappropriate = sum(1 for c in verdicts.values() if c == "INAPPROPRIATE")
    returned = sum(1 for c in verdicts.values() if c == "RETURNED_NOT_TRANSFUSED")
    periop_exempt = sum(
        1 for c in verdicts.values() if c == "PERIOP_TRANSFUSION_EXEMPT"
    )
    scorable_total = len(verdicts) - returned - periop_exempt
    return BucketTotals(
        appropriate=appropriate,
        inappropriate=inappropriate,
        unresolved=scorable_total - appropriate - inappropriate,
        returned_not_transfused=returned,
        periop_transfusion_exempt=periop_exempt,
        total=scorable_total,
    )


def build_rankings(
    *,
    verdicts: Mapping[str, str],
    reqno_to_doctor: Mapping[str, str],
    dct_registry: Mapping[str, DoctorRecord],
    bucket: Bucket = "inappropriate",
    n: int = DEFAULT_TOP_N,
    min_orders: int = DEFAULT_MIN_ORDERS,
    order_labs: Mapping[str, OrderLabValue] | None = None,
) -> RankingResult:
    """Build both top-N ranking tables plus the reconciliation totals.

    ``bucket`` defaults to ``"inappropriate"`` — the audit's actionable
    metric, consistent with ``inappropriate_rate`` being the ranked
    quantity everywhere else in the report generator. All three bucket
    counts still travel on every row.

    When ``order_labs`` (the per-order lab join from ``report.csv``) is
    supplied, the mean pre-transfusion trigger is aggregated here — over
    the same scorable verdict cohort — and threaded onto both tables' rows;
    omitting it leaves the mean fields at their defaults so existing
    callers are unaffected.
    """
    doctor_stats = (
        aggregate_doctor_lab_stats(verdicts, reqno_to_doctor, order_labs)
        if order_labs is not None
        else None
    )
    department_stats = (
        aggregate_department_lab_stats(
            verdicts, reqno_to_doctor, dct_registry, order_labs
        )
        if order_labs is not None
        else None
    )
    doctors = rank_doctor_scorecards(
        build_doctor_scorecards(verdicts, reqno_to_doctor, dct_registry),
        bucket,
        n=n,
        min_orders=min_orders,
        group_stats=doctor_stats,
    )
    departments = rank_department_scorecards(
        build_department_scorecards(verdicts, reqno_to_doctor, dct_registry),
        bucket,
        n=n,
        min_orders=min_orders,
        group_stats=department_stats,
    )
    return RankingResult(
        doctors=RankingTable(
            dimension="doctor",
            bucket=bucket,
            n=n,
            min_orders=min_orders,
            rows=doctors,
        ),
        departments=RankingTable(
            dimension="department",
            bucket=bucket,
            n=n,
            min_orders=min_orders,
            rows=departments,
        ),
        totals=_bucket_totals(verdicts),
    )
