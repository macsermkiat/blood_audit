"""End-to-end assembly: verdicts + attribution → :class:`RankingResult`.

This is the single function the pilot runner (and, next build, the
full-cohort runner) calls. Swapping the verdict source is the ONLY
change between the two builds — everything below the ``verdicts``
mapping is verdict-source-agnostic.
"""

from __future__ import annotations

from collections.abc import Mapping

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
    scorable_total = len(verdicts) - returned
    return BucketTotals(
        appropriate=appropriate,
        inappropriate=inappropriate,
        unresolved=scorable_total - appropriate - inappropriate,
        returned_not_transfused=returned,
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
) -> RankingResult:
    """Build both top-N ranking tables plus the reconciliation totals.

    ``bucket`` defaults to ``"inappropriate"`` — the audit's actionable
    metric, consistent with ``inappropriate_rate`` being the ranked
    quantity everywhere else in the report generator. All three bucket
    counts still travel on every row.
    """
    doctors = rank_doctor_scorecards(
        build_doctor_scorecards(verdicts, reqno_to_doctor, dct_registry),
        bucket,
        n=n,
        min_orders=min_orders,
    )
    departments = rank_department_scorecards(
        build_department_scorecards(verdicts, reqno_to_doctor, dct_registry),
        bucket,
        n=n,
        min_orders=min_orders,
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
