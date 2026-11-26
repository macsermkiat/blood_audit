"""End-to-end assembly: verdicts + attribution → :class:`RankingResult`.

This is the single function the pilot runner (and, next build, the
full-cohort runner) calls. Swapping the verdict source is the ONLY
change between the two builds — everything below the ``verdicts``
records is verdict-source-agnostic. The records are per issue event
(blood_audit #276); each is credited to the doctor of its linked order.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from bba.attribution.lab_stats import (
    OrderLabValue,
    aggregate_department_lab_stats,
    aggregate_doctor_lab_stats,
    missing_lab_audit_ids,
)
from bba.attribution.models import (
    Bucket,
    BucketTotals,
    DoctorRecord,
    EventVerdict,
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


def _bucket_totals(events: Sequence[EventVerdict]) -> BucketTotals:
    """Collapse raw verdicts into the 3-bucket totals used for
    reconciliation (the 300 human labels must land on 162/32/106)."""
    verdicts = [event.verdict for event in events]
    appropriate = sum(1 for c in verdicts if c == "APPROPRIATE")
    # PREOP_OVER_RESERVATION folds into inappropriate (scorable), mirroring the
    # scorecard counters (#162); without it a passed-through over-reservation
    # would fall into the `unresolved` remainder and the cohort total would
    # disagree with the per-doctor scorecards (Codex PR #168 P2).
    inappropriate = sum(
        1 for c in verdicts if c in ("INAPPROPRIATE", "PREOP_OVER_RESERVATION")
    )
    returned = sum(1 for c in verdicts if c == "RETURNED_NOT_TRANSFUSED")
    periop_exempt = sum(1 for c in verdicts if c == "PERIOP_TRANSFUSION_EXEMPT")
    scorable_total = len(verdicts) - returned - periop_exempt
    return BucketTotals(
        appropriate=appropriate,
        inappropriate=inappropriate,
        unresolved=scorable_total - appropriate - inappropriate,
        returned_not_transfused=returned,
        periop_transfusion_exempt=periop_exempt,
        total=scorable_total,
    )


def _require_unique_audit_ids(verdicts: Sequence[EventVerdict]) -> None:
    """Each event is one decision and counts once: a repeated audit_id means
    a concatenated or double-merged verdict source."""
    repeated = sorted(
        a for a, n in Counter(e.audit_id for e in verdicts).items() if n > 1
    )
    if repeated:
        raise ValueError(
            f"{len(repeated)} audit_id(s) appear more than once in the verdict "
            f"records (e.g. {repeated[:5]}); each issue event must count once"
        )


def build_rankings(
    *,
    verdicts: Sequence[EventVerdict],
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
    _require_unique_audit_ids(verdicts)
    if order_labs is not None:
        # Fail loud before aggregating (guards both the doctor and department
        # paths): a scorable event absent from the lab source means a stale,
        # header-only, or wrong-run report.csv, and a partial join must never
        # be presented as a trigger (spec #131, "fail loudly if the per-order
        # lab source is missing"). A legitimately absent Hb still carries a
        # report row with hb_freshness == "missing".
        missing = missing_lab_audit_ids(verdicts, order_labs)
        if missing:
            raise ValueError(
                f"per-event lab source has no row for {len(missing)} scorable "
                f"audit_id(s) (e.g. {missing[:5]}); a stale, header-only, or "
                "wrong-run report.csv cannot back the mean trigger columns — "
                "refusing to present a partial join. Every scorable event must "
                "have a report row (a genuinely absent value still carries "
                "freshness == 'missing')."
            )
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
