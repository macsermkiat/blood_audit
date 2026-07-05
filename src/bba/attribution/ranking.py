"""Pure top-N ranking over classification scorecards.

The thin-sample rule (feature plan, "Caveats to surface"): with only
300 human-labelled orders, most doctors have 1–3 orders, and a
1/1 = 100% rate must never be presented as a "top" rate. Groups meeting
``min_orders`` rank first, by bucket *rate*; groups below the threshold
follow, ranked by bucket *count*, and every row carries
``meets_min_orders`` so output writers can flag them.

``DEFAULT_MIN_ORDERS`` was frozen before scoring the 300 (overfit guard
from the plan); changing it is a clinical-review decision.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

from bba.attribution.models import Bucket, RankedRow
from bba.dashboard.models import PhysicianScorecard, WardScorecard


DEFAULT_TOP_N = 10
DEFAULT_MIN_ORDERS = 5


class SupportsClassificationCounts(Protocol):
    """The count surface shared by the dashboard's scorecard models."""

    total_orders: int
    appropriate_count: int
    inappropriate_count: int
    needs_review_count: int
    insufficient_evidence_count: int


_S = TypeVar("_S", bound=SupportsClassificationCounts)


def _unresolved_count(card: SupportsClassificationCounts) -> int:
    return card.needs_review_count + card.insufficient_evidence_count


def _bucket_count(card: SupportsClassificationCounts, bucket: Bucket) -> int:
    if bucket == "appropriate":
        return card.appropriate_count
    if bucket == "inappropriate":
        return card.inappropriate_count
    return _unresolved_count(card)


def _rate(numerator: int, denominator: int) -> float:
    """Zero-volume guard, mirroring
    :func:`bba.report_generator.aggregate._rate`."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def rank_top_n(
    scorecards: Sequence[_S],
    bucket: Bucket,
    *,
    group_id: Callable[[_S], str],
    group_name: Callable[[_S], str],
    n: int = DEFAULT_TOP_N,
    min_orders: int = DEFAULT_MIN_ORDERS,
) -> tuple[RankedRow, ...]:
    """Rank ``scorecards`` by ``bucket`` and return the top ``n`` rows.

    Groups with zero orders in the ranked bucket are excluded outright:
    a doctor with no inappropriate orders cannot appear in the "top
    inappropriate" table — on the real 300 the only two threshold-
    qualified doctors are inappropriate-free and would otherwise head
    the table above every actual finding.

    Ordering among the bucket-positive groups: qualified groups
    (``total_orders >= min_orders``) first, sorted by bucket rate
    descending; unqualified groups follow, sorted by bucket count
    descending. Ties break by bucket count, then total orders (both
    descending), then ``group_id`` ascending, so the output is
    deterministic for byte-stable artifacts.
    """
    candidates = [c for c in scorecards if _bucket_count(c, bucket) > 0]
    qualified = [c for c in candidates if c.total_orders >= min_orders]
    unqualified = [c for c in candidates if c.total_orders < min_orders]

    qualified.sort(
        key=lambda c: (
            -_rate(_bucket_count(c, bucket), c.total_orders),
            -_bucket_count(c, bucket),
            -c.total_orders,
            group_id(c),
        )
    )
    unqualified.sort(
        key=lambda c: (
            -_bucket_count(c, bucket),
            -c.total_orders,
            group_id(c),
        )
    )

    rows: list[RankedRow] = []
    for rank, card in enumerate((qualified + unqualified)[:n], start=1):
        count = _bucket_count(card, bucket)
        rows.append(
            RankedRow(
                rank=rank,
                group_id=group_id(card),
                group_name=group_name(card),
                total_orders=card.total_orders,
                appropriate_count=card.appropriate_count,
                inappropriate_count=card.inappropriate_count,
                unresolved_count=_unresolved_count(card),
                bucket=bucket,
                bucket_count=count,
                bucket_rate=_rate(count, card.total_orders),
                meets_min_orders=card.total_orders >= min_orders,
            )
        )
    return tuple(rows)


def rank_doctor_scorecards(
    scorecards: Sequence[PhysicianScorecard],
    bucket: Bucket,
    *,
    n: int = DEFAULT_TOP_N,
    min_orders: int = DEFAULT_MIN_ORDERS,
) -> tuple[RankedRow, ...]:
    """Doctor-dimension convenience wrapper over :func:`rank_top_n`."""
    return rank_top_n(
        scorecards,
        bucket,
        group_id=lambda c: c.physician_id,
        group_name=lambda c: c.physician_name,
        n=n,
        min_orders=min_orders,
    )


def rank_department_scorecards(
    scorecards: Sequence[WardScorecard],
    bucket: Bucket,
    *,
    n: int = DEFAULT_TOP_N,
    min_orders: int = DEFAULT_MIN_ORDERS,
) -> tuple[RankedRow, ...]:
    """Department-dimension convenience wrapper over :func:`rank_top_n`."""
    return rank_top_n(
        scorecards,
        bucket,
        group_id=lambda c: c.ward_id,
        group_name=lambda c: c.ward_name,
        n=n,
        min_orders=min_orders,
    )
