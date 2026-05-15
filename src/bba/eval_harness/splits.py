"""Temporal cross-validation splits: LOMO-CV and blocked temporal (PRD §11).

When the dataset spans less than 12 months, leave-one-month-out CV gives
finer-grained generalization estimates per held-out month; once the dataset
has at least 12 months of history, the seasonal-confounder horizon is
already represented, and blocked temporal splits give a more stable estimate
with less computational cost. The split strategy is therefore *auto-selected*
from the data — the harness does not expect the caller to know which to use.

Both strategies split by ``order_datetime`` truncated to the calendar month
(UTC, naive-datetime-banned upstream). The output uses ``audit_id`` so the
downstream caller can re-key into its own data structure.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from bba.eval_harness.exceptions import EmptyInputError
from bba.eval_harness.models import (
    LOMO_DATASET_THRESHOLD_MONTHS,
    AuditCase,
    SplitStrategy,
    TemporalSplit,
)


def _month_tag(case: AuditCase) -> str:
    return f"{case.order_datetime.year:04d}-{case.order_datetime.month:02d}"


def dataset_month_span(cases: Sequence[AuditCase]) -> int:
    """Inclusive calendar-month span between the earliest and latest case.

    The span is the *temporal range* a dataset covers, not the count of
    distinct months that happen to contain observations (codex P0 finding).
    A dataset with one case in Jan 2026 and one in Dec 2026 has span 12
    even though only 2 months have data — the seasonal-confounder horizon
    the PRD §11 threshold reasons about is the calendar range, not the
    observed-month density.
    """
    if not cases:
        raise EmptyInputError("dataset_month_span: cases must be non-empty")
    months = sorted(
        {(c.order_datetime.year, c.order_datetime.month) for c in cases}
    )
    first_year, first_month = months[0]
    last_year, last_month = months[-1]
    return (last_year - first_year) * 12 + (last_month - first_month) + 1


def select_split_strategy(cases: Sequence[AuditCase]) -> SplitStrategy:
    """Auto-pick LOMO-CV vs blocked temporal split (PRD §11 threshold)."""
    return (
        "lomo"
        if dataset_month_span(cases) < LOMO_DATASET_THRESHOLD_MONTHS
        else "blocked"
    )


def lomo_cv_splits(cases: Sequence[AuditCase]) -> tuple[TemporalSplit, ...]:
    """Leave-one-month-out cross-validation splits."""
    if not cases:
        raise EmptyInputError("lomo_cv_splits: cases must be non-empty")
    by_month: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        by_month[_month_tag(case)].append(case.audit_id)
    splits: list[TemporalSplit] = []
    sorted_months = sorted(by_month.keys())
    all_ids = [c.audit_id for c in cases]
    for month in sorted_months:
        holdout_set = set(by_month[month])
        train_ids = tuple(aid for aid in all_ids if aid not in holdout_set)
        splits.append(
            TemporalSplit(
                train_audit_ids=train_ids,
                holdout_audit_ids=tuple(by_month[month]),
                holdout_label=month,
            )
        )
    return tuple(splits)


def blocked_temporal_split(
    cases: Sequence[AuditCase], *, n_blocks: int = 4
) -> tuple[TemporalSplit, ...]:
    """Blocked temporal cross-validation splits."""
    if not cases:
        raise EmptyInputError("blocked_temporal_split: cases must be non-empty")
    if n_blocks < 1:
        raise ValueError(f"blocked_temporal_split: n_blocks must be >= 1, got {n_blocks}")
    ordered = sorted(
        cases, key=lambda c: (c.order_datetime, c.audit_id)
    )
    n = len(ordered)
    # When fewer cases than requested blocks (e.g., sparse-span dataset that
    # auto-routed to blocked but only contains a handful of orders), cap
    # n_blocks at the case count — otherwise later splits emit empty
    # holdouts which corrupt downstream per-fold metrics (codex P1 round 2).
    n_blocks = min(n_blocks, n)
    # Roughly equal contiguous blocks; remainder distributed to early blocks.
    base_size, remainder = divmod(n, n_blocks)
    splits: list[TemporalSplit] = []
    cursor = 0
    all_ids = [c.audit_id for c in ordered]
    for i in range(n_blocks):
        block_size = base_size + (1 if i < remainder else 0)
        block_ids = all_ids[cursor : cursor + block_size]
        cursor += block_size
        holdout_set = set(block_ids)
        train_ids = tuple(aid for aid in all_ids if aid not in holdout_set)
        splits.append(
            TemporalSplit(
                train_audit_ids=train_ids,
                holdout_audit_ids=tuple(block_ids),
                holdout_label=f"block-{i + 1}",
            )
        )
    return tuple(splits)


def temporal_cv_splits(cases: Sequence[AuditCase]) -> tuple[TemporalSplit, ...]:
    """Auto-pick LOMO or blocked, then return the corresponding split set."""
    strategy = select_split_strategy(cases)
    if strategy == "lomo":
        return lomo_cv_splits(cases)
    return blocked_temporal_split(cases)
