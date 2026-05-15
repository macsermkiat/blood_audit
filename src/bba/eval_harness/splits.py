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

from collections.abc import Sequence

from bba.eval_harness.models import (
    AuditCase,
    SplitStrategy,
    TemporalSplit,
)


def dataset_month_span(cases: Sequence[AuditCase]) -> int:
    """Number of distinct calendar months present in ``cases``.

    A dataset spanning Jan 2026 - Mar 2026 has span 3 (Jan, Feb, Mar), not
    the integer number of weeks. Raises
    :class:`bba.eval_harness.exceptions.EmptyInputError` for an empty input.
    """
    raise NotImplementedError("eval_harness.splits: RED phase, see issue #20")


def select_split_strategy(cases: Sequence[AuditCase]) -> SplitStrategy:
    """Auto-pick LOMO-CV vs blocked temporal split (PRD §11 threshold).

    Returns ``"lomo"`` when the dataset spans strictly fewer than
    :data:`bba.eval_harness.models.LOMO_DATASET_THRESHOLD_MONTHS` months;
    ``"blocked"`` otherwise. The boundary is at the threshold itself
    (12 months → ``"blocked"``) so the design choice is one-sided and not
    susceptible to off-by-one drift between report runs.
    """
    raise NotImplementedError("eval_harness.splits: RED phase, see issue #20")


def lomo_cv_splits(cases: Sequence[AuditCase]) -> tuple[TemporalSplit, ...]:
    """Leave-one-month-out cross-validation splits.

    Returns one :class:`TemporalSplit` per distinct calendar month present
    in the data; the holdout is exactly that month's audit_ids and the train
    set is every other audit_id. Splits are ordered chronologically by
    holdout month (oldest first) so report tables read left-to-right in time.
    """
    raise NotImplementedError("eval_harness.splits: RED phase, see issue #20")


def blocked_temporal_split(
    cases: Sequence[AuditCase], *, n_blocks: int = 4
) -> tuple[TemporalSplit, ...]:
    """Blocked temporal cross-validation splits.

    Sorts cases by ``order_datetime`` and partitions into ``n_blocks``
    contiguous blocks of (roughly) equal size; the holdout for split ``i``
    is block ``i`` and the train set is the remaining blocks. ``n_blocks``
    defaults to 4 (one seasonal quarter per block).
    """
    raise NotImplementedError("eval_harness.splits: RED phase, see issue #20")


def temporal_cv_splits(cases: Sequence[AuditCase]) -> tuple[TemporalSplit, ...]:
    """Auto-pick LOMO or blocked, then return the corresponding split set.

    Convenience wrapper for the typical caller path: it dispatches via
    :func:`select_split_strategy` and returns the corresponding splits. The
    standalone strategy functions remain public so a caller can override
    the auto-pick when running a sensitivity analysis.
    """
    raise NotImplementedError("eval_harness.splits: RED phase, see issue #20")
