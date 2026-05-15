"""k-anonymity gate for the deid_redactor.

PRD §8: bundles with quasi-identifier combos ``k < 5`` are routed to
NEEDS_REVIEW (not silently dropped). Two public entry points:

* :func:`compute_k_groups` — pure utility the audit-pipeline caller uses to
  precompute group sizes across the full population (a month's worth of
  orders).
* :func:`k_anonymity_passed` — single-record decision; the wrapper composes
  it with the result of :func:`compute_k_groups`.

The gate itself enters :func:`bba.deid_redactor.redactor.redact_bundle`
via the :class:`bba.deid_redactor.models.KAnonymityGate` Protocol, which
is just ``Callable[[QuasiIdentifiers], int]``. Keeping it a Protocol means
tests can supply a fixed-group-size lambda without constructing a full
population.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from bba.deid_redactor.models import K_ANONYMITY_MIN, QuasiIdentifiers


def compute_k_groups(
    records: Sequence[QuasiIdentifiers],
) -> Mapping[QuasiIdentifiers, int]:
    """Count occurrences of every quasi-identifier tuple in ``records``.

    Returns a read-only mapping ``QuasiIdentifiers → group size``. The
    caller adapts this to the :class:`KAnonymityGate` Protocol with
    ``groups.get(qi, 0)`` so a record whose QI tuple is absent from the
    precomputed population is treated as group-size-zero (the maximally
    conservative interpretation).

    Pure function: same input → same mapping. The order of records does
    NOT affect output.
    """
    counter = _count_qi(records)
    return MappingProxyType(dict(counter))


def k_anonymity_passed(group_size: int, *, k: int = K_ANONYMITY_MIN) -> bool:
    """Single-record decision: does this group meet the k threshold?

    PRD §8: the threshold is ``k ≥ 5``. Below 5 → fail → route to
    NEEDS_REVIEW. The default ``k`` is the module constant
    :data:`bba.deid_redactor.models.K_ANONYMITY_MIN`; callers can override
    for a stricter (k=10) policy if KCMH compliance later raises the bar.
    """
    return group_size >= k


def _count_qi(records: Sequence[QuasiIdentifiers]) -> Counter[QuasiIdentifiers]:
    """Implementation seam — :class:`Counter`-backed groupby on QI tuples."""
    return Counter(records)
