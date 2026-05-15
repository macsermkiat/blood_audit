"""End-to-end audit_orders filter: BloodOrderInput sequence → FilterResult.

Public entry point: :func:`build_audit_orders`. Applies the inclusion /
exclusion rules in a fixed order and partitions every input record into
either an :class:`~bba.audit_orders.models.AuditOrder` or an
:class:`~bba.audit_orders.models.ExcludedRecord` with a typed reason.

Rule ordering (cheap first, semantic last) — the first matching exclusion
wins so the typed reason matches the most specific gate that fired:

1. ``check_rbc_product``  — wrong product family
2. ``check_status``       — BDVSTST not in {4, 5} (covers refused = 6)
3. ``check_cancelled``    — CANCELDATE non-null
4. ``check_an_scoped``    — OPD (no AN)
5. ``check_request_type`` — inter-hospital referral (REQTYPE != 'P')
6. ``check_hemoglobinopathy`` — D55/D56/D57/D58
7. ``check_aiha``         — D59.x
8. ``check_tma``          — M31.1
9. ``check_obstetric``    — O-chapter
10. (anchor resolution + age) — last, because age depends on the resolved
    anchor date; anchor-unrecoverable raises rather than silently drops.

Per PRD §"Testing Decisions" the test suite asserts the partition invariant
(every input lands in exactly one bucket), not the exact rule firing order.
The order above is the conservative reading and any future re-ordering must
not change which records land in which bucket.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.audit_orders.models import (
    AuditOrdersConfig,
    BloodOrderInput,
    FilterResult,
)


def build_audit_orders(
    records: Sequence[BloodOrderInput],
    config: AuditOrdersConfig,
) -> FilterResult:
    """Apply the audit_orders filter to ``records`` and return the partition.

    Contract:

    * Every input record appears in exactly one of ``result.included`` or
      ``result.excluded`` (no silent drops; no double-counting).
    * ``included`` and ``excluded`` preserve the input ordering of their
      respective subsets.
    * Re-running with the same input produces the same ``audit_id`` for
      every included record (identity determinism, asserted by the
      property tests).
    * Raises :class:`bba.audit_orders.exceptions.UnrecoverableAnchorError`
      if a record passes every other gate but neither anchor pair is
      usable. Per PRD §"Output schema", a row without an anchor is a bug
      to surface, not a silent omission.
    """
    raise NotImplementedError


__all__ = ("build_audit_orders",)
