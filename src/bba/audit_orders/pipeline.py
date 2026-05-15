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

from collections.abc import Callable, Sequence

from bba.audit_orders.anchor import resolve_anchor
from bba.audit_orders.exceptions import UnrecoverableAnchorError
from bba.audit_orders.identity import build_audit_id
from bba.audit_orders.models import (
    AuditOrder,
    AuditOrdersConfig,
    BloodOrderInput,
    ExcludedRecord,
    FilterResult,
)
from bba.audit_orders.rules import (
    check_age,
    check_aiha,
    check_an_scoped,
    check_cancelled,
    check_hemoglobinopathy,
    check_obstetric,
    check_rbc_product,
    check_request_type,
    check_status,
    check_tma,
    rbc_products_in,
    years_between,
)

# Pre-anchor rules run in this fixed order. Age is intentionally NOT here:
# it depends on the resolved anchor's local date and runs after anchor
# resolution.
_PRE_ANCHOR_RULES: tuple[Callable[[BloodOrderInput], ExcludedRecord | None], ...] = (
    check_rbc_product,
    check_status,
    check_cancelled,
    check_an_scoped,
    check_request_type,
    check_hemoglobinopathy,
    check_aiha,
    check_tma,
    check_obstetric,
)


def _first_exclusion(record: BloodOrderInput) -> ExcludedRecord | None:
    """Apply pre-anchor rules in fixed order; return the first exclusion or None."""
    for rule in _PRE_ANCHOR_RULES:
        excluded = rule(record)
        if excluded is not None:
            return excluded
    return None


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
    included: list[AuditOrder] = []
    excluded: list[ExcludedRecord] = []

    for record in records:
        pre_anchor = _first_exclusion(record)
        if pre_anchor is not None:
            excluded.append(pre_anchor)
            continue

        resolved = resolve_anchor(record, config.tz_source)
        if resolved.anchor is None or resolved.local_date is None:
            raise UnrecoverableAnchorError(
                f"record (hn={record.hn!r}, reqno={record.reqno!r}) has neither "
                f"REQ nor BDVST anchor pair; cannot derive order_datetime"
            )

        # Age depends on the resolved local date (Bangkok wall-clock day
        # of the order, not UTC date — they can disagree near midnight).
        age_exclusion = check_age(record, resolved.local_date)
        if age_exclusion is not None:
            excluded.append(age_exclusion)
            continue

        # All gates passed — construct the canonical row.
        # an + birthdate are non-None at this point (check_an_scoped +
        # check_age passed); assert to narrow for mypy.
        assert record.an is not None  # noqa: S101 — type narrowing
        assert record.birthdate is not None  # noqa: S101 — type narrowing
        included.append(
            AuditOrder(
                audit_id=build_audit_id(record.hn, record.reqno),
                hn=record.hn,
                an=record.an,
                reqno=record.reqno,
                order_datetime=resolved.anchor.utc,
                anchor_imputed=resolved.imputed,
                products_ordered=rbc_products_in(record.products),
                age_years=years_between(record.birthdate, resolved.local_date),
                sex=record.sex,
                diagnosis_codes=record.diagnosis_codes,
            )
        )

    return FilterResult(included=tuple(included), excluded=tuple(excluded))


__all__ = ("build_audit_orders",)
