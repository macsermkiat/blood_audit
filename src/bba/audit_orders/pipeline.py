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
7. (anchor resolution)   — last, because anchor-unrecoverable raises
    rather than silently drops. Age-based ``"pediatric"`` exclusion is
    not in the audit pipeline: post-schema-lock (2026-05-19) the bundle
    has no per-row age column and the upstream IT pre-filter handles
    ``age > 15`` exclusively. See ``docs/ingest-mapping.md``.

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
    Component,
    ExcludedRecord,
    FilterResult,
)
from bba.audit_orders.rules import (
    check_an_scoped,
    check_cancelled,
    check_hemoglobinopathy,
    check_rbc_product,
    check_request_type,
    check_status,
    is_rbc_product,
    platelet_products_in,
    rbc_products_in,
)

# Pre-anchor rules run in this fixed order. The only post-anchor step is
# anchor resolution itself — there is no longer an age gate at this layer
# (per the 2026-05-19 schema lock; see ``docs/ingest-mapping.md``).
_PRE_ANCHOR_RULES: tuple[Callable[[BloodOrderInput], ExcludedRecord | None], ...] = (
    check_rbc_product,
    check_status,
    check_cancelled,
    check_an_scoped,
    check_request_type,
    check_hemoglobinopathy,
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

        # All gates passed — construct the canonical row. an is non-None at
        # this point (check_an_scoped passed); assert to narrow for mypy.
        assert record.an is not None  # noqa: S101 — type narrowing

        # Determine component: if the order contains at least one RBC product it
        # is a red_cell order (mixed RBC+platelet orders fall here too — the
        # platelet codes are stripped by rbc_products_in, matching Phase 1 behavior
        # byte-for-byte). Only a platelet-only order (no RBC, all platelet) becomes
        # component="platelet"; check_rbc_product already admitted it through the
        # gate, so the branch is safe and no re-validation is needed.
        component: Component
        if any(is_rbc_product(p) for p in record.products):
            component = "red_cell"
            products_tuple: tuple[str, ...] = rbc_products_in(record.products)
        else:
            component = "platelet"
            products_tuple = platelet_products_in(record.products)

        included.append(
            AuditOrder(
                audit_id=build_audit_id(record.hn, record.reqno),
                hn=record.hn,
                an=record.an,
                reqno=record.reqno,
                order_datetime=resolved.anchor.utc,
                anchor_imputed=resolved.imputed,
                products_ordered=products_tuple,
                diagnosis_codes=record.diagnosis_codes,
                component=component,
            )
        )

    return FilterResult(included=tuple(included), excluded=tuple(excluded))


__all__ = ("build_audit_orders",)
