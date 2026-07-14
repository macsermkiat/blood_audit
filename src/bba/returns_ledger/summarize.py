"""Pure aggregation of BDVSTTRANS rows into a :class:`ReturnsSummary` (spec #119).

``summarize_returns`` is a deterministic function of one order's ledger rows and
its ordered-unit-quantity lines. It performs no I/O and never calls a model.

The complete production export records a physical unit's lifecycle as MULTIPLE
rows (a dispense row plus a later return row, or split aliquots of one donor
bag), keyed apart by ``SEQNO``. So the rows are first collapsed to one unit per
``(DNRNO, SEQNO)`` at its terminal status (:func:`physical_units`) before
counting — ``units_total`` is a physical-unit count, not a row count.

Completeness fails CLOSED (spec #119, decision 5 + Codex completeness finding):
any blank, non-numeric, zero, or negative ``UNITAMT`` line — or the absence of
any line at all — makes the whole order ``inconclusive``. The ordered amount is
never computed from only the parseable lines, because a reduced threshold could
let a partially-exported ledger read as complete. ``ledger_complete`` means the
ledger COVERS the order (``units_total >= ordered_unit_amount``); the earlier
NARROW exact-count guard is relaxed for the complete-ledger go-live (see
:meth:`ReturnsSummary.disposition`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from bba.returns_ledger.models import ReturnsSummary

_RETURNED_STATUS = "3"
_TRANSFUSED_STATUS = "5"
_INCOMPAT_STATUS = "7"

# Terminal disposition precedence when one physical unit carries several
# lifecycle rows. Transfused wins so a given unit is never downgraded to
# returned (fail safe — a false "not transfused" would clear a real
# transfusion); then returned, then incompatible, then dispensed. Any other
# status falls back to first-seen.
_TERMINAL_PRECEDENCE = (_TRANSFUSED_STATUS, _RETURNED_STATUS, _INCOMPAT_STATUS, "2")


def terminal_status(statuses: Sequence[str]) -> str:
    """The most-final disposition among one physical unit's lifecycle rows.

    Precedence :data:`_TERMINAL_PRECEDENCE` (transfused > returned > incompatible
    > dispensed); any other status falls back to first-seen. Public so callers
    that dedup ledger rows themselves (e.g. the pre-flight's sibling attribution)
    reuse the exact same rule instead of re-deriving it.
    """
    for status in _TERMINAL_PRECEDENCE:
        if status in statuses:
            return status
    for status in statuses:
        if status:
            return status
    return ""


def rows_for_admission(
    trans_rows: Sequence[Mapping[str, str]], an: str | None
) -> list[dict[str, str]]:
    """Restrict a REQNO's ledger rows to one admission's ``AN``.

    A REQNO can recur across admissions in the complete export, so a REQNO-only
    lookup can feed a foreign admission's units into this order's disposition —
    foreign returned/incompatible rows could even derive ``not_transfused`` for
    an order whose own units were transfused. Every consumer that summarizes an
    order's ledger rows (pre-flight, both pilot legs, production wiring) MUST
    scope through this first so they stay in lockstep. When the order has no
    ``an`` the rows are returned unscoped (cannot scope); audited orders always
    carry an AN, so this only fails open on a malformed order.
    """
    if not an:
        return [dict(r) for r in trans_rows]
    return [dict(r) for r in trans_rows if str(r.get("AN") or "").strip() == an]


def physical_units(trans_rows: Sequence[Mapping[str, str]]) -> list[str]:
    """Collapse ledger rows to one terminal ``UNITSTAT`` per physical unit.

    Groups rows sharing a non-blank ``(DNRNO, SEQNO, BDTYPE)`` — the
    physical-unit key in the complete export — and reduces each group to its
    terminal status (:func:`terminal_status`). ``BDTYPE`` is part of the key
    because ``(DNRNO, SEQNO)`` alone collides across distinct products of one
    donation (e.g. split ``SDRF``/``SDRF2`` units, or a pooled ``LDPC1..4``) and
    across a unit's irradiation relabel (``X`` vs ``X+I``); keying on ``BDTYPE``
    keeps genuinely-distinct units apart so a presumed-transfused unit can never
    be masked behind another product's return (a false ``not_transfused``). Only
    true same-item lifecycle rows (identical DNRNO/SEQNO/BDTYPE, e.g. a
    dispense + a later return) collapse.

    Collapsing requires ALL THREE key components present: a row missing any of
    ``DNRNO``/``SEQNO``/``BDTYPE`` is treated as its own physical unit (never
    collapsed), so a partial key can never group a dispensed row with a returned
    one and mask the transfusion (fail closed). The real export fills all three
    on every row; this only guards a malformed line. Order is preserved for
    determinism.
    """
    groups: dict[tuple[object, ...], list[str]] = {}
    order: list[tuple[object, ...]] = []
    for idx, row in enumerate(trans_rows):
        dnrno = str(row.get("DNRNO") or "").strip()
        seqno = str(row.get("SEQNO") or "").strip()
        bdtype = str(row.get("BDTYPE") or "").strip()
        # Only a complete physical-unit key collapses rows; any missing
        # component falls back to a per-row identifier so distinct rows are
        # never merged behind a partial key.
        key: tuple[object, ...] = (
            (dnrno, seqno, bdtype) if (dnrno and seqno and bdtype) else (None, idx)
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(str(row.get("UNITSTAT") or "").strip())
    return [terminal_status(groups[key]) for key in order]


def _parse_ordered_amount(unitamt_lines: list[str]) -> int | None:
    """Sum the ordered unit amount, or ``None`` if any line is invalid.

    A line is valid only when it is a non-empty base-10 integer strictly
    greater than zero. An empty ``unitamt_lines`` list is invalid: the ordered
    quantity is unknown, so the order cannot be judged complete.
    """
    if not unitamt_lines:
        return None
    total = 0
    for raw in unitamt_lines:
        text = (raw or "").strip()
        if not text:
            return None
        try:
            value = int(text)
        except ValueError:
            return None
        if value <= 0:
            return None
        total += value
    return total


def summarize_returns(
    trans_rows: list[dict[str, str]],
    unitamt_lines: list[str],
) -> ReturnsSummary:
    """Aggregate one REQNO's ledger rows into a :class:`ReturnsSummary`.

    Parameters
    ----------
    trans_rows:
        BDVSTTRANS rows for a single REQNO with UPPERCASE keys. Lifecycle rows
        of one physical unit share ``(DNRNO, SEQNO)``; ``UNITSTAT``, ``DNRNO``
        and ``SEQNO`` are read here.
    unitamt_lines:
        Raw ``UNITAMT`` strings from the order's BDVSTDT detail lines.
    """
    units = physical_units(trans_rows)
    units_total = len(units)
    units_returned = sum(1 for status in units if status == _RETURNED_STATUS)
    units_transfused = sum(1 for status in units if status == _TRANSFUSED_STATUS)
    units_incompat = sum(1 for status in units if status == _INCOMPAT_STATUS)

    ordered_unit_amount = _parse_ordered_amount(unitamt_lines)
    ledger_complete = (
        units_total > 0
        and ordered_unit_amount is not None
        and units_total >= ordered_unit_amount
    )

    return ReturnsSummary(
        units_total=units_total,
        units_returned=units_returned,
        units_transfused=units_transfused,
        units_incompat=units_incompat,
        ordered_unit_amount=ordered_unit_amount,
        ledger_complete=ledger_complete,
    )
