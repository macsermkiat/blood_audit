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


def _terminal_status(statuses: Sequence[str]) -> str:
    """The most-final disposition among one physical unit's lifecycle rows."""
    for status in _TERMINAL_PRECEDENCE:
        if status in statuses:
            return status
    for status in statuses:
        if status:
            return status
    return ""


def physical_units(trans_rows: Sequence[Mapping[str, str]]) -> list[str]:
    """Collapse ledger rows to one terminal ``UNITSTAT`` per physical unit.

    Groups rows sharing a non-blank ``(DNRNO, SEQNO)`` — the physical-unit key
    in the complete export — and reduces each group to its terminal status
    (:func:`_terminal_status`). A row with a BLANK ``DNRNO`` is treated as its
    own unit (never collapsed), so identifier-free rows fall back to
    one-row-per-unit counting. Order is preserved for determinism.
    """
    groups: dict[tuple[object, object], list[str]] = {}
    order: list[tuple[object, object]] = []
    for idx, row in enumerate(trans_rows):
        dnrno = str(row.get("DNRNO") or "").strip()
        seqno = str(row.get("SEQNO") or "").strip()
        key: tuple[object, object] = (dnrno, seqno) if dnrno else (None, idx)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(str(row.get("UNITSTAT") or "").strip())
    return [_terminal_status(groups[key]) for key in order]


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
