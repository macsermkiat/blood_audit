"""Pure aggregation of BDVSTTRANS rows into a :class:`ReturnsSummary` (spec #119).

``summarize_returns`` is a deterministic function of one order's ledger rows and
its ordered-unit-quantity lines. It performs no I/O and never calls a model.

Completeness fails CLOSED (spec #119, decision 5 + Codex completeness finding):
any blank, non-numeric, zero, or negative ``UNITAMT`` line — or the absence of
any line at all — makes the whole order ``inconclusive``. The ordered amount is
never computed from only the parseable lines, because a reduced threshold could
let a partially-exported ledger read as complete. ``ledger_complete`` means the
ledger COVERS the order (``units_total >= ordered_unit_amount``); the stricter
EXACT-count requirement that guards the not-transfused screen against a reissue
lives in :meth:`ReturnsSummary.disposition`, not here (spec #119 NARROW).
"""

from __future__ import annotations

from bba.returns_ledger.models import ReturnsSummary

_RETURNED_STATUS = "3"
_TRANSFUSED_STATUS = "5"


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
        BDVSTTRANS rows for a single REQNO with UPPERCASE keys (each row is one
        dispensed physical unit). Only ``UNITSTAT`` is read here.
    unitamt_lines:
        Raw ``UNITAMT`` strings from the order's BDVSTDT detail lines.
    """
    units_total = len(trans_rows)
    units_returned = sum(
        1
        for r in trans_rows
        if str(r.get("UNITSTAT") or "").strip() == _RETURNED_STATUS
    )
    units_transfused = sum(
        1
        for r in trans_rows
        if str(r.get("UNITSTAT") or "").strip() == _TRANSFUSED_STATUS
    )

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
        ordered_unit_amount=ordered_unit_amount,
        ledger_complete=ledger_complete,
    )
