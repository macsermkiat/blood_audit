"""Operations that let a fresh order's declared surgery exempt its event.

Owner decision 8.15 (c), amended 2026-11-22 (blood_audit #276): an operation
is an IPTSUMOPRT record, an INCPT surgery-fee charge (INCGRP 111) or an INCPT
general-anaesthesia charge (AS056-AS059). The GA code is read from O__OPRTACT
only: ORDERCODE holds the billing code and never carries it. Bedside
procedures and add-on charges alone do not count.

One definition for ``run_pipeline.py`` and ``issue_events_gold_check.py``.
Rows come in normalized by ``run_pipeline._normalize_iptsumoprt`` and
``_normalize_incpt``; times are read as ``run_pipeline._build_op_events``
reads them, including its midnight placement of a blank or unreadable time,
which ``time_imputed`` records.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime

from bba.ingest.models import ParsedTimeOfDay

from _hosxp_dt import _combine, _parse_time

SURGERY_FEE_GROUP = "111"
GENERAL_ANAESTHESIA = frozenset({"AS056", "AS057", "AS058", "AS059"})
_MIDNIGHT = ParsedTimeOfDay(hour=0, minute=0, second=0)


@dataclass(frozen=True)
class Operation:
    """One operation on an admission, at a UTC time."""

    at: datetime
    time_imputed: bool


def is_operation_charge(row: Mapping[str, object]) -> bool:
    """An INCPT charge that counts as an operation under 8.15 (c)."""
    if _cell(row, "CANCELDATE"):
        return False
    return (
        _cell(row, "INCGRP") == SURGERY_FEE_GROUP
        or _cell(row, "O__OPRTACT") in GENERAL_ANAESTHESIA
    )


def operations_by_an(
    iptsumoprt: Iterable[Mapping[str, object]],
    incpt: Iterable[Mapping[str, object]],
) -> dict[str, list[Operation]]:
    """Every operation per admission; rows with no readable date are skipped."""
    out: dict[str, list[Operation]] = defaultdict(list)
    for row in iptsumoprt:
        operation = _operation(row.get("INDATE"), row.get("INTIME"))
        if operation is not None:
            out[_cell(row, "AN")].append(operation)
    for row in incpt:
        if not is_operation_charge(row):
            continue
        operation = _operation(row.get("INCDATE"), row.get("INCTIME"))
        if operation is not None:
            out[_cell(row, "AN")].append(operation)
    return dict(out)


def _operation(day_raw: object, time_raw: object) -> Operation | None:
    if isinstance(day_raw, date):
        day: date = day_raw
    else:
        try:
            day = date.fromisoformat(str(day_raw or "").strip()[:10])
        except ValueError:
            return None
    time = _parse_time(str(time_raw or "").strip())
    at = _combine(day, time or _MIDNIGHT)
    return None if at is None else Operation(at=at, time_imputed=time is None)


def _cell(row: Mapping[str, object], key: str) -> str:
    return str(row.get(key) or "").strip()
