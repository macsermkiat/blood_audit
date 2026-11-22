"""Parse one REQNO's BDVSTTRANS rows into issued physical units (plan 1.1).

This is the boundary of the package: raw ledger rows (UPPERCASE keys, HOSxP
date and integer-like time text) in, typed units out. It reuses the returns
ledger's admission scoping and physical-unit key so both read the same units.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime

from bba.ingest.row_timestamp import RowTimestamp
from bba.ingest.time_parser import parse_hosxp_time
from bba.issue_events.models import IssuedUnit, LedgerUnits
from bba.returns_ledger import physical_unit_rows, rows_for_admission, terminal_status


def units_from_ledger(trans_rows: Sequence[Mapping[str, str]], an: str) -> LedgerUnits:
    """Collapse a REQNO's ledger rows on admission ``an`` to physical units.

    Rows on another admission are dropped. A unit with no parseable
    PAYDATE/PAYTIME on any of its rows was never issued and is only counted.
    """
    if not an:
        raise ValueError(
            "units_from_ledger needs the order's AN to scope the ledger rows"
        )
    issued: list[IssuedUnit] = []
    reserved = 0
    for rows in physical_unit_rows(rows_for_admission(trans_rows, an)):
        issue_at = _earliest(rows, "PAYDATE", "PAYTIME")
        if issue_at is None:
            reserved += 1
            continue
        first = rows[0]
        issued.append(
            IssuedUnit(
                dnrno=_cell(first, "DNRNO"),
                seqno=_cell(first, "SEQNO"),
                bdtype=_cell(first, "BDTYPE"),
                status=terminal_status([_cell(row, "UNITSTAT") for row in rows]),
                issue_at=issue_at,
                returned_at=_earliest(rows, "RTNDATE", "RTNTIME"),
                given_at=_earliest(rows, "GIVEDATE", "GIVETIME"),
            )
        )
    return LedgerUnits(issued=tuple(issued), units_reserved_not_issued=reserved)


def _cell(row: Mapping[str, str], key: str) -> str:
    return str(row.get(key) or "").strip()


def _earliest(
    rows: Sequence[Mapping[str, str]], date_key: str, time_key: str
) -> datetime | None:
    moments = [
        moment
        for row in rows
        if (moment := _hosxp_datetime(_cell(row, date_key), _cell(row, time_key)))
        is not None
    ]
    return min(moments, default=None)


def _hosxp_datetime(date_raw: str, time_raw: str) -> datetime | None:
    """Bangkok wall-clock date + integer-like time as UTC, as the pilot reads them."""
    try:
        day = date.fromisoformat(date_raw.split(" ", 1)[0])
    except ValueError:
        return None
    if time_raw.isdigit() and len(time_raw) <= 6:
        time_raw = time_raw.zfill(6)
    time = parse_hosxp_time(time_raw).value
    return None if time is None else RowTimestamp.from_parts(day, time).utc
