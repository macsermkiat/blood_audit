"""Strict non-HOSxP date parser for KCMH-export columns.

The HOSxP standard date format ``2025-06-07 00:00:00.000`` is handled by the
existing pandas/Polars datetime parsers in the ingest pipeline. This module
exists for the **English-locale forms** that appear in KCMH exports for
the procedure-family tables (``IPTSUMOPRT.INDATE`` and ``INCPT.INCDATE``
today; ``ICD9CM.Firstdate`` and ``Lastdate`` are dropped from the schema
but use the same shape):

    "June 7, 2025, 12:00 AM"
    "January 1, 2014, 12:00 AM"
    "January 9, 2025"

The format string is shared with the long-form branch of
:func:`bba.ingest.time_parser.parse_hosxp_time`, which extracts only the
**time** component for ``OPDATETIME`` columns and intentionally discards the
date. :func:`parse_iptsumoprt_date` is the inverse: it extracts only the
**date** component and intentionally discards the embedded time, because
``IPTSUMOPRT`` carries its time in the separate ``INTIME`` column (HOSxP
``HHMMSS`` int format, handled by ``parse_hosxp_time``). ``INCPT`` follows
the same split, with ``INCDATE`` and ``INCTIME``.

NEVER silently shifts: an unrecognized format yields ``value=None`` with a
populated ``parse_warning`` — same contract as ``parse_hosxp_time`` (PRD §1,
Round 2 fix E35). Calendar validation rejects impossible dates such as
``"June 31, 2025"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from bba.ingest.time_parser import _LONG_FORM_RE, _MONTH_NUMBER


_DATE_ONLY_RE: re.Pattern[str] = re.compile(
    r"^(?P<month>[A-Z][a-z]+) "
    r"(?P<day>\d{1,2}), "
    r"(?P<year>\d{4})$"
)


@dataclass(frozen=True, slots=True)
class DateParseResult:
    """Result of strict English-locale date parsing.

    Invariant: exactly one of ``value`` and ``parse_warning`` is non-None.
    The ``raw`` field preserves the input for downstream logging / drift
    reporting.

    Mirrors :class:`bba.ingest.ParseResult` for the date side. Kept as a
    dataclass (not a Pydantic model) because this parser is internal to the
    normalize layer — it is not part of the public ``bba.ingest`` surface that
    downstream tickets consume.
    """

    value: date | None
    parse_warning: str | None
    raw: str


def _date_result(raw: str, month_name: str, day: int, year: int) -> DateParseResult:
    if month_name not in _MONTH_NUMBER:
        return DateParseResult(
            value=None,
            parse_warning=f"long-form: unknown month name {month_name!r}",
            raw=raw,
        )

    try:
        d = date(year, _MONTH_NUMBER[month_name], day)
    except ValueError as exc:
        return DateParseResult(
            value=None,
            parse_warning=f"long-form: invalid calendar date ({exc})",
            raw=raw,
        )
    return DateParseResult(value=d, parse_warning=None, raw=raw)


def parse_kcmh_english_date(raw: str | None) -> DateParseResult:
    """Parse a KCMH English-locale date string against the strict allow-list.

    Accepts ``"Month Day, Year, H:MM AM/PM"`` and ``"Month Day, Year"``.
    The embedded time component, when present, is matched for shape but its
    value is discarded because procedure-family exports carry time-of-day
    in a sibling time column.

    Returns a frozen :class:`DateParseResult`. Unrecognized formats, calendar-
    invalid dates, empty strings, ``None``, and HOSxP-shaped inputs all yield
    ``value=None`` with a descriptive ``parse_warning``.
    """
    if raw is None:
        return DateParseResult(
            value=None, parse_warning="empty: input was None", raw=""
        )
    if raw == "":
        return DateParseResult(value=None, parse_warning="empty: empty string", raw=raw)

    match = _LONG_FORM_RE.match(raw) or _DATE_ONLY_RE.match(raw)
    if match is None:
        return DateParseResult(
            value=None,
            parse_warning=f"unrecognized: not English-locale date ({raw!r})",
            raw=raw,
        )

    return _date_result(
        raw,
        month_name=match.group("month"),
        day=int(match.group("day")),
        year=int(match.group("year")),
    )


def parse_iptsumoprt_date(raw: str | None) -> DateParseResult:
    """Backward-compatible alias for procedure-family English-date parsing."""
    return parse_kcmh_english_date(raw)
