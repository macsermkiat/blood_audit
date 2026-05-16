"""Strict HOSxP time-of-day parser.

Allow-list of accepted formats:

* ``HHMMSS`` — 6 zero-padded digits (e.g., ``"083045"``).
* ``HH:MM`` — 5 chars with a literal colon (e.g., ``"08:30"``).
* ``"Month Day, Year, H:MM AM/PM"`` long form (e.g.,
  ``"June 7, 2025, 12:00 AM"``) — the format used by the ``OPDATETIME``
  column in the IPTSUMOPRT export added for issue #7. Only the
  time-of-day component is returned in :class:`ParsedTimeOfDay`; the
  date is intentionally discarded so the caller's row-date column
  remains the single source of truth (mirrors the rule that prevents a
  sentinel ``1900-01-01`` leaking out of the parser).

Everything else — decimal hour (``8.5``), Excel serial fraction (``0.354166``),
Buddhist-year-prefixed dates (``2568-01-01``), sentinels ``0`` / ``9999`` /
``null``, empty string, garbage, ``None``, misspelled month names, transposed
fields, hour-out-of-range — yields a :class:`ParseResult` with ``value=None``
and a populated ``parse_warning``.

Returns a :class:`ParsedTimeOfDay`, not a sentinel-dated ``datetime``: the time
parser does not know which date the time belongs to, so it does not invent one.
Callers combine the time with the row's date column via
:class:`~bba.ingest.row_timestamp.RowTimestamp.from_parts`.

NEVER silently shifts: this is the contract that protects the +/-24 h evidence
window from being anchored on a wrong-but-plausible time (PRD §1, Round 2 fix E35).
"""

from __future__ import annotations

import re
from datetime import date

from bba.ingest.models import ParsedTimeOfDay, ParseResult


_MONTH_NAMES: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_MONTH_NUMBER: dict[str, int] = {name: idx + 1 for idx, name in enumerate(_MONTH_NAMES)}

# "Month Day, Year, H:MM AM/PM" — the export shape used by IPTSUMOPRT.
# Anchored ^…$ so trailing junk does not pass; non-greedy matching is
# unnecessary because every group is bounded.
_LONG_FORM_RE: re.Pattern[str] = re.compile(
    r"^(?P<month>[A-Z][a-z]+) "
    r"(?P<day>\d{1,2}), "
    r"(?P<year>\d{4}), "
    r"(?P<hour>\d{1,2}):(?P<minute>\d{2}) "
    r"(?P<ampm>AM|PM)$"
)


def parse_hosxp_time(raw: str | None) -> ParseResult:
    """Parse a HOSxP time string against the strict allow-list.

    See the module docstring for the full allow-list. Returns a frozen
    :class:`ParseResult` with either a populated ``value`` (a
    :class:`ParsedTimeOfDay`) and ``parse_warning=None``, or ``value=None``
    plus a descriptive ``parse_warning``. Never produces both or neither
    (mutual-exclusion invariant covered by the property test suite).
    """
    if raw is None:
        return ParseResult(value=None, parse_warning="empty: input was None", raw="")
    if raw == "":
        return ParseResult(value=None, parse_warning="empty: empty string", raw=raw)

    # HHMMSS: 6 zero-padded digits with each component in range.
    if len(raw) == 6 and raw.isdigit():
        h, m, s = int(raw[0:2]), int(raw[2:4]), int(raw[4:6])
        if 0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59:
            return ParseResult(
                value=ParsedTimeOfDay(hour=h, minute=m, second=s),
                parse_warning=None,
                raw=raw,
            )
        return ParseResult(
            value=None,
            parse_warning=f"hhmmss: out-of-range {h:02d}:{m:02d}:{s:02d}",
            raw=raw,
        )

    # HH:MM: 5 chars with a literal colon at position 2.
    if (
        len(raw) == 5
        and raw[2] == ":"
        and raw[:2].isdigit()
        and raw[3:].isdigit()
    ):
        h, m = int(raw[:2]), int(raw[3:])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return ParseResult(
                value=ParsedTimeOfDay(hour=h, minute=m, second=0),
                parse_warning=None,
                raw=raw,
            )
        return ParseResult(
            value=None,
            parse_warning=f"hh:mm: out-of-range {h:02d}:{m:02d}",
            raw=raw,
        )

    # Explicit sentinels — clearer warnings than the generic fall-through.
    if raw == "0":
        return ParseResult(value=None, parse_warning="sentinel: '0'", raw=raw)
    if raw == "9999":
        return ParseResult(value=None, parse_warning="sentinel: '9999'", raw=raw)
    if raw.lower() == "null":
        return ParseResult(value=None, parse_warning="sentinel: 'null'", raw=raw)

    # "Month Day, Year, H:MM AM/PM" long form — IPTSUMOPRT.OPDATETIME shape.
    long_form = _LONG_FORM_RE.match(raw)
    if long_form is not None:
        month = long_form.group("month")
        if month not in _MONTH_NUMBER:
            return ParseResult(
                value=None,
                parse_warning=f"long-form: unknown month name {month!r}",
                raw=raw,
            )
        day = int(long_form.group("day"))
        year = int(long_form.group("year"))
        h12 = int(long_form.group("hour"))
        minute = int(long_form.group("minute"))
        ampm = long_form.group("ampm")
        # Calendar validation: reject "June 31, 2025" et al. The date
        # constructor enforces day-in-month and leap-year rules; without
        # this, the parser would silently accept impossible dates and
        # the orchestrator would re-parse them via a separate path.
        try:
            date(year, _MONTH_NUMBER[month], day)
        except ValueError as exc:
            return ParseResult(
                value=None,
                parse_warning=f"long-form: invalid calendar date ({exc})",
                raw=raw,
            )
        if not (1 <= h12 <= 12):
            return ParseResult(
                value=None,
                parse_warning=f"long-form: hour out of 12-h range ({h12})",
                raw=raw,
            )
        if not (0 <= minute <= 59):
            return ParseResult(
                value=None,
                parse_warning=f"long-form: minute out of range ({minute})",
                raw=raw,
            )
        # 12-h to 24-h conversion: 12 AM -> 0, 12 PM -> 12, otherwise +12 if PM.
        if ampm == "AM":
            hour24 = 0 if h12 == 12 else h12
        else:
            hour24 = 12 if h12 == 12 else h12 + 12
        return ParseResult(
            value=ParsedTimeOfDay(hour=hour24, minute=minute, second=0),
            parse_warning=None,
            raw=raw,
        )

    # Decimal hour (8.5) or Excel serial fraction (0.354166) — refused.
    if "." in raw:
        return ParseResult(
            value=None,
            parse_warning=f"unrecognized: decimal-hour or excel-serial not allowed ({raw!r})",
            raw=raw,
        )

    return ParseResult(
        value=None,
        parse_warning=f"unrecognized: format not in allow-list ({raw!r})",
        raw=raw,
    )
