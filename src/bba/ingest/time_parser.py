"""Strict HOSxP time-of-day parser.

Allow-list of accepted formats: ``HHMMSS`` (6 zero-padded digits) and ``HH:MM``.
Everything else — decimal hour (``8.5``), Excel serial fraction (``0.354166``),
Buddhist-year-prefixed dates (``2568-01-01``), sentinels ``0`` / ``9999`` /
``null``, empty string, garbage, ``None`` — yields a :class:`ParseResult` with
``value=None`` and a populated ``parse_warning``.

Returns a :class:`ParsedTimeOfDay`, not a sentinel-dated ``datetime``: the time
parser does not know which date the time belongs to, so it does not invent one.
Callers combine the time with the row's date column via
:class:`~bba.ingest.row_timestamp.RowTimestamp.from_parts`.

NEVER silently shifts: this is the contract that protects the +/-24 h evidence
window from being anchored on a wrong-but-plausible time (PRD §1, Round 2 fix E35).
"""

from __future__ import annotations

from bba.ingest.models import ParsedTimeOfDay, ParseResult


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
