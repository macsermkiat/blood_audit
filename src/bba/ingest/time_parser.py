"""Strict HOSxP time/datetime parser.

Allow-list of accepted formats: ``HHMMSS`` (6-digit zero-padded), ``HH:MM``.
Everything else — decimal hour (``8.5``), Excel serial fraction (``0.354166``),
Buddhist-year-prefixed dates (``2568-01-01``), sentinels ``0`` / ``9999`` /
``null``, empty string, garbage, ``None`` — yields a ``ParseResult`` with
``value=None`` and a populated ``parse_warning``.

NEVER silently shifts: this is the contract that protects the ±24-h evidence
window from being anchored on a wrong-but-plausible time. (PRD §1, Round 2 fix E35.)
"""

from __future__ import annotations

from bba.ingest.models import ParseResult


def parse_hosxp_time(raw: str | None) -> ParseResult:
    """Parse a HOSxP time/datetime string against the strict allow-list.

    Returns a frozen :class:`ParseResult` — either a parsed ``value`` and
    ``parse_warning=None``, or ``value=None`` with a descriptive ``parse_warning``.
    Implementations MUST NOT silently coerce decimal hours, Excel serials, or
    Buddhist-year strings into a datetime.
    """
    raise NotImplementedError
