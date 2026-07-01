"""Canonical HOSxP date/time parsers for the pilot scripts.

Single owner of the small date/time helper set so ``run_pipeline.py``,
``run_llm_leg.py`` and ``_anchor_candidates.py`` import one implementation
instead of carrying their own copies (which had already drifted in scope).
Extracted verbatim from the previously-duplicated copies in the two pilot
scripts. See ``docs/handoff-hb-anchor-unification.md``.
"""

from __future__ import annotations

from datetime import date, datetime

from bba.ingest.models import ParsedTimeOfDay
from bba.ingest.row_timestamp import RowTimestamp
from bba.ingest.time_parser import parse_hosxp_time

TZ_LOCAL = "Asia/Bangkok"


def _parse_hosxp_date(raw: str) -> date | None:
    if not raw:
        return None
    head = raw.split(" ", 1)[0]
    try:
        return date.fromisoformat(head)
    except ValueError:
        return None


def _parse_time(raw: str | None) -> ParsedTimeOfDay | None:
    if not raw:
        return None
    stripped = str(raw).strip()
    if stripped.isdigit() and 1 <= len(stripped) <= 6:
        stripped = stripped.zfill(6)
    return parse_hosxp_time(stripped).value


def _fmt_hosxp_time(raw: str | None) -> str:
    """Render HOSxP integer-like time cells as HH:MM:SS."""
    if raw is None:
        return ""
    stripped = str(raw).strip()
    if not stripped:
        return ""
    if stripped.isdigit() and 1 <= len(stripped) <= 6:
        padded = stripped.zfill(6)
        return f"{padded[:2]}:{padded[2:4]}:{padded[4:6]}"
    return stripped


def _fmt_local_datetime(date_raw: str | None, time_raw: str | None = None) -> str:
    """Render split HOSxP date/time cells as a local datetime string."""
    d = (date_raw or "").strip().split(" ")[0]
    t = _fmt_hosxp_time(time_raw)
    if d and t:
        return f"{d} {t}"
    return d or t or ""


def _combine(d: date | None, t: ParsedTimeOfDay | None) -> datetime | None:
    if d is None or t is None:
        return None
    return RowTimestamp.from_parts(d, t, tz=TZ_LOCAL).utc
