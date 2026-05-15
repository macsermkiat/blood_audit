"""Date-shift transform: rewrite in-text dates as Δ-days-from-admission.

PRD §8: "Date-shift to relative offsets (Δ-days-from-admission)". The
wrapper applies this AFTER ``thai-medical-deid`` has redacted explicit
date PHI to ``[DATE]`` tokens. Any remaining literal date strings inside
the note text — typically inside running prose ("admitted on 2026-05-10")
that the backend missed, or pre-redaction-resistant formats like "5/10/26"
— are converted to anchored offsets like ``Day 0``, ``Day +3``, ``Day -2``.

Determinism contract: same input text + same ``admission_date`` →
byte-identical output. The bundle-hash stability AC depends on this —
two runs over the same source must produce the same redacted bytes.

Supported date formats (regexes in :data:`DATE_PATTERNS`):

* ``YYYY-MM-DD`` — ISO 8601 calendar date
* ``YYYY/MM/DD`` — common ISO variant
* ``DD/MM/YYYY`` — Thai/EU convention
* ``DD-MM-YYYY`` — Thai/EU variant

Unrecognized formats (Buddhist-year prefix, decimal hour fragments, etc.)
are left untouched — the strict ingest-time parser
(:mod:`bba.ingest.time_parser`) has already routed those records to a
``parse_warning`` column upstream.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date


DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<![\d\-/])(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})(?![\d\-])"),
    re.compile(r"(?<![\d/])(?P<y>\d{4})/(?P<m>\d{2})/(?P<d>\d{2})(?![\d/])"),
    re.compile(r"(?<![\d/])(?P<d>\d{2})/(?P<m>\d{2})/(?P<y>\d{4})(?![\d/])"),
    re.compile(r"(?<![\d\-])(?P<d>\d{2})-(?P<m>\d{2})-(?P<y>\d{4})(?![\d\-])"),
)
"""The recognized literal-date patterns. ISO forms are tried first; the
day-first variants follow so a string like ``"2026-05-10"`` is not
mis-parsed as DD-MM-YYYY. The patterns use look-behind / look-ahead
boundary classes that reject neighbouring digits and the separator char,
so an identifier-like fragment ``"H123-456-789"`` does not produce a
false date match.
"""


class DateMatch:
    """One detected date in a note's redacted text."""

    __slots__ = ("start", "end", "parsed")

    def __init__(self, *, start: int, end: int, parsed: date) -> None:
        self.start = start
        self.end = end
        self.parsed = parsed

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"DateMatch(start={self.start}, end={self.end}, parsed={self.parsed!r})"


def parse_dates(text: str) -> Sequence[DateMatch]:
    """Return every parseable date match in ``text``, sorted by ``start``.

    Overlapping matches are resolved in favor of the EARLIER pattern in
    :data:`DATE_PATTERNS` (ISO-first), so a literal ``"2026-05-10"`` is
    parsed as ISO and not also as DD-MM-YYYY at a shifted offset.

    Pure function; no side effects, no I/O.
    """
    matches: list[DateMatch] = []
    claimed: list[tuple[int, int]] = []

    for pattern in DATE_PATTERNS:
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            if _overlaps_any(start, end, claimed):
                continue
            try:
                parsed = date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
            except ValueError:
                continue
            matches.append(DateMatch(start=start, end=end, parsed=parsed))
            claimed.append((start, end))

    matches.sort(key=lambda dm: dm.start)
    return tuple(matches)


def _overlaps_any(start: int, end: int, claimed: Sequence[tuple[int, int]]) -> bool:
    """Return ``True`` iff ``[start, end)`` overlaps any prior claimed span."""
    for cs, ce in claimed:
        if start < ce and cs < end:
            return True
    return False


def format_offset(*, days: int) -> str:
    """Render a day offset in the canonical ``Day N`` form.

    Examples: ``0`` → ``"Day 0"``, ``3`` → ``"Day +3"``, ``-2`` → ``"Day -2"``.
    """
    if days == 0:
        return "Day 0"
    if days > 0:
        return f"Day +{days}"
    return f"Day {days}"


def shift_dates_in_text(text: str, *, admission_date: date) -> str:
    """Replace every detected literal date in ``text`` with its offset form.

    ``admission_date`` is the patient's admission day; offsets are
    ``(date_in_text - admission_date).days``. The transform is total
    (every match in :func:`parse_dates` is replaced) and order-preserving
    (no re-flow of unmatched text).

    Pure function. Returns a NEW string; the input is never mutated.
    """
    matches = parse_dates(text)
    if not matches:
        return text

    parts: list[str] = []
    cursor = 0
    for m in matches:
        parts.append(text[cursor : m.start])
        delta_days = (m.parsed - admission_date).days
        parts.append(format_offset(days=delta_days))
        cursor = m.end
    parts.append(text[cursor:])
    return "".join(parts)
