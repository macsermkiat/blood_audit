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

Supported literal-date formats (regexes in :data:`DATE_PATTERNS`):

* ``YYYY-MM-DD`` — ISO 8601 calendar date
* ``YYYY/MM/DD`` — common ISO variant
* ``DD/MM/YYYY`` — Thai/EU convention
* ``DD-MM-YYYY`` — Thai/EU variant

All four formats require a four-digit year and zero-padded day/month so
ambiguous strings like ``"5/10/26"`` (two-digit year, unpadded) are
NOT matched — those land in the upstream ingest layer's
``parse_warning`` channel rather than risk a wrong-century shift here.
Unrecognized formats (Buddhist-year prefix, decimal hour fragments, etc.)
are likewise left untouched — the strict ingest-time parser
(:mod:`bba.ingest.time_parser`) has already routed those records to a
``parse_warning`` column upstream.

Backend-tagged date spans (``entity_type == "DATE"``) carrying a
``YYYY-MM-DD`` ``original_text`` are converted by
:func:`shift_date_spans_in_text` — the in-text-regex path only catches
literal dates the backend missed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date

from bba.deid_redactor.exceptions import BackendRedactionError
from bba.deid_redactor.models import RedactionSpan, RoleToken


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


_DATE_PLACEHOLDER: str = RoleToken.DATE.value


def shift_date_spans_in_text(
    redacted_text: str,
    *,
    spans: Sequence[RedactionSpan],
    admission_date: date,
) -> str:
    """Replace ``[DATE]`` placeholders with their per-span ``Day N`` offsets.

    Iterates the spans whose ``entity_type == "DATE"`` in document order
    and substitutes the next ``[DATE]`` placeholder in ``redacted_text``
    with the offset form derived from the span's ``original_text``.

    Per-span behavior:

    * If ``span.original_text`` parses as one of the recognized formats
      (:data:`DATE_PATTERNS`), the placeholder becomes
      :func:`format_offset` of the day delta from ``admission_date``.
    * If ``span.original_text`` is empty or unparseable, the placeholder
      is left as ``[DATE]`` (the backend tagged a date PHI but did not
      surface a parseable form — typically because the backend's own
      regex used a format the wrapper does not recognize). This is the
      safe fail-open behavior: a non-rewritten ``[DATE]`` is still
      redacted PHI, just without the Δ-day annotation.

    Invariant: the count of ``[DATE]`` placeholders in ``redacted_text``
    must equal the count of DATE-typed entries in ``spans``. A mismatch
    raises :class:`BackendRedactionError` (mirrors the contract on
    :func:`bba.deid_redactor.roles.upgrade_person_tokens`).
    """
    date_spans = tuple(s for s in spans if s.entity_type == "DATE")
    placeholder_count = redacted_text.count(_DATE_PLACEHOLDER)
    if placeholder_count != len(date_spans):
        raise BackendRedactionError(
            "DATE placeholder count mismatch: redacted_text has "
            f"{placeholder_count} '[DATE]' tokens but spans has "
            f"{len(date_spans)} DATE entries"
        )
    if not date_spans:
        return redacted_text

    parts: list[str] = []
    cursor = 0
    placeholder_len = len(_DATE_PLACEHOLDER)

    for span in date_spans:
        idx = redacted_text.find(_DATE_PLACEHOLDER, cursor)
        if idx == -1:  # pragma: no cover - guarded by placeholder count check
            break
        parts.append(redacted_text[cursor:idx])
        parts.append(_format_date_span(span, admission_date=admission_date))
        cursor = idx + placeholder_len

    parts.append(redacted_text[cursor:])
    return "".join(parts)


def _format_date_span(span: RedactionSpan, *, admission_date: date) -> str:
    """Render one DATE span as ``Day N`` or fall back to the placeholder.

    Unparseable spans (empty ``original_text`` or non-canonical format)
    keep the generic ``[DATE]`` placeholder so the audit chain still
    carries a redacted token rather than leaking the raw original.
    """
    if not span.original_text:
        return _DATE_PLACEHOLDER
    matches = parse_dates(span.original_text)
    if not matches:
        return _DATE_PLACEHOLDER
    return format_offset(days=(matches[0].parsed - admission_date).days)
