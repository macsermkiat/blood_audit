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
    re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b"),
    re.compile(r"\b(?P<y>\d{4})/(?P<m>\d{2})/(?P<d>\d{2})\b"),
    re.compile(r"\b(?P<d>\d{2})/(?P<m>\d{2})/(?P<y>\d{4})\b"),
    re.compile(r"\b(?P<d>\d{2})-(?P<m>\d{2})-(?P<y>\d{4})\b"),
)
"""The recognized literal-date patterns. ISO forms are tried first; the
day-first variants follow so a string like ``"2026-05-10"`` is not
mis-parsed as DD-MM-YYYY. The patterns are word-boundary-anchored
(``\\b``) to avoid grabbing fragments out of longer identifiers (e.g.
``"H123-456-789"``).
"""


class DateMatch:
    """One detected date in a note's redacted text.

    Carries the regex span (so the replacer can write back into the
    string without re-scanning) and the parsed :class:`datetime.date` (so
    the offset math is unit-testable separately from the regex match).
    Public-ish — not exported from the module's ``__init__`` but stable
    so :mod:`bba.deid_redactor.redactor` and the test suite can both
    pattern-match on it.
    """

    __slots__ = ("start", "end", "parsed")

    def __init__(self, *, start: int, end: int, parsed: date) -> None:
        self.start = start
        self.end = end
        self.parsed = parsed


def parse_dates(text: str) -> Sequence[DateMatch]:
    """Return every parseable date match in ``text``, sorted by ``start``.

    Overlapping matches are resolved in favor of the EARLIER pattern in
    :data:`DATE_PATTERNS` (ISO-first), so a literal ``"2026-05-10"`` is
    parsed as ISO and not also as DD-MM-YYYY at a shifted offset.

    Pure function; no side effects, no I/O.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #17")


def format_offset(*, days: int) -> str:
    """Render a day offset in the canonical ``Day N`` form.

    Examples: ``0`` → ``"Day 0"``, ``3`` → ``"Day +3"``, ``-2`` → ``"Day -2"``.
    The explicit ``+`` sign on positive offsets prevents misreading
    ``Day 3`` as ``Day 3 of admission`` (an absolute index) — the audit
    chain semantic is "days SINCE admission", and the sign disambiguates.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #17")


def shift_dates_in_text(text: str, *, admission_date: date) -> str:
    """Replace every detected literal date in ``text`` with its offset form.

    ``admission_date`` is the patient's admission day; offsets are
    ``(date_in_text - admission_date).days``. The transform is total
    (every match in :func:`parse_dates` is replaced) and order-preserving
    (no re-flow of unmatched text).

    Pure function. Returns a NEW string; the input is never mutated
    (Python strings are immutable, but the spec is explicit here so
    callers reason about determinism the same way they do for
    :func:`bba.evidence_bundle_builder.canonical.canonical_serialize`).
    """
    raise NotImplementedError("RED-phase scaffold; see issue #17")
