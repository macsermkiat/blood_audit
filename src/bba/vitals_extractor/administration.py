"""Affirmative-only blood-administration evidence scan (issue #107).

This scan recovers charted administration facts from the shipped narrative. It
can only confirm administration, never deny it: absence of a ``ให้เลือด`` note
is NEVER evidence of non-transfusion. An empty result therefore means that
administration is unconfirmed, not that blood was not administered.

Precision is deliberately favoured over recall. A cue on a newline-delimited
line carrying planning, preparation, reservation, or transport language is
discarded by a strict negative-context guard. This bias toward false negatives
is safe by design because a missed marker remains merely unconfirmed, whereas a
false affirmative marker could incorrectly assert that administration occurred.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from bba.vitals_extractor.components import BLOOD_COMPONENT
from bba.vitals_extractor.models import (
    AdministrationFinding,
    AdministrationSummary,
    VitalsNote,
)

_GAVE_BLOOD_RE = re.compile(
    rf"(?:ให้เลือด|ให้\s*{BLOOD_COMPONENT})",
    re.IGNORECASE,
)
_UNIT_COUNT = r"\d+\s*(?:units?|ยูนิต|ถุง)"
_UNIT_COUNT_RE = re.compile(
    rf"{BLOOD_COMPONENT}[^\n]{{0,40}}?{_UNIT_COUNT}"
    rf"|{_UNIT_COUNT}[^\n]{{0,40}}?{BLOOD_COMPONENT}",
    re.IGNORECASE,
)
_POST_TRANSFUSION_RE = re.compile(
    r"(?:post[\s-]?transfusion|หลังให้เลือด|transfusion\s+reaction)",
    re.IGNORECASE,
)
_NEGATIVE_CONTEXT_RE = re.compile(
    r"(?:จะให้|เตรียม|แผน|\bplan\b|ส่ง\S*ไป|\bG/?M\b|จอง)",
    re.IGNORECASE,
)

_CUES = (
    ("gave_blood", _GAVE_BLOOD_RE),
    ("unit_count", _UNIT_COUNT_RE),
    ("post_transfusion", _POST_TRANSFUSION_RE),
)
_SNIPPET_RADIUS = 60


def scan_administration(notes: Sequence[VitalsNote]) -> AdministrationSummary:
    """Scan ``notes`` for affirmative blood-administration markers.

    ``notes`` are assumed already filtered to the relevant window by the caller
    (the builder/pilot owns windowing); this function is a pure aggregation over
    whatever it is given and is total — empty input yields an empty summary.

    At most one finding per category is kept: the earliest affirmative cue on a
    line that does not match the strict negative-context guard. Iterating notes
    in timestamp order makes selection deterministic across input shuffles.
    """
    found: dict[str, AdministrationFinding] = {}

    for note in sorted(notes, key=lambda n: (n.timestamp, n.source, n.text)):
        line_start = 0
        for line in note.text.split("\n"):
            if _NEGATIVE_CONTEXT_RE.search(line) is None:
                for category, cue_re in _CUES:
                    if category in found:
                        continue
                    match = cue_re.search(line)
                    if match is not None:
                        found[category] = _finding(
                            category,
                            note,
                            line_start + match.start(),
                            line_start + match.end(),
                        )
            line_start += len(line) + 1

    findings = tuple(found[c] for c, _ in _CUES if c in found)
    return AdministrationSummary(
        has_affirmative_marker=bool(findings),
        findings=findings,
    )


def _finding(
    category: str, note: VitalsNote, start: int, end: int
) -> AdministrationFinding:
    return AdministrationFinding(
        category=category,  # type: ignore[arg-type]  # narrowed by call sites
        snippet=_snippet(note.text, start, end),
        at=note.timestamp,
        source=note.source,
    )


def _snippet(text: str, start: int, end: int) -> str:
    """A whitespace-collapsed window around ``[start, end)`` with ellipses."""
    lo = max(0, start - _SNIPPET_RADIUS)
    hi = min(len(text), end + _SNIPPET_RADIUS)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return prefix + " ".join(text[lo:hi].split()) + suffix
