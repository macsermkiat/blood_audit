"""Affirmative-only blood-administration evidence scan (issue #107).

This scan recovers charted administration facts from the shipped narrative. It
can only confirm administration, never deny it: absence of a ``ให้เลือด`` note
is NEVER evidence of non-transfusion. An empty result therefore means that
administration is unconfirmed, not that blood was not administered.

Precision is deliberately favoured over recall. A cue on a newline-delimited
line carrying negation, withholding, planning, preparation, reservation, or
transport language is discarded by a strict negative-context guard. This bias
toward false negatives is safe by design because a missed marker remains merely
unconfirmed, whereas a false affirmative marker could incorrectly assert that
administration occurred.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from bba.vitals_extractor.components import BLOOD_COMPONENT, RBC_COMPONENT
from bba.vitals_extractor.models import (
    AdministrationFinding,
    AdministrationSummary,
    VitalsNote,
)

# Affirmative cues key on RED-CELL products only (Codex round 5 on PR #112):
# this scan runs for red-cell bundles, where a charted FFP/platelet/cryo
# administration does not confirm the reserved red cells were given. The
# negative guard below stays on the broad component set (broader guard = more
# false negatives, which are safe).
_GAVE_BLOOD_RE = re.compile(
    rf"(?:ให้เลือด|ให้\s*{RBC_COMPONENT})",
    re.IGNORECASE,
)
_UNIT_COUNT = r"\d+\s*(?:units?|ยูนิต|ถุง)"
_UNIT_COUNT_RE = re.compile(
    rf"{RBC_COMPONENT}[^\n]{{0,40}}?{_UNIT_COUNT}"
    rf"|{_UNIT_COUNT}[^\n]{{0,40}}?{RBC_COMPONENT}",
    re.IGNORECASE,
)
_POST_TRANSFUSION_RE = re.compile(
    r"(?:post[\s-]?transfusion|หลังให้เลือด|transfusion\s+reaction)",
    re.IGNORECASE,
)
_NEGATIVE_CONTEXT_RE = re.compile(
    rf"(?:ไม่(?:ได้)?(?:ให้|รับ)|งดให้|จะให้|เตรียม|แผน|\bplan\b|ส่ง\S*ไป|\bG/?M\b|จอง"
    rf"|\bno\s+(?:{BLOOD_COMPONENT}|blood\b|transfusion\b(?!\s+reaction))"
    r"|\bnot\s+(?:yet\s+)?(?:given|transfused|received)\b"
    r"|ปฏิเสธ|\b(?:refused|declined)\b"
    # History screens and pre-transfusion checks mention reactions without
    # documenting administration (Codex round 3 on PR #112). A bare
    # post-transfusion "no transfusion reaction" line remains affirmative.
    r"|\bhistory\b|ประวัติ|\bpre[\s-]?transfusion\b)",
    re.IGNORECASE,
)

# A bare component+count line is ambiguous: "order LPRC 2 units" restates the
# very order being audited and must not confirm administration (it would beg
# the question the #109 gate exists to answer). The unit_count cue therefore
# also requires administered wording in the SAME ;/,-delimited clause as the
# component-count match (a verb bound to a different clause, e.g. "order PRC
# 2 units; patient received Lasix", must not count); ให้-prefixed forms are
# already covered by the gave_blood cue.
_ADMINISTERED_VERB_RE = re.compile(
    r"(?:ให้|ได้รับ|\bgiven\b|\btransfused\b|\breceived\b)",
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
                    if match is None:
                        continue
                    if category == "unit_count" and (
                        _ADMINISTERED_VERB_RE.search(
                            _clause_around(line, match.start(), match.end())
                        )
                        is None
                    ):
                        continue
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


_CLAUSE_DELIMITERS = (";", ",", ".")


def _clause_around(line: str, start: int, end: int) -> str:
    """The clause of ``line`` containing ``[start, end)``.

    Clauses are delimited by ``;``/``,``/``.`` so an administered verb in a
    neighbouring clause or sentence never binds to the component count."""
    lo = max(line.rfind(d, 0, start) for d in _CLAUSE_DELIMITERS) + 1
    ends = [p for d in _CLAUSE_DELIMITERS if (p := line.find(d, end)) != -1]
    hi = min(ends) if ends else len(line)
    return line[lo:hi]


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
