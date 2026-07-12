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
    rf"(?:ไม่(?:ได้)?(?:ให้|รับ)|งดให้|จะให้|เตรียม|แผน|\bplan\b"
    rf"|ส่ง[^\n]{{0,80}}?ไป|\bG/?M\b|จอง"
    # Reversed group-and-match crossmatch shorthand ("M/G LPRC 2Unit"): the
    # mirror of the \bG/?M\b token above. The slash is REQUIRED so this never
    # matches "mg" (milligrams). (#117 case 68080335)
    rf"|\bM/G\b"
    # Portering reserved units to theatre ("ให้นำ LPRC ... IV to OR"): the
    # นำ ("carry/bring") counterpart of the ส่ง...ไป dispatch cue. (#117 case
    # 68046079)
    rf"|นำ[^\n]{{0,60}}?(?:ไป|to)\b"
    # Patient counselling ("แจ้งให้ผู้ป่วยทราบ...การให้เลือด") and the
    # prospective bedside complication-watch instruction ("Observe complication
    # ขณะให้เลือด...") are transfusion education/monitoring text, not
    # administration. The complication cue is anchored on the ขณะ(ให้|ได้รับ)
    # instruction form so it never suppresses a completed "no complication
    # post-transfusion" note that carries its own affirmative marker. (#117
    # case 68046079)
    rf"|แจ้ง[^\n]{{0,20}}?ทราบ|Observe\s+complication\s+ขณะ(?:ให้|ได้รับ)"
    # A "ก่อนให้เลือด" (before-transfusion) reference is a pre-transfusion
    # baseline — a V/S reading or a premed order given BEFORE the unit — not
    # evidence the reserved unit was administered. Like every guard here it is
    # line-scoped, so on the rare same-line multi-unit note that also charts a
    # completed unit it degrades to a (safe, false-negative-biased) unconfirmed
    # rather than a false confirmation. (#117 case 68046079)
    rf"|ก่อนให้เลือด"
    # Retrospective history-summary arrow: a dated lab line whose "-->" records
    # a PAST transfusion decision ("2/9/68 ...PTT=36 --> LPRC 1 unit"), not
    # administration of the reserved unit. Anchored on a leading DD/MM/YY date
    # so a bare "-->" used as a live connective is not caught. (#117 case
    # 68055153)
    rf"|\d{{1,2}}/\d{{1,2}}/\d{{2,4}}[^\n]{{0,100}}?-->\s*(?:ได้\s*)?(?:{RBC_COMPONENT}|เลือด)"
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
_RED_CELL_CITATION_RE = re.compile(
    rf"(?:ให้เลือด|ได้รับเลือด|หลังให้เลือด|{RBC_COMPONENT})",
    re.IGNORECASE,
)


def _citation_lines(source_text: str, quote: str) -> tuple[str, ...]:
    """Return every source line containing an exact citation occurrence."""
    if not quote:
        return ()
    lines: list[str] = []
    start = 0
    while (match_start := source_text.find(quote, start)) != -1:
        line_start = source_text.rfind("\n", 0, match_start) + 1
        line_end = source_text.find("\n", match_start + len(quote))
        if line_end == -1:
            line_end = len(source_text)
        lines.append(source_text[line_start:line_end])
        start = match_start + 1
    return tuple(lines)


def administration_citation_has_negative_context(source_text: str, quote: str) -> bool:
    """Return whether a grounded quote occurs on a non-administration line.

    Citation grounding proves that words exist in a source, not that those
    words document administration. Reuse the scan's false-negative-biased
    line guard so reservation, planning, and dispatch citations cannot become
    a separate path around :func:`scan_administration`.
    """
    return any(
        _NEGATIVE_CONTEXT_RE.search(line) is not None
        for line in _citation_lines(source_text, quote)
    )


def administration_citation_supports_red_cell(source_text: str, quote: str) -> bool:
    """Return whether every cited line identifies red-cell administration.

    For an RBC order, FFP/cryo/platelet administration and component-free
    blood-loss text do not confirm that the reserved red cells were given.
    Generic Thai give/receive-blood or post-transfusion wording is accepted
    because the audited order supplies the red-cell component context.
    Ambiguous repeated quotes fail closed if any occurrence lacks a red-cell
    marker.
    """
    lines = _citation_lines(source_text, quote)
    return bool(lines) and all(_RED_CELL_CITATION_RE.search(line) for line in lines)


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
