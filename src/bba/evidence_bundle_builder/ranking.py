"""Ranking + truncation primitives for the evidence bundle.

Three responsibilities:

* :func:`parse_soap_sections` — split an IPDADMPROGRESS note text into its
  Subjective / Objective / Assessment / Plan sections so the bundle can
  emit them in priority order (issue #16 AC: A + P first, O next, S last).
* :func:`split_focus_notes_5_5` — split IPDNRFOCUSDT notes into the 5-before
  / 5-after order anchor with closest-to-anchor first (issue #16 AC).
* :func:`truncate_to_char_cap` — drop entries from the end of an item list
  greedily so the rendered bundle fits the LLM token budget (issue #16 AC).
  Section-level truncation within IPDADMPROGRESS items is handled in
  :mod:`bba.evidence_bundle_builder.builder`, where the section structure
  is known.

These are exposed publicly so the test suite can drive the contracts
directly — the builder pipeline is a thin assembler around them.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime

from bba.evidence_bundle_builder.canonical import canonical_serialize
from bba.evidence_bundle_builder.models import (
    EvidenceItem,
    FocusNote,
    SOAPSection,
)


SECTION_PRIORITY: tuple[SOAPSection, ...] = (
    "ASSESSMENT",
    "PLAN",
    "OBJECTIVE",
    "SUBJECTIVE",
)
"""Section emission order: most-important-first.

Truncation walks this tuple in REVERSE — SUBJECTIVE drops first because the
patient's own report is the most easily reconstructed from the rest of the
chart. ASSESSMENT is last to drop because it is the clinician's diagnosis-time
interpretation; losing it would change what the LLM is auditing."""


# Inline-header regex: matches SOAP headers anywhere in the text, not just at
# line starts. The leading ``(?:^|\s)`` requires a boundary so "BP" doesn't
# match (B is not preceded by whitespace before P). Real charts often pack
# multiple sections into a single line ("S: tired O: BP 90/60 A: anemia
# P: PRBC"); the line-anchored variant misclassified those entirely as
# SUBJECTIVE and let truncation drop ASSESSMENT first — exactly inverted from
# the AC's priority order.
# Note: the trailing ``\s*`` is INTENTIONALLY OMITTED. Including it would
# greedy-consume the newline between consecutive header-only lines like
# ``S:\nO:\n...``, leaving the next "O" without a whitespace boundary so the
# regex would skip it and misclassify "O:" as content of SUBJECTIVE.
# strip() on the per-section segment trims any leading whitespace anyway.
_INLINE_HEADER_PATTERN = re.compile(
    r"(?:^|\s)\s*"
    r"(?P<header>S|Subjective|CC|HPI|O|Objective|A|Assessment|Impression|P|Plan)"
    r"\s*:",
    re.IGNORECASE,
)

_HEADER_TO_SECTION: dict[str, SOAPSection] = {
    "s": "SUBJECTIVE",
    "subjective": "SUBJECTIVE",
    "cc": "SUBJECTIVE",
    "hpi": "SUBJECTIVE",
    "o": "OBJECTIVE",
    "objective": "OBJECTIVE",
    "a": "ASSESSMENT",
    "assessment": "ASSESSMENT",
    "impression": "ASSESSMENT",
    "p": "PLAN",
    "plan": "PLAN",
}


def parse_soap_sections(text: str) -> Mapping[SOAPSection, str]:
    """Split a SOAP-formatted note into its four sections.

    Returns a mapping containing every key in :data:`SECTION_PRIORITY`. Empty
    sections map to ``""`` rather than being absent — callers can rely on the
    full key set without an existence check.

    Recognized header forms (case-insensitive, with an inline boundary so
    headers may appear mid-line as well as at line starts):

    * Subjective: ``S:``, ``Subjective:``, ``CC:``, ``HPI:``
    * Objective:  ``O:``, ``Objective:``
    * Assessment: ``A:``, ``Assessment:``, ``Impression:``
    * Plan:       ``P:``, ``Plan:``

    Notes without any recognized header are treated as a single OBJECTIVE
    section — the IPDADMPROGRESS column is itself named ``OBJECTIVE`` in the
    HOSxP schema, so the no-header default is the most truthful fallback.

    Text appearing before the first header is treated as OBJECTIVE preamble
    (same default), preventing pre-header chart metadata from being silently
    captured under the wrong section."""
    sections: dict[SOAPSection, list[str]] = {k: [] for k in SECTION_PRIORITY}

    matches = list(_INLINE_HEADER_PATTERN.finditer(text))

    if not matches:
        if text.strip():
            sections["OBJECTIVE"].append(text.strip())
        return {k: "\n".join(lines).strip() for k, lines in sections.items()}

    # Pre-first-header text → OBJECTIVE preamble.
    leading = text[: matches[0].start()].strip()
    if leading:
        sections["OBJECTIVE"].append(leading)

    for i, m in enumerate(matches):
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = _HEADER_TO_SECTION[m.group("header").lower()]
        content = text[m.end() : next_start].strip()
        if content:
            sections[section].append(content)

    return {k: "\n".join(lines).strip() for k, lines in sections.items()}


FOCUS_INDICATION_TERMS: tuple[str, ...] = (
    r"เลือดออก",
    r"ซีด",
    r"เลือดจาง",
    r"ถ่ายดำ",
    r"อาเจียนเป็นเลือด",
    r"ความดันต่ำ",
    r"\bhb\s*[=:]?\s*\d",
    r"\bhct\s*[=:]?\s*\d",
    r"bleed",
    r"melena",
    r"hematemesis",
    r"hypotens",
    r"bp\s*drop",
    r"\bshock",
    r"anemi",
)
"""Observed-indication vocabulary for IPDNRFOCUSDT salience.

Measured 2026-09-22 over all 39,749 IPD orders: 72.7% exceed the 5+5 cap and
closest-first kept only 33.6% of the pre-order notes matching this list; on
8.1% of orders it kept none. Ranking hits first keeps 91.8% and leaves 3.0%
of orders with more than five hits on the pre-order side.

``hb``/``hct`` require a following number: a bare mention ("follow Hb") is a
plan, a value ("Hb 6.8") is an observation. Bare ``prc``/``ให้เลือด`` were
dropped for the same reason (administration, not indication)."""

_FOCUS_INDICATION_PATTERN = re.compile("|".join(FOCUS_INDICATION_TERMS), re.IGNORECASE)

# A match preceded, within the same clause, by a negation or an example-list
# marker is a non-finding ("no bleed", "ไม่มีเลือดออก", "no sign of septic
# shock") or a templated surveillance list ("sign of shock เช่น ...") —
# neither is an observation. The look-behind stops at a clause separator so
# "no pain, bleeding at wound" still counts the bleeding.
_NEGATION_LOOKBEHIND_CHARS = 24
_NEGATION_PATTERN = re.compile(
    r"(\bno\b|\bnot\b|ไม่|without|\bneg|\bnil\b|\bnad\b|เช่น)", re.IGNORECASE
)
_CLAUSE_SEPARATOR_PATTERN = re.compile(r"[,;/\n]|\bbut\b|แต่", re.IGNORECASE)


def _is_negated(segment: str, start: int) -> bool:
    window = segment[max(0, start - _NEGATION_LOOKBEHIND_CHARS) : start]
    separators = list(_CLAUSE_SEPARATOR_PATTERN.finditer(window))
    clause = window[separators[-1].end() :] if separators else window
    return _NEGATION_PATTERN.search(clause) is not None


# The pilot joins the HOSxP columns as "Action: ...\nResponse: ..." (mirroring
# the labelled SOAP join for IPDADMPROGRESS). ACTION is templated care-plan
# text ("Observe signs bleeding เช่น แผลผ่าตัด") that matches on nearly every
# surgical patient, so only the Response segment is scanned when the label
# is present.
_RESPONSE_SEGMENT_PATTERN = re.compile(r"(?:^|\n)Response:(?P<body>.*)\Z", re.DOTALL)


def focus_indication_hit(text: str) -> bool:
    """True when the note records an OBSERVED transfusion indication.

    Scans the ``Response:`` segment when the text is labelled, else the whole
    text. A vocabulary match is discarded when a negation / example marker
    precedes it in the same clause (see :func:`_is_negated`).
    """
    segment_match = _RESPONSE_SEGMENT_PATTERN.search(text)
    segment = segment_match.group("body") if segment_match else text
    return any(
        not _is_negated(segment, m.start())
        for m in _FOCUS_INDICATION_PATTERN.finditer(segment)
    )


def split_focus_notes_5_5(
    *,
    notes: Sequence[FocusNote],
    anchor: datetime,
    cap_before: int = 5,
    cap_after: int = 5,
) -> tuple[FocusNote, ...]:
    """Return up to ``cap_before`` + ``cap_after`` notes around ``anchor``.

    Selection rule (PRD §7 + issue #16 AC, salience tier added 2026-09-22):

    1. Partition into ``before = timestamp <= anchor`` and ``after =
       timestamp > anchor``. An at-anchor note belongs to ``before``: at-
       anchor is the latest possible "what was true at decision time" data
       point, so attaching it to the post-order side would silently demote it.
    2. Within each side, notes with an observed indication
       (:func:`focus_indication_hit`) rank ahead of the rest; within a tier,
       closest-to-anchor first (``before`` by descending timestamp, ``after``
       by ascending). Take the first ``cap_before`` / ``cap_after``.
    3. Concatenate ``before + after`` — the returned tuple's order is
       deterministic across input shuffles, which is what makes the
       stable-IDs AC hold downstream.

    No padding when fewer notes are available; when only 3 ``before`` exist,
    3 are returned (the cap is a ceiling, not a target)."""

    # Sort key includes ``n.text`` so the order is TOTAL — without the
    # tiebreak, two focus notes charted at the same minute would retain
    # caller order (Python's stable sort), leaking input shuffle into the
    # bundle and breaking the reorder-invariance AC. The tiebreak is on the
    # whole text (the only model field besides timestamp) so any two
    # genuinely-distinct rows have a deterministic order; two byte-identical
    # rows are operationally a duplicate and order does not matter for hash.
    # NFC-normalize the text tiebreak so NFD vs NFC variants of the same
    # text sort identically — without this, the bundle hash would leak
    # the input encoding even though canonical_serialize unifies it.
    def _tier(n: FocusNote) -> int:
        return 0 if focus_indication_hit(n.text) else 1

    before = sorted(
        (n for n in notes if n.timestamp <= anchor),
        key=lambda n: (-_tier(n), n.timestamp, unicodedata.normalize("NFC", n.text)),
        reverse=True,
    )[:cap_before]
    after = sorted(
        (n for n in notes if n.timestamp > anchor),
        key=lambda n: (_tier(n), n.timestamp, unicodedata.normalize("NFC", n.text)),
    )[:cap_after]
    return tuple(before) + tuple(after)


def truncate_to_char_cap(
    *,
    items: Sequence[EvidenceItem],
    char_cap: int,
) -> tuple[EvidenceItem, ...]:
    """Greedily drop trailing items until the rendered list fits ``char_cap``.

    Walks ``items`` in input order, accumulating into the kept tuple and
    re-rendering after each addition. Stops the moment the next addition would
    exceed the cap. Returns the surviving prefix as a tuple — guaranteed to be
    a contiguous subsequence of the input, which is the property the
    ``test_truncate_to_char_cap_returns_subsequence`` invariant locks in.

    The builder calls this AFTER doing section-level truncation on
    IPDADMPROGRESS payloads, so the items handed in here are already as small
    as section-priority allows. Whole-item drop is the last-resort layer."""
    if not items:
        return ()

    kept: list[EvidenceItem] = []
    for item in items:
        candidate = [*kept, item]
        rendered = canonical_serialize(
            [
                {
                    "id": it.id,
                    "source": it.source,
                    "timestamp_utc": it.timestamp_utc,
                    "payload": dict(it.payload),
                }
                for it in candidate
            ]
        )
        if len(rendered) <= char_cap:
            kept.append(item)
        else:
            break
    return tuple(kept)


__all__ = (
    "FOCUS_INDICATION_TERMS",
    "SECTION_PRIORITY",
    "focus_indication_hit",
    "parse_soap_sections",
    "split_focus_notes_5_5",
    "truncate_to_char_cap",
)
