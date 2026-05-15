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


# Header patterns: (section, regex). Order matches SECTION_PRIORITY's reverse
# so that "S:" is checked before "A:" if both are valid prefixes — they are
# disjoint here, but the order is documentation. ``\s*:`` admits "S :" too.
_HEADER_PATTERNS: tuple[tuple[SOAPSection, re.Pattern[str]], ...] = (
    ("SUBJECTIVE", re.compile(r"^\s*(?:S|Subjective|CC|HPI)\s*:\s*", re.IGNORECASE)),
    ("OBJECTIVE", re.compile(r"^\s*(?:O|Objective)\s*:\s*", re.IGNORECASE)),
    ("ASSESSMENT", re.compile(r"^\s*(?:A|Assessment|Impression)\s*:\s*", re.IGNORECASE)),
    ("PLAN", re.compile(r"^\s*(?:P|Plan)\s*:\s*", re.IGNORECASE)),
)


def parse_soap_sections(text: str) -> Mapping[SOAPSection, str]:
    """Split a SOAP-formatted note into its four sections.

    Returns a mapping containing every key in :data:`SECTION_PRIORITY`. Empty
    sections map to ``""`` rather than being absent — callers can rely on the
    full key set without an existence check.

    Recognized header forms (case-insensitive, anchored at line start, optional
    whitespace around the colon):

    * Subjective: ``S:``, ``Subjective:``, ``CC:``, ``HPI:``
    * Objective:  ``O:``, ``Objective:``
    * Assessment: ``A:``, ``Assessment:``, ``Impression:``
    * Plan:       ``P:``, ``Plan:``

    Notes without any recognized header are treated as a single OBJECTIVE
    section — the IPDADMPROGRESS column is itself named ``OBJECTIVE`` in the
    HOSxP schema, so the no-header default is the most truthful fallback.
    """
    sections: dict[SOAPSection, list[str]] = {k: [] for k in SECTION_PRIORITY}
    current: SOAPSection = "OBJECTIVE"  # no-header default
    for line in text.splitlines():
        matched = False
        for section, pattern in _HEADER_PATTERNS:
            m = pattern.match(line)
            if m:
                current = section
                rest = line[m.end():]
                if rest:
                    sections[current].append(rest)
                matched = True
                break
        if not matched:
            sections[current].append(line)
    return {k: "\n".join(lines).strip() for k, lines in sections.items()}


def split_focus_notes_5_5(
    *,
    notes: Sequence[FocusNote],
    anchor: datetime,
    cap_before: int = 5,
    cap_after: int = 5,
) -> tuple[FocusNote, ...]:
    """Return up to ``cap_before`` + ``cap_after`` notes around ``anchor``.

    Selection rule (PRD §7 + issue #16 AC):

    1. Partition into ``before = timestamp <= anchor`` and ``after =
       timestamp > anchor``. An at-anchor note belongs to ``before``: at-
       anchor is the latest possible "what was true at decision time" data
       point, so attaching it to the post-order side would silently demote it.
    2. Sort ``before`` by descending timestamp (closest-to-anchor first); take
       the first ``cap_before``.
    3. Sort ``after`` by ascending timestamp (closest-to-anchor first); take
       the first ``cap_after``.
    4. Concatenate ``before + after`` — the returned tuple's order is
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
    before = sorted(
        (n for n in notes if n.timestamp <= anchor),
        key=lambda n: (n.timestamp, n.text),
        reverse=True,
    )[:cap_before]
    after = sorted(
        (n for n in notes if n.timestamp > anchor),
        key=lambda n: (n.timestamp, n.text),
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
    "SECTION_PRIORITY",
    "parse_soap_sections",
    "split_focus_notes_5_5",
    "truncate_to_char_cap",
)
