"""Ranking + truncation primitives for the evidence bundle.

Three responsibilities:

* :func:`parse_soap_sections` — split an IPDADMPROGRESS note text into its
  Subjective / Objective / Assessment / Plan sections so the bundle can
  emit them in priority order (issue #16 AC: A + P first, O next, S last).
* :func:`split_focus_notes_5_5` — split IPDNRFOCUSDT notes into the 5-before
  / 5-after order anchor with closest-to-anchor first (issue #16 AC).
* :func:`truncate_to_char_cap` — drop low-priority sections / items first
  when the rendered bundle exceeds the 8 K char proxy for the LLM token
  budget (issue #16 AC).

These are exposed publicly so the test suite can drive the contracts
directly — the builder pipeline is a thin assembler around them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

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


def parse_soap_sections(text: str) -> Mapping[SOAPSection, str]:
    """Split a SOAP-formatted note into its four sections.

    Returns a mapping containing every key in :data:`SECTION_PRIORITY`. Empty
    sections map to ``""`` rather than being absent — callers can rely on the
    full key set without an existence check.

    Recognized header forms (case-insensitive, anchored at line start or after
    whitespace):

    * Subjective: ``S:``, ``Subjective:``
    * Objective:  ``O:``, ``Objective:``
    * Assessment: ``A:``, ``Assessment:``, ``Impression:``
    * Plan:       ``P:``, ``Plan:``

    Notes without any recognized header are treated as a single OBJECTIVE
    section — the IPDADMPROGRESS column is itself named ``OBJECTIVE`` in the
    HOSxP schema, so the no-header default is the most truthful fallback.
    """
    raise NotImplementedError("parse_soap_sections: not implemented in RED phase")


def split_focus_notes_5_5(
    *,
    notes: Sequence[FocusNote],
    anchor: datetime,
    cap_before: int = 5,
    cap_after: int = 5,
) -> tuple[FocusNote, ...]:
    """Return up to ``cap_before`` + ``cap_after`` notes around ``anchor``.

    Selection rule (PRD §7 + issue #16 AC):

    1. Within the +/- 24 h window (caller pre-filters or calls the builder),
       partition into ``before = timestamp <= anchor`` and ``after = timestamp
       > anchor``.
    2. Sort ``before`` by descending timestamp (closest-to-anchor first); take
       the first ``cap_before``.
    3. Sort ``after`` by ascending timestamp (closest-to-anchor first); take
       the first ``cap_after``.
    4. Concatenate ``before + after`` — the returned tuple's order is
       deterministic across input shuffles, which is what makes the stable-IDs
       AC hold.

    No padding when fewer notes are available; when only 3 ``before`` exist,
    3 are returned (the cap is a ceiling, not a target)."""
    raise NotImplementedError("split_focus_notes_5_5: not implemented in RED phase")


def truncate_to_char_cap(
    *,
    items: Sequence[EvidenceItem],
    char_cap: int,
) -> tuple[EvidenceItem, ...]:
    """Drop low-priority content until the rendered bundle fits ``char_cap``.

    Priority of preservation (most-important last to drop):

    1. Diagnoses (AN-scoped, smallest, anchor-defining).
    2. Hb history (the central audit signal).
    3. Vitals (clinical state at order time).
    4. MED, focus, progress notes.

    Within IPDADMPROGRESS items, drop SECTION_PRIORITY in reverse:
    SUBJECTIVE first, then OBJECTIVE, then PLAN, then (only if still over)
    drop the entire item.

    Within IPDNRFOCUSDT items, drop the farthest-from-anchor entry first
    (the "post-order side" cap is hit later than the "pre-order side").

    Returns the surviving items in input order. The bundle as serialized is
    guaranteed to satisfy ``len(canonical_json) <= char_cap`` for the cap
    enforcement AC.
    """
    raise NotImplementedError("truncate_to_char_cap: not implemented in RED phase")


__all__ = (
    "SECTION_PRIORITY",
    "parse_soap_sections",
    "split_focus_notes_5_5",
    "truncate_to_char_cap",
)
