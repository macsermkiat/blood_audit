"""Pre-LLM injection scanner.

PRD §38 + issue #21 scope: detect imperative verbs, fake-guideline
patterns, and bilingual Thai/EN jailbreaks inside redacted evidence text.
Flagged rows route to ``NEEDS_REVIEW`` without an LLM call.

Design contract:

* The scanner is a **pure function** over post-redaction evidence text.
  It performs no I/O, imports no model framework, and never depends on
  the LLM client.
* Pattern matching is **case-insensitive** for ASCII letters and
  **NFC-normalized** for Thai script. Adversarial inputs encoded in NFD
  must produce the same verdict as the same content in NFC.
* The scanner returns ALL matches (not just the first) so monitoring
  can count per-pattern hit rates. Flagging is the OR over all patterns.
* The shipped pattern set must satisfy
  :data:`bba.prompt_builder.MIN_REQUIRED_INJECTION_PATTERNS` (>= 20)
  covering imperative verbs (EN + TH), fake-guideline references,
  bilingual jailbreaks, system-prompt exfiltration, and role-pretend
  attempts.

RED-phase scaffold: :func:`scan_injection` and :func:`scan_chunks` raise
:class:`NotImplementedError`. The shipped pattern catalog is also
declared as :data:`INJECTION_PATTERNS` (a stub tuple) so the public
import surface is stable from RED forward; the catalog is filled in
during the GREEN phase against a curated adversarial fixture set.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from bba.prompt_builder.models import (
    EvidenceChunk,
    InjectionCategory,
    InjectionMatch,
    InjectionVerdict,
)


class InjectionPattern(NamedTuple):
    """One named injection-detection pattern.

    ``pattern_id`` is the stable identifier surfaced on
    :class:`InjectionMatch.pattern_id`. ``category`` tags the match for
    monitoring. ``regex`` is the pattern source (compiled lazily inside
    the scanner so import-time cost stays low). ``language`` is a
    coarse hint for documentation only — the scanner does not branch on
    it, but a corpus update can grep for "th" patterns to audit Thai
    coverage."""

    pattern_id: str
    category: InjectionCategory
    regex: str
    language: str


INJECTION_PATTERNS: tuple[InjectionPattern, ...] = ()
"""The shipped pattern catalog.

RED-phase: empty tuple (the GREEN phase fills it). The
:data:`bba.prompt_builder.MIN_REQUIRED_INJECTION_PATTERNS` invariant is
asserted by the test suite — the empty tuple is the RED-phase signal,
not a release shape.
"""


def scan_injection(*, evidence_id: str, text: str) -> tuple[InjectionMatch, ...]:
    """Scan one evidence chunk's text against :data:`INJECTION_PATTERNS`.

    Returns a tuple of :class:`InjectionMatch` (potentially empty). The
    function is pure — same input always yields the same output. The
    ``evidence_id`` is echoed onto every match so downstream aggregation
    can attribute the hit to the original chunk.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #21")


def scan_chunks(chunks: Sequence[EvidenceChunk]) -> InjectionVerdict:
    """Apply :func:`scan_injection` over every chunk and aggregate.

    Returns the :class:`InjectionVerdict` over the chunk sequence:
    ``flagged`` is ``True`` iff any chunk produced a match; ``matches``
    is the concatenated tuple of per-chunk hits in chunk-iteration
    order.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #21")
