"""Top-level grounding pipeline: combine the six (+1 optional) layers.

:func:`verify_citation` is the single-citation entry point; it short-circuits
on the first failed layer and returns a :class:`Verdict` whose ``reason``
names that layer. :func:`verify_citations` is the batch convenience over an
LLM-output indication list — it preserves input order and never raises on a
single rejection (rejections are values, not exceptions).

Canonical layer order (the order the verifier short-circuits in):

1. EMPTY_QUOTE — trivial guard
2. CITED_ID_NOT_FOUND — Layer 3 (no source to compare against)
3. TOO_SHORT — Layer 5 (cheap reject before substring scan)
4. NO_CONTIGUOUS_MATCH — Layer 2
5. NOT_UNIQUE — Layer 4 (only reached when the substring exists)
6. LAB_TUPLE_MISMATCH — Layer 6 (only when the citation carries a tuple)
7. NLI_NOT_ENTAILED — Layer 7 (only when an NLI gate is provided)

The order is part of the contract: tests assert that, given a citation that
violates layers 3 and 5 simultaneously, the verdict's ``reason`` is
``CITED_ID_NOT_FOUND`` (the higher-ranked layer). Re-ordering means
re-classifying every failure mode the reviewer dashboard already labels.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.quote_grounder.models import (
    Citation,
    NLIEntailmentGate,
    Verdict,
    VerdictTuple,
)


def verify_citation(
    citation: Citation,
    sources: Sequence,  # Sequence[EvidenceSource] — kept generic to avoid a circular import at scaffold time
    *,
    nli_gate: NLIEntailmentGate | None = None,
    min_length: int | None = None,
) -> Verdict:
    """Run all applicable grounding layers against one citation.

    Returns a :class:`Verdict` whose ``passed`` is ``True`` only when every
    applicable layer accepts. ``nli_gate`` is opt-in (PRD §9 explicitly
    allows omitting Layer 7); ``min_length`` overrides the module default
    of :data:`bba.quote_grounder.layers.MIN_QUOTE_LENGTH` (eval-harness
    sweep parameter).

    The function is pure: no I/O, no global mutation, no logging.
    Identical inputs yield identical outputs.
    """
    raise NotImplementedError


def verify_citations(
    citations: Sequence[Citation],
    sources: Sequence,  # Sequence[EvidenceSource]
    *,
    nli_gate: NLIEntailmentGate | None = None,
    min_length: int | None = None,
) -> VerdictTuple:
    """Batch convenience: verify every citation, preserve input order.

    Returns a tuple of verdicts ``len(verdicts) == len(citations)``. The
    output is a tuple (not a list) so the caller cannot mutate the result
    in place — mirroring the immutability contract of the audit_store
    persisted record (``indications_json``).
    """
    raise NotImplementedError
