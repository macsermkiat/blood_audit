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

from bba.quote_grounder.layers import (
    MIN_QUOTE_LENGTH,
    contiguous_match,
    find_cited_source,
    lab_tuple_match,
    min_length_ok,
    nfc_normalize,
    within_doc_unique,
)
from bba.quote_grounder.models import (
    Citation,
    EvidenceSource,
    NLIEntailmentGate,
    Verdict,
    VerdictReason,
    VerdictTuple,
)


def verify_citation(
    citation: Citation,
    sources: Sequence[EvidenceSource],
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
    threshold = MIN_QUOTE_LENGTH if min_length is None else min_length
    reason = _evaluate(citation, sources, nli_gate=nli_gate, min_length=threshold)
    return Verdict(
        passed=reason is VerdictReason.PASS, reason=reason, citation=citation
    )


def verify_citations(
    citations: Sequence[Citation],
    sources: Sequence[EvidenceSource],
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
    return tuple(
        verify_citation(c, sources, nli_gate=nli_gate, min_length=min_length)
        for c in citations
    )


def _evaluate(
    citation: Citation,
    sources: Sequence[EvidenceSource],
    *,
    nli_gate: NLIEntailmentGate | None,
    min_length: int,
) -> VerdictReason:
    """Run the layers in canonical order; return the first-failed reason.

    Kept as a separate function from :func:`verify_citation` so the Verdict
    construction sits in one place and the layer logic stays a pure
    "input → reason" computation. The short-circuit order is the contract;
    moving any branch up or down re-labels failure modes.
    """
    if not citation.quote:
        return VerdictReason.EMPTY_QUOTE

    source = find_cited_source(citation.cited_id, sources)
    if source is None:
        return VerdictReason.CITED_ID_NOT_FOUND

    quote_n = nfc_normalize(citation.quote)
    if not min_length_ok(quote_n, minimum=min_length):
        return VerdictReason.TOO_SHORT

    source_n = nfc_normalize(source.text)
    if not contiguous_match(quote_n, source_n):
        return VerdictReason.NO_CONTIGUOUS_MATCH

    if not within_doc_unique(quote_n, source_n):
        return VerdictReason.NOT_UNIQUE

    if citation.lab_tuple is not None and not lab_tuple_match(
        citation.lab_tuple, source_n
    ):
        return VerdictReason.LAB_TUPLE_MISMATCH

    if nli_gate is not None and not nli_gate(premise=source_n, hypothesis=quote_n):
        return VerdictReason.NLI_NOT_ENTAILED

    return VerdictReason.PASS
