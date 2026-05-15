"""bba.quote_grounder — six-layer anti-hallucination verifier.

See issue #18 for acceptance criteria. PRD §9 (Implementation Decisions)
defines the layered grounding contract: NFC normalization, contiguous
substring match in the cited source's redacted text, strict cited_id match,
within-document uniqueness, minimum span ≥ 25 chars, numeric-tuple grounding
for lab citations, and an optional medical-NLI semantic-entailment gate.

This module is a *pure function* (no I/O, zero deps on the Anthropic SDK).
It is the failure-mode gate for the LLM-stage pipeline (#24): if every
Tier-1 citation in an indication list is rejected, the caller retries on
Sonnet and then escalates to Opus 4.7 (PRD §"Anti-hallucination is layered,
not single-shot.").
"""

from bba.quote_grounder.layers import (
    MIN_QUOTE_LENGTH,
    contiguous_match,
    extract_lab_triples,
    find_cited_source,
    lab_tuple_match,
    min_length_ok,
    nfc_normalize,
    within_doc_unique,
)
from bba.quote_grounder.metrics import confusion_matrix
from bba.quote_grounder.models import (
    Citation,
    ConfusionMatrix,
    EvidenceSource,
    LabTuple,
    NLIEntailmentGate,
    Verdict,
    VerdictReason,
    VerdictSequence,
    VerdictTuple,
)
from bba.quote_grounder.verifier import verify_citation, verify_citations

__all__ = [
    "MIN_QUOTE_LENGTH",
    "Citation",
    "ConfusionMatrix",
    "EvidenceSource",
    "LabTuple",
    "NLIEntailmentGate",
    "Verdict",
    "VerdictReason",
    "VerdictSequence",
    "VerdictTuple",
    "confusion_matrix",
    "contiguous_match",
    "extract_lab_triples",
    "find_cited_source",
    "lab_tuple_match",
    "min_length_ok",
    "nfc_normalize",
    "verify_citation",
    "verify_citations",
    "within_doc_unique",
]
