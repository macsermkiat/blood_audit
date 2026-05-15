"""The six grounding layers of the quote_grounder, exposed individually.

Each layer is a pure predicate (or transformer for Layer 1) testable in
isolation against the issue #18 acceptance criteria:

1. :func:`nfc_normalize` — Unicode NFC normalization on both sides of the
   comparison. Defeats Thai NFC-vs-NFD adversarial fixtures.
2. :func:`contiguous_match` — quote is a contiguous substring of the cited
   source's redacted text (both NFC). Defeats concatenated-quote attacks.
3. :func:`find_cited_source` — strict ``cited_id`` match against the bundle.
   Defeats cross-source attribution attacks where the quote appears in
   some other source.
4. :func:`within_doc_unique` — the quote occurs exactly once in the cited
   source. Defeats short-common-phrase attacks ("no bleeding" appearing in
   unrelated context elsewhere in the same source).
5. :func:`min_length_ok` — NFC-length ≥ ``MIN_QUOTE_LENGTH``. Defeats the
   trivial-substring attack.
6. :func:`lab_tuple_match` — for lab citations, the (analyte, value, unit)
   tuple must be present in the cited source after analyte aliasing +
   unit normalization. Defeats numeric paraphrase attacks.

The combined ordered pipeline lives in :mod:`bba.quote_grounder.verifier`;
the layers themselves know nothing about each other.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.quote_grounder.models import EvidenceSource, LabTuple


MIN_QUOTE_LENGTH: int = 25
"""Minimum NFC-normalized character length for a citation quote (Layer 5).

PRD §9 ("minimum span ≥ 25 characters"). Below this threshold, even a
contiguous-substring match is suspect — a 24-character span has a high
probability of co-occurring in unrelated source content.
"""


def nfc_normalize(text: str) -> str:
    """Layer 1 — apply Unicode NFC normalization.

    Idempotent: ``nfc_normalize(nfc_normalize(x)) == nfc_normalize(x)``.
    Required because Thai medical text appears in both NFC and NFD forms in
    KCMH note exports; without normalization a visually-identical quote
    would fail :func:`contiguous_match` (issue #18 adversarial fixture).
    """
    raise NotImplementedError


def contiguous_match(quote: str, source_text: str) -> bool:
    """Layer 2 — quote is a contiguous substring of ``source_text``.

    Both arguments MUST be pre-normalized via :func:`nfc_normalize` by the
    caller; this layer does not re-normalize so the combined pipeline can
    keep normalization cost O(sources) instead of O(citations × sources).
    """
    raise NotImplementedError


def find_cited_source(
    cited_id: str, sources: Sequence[EvidenceSource]
) -> EvidenceSource | None:
    """Layer 3 — return the unique source whose ``source_id`` == ``cited_id``.

    Returns ``None`` when no source matches. A bundle with duplicate
    ``source_id`` values is rejected (the upstream evidence-bundle builder
    is contractually required to produce unique ids); the verifier treats
    ambiguous-cited_id as the same failure as missing-cited_id so the
    reviewer dashboard sees a single failure mode.
    """
    raise NotImplementedError


def within_doc_unique(quote: str, source_text: str) -> bool:
    """Layer 4 — ``quote`` occurs exactly once in ``source_text``.

    Both arguments MUST be pre-NFC-normalized. Returns ``False`` for zero
    occurrences AND for ≥ 2 occurrences; the latter is the boilerplate /
    short-common-phrase defense. A quote that legitimately repeats in the
    source is suspect (the LLM cannot disambiguate which occurrence it
    meant) and is rejected.
    """
    raise NotImplementedError


def min_length_ok(quote: str, *, minimum: int = MIN_QUOTE_LENGTH) -> bool:
    """Layer 5 — NFC-normalized character length ≥ ``minimum``.

    The caller is responsible for passing an NFC-normalized quote; this
    layer measures ``len(quote)`` directly. ``minimum`` is parameterized so
    the eval harness can sweep the threshold during calibration.
    """
    raise NotImplementedError


def lab_tuple_match(lab_tuple: LabTuple, source_text: str) -> bool:
    """Layer 6 — numeric tuple grounding for lab citations.

    Matches when ``source_text`` contains a (analyte, value, unit) triple
    equivalent to ``lab_tuple`` after analyte aliasing (e.g. ``"Hgb"`` ==
    ``"Hb"``) and unit normalization (e.g. ``"g/dl"`` == ``"g/dL"``). The
    value comparison is exact-equal on the parsed float — paraphrase is on
    the surrounding tokens, not the number itself.
    """
    raise NotImplementedError
