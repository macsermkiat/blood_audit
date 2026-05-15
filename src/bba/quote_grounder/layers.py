"""The six grounding layers of the quote_grounder, exposed individually.

Each layer is a pure predicate (or transformer for Layer 1) testable in
isolation against the issue #18 acceptance criteria:

1. :func:`nfc_normalize` — Unicode NFC normalization on both sides of the
   comparison. Defeats Thai NFC-vs-NFD adversarial fixtures.
2. :func:`contiguous_match` — quote is a word-boundary-anchored contiguous
   substring of the cited source's redacted text (both NFC). Defeats
   concatenated-quote attacks and 1-char-shift attacks where deleting a
   boundary character produces a still-contained but mid-word run.
3. :func:`find_cited_source` — strict ``cited_id`` match against the bundle.
   Defeats cross-source attribution attacks where the quote appears in
   some other source.
4. :func:`within_doc_unique` — the quote occurs at exactly one word-boundary-
   anchored position in the cited source. Defeats short-common-phrase
   attacks ("no bleeding" appearing in unrelated context elsewhere in the
   same source).
5. :func:`min_length_ok` — NFC-length ≥ ``MIN_QUOTE_LENGTH``. Defeats the
   trivial-substring attack.
6. :func:`lab_tuple_match` — for lab citations, the (analyte, value, unit)
   tuple must be present in the cited source after analyte aliasing +
   unit normalization. Defeats numeric paraphrase attacks. The full
   pipeline (in :mod:`bba.quote_grounder.verifier`) additionally extracts
   triples from the quote itself and requires the LLM-supplied
   ``citation.lab_tuple`` (if any) to match a tuple parsed from the quote
   — guarding against the hallucination "quote one Hb value but emit a
   different one in the structured tuple".

The combined ordered pipeline lives in :mod:`bba.quote_grounder.verifier`;
the layers themselves know nothing about each other.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from bba.quote_grounder.models import EvidenceSource, LabTuple


_ANALYTE_ALIASES: dict[str, str] = {
    "hb": "hb",
    "hgb": "hb",
    "hemoglobin": "hb",
    "wbc": "wbc",
    "platelets": "plt",
    "plt": "plt",
}
"""Map every accepted phrasing of a lab analyte to a canonical key.

Layer 6 collapses "Hgb" / "Hb" / "hemoglobin" to one key so paraphrase on
the analyte name (PRD §9 fixture: "Hgb=8.3 vs Hb 8.3 g/dL") does not bypass
tuple-equivalence. Keys are stored lowercase; the lookup also lowercases.
"""


_LAB_TRIPLE_RE = re.compile(
    r"(?P<analyte>\bHgb|\bHb|\bhemoglobin|\bWBC|\bPlatelets|\bPlt)\s*[:=]?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>g\s*/\s*dL|K\s*/\s*uL|mmol\s*/\s*L)\b",
    re.IGNORECASE,
)
"""Regex over a text to capture every (analyte, value, unit) triple.

The unit alternation tolerates optional whitespace around the ``/``
separator (``"g / dL"`` is canonicalized by :func:`_canonical_unit` to
``"g/dl"``, matching ``"g/dL"``). Case-insensitive so "HGB" and "hgb" and
"Hgb" all extract the same canonical analyte key.
"""


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
    return unicodedata.normalize("NFC", text)


def contiguous_match(quote: str, source_text: str) -> bool:
    """Layer 2 — quote is a word-boundary-anchored substring of ``source_text``.

    Both arguments MUST be pre-normalized via :func:`nfc_normalize` by the
    caller; this layer does not re-normalize so the combined pipeline can
    keep normalization cost O(sources) instead of O(citations × sources).

    Word-boundary anchoring: the quote must start at the beginning of the
    source or after a non-alphanumeric character, and end at the end of the
    source or before a non-alphanumeric character. This rejects the 1-char-
    deletion shift attack — deleting the first ``P`` of "Patient has melena…"
    yields "atient has melena…" which is still a substring but is now
    pinned mid-word and the adjacent ``P`` in the source flags it as a
    shifted citation.

    An empty ``quote`` returns ``False`` even though ``"" in s`` is ``True``
    in Python — defense in depth against the trivial-substring backdoor.
    """
    if not quote:
        return False
    return len(_aligned_positions(quote, source_text)) >= 1


def find_cited_source(
    cited_id: str, sources: Sequence[EvidenceSource]
) -> EvidenceSource | None:
    """Layer 3 — return the unique source whose ``source_id`` == ``cited_id``.

    Returns ``None`` when no source matches AND when ≥ 2 sources match. A
    bundle with duplicate ``source_id`` values is rejected (the upstream
    evidence-bundle builder is contractually required to produce unique
    ids); the verifier treats ambiguous-cited_id as the same failure as
    missing-cited_id so the reviewer dashboard sees a single failure mode.
    """
    matches = [s for s in sources if s.source_id == cited_id]
    if len(matches) != 1:
        return None
    return matches[0]


def within_doc_unique(quote: str, source_text: str) -> bool:
    """Layer 4 — ``quote`` occurs at exactly one aligned position in source.

    Both arguments MUST be pre-NFC-normalized. Counts only word-boundary-
    anchored occurrences (same definition as :func:`contiguous_match`).
    Returns ``False`` for zero occurrences AND for ≥ 2 — the latter is the
    boilerplate / short-common-phrase defense. A quote that legitimately
    repeats in the source is suspect (the LLM cannot disambiguate which
    occurrence it meant) and is rejected.
    """
    if not quote:
        return False
    return len(_aligned_positions(quote, source_text)) == 1


def min_length_ok(quote: str, *, minimum: int = MIN_QUOTE_LENGTH) -> bool:
    """Layer 5 — NFC-normalized character length ≥ ``minimum``.

    The caller is responsible for passing an NFC-normalized quote; this
    layer measures ``len(quote)`` directly. ``minimum`` is parameterized so
    the eval harness can sweep the threshold during calibration.
    """
    return len(quote) >= minimum


def lab_tuple_match(lab_tuple: LabTuple, source_text: str) -> bool:
    """Layer 6 — numeric tuple grounding for lab citations.

    Matches when ``source_text`` contains a (analyte, value, unit) triple
    equivalent to ``lab_tuple`` after analyte aliasing (e.g. ``"Hgb"`` ==
    ``"Hb"``) and unit normalization (e.g. ``"g/dl"`` == ``"g/dL"``). The
    value comparison is exact-equal on the parsed float — paraphrase is on
    the surrounding tokens, not the number itself.

    Note: the full verifier pipeline (:func:`bba.quote_grounder.verify_citation`)
    pairs this against :func:`extract_lab_triples` over the quote itself to
    additionally require the supplied tuple to match a triple parsed from
    the quote, not just from somewhere in the source. That stricter check
    lives in the verifier so layer-level callers can use the more permissive
    "tuple appears in source" semantic when they need it.
    """
    cited_analyte = _canonical_analyte(lab_tuple.analyte)
    if cited_analyte is None:
        return False
    cited_unit = _canonical_unit(lab_tuple.unit)
    target = (cited_analyte, lab_tuple.value, cited_unit)
    return target in extract_lab_triples(source_text)


def extract_lab_triples(text: str) -> list[tuple[str, float, str]]:
    """Return every (canonical_analyte, value, canonical_unit) triple in ``text``.

    Exposed on the public surface so the verifier (and downstream consumers)
    can extract triples from BOTH a citation's quote and the cited source
    text. The verifier requires:

    * every triple in the quote is present in the source (auto-satisfied
      when Layer 2 passes, but defense-in-depth)
    * any LLM-supplied ``citation.lab_tuple`` matches a triple parsed from
      the QUOTE — guarding against "quote one Hb value but emit a different
      one in the structured tuple" hallucinations.
    """
    triples: list[tuple[str, float, str]] = []
    for m in _LAB_TRIPLE_RE.finditer(text):
        canonical = _canonical_analyte(m.group("analyte"))
        if canonical is None:
            continue
        triples.append(
            (canonical, float(m.group("value")), _canonical_unit(m.group("unit")))
        )
    return triples


def _canonical_analyte(name: str) -> str | None:
    """Return the canonical analyte key for ``name`` (None if unknown).

    Layer 6's analyte aliasing: "Hgb" and "Hb" collapse to the same key so
    a paraphrased LLM emission matches a source phrasing. An unknown
    analyte returns None — the citation cannot be tuple-grounded.
    """
    return _ANALYTE_ALIASES.get(name.lower().strip())


def _canonical_unit(unit: str) -> str:
    """Normalize a unit literal for tuple comparison.

    Strips ALL whitespace (including spaces around the ``/`` separator) and
    lowercases. ``"g/dL"``, ``"g/dl"``, ``"g / dL"`` all collapse to
    ``"g/dl"``. Cross-family units ("g/dl" vs "mmol/l") stay distinct —
    the regulator's "different unit, different semantics" contract still
    holds at the canonical-form layer.
    """
    return "".join(unit.split()).lower()


def _aligned_positions(quote: str, source: str) -> list[int]:
    """Word-boundary-aligned occurrence positions of ``quote`` in ``source``.

    A position is aligned when:

    * It is 0, OR the source character immediately before it is not
      alphanumeric (Unicode :meth:`str.isalnum` — Latin letters, digits,
      and Thai/CJK characters all count as alphanumeric).
    * Quote ends at the end of source, OR the character immediately after
      is not alphanumeric.

    Used by Layer 2 (existence) and Layer 4 (uniqueness). The shared helper
    keeps the alignment definition in one place so a quote that passes one
    layer cannot fail the other for a different reason.
    """
    positions: list[int] = []
    start = 0
    while True:
        idx = source.find(quote, start)
        if idx == -1:
            return positions
        end = idx + len(quote)
        before_ok = idx == 0 or not source[idx - 1].isalnum()
        after_ok = end == len(source) or not source[end].isalnum()
        if before_ok and after_ok:
            positions.append(idx)
        start = idx + 1
