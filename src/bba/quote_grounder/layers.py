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


_UNITS: tuple[str, ...] = (
    "g/dL",
    "g/dl",
    "K/uL",
    "k/ul",
    "mmol/L",
    "mmol/l",
)
"""Recognised lab unit literals — surface forms only.

Canonical comparison is whitespace-stripped lowercase via
:func:`_canonical_unit`, so ``"g/dL"`` and ``"g/dl"`` match the same source.
The list is the regex alternation; surface forms here must stay in sync
with the analyte family they accompany (Hb → g/dL/mmol/L; WBC → K/uL).
"""


_LAB_TRIPLE_RE = re.compile(
    r"\b(?P<analyte>Hgb|Hb|hemoglobin|WBC|Platelets|Plt)\s*[:=]?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>g/dL|g/dl|K/uL|k/ul|mmol/L|mmol/l)\b",
    re.IGNORECASE,
)
"""Regex over a source text to capture every (analyte, value, unit) triple.

The unit must follow the value with at most whitespace in between; this
keeps "Hb 8.3 (admission)" from claiming a triple via a trailing token.
``case insensitive`` so "HGB" and "hgb" extract the same way.
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
    """Layer 2 — quote is a contiguous substring of ``source_text``.

    Both arguments MUST be pre-normalized via :func:`nfc_normalize` by the
    caller; this layer does not re-normalize so the combined pipeline can
    keep normalization cost O(sources) instead of O(citations × sources).

    An empty ``quote`` returns ``False`` even though ``"" in s`` is ``True``
    in Python — defense in depth against the trivial-substring backdoor.
    """
    if not quote:
        return False
    return quote in source_text


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
    matches = [s for s in sources if s.source_id == cited_id]
    if len(matches) != 1:
        return None
    return matches[0]


def within_doc_unique(quote: str, source_text: str) -> bool:
    """Layer 4 — ``quote`` occurs exactly once in ``source_text``.

    Both arguments MUST be pre-NFC-normalized. Returns ``False`` for zero
    occurrences AND for ≥ 2 occurrences; the latter is the boilerplate /
    short-common-phrase defense. A quote that legitimately repeats in the
    source is suspect (the LLM cannot disambiguate which occurrence it
    meant) and is rejected.
    """
    if not quote:
        return False
    return source_text.count(quote) == 1


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
    """
    cited_analyte = _canonical_analyte(lab_tuple.analyte)
    if cited_analyte is None:
        return False
    cited_unit = _canonical_unit(lab_tuple.unit)
    target = (cited_analyte, lab_tuple.value, cited_unit)
    return target in _extract_lab_triples(source_text)


def _canonical_analyte(name: str) -> str | None:
    """Return the canonical analyte key for ``name`` (None if unknown).

    Layer 6's analyte aliasing: "Hgb" and "Hb" collapse to the same key so
    a paraphrased LLM emission matches a source phrasing. An unknown
    analyte returns None — the citation cannot be tuple-grounded.
    """
    return _ANALYTE_ALIASES.get(name.lower().strip())


def _canonical_unit(unit: str) -> str:
    """Normalize a unit literal for tuple comparison.

    Lowercases and strips internal whitespace. ``"g/dL"`` and ``"g/dl"`` and
    ``"g / dL"`` all collapse to ``"g/dl"``. Cross-family units ("g/dl" vs
    "mmol/l") stay distinct — the regulator's "different unit, different
    semantics" contract still holds at the canonical-form layer.
    """
    return "".join(unit.split()).lower()


def _extract_lab_triples(source: str) -> list[tuple[str, float, str]]:
    """Return every (canonical_analyte, value, canonical_unit) triple in ``source``.

    Uses :data:`_LAB_TRIPLE_RE` to find analyte-value-unit runs. The list is
    the source's grounded-triple inventory; Layer 6 checks tuple-equivalence
    against this set.
    """
    triples: list[tuple[str, float, str]] = []
    for m in _LAB_TRIPLE_RE.finditer(source):
        canonical = _canonical_analyte(m.group("analyte"))
        if canonical is None:
            continue
        triples.append(
            (canonical, float(m.group("value")), _canonical_unit(m.group("unit")))
        )
    return triples
