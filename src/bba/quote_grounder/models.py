"""Pydantic v2 models + enums + protocol for the quote_grounder module.

The quote grounder is a *pure function* (issue #18, PRD §9). Its public types
mirror the producer/consumer boundary:

* :class:`EvidenceSource` — one redacted source in the bundle. ``source_id``
  is the opaque ``cited_id`` the LLM is expected to reference (e.g. ``"E1"``);
  ``text`` is the redacted source text the LLM was shown.
* :class:`LabTuple` — the numeric grounding triple (analyte, value, unit)
  attached to a lab citation. Layer 6 verifies tuple-equivalence against the
  source's extracted tuples; the analyte aliasing ("Hgb" == "Hb") and unit
  normalization happen inside the grounding layer.
* :class:`Citation` — one LLM-output citation: a verbatim ``quote`` attributed
  to ``cited_id``, optionally carrying a :class:`LabTuple` when it references
  a structured lab value.
* :class:`Verdict` — the per-citation grounding outcome. ``passed`` is the
  pipeline's escalation signal; ``reason`` is the first-failed layer's tag,
  surfaced to the reviewer dashboard as the "why".
* :class:`ConfusionMatrix` — verifier-as-classifier output: the 2x2 contingency
  table the eval harness (#20) ingests for the 200-row hand-labeled set.
* :class:`NLIEntailmentGate` — duck-typed callable boundary for the optional
  medical-NLI layer (Layer 7). The grounder NEVER imports a model directly;
  callers supply the gate so the module remains pure-function and zero-deps
  on the Anthropic SDK or HuggingFace transformers (issue #18 AC).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict


class EvidenceSource(BaseModel):
    """One redacted source in the evidence bundle.

    The grounder treats ``text`` as opaque post-redaction string content;
    redaction itself happens upstream in :mod:`bba.deid_redactor`. ``source_id``
    is the contract identifier the LLM must echo back as a citation's
    ``cited_id``; strict equality on this field is Layer 3 of the verifier.
    """

    model_config = ConfigDict(frozen=True)

    source_id: str
    text: str


class LabTuple(BaseModel):
    """Numeric grounding triple for a lab citation.

    Layer 6 (issue #18 §9.f) compares this tuple against tuples extracted
    from the cited source. ``analyte`` is canonicalized by the grounder's
    aliasing table (e.g. ``"Hgb"`` / ``"Hb"`` / ``"hemoglobin"`` collapse to
    one canonical key) so an LLM phrasing skew ("Hgb=8.3") cannot reject a
    source phrasing skew ("Hb 8.3 g/dL").
    """

    model_config = ConfigDict(frozen=True)

    analyte: str
    value: float
    unit: str


class Citation(BaseModel):
    """One LLM-output citation: a quote attributed to a source by id.

    ``lab_tuple`` is the structured paraphrase the LLM emits for lab-value
    citations. The verifier ALWAYS evaluates Layer 6: it extracts triples
    from the quote text and requires every quote-triple to be present in
    the cited source. When ``lab_tuple`` is supplied, it must additionally
    match a triple parsed from the QUOTE itself (after analyte aliasing +
    unit canonicalization) — guarding against the hallucination "quote one
    Hb value but emit a different Hb value in the structured tuple".
    """

    model_config = ConfigDict(frozen=True)

    quote: str
    cited_id: str
    lab_tuple: LabTuple | None = None


class VerdictReason(StrEnum):
    """Mutually-exclusive grounding outcome tag.

    ``PASS`` is the one accepting state; every other value is the
    first-failed layer in the canonical layer order. The order matters: the
    verifier short-circuits on the FIRST failed layer so the reason maps
    1:1 to the explanation the reviewer dashboard surfaces.
    """

    PASS = "pass"
    EMPTY_QUOTE = "empty_quote"
    CITED_ID_NOT_FOUND = "cited_id_not_found"
    TOO_SHORT = "too_short"
    NO_CONTIGUOUS_MATCH = "no_contiguous_match"
    NOT_UNIQUE = "not_unique"
    LAB_TUPLE_MISMATCH = "lab_tuple_mismatch"
    NLI_NOT_ENTAILED = "nli_not_entailed"


class Verdict(BaseModel):
    """Outcome of grounding one citation against the evidence bundle.

    ``passed`` is the boolean the pipeline branches on; ``reason`` is the
    structured tag (``PASS`` when ``passed`` else the first-failed layer).
    The original ``citation`` is echoed so the verdict is self-describing
    on persistence — the audit row can store a ``tuple[Verdict, ...]``
    without losing the link to the source quote.
    """

    model_config = ConfigDict(frozen=True)

    passed: bool
    reason: VerdictReason
    citation: Citation


class ConfusionMatrix(BaseModel):
    """2x2 contingency table for the verifier-as-classifier evaluation.

    Convention: gold-positive == "the citation is genuinely grounded";
    predicted-positive == ``Verdict.passed``. The 200-row hand-labeled set
    (PRD §"Acceptance criteria" — verifier-as-classifier confusion matrix
    on 200 hand-labeled verdicts) is consumed by :mod:`bba.eval_harness`.
    """

    model_config = ConfigDict(frozen=True)

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


class NLIEntailmentGate(Protocol):
    """Callable boundary for the optional medical-NLI layer (Layer 7).

    The grounder invokes the gate as ``gate(premise=source_text,
    hypothesis=quote)`` and treats a truthy return as "entails". The PRD
    explicitly allows omitting this layer ("if unavailable, omit and accept
    the false-positive risk, with this being a publishable methodology gap
    rather than a launch blocker"), so the gate is always optional at the
    grounder API boundary.
    """

    def __call__(
        self, *, premise: str, hypothesis: str
    ) -> bool:  # pragma: no cover - protocol
        ...


VerdictTuple = tuple[Verdict, ...]
"""Output type alias of :func:`verify_citations` — a frozen sequence of verdicts.

Verdicts preserve input-citation order so a caller can zip them back to the
LLM's structured-output list without recomputing identities.
"""


VerdictSequence = Sequence[Verdict]
"""Input alias for :func:`confusion_matrix`. Accepts any ``Sequence[Verdict]``
so the eval harness can pass either the ``VerdictTuple`` returned by
:func:`verify_citations` or a list re-read from disk.
"""
