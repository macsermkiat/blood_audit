"""Read the class a reasoning summary concludes, so replay can check it against
the label the model returned.

Full hematology rerun 2026-09-22: 6 of 364 answers ended their reasoning on one
class and put another in ``classification``; five had written the label last,
so schema order does not prevent it. Two readers:

* :func:`closing_classification` reads the fixed sentence the prompt now asks
  for at the end of ``reasoning_summary_en`` ("Final classification: X").
  Binding when present.
* :func:`concluded_class` is the fallback for answers without that sentence
  (every answer stored before the prompt asked for it): the last class the
  text names without rejecting it. A candidate reader, not a proof: on the
  first real-data run a model consortium overruled 4 of its 10 hits.
"""

from __future__ import annotations

import re

CLASSES: tuple[str, ...] = (
    "POTENTIALLY_INAPPROPRIATE",
    "INSUFFICIENT_EVIDENCE",
    "INAPPROPRIATE",
    "NEEDS_REVIEW",
    "APPROPRIATE",
)
_CLOSING_RE = re.compile(
    r"final classification\s*:\s*\**\s*([A-Za-z_ ]+?)\s*\**\s*[.!]?\s*$", re.IGNORECASE
)
# Upper-case only: lower-case "appropriate" is ordinary prose.
_CLASS_RE = re.compile(
    r"\b(" + "|".join(c.replace("_", "[ _]") for c in CLASSES) + r")\b"
)
# A class is REJECTED when a cue sits right before it ("not X", "rather than a
# clean X") or when it continues a rejected list ("X or Y"). Only the mention
# is rejected, never the rest of the sentence: in "not APPROPRIATE but
# INAPPROPRIATE" the correction is the conclusion.
_FILLER = r"(?:[a-z][a-z-]*\s+){0,3}"
_CUE_RE = re.compile(
    r"(?:\brather than|\binstead of|\bas opposed to|\bnot|\bnor|\bnever)\s+"
    + _FILLER
    + r"$",
    re.IGNORECASE,
)
_CHAIN_RE = re.compile(
    r"\s*(?:,|/|\bor\b|\bnor\b|,\s*or\b)\s*" + _FILLER, re.IGNORECASE
)


def closing_classification(reasoning: str) -> str | None:
    """The class named by the closing "Final classification: X" sentence, or
    ``None`` when the summary does not end with one."""
    match = _CLOSING_RE.search(reasoning.strip())
    if match is None:
        return None
    named = match.group(1).strip().upper().replace(" ", "_")
    return named if named in CLASSES else None


def concluded_class(reasoning: str) -> str | None:
    """The last class the reasoning names without rejecting it, else ``None``."""
    concluded: str | None = None
    previous_end = 0
    previous_rejected = False
    for match in _CLASS_RE.finditer(reasoning):
        between = reasoning[previous_end : match.start()]
        rejected = bool(_CUE_RE.search(between)) or (
            previous_rejected and _CHAIN_RE.fullmatch(between) is not None
        )
        if not rejected:
            concluded = match.group(1).replace(" ", "_")
        previous_end, previous_rejected = match.end(), rejected
    return concluded


def reasoning_conclusion(reasoning: str) -> str | None:
    """What the reasoning concludes: the closing sentence when present, else
    the fallback reader."""
    return closing_classification(reasoning) or concluded_class(reasoning)


__all__ = (
    "CLASSES",
    "closing_classification",
    "concluded_class",
    "reasoning_conclusion",
)
