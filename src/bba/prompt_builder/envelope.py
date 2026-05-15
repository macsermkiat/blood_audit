"""``<evidence>`` envelope wrapping for the prompt builder's user payload.

PRD §38 + issue #21 scope: "Wraps each evidence chunk in
``<evidence id="E1" untrusted="true">...</evidence>``". The ``untrusted="true"``
attribute is a fixed signal to the LLM — every redacted bundle item has
crossed the deid boundary and must be treated as adversarial regardless
of source.

The wrapper additionally XML-escapes the content (``&``, ``<``, ``>``) so
a chunk text containing literal ``<evidence>`` substrings cannot break
the envelope shape the LLM parses.
"""

from __future__ import annotations

import unicodedata
from xml.sax.saxutils import escape

from bba.prompt_builder.models import (
    EVIDENCE_TAG_CLOSE,
    EVIDENCE_TAG_OPEN_TEMPLATE,
    EvidenceChunk,
)


def wrap_evidence(chunk: EvidenceChunk) -> str:
    """Return the ``<evidence>``-wrapped form of ``chunk``.

    Output shape: ``<evidence id="{evidence_id}" untrusted="true">{escaped_text}</evidence>``.
    The text is NFC-normalized and XML-escaped (``&`` -> ``&amp;``,
    ``<`` -> ``&lt;``, ``>`` -> ``&gt;``).
    """
    nfc_text = unicodedata.normalize("NFC", chunk.text)
    escaped = escape(nfc_text)
    return (
        f"{EVIDENCE_TAG_OPEN_TEMPLATE.format(evidence_id=chunk.evidence_id)}"
        f"{escaped}{EVIDENCE_TAG_CLOSE}"
    )


def wrap_evidence_chunks(chunks: tuple[EvidenceChunk, ...]) -> str:
    """Wrap every chunk and join with a single newline (no trailing newline)."""
    if not chunks:
        return ""
    return "\n".join(wrap_evidence(c) for c in chunks)
