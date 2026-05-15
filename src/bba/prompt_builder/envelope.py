"""``<evidence>`` envelope wrapping for the prompt builder's user payload.

PRD §38 + issue #21 scope: "Wraps each evidence chunk in
``<evidence id="E1" untrusted="true">...</evidence>``". The ``untrusted="true"``
attribute is a fixed signal to the LLM — every redacted bundle item has
crossed the deid boundary and must be treated as adversarial regardless
of source.

**Byte-identity contract:** chunk text is NFC-normalized and inserted
verbatim into the envelope — no XML escaping of ``<``, ``>``, or ``&``.
Clinical text routinely contains these characters (``Hb < 8``, ``SBP >
90``, ``K&Na panel``); escaping them would make the LLM see a different
byte sequence than the redacted source text that
:mod:`bba.quote_grounder` verifies citations against, silently failing
all grounding for affected chunks (codex GitHub bot review #43).

**Envelope-escape attacks are handled upstream by the injection scanner**
(:mod:`bba.prompt_builder.injection` — ``ENVELOPE_ESCAPE`` category):
chunks containing literal ``</evidence>`` or ``<evidence `` boundary
tokens flag the row to ``NEEDS_REVIEW`` before the prompt is assembled,
so adversarial envelope escapes never reach this wrapper.
"""

from __future__ import annotations

import unicodedata

from bba.prompt_builder.models import (
    EVIDENCE_TAG_CLOSE,
    EVIDENCE_TAG_OPEN_TEMPLATE,
    EvidenceChunk,
)


def wrap_evidence(chunk: EvidenceChunk) -> str:
    """Return the ``<evidence>``-wrapped form of ``chunk`` with byte-identical content.

    The text is NFC-normalized; no other transformation is applied. See
    module docstring for why escaping is intentionally absent.
    """
    nfc_text = unicodedata.normalize("NFC", chunk.text)
    return (
        f"{EVIDENCE_TAG_OPEN_TEMPLATE.format(evidence_id=chunk.evidence_id)}"
        f"{nfc_text}{EVIDENCE_TAG_CLOSE}"
    )


def wrap_evidence_chunks(chunks: tuple[EvidenceChunk, ...]) -> str:
    """Wrap every chunk and join with a single newline (no trailing newline)."""
    if not chunks:
        return ""
    return "\n".join(wrap_evidence(c) for c in chunks)
