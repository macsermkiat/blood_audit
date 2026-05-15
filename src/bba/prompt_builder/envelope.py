"""``<evidence>`` envelope wrapping for the prompt builder's user payload.

PRD §38 + issue #21 scope: "Wraps each evidence chunk in
``<evidence id="E1" untrusted="true">...</evidence>``". The ``untrusted="true"``
attribute is a fixed signal to the LLM — every redacted bundle item has
crossed the deid boundary and must be treated as adversarial regardless
of source.

The wrapper additionally XML-escapes the content (``&``, ``<``, ``>``) so
a chunk text containing literal ``<evidence>`` substrings cannot break
the envelope shape the LLM parses.

RED-phase scaffold: every public function below raises
:class:`NotImplementedError` so the test suite goes red on contract
assertions, not on import failure.
"""

from __future__ import annotations

from bba.prompt_builder.models import EvidenceChunk


def wrap_evidence(chunk: EvidenceChunk) -> str:
    """Return the ``<evidence>``-wrapped form of ``chunk``.

    Output shape: ``<evidence id="{evidence_id}" untrusted="true">{escaped_text}</evidence>``.
    The text is NFC-normalized and XML-escaped (``&`` -> ``&amp;``,
    ``<`` -> ``&lt;``, ``>`` -> ``&gt;``). The ``evidence_id`` is NOT
    escaped — Pydantic's pattern validator already constrains it to
    ``E\\d+``.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #21")


def wrap_evidence_chunks(chunks: tuple[EvidenceChunk, ...]) -> str:
    """Wrap every chunk and join with a single newline.

    A trailing newline is NOT appended — the join is between-only so the
    canonical bytes are byte-stable for hashing. Empty input yields the
    empty string.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #21")
