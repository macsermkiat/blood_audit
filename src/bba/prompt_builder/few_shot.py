"""Few-shot block assembly.

The few-shot block is a single ``user``-role string concatenating
committee-approved exemplars. It is the LAST cacheable block (Anthropic
prompt-cache marker boundary), so its byte stability across audit rows
is required for cache-hit rate (PRD §"Stack" — Anthropic Batch API +
prompt caching).

RED-phase scaffold: :func:`build_few_shot_block` raises
:class:`NotImplementedError`.
"""

from __future__ import annotations

from bba.prompt_builder.models import FewShotExample


def build_few_shot_block(examples: tuple[FewShotExample, ...]) -> str:
    """Concatenate ``examples`` into a single cacheable user-role text block.

    Output contract:

    * NFC-normalized.
    * Empty ``examples`` yields the empty string (the caller's
      :func:`build_prompt` then omits the few-shot block entirely).
    * Stable ordering: input order is preserved (the clinical committee
      ranks examples, and reordering would change the LLM's
      prioritization signal).
    * No trailing newline; between-example separator is exactly
      ``"\\n\\n"`` so the byte boundary is unambiguous when the block
      lands inside the cache boundary.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #21")
