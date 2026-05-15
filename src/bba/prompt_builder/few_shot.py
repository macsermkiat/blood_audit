"""Few-shot block assembly.

The few-shot block is a single ``user``-role string concatenating
committee-approved exemplars. It is the LAST cacheable block (Anthropic
prompt-cache marker boundary), so its byte stability across audit rows
is required for cache-hit rate (PRD §"Stack" — Anthropic Batch API +
prompt caching).
"""

from __future__ import annotations

import unicodedata

from bba.prompt_builder.models import FewShotExample


def build_few_shot_block(examples: tuple[FewShotExample, ...]) -> str:
    """Concatenate ``examples`` into a single cacheable user-role text block.

    Output contract: NFC-normalized; empty input -> empty string; input
    order preserved; between-example separator is ``"\\n\\n"``; no
    trailing newline.
    """
    if not examples:
        return ""
    rendered_examples: list[str] = []
    for ex in examples:
        rendered = (
            f"Example: {ex.name}\n"
            f"<input>\n{ex.user_payload}\n</input>\n"
            f"<assistant_output>\n{ex.assistant_output}\n</assistant_output>"
        )
        rendered_examples.append(rendered)
    block = "\n\n".join(rendered_examples)
    return unicodedata.normalize("NFC", block)
