"""The tool-call output limit leaves room for the label, which is written last.

Sandbox 2026-09-21: 3 of 107 answers stopped at the old 4,096-token limit. Every
schema asks for `classification` LAST (#239, #242), so a cut-off answer carries
no label and the order falls to schema_mismatch / NEEDS_REVIEW. Long bilingual
summaries with several quoted indications need more headroom.
"""

from __future__ import annotations

from bba.llm_client.transport import MAX_OUTPUT_TOKENS


def test_output_limit_has_headroom_for_bilingual_reasoning_plus_the_label() -> None:
    assert MAX_OUTPUT_TOKENS >= 8192
