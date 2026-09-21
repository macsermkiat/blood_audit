"""The platelet tool schema asks for the label AFTER the reasoning (issue #239).

A forced tool call is generated in schema order. With ``classification`` first,
the model committed to a label before writing its reasoning; on the first
real-data platelet run 22 of 297 answers carried ``APPROPRIATE`` while their own
reasoning concluded INAPPROPRIATE or NEEDS_REVIEW (e.g. "... however, the count
here is 22,000 ... Re-evaluating ... should be classified as INAPPROPRIATE").
The guardrails floored all 22 to review, so the cost was a padded review queue
and an under-count of INAPPROPRIATE.
"""

from __future__ import annotations

import hashlib
import json

from bba.llm_client.transport import (
    _PLATELET_TOOL_INPUT_SCHEMA,
    _RESERVE_AHEAD_TOOL_INPUT_SCHEMA,
    _TOOL_INPUT_SCHEMA,
)

_HARD_SIGNALS = (
    "active_bleeding",
    "procedure_indication",
    "prophylactic_marrow_failure",
    "intracranial_bleed_indication",
)


def test_platelet_label_is_the_last_field_the_model_writes() -> None:
    properties = list(_PLATELET_TOOL_INPUT_SCHEMA["properties"])
    required = list(_PLATELET_TOOL_INPUT_SCHEMA["required"])

    assert properties[-1] == "classification"
    assert required[-1] == "classification"


def test_platelet_label_follows_the_reasoning_and_the_hard_signals() -> None:
    # The label must be consistent with the signals too: the over-clear pattern
    # was APPROPRIATE with every signal false.
    properties = list(_PLATELET_TOOL_INPUT_SCHEMA["properties"])
    label_at = properties.index("classification")

    for field in ("reasoning_summary_en", "reasoning_summary_th", *_HARD_SIGNALS):
        assert properties.index(field) < label_at


def test_platelet_schema_keeps_every_field_and_the_same_definitions() -> None:
    # A reorder only: the parser contract and the field descriptions must not move.
    assert set(_PLATELET_TOOL_INPUT_SCHEMA["properties"]) == {
        *_TOOL_INPUT_SCHEMA["properties"],
        *_HARD_SIGNALS,
    }
    assert set(_PLATELET_TOOL_INPUT_SCHEMA["required"]) == {
        *_TOOL_INPUT_SCHEMA["required"],
        *_HARD_SIGNALS,
    }
    for name, definition in _TOOL_INPUT_SCHEMA["properties"].items():
        if name == "classification":
            continue
        assert _PLATELET_TOOL_INPUT_SCHEMA["properties"][name] == definition


def test_platelet_label_field_says_in_words_that_it_follows_the_reasoning() -> None:
    # Sandbox rerun 2026-09-21: 3 of 26 answers ignored the property order, wrote
    # the label before the reasoning, and all 3 contradicted themselves. The
    # order is a hint to the model, so the field also states the rule.
    label = _PLATELET_TOOL_INPUT_SCHEMA["properties"]["classification"]

    assert label["enum"] == _TOOL_INPUT_SCHEMA["properties"]["classification"]["enum"]
    assert "LAST" in label["description"]
    assert "reasoning_summary_en" in label["description"]


def test_rbc_schemas_keep_their_field_order() -> None:
    # The RBC verdicts on file were produced with this exact request; this fix
    # is platelet-only.
    expected = [
        "classification",
        "indications",
        "negative_evidence",
        "reasoning_summary_en",
        "reasoning_summary_th",
    ]
    assert list(_TOOL_INPUT_SCHEMA["properties"]) == expected
    assert list(_TOOL_INPUT_SCHEMA["required"]) == expected
    assert list(_RESERVE_AHEAD_TOOL_INPUT_SCHEMA["properties"])[:5] == expected
    assert list(_RESERVE_AHEAD_TOOL_INPUT_SCHEMA["required"])[:5] == expected


def test_rbc_schemas_are_byte_identical_to_main() -> None:
    # Field order alone would let a description or enum edit slip through. These
    # digests were taken from main @ 3fa97e9; change them only when an RBC
    # prompt change is intended and the RBC verdicts on file are being re-run.
    def digest(schema: dict[str, object]) -> str:
        return hashlib.sha256(
            json.dumps(schema, ensure_ascii=False).encode()
        ).hexdigest()

    assert digest(_TOOL_INPUT_SCHEMA) == (
        "a9152445d7a67532a21ec9573b4a39293a3dffa23cdbb28d2e988fc8405b1164"
    )
    assert digest(_RESERVE_AHEAD_TOOL_INPUT_SCHEMA) == (
        "d17f078d31acfd3d3b83f64f0ed751b1bf5ee1087e7b2133ea7ea53f9f76a5ef"
    )
