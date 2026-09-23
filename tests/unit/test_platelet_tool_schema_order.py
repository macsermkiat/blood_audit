"""Every tool schema asks for the label AFTER the reasoning (issues #239, #242).

A forced tool call is generated in schema order. With ``classification`` first,
the model committed to a label before writing its reasoning; on the first
real-data platelet run 22 of 297 answers carried ``APPROPRIATE`` while their own
reasoning concluded INAPPROPRIATE or NEEDS_REVIEW (e.g. "... however, the count
here is 22,000 ... Re-evaluating ... should be classified as INAPPROPRIATE").
The platelet schema was fixed first (#239). The consortium audit then found the
same defect in RBC answers (68023033, 68042514, 68055542: label APPROPRIATE,
reasoning INAPPROPRIATE / NEEDS_REVIEW), so the RBC and reserve-ahead schemas
follow (user ruling 2026-09-21; RBC verdicts on file predate this order).
"""

from __future__ import annotations

import pytest

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
    "aplastic_active_therapy_indication",
    "specialist_platelet_target_per_ul",
)
_RESERVE_FIELDS = (
    "administration_evidence",
    "administration_claimed",
    "reservation_assessment",
)
_BASE_FIELDS = (
    "indications",
    "negative_evidence",
    "reasoning_summary_en",
    "reasoning_summary_th",
)
_SCHEMAS = {
    "rbc": (_TOOL_INPUT_SCHEMA, ()),
    "platelet": (_PLATELET_TOOL_INPUT_SCHEMA, _HARD_SIGNALS),
    "reserve_ahead": (_RESERVE_AHEAD_TOOL_INPUT_SCHEMA, _RESERVE_FIELDS),
}


@pytest.mark.parametrize("name", sorted(_SCHEMAS))
def test_label_is_the_last_field_the_model_writes(name: str) -> None:
    schema, _extra = _SCHEMAS[name]

    assert list(schema["properties"])[-1] == "classification"
    assert list(schema["required"])[-1] == "classification"


@pytest.mark.parametrize("name", sorted(_SCHEMAS))
def test_label_follows_the_reasoning_and_every_other_judgment(name: str) -> None:
    # The label must be consistent with the other judgments too: the platelet
    # over-clear pattern was APPROPRIATE with every hard signal false, and the
    # reserve-ahead schema carries a second verdict (reservation_assessment).
    schema, extra = _SCHEMAS[name]
    properties = list(schema["properties"])

    assert properties == [*_BASE_FIELDS, *extra, "classification"]
    assert list(schema["required"]) == properties


@pytest.mark.parametrize("name", sorted(_SCHEMAS))
def test_label_field_says_in_words_that_it_follows_the_reasoning(name: str) -> None:
    # Sandbox rerun 2026-09-21: 3 of 26 platelet answers ignored the property
    # order, wrote the label first, and all 3 contradicted themselves. The
    # order is a hint to the model, so the field also states the rule.
    label = _SCHEMAS[name][0]["properties"]["classification"]

    assert label["enum"] == [
        "APPROPRIATE",
        "INAPPROPRIATE",
        "NEEDS_REVIEW",
        "INSUFFICIENT_EVIDENCE",
    ]
    assert "LAST" in label["description"]
    assert "reasoning_summary_en" in label["description"]


def test_shared_fields_keep_one_definition_across_schemas() -> None:
    for name in _BASE_FIELDS:
        definition = _TOOL_INPUT_SCHEMA["properties"][name]
        assert _PLATELET_TOOL_INPUT_SCHEMA["properties"][name] == definition
        assert _RESERVE_AHEAD_TOOL_INPUT_SCHEMA["properties"][name] == definition


@pytest.mark.parametrize("signal", _HARD_SIGNALS)
def test_hard_signals_say_in_words_that_they_follow_the_reasoning(signal: str) -> None:
    # Sandbox 2026-09-21 (68042732, 68051598): once the label was pinned last,
    # the model moved the four booleans to the FRONT, set
    # prophylactic_marrow_failure true for a chemotherapy patient, then reasoned
    # that 17,000 /uL is above the 10,000 threshold. Same defect as the label,
    # same remedy: the field states when it is written.
    description = _PLATELET_TOOL_INPUT_SCHEMA["properties"][signal]["description"]

    assert "after both reasoning summaries" in description
    assert "reasoning_summary_en" in description
