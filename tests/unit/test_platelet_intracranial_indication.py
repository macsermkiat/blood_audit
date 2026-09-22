"""Intracranial bleed as a platelet indication (clinician ruling 2026-09-21).

Ruling, from review of hematology case REQNO 68001379 (AML, acute-on-chronic
bilateral SDH, count 24,000 /uL, neurosurgery "keep plt > 100,000"):

* acute, acute-on-chronic, growing / expanding, or operative intracranial
  bleed: keep the count above 100,000 /uL. A growing SDH qualifies because
  emergency surgery may follow and platelets take time to prepare.
* stable, non-operative intracranial bleed, including a stable chronic SDH:
  keep the count above 50,000 /uL.
* expansion of a chronic SDH is NOT "active bleeding" for this audit, so it
  must not be reported under active_bleeding. (The prompt states the rule
  without naming a mechanism; the growth mechanism is debated.)

Before the ruling the prompt named intracranial bleeds only as an exclusion
above 100,000 /uL, and about half of the orders discussing one ended as
NEEDS_REVIEW.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from bba.llm_client import SONNET_MODEL_ID, BatchSubmissionResult
from bba.llm_client.models import PlateletLlmClassificationResponse
from bba.llm_client.parser import parse_platelet_structured_response
from bba.llm_client.transport import _PLATELET_TOOL_INPUT_SCHEMA
from bba.platelet_guardrail.guardrail import platelet_overclear_suspect
from bba.platelet_guardrail.models import PlateletHardSignals
from bba.prompt_builder.system_prompt import platelet_system_prompt, system_prompt_for


def _tool_input(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "classification": "APPROPRIATE",
        "indications": [
            {
                "code": "PLT.intracranial_bleed",
                "quote": "Suggest keep plt > 100,000",
                "source_id": "E17",
                "confidence": 0.9,
            }
        ],
        "negative_evidence": [],
        "reasoning_summary_en": "Acute-on-chronic SDH below the 100,000 threshold.",
        "reasoning_summary_th": "SDH เฉียบพลันบนเรื้อรัง เกล็ดเลือดต่ำกว่าเกณฑ์",
        "active_bleeding": False,
        "procedure_indication": False,
        "prophylactic_marrow_failure": False,
    }
    return {**base, **overrides}


def _batch_result(tool_input: dict[str, Any]) -> BatchSubmissionResult:
    return BatchSubmissionResult(
        custom_id="audit-plt-ich",
        model_id=SONNET_MODEL_ID,  # type: ignore[arg-type]
        raw_response_json={
            "id": "msg_plt_ich",
            "type": "message",
            "role": "assistant",
            "model": SONNET_MODEL_ID,
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_plt_ich",
                    "name": "classify_transfusion_order",
                    "input": tool_input,
                }
            ],
        },
        request_json={"model": SONNET_MODEL_ID, "messages": []},
        response_headers={"anthropic-version": "2023-06-01"},
        request_timestamp=datetime(2026, 9, 21, 12, 0, 0, tzinfo=UTC),
        latency_ms=900,
        anthropic_version="2023-06-01",
    )


class TestPromptStatesTheRuling:
    def test_acute_growing_or_operative_bleed_uses_100k(self) -> None:
        prompt = platelet_system_prompt()
        assert "  6. Intracranial bleed" in prompt
        section = prompt.split("  6. Intracranial bleed", 1)[1].split("\n\n", 1)[0]
        assert "<100,000 /μL" in section
        for word in ("acute", "acute-on-chronic", "growing", "neurosurgery"):
            assert word in section

    def test_stable_non_operative_bleed_uses_50k(self) -> None:
        section = (
            platelet_system_prompt()
            .split("  6. Intracranial bleed", 1)[1]
            .split("\n\n", 1)[0]
        )
        assert "<50,000 /μL" in section
        assert "stable" in section and "chronic SDH" in section

    def test_chronic_sdh_expansion_is_not_active_bleeding(self) -> None:
        # The clinician's correction: radiographic progression of a chronic SDH
        # must not be reported as active bleeding.
        prompt = platelet_system_prompt()
        assert "is NOT active bleeding" in prompt
        assert "intracranial_bleed_indication" in prompt

    def test_terminal_line_counts_indication_6(self) -> None:
        # Codex P1 on #238: the closing INAPPROPRIATE rule listed only
        # indications 4 and 5, so a stable intracranial bleed below 50,000
        # matched both "INAPPROPRIATE" and indication 6. The guardrail cannot
        # repair a wrong INAPPROPRIATE, so the prompt must not contradict itself.
        prompt = platelet_system_prompt()
        # Widened to 4–7 with the 2026-09-22 rulings (active bleeding became
        # indication 7); the point stands: the terminal line must name every
        # non-procedure indication so it cannot contradict one of them.
        assert "no indication 4–8 below its threshold" in prompt
        assert "no indication 4 or 5 below its threshold" not in prompt
        assert "no indication 4, 5 or 6 below its threshold" not in prompt

    def test_exclusion_above_100k_is_kept(self) -> None:
        assert "intracranial bleed with platelet count >100,000" in (
            platelet_system_prompt()
        )

    def test_rbc_prompt_is_untouched(self) -> None:
        rbc = system_prompt_for(task_mode="HB_7_10_REVIEW", cohort_threshold=7.0)
        assert "intracranial_bleed_indication" not in rbc


class TestSignalReachesTheGuardrail:
    def test_intracranial_signal_alone_supports_an_appropriate_verdict(self) -> None:
        # Without its own signal a correct clear would have to be mislabelled
        # as active_bleeding, or be floored to NEEDS_REVIEW by the guardrail.
        outcome = parse_platelet_structured_response(
            _batch_result(_tool_input(intracranial_bleed_indication=True))
        )

        assert outcome.parse_failure is False
        signals = outcome.platelet_hard_signals
        assert signals is not None
        assert signals.intracranial_bleed_indication is True
        assert signals.active_bleeding is False
        assert signals.any_signal() is True
        assert (
            platelet_overclear_suspect(
                final_classification="APPROPRIATE",
                rule_classification="NEEDS_REVIEW",
                hard_signals=signals,
            )
            is False
        )

    def test_no_signal_still_floors_the_clear(self) -> None:
        signals = PlateletHardSignals()
        assert (
            platelet_overclear_suspect(
                final_classification="APPROPRIATE",
                rule_classification="NEEDS_REVIEW",
                hard_signals=signals,
            )
            is True
        )

    def test_stored_three_signal_responses_still_parse(self) -> None:
        # Responses persisted before this change carry three booleans; replay
        # of the audit store must not turn them into SCHEMA_MISMATCH.
        response = PlateletLlmClassificationResponse.model_validate(_tool_input())
        assert response.intracranial_bleed_indication is False

    def test_tool_schema_requires_the_new_signal(self) -> None:
        # New requests must always carry it, so the model cannot omit it.
        assert (
            "intracranial_bleed_indication"
            in (_PLATELET_TOOL_INPUT_SCHEMA["properties"])
        )
        assert (
            "intracranial_bleed_indication" in _PLATELET_TOOL_INPUT_SCHEMA["required"]
        )
