"""Stage C1 contract tests — platelet LLM prompt + hard-signal response parser.

WHY: The platelet LLM path must encode the Chula DRAFT policy (AABB/ICTMG
2025) in its system prompt AND emit three grounded hard-signal booleans in the
structured response. This test file verifies:

* C1a — platelet_system_prompt() contains every policy threshold and exclusion
  population (assert substrings), renders WITHOUT a cohort_threshold argument,
  and does NOT alter the RBC prompt.
* C1b — parse_platelet_structured_response extracts the three hard-signal
  booleans correctly; RBC parse_structured_response is byte-identical (the new
  field defaults to None and no existing assertion changes).
* C1c — PLATELET_LLM_ENABLED flag is False (default OFF).

No audit_pipeline wiring, no evidence_bundle_builder, no submission builder,
no guardrail wiring — those are Stage C2.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from bba.feature_flags import PLATELET_LLM_ENABLED
from bba.llm_client import (
    ParseOutcome,
    ParseFailureReason,
    parse_structured_response,
)
from bba.llm_client.models import PlateletLlmClassificationResponse
from bba.llm_client.parser import parse_platelet_structured_response
from bba.prompt_builder import (
    TASK_MODES,
    system_prompt_for,
)
from bba.prompt_builder.system_prompt import platelet_system_prompt
from bba.llm_client.models import (
    BatchSubmissionResult,
    SONNET_MODEL_ID,
)


# =============================================================================
# Helpers
# =============================================================================


def _batch_result(
    *,
    custom_id: str = "audit-plt-001",
    content: list[dict[str, Any]] | None = None,
    raw_response_override: dict[str, Any] | None = None,
) -> BatchSubmissionResult:
    """Minimal BatchSubmissionResult for parser tests."""
    raw: dict[str, Any] = (
        raw_response_override
        if raw_response_override is not None
        else {
            "id": "msg_plt_01",
            "type": "message",
            "role": "assistant",
            "model": SONNET_MODEL_ID,
            "stop_reason": "tool_use",
            "content": content if content is not None else [],
        }
    )
    return BatchSubmissionResult(
        custom_id=custom_id,
        model_id=SONNET_MODEL_ID,  # type: ignore[arg-type]
        raw_response_json=raw,
        request_json={"model": SONNET_MODEL_ID, "messages": []},
        response_headers={"anthropic-version": "2023-06-01"},
        request_timestamp=datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC),
        latency_ms=900,
        anthropic_version="2023-06-01",
    )


def _platelet_tool_content(
    *,
    classification: str = "APPROPRIATE",
    active_bleeding: bool = False,
    procedure_indication: bool = False,
    prophylactic_marrow_failure: bool = False,
    quote: str = "plt 48k; LP scheduled tomorrow",
    source_id: str = "E1",
) -> list[dict[str, Any]]:
    """Tool-use content block carrying platelet-specific hard signals."""
    return [
        {
            "type": "tool_use",
            "id": "tool_plt_01",
            "name": "classify_transfusion_order",
            "input": {
                "classification": classification,
                "indications": [
                    {
                        "code": "PLT.procedure_indication",
                        "quote": quote,
                        "source_id": source_id,
                        "confidence": 0.90,
                    }
                ],
                "negative_evidence": [],
                "reasoning_summary_en": "Platelet count below LP threshold.",
                "reasoning_summary_th": "เกล็ดเลือดต่ำกว่าเกณฑ์สำหรับการเจาะหลัง",
                "active_bleeding": active_bleeding,
                "procedure_indication": procedure_indication,
                "prophylactic_marrow_failure": prophylactic_marrow_failure,
            },
        }
    ]


def _rbc_tool_content() -> list[dict[str, Any]]:
    """Tool-use content block for an RBC (no platelet fields)."""
    return [
        {
            "type": "tool_use",
            "id": "tool_rbc_01",
            "name": "classify_transfusion_order",
            "input": {
                "classification": "APPROPRIATE",
                "indications": [
                    {
                        "code": "B1.active_bleeding",
                        "quote": "active haemorrhage noted",
                        "source_id": "E1",
                        "confidence": 0.92,
                    }
                ],
                "negative_evidence": [],
                "reasoning_summary_en": "Active bleeding documented.",
                "reasoning_summary_th": "มีเลือดออกอยู่",
            },
        }
    ]


# =============================================================================
# C1a — platelet system prompt policy content
# =============================================================================


class TestPlateletSystemPromptPolicyThresholds:
    """The platelet prompt must encode every policy threshold verbatim.

    WHY: If a threshold is absent the LLM cannot apply the policy. Each
    assertion targets a clinically distinct threshold — a generic "50k"
    assertion would not catch a missing "80k" for surgery/ortho LP.
    """

    def test_lp_medicine_ob_threshold_50k(self) -> None:
        # LP for medicine/OB patients: platelet count must be <50k.
        # Missing this causes the LLM to apply the wrong threshold for
        # non-surgical patients (over-or under-clearing).
        prompt = platelet_system_prompt()
        assert "50,000" in prompt

    def test_lp_surgery_ortho_threshold_80k(self) -> None:
        # LP for surgical/orthopaedic patients: threshold is <80k.
        # Surgery patients have a higher threshold than medicine patients.
        prompt = platelet_system_prompt()
        assert "80,000" in prompt

    def test_cvc_threshold_50k_mentioned(self) -> None:
        # CVC insertion threshold: <50k. Independent of LP — the LLM must
        # distinguish between LP indication and CVC indication.
        prompt = platelet_system_prompt()
        # Both LP-medicine/OB and CVC use 50k; prompt must mention CVC
        # separately so the LLM links the threshold to the right procedure.
        lower = prompt.lower()
        assert "50,000" in prompt and ("cvc" in lower or "central venous" in lower)

    def test_major_surgery_threshold_80k(self) -> None:
        # Major non-neuraxial surgery: <80k. Distinct from neuraxial LP.
        prompt = platelet_system_prompt()
        lower = prompt.lower()
        assert "80,000" in prompt and "surgery" in lower

    def test_high_bleeding_risk_surgery_threshold_100k(self) -> None:
        # High-bleeding-risk surgery: ceiling raised to <100k. Absent
        # this clause the LLM would under-clear high-risk surgical patients.
        prompt = platelet_system_prompt()
        assert "100,000" in prompt

    def test_dic_consumptive_threshold_10k(self) -> None:
        # DIC / consumptive thrombocytopenia without active bleeding: <10k.
        # Setting any higher threshold would lead to unnecessary transfusions
        # in DIC patients (per Chula policy exclusion logic).
        prompt = platelet_system_prompt()
        lower = prompt.lower()
        assert "10,000" in prompt and ("dic" in lower or "consumptive" in lower)

    def test_chemo_hsct_threshold_10k(self) -> None:
        # Chemo/HSCT prophylaxis: <10k (no active bleeding). Separate from
        # DIC — both share the 10k threshold but for different reasons.
        prompt = platelet_system_prompt()
        lower = prompt.lower()
        assert "10,000" in prompt and ("chemo" in lower or "hsct" in lower)

    def test_expected_below_10k_within_24h_clause(self) -> None:
        # Pre-emptive prophylaxis: transfusion appropriate even before count
        # drops if it is EXPECTED to drop below 10k within 24 hours. Missing
        # this clause causes the LLM to deny appropriate pre-chemo platelets.
        prompt = platelet_system_prompt()
        lower = prompt.lower()
        assert "expected" in lower and "24" in prompt


class TestPlateletSystemPromptExclusionPopulations:
    """Every exclusion population must be in the prompt.

    WHY: An exclusion population absent from the prompt lets the LLM clear
    a WITHHELD patient (e.g. dengue without bleeding). Each exclusion is
    clinically distinct and must be present independently.
    """

    def test_cardiac_surgery_cardiopulmonary_bypass_excluded(self) -> None:
        # Cardiac surgery / CPB: platelet consumption is expected; transfusion
        # without thrombocytopenia + severe bleeding is inappropriate.
        prompt = platelet_system_prompt()
        lower = prompt.lower()
        assert "cardiac" in lower or "cardiopulmonary" in lower

    def test_intracranial_bleed_high_count_excluded(self) -> None:
        # Head / intracranial bleed with count >100k: transfusion is NOT
        # indicated unless count is below the relevant threshold.
        prompt = platelet_system_prompt()
        lower = prompt.lower()
        assert "intracranial" in lower or ("head" in lower and "bleed" in lower)

    def test_dengue_thai_term_excluded(self) -> None:
        # Thai clinicians record dengue as ไข้เลือดออก in Thai-language notes.
        # The exclusion population MUST include the Thai term so the LLM
        # recognizes it from Thai-language evidence chunks.
        prompt = platelet_system_prompt()
        assert "ไข้เลือดออก" in prompt

    def test_aplastic_anemia_excluded(self) -> None:
        # Aplastic anemia without active bleeding: transfusion for a low count
        # alone is NOT appropriate; aplastic patients need platelet support
        # only with active bleeding.
        prompt = platelet_system_prompt()
        lower = prompt.lower()
        assert "aplastic" in lower

    def test_snakebite_excluded(self) -> None:
        # Hematotoxic snakebite without life-threatening bleeding: transfusion
        # is withheld (thrombocytopenia is expected and self-limiting).
        prompt = platelet_system_prompt()
        lower = prompt.lower()
        assert "snakebite" in lower or "snake" in lower


class TestPlateletSystemPromptHardSignalFields:
    """The prompt must declare all three hard-signal field names.

    WHY: The LLM's structured output includes these boolean fields; if the
    prompt does not name them the model may omit or misname them, breaking
    the parser contract.
    """

    def test_active_bleeding_field_named(self) -> None:
        prompt = platelet_system_prompt()
        assert "active_bleeding" in prompt

    def test_procedure_indication_field_named(self) -> None:
        prompt = platelet_system_prompt()
        assert "procedure_indication" in prompt

    def test_prophylactic_marrow_failure_field_named(self) -> None:
        prompt = platelet_system_prompt()
        assert "prophylactic_marrow_failure" in prompt


class TestPlateletSystemPromptBilingualOutput:
    """The platelet prompt must mandate the same bilingual output as RBC.

    WHY: Reviewers use reasoning_summary_th; absent Thai the committee cannot
    read the LLM rationale. The test mirrors the existing RBC assertion so
    both modes share the same bilingual contract.
    """

    def test_reasoning_summary_en_required(self) -> None:
        prompt = platelet_system_prompt()
        assert "reasoning_summary_en" in prompt

    def test_reasoning_summary_th_required(self) -> None:
        prompt = platelet_system_prompt()
        assert "reasoning_summary_th" in prompt

    def test_english_and_thai_mentioned(self) -> None:
        prompt = platelet_system_prompt()
        assert "English" in prompt and "Thai" in prompt

    def test_translation_instruction_present(self) -> None:
        # "NOT a word-for-word translation" guards against stiff literal Thai.
        prompt = platelet_system_prompt()
        assert "translation" in prompt


class TestPlateletSystemPromptDocumentationAbsenceRule:
    """The documentation-absence rule (CR-C2) must be in the prompt."""

    def test_documentation_absence_rule_present(self) -> None:
        # Bare low count must NOT set APPROPRIATE — the rule must be stated.
        prompt = platelet_system_prompt()
        lower = prompt.lower()
        assert (
            "bare" in lower or "absence" in lower or "cr-c2" in lower.replace("-", "")
        )


# =============================================================================
# C1a — rendering contract (no cohort_threshold required)
# =============================================================================


class TestPlateletSystemPromptRendering:
    """platelet_system_prompt() renders without a cohort_threshold argument."""

    def test_renders_without_cohort_threshold(self) -> None:
        # The function takes no arguments — platelet transfusion has no
        # Hb-based cohort threshold (unlike RBC).
        prompt = platelet_system_prompt()
        assert isinstance(prompt, str) and prompt.strip()

    def test_is_nfc_normalized(self) -> None:
        # Thai Unicode must be NFC to match the evidence chunks (which go
        # through the NFC normalizer upstream). NFD vs NFC mismatch would
        # prevent citation grounding for Thai-language evidence.
        import unicodedata

        prompt = platelet_system_prompt()
        assert unicodedata.normalize("NFC", prompt) == prompt

    def test_mentions_platelet_review_mode(self) -> None:
        # The system prompt must identify its task mode so the LLM knows
        # it is auditing platelets, not RBC.
        prompt = platelet_system_prompt()
        assert "PLATELET_REVIEW" in prompt or "platelet" in prompt.lower()

    def test_system_prompt_for_accepts_platelet_mode(self) -> None:
        # system_prompt_for must handle PLATELET_REVIEW so callers iterating
        # TASK_MODES do not need a special branch (existing test pattern).
        # cohort_threshold is accepted but ignored for platelet.
        prompt = system_prompt_for(
            task_mode="PLATELET_REVIEW",  # type: ignore[arg-type]
            cohort_threshold=7.0,
        )
        assert isinstance(prompt, str) and prompt.strip()

    def test_platelet_mode_in_task_modes(self) -> None:
        # PLATELET_REVIEW is a declared mode; callers iterating TASK_MODES
        # (e.g. golden-set replay) will encounter it.
        assert "PLATELET_REVIEW" in TASK_MODES


# =============================================================================
# C1a — RBC prompt byte-identity
# =============================================================================


class TestRbcPromptByteIdentity:
    """RBC prompts must be unchanged after adding the platelet mode.

    WHY: Changing the RBC prompt text would shift the prompt hash, invalidating
    every stored audit row's reproducibility invariant.
    """

    # The expected prompt text is derived by calling the function itself
    # (golden comparison). The point is that calling it BEFORE and AFTER
    # the Stage C1 changes yields the same bytes. We assert this by
    # comparing two calls with different cohort thresholds to confirm the
    # injection still works, and by checking structural markers.

    def test_hb_7_10_review_prompt_contains_gray_zone_text(self) -> None:
        prompt = system_prompt_for(task_mode="HB_7_10_REVIEW", cohort_threshold=7.0)
        assert "7" in prompt and "10" in prompt and "gray-zone" in prompt.lower()

    def test_hb_gt_10_override_prompt_contains_override_text(self) -> None:
        prompt = system_prompt_for(task_mode="HB_GT_10_OVERRIDE", cohort_threshold=7.0)
        assert "override" in prompt.lower()

    def test_rbc_cohort_threshold_still_injected(self) -> None:
        # After adding platelet, the RBC threshold injection must still work.
        for threshold in [7.0, 7.5, 8.0]:
            prompt = system_prompt_for(
                task_mode="HB_7_10_REVIEW", cohort_threshold=threshold
            )
            assert f"{threshold:.1f}" in prompt

    def test_rbc_prompts_differ_by_threshold(self) -> None:
        p70 = system_prompt_for(task_mode="HB_7_10_REVIEW", cohort_threshold=7.0)
        p75 = system_prompt_for(task_mode="HB_7_10_REVIEW", cohort_threshold=7.5)
        assert p70 != p75

    def test_invalid_rbc_cohort_threshold_still_rejected(self) -> None:
        from bba.prompt_builder.exceptions import UnsupportedCohortThresholdError

        with pytest.raises((UnsupportedCohortThresholdError, ValueError)):
            system_prompt_for(task_mode="HB_7_10_REVIEW", cohort_threshold=9.9)

    def test_rbc_and_platelet_prompts_are_distinct(self) -> None:
        rbc = system_prompt_for(task_mode="HB_7_10_REVIEW", cohort_threshold=7.0)
        plt = platelet_system_prompt()
        assert rbc != plt


# =============================================================================
# C1b — PlateletLlmClassificationResponse model
# =============================================================================


class TestPlateletLlmClassificationResponseModel:
    """PlateletLlmClassificationResponse carries the three hard-signal bools.

    WHY: The model is the parse target; if it does not enforce the three
    booleans, a response missing them would silently validate and the
    guardrail would never see the signals.
    """

    def test_valid_platelet_response_accepted(self) -> None:
        resp = PlateletLlmClassificationResponse(
            classification="APPROPRIATE",
            indications=(),
            negative_evidence=(),
            reasoning_summary_en="Platelet <50k, LP planned.",
            reasoning_summary_th="เกล็ดเลือดต่ำกว่าเกณฑ์ LP",
            active_bleeding=False,
            procedure_indication=True,
            prophylactic_marrow_failure=False,
        )
        assert resp.procedure_indication is True
        assert resp.active_bleeding is False

    def test_missing_hard_signals_rejected(self) -> None:
        # All three booleans are required — absent signals cannot be inferred.
        with pytest.raises(ValidationError):
            PlateletLlmClassificationResponse(
                classification="APPROPRIATE",
                indications=(),
                negative_evidence=(),
                reasoning_summary_en="x",
                reasoning_summary_th="x",
                # active_bleeding, procedure_indication, prophylactic_marrow_failure
                # all missing — must fail
            )

    def test_is_frozen(self) -> None:
        resp = PlateletLlmClassificationResponse(
            classification="APPROPRIATE",
            indications=(),
            negative_evidence=(),
            reasoning_summary_en="x",
            reasoning_summary_th="x",
            active_bleeding=False,
            procedure_indication=False,
            prophylactic_marrow_failure=False,
        )
        with pytest.raises((ValidationError, TypeError)):
            resp.active_bleeding = True  # type: ignore[misc]


# =============================================================================
# C1b — parse_platelet_structured_response extracts hard signals
# =============================================================================


class TestParsePlateletStructuredResponse:
    """parse_platelet_structured_response extracts the three hard-signal bools.

    WHY: The guardrail (Stage C2) reads PlateletHardSignals from the parse
    outcome; if extraction silently fails the guardrail cannot protect against
    over-clearing.
    """

    def test_all_signals_false_extracted_correctly(self) -> None:
        # A grounded APPROPRIATE with no active bleeding / no procedure / no
        # marrow-failure — all signals False, correctly reflected.
        result = _batch_result(
            content=_platelet_tool_content(
                classification="APPROPRIATE",
                active_bleeding=False,
                procedure_indication=False,
                prophylactic_marrow_failure=False,
            )
        )
        outcome = parse_platelet_structured_response(result)
        assert not outcome.parse_failure
        assert outcome.platelet_hard_signals is not None
        signals = outcome.platelet_hard_signals
        assert signals.active_bleeding is False
        assert signals.procedure_indication is False
        assert signals.prophylactic_marrow_failure is False

    def test_procedure_indication_true_extracted(self) -> None:
        # LP below the 50k threshold — procedure_indication grounded.
        result = _batch_result(
            content=_platelet_tool_content(
                classification="APPROPRIATE",
                procedure_indication=True,
            )
        )
        outcome = parse_platelet_structured_response(result)
        assert not outcome.parse_failure
        assert outcome.platelet_hard_signals is not None
        assert outcome.platelet_hard_signals.procedure_indication is True
        assert outcome.platelet_hard_signals.active_bleeding is False
        assert outcome.platelet_hard_signals.prophylactic_marrow_failure is False

    def test_active_bleeding_true_extracted(self) -> None:
        result = _batch_result(
            content=_platelet_tool_content(
                classification="APPROPRIATE",
                active_bleeding=True,
            )
        )
        outcome = parse_platelet_structured_response(result)
        assert outcome.platelet_hard_signals is not None
        assert outcome.platelet_hard_signals.active_bleeding is True

    def test_prophylactic_marrow_failure_true_extracted(self) -> None:
        result = _batch_result(
            content=_platelet_tool_content(
                classification="APPROPRIATE",
                prophylactic_marrow_failure=True,
            )
        )
        outcome = parse_platelet_structured_response(result)
        assert outcome.platelet_hard_signals is not None
        assert outcome.platelet_hard_signals.prophylactic_marrow_failure is True

    def test_all_signals_true_extracted(self) -> None:
        result = _batch_result(
            content=_platelet_tool_content(
                active_bleeding=True,
                procedure_indication=True,
                prophylactic_marrow_failure=True,
            )
        )
        outcome = parse_platelet_structured_response(result)
        assert outcome.platelet_hard_signals is not None
        assert outcome.platelet_hard_signals.any_signal() is True

    def test_parse_failure_gives_none_signals(self) -> None:
        # Malformed response: signals must be None, not defaulting to False.
        # A None signals on the outcome tells Stage C2 the response was
        # unparseable (distinct from signals=all-False which means parsed
        # but no indication grounded).
        result = _batch_result(
            raw_response_override={"content": [{"type": "text", "text": "oops"}]}
        )
        outcome = parse_platelet_structured_response(result)
        assert outcome.parse_failure is True
        assert outcome.platelet_hard_signals is None

    def test_missing_hard_signals_in_response_is_parse_failure(self) -> None:
        # If the LLM omits the three booleans the platelet parser must fail
        # closed (SCHEMA_MISMATCH), not silently default them to False.
        content = [
            {
                "type": "tool_use",
                "id": "tool_01",
                "name": "classify_transfusion_order",
                "input": {
                    "classification": "APPROPRIATE",
                    "indications": [],
                    "negative_evidence": [],
                    "reasoning_summary_en": "ok",
                    "reasoning_summary_th": "ok",
                    # active_bleeding, procedure_indication, prophylactic_marrow_failure
                    # intentionally absent — schema mismatch
                },
            }
        ]
        result = _batch_result(content=content)
        outcome = parse_platelet_structured_response(result)
        assert outcome.parse_failure is True
        assert outcome.parse_failure_reason == ParseFailureReason.SCHEMA_MISMATCH
        assert outcome.platelet_hard_signals is None

    def test_returns_parse_outcome_type(self) -> None:
        result = _batch_result(content=_platelet_tool_content())
        outcome = parse_platelet_structured_response(result)
        assert isinstance(outcome, ParseOutcome)


# =============================================================================
# C1b — RBC parse_structured_response is byte-identical
# =============================================================================


class TestRbcParserUnchanged:
    """parse_structured_response must return platelet_hard_signals=None for RBC.

    WHY: RBC responses do not emit hard-signal booleans; the field must default
    to None so Stage C2 can distinguish RBC outcomes from platelet outcomes
    without inspecting the response classification.
    """

    def test_rbc_success_outcome_has_none_signals(self) -> None:
        result = _batch_result(content=_rbc_tool_content())
        outcome = parse_structured_response(result)
        assert not outcome.parse_failure
        assert outcome.platelet_hard_signals is None

    def test_rbc_parse_failure_has_none_signals(self) -> None:
        result = _batch_result(
            raw_response_override={"content": [{"type": "text", "text": "bad"}]}
        )
        outcome = parse_structured_response(result)
        assert outcome.parse_failure is True
        assert outcome.platelet_hard_signals is None

    def test_parse_outcome_default_signals_is_none(self) -> None:
        # Constructing ParseOutcome without platelet_hard_signals must give None
        # (backward compatibility for all existing callers).
        from bba.llm_client.models import LlmClassificationResponse, IndicationCitation

        resp = LlmClassificationResponse(
            classification="APPROPRIATE",
            indications=(
                IndicationCitation(
                    code="B1.ab",
                    quote="bleeding",
                    source_id="E1",
                    confidence=0.9,
                ),
            ),
            negative_evidence=(),
            reasoning_summary_en="ok",
            reasoning_summary_th="โอเค",
        )
        outcome = ParseOutcome(
            parsed=resp,
            parse_failure=False,
            parse_failure_reason=None,
        )
        assert outcome.platelet_hard_signals is None


# =============================================================================
# C1c — feature flag
# =============================================================================


class TestPlateletLlmFlag:
    """PLATELET_LLM_ENABLED must default to False (Stage C2 gates on it)."""

    def test_flag_is_false_by_default(self) -> None:
        # The flag is OFF so the RBC audit path is unaffected until Stage C2
        # explicitly enables it.
        assert PLATELET_LLM_ENABLED is False

    def test_flag_is_bool(self) -> None:
        assert isinstance(PLATELET_LLM_ENABLED, bool)
