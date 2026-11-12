"""A consultant's higher platelet target routes an INAPPROPRIATE order to review.

Ruling (review of hematology case REQNO 68012561, 2026-09-23): the patient had
hemoptysis on a ventilator, count 89,000 /uL, and the chest team wrote "chest
keep plt 100,000". Policy transfuses active bleeding only below 50,000 /uL, so
the order is INAPPROPRIATE on policy. But the ordering doctor followed a
specialist's documented advice, so the audit must not blame them: the order goes
to human review instead.

The advice must be documented before the order (user decision 2026-09-23). In
68012561 itself it first appears 32 h after the order, so that case stays
INAPPROPRIATE; the tests below use the same numbers with the advice on file.

The model reports the target it read (a number, not an indication); code does
the comparison against the trigger count and owns the downgrade. The downgrade
only ever moves INAPPROPRIATE to NEEDS_REVIEW, never to APPROPRIATE.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import bba.feature_flags as feature_flags
import pytest
from bba.audit_orders import AuditOrder
from bba.audit_pipeline import PipelineRowContext, apply_batch_results
from bba.audit_store import AuditRow, AuditStore
from bba.audit_store.models import AuditStoreConfig
from bba.llm_client import RawBatchResponse
from bba.llm_client.models import (
    SONNET_MODEL_ID,
    BatchSubmissionResult,
    PlateletLlmClassificationResponse,
)
from bba.llm_client.parser import parse_platelet_structured_response
from bba.llm_client.transport import _PLATELET_TOOL_INPUT_SCHEMA
from bba.platelet_guardrail import (
    PLATELET_SPECIALIST_TARGET_REVIEW_REASON,
    PlateletHardSignals,
    platelet_specialist_target_review,
)
from bba.platelet_lookup.models import PlateletLookupResult
from bba.prompt_builder.system_prompt import platelet_system_prompt, system_prompt_for

_FIELD = "specialist_platelet_target_per_ul"
_RUN_TS = datetime(2026, 9, 23, 4, 0, tzinfo=UTC)
_AUDIT_ID = "audit-plt-specialist"


def _payload(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "indications": [],
        "negative_evidence": [],
        "reasoning_summary_en": (
            "Hemoptysis at 89,000 /uL is above the 50,000 active-bleeding "
            "threshold; the chest team wrote 'chest keep plt 100,000'."
        ),
        "reasoning_summary_th": "hemoptysis ที่ plt 89,000 สูงกว่าเกณฑ์ 50,000",
        "active_bleeding": False,
        "procedure_indication": False,
        "prophylactic_marrow_failure": False,
        "intracranial_bleed_indication": False,
        _FIELD: 100000,
        "classification": "INAPPROPRIATE",
    }
    return {**base, **overrides}


def _result(payload: dict[str, Any]) -> BatchSubmissionResult:
    return BatchSubmissionResult(
        custom_id=_AUDIT_ID,
        model_id=SONNET_MODEL_ID,  # type: ignore[arg-type]
        raw_response_json={
            "content": [
                {
                    "type": "tool_use",
                    "name": "classify_transfusion_order",
                    "input": payload,
                }
            ]
        },
        request_json={"messages": [{"role": "user", "content": "..."}]},
        response_headers={"anthropic-version": "2023-06-01"},
        request_timestamp=_RUN_TS,
        latency_ms=500,
        anthropic_version="2023-06-01",
        prompt_cache_id=None,
        extended_thinking_blocks=None,
    )


def _ctx(count: float) -> PipelineRowContext:
    return PipelineRowContext.for_platelet(
        order=AuditOrder(
            audit_id=_AUDIT_ID,
            hn="HN-1",
            an="AN-1",
            reqno="REQ-1",
            order_datetime=_RUN_TS,
            anchor_imputed=False,
            products_ordered=("LDPPC",),
            diagnosis_codes=("R04.2",),
            component="platelet",
        ),
        platelet_result=PlateletLookupResult(
            value_k_ul=count,
            datetime_utc=_RUN_TS,
            source="HEMATOLOGY",
            freshness="fresh",
        ),
        hn_hash="hn",
        an_hash="an",
        redactor_version="0.4.1+test",
        redactor_model_sha="sha",
        policy_version="kcmh-pr17.2-2024",
        prompt_hash="ph",
        evidence_bundle_hash="bh",
    )


def _apply(tmp_path, count: float, payload: dict[str, Any]) -> AuditRow:
    store = AuditStore(
        AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
    )
    apply_batch_results(
        RawBatchResponse(batch_id="msgbatch_specialist", results=(_result(payload),)),
        audit_store=store,
        run_id="run-specialist",
        contexts={_AUDIT_ID: _ctx(count)},
    )
    return store.read_audit_results()[0]


@pytest.fixture
def guardrail_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        feature_flags, "PLATELET_SPECIALIST_TARGET_GUARDRAIL_ENABLED", True
    )


class TestPromptAsksForTheTarget:
    def test_prompt_names_the_field_and_the_example(self) -> None:
        prompt = platelet_system_prompt()
        assert "SPECIALIST TARGET." in prompt
        assert _FIELD in prompt

    def test_target_is_not_an_indication(self) -> None:
        # The model must keep judging against policy; code does the routing.
        # If the model itself returned NEEDS_REVIEW, the INAPPROPRIATE-only
        # downgrade would have nothing to act on and the reason would be lost.
        section = platelet_system_prompt().split("SPECIALIST TARGET.", 1)[1]
        section = section.split("\n\n", 1)[0]
        assert "not a positive indication" in section
        assert "LOCAL TRIGGERS" in section

    def test_only_advice_documented_before_the_order_counts(self) -> None:
        # Advice written after the order would justify it after the fact.
        section = platelet_system_prompt().split("SPECIALIST TARGET.", 1)[1]
        assert "before the order" in section.split("\n\n", 1)[0]
        prop = _PLATELET_TOOL_INPUT_SCHEMA["properties"][_FIELD]
        assert "before the order" in prop["description"]

    def test_rbc_prompt_is_untouched(self) -> None:
        rbc = system_prompt_for(task_mode="HB_7_10_REVIEW", cohort_threshold=7.0)
        assert _FIELD not in rbc


class TestSchemaAndParse:
    def test_tool_schema_requires_a_nullable_integer(self) -> None:
        prop = _PLATELET_TOOL_INPUT_SCHEMA["properties"][_FIELD]
        assert prop["type"] == ["integer", "null"]
        assert _FIELD in _PLATELET_TOOL_INPUT_SCHEMA["required"]

    def test_parse_carries_the_target_to_the_signals(self) -> None:
        outcome = parse_platelet_structured_response(_result(_payload()))
        assert outcome.parse_failure is False
        assert outcome.platelet_hard_signals is not None
        assert outcome.platelet_hard_signals.specialist_target_per_ul == 100000

    def test_target_is_not_a_hard_signal(self) -> None:
        # A target alone must never let an APPROPRIATE past the over-clear floor.
        signals = PlateletHardSignals(specialist_target_per_ul=100000)
        assert signals.any_signal() is False

    def test_stored_responses_without_the_field_still_parse(self) -> None:
        payload = _payload()
        del payload[_FIELD]
        response = PlateletLlmClassificationResponse.model_validate(payload)
        assert response.specialist_platelet_target_per_ul is None

    def test_a_string_target_fails_closed(self) -> None:
        outcome = parse_platelet_structured_response(
            _result(_payload(**{_FIELD: "100000"}))
        )
        assert outcome.parse_failure is True


class TestDecision:
    @pytest.mark.parametrize(
        ("final", "target", "count", "expected"),
        [
            ("INAPPROPRIATE", 100000, 89.0, True),  # the ruling case
            ("INAPPROPRIATE", 100000, 100.0, False),  # at target: advice met
            ("INAPPROPRIATE", 100000, 120.0, False),  # beyond the advice
            ("INAPPROPRIATE", None, 89.0, False),  # no advice on file
            ("INAPPROPRIATE", 100000, None, False),  # no count to compare
            ("APPROPRIATE", 100000, 89.0, False),  # never touches a clear
            ("INSUFFICIENT_EVIDENCE", 100000, 89.0, False),
        ],
    )
    def test_fires_only_below_the_target_on_inappropriate(
        self, final, target, count, expected
    ) -> None:
        signals = PlateletHardSignals(specialist_target_per_ul=target)
        assert platelet_specialist_target_review(final, signals, count) is expected


class TestReplay:
    def test_ruling_case_goes_to_review_not_inappropriate(
        self, tmp_path, guardrail_on
    ) -> None:
        row = _apply(tmp_path, 89.0, _payload())
        assert row.final_classification == "NEEDS_REVIEW"
        assert row.review_reason == PLATELET_SPECIALIST_TARGET_REVIEW_REASON
        assert row.needs_human_review is True

    def test_count_above_the_target_stays_inappropriate(
        self, tmp_path, guardrail_on
    ) -> None:
        row = _apply(tmp_path, 120.0, _payload())
        assert row.final_classification == "INAPPROPRIATE"

    def test_no_target_stays_inappropriate(self, tmp_path, guardrail_on) -> None:
        row = _apply(tmp_path, 89.0, _payload(**{_FIELD: None}))
        assert row.final_classification == "INAPPROPRIATE"

    def test_flag_off_leaves_the_verdict_untouched(self, tmp_path) -> None:
        # Default-off until the sandbox result is read (repo convention).
        assert feature_flags.PLATELET_SPECIALIST_TARGET_GUARDRAIL_ENABLED is False
        row = _apply(tmp_path, 89.0, _payload())
        assert row.final_classification == "INAPPROPRIATE"
