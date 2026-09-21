"""Replay wiring of the platelet count-trend floor (issue #237).

The first real-data platelet run cleared 67 orders at a count of 10,000 /uL or
more on ``prophylactic_marrow_failure`` alone, 19 of them with a flat or rising
count. The model sets that hard signal itself, so the over-clear guardrail
cannot see it; the projected 24 h count carried on the row context can.
"""

from __future__ import annotations

from datetime import UTC, datetime

import bba.feature_flags as feature_flags
import pytest
from bba.audit_orders import AuditOrder
from bba.audit_pipeline import PipelineRowContext, apply_batch_results
from bba.audit_store import AuditRow, AuditStore
from bba.audit_store.models import AuditStoreConfig
from bba.llm_client import RawBatchResponse
from bba.llm_client.models import SONNET_MODEL_ID, BatchSubmissionResult
from bba.platelet_guardrail import (
    PLATELET_OVERCLEAR_REVIEW_REASON,
    PLATELET_TREND_REVIEW_REASON,
)
from bba.platelet_lookup.models import PlateletLookupResult

_RUN_TS = datetime(2026, 9, 21, 4, 0, tzinfo=UTC)
_AUDIT_ID = "audit-plt-trend"


def _ctx(
    *, count: float, projected: float | None, computed: bool = True
) -> PipelineRowContext:
    return PipelineRowContext.for_platelet(
        order=AuditOrder(
            audit_id=_AUDIT_ID,
            hn="HN-1",
            an="AN-1",
            reqno="REQ-1",
            order_datetime=_RUN_TS,
            anchor_imputed=False,
            products_ordered=("PLT-POOL",),
            diagnosis_codes=("C92.0",),
            component="platelet",
        ),
        platelet_result=PlateletLookupResult(
            value_k_ul=count,
            datetime_utc=_RUN_TS,
            source="HEMATOLOGY",
            freshness="fresh",
        ),
        platelet_projected_24h_k_ul=projected,
        platelet_projection_computed=computed,
        hn_hash="hn",
        an_hash="an",
        redactor_version="0.4.1+test",
        redactor_model_sha="sha",
        policy_version="kcmh-pr17.2-2024",
        prompt_hash="ph",
        evidence_bundle_hash="bh",
    )


def _response(**signals: bool) -> RawBatchResponse:
    payload = {
        "classification": "APPROPRIATE",
        "indications": [],
        "negative_evidence": [],
        "reasoning_summary_en": "count expected to fall below 10,000 within 24 h",
        "reasoning_summary_th": "คาดว่าเกล็ดเลือดจะลดลง",
        "active_bleeding": False,
        "procedure_indication": False,
        "prophylactic_marrow_failure": False,
        "intracranial_bleed_indication": False,
        **signals,
    }
    return RawBatchResponse(
        batch_id="msgbatch_trend",
        results=(
            BatchSubmissionResult(
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
            ),
        ),
    )


def _apply(tmp_path, ctx: PipelineRowContext, response: RawBatchResponse) -> AuditRow:
    store = AuditStore(
        AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
    )
    apply_batch_results(
        response, audit_store=store, run_id="run-trend", contexts={_AUDIT_ID: ctx}
    )
    return store.read_audit_results()[0]


@pytest.fixture
def trend_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(feature_flags, "PLATELET_TREND_GUARDRAIL_ENABLED", True)


def test_flat_count_clear_on_the_clause_alone_goes_to_review(tmp_path, trend_on):
    row = _apply(
        tmp_path,
        _ctx(count=14.0, projected=14.0),
        _response(prophylactic_marrow_failure=True),
    )
    # Review, never INAPPROPRIATE: the floor disputes one clause, not the order.
    assert row.final_classification == "NEEDS_REVIEW"
    assert row.review_reason == PLATELET_TREND_REVIEW_REASON
    assert row.needs_human_review is True


def test_clear_with_no_second_draw_goes_to_review(tmp_path, trend_on):
    row = _apply(
        tmp_path,
        _ctx(count=14.0, projected=None),
        _response(prophylactic_marrow_failure=True),
    )
    assert row.final_classification == "NEEDS_REVIEW"
    assert row.review_reason == PLATELET_TREND_REVIEW_REASON


def test_clear_supported_by_the_projection_stands(tmp_path, trend_on):
    row = _apply(
        tmp_path,
        _ctx(count=14.0, projected=6.0),
        _response(prophylactic_marrow_failure=True),
    )
    assert row.final_classification == "APPROPRIATE"
    assert row.review_reason is None


def test_bleeding_clear_is_not_judged_on_the_trend(tmp_path, trend_on):
    row = _apply(
        tmp_path,
        _ctx(count=14.0, projected=14.0),
        _response(prophylactic_marrow_failure=True, active_bleeding=True),
    )
    assert row.final_classification == "APPROPRIATE"


def test_ungrounded_clear_keeps_the_overclear_reason(tmp_path, trend_on):
    # The two floors must stay separately triageable on the dashboard.
    row = _apply(tmp_path, _ctx(count=14.0, projected=14.0), _response())
    assert row.review_reason == PLATELET_OVERCLEAR_REVIEW_REASON


def test_caller_that_never_computed_a_projection_is_not_floored(tmp_path, trend_on):
    # None means "fewer than two usable draws" only when the caller says it
    # looked. A context builder that predates #237 must not send every
    # marrow-failure clear to review the day the flag is switched on.
    row = _apply(
        tmp_path,
        _ctx(count=14.0, projected=None, computed=False),
        _response(prophylactic_marrow_failure=True),
    )
    assert row.final_classification == "APPROPRIATE"


def test_flag_off_leaves_the_clear_untouched(tmp_path):
    # Default-off until the sandbox result is read (repo convention).
    assert feature_flags.PLATELET_TREND_GUARDRAIL_ENABLED is False
    row = _apply(
        tmp_path,
        _ctx(count=14.0, projected=14.0),
        _response(prophylactic_marrow_failure=True),
    )
    assert row.final_classification == "APPROPRIATE"
