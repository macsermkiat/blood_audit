"""Precedence pins for the ordered guardrails in ``_build_audit_row``."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from bba import feature_flags
from bba.audit_orders import AuditOrder
from bba.audit_pipeline import PipelineRowContext, apply_batch_results
from bba.audit_pipeline import replay
from bba.audit_pipeline.replay import (
    ADMINISTRATION_CONTRADICTION_REVIEW_REASON,
    EMPTY_REASONING_REVIEW_REASON,
    LLM_OVERCLEAR_ASSERT_REASON,
    PERIOP_CONTRADICTION_REVIEW_REASON,
    PREOP_RESERVATION_UNCONFIRMED_REVIEW_REASON,
)
from bba.audit_store import AuditRow, AuditStore
from bba.audit_store.models import AuditStoreConfig, Classification
from bba.cohort_detector import CohortAssignment, CohortLabel
from bba.hb_lookup import HbLookupResult
from bba.llm_client.models import (
    SONNET_MODEL_ID,
    BatchSubmissionResult,
    RawBatchResponse,
)
from bba.platelet_guardrail import PLATELET_OVERCLEAR_REVIEW_REASON
from bba.platelet_lookup.models import PlateletLookupResult
from bba.prompt_builder import EvidenceChunk
from bba.vitals_extractor import (
    PeriopSummary,
    SourceProvenance,
    VitalSigns,
    VitalsResult,
)

_RUN_TS = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _legacy_reserve_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make structured upcoming operations use the reserve-ahead LLM route."""
    monkeypatch.setattr(feature_flags, "DECLARED_USE_PREOP_EXEMPT_ENABLED", False)
    monkeypatch.setattr(feature_flags, "RESERVE_AHEAD_ROUTER_ENABLED", True)


def _rbc_context(
    audit_id: str,
    *,
    hb_value: float = 9.4,
    periop_summary: PeriopSummary | None = None,
    upcoming_procedure_hours: float | None = None,
) -> PipelineRowContext:
    order = AuditOrder(
        audit_id=audit_id,
        hn=f"HN-{audit_id}",
        an=f"AN-{audit_id}",
        reqno=f"REQ-{audit_id}",
        order_datetime=_RUN_TS,
        anchor_imputed=False,
        products_ordered=("LPRC",),
        diagnosis_codes=("D62",),
    )
    return PipelineRowContext(
        order=order,
        hb_result=HbLookupResult(
            value_g_dl=hb_value,
            datetime_utc=_RUN_TS,
            source="HEMATOLOGY",
            freshness="fresh",
            delta_hb_bypass=False,
            delta_hb_windows=(),
            needs_review_single_low_hb=False,
        ),
        vitals_result=VitalsResult(
            vitals=VitalSigns(sbp=110.0, hr=85.0),
            source=SourceProvenance.IPDADMPROGRESS,
            flags=frozenset(),
            note_timestamp=_RUN_TS,
        ),
        cohort_assignment=CohortAssignment(
            label=CohortLabel.CARDIAC_SURGERY,
            threshold=7.5,
            evidence_code=None,
            evidence_name=None,
        ),
        procedure_proximity_hours=None,
        upcoming_procedure_hours=upcoming_procedure_hours,
        crystalloid_liters_prior_4h=0.0,
        hn_hash=f"hn_{audit_id}",
        an_hash=f"an_{audit_id}",
        prior_rbc_units_24h=0,
        prior_rbc_units_7d=0,
        redactor_version="0.4.1+test",
        redactor_model_sha=f"sha_{audit_id}",
        policy_version="kcmh-pr17.2-2024",
        prompt_hash=f"ph_{audit_id}",
        evidence_bundle_hash=f"bh_{audit_id}",
        evidence_chunks=(
            EvidenceChunk(evidence_id="E1", source="Note", text="No transfusion"),
        ),
        periop_summary=periop_summary,
    )


def _platelet_context(audit_id: str) -> PipelineRowContext:
    order = AuditOrder(
        audit_id=audit_id,
        hn=f"HN-{audit_id}",
        an=f"AN-{audit_id}",
        reqno=f"REQ-{audit_id}",
        order_datetime=_RUN_TS,
        anchor_imputed=False,
        products_ordered=("PLT-POOL",),
        diagnosis_codes=("D62",),
        component="platelet",
    )
    return PipelineRowContext.for_platelet(
        order=order,
        platelet_result=PlateletLookupResult(
            value_k_ul=50.0,
            datetime_utc=_RUN_TS,
            source="HEMATOLOGY",
            freshness="fresh",
        ),
        hn_hash=f"hn_{audit_id}",
        an_hash=f"an_{audit_id}",
        redactor_version="0.4.1+test",
        redactor_model_sha=f"sha_{audit_id}",
        policy_version="kcmh-pr17.2-2024",
        prompt_hash=f"ph_{audit_id}",
        evidence_bundle_hash=f"bh_{audit_id}",
        evidence_chunks=(),
    )


def _response(
    audit_id: str,
    classification: Classification,
    *,
    reasoning_en: str = "model rationale",
    reasoning_th: str = "th",
    administration_claimed: bool | None = None,
    platelet_signals: bool | None = None,
) -> RawBatchResponse:
    input_payload: dict[str, object] = {
        "classification": classification,
        "indications": [],
        "negative_evidence": [],
        "reasoning_summary_en": reasoning_en,
        "reasoning_summary_th": reasoning_th,
    }
    if administration_claimed is not None:
        input_payload.update(
            {
                "administration_evidence": [],
                "administration_claimed": administration_claimed,
                "reservation_assessment": "INSUFFICIENT_EVIDENCE",
            }
        )
    if platelet_signals is not None:
        input_payload.update(
            {
                "active_bleeding": platelet_signals,
                "procedure_indication": platelet_signals,
                "prophylactic_marrow_failure": platelet_signals,
            }
        )
    result = BatchSubmissionResult(
        custom_id=audit_id,
        model_id=SONNET_MODEL_ID,
        raw_response_json={
            "id": f"msg-{audit_id}",
            "type": "message",
            "content": [
                {
                    "type": "tool_use",
                    "name": "classify_audit",
                    "input": input_payload,
                }
            ],
            "stop_reason": "tool_use",
        },
        request_json={"messages": []},
        response_headers={"anthropic-version": "2023-06-01"},
        request_timestamp=_RUN_TS,
        latency_ms=100,
        anthropic_version="2023-06-01",
        prompt_cache_id=None,
        extended_thinking_blocks=None,
    )
    return RawBatchResponse(batch_id=f"batch-{audit_id}", results=(result,))


def _apply_single_row(
    tmp_path: Path, context: PipelineRowContext, response: RawBatchResponse
) -> AuditRow:
    store = AuditStore(
        AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
    )
    apply_batch_results(
        response,
        audit_store=store,
        run_id="run-precedence",
        contexts={context.order.audit_id: context},
    )
    (row,) = store.read_audit_results(run_id="run-precedence")
    return row


def _assert_terminal(
    row: AuditRow, classification: Classification, review_reason: str
) -> None:
    assert (row.final_classification, row.review_reason) == (
        classification,
        review_reason,
    )


type _GuardrailTrace = list[tuple[str, Classification, str | None]]


def _trace_guardrail_stages(monkeypatch: pytest.MonkeyPatch) -> _GuardrailTrace:
    """Record primary/overlay inputs without changing guardrail behaviour."""
    trace: _GuardrailTrace = []

    def traced(stage: str, guardrail: replay._Guardrail) -> replay._Guardrail:
        def wrapper(
            verdict: replay._Verdict, context: replay._GuardrailContext, /
        ) -> replay._Verdict | None:
            trace.append((stage, verdict.final_classification, verdict.review_reason))
            return guardrail(verdict, context)

        return wrapper

    monkeypatch.setattr(
        replay,
        "_PRIMARY_GUARDRAILS",
        tuple(traced("primary", guardrail) for guardrail in replay._PRIMARY_GUARDRAILS),
    )
    monkeypatch.setattr(
        replay,
        "_POST_TERMINAL_OVERLAYS",
        tuple(
            traced("overlay", guardrail) for guardrail in replay._POST_TERMINAL_OVERLAYS
        ),
    )
    return trace


def _assert_overlay_received(
    trace: _GuardrailTrace,
    classification: Classification,
    review_reason: str | None,
) -> None:
    """Pin that the overlay ran once, after the primary stage's terminal."""
    assert [event for event in trace if event[0] == "overlay"] == [
        ("overlay", classification, review_reason)
    ]
    assert trace[-1][0] == "overlay"


def test_reserve_wins_over_periop(tmp_path: Path) -> None:
    context = _rbc_context(
        "reserve-periop",
        hb_value=12.9,
        upcoming_procedure_hours=24.0,
        periop_summary=PeriopSummary(surgical_context=True),
    )
    row = _apply_single_row(
        tmp_path,
        context,
        _response(context.order.audit_id, "INSUFFICIENT_EVIDENCE"),
    )
    _assert_terminal(
        row,
        "PREOP_RESERVATION_UNCONFIRMED",
        PREOP_RESERVATION_UNCONFIRMED_REVIEW_REASON,
    )


def test_reserve_administration_contradiction_wins_over_periop(
    tmp_path: Path,
) -> None:
    context = _rbc_context(
        "reserve-admin-periop",
        hb_value=12.9,
        upcoming_procedure_hours=24.0,
        periop_summary=PeriopSummary(intraop_transfusion=True),
    )
    row = _apply_single_row(
        tmp_path,
        context,
        _response(
            context.order.audit_id,
            "INSUFFICIENT_EVIDENCE",
            administration_claimed=False,
        ),
    )
    _assert_terminal(row, "NEEDS_REVIEW", ADMINISTRATION_CONTRADICTION_REVIEW_REASON)


def test_reserve_wins_over_native_review(tmp_path: Path) -> None:
    context = _rbc_context(
        "reserve-native", hb_value=12.9, upcoming_procedure_hours=24.0
    )
    row = _apply_single_row(
        tmp_path,
        context,
        _response(context.order.audit_id, "NEEDS_REVIEW"),
    )
    _assert_terminal(
        row,
        "PREOP_RESERVATION_UNCONFIRMED",
        PREOP_RESERVATION_UNCONFIRMED_REVIEW_REASON,
    )


def test_empty_reasoning_overlays_reserve_and_preserves_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = _trace_guardrail_stages(monkeypatch)
    context = _rbc_context(
        "reserve-empty", hb_value=12.9, upcoming_procedure_hours=24.0
    )
    row = _apply_single_row(
        tmp_path,
        context,
        _response(
            context.order.audit_id,
            "NEEDS_REVIEW",
            reasoning_en="",
            reasoning_th="",
        ),
    )
    _assert_terminal(row, "NEEDS_REVIEW", PREOP_RESERVATION_UNCONFIRMED_REVIEW_REASON)
    _assert_overlay_received(
        trace,
        "PREOP_RESERVATION_UNCONFIRMED",
        PREOP_RESERVATION_UNCONFIRMED_REVIEW_REASON,
    )


def test_empty_reasoning_overlays_periop_and_preserves_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = _trace_guardrail_stages(monkeypatch)
    context = _rbc_context(
        "periop-empty", periop_summary=PeriopSummary(surgical_context=True)
    )
    row = _apply_single_row(
        tmp_path,
        context,
        _response(
            context.order.audit_id,
            "INSUFFICIENT_EVIDENCE",
            reasoning_en="",
            reasoning_th="",
        ),
    )
    _assert_terminal(row, "NEEDS_REVIEW", PERIOP_CONTRADICTION_REVIEW_REASON)
    _assert_overlay_received(trace, "NEEDS_REVIEW", PERIOP_CONTRADICTION_REVIEW_REASON)


def test_empty_reasoning_overlays_b1_and_replaces_assertion_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = _trace_guardrail_stages(monkeypatch)
    context = _rbc_context("b1-empty")
    row = _apply_single_row(
        tmp_path,
        context,
        _response(
            context.order.audit_id,
            "APPROPRIATE",
            reasoning_en="",
            reasoning_th="",
        ),
    )
    _assert_terminal(row, "NEEDS_REVIEW", EMPTY_REASONING_REVIEW_REASON)
    _assert_overlay_received(trace, "INAPPROPRIATE", LLM_OVERCLEAR_ASSERT_REASON)


def test_empty_reasoning_overlays_platelet_overclear_and_preserves_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = _trace_guardrail_stages(monkeypatch)
    context = _platelet_context("platelet-overclear-empty")
    row = _apply_single_row(
        tmp_path,
        context,
        _response(
            context.order.audit_id,
            "APPROPRIATE",
            reasoning_en="",
            reasoning_th="",
            platelet_signals=False,
        ),
    )
    _assert_terminal(row, "NEEDS_REVIEW", PLATELET_OVERCLEAR_REVIEW_REASON)
    _assert_overlay_received(trace, "NEEDS_REVIEW", PLATELET_OVERCLEAR_REVIEW_REASON)


def test_empty_reasoning_overlays_platelet_parse_failure_and_preserves_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace = _trace_guardrail_stages(monkeypatch)
    context = _platelet_context("platelet-parse-empty")
    row = _apply_single_row(
        tmp_path,
        context,
        _response(
            context.order.audit_id,
            "INAPPROPRIATE",
            reasoning_en="",
            reasoning_th="",
        ),
    )
    _assert_terminal(row, "NEEDS_REVIEW", "schema_mismatch")
    _assert_overlay_received(trace, "NEEDS_REVIEW", "schema_mismatch")


def test_inappropriate_rbc_without_guardrail_need_does_not_ground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A terminal RBC row must not touch malformed evidence unnecessarily."""
    context = _rbc_context("inappropriate-none-text").model_copy(
        update={
            "evidence_chunks": (
                EvidenceChunk.model_construct(
                    evidence_id="E1", source="Note", text=None
                ),
            )
        }
    )
    grounding_calls = 0
    original_grounder = replay._grounded_indications

    def count_grounding_calls(
        indications: tuple[dict[str, object], ...],
        row_context: PipelineRowContext,
    ) -> tuple[dict[str, object], ...]:
        nonlocal grounding_calls
        grounding_calls += 1
        return original_grounder(indications, row_context)

    monkeypatch.setattr(replay, "_grounded_indications", count_grounding_calls)

    row = _apply_single_row(
        tmp_path,
        context,
        _response(context.order.audit_id, "INAPPROPRIATE"),
    )

    assert (row.final_classification, row.review_reason) == ("INAPPROPRIATE", None)
    assert grounding_calls == 0
