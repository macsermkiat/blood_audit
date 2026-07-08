"""Stage B platelet dispatch tests.

Tests for the platelet order routing in run_pipeline and apply_batch_results.
RED phase: these tests will fail until the implementation is in place.
"""

from __future__ import annotations

from datetime import UTC, datetime

import bba.feature_flags as feature_flags
from bba.audit_orders import AuditOrder
from bba.audit_pipeline import (
    AuditPipelineConfig,
    InMemoryBatchRunStore,
    PipelineRowContext,
    apply_batch_results,
    run_pipeline,
)
from bba.audit_pipeline.pipeline import (
    _build_submission_requests,
    _deterministic_audit_row,
)
from bba.audit_store import AuditRow, AuditStore
from bba.audit_store.models import AuditStoreConfig
from bba.deterministic_classifier import BypassReason, ClassifierResult
from bba.llm_client import CassetteTransport, LlmClientConfig, RawBatchResponse
from bba.llm_client.models import (
    SONNET_MODEL_ID,
    BatchSubmissionResult,
    CassetteInteraction,
)
from bba.platelet_guardrail import PLATELET_OVERCLEAR_REVIEW_REASON
from bba.platelet_lookup.models import PlateletLookupResult
from bba.prompt_builder import EvidenceChunk
from bba.vitals_extractor import PeriopSummary

_RUN_TS = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)
_PIPELINE_CONFIG = AuditPipelineConfig(
    db_url="sqlite:///:memory:",
    code_version="v0.1.0+test",
)
_LLM_CONFIG = LlmClientConfig(code_version="v0.1.0+test")


def _platelet_order(audit_id: str) -> AuditOrder:
    return AuditOrder(
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


def _platelet_ctx(
    audit_id: str,
    *,
    platelet_count: float | None = None,
    platelet_mtp_suppressed: bool = False,
    evidence_chunks: tuple[EvidenceChunk, ...] = (),
    periop_summary: PeriopSummary | None = None,
) -> PipelineRowContext:
    plt_result = (
        PlateletLookupResult(
            value_k_ul=platelet_count,
            datetime_utc=_RUN_TS if platelet_count is not None else None,
            source="HEMATOLOGY" if platelet_count is not None else None,
            freshness="fresh" if platelet_count is not None else "missing",
        )
        if platelet_count is not None
        else PlateletLookupResult(
            value_k_ul=None,
            datetime_utc=None,
            source=None,
            freshness="missing",
        )
    )
    return PipelineRowContext.for_platelet(
        order=_platelet_order(audit_id),
        platelet_result=plt_result,
        hn_hash=f"hn_{audit_id}",
        an_hash=f"an_{audit_id}",
        redactor_version="0.4.1+test",
        redactor_model_sha=f"sha_{audit_id}",
        policy_version="kcmh-pr17.2-2024",
        prompt_hash=f"ph_{audit_id}",
        evidence_bundle_hash=f"bh_{audit_id}",
        evidence_chunks=evidence_chunks,
        periop_summary=periop_summary,
        platelet_mtp_suppressed=platelet_mtp_suppressed,
    )


def _platelet_result_item(
    audit_id: str,
    *,
    classification: str = "APPROPRIATE",
    active_bleeding: bool = False,
    procedure_indication: bool = False,
    prophylactic_marrow_failure: bool = False,
) -> BatchSubmissionResult:
    """A platelet-shaped tool-use response carrying the three hard-signal bools.

    Distinct from tests 8/9's RBC-shaped payloads: the platelet parser validates
    the three booleans, so the over-clear guardrail sees grounded (or absent)
    signals rather than fail-closing on a schema mismatch."""
    return BatchSubmissionResult(
        custom_id=audit_id,
        model_id=SONNET_MODEL_ID,  # type: ignore[arg-type]
        raw_response_json={
            "content": [
                {
                    "type": "tool_use",
                    "name": "classify_transfusion_order",
                    "input": {
                        "classification": classification,
                        "indications": [],
                        "negative_evidence": [],
                        "reasoning_summary_en": "platelet audit rationale",
                        "reasoning_summary_th": "เหตุผลการตรวจสอบเกล็ดเลือด",
                        "active_bleeding": active_bleeding,
                        "procedure_indication": procedure_indication,
                        "prophylactic_marrow_failure": prophylactic_marrow_failure,
                    },
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


def _evidence_chunk() -> tuple[EvidenceChunk, ...]:
    return (EvidenceChunk(evidence_id="E1", source="Lab", text="platelet 48k"),)


def _audit_store(tmp_path):
    return AuditStore(
        AuditStoreConfig(root_dir=tmp_path / "store", code_version="v0.1.0+test")
    )


# ───────────────────────── Test 1 ─────────────────────────


def test_for_platelet_factory_builds_correct_sentinels(tmp_path):
    ctx = _platelet_ctx("audit-plt-001")
    assert ctx.component == "platelet"
    assert ctx.hb_result.freshness == "missing"
    assert ctx.hb_result.value_g_dl is None
    assert ctx.cohort_assignment.threshold is None


# ───────────────────────── Test 2 ─────────────────────────


def test_platelet_insufficient_evidence_round_trip(tmp_path):
    """CRITICAL: platelet with missing count → INSUFFICIENT_EVIDENCE persisted
    with component=='platelet' and platelet_value==None."""
    ctx = _platelet_ctx("audit-plt-002")  # count=None, defer=False → IE
    store = _audit_store(tmp_path)
    result = run_pipeline(
        [ctx],
        transport=CassetteTransport(interactions=()),
        audit_store=store,
        batch_run_store=InMemoryBatchRunStore(),
        llm_config=_LLM_CONFIG,
        pipeline_config=_PIPELINE_CONFIG,
        run_id="run-plt-002",
    )
    assert "audit-plt-002" in result.audit_ids_persisted
    rows = store.read_audit_results()
    assert len(rows) == 1
    row = rows[0]
    assert row.component == "platelet"
    assert row.platelet_value is None
    assert row.platelet_freshness == "missing"


# ───────────────────────── Test 3 ─────────────────────────


def test_platelet_potentially_inappropriate_routes_onward(tmp_path):
    """count>=100 → POTENTIALLY_INAPPROPRIATE → NOT persisted in Stage B (LLM pending).

    With the platelet LLM leg OFF (default) the row is neither persisted NOR
    silently dropped: it must surface as an ORPHAN so an operator sees it is
    still pending, never lost."""
    ctx = _platelet_ctx("audit-plt-003", platelet_count=150.0)
    store = _audit_store(tmp_path)
    result = run_pipeline(
        [ctx],
        transport=CassetteTransport(interactions=()),
        audit_store=store,
        batch_run_store=InMemoryBatchRunStore(),
        llm_config=_LLM_CONFIG,
        pipeline_config=_PIPELINE_CONFIG,
        run_id="run-plt-003",
    )
    assert "audit-plt-003" not in result.audit_ids_persisted
    assert "audit-plt-003" in result.orphan_audit_ids
    assert store.read_audit_results() == ()


# ───────────────────────── Test 4 ─────────────────────────


def test_platelet_needs_review_routes_onward(tmp_path):
    """count<100 → NEEDS_REVIEW → NOT persisted in Stage B (LLM pending).

    Same orphan-not-dropped contract as the POTENTIALLY_INAPPROPRIATE case."""
    ctx = _platelet_ctx("audit-plt-004", platelet_count=50.0)
    store = _audit_store(tmp_path)
    result = run_pipeline(
        [ctx],
        transport=CassetteTransport(interactions=()),
        audit_store=store,
        batch_run_store=InMemoryBatchRunStore(),
        llm_config=_LLM_CONFIG,
        pipeline_config=_PIPELINE_CONFIG,
        run_id="run-plt-004",
    )
    assert "audit-plt-004" not in result.audit_ids_persisted
    assert "audit-plt-004" in result.orphan_audit_ids
    assert store.read_audit_results() == ()


# ───────────────────────── Test 5 ─────────────────────────


def test_hb_none_guard_never_trips_on_platelet_row(tmp_path):
    """_deterministic_audit_row must return an AuditRow for a platelet context,
    not raise the Hb=None ValueError."""
    ctx = _platelet_ctx("audit-plt-005")
    # The guard in _deterministic_audit_row should detect component=="platelet"
    # and return via the platelet path, never reaching the Hb=None raise.
    # Since _deterministic_audit_row short-circuits on platelet, any
    # ClassifierResult with INSUFFICIENT_EVIDENCE works (the guard fires
    # before the Hb=None raise).
    clf = ClassifierResult(
        classification="INSUFFICIENT_EVIDENCE",
        rationale="missing_hb",
        cohort_threshold=None,
        bypass_reason=BypassReason.NONE,
    )
    row = _deterministic_audit_row(
        context=ctx, classifier_result=clf, run_id="run-plt-005"
    )
    assert isinstance(row, AuditRow)
    assert row.component == "platelet"


# ───────────────────────── Test 6 ─────────────────────────


def test_rbc_dispatch_unchanged(tmp_path):
    """RBC context with Hb=None (INSUFFICIENT_EVIDENCE) still persists via
    the RBC path unchanged; component=='red_cell'."""
    from bba.cohort_detector import CohortAssignment, CohortLabel
    from bba.hb_lookup import HbLookupResult
    from bba.vitals_extractor import SourceProvenance, VitalSigns, VitalsResult

    order = AuditOrder(
        audit_id="audit-rbc-006",
        hn="HN-rbc-006",
        an="AN-rbc-006",
        reqno="REQ-rbc-006",
        order_datetime=_RUN_TS,
        anchor_imputed=False,
        products_ordered=("LPRC",),
        diagnosis_codes=("D62",),
    )
    hb_result = HbLookupResult(
        value_g_dl=None,
        datetime_utc=None,
        source=None,
        freshness="missing",
        delta_hb_bypass=False,
        delta_hb_windows=(),
        needs_review_single_low_hb=False,
    )
    vitals = VitalsResult(
        vitals=VitalSigns(),
        source=SourceProvenance.NONE_IN_WINDOW,
        flags=frozenset(),
        note_timestamp=None,
    )
    cohort = CohortAssignment(
        label=CohortLabel.UNKNOWN,
        threshold=None,
        evidence_code=None,
        evidence_name=None,
    )
    ctx = PipelineRowContext(
        order=order,
        hb_result=hb_result,
        vitals_result=vitals,
        cohort_assignment=cohort,
        procedure_proximity_hours=None,
        crystalloid_liters_prior_4h=0.0,
        hn_hash="hn_rbc_006",
        an_hash="an_rbc_006",
        prior_rbc_units_24h=0,
        prior_rbc_units_7d=0,
        redactor_version="0.4.1+test",
        redactor_model_sha="sha_rbc_006",
        policy_version="kcmh-pr17.2-2024",
        prompt_hash="ph_rbc_006",
        evidence_bundle_hash="bh_rbc_006",
    )
    store = _audit_store(tmp_path)
    result = run_pipeline(
        [ctx],
        transport=CassetteTransport(interactions=()),
        audit_store=store,
        batch_run_store=InMemoryBatchRunStore(),
        llm_config=_LLM_CONFIG,
        pipeline_config=_PIPELINE_CONFIG,
        run_id="run-rbc-006",
    )
    assert "audit-rbc-006" in result.audit_ids_persisted
    rows = store.read_audit_results()
    assert len(rows) == 1
    assert rows[0].component == "red_cell"


# ───────────────────────── Test 7 ─────────────────────────


def test_mtp_suppressed_platelet_emits_no_row(tmp_path):
    """A platelet context with platelet_mtp_suppressed=True emits no AuditRow."""
    ctx = _platelet_ctx("audit-plt-007", platelet_mtp_suppressed=True)
    store = _audit_store(tmp_path)
    result = run_pipeline(
        [ctx],
        transport=CassetteTransport(interactions=()),
        audit_store=store,
        batch_run_store=InMemoryBatchRunStore(),
        llm_config=_LLM_CONFIG,
        pipeline_config=_PIPELINE_CONFIG,
        run_id="run-plt-007",
    )
    assert "audit-plt-007" not in result.audit_ids_persisted
    assert len(store.read_audit_results()) == 0


# ───────────────────────── Test 8 ─────────────────────────


def test_rbc_overclear_guardrail_not_called_on_platelet(tmp_path):
    """The RBC llm_overclear_suspect guardrail must NOT fire on platelet rows.

    Without component-gating the guardrail would read the inert sentinel
    hb_result (value_g_dl=None) and cohort_assignment (UNKNOWN) and
    incorrectly floor APPROPRIATE → NEEDS_REVIEW. With the guard it is
    skipped and APPROPRIATE is preserved.
    """
    ctx = _platelet_ctx("audit-plt-008", platelet_count=50.0)
    # Build a minimal LLM-APPROPRIATE response for the platelet audit_id
    result_item = BatchSubmissionResult(
        custom_id="audit-plt-008",
        model_id="claude-sonnet-5",
        raw_response_json={
            "content": [
                {
                    "type": "tool_use",
                    "name": "classify_audit",
                    "input": {
                        "classification": "APPROPRIATE",
                        "indications": [{"text": "active bleed", "confidence": 0.9}],
                        "negative_evidence": [],
                        "reasoning_summary_en": "platelet transfusion appropriate for active bleeding",
                        "reasoning_summary_th": "เหมาะสม",
                    },
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
    response = RawBatchResponse(batch_id="msgbatch_plt", results=(result_item,))
    store = _audit_store(tmp_path)
    summary = apply_batch_results(
        response,
        audit_store=store,
        run_id="run-plt-008",
        contexts={"audit-plt-008": ctx},
    )
    assert "audit-plt-008" in summary.audit_ids_persisted
    rows = store.read_audit_results()
    assert len(rows) == 1
    # The LLM said APPROPRIATE; the RBC guardrail must NOT have fired
    assert rows[0].final_classification == "APPROPRIATE"
    assert rows[0].component == "platelet"


# ───────────────────────── Test 9 ─────────────────────────


def test_apply_batch_results_platelet_row_sets_component(tmp_path):
    """apply_batch_results with a platelet context persists component=='platelet'."""
    ctx = _platelet_ctx("audit-plt-009", platelet_count=75.0)
    result_item = BatchSubmissionResult(
        custom_id="audit-plt-009",
        model_id="claude-sonnet-5",
        raw_response_json={
            "content": [
                {
                    "type": "tool_use",
                    "name": "classify_audit",
                    "input": {
                        "classification": "NEEDS_REVIEW",
                        "indications": [],
                        "negative_evidence": [],
                        "reasoning_summary_en": "platelet count 75, needs review",
                        "reasoning_summary_th": "ต้องตรวจสอบ",
                    },
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
    response = RawBatchResponse(batch_id="msgbatch_plt9", results=(result_item,))
    store = _audit_store(tmp_path)
    summary = apply_batch_results(
        response,
        audit_store=store,
        run_id="run-plt-009",
        contexts={"audit-plt-009": ctx},
    )
    assert "audit-plt-009" in summary.audit_ids_persisted
    rows = store.read_audit_results()
    assert len(rows) == 1
    assert rows[0].component == "platelet"


# ═════════════════ Stage C2 — replay wiring ═════════════════


def test_platelet_rule_classification_is_platelet_derived(tmp_path):
    """C2a (Stage B MED-1): a platelet row's rule_classification comes from the
    platelet gate (count 50 → NEEDS_REVIEW, ceiling 100), NOT the RBC classifier
    run on the inert Hb sentinels (which would yield INSUFFICIENT_EVIDENCE)."""
    ctx = _platelet_ctx("audit-plt-c2a", platelet_count=50.0)
    # LLM returns NEEDS_REVIEW so no guardrail rewrites the verdict; the point is
    # the RULE column, which must reflect the platelet gate.
    response = RawBatchResponse(
        batch_id="msgbatch_c2a",
        results=(
            _platelet_result_item("audit-plt-c2a", classification="NEEDS_REVIEW"),
        ),
    )
    store = _audit_store(tmp_path)
    apply_batch_results(
        response,
        audit_store=store,
        run_id="run-c2a",
        contexts={"audit-plt-c2a": ctx},
    )
    row = store.read_audit_results()[0]
    assert row.rule_classification == "NEEDS_REVIEW"
    assert row.rule_classification != "INSUFFICIENT_EVIDENCE"
    assert row.platelet_review_ceiling == 100.0


def test_platelet_overclear_floors_ungrounded_appropriate(tmp_path, monkeypatch):
    """C2b: flag ON — an LLM APPROPRIATE on a sub-ceiling count with NO grounded
    hard signal floors to NEEDS_REVIEW with the platelet over-clear reason. This
    makes the ADD-hard-signals ruling real: a bare low count can never clear."""
    monkeypatch.setattr(feature_flags, "PLATELET_LLM_ENABLED", True)
    ctx = _platelet_ctx("audit-plt-c2b", platelet_count=50.0)
    response = RawBatchResponse(
        batch_id="msgbatch_c2b",
        results=(
            _platelet_result_item(
                "audit-plt-c2b",
                classification="APPROPRIATE",  # ungrounded: all hard signals False
            ),
        ),
    )
    store = _audit_store(tmp_path)
    apply_batch_results(
        response,
        audit_store=store,
        run_id="run-c2b",
        contexts={"audit-plt-c2b": ctx},
    )
    row = store.read_audit_results()[0]
    assert row.final_classification == "NEEDS_REVIEW"
    assert row.review_reason == PLATELET_OVERCLEAR_REVIEW_REASON
    assert row.needs_human_review is True


def test_platelet_grounded_hard_signal_clears(tmp_path, monkeypatch):
    """C2b: flag ON — an LLM APPROPRIATE WITH a grounded hard signal
    (active_bleeding) is NOT floored; the clear stands."""
    monkeypatch.setattr(feature_flags, "PLATELET_LLM_ENABLED", True)
    ctx = _platelet_ctx("audit-plt-c2c", platelet_count=50.0)
    response = RawBatchResponse(
        batch_id="msgbatch_c2c",
        results=(
            _platelet_result_item(
                "audit-plt-c2c",
                classification="APPROPRIATE",
                active_bleeding=True,
            ),
        ),
    )
    store = _audit_store(tmp_path)
    apply_batch_results(
        response,
        audit_store=store,
        run_id="run-c2c",
        contexts={"audit-plt-c2c": ctx},
    )
    row = store.read_audit_results()[0]
    assert row.final_classification == "APPROPRIATE"


def test_platelet_overclear_inert_when_flag_off(tmp_path):
    """C2e: flag OFF — the platelet over-clear guardrail does not fire, so an
    ungrounded APPROPRIATE is preserved (the leg is entirely inert)."""
    assert feature_flags.PLATELET_LLM_ENABLED is False
    ctx = _platelet_ctx("audit-plt-c2e", platelet_count=50.0)
    response = RawBatchResponse(
        batch_id="msgbatch_c2e",
        results=(_platelet_result_item("audit-plt-c2e", classification="APPROPRIATE"),),
    )
    store = _audit_store(tmp_path)
    apply_batch_results(
        response,
        audit_store=store,
        run_id="run-c2e",
        contexts={"audit-plt-c2e": ctx},
    )
    row = store.read_audit_results()[0]
    assert row.final_classification == "APPROPRIATE"


def test_periop_contradiction_does_not_fire_on_platelet(tmp_path):
    """C2b (Stage B MED-2): the RBC peri-op contradiction guardrail reads
    context.periop_summary; on a platelet row it must be gated OFF. An LLM
    INSUFFICIENT_EVIDENCE against a hard peri-op signal must stay
    INSUFFICIENT_EVIDENCE, not be escalated to NEEDS_REVIEW."""
    ctx = _platelet_ctx(
        "audit-plt-periop",
        platelet_count=50.0,
        periop_summary=PeriopSummary(surgical_context=True),
    )
    response = RawBatchResponse(
        batch_id="msgbatch_periop",
        results=(
            _platelet_result_item(
                "audit-plt-periop", classification="INSUFFICIENT_EVIDENCE"
            ),
        ),
    )
    store = _audit_store(tmp_path)
    apply_batch_results(
        response,
        audit_store=store,
        run_id="run-periop-plt",
        contexts={"audit-plt-periop": ctx},
    )
    row = store.read_audit_results()[0]
    assert row.final_classification == "INSUFFICIENT_EVIDENCE"


# ═════════════════ Stage C2 — submission (C2c) ═════════════════


def test_platelet_submission_uses_platelet_prompt():
    """C2c: a platelet context builds a PLATELET_REVIEW request with the platelet
    system prompt and cohort_threshold=None; an RBC context is unchanged."""
    plt_ctx = _platelet_ctx(
        "audit-plt-sub", platelet_count=50.0, evidence_chunks=_evidence_chunk()
    )
    requests = _build_submission_requests([plt_ctx], run_id="run-sub")
    assert len(requests) == 1
    assert requests[0].task_mode == "PLATELET_REVIEW"
    assert requests[0].prompt.task_mode == "PLATELET_REVIEW"
    assert requests[0].prompt.cohort_threshold is None


def test_rbc_submission_unchanged_by_platelet_branch(tmp_path):
    """C2c: the RBC submission path stays HB_7_10_REVIEW with a cohort threshold —
    the platelet branch must not touch RBC request-building."""
    from bba.cohort_detector import CohortAssignment, CohortLabel
    from bba.hb_lookup import HbLookupResult
    from bba.vitals_extractor import SourceProvenance, VitalSigns, VitalsResult

    order = AuditOrder(
        audit_id="audit-rbc-sub",
        hn="HN-rbc-sub",
        an="AN-rbc-sub",
        reqno="REQ-rbc-sub",
        order_datetime=_RUN_TS,
        anchor_imputed=False,
        products_ordered=("LPRC",),
        diagnosis_codes=("D62",),
    )
    ctx = PipelineRowContext(
        order=order,
        hb_result=HbLookupResult(
            value_g_dl=8.0,
            datetime_utc=_RUN_TS,
            source="HEMATOLOGY",
            freshness="fresh",
            delta_hb_bypass=False,
            delta_hb_windows=(),
            needs_review_single_low_hb=False,
        ),
        vitals_result=VitalsResult(
            vitals=VitalSigns(),
            source=SourceProvenance.NONE_IN_WINDOW,
            flags=frozenset(),
            note_timestamp=None,
        ),
        cohort_assignment=CohortAssignment(
            label=CohortLabel.UNKNOWN,
            threshold=7.5,
            evidence_code=None,
            evidence_name=None,
        ),
        procedure_proximity_hours=None,
        crystalloid_liters_prior_4h=0.0,
        hn_hash="hn_rbc_sub",
        an_hash="an_rbc_sub",
        prior_rbc_units_24h=0,
        prior_rbc_units_7d=0,
        redactor_version="0.4.1+test",
        redactor_model_sha="sha_rbc_sub",
        policy_version="kcmh-pr17.2-2024",
        prompt_hash="ph_rbc_sub",
        evidence_bundle_hash="bh_rbc_sub",
        evidence_chunks=_evidence_chunk(),
    )
    requests = _build_submission_requests([ctx], run_id="run-sub")
    assert requests[0].task_mode == "HB_7_10_REVIEW"
    assert requests[0].prompt.cohort_threshold == 7.5


def test_run_pipeline_platelet_leg_submits_when_flag_on(tmp_path, monkeypatch):
    """C2c + C2e end-to-end: with the flag ON, a NEEDS_REVIEW platelet row is
    submitted through the platelet prompt and persisted, with the platelet-
    derived rule_classification and the over-clear guardrail applied (ungrounded
    APPROPRIATE → floored to NEEDS_REVIEW)."""
    monkeypatch.setattr(feature_flags, "PLATELET_LLM_ENABLED", True)
    ctx = _platelet_ctx(
        "audit-plt-run", platelet_count=50.0, evidence_chunks=_evidence_chunk()
    )
    interaction = CassetteInteraction(
        model=SONNET_MODEL_ID,
        custom_ids=("audit-plt-run",),
        response=RawBatchResponse(
            batch_id="msgbatch_run",
            results=(
                _platelet_result_item("audit-plt-run", classification="APPROPRIATE"),
            ),
        ),
    )
    store = _audit_store(tmp_path)
    result = run_pipeline(
        [ctx],
        transport=CassetteTransport(interactions=(interaction,)),
        audit_store=store,
        batch_run_store=InMemoryBatchRunStore(),
        llm_config=_LLM_CONFIG,
        pipeline_config=_PIPELINE_CONFIG,
        run_id="run-plt-run",
    )
    assert "audit-plt-run" in result.audit_ids_persisted
    row = store.read_audit_results()[0]
    assert row.component == "platelet"
    assert row.rule_classification == "NEEDS_REVIEW"
    # ungrounded APPROPRIATE on a sub-ceiling count → over-clear guardrail floors
    assert row.final_classification == "NEEDS_REVIEW"
    assert row.review_reason == PLATELET_OVERCLEAR_REVIEW_REASON
