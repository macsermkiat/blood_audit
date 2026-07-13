"""Regression tests for Hb-keyed RBC task-mode dispatch (ticket #93)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from bba import feature_flags
from bba.audit_orders import AuditOrder
from bba.audit_pipeline.models import BatchRun, BatchRunState, PipelineRowContext
from bba.audit_pipeline.pipeline import (
    _build_submission_requests,
    _classifier_inputs_for,
    rbc_task_mode,
)
from bba.audit_pipeline.resume import _rebuild_submission_requests
from bba.cohort_detector import CohortAssignment, CohortLabel
from bba.deterministic_classifier import classify
from bba.hb_lookup import HbLookupResult
from bba.platelet_lookup.models import PlateletLookupResult
from bba.prompt_builder import EvidenceChunk
from bba.returns_ledger import ReturnsSummary
from bba.vitals_extractor import SourceProvenance, VitalSigns, VitalsResult


_RUN_TS = datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)


def _evidence_chunk() -> tuple[EvidenceChunk, ...]:
    return (EvidenceChunk(evidence_id="E1", source="Lab", text="Hb reviewed"),)


def _rbc_ctx(
    audit_id: str,
    hb_value: float | None,
    *,
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
    hb_result = HbLookupResult(
        value_g_dl=hb_value,
        datetime_utc=_RUN_TS if hb_value is not None else None,
        source="HEMATOLOGY" if hb_value is not None else None,
        freshness="fresh" if hb_value is not None else "missing",
        delta_hb_bypass=False,
        delta_hb_windows=(),
        needs_review_single_low_hb=False,
    )
    vitals_result = VitalsResult(
        vitals=VitalSigns(),
        source=SourceProvenance.NONE_IN_WINDOW,
        flags=frozenset(),
        note_timestamp=None,
    )
    cohort_assignment = CohortAssignment(
        label=CohortLabel.CARDIAC_SURGERY,
        threshold=7.5,
        evidence_code=None,
        evidence_name=None,
    )
    return PipelineRowContext(
        order=order,
        hb_result=hb_result,
        vitals_result=vitals_result,
        cohort_assignment=cohort_assignment,
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
        evidence_chunks=_evidence_chunk(),
    )


def _batch_run(*audit_ids: str) -> BatchRun:
    return BatchRun(
        batch_id="batch-rbc-task-mode",
        state=BatchRunState.SUBMITTED,
        run_id="run-rbc-task-mode",
        code_version="v0.1.0+test",
        audit_ids=audit_ids,
        anthropic_batch_id="msgbatch_rbc_task_mode",
        submitted_at=_RUN_TS,
        updated_at=_RUN_TS,
    )


@pytest.mark.parametrize("hb_value", [12.3, 10.1, 10.0])
def test_rbc_task_mode_selects_override_at_or_above_ten(hb_value: float):
    """Ticket #93 occurred because high-Hb rows were hard-coded to gray-zone mode.

    Exactly 10.0 belongs to the override side: the deterministic engine's
    ``hb_ge_10`` branch assigns POTENTIALLY_INAPPROPRIATE for Hb >= 10.0
    (classifier.py), and the dispatch must agree with the rule verdict.
    """
    assert rbc_task_mode(hb_value) == "HB_GT_10_OVERRIDE"


@pytest.mark.parametrize("hb_value", [9.9, 8.0, 7.0])
def test_rbc_task_mode_keeps_sub_ten_hb_in_gray_zone(hb_value: float):
    """Ticket #93 must not move gray-zone (Hb < 10) reviews onto the override prompt."""
    assert rbc_task_mode(hb_value) == "HB_7_10_REVIEW"


def test_rbc_task_mode_keeps_missing_hb_in_gray_zone():
    """Ticket #93 cannot infer Hb >10 when the measured Hb value is missing."""
    assert rbc_task_mode(None) == "HB_7_10_REVIEW"


def test_rbc_task_mode_selects_reserve_ahead_without_changing_default() -> None:
    assert rbc_task_mode(12.9) == "HB_GT_10_OVERRIDE"
    assert rbc_task_mode(12.9, reserve_ahead=True) == "RESERVE_AHEAD_REVIEW"


def test_pipeline_dispatches_high_hb_rbc_to_override():
    """Ticket #93 broke live high-Hb dispatch and threshold pass-through."""
    requests = _build_submission_requests(
        [_rbc_ctx("audit-rbc-high", 12.3)], run_id="run-rbc-task-mode"
    )

    assert len(requests) == 1
    request = requests[0]
    assert request.task_mode == "HB_GT_10_OVERRIDE"
    assert request.prompt.task_mode == "HB_GT_10_OVERRIDE"
    assert request.prompt.cohort_threshold == 7.5


def test_pipeline_routes_exact_threshold_rbc_to_override():
    """Ticket #93: at exactly Hb 10.0 the engine's rule verdict is
    POTENTIALLY_INAPPROPRIATE (``hb_ge_10``), so the live request must carry
    the override prompt — a gray-zone prompt here would recreate the
    prompt/rule disagreement the ticket removes."""
    request = _build_submission_requests(
        [_rbc_ctx("audit-rbc-boundary", 10.0)], run_id="run-rbc-task-mode"
    )[0]

    assert request.task_mode == "HB_GT_10_OVERRIDE"
    assert request.prompt.task_mode == "HB_GT_10_OVERRIDE"


def test_pipeline_high_hb_preop_reserve_still_uses_override():
    """Ticket #93 must key on Hb, not the preop_defer_llm NEEDS_REVIEW verdict."""
    request = _build_submission_requests(
        [
            _rbc_ctx(
                "audit-rbc-high-preop",
                12.3,
                upcoming_procedure_hours=24.0,
            )
        ],
        run_id="run-rbc-task-mode",
    )[0]

    assert request.task_mode == "HB_GT_10_OVERRIDE"
    assert request.prompt.task_mode == "HB_GT_10_OVERRIDE"


def test_flag_off_preop_dispatch_is_byte_identical_with_classifier_threaded() -> None:
    ctx = _rbc_ctx("audit-rbc-flag-off", 12.9, upcoming_procedure_hours=24.0)
    classifier_result = classify(_classifier_inputs_for(ctx))
    assert feature_flags.RESERVE_AHEAD_ROUTER_ENABLED is False
    before = _build_submission_requests([ctx], run_id="run-rbc-task-mode")[
        0
    ].model_dump_json()
    threaded = _build_submission_requests(
        [ctx],
        run_id="run-rbc-task-mode",
        classifier_results={ctx.order.audit_id: classifier_result},
    )[0].model_dump_json()

    assert threaded == before
    assert "HB_GT_10_OVERRIDE" in threaded


def test_pipeline_routes_real_preop_classifier_result_when_flag_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _rbc_ctx("audit-rbc-case-68026306", 12.9, upcoming_procedure_hours=24.0)
    classifier_result = classify(_classifier_inputs_for(ctx))
    assert classifier_result.rationale == "preop_defer_llm"
    monkeypatch.setattr(feature_flags, "RESERVE_AHEAD_ROUTER_ENABLED", True)

    request = _build_submission_requests(
        [ctx],
        run_id="run-rbc-task-mode",
        classifier_results={ctx.order.audit_id: classifier_result},
    )[0]

    assert request.task_mode == "RESERVE_AHEAD_REVIEW"
    assert request.prompt.task_mode == "RESERVE_AHEAD_REVIEW"


def test_resume_dispatches_high_hb_rbc_to_override():
    """Ticket #93 also hard-coded resumed high-Hb RBC rows to gray-zone mode."""
    ctx = _rbc_ctx("audit-rbc-resume-high", 12.3)
    request = _rebuild_submission_requests(
        run=_batch_run(ctx.order.audit_id),
        contexts={ctx.order.audit_id: ctx},
        audit_ids=(ctx.order.audit_id,),
    )[0]

    assert request.task_mode == "HB_GT_10_OVERRIDE"
    assert request.prompt.task_mode == "HB_GT_10_OVERRIDE"


def test_resume_routes_exact_threshold_rbc_to_override():
    """Ticket #93 requires the resume rebuild to agree with live dispatch at
    the Hb 10.0 boundary (engine ``hb_ge_10`` → override prompt)."""
    ctx = _rbc_ctx("audit-rbc-resume-boundary", 10.0)
    request = _rebuild_submission_requests(
        run=_batch_run(ctx.order.audit_id),
        contexts={ctx.order.audit_id: ctx},
        audit_ids=(ctx.order.audit_id,),
    )[0]

    assert request.task_mode == "HB_GT_10_OVERRIDE"
    assert request.prompt.task_mode == "HB_GT_10_OVERRIDE"


def test_resume_flag_off_does_not_reclassify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bba.audit_pipeline.resume as resume_module

    ctx = _rbc_ctx("audit-rbc-resume-flag-off", 12.9, upcoming_procedure_hours=24.0)

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("flag-off resume must not call classify")

    monkeypatch.setattr(resume_module, "classify", fail_if_called)
    request = _rebuild_submission_requests(
        run=_batch_run(ctx.order.audit_id),
        contexts={ctx.order.audit_id: ctx},
        audit_ids=(ctx.order.audit_id,),
    )[0]

    assert request.task_mode == "HB_GT_10_OVERRIDE"


def test_resume_routes_real_preop_classifier_result_when_flag_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = _rbc_ctx(
        "audit-rbc-resume-case-68026306", 12.9, upcoming_procedure_hours=24.0
    )
    monkeypatch.setattr(feature_flags, "RESERVE_AHEAD_ROUTER_ENABLED", True)

    request = _rebuild_submission_requests(
        run=_batch_run(ctx.order.audit_id),
        contexts={ctx.order.audit_id: ctx},
        audit_ids=(ctx.order.audit_id,),
    )[0]

    assert request.task_mode == "RESERVE_AHEAD_REVIEW"
    assert request.prompt.task_mode == "RESERVE_AHEAD_REVIEW"


_ALL_RETURNED = ReturnsSummary(
    units_total=2,
    units_returned=2,
    ordered_unit_amount=2,
    ledger_complete=True,
)


def test_resume_reserve_ahead_reflects_real_returns_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ticket #124: resume must re-derive returns routing from the context's
    returns_summary, NOT the #122/#123 forced ``inconclusive``.

    A fully-returned order carrying an upcoming procedure would, under the old
    force, be re-classified as ``preop_defer_llm`` and mis-dispatched to
    RESERVE_AHEAD_REVIEW on resume. With the real disposition threaded, the
    classifier returns the terminal RETURNED_NOT_TRANSFUSED (rationale is not
    ``preop_defer_llm``), so reserve-ahead is off and the rebuild agrees with
    live dispatch. Encodes AC "resume re-derives the same terminal routing".
    """
    monkeypatch.setattr(feature_flags, "RETURNS_LEDGER_ENABLED", True)
    monkeypatch.setattr(feature_flags, "RESERVE_AHEAD_ROUTER_ENABLED", True)
    ctx = _rbc_ctx(
        "audit-rbc-resume-returned", 12.9, upcoming_procedure_hours=24.0
    ).model_copy(update={"returns_summary": _ALL_RETURNED})

    # The production composer that resume defers to now sees the real
    # disposition (proving the force is gone).
    assert _classifier_inputs_for(ctx).returns_disposition == "not_transfused"
    assert classify(_classifier_inputs_for(ctx)).rationale == "returned_not_transfused"

    live_task_mode = _build_submission_requests(
        [ctx],
        run_id="run-rbc-task-mode",
        classifier_results={ctx.order.audit_id: classify(_classifier_inputs_for(ctx))},
    )[0].task_mode
    resume_task_mode = _rebuild_submission_requests(
        run=_batch_run(ctx.order.audit_id),
        contexts={ctx.order.audit_id: ctx},
        audit_ids=(ctx.order.audit_id,),
    )[0].task_mode

    # No divergence across run / resume for the same returns-bearing context.
    assert resume_task_mode == live_task_mode == "HB_GT_10_OVERRIDE"


def test_resume_flag_off_ignores_returns_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ticket #124 flag-off byte identity: with RETURNS_LEDGER_ENABLED off, a
    context carrying a returns_summary must be inert — the composer yields
    ``inconclusive`` and resume keeps today's task-mode selection (the same the
    #122/#123 force produced), so enabling the flag is the only behavior change.
    """
    monkeypatch.setattr(feature_flags, "RETURNS_LEDGER_ENABLED", False)
    monkeypatch.setattr(feature_flags, "RESERVE_AHEAD_ROUTER_ENABLED", True)
    ctx = _rbc_ctx(
        "audit-rbc-resume-returned-flagoff", 12.9, upcoming_procedure_hours=24.0
    ).model_copy(update={"returns_summary": _ALL_RETURNED})

    assert _classifier_inputs_for(ctx).returns_disposition == "inconclusive"
    request = _rebuild_submission_requests(
        run=_batch_run(ctx.order.audit_id),
        contexts={ctx.order.audit_id: ctx},
        audit_ids=(ctx.order.audit_id,),
    )[0]

    # Unchanged from pre-#124: the preop reserve-ahead path still fires.
    assert request.task_mode == "RESERVE_AHEAD_REVIEW"


def test_platelet_dispatch_never_uses_rbc_selector():
    """Ticket #93's Hb selector must never reroute platelet rows, even with Hb >10."""
    rbc_ctx = _rbc_ctx("audit-platelet-high-hb", 12.9)
    platelet_ctx = rbc_ctx.model_copy(
        update={
            "component": "platelet",
            "platelet_result": PlateletLookupResult(
                value_k_ul=50.0,
                datetime_utc=_RUN_TS,
                source="HEMATOLOGY",
                freshness="fresh",
            ),
        }
    )

    request = _build_submission_requests([platelet_ctx], run_id="run-rbc-task-mode")[0]

    assert platelet_ctx.hb_result.value_g_dl == 12.9
    assert request.task_mode == "PLATELET_REVIEW"
    assert request.prompt.task_mode == "PLATELET_REVIEW"
