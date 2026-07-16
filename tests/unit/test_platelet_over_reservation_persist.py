"""Platelet reservation overlay and persistence tests for ticket #166."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from bba import feature_flags
from bba.audit_orders import AuditOrder
from bba.audit_pipeline.models import PipelineRowContext
from bba.audit_pipeline.pipeline import (
    _persist_platelet_over_reservation_row,
    _persist_platelet_reservation_review_row,
)
from bba.audit_pipeline.replay import (
    PLATELET_RESERVATION_REVIEW_REASON,
    PREOP_OVER_RESERVATION_REVIEW_REASON,
    _LLM_ASSERT_REASONS,
    is_platelet_over_reservation,
    is_platelet_reservation_review,
)
from bba.audit_store import AuditStore, AuditStoreConfig
from bba.eval_harness import FalsificationOutcome, outcome_anchored_falsification
from bba.platelet_classifier import PlateletClassifierResult
from bba.platelet_lookup import PlateletLookupResult
from bba.preop_reservation import PlateletReservationDecision


_NOW = datetime(2026, 7, 16, 4, 0, tzinfo=UTC)
_RUN_ID = "run-platelet-msbos-test"


def _platelet_result(classification: str) -> PlateletClassifierResult:
    return PlateletClassifierResult(
        classification=classification,
        rationale="platelet_count_gate",
        review_ceiling=100.0,
    )


def _decision(*, review: bool = False) -> PlateletReservationDecision:
    return PlateletReservationDecision(
        resolved_icd9="1234",
        category="major_non_neuraxial",
        pre_op_count_k_ul=120.0,
        threshold_per_ul=80_000,
        high_risk_ceiling_per_ul=100_000,
        reserved_units=2,
        is_over=not review,
        reason=(
            "gray_band_major_non_neuraxial" if review else "over_major_non_neuraxial"
        ),
        reference_hash="a" * 64,
        seed_pending_signoff=True,
    )


def _context(*, review: bool = False) -> PipelineRowContext:
    decision = _decision(review=review)
    return PipelineRowContext.for_platelet(
        order=AuditOrder(
            audit_id=f"audit-platelet-msbos-{'review' if review else 'over'}",
            hn="HN1",
            an="AN1",
            reqno=f"REQ-{'review' if review else 'over'}",
            order_datetime=_NOW,
            anchor_imputed=False,
            products_ordered=("PLT-POOL",),
            diagnosis_codes=(),
            component="platelet",
        ),
        platelet_result=PlateletLookupResult(
            value_k_ul=decision.pre_op_count_k_ul,
            datetime_utc=_NOW,
            source="HEMATOLOGY",
            freshness="fresh",
        ),
        hn_hash="hn_hash",
        an_hash="an_hash",
        redactor_version="test",
        redactor_model_sha="sha",
        policy_version="test",
        prompt_hash="prompt",
        evidence_bundle_hash="bundle",
        platelet_reservation_decision=decision,
    )


@pytest.mark.parametrize(
    ("enabled", "classification", "expected_over", "expected_review"),
    [
        (False, "POTENTIALLY_INAPPROPRIATE", False, False),
        (True, "RETURNED_NOT_TRANSFUSED", False, False),
        (True, "PERIOP_TRANSFUSION_EXEMPT", False, False),
        (True, "POTENTIALLY_INAPPROPRIATE", True, True),
    ],
)
def test_platelet_overlays_are_flag_gated_snapshot_only_and_returns_safe(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    classification: str,
    expected_over: bool,
    expected_review: bool,
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", enabled)
    result = _platelet_result(classification)

    assert (
        is_platelet_over_reservation(
            classifier_result=result, context=_context(review=False)
        )
        is expected_over
    )
    assert (
        is_platelet_reservation_review(
            classifier_result=result, context=_context(review=True)
        )
        is expected_review
    )


@pytest.mark.parametrize(
    ("review", "expected_final", "expected_reason"),
    [
        (False, "PREOP_OVER_RESERVATION", PREOP_OVER_RESERVATION_REVIEW_REASON),
        (True, "NEEDS_REVIEW", PLATELET_RESERVATION_REVIEW_REASON),
    ],
)
def test_platelet_persist_helpers_write_paired_marker_idempotently(
    tmp_path: Path,
    review: bool,
    expected_final: str,
    expected_reason: str,
) -> None:
    store = AuditStore(
        AuditStoreConfig(root_dir=tmp_path / "store", code_version="test+msbos3")
    )
    context = _context(review=review)
    persist = (
        _persist_platelet_reservation_review_row
        if review
        else _persist_platelet_over_reservation_row
    )

    persist(context, audit_store=store, run_id=_RUN_ID)
    persist(context, audit_store=store, run_id=_RUN_ID)

    (row,) = store.read_audit_results(run_id=_RUN_ID)
    (call,) = store.read_llm_calls(run_id=_RUN_ID)
    decision = context.platelet_reservation_decision
    assert decision is not None
    assert row.final_classification == expected_final
    assert row.review_reason == expected_reason
    assert row.component == "platelet"
    assert call.model_id == "msbos-platelet-reservation"
    expected_request = {
        "audit_id": context.order.audit_id,
        "reason": decision.reason,
        "resolved_icd9": decision.resolved_icd9,
        "category": decision.category,
        "pre_op_count_k_ul": decision.pre_op_count_k_ul,
        "reserved_units": decision.reserved_units,
        "seed_pending_signoff": decision.seed_pending_signoff,
    }
    expected_request[
        "platelet_reservation_review" if review else "platelet_over_reservation"
    ] = True
    assert call.request_json == expected_request
    assert call.response_json == {
        "classification": expected_final,
        "review_reason": expected_reason,
    }
    assert store.reconcile(_RUN_ID).orphan_audit_ids == ()
    assert store.reconcile(_RUN_ID).orphan_call_ids == ()


def test_platelet_reservation_review_is_not_an_llm_assert_reason() -> None:
    assert PLATELET_RESERVATION_REVIEW_REASON not in _LLM_ASSERT_REASONS


def test_platelet_over_row_uses_existing_component_blind_inappropriate_fold(
    tmp_path: Path,
) -> None:
    store = AuditStore(
        AuditStoreConfig(root_dir=tmp_path / "store", code_version="test+msbos3")
    )
    _persist_platelet_over_reservation_row(
        _context(review=False), audit_store=store, run_id=_RUN_ID
    )
    (row,) = store.read_audit_results(run_id=_RUN_ID)

    result = outcome_anchored_falsification(
        predictions=[row.final_classification],
        outcomes=[FalsificationOutcome.FURTHER_TRANSFUSION_24H],
    )

    assert row.component == "platelet"
    assert result.n_inappropriate_pred == 1, (
        "the pure fold is component-blind; T4 does not broaden RBC-only reports"
    )
