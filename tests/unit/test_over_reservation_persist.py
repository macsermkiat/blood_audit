"""MSBOS overlay and persistence tests for ticket #163."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from bba import feature_flags
from bba.audit_orders import AuditOrder
from bba.audit_pipeline.models import PipelineRowContext
from bba.audit_pipeline.pipeline import _persist_over_reservation_row
from bba.audit_pipeline.replay import (
    PREOP_OVER_RESERVATION_REVIEW_REASON,
    _audit_row_for_over_reservation,
    is_over_reservation,
)
from bba.audit_store import AuditStore, AuditStoreConfig
from bba.cohort_detector import CohortAssignment, CohortLabel
from bba.deterministic_classifier import BypassReason, ClassifierResult
from bba.hb_lookup import HbLookupResult
from bba.preop_reservation import ReservationDecision
from bba.vitals_extractor import SourceProvenance, VitalSigns, VitalsResult


_NOW = datetime(2026, 7, 16, 4, 0, tzinfo=UTC)
_RUN_ID = "run-msbos-test"


def _context(*, decision: ReservationDecision | None) -> PipelineRowContext:
    return PipelineRowContext(
        order=AuditOrder(
            audit_id="audit-msbos-test",
            hn="HN1",
            an="AN1",
            reqno="REQ1",
            order_datetime=_NOW,
            anchor_imputed=False,
            products_ordered=("LPRC",),
            diagnosis_codes=(),
        ),
        hb_result=HbLookupResult(
            value_g_dl=9.0,
            datetime_utc=_NOW,
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
            label=CohortLabel.DEFAULT,
            threshold=7.0,
            evidence_code=None,
            evidence_name=None,
        ),
        procedure_proximity_hours=None,
        crystalloid_liters_prior_4h=0.0,
        hn_hash="hn_hash",
        an_hash="an_hash",
        prior_rbc_units_24h=0,
        prior_rbc_units_7d=0,
        redactor_version="test",
        redactor_model_sha="sha",
        policy_version="test",
        prompt_hash="prompt",
        evidence_bundle_hash="bundle",
        reservation_decision=decision,
    )


def _decision(*, is_over: bool = True) -> ReservationDecision:
    return ReservationDecision(
        resolved_icd9="1000",
        msbos="none",
        recommended_units=0,
        reserved_units=2,
        is_over=is_over,
        reason="over_none" if is_over else "within_recommendation",
        reference_hash="a" * 64,
    )


def _classifier(classification: str = "POTENTIALLY_INAPPROPRIATE") -> ClassifierResult:
    return ClassifierResult(
        classification=classification,
        bypass_reason=BypassReason.NONE,
        cohort_threshold=7.0,
        rationale="preop_defer_llm",
    )


def test_over_reservation_builder_preserves_rule_and_survives_empty_reasoning() -> None:
    row = _audit_row_for_over_reservation(
        run_id=_RUN_ID,
        context=_context(decision=_decision()),
        classifier_result=_classifier(),
        review_reason=PREOP_OVER_RESERVATION_REVIEW_REASON,
        verifier_pass=True,
        verifier_retries=0,
        model_id="msbos-reservation",
        reasoning_en=(
            "Pre-op RBC reservation exceeds the MSBOS recommendation for the "
            "planned operation; not submitted to LLM."
        ),
        reasoning_th="",
        indications=(),
        negative_evidence=(),
        confidence=1.0,
        escalated=False,
    )

    assert row.final_classification == "PREOP_OVER_RESERVATION"
    assert row.rule_classification == "POTENTIALLY_INAPPROPRIATE"
    assert row.review_reason == PREOP_OVER_RESERVATION_REVIEW_REASON
    assert row.needs_human_review is True
    assert row.reasoning_summary_en
    assert row.reasoning_summary_thai == ""
    assert row.confidence == 1.0
    assert row.model_id == "msbos-reservation"


def test_persist_over_reservation_writes_one_row_and_call_idempotently(
    tmp_path: Path,
) -> None:
    store = AuditStore(
        AuditStoreConfig(root_dir=tmp_path / "store", code_version="test+msbos")
    )
    context = _context(decision=_decision())
    classifier = _classifier()

    _persist_over_reservation_row(
        context,
        classifier_result=classifier,
        audit_store=store,
        run_id=_RUN_ID,
    )
    _persist_over_reservation_row(
        context,
        classifier_result=classifier,
        audit_store=store,
        run_id=_RUN_ID,
    )

    (row,) = store.read_audit_results(run_id=_RUN_ID)
    (call,) = store.read_llm_calls(run_id=_RUN_ID)
    assert row.final_classification == "PREOP_OVER_RESERVATION"
    assert call.model_id == "msbos-reservation"
    assert call.response_json["classification"] == "PREOP_OVER_RESERVATION"
    assert store.reconcile(_RUN_ID).orphan_audit_ids == ()
    assert store.reconcile(_RUN_ID).orphan_call_ids == ()


@pytest.mark.parametrize(
    ("enabled", "decision", "classification", "expected"),
    [
        (False, _decision(), "POTENTIALLY_INAPPROPRIATE", False),
        (True, None, "POTENTIALLY_INAPPROPRIATE", False),
        (True, _decision(is_over=False), "POTENTIALLY_INAPPROPRIATE", False),
        (True, _decision(), "RETURNED_NOT_TRANSFUSED", False),
        (True, _decision(), "PERIOP_TRANSFUSION_EXEMPT", False),
        (True, _decision(), "POTENTIALLY_INAPPROPRIATE", True),
    ],
)
def test_is_over_reservation_overlay(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    decision: ReservationDecision | None,
    classification: str,
    expected: bool,
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", enabled)

    assert (
        is_over_reservation(
            classifier_result=_classifier(classification),
            context=_context(decision=decision),
        )
        is expected
    )
