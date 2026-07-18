"""MSBOS overlay and persistence tests for ticket #163."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from bba import feature_flags
from bba.audit_orders import AuditOrder
from bba.audit_pipeline.models import PipelineRowContext
from bba.audit_pipeline.pipeline import (
    _persist_operation_unresolved_row,
    _persist_over_reservation_row,
)
from bba.audit_pipeline.replay import (
    OPERATION_UNRESOLVED_REVIEW_REASON,
    PREOP_OVER_RESERVATION_REVIEW_REASON,
    _LLM_ASSERT_REASONS,
    _audit_row_for_operation_unresolved,
    _audit_row_for_over_reservation,
    is_operation_unresolved,
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
        note_resolved=is_over,
    )


def _unresolved_decision() -> ReservationDecision:
    return ReservationDecision(
        resolved_icd9="0124",
        reserved_units=2,
        reason="operation_unresolved",
        reference_hash="a" * 64,
    )


def _classifier(
    classification: str = "POTENTIALLY_INAPPROPRIATE",
    *,
    rationale: str = "preop_defer_llm",
) -> ClassifierResult:
    return ClassifierResult(
        classification=classification,
        bypass_reason=BypassReason.NONE,
        cohort_threshold=7.0,
        rationale=rationale,
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
    assert call.request_json["reason"] == "over_none", (
        "the marker must persist the exact once-computed decision reason"
    )
    assert call.request_json["resolved_icd9"] == "1000"
    assert call.request_json["note_resolved"] is True, (
        "T5 must be able to separate note-resolved precision from persisted markers"
    )
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


@pytest.mark.parametrize(
    ("enabled", "decision", "classification", "expected"),
    [
        (False, _unresolved_decision(), "POTENTIALLY_INAPPROPRIATE", False),
        (True, None, "POTENTIALLY_INAPPROPRIATE", False),
        (True, _decision(), "POTENTIALLY_INAPPROPRIATE", False),
        (True, _unresolved_decision(), "RETURNED_NOT_TRANSFUSED", False),
        (True, _unresolved_decision(), "PERIOP_TRANSFUSION_EXEMPT", False),
        (True, _unresolved_decision(), "POTENTIALLY_INAPPROPRIATE", True),
    ],
)
def test_is_operation_unresolved_overlay_is_flagged_snapshot_and_returns_safe(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    decision: ReservationDecision | None,
    classification: str,
    expected: bool,
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", enabled)

    assert (
        is_operation_unresolved(
            classifier_result=_classifier(classification),
            context=_context(decision=decision),
        )
        is expected
    ), "only a flag-on unresolved snapshot outside returns terminals may overlay"


@pytest.mark.parametrize(
    "classification", ["RETURNED_NOT_TRANSFUSED", "PERIOP_TRANSFUSION_EXEMPT"]
)
def test_returns_terminal_fires_neither_reservation_overlay(
    monkeypatch: pytest.MonkeyPatch, classification: str
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)
    classifier = _classifier(classification)

    assert not is_over_reservation(
        classifier_result=classifier, context=_context(decision=_decision())
    ), "returns disposition outranks the over-reservation terminal"
    assert not is_operation_unresolved(
        classifier_result=classifier, context=_context(decision=_unresolved_decision())
    ), "returns disposition also outranks the unresolved-review terminal"


def test_declared_preop_exempt_remains_msbos_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)
    declared = _classifier(
        "PERIOP_TRANSFUSION_EXEMPT", rationale="preop_declared_exempt"
    )
    legacy = _classifier(
        "PERIOP_TRANSFUSION_EXEMPT", rationale="periop_transfusion_exempt"
    )

    assert is_over_reservation(
        classifier_result=declared, context=_context(decision=_decision())
    )
    assert not is_over_reservation(
        classifier_result=legacy, context=_context(decision=_decision())
    )


def test_operation_unresolved_builder_is_needs_review_not_llm_assertion() -> None:
    kwargs = {
        "run_id": _RUN_ID,
        "context": _context(decision=_unresolved_decision()),
        "classifier_result": _classifier(),
        "review_reason": "caller-value-is-overridden",
        "verifier_pass": True,
        "verifier_retries": 0,
        "model_id": "msbos-reservation",
        "reasoning_en": "The conflicting operation code could not be resolved.",
        "reasoning_th": "",
        "indications": (),
        "negative_evidence": (),
        "confidence": 1.0,
        "escalated": False,
    }

    first = _audit_row_for_operation_unresolved(**kwargs)
    second = _audit_row_for_operation_unresolved(**kwargs)

    assert first == second, "the same in-run snapshot must replay identically"
    assert first.final_classification == "NEEDS_REVIEW", (
        "unresolved operation identity requires a human; it is not a new verdict class"
    )
    assert first.review_reason == OPERATION_UNRESOLVED_REVIEW_REASON
    assert OPERATION_UNRESOLVED_REVIEW_REASON not in _LLM_ASSERT_REASONS, (
        "the unresolved path must never become an asserted LLM verdict"
    )


def test_persist_operation_unresolved_writes_deterministic_review_marker(
    tmp_path: Path,
) -> None:
    store = AuditStore(
        AuditStoreConfig(root_dir=tmp_path / "store", code_version="test+msbos2")
    )
    context = _context(decision=_unresolved_decision())

    _persist_operation_unresolved_row(
        context,
        classifier_result=_classifier(),
        audit_store=store,
        run_id=_RUN_ID,
    )

    (row,) = store.read_audit_results(run_id=_RUN_ID)
    (call,) = store.read_llm_calls(run_id=_RUN_ID)
    expected_fingerprint = hashlib.sha256(
        f"{_RUN_ID}|{context.order.audit_id}|operation-unresolved".encode()
    ).hexdigest()[:16]
    assert row.final_classification == "NEEDS_REVIEW", (
        "unresolved conflicting codes are persisted for clinician review"
    )
    assert row.review_reason == OPERATION_UNRESOLVED_REVIEW_REASON
    assert call.model_id == "msbos-reservation"
    assert call.call_id.endswith(expected_fingerprint), (
        "the distinct operation-unresolved suffix must determine marker identity"
    )
    assert call.response_json == {
        "classification": "NEEDS_REVIEW",
        "review_reason": OPERATION_UNRESOLVED_REVIEW_REASON,
    }
    assert store.reconcile(_RUN_ID).orphan_audit_ids == ()
    assert store.reconcile(_RUN_ID).orphan_call_ids == ()
