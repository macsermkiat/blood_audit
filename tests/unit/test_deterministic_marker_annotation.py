"""Deterministic marker reservation annotation tests for ticket #178."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from bba import feature_flags
from bba.audit_orders import AuditOrder
from bba.audit_pipeline.models import PipelineRowContext
from bba.audit_pipeline.pipeline import (
    _deterministic_marker_call,
    _platelet_marker_call,
)
from bba.cohort_detector import CohortAssignment, CohortLabel
from bba.deterministic_classifier import BypassReason, ClassifierResult
from bba.hb_lookup import HbLookupResult
from bba.platelet_classifier import PlateletClassifierResult
from bba.platelet_lookup import PlateletLookupResult
from bba.preop_reservation import (
    PlateletReservationDecision,
    ReservationDecision,
)
from bba.vitals_extractor import SourceProvenance, VitalSigns, VitalsResult


_NOW = datetime(2026, 7, 16, 4, 0, tzinfo=UTC)
_RUN_ID = "run-deterministic-marker-annotation"


def _rbc_decision() -> ReservationDecision:
    return ReservationDecision(
        resolved_icd9="1000",
        msbos="none",
        recommended_units=0,
        reserved_units=2,
        is_over=True,
        reason="over_none",
        reference_hash="a" * 64,
        note_resolved=True,
    )


def _rbc_context(*, decision: ReservationDecision | None) -> PipelineRowContext:
    return PipelineRowContext(
        order=AuditOrder(
            audit_id="audit-deterministic-marker-rbc",
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


def _rbc_classifier(classification: str) -> ClassifierResult:
    return ClassifierResult(
        classification=classification,
        bypass_reason=BypassReason.NONE,
        cohort_threshold=7.0,
        rationale="returns_terminal",
    )


def _platelet_decision() -> PlateletReservationDecision:
    return PlateletReservationDecision(
        resolved_icd9="1234",
        category="major_non_neuraxial",
        pre_op_count_k_ul=120.0,
        over_above_per_ul=80_000,
        reserved_units=2,
        is_over=True,
        reason="over_major_non_neuraxial",
        reference_hash="b" * 64,
        clinician_signed=True,
    )


def _platelet_context(
    *, decision: PlateletReservationDecision | None
) -> PipelineRowContext:
    return PipelineRowContext.for_platelet(
        order=AuditOrder(
            audit_id="audit-deterministic-marker-platelet",
            hn="HN1",
            an="AN1",
            reqno="REQ-PLT",
            order_datetime=_NOW,
            anchor_imputed=False,
            products_ordered=("PLT-POOL",),
            diagnosis_codes=(),
            component="platelet",
        ),
        platelet_result=PlateletLookupResult(
            value_k_ul=120.0,
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


def _platelet_classifier(classification: str) -> PlateletClassifierResult:
    return PlateletClassifierResult(
        classification=classification,
        rationale="returns_terminal",
        review_ceiling=100.0,
    )


@pytest.mark.parametrize(
    "classification", ["RETURNED_NOT_TRANSFUSED", "PERIOP_TRANSFUSION_EXEMPT"]
)
def test_rbc_marker_includes_reservation_annotation_for_returns_terminals(
    monkeypatch: pytest.MonkeyPatch, classification: str
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)

    call = _deterministic_marker_call(
        context=_rbc_context(decision=_rbc_decision()),
        classifier_result=_rbc_classifier(classification),
        run_id=_RUN_ID,
    )

    assert call.model_id == "deterministic"
    assert call.response_json == {
        "classification": classification,
        "rationale": "returns_terminal",
        "reservation_annotation": {
            "reserved_units": 2,
            "msbos": "none",
            "recommended_units": 0,
            "is_over": True,
            "reason": "over_none",
            "resolved_icd9": "1000",
            "note_resolved": True,
            "reference_hash": "a" * 64,
        },
    }


def test_rbc_marker_omits_annotation_when_flag_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", False)

    call = _deterministic_marker_call(
        context=_rbc_context(decision=_rbc_decision()),
        classifier_result=_rbc_classifier("RETURNED_NOT_TRANSFUSED"),
        run_id=_RUN_ID,
    )

    assert call.response_json == {
        "classification": "RETURNED_NOT_TRANSFUSED",
        "rationale": "returns_terminal",
    }


def test_rbc_marker_omits_annotation_without_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)

    call = _deterministic_marker_call(
        context=_rbc_context(decision=None),
        classifier_result=_rbc_classifier("RETURNED_NOT_TRANSFUSED"),
        run_id=_RUN_ID,
    )

    assert call.response_json == {
        "classification": "RETURNED_NOT_TRANSFUSED",
        "rationale": "returns_terminal",
    }


def test_rbc_marker_omits_annotation_for_non_returns_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)

    call = _deterministic_marker_call(
        context=_rbc_context(decision=_rbc_decision()),
        classifier_result=_rbc_classifier("APPROPRIATE"),
        run_id=_RUN_ID,
    )

    assert call.response_json == {
        "classification": "APPROPRIATE",
        "rationale": "returns_terminal",
    }


@pytest.mark.parametrize(
    "classification", ["RETURNED_NOT_TRANSFUSED", "PERIOP_TRANSFUSION_EXEMPT"]
)
def test_platelet_marker_includes_reservation_annotation_for_returns_terminals(
    monkeypatch: pytest.MonkeyPatch, classification: str
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)

    call = _platelet_marker_call(
        context=_platelet_context(decision=_platelet_decision()),
        classifier_result=_platelet_classifier(classification),
        run_id=_RUN_ID,
    )

    assert call.model_id == "deterministic"
    assert call.response_json == {
        "classification": classification,
        "rationale": "returns_terminal",
        "reservation_annotation": {
            "reserved_units": 2,
            "category": "major_non_neuraxial",
            "pre_op_count_k_ul": 120.0,
            "over_above_per_ul": 80_000,
            "is_over": True,
            "reason": "over_major_non_neuraxial",
            "clinician_signed": True,
            "reference_hash": "b" * 64,
        },
    }


def test_platelet_marker_omits_annotation_when_flag_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", False)

    call = _platelet_marker_call(
        context=_platelet_context(decision=_platelet_decision()),
        classifier_result=_platelet_classifier("RETURNED_NOT_TRANSFUSED"),
        run_id=_RUN_ID,
    )

    assert call.response_json == {
        "classification": "RETURNED_NOT_TRANSFUSED",
        "rationale": "returns_terminal",
    }


def test_platelet_marker_omits_annotation_without_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)

    call = _platelet_marker_call(
        context=_platelet_context(decision=None),
        classifier_result=_platelet_classifier("RETURNED_NOT_TRANSFUSED"),
        run_id=_RUN_ID,
    )

    assert call.response_json == {
        "classification": "RETURNED_NOT_TRANSFUSED",
        "rationale": "returns_terminal",
    }


def test_platelet_marker_omits_annotation_for_non_returns_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)

    call = _platelet_marker_call(
        context=_platelet_context(decision=_platelet_decision()),
        classifier_result=_platelet_classifier("INSUFFICIENT_EVIDENCE"),
        run_id=_RUN_ID,
    )

    assert call.response_json == {
        "classification": "INSUFFICIENT_EVIDENCE",
        "rationale": "returns_terminal",
    }
