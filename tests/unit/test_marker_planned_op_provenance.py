"""Audit-store marker picker-v2 provenance tests for ticket #200.

Every affected marker must embed the FULL planned-op provenance when the
picker-v2 seam produced the pick, and stay byte-identical to the legacy
payload when it did not (``planned_op`` is None). Strict whole-dict
assertions; import-level only (recorder store, no filesystem writes).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from bba import feature_flags
from bba.audit_orders import AuditOrder
from bba.audit_pipeline.models import PipelineRowContext
from bba.audit_pipeline.pipeline import (
    _deterministic_marker_call,
    _persist_bridge_gate_review_row,
    _persist_over_reservation_row,
    _persist_platelet_bridge_gate_review_row,
    _persist_platelet_over_reservation_row,
    _platelet_marker_call,
)
from bba.cohort_detector import CohortAssignment, CohortLabel
from bba.deterministic_classifier import BypassReason, ClassifierResult
from bba.hb_lookup import HbLookupResult
from bba.platelet_classifier import PlateletClassifierResult
from bba.platelet_lookup import PlateletLookupResult
from bba.preop_reservation import PlateletReservationDecision, ReservationDecision
from bba.preop_reservation.models import PlannedOpProvenance
from bba.vitals_extractor import SourceProvenance, VitalSigns, VitalsResult

_NOW = datetime(2026, 7, 19, 4, 0, tzinfo=UTC)
_RUN_ID = "run-marker-planned-op"

_PROVENANCE = PlannedOpProvenance(
    source_code="P0614",
    source="incpt_bridge",
    bridge_icd9="3611",
    bridge_score=1.0,
    human_index="1",
    human_agreed=False,
    human_icd9="3612",
    pick_status="selected",
    candidate_count=3,
    tie_count=1,
    bridge_hash="f" * 64,
    gate="bridge_disagreement",
)

_PROVENANCE_DICT = {
    "source_code": "P0614",
    "source": "incpt_bridge",
    "bridge_icd9": "3611",
    "bridge_score": 1.0,
    "human_index": "1",
    "human_agreed": False,
    "human_icd9": "3612",
    "pick_status": "selected",
    "candidate_count": 3,
    "tie_count": 1,
    "bridge_hash": "f" * 64,
    "gate": "bridge_disagreement",
    "ceiling_token": "",
    "ceiling_units": None,
    "ceiling_codes": "",
}


class _RecorderStore:
    def __init__(self) -> None:
        self.rows: list[object] = []
        self.calls: list[object] = []

    def write(self, row: object, calls: list[object]) -> None:
        self.rows.append(row)
        self.calls.extend(calls)


def _rbc_decision(*, planned_op: PlannedOpProvenance | None) -> ReservationDecision:
    return ReservationDecision(
        resolved_icd9="3611",
        msbos="none",
        recommended_units=0,
        reserved_units=2,
        is_over=True,
        reason="over_none",
        reference_hash="a" * 64,
        note_resolved=False,
        planned_op=planned_op,
    )


def _platelet_decision(
    *, planned_op: PlannedOpProvenance | None
) -> PlateletReservationDecision:
    return PlateletReservationDecision(
        resolved_icd9="3611",
        category="major_non_neuraxial",
        pre_op_count_k_ul=120.0,
        over_above_per_ul=80_000,
        reserved_units=2,
        is_over=True,
        reason="over_major_non_neuraxial",
        reference_hash="b" * 64,
        clinician_signed=True,
        planned_op=planned_op,
    )


def _rbc_context(decision: ReservationDecision | None) -> PipelineRowContext:
    return PipelineRowContext(
        order=AuditOrder(
            audit_id="audit-marker-provenance-rbc",
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


def _platelet_context(
    decision: PlateletReservationDecision | None,
) -> PipelineRowContext:
    return PipelineRowContext.for_platelet(
        order=AuditOrder(
            audit_id="audit-marker-provenance-plt",
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


def _rbc_classifier() -> ClassifierResult:
    return ClassifierResult(
        classification="PERIOP_TRANSFUSION_EXEMPT",
        bypass_reason=BypassReason.NONE,
        cohort_threshold=7.0,
        rationale="preop_declared_exempt",
    )


def _platelet_classifier() -> PlateletClassifierResult:
    return PlateletClassifierResult(
        classification="PERIOP_TRANSFUSION_EXEMPT",
        rationale="returns_terminal",
        review_ceiling=100.0,
    )


# --- returns-annotation markers (#178 shape + planned_op extension) -----------


def test_rbc_returns_marker_embeds_full_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)
    call = _deterministic_marker_call(
        context=_rbc_context(_rbc_decision(planned_op=_PROVENANCE)),
        classifier_result=_rbc_classifier(),
        run_id=_RUN_ID,
    )

    assert call.response_json["reservation_annotation"] == {
        "reserved_units": 2,
        "msbos": "none",
        "recommended_units": 0,
        "is_over": True,
        "reason": "over_none",
        "resolved_icd9": "3611",
        "note_resolved": False,
        "reference_hash": "a" * 64,
        "planned_op": _PROVENANCE_DICT,
    }


def test_rbc_returns_marker_without_pick_keeps_legacy_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)
    call = _deterministic_marker_call(
        context=_rbc_context(_rbc_decision(planned_op=None)),
        classifier_result=_rbc_classifier(),
        run_id=_RUN_ID,
    )

    assert call.response_json["reservation_annotation"] == {
        "reserved_units": 2,
        "msbos": "none",
        "recommended_units": 0,
        "is_over": True,
        "reason": "over_none",
        "resolved_icd9": "3611",
        "note_resolved": False,
        "reference_hash": "a" * 64,
    }


def test_platelet_returns_marker_embeds_full_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)
    call = _platelet_marker_call(
        context=_platelet_context(_platelet_decision(planned_op=_PROVENANCE)),
        classifier_result=_platelet_classifier(),
        run_id=_RUN_ID,
    )

    assert call.response_json["reservation_annotation"] == {
        "reserved_units": 2,
        "category": "major_non_neuraxial",
        "pre_op_count_k_ul": 120.0,
        "over_above_per_ul": 80_000,
        "is_over": True,
        "reason": "over_major_non_neuraxial",
        "clinician_signed": True,
        "reference_hash": "b" * 64,
        "planned_op": _PROVENANCE_DICT,
    }


def test_platelet_returns_marker_without_pick_keeps_legacy_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)
    call = _platelet_marker_call(
        context=_platelet_context(_platelet_decision(planned_op=None)),
        classifier_result=_platelet_classifier(),
        run_id=_RUN_ID,
    )

    assert "planned_op" not in call.response_json["reservation_annotation"]


# --- verdict-writing markers --------------------------------------------------


def test_rbc_over_marker_embeds_full_provenance() -> None:
    store = _RecorderStore()
    _persist_over_reservation_row(
        _rbc_context(_rbc_decision(planned_op=_PROVENANCE)),
        classifier_result=_rbc_classifier(),
        audit_store=store,  # type: ignore[arg-type]
        run_id=_RUN_ID,
    )

    assert len(store.calls) == 1
    request = store.calls[0].request_json  # type: ignore[attr-defined]
    assert request == {
        "over_reservation": True,
        "audit_id": "audit-marker-provenance-rbc",
        "reason": "over_none",
        "resolved_icd9": "3611",
        "note_resolved": False,
        "planned_op": _PROVENANCE_DICT,
    }


def test_rbc_marker_carries_ceiling_fields_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A ceiling-judged pick propagates ceiling_token/units/codes into the marker.
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)
    ceiling_provenance = _PROVENANCE.model_copy(
        update={
            "pick_status": "ambiguous_top_rank",
            "gate": "",
            "ceiling_token": "G/M",
            "ceiling_units": 2,
            "ceiling_codes": "8101,8103",
        }
    )
    call = _deterministic_marker_call(
        context=_rbc_context(_rbc_decision(planned_op=ceiling_provenance)),
        classifier_result=_rbc_classifier(),
        run_id=_RUN_ID,
    )

    planned_op = call.response_json["reservation_annotation"]["planned_op"]
    assert planned_op["ceiling_token"] == "G/M"
    assert planned_op["ceiling_units"] == 2
    assert planned_op["ceiling_codes"] == "8101,8103"


def test_rbc_over_marker_without_pick_keeps_legacy_shape() -> None:
    store = _RecorderStore()
    _persist_over_reservation_row(
        _rbc_context(_rbc_decision(planned_op=None)),
        classifier_result=_rbc_classifier(),
        audit_store=store,  # type: ignore[arg-type]
        run_id=_RUN_ID,
    )

    request = store.calls[0].request_json  # type: ignore[attr-defined]
    assert request == {
        "over_reservation": True,
        "audit_id": "audit-marker-provenance-rbc",
        "reason": "over_none",
        "resolved_icd9": "3611",
        "note_resolved": False,
    }


def test_platelet_over_marker_embeds_full_provenance() -> None:
    store = _RecorderStore()
    _persist_platelet_over_reservation_row(
        _platelet_context(_platelet_decision(planned_op=_PROVENANCE)),
        audit_store=store,  # type: ignore[arg-type]
        run_id=_RUN_ID,
    )

    request = store.calls[0].request_json  # type: ignore[attr-defined]
    assert request["planned_op"] == _PROVENANCE_DICT
    assert request["platelet_over_reservation"] is True


def test_bridge_gate_review_markers_embed_full_provenance() -> None:
    rbc_store = _RecorderStore()
    _persist_bridge_gate_review_row(
        _rbc_context(_rbc_decision(planned_op=_PROVENANCE)),
        classifier_result=_rbc_classifier(),
        audit_store=rbc_store,  # type: ignore[arg-type]
        run_id=_RUN_ID,
        review_reason="preop_reservation_bridge_disagreement",
        marker_tag="bridge-disagreement",
        reasoning_en="test",
    )
    plt_store = _RecorderStore()
    _persist_platelet_bridge_gate_review_row(
        _platelet_context(_platelet_decision(planned_op=_PROVENANCE)),
        audit_store=plt_store,  # type: ignore[arg-type]
        run_id=_RUN_ID,
        review_reason="preop_over_reservation_bridge_unconfirmed",
        marker_tag="platelet-bridge-over-unconfirmed",
        reasoning_en="test",
    )

    rbc_request = rbc_store.calls[0].request_json  # type: ignore[attr-defined]
    assert rbc_request["planned_op"] == _PROVENANCE_DICT
    assert rbc_request["bridge_disagreement"] is True
    plt_request = plt_store.calls[0].request_json  # type: ignore[attr-defined]
    assert plt_request["planned_op"] == _PROVENANCE_DICT
    assert plt_request["platelet_bridge_over_unconfirmed"] is True
