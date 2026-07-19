"""Verdict-gate + provenance library tests for ticket #199.

Covers the shared pick->evaluation mapping, the PlannedOpProvenance gate
matrix (disagreement guard, score gate), the decision-model attachment, and
the replay predicates' gate awareness. All fixtures synthetic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

import bba.feature_flags as feature_flags
from bba.audit_pipeline.replay import (
    PLANNED_OP_AMBIGUOUS_REVIEW_REASON,
    PREOP_OVER_RESERVATION_BRIDGE_UNCONFIRMED_REVIEW_REASON,
    PREOP_RESERVATION_BRIDGE_DISAGREEMENT_REVIEW_REASON,
)
from bba.cohort_detector.models import OperativeEvent
from bba.preop_reservation.bridge import OprtactBridge, _bridge_from_rows
from bba.preop_reservation.models import PlannedOpProvenance, ReservationDecision
from bba.preop_reservation.planned_op import (
    AMBIGUOUS_PLANNED_OP_SENTINEL,
    BRIDGE_HARD_VERDICT_MIN_SCORE,
    PlannedOpPick,
    attach_planned_op,
    planned_op_v2_for_events,
)
from bba.preop_reservation.reference import _reference_from_rows

_ORDER_DT = datetime(2026, 3, 1, 8, 0, 0, tzinfo=UTC)
_BRIDGE_HASH = "c" * 64


def _event(icd9: str, *, hours: float = 1.0, seconds: float = 0.0) -> OperativeEvent:
    return OperativeEvent(
        icd9=icd9,
        or_flag=False,
        operative_datetime=_ORDER_DT + timedelta(hours=hours, seconds=seconds),
        name=None,
    )


def _bridge_row(
    oprtact: str,
    *,
    icd9: str = "8151",
    score: str = "0.99",
    human_index: str = "0",
    human_icd9: str = "",
) -> dict[str, str]:
    return {
        "oprtact": oprtact,
        "icd9": icd9,
        "icd9_nodot": icd9.replace(".", ""),
        "score": score,
        "human_index": human_index,
        "human_agreed": "true" if human_index == "0" else "false",
        "human_icd9": human_icd9,
        "name": f"Synthetic op {oprtact}",
    }


def _bridge(rows: list[dict[str, str]]) -> OprtactBridge:
    return _bridge_from_rows(rows, content_hash=_BRIDGE_HASH)


def _reference(codes: list[str]):
    rows = [
        {
            "icd9_code_nodot": code,
            "msbos": "G/M",
            "recommended_units": "2",
            "operation": f"Synthetic operation {code}",
            "procedure_group": "Synthetic group",
        }
        for code in codes
    ]
    return _reference_from_rows(rows, content_hash="d" * 64)


def _pick(
    *,
    source: str | None = "incpt_bridge",
    resolved: str = "8151",
    score: float | None = 0.99,
    human_index: str | None = "0",
    human_agreed: bool | None = True,
    human_icd9: str | None = "",
    pick_status: str = "selected",
) -> PlannedOpPick:
    return PlannedOpPick(
        resolved_icd9=resolved,
        source_code="PX001",
        source=source,  # type: ignore[arg-type]
        bridge_score=score,
        human_index=human_index,
        human_agreed=human_agreed,
        human_icd9=human_icd9,
        or_flag=False,
        matched_datetime=_ORDER_DT + timedelta(hours=1),
        pick_status=pick_status,  # type: ignore[arg-type]
        candidate_count=1,
        tie_count=1,
    )


def _decision(*, is_over: bool = False) -> ReservationDecision:
    return ReservationDecision(
        resolved_icd9="8151",
        reserved_units=4,
        is_over=is_over,
        reason="over_gm_excess" if is_over else "within_recommendation",
        reference_hash="d" * 64,
    )


# --- planned_op_v2_for_events -------------------------------------------------


def test_planned_for_events_selected_returns_resolved_code() -> None:
    bridge = _bridge([_bridge_row("PX001")])

    planned, pick = planned_op_v2_for_events(
        [_event("INCPT:PX001")],
        _ORDER_DT,
        bridge=bridge,
        msbos_codes=frozenset(),
        approved_non_blood_codes=frozenset(),
    )

    assert planned == "8151"
    assert pick.pick_status == "selected"


def test_planned_for_events_ambiguous_maps_to_legacy_sentinel() -> None:
    bridge = _bridge([])
    events = [_event("544"), _event("6561")]

    planned, pick = planned_op_v2_for_events(
        events,
        _ORDER_DT,
        bridge=bridge,
        msbos_codes=frozenset(),
        approved_non_blood_codes=frozenset(),
    )

    assert planned == AMBIGUOUS_PLANNED_OP_SENTINEL
    assert pick.pick_status == "ambiguous_top_rank"


def test_planned_for_events_failure_returns_blank() -> None:
    planned, pick = planned_op_v2_for_events(
        [],
        _ORDER_DT,
        bridge=_bridge([]),
        msbos_codes=frozenset(),
        approved_non_blood_codes=frozenset(),
    )

    assert planned == ""
    assert pick.pick_status == "no_future_event"


# --- attach_planned_op: provenance + gate matrix ------------------------------


def test_attach_carries_full_provenance() -> None:
    reference = _reference(["8151"])
    decision = attach_planned_op(
        _decision(),
        _pick(),
        reference=reference,
        bridge_hash=_BRIDGE_HASH,
    )

    provenance = decision.planned_op
    assert provenance is not None
    assert provenance.source_code == "PX001"
    assert provenance.source == "incpt_bridge"
    assert provenance.bridge_icd9 == "8151"
    assert provenance.bridge_score == 0.99
    assert provenance.human_index == "0"
    assert provenance.human_agreed is True
    assert provenance.human_icd9 == ""
    assert provenance.pick_status == "selected"
    assert provenance.candidate_count == 1
    assert provenance.tie_count == 1
    assert provenance.bridge_hash == _BRIDGE_HASH
    assert provenance.gate == ""


def test_disagreement_gate_fires_when_first_choice_hits_msbos() -> None:
    # P0580 class: First Choice resolves in MSBOS, human picked another code.
    reference = _reference(["8151"])
    decision = attach_planned_op(
        _decision(),
        _pick(human_index="1", human_agreed=False, human_icd9="7935"),
        reference=reference,
        bridge_hash=_BRIDGE_HASH,
    )

    assert decision.planned_op is not None
    assert decision.planned_op.gate == "bridge_disagreement"


def test_disagreement_gate_fires_when_human_code_hits_msbos() -> None:
    # P0752 class: First Choice misses MSBOS, the human-selected code hits.
    reference = _reference(["4573"])
    decision = attach_planned_op(
        _decision(),
        _pick(resolved="1733", human_index="1", human_agreed=False, human_icd9="4573"),
        reference=reference,
        bridge_hash=_BRIDGE_HASH,
    )

    assert decision.planned_op is not None
    assert decision.planned_op.gate == "bridge_disagreement"


def test_disagreement_gate_silent_when_neither_code_hits_msbos() -> None:
    reference = _reference(["9999"])
    decision = attach_planned_op(
        _decision(),
        _pick(resolved="1733", human_index="1", human_agreed=False, human_icd9="4573"),
        reference=reference,
        bridge_hash=_BRIDGE_HASH,
    )

    assert decision.planned_op is not None
    assert decision.planned_op.gate == ""


def test_disagreement_gate_silent_on_agreement_or_no_selection() -> None:
    reference = _reference(["8151"])
    agreed = attach_planned_op(
        _decision(),
        _pick(human_index="0", human_agreed=True, human_icd9=""),
        reference=reference,
        bridge_hash=_BRIDGE_HASH,
    )
    no_selection = attach_planned_op(
        _decision(),
        _pick(human_index="3", human_agreed=False, human_icd9=""),
        reference=reference,
        bridge_hash=_BRIDGE_HASH,
    )

    assert agreed.planned_op is not None and agreed.planned_op.gate == ""
    assert no_selection.planned_op is not None and no_selection.planned_op.gate == ""


def test_over_gate_confirmed_high_score_agreed_stays_hard() -> None:
    reference = _reference(["8151"])
    decision = attach_planned_op(
        _decision(is_over=True),
        _pick(score=BRIDGE_HARD_VERDICT_MIN_SCORE),
        reference=reference,
        bridge_hash=_BRIDGE_HASH,
    )

    assert decision.planned_op is not None
    assert decision.planned_op.gate == ""
    assert decision.is_over is True


@pytest.mark.parametrize(
    ("score", "human_index", "human_agreed"),
    [
        (0.94, "0", True),  # below threshold
        (0.99, "3", False),  # human never confirmed (out-of-range selection)
        (0.94, "4", False),  # both fail
    ],
)
def test_over_gate_unconfirmed_routes_to_review(
    score: float, human_index: str, human_agreed: bool
) -> None:
    reference = _reference(["8151"])
    decision = attach_planned_op(
        _decision(is_over=True),
        _pick(score=score, human_index=human_index, human_agreed=human_agreed),
        reference=reference,
        bridge_hash=_BRIDGE_HASH,
    )

    assert decision.planned_op is not None
    assert decision.planned_op.gate == "bridge_over_unconfirmed"
    # The raw over judgment is preserved for audit; only the verdict is gated.
    assert decision.is_over is True


def test_disagreement_outranks_over_gate() -> None:
    reference = _reference(["8151"])
    decision = attach_planned_op(
        _decision(is_over=True),
        _pick(score=0.5, human_index="1", human_agreed=False, human_icd9="7935"),
        reference=reference,
        bridge_hash=_BRIDGE_HASH,
    )

    assert decision.planned_op is not None
    assert decision.planned_op.gate == "bridge_disagreement"


def test_ambiguous_pick_suppresses_gate() -> None:
    reference = _reference(["8151"])
    decision = attach_planned_op(
        _decision(is_over=True),
        _pick(
            pick_status="ambiguous_top_rank",
            human_index="1",
            human_agreed=False,
            human_icd9="7935",
        ),
        reference=reference,
        bridge_hash=_BRIDGE_HASH,
    )

    assert decision.planned_op is not None
    assert decision.planned_op.gate == ""


def test_exact_icd9_pick_never_gated() -> None:
    reference = _reference(["8151"])
    decision = attach_planned_op(
        _decision(is_over=True),
        _pick(
            source="icd9",
            score=None,
            human_index=None,
            human_agreed=None,
            human_icd9=None,
        ),
        reference=reference,
        bridge_hash=_BRIDGE_HASH,
    )

    assert decision.planned_op is not None
    assert decision.planned_op.gate == ""
    assert decision.is_over is True


# --- decision model extension -------------------------------------------------


def test_reservation_decision_default_has_no_planned_op() -> None:
    assert _decision().planned_op is None


def test_platelet_decision_default_has_no_planned_op() -> None:
    from bba.preop_reservation.platelet_evaluate import PlateletReservationDecision

    decision = PlateletReservationDecision(
        reason="no_planned_op", reference_hash="d" * 64
    )
    assert decision.planned_op is None


# --- replay predicate gate awareness ------------------------------------------


def _provenance(gate: str, *, pick_status: str = "selected") -> PlannedOpProvenance:
    return PlannedOpProvenance(
        source_code="PX001",
        source="incpt_bridge",
        bridge_icd9="8151",
        bridge_score=0.5,
        human_index="1",
        human_agreed=False,
        human_icd9="",
        pick_status=pick_status,
        candidate_count=1,
        tie_count=1,
        bridge_hash=_BRIDGE_HASH,
        gate=gate,
    )


def test_replay_over_reservation_predicate_ignores_gated_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bba.audit_pipeline.replay import (
        is_bridge_disagreement_review,
        is_bridge_over_unconfirmed_review,
        is_over_reservation,
        is_planned_op_ambiguous_review,
    )
    from bba.deterministic_classifier import ClassifierResult
    from bba.deterministic_classifier.models import BypassReason

    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)
    cres = ClassifierResult(
        classification="PERIOP_TRANSFUSION_EXEMPT",
        rationale="preop_declared_exempt",
        cohort_threshold=None,
        bypass_reason=BypassReason.NONE,
    )

    class _Ctx:
        reservation_decision = _decision(is_over=True).model_copy(
            update={"planned_op": _provenance("bridge_over_unconfirmed")}
        )

    class _CtxDisagreement:
        reservation_decision = _decision(is_over=True).model_copy(
            update={"planned_op": _provenance("bridge_disagreement")}
        )

    class _CtxAmbiguous:
        reservation_decision = ReservationDecision(
            reserved_units=2,
            is_over=False,
            reason="ambiguous_planned_op",
            reference_hash="d" * 64,
        ).model_copy(
            update={"planned_op": _provenance("", pick_status="ambiguous_top_rank")}
        )

    class _CtxHard:
        reservation_decision = _decision(is_over=True)

    assert is_over_reservation(classifier_result=cres, context=_Ctx) is False  # type: ignore[arg-type]
    assert (
        is_bridge_over_unconfirmed_review(classifier_result=cres, context=_Ctx)  # type: ignore[arg-type]
        is True
    )
    assert (
        is_over_reservation(classifier_result=cres, context=_CtxDisagreement)  # type: ignore[arg-type]
        is False
    )
    assert (
        is_bridge_disagreement_review(
            classifier_result=cres,
            context=_CtxDisagreement,  # type: ignore[arg-type]
        )
        is True
    )
    assert (
        is_planned_op_ambiguous_review(
            classifier_result=cres,
            context=_CtxAmbiguous,  # type: ignore[arg-type]
        )
        is True
    )
    assert is_over_reservation(classifier_result=cres, context=_CtxHard) is True  # type: ignore[arg-type]


def test_replay_new_predicates_false_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bba.audit_pipeline.replay import (
        is_bridge_disagreement_review,
        is_bridge_over_unconfirmed_review,
        is_planned_op_ambiguous_review,
    )
    from bba.deterministic_classifier import ClassifierResult
    from bba.deterministic_classifier.models import BypassReason

    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", False)
    cres = ClassifierResult(
        classification="PERIOP_TRANSFUSION_EXEMPT",
        rationale="preop_declared_exempt",
        cohort_threshold=None,
        bypass_reason=BypassReason.NONE,
    )

    class _Ctx:
        reservation_decision = _decision(is_over=True).model_copy(
            update={"planned_op": _provenance("bridge_over_unconfirmed")}
        )

    assert (
        is_bridge_over_unconfirmed_review(classifier_result=cres, context=_Ctx)  # type: ignore[arg-type]
        is False
    )
    assert (
        is_bridge_disagreement_review(classifier_result=cres, context=_Ctx)  # type: ignore[arg-type]
        is False
    )
    assert (
        is_planned_op_ambiguous_review(classifier_result=cres, context=_Ctx)  # type: ignore[arg-type]
        is False
    )


def test_review_reason_constants_are_frozen_strings() -> None:
    assert (
        PREOP_RESERVATION_BRIDGE_DISAGREEMENT_REVIEW_REASON
        == "preop_reservation_bridge_disagreement"
    )
    assert (
        PREOP_OVER_RESERVATION_BRIDGE_UNCONFIRMED_REVIEW_REASON
        == "preop_over_reservation_bridge_unconfirmed"
    )
    assert PLANNED_OP_AMBIGUOUS_REVIEW_REASON == "ambiguous_planned_op"


def test_is_planned_op_ambiguous_review_rekeyed_on_reason_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #210/#213: the predicate keys on the decision REASON (+ provenance
    # presence), not the pick_status, so a ceiling-judged row routes as an over
    # and a legacy picker-off \x00AMBIG row (no provenance) stays out.
    from bba.audit_pipeline.replay import is_planned_op_ambiguous_review
    from bba.deterministic_classifier import ClassifierResult
    from bba.deterministic_classifier.models import BypassReason

    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)
    cres = ClassifierResult(
        classification="PERIOP_TRANSFUSION_EXEMPT",
        rationale="preop_declared_exempt",
        cohort_threshold=None,
        bypass_reason=BypassReason.NONE,
    )

    class _CtxCeiling:
        # Keeps its ambiguous_top_rank pick, but reason is now over_ceiling.
        reservation_decision = ReservationDecision(
            reserved_units=5,
            is_over=True,
            reason="over_ceiling",
            reference_hash="d" * 64,
        ).model_copy(
            update={"planned_op": _provenance("", pick_status="ambiguous_top_rank")}
        )

    class _CtxLegacyAmbig:
        # Picker-off \x00AMBIG: ambiguous reason but no provenance.
        reservation_decision = ReservationDecision(
            reserved_units=2,
            is_over=False,
            reason="ambiguous_planned_op",
            reference_hash="d" * 64,
        )

    class _CtxAmbiguous:
        reservation_decision = ReservationDecision(
            reserved_units=2,
            is_over=False,
            reason="ambiguous_planned_op",
            reference_hash="d" * 64,
        ).model_copy(
            update={"planned_op": _provenance("", pick_status="ambiguous_top_rank")}
        )

    assert (
        is_planned_op_ambiguous_review(classifier_result=cres, context=_CtxCeiling)  # type: ignore[arg-type]
        is False
    )
    assert (
        is_planned_op_ambiguous_review(
            classifier_result=cres,
            context=_CtxLegacyAmbig,  # type: ignore[arg-type]
        )
        is False
    )
    assert (
        is_planned_op_ambiguous_review(
            classifier_result=cres,
            context=_CtxAmbiguous,  # type: ignore[arg-type]
        )
        is True
    )


def test_is_all_candidates_excluded_review_matches_det_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # #210: the LLM leg must route all_candidates_excluded to review like the det
    # overlay (twin parity). The pick leaves a no_planned_op decision, so only
    # the all_candidates_excluded provenance distinguishes it from a plain
    # no-plan row.
    from bba.audit_pipeline.replay import is_all_candidates_excluded_review
    from bba.deterministic_classifier import ClassifierResult
    from bba.deterministic_classifier.models import BypassReason

    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", True)
    cres = ClassifierResult(
        classification="PERIOP_TRANSFUSION_EXEMPT",
        rationale="preop_declared_exempt",
        cohort_threshold=None,
        bypass_reason=BypassReason.NONE,
    )

    class _CtxExcluded:
        reservation_decision = ReservationDecision(
            reserved_units=2,
            is_over=False,
            reason="no_planned_op",
            reference_hash="d" * 64,
        ).model_copy(
            update={
                "planned_op": _provenance("", pick_status="all_candidates_excluded")
            }
        )

    class _CtxPlainNoPlan:
        # no_planned_op WITHOUT the all_candidates_excluded provenance (e.g. a
        # legacy picker-off row) must NOT route to review.
        reservation_decision = ReservationDecision(
            reserved_units=2,
            is_over=False,
            reason="no_planned_op",
            reference_hash="d" * 64,
        )

    assert (
        is_all_candidates_excluded_review(
            classifier_result=cres,
            context=_CtxExcluded,  # type: ignore[arg-type]
        )
        is True
    )
    assert (
        is_all_candidates_excluded_review(
            classifier_result=cres,
            context=_CtxPlainNoPlan,  # type: ignore[arg-type]
        )
        is False
    )
    monkeypatch.setattr(feature_flags, "MSBOS_RESERVATION_ENABLED", False)
    assert (
        is_all_candidates_excluded_review(
            classifier_result=cres,
            context=_CtxExcluded,  # type: ignore[arg-type]
        )
        is False
    )
