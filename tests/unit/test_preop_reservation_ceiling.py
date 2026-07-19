"""Dominance-ceiling library tests for ticket #212 (spec #210).

Covers ``rows_for``/``ceiling_row``, the ``finalize_planned_op`` seam (cluster
and single-code ceilings), the dominance validity precondition, the per-member
verdict gate, and the platelet no-op. All fixtures synthetic; the picker is
pure and every branch is driven directly through the library.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from bba.preop_reservation.evaluate import evaluate_reservation
from bba.preop_reservation.models import MsbosRow, ReservationDecision
from bba.preop_reservation.planned_op import (
    BRIDGE_HARD_VERDICT_MIN_SCORE,
    PlannedOpClusterMember,
    PlannedOpPick,
    ceiling_row,
    finalize_planned_op,
)
from bba.preop_reservation.platelet_evaluate import PlateletReservationDecision
from bba.preop_reservation.reference import _reference_from_rows

_ORDER_DT = datetime(2026, 3, 1, 8, 0, 0, tzinfo=UTC)
_BRIDGE_HASH = "c" * 64
_REF_HASH = "d" * 64


def _reference(rows: list[tuple[str, str, int]]):
    return _reference_from_rows(
        [
            {
                "icd9_code_nodot": code,
                "msbos": token,
                "recommended_units": str(units),
                "operation": f"op {code}",
                "procedure_group": "grp",
            }
            for code, token, units in rows
        ],
        content_hash=_REF_HASH,
    )


def _member(
    resolved: str,
    *,
    source: str = "icd9",
    score: float | None = None,
    human_agreed: bool | None = None,
    human_icd9: str | None = "",
) -> PlannedOpClusterMember:
    return PlannedOpClusterMember(
        resolved_icd9=resolved,
        source=source,  # type: ignore[arg-type]
        bridge_score=score,
        human_agreed=human_agreed,
        human_icd9=human_icd9,
    )


def _cluster_pick(
    *,
    codes: tuple[str, ...],
    members: tuple[PlannedOpClusterMember, ...],
    all_eligible: bool = True,
    resolved: str | None = None,
) -> PlannedOpPick:
    return PlannedOpPick(
        resolved_icd9=resolved if resolved is not None else codes[0],
        source_code="PX001",
        source="icd9",
        bridge_score=None,
        human_index=None,
        human_agreed=None,
        human_icd9=None,
        or_flag=False,
        matched_datetime=_ORDER_DT + timedelta(hours=1),
        pick_status="ambiguous_top_rank",
        candidate_count=len(members),
        tie_count=len(members),
        cluster_codes=codes,
        cluster_all_eligible=all_eligible,
        cluster_members=members,
    )


def _selected_pick(
    *,
    resolved: str,
    source: str = "icd9",
    score: float | None = None,
    human_agreed: bool | None = None,
    human_icd9: str | None = "",
) -> PlannedOpPick:
    return PlannedOpPick(
        resolved_icd9=resolved,
        source_code="PX001",
        source=source,  # type: ignore[arg-type]
        bridge_score=score,
        human_index="0" if human_agreed else "1",
        human_agreed=human_agreed,
        human_icd9=human_icd9,
        or_flag=False,
        matched_datetime=_ORDER_DT + timedelta(hours=1),
        pick_status="selected",
        candidate_count=1,
        tie_count=1,
    )


def _ambig_decision(reserved: int) -> ReservationDecision:
    return ReservationDecision(
        reserved_units=reserved,
        is_over=False,
        reason="ambiguous_planned_op",
        reference_hash=_REF_HASH,
    )


# --- rows_for ----------------------------------------------------------------


def test_rows_for_returns_all_recommendations_for_a_code() -> None:
    reference = _reference([("1234", "G/M", 2), ("1234", "T/S", 0)])

    rows = reference.rows_for("1234")

    assert rows == frozenset(
        {
            MsbosRow(msbos="G/M", recommended_units=2),
            MsbosRow(msbos="T/S", recommended_units=0),
        }
    )


def test_rows_for_absent_code_is_empty() -> None:
    reference = _reference([("1234", "G/M", 2)])

    assert reference.rows_for("9999") == frozenset()


# --- ceiling_row -------------------------------------------------------------


def test_ceiling_row_prefers_gm_max_units() -> None:
    rows = frozenset(
        {
            MsbosRow(msbos="G/M", recommended_units=2),
            MsbosRow(msbos="G/M", recommended_units=4),
            MsbosRow(msbos="T/S", recommended_units=0),
        }
    )

    assert ceiling_row(rows) == MsbosRow(msbos="G/M", recommended_units=4)


def test_ceiling_row_falls_back_to_ts_when_no_gm() -> None:
    rows = frozenset(
        {
            MsbosRow(msbos="T/S", recommended_units=0),
            MsbosRow(msbos="none", recommended_units=0),
        }
    )

    assert ceiling_row(rows) == MsbosRow(msbos="T/S", recommended_units=0)


def test_ceiling_row_falls_back_to_none() -> None:
    rows = frozenset({MsbosRow(msbos="none", recommended_units=0)})

    assert ceiling_row(rows) == MsbosRow(msbos="none", recommended_units=0)


# --- finalize_planned_op: cluster ceiling ------------------------------------


def test_cluster_over_ceiling_exact_members_keeps_hard_over() -> None:
    # Reserved 5 exceeds the most permissive G/M 2 -> over under every reading.
    reference = _reference([("1111", "G/M", 2), ("2222", "T/S", 0)])
    pick = _cluster_pick(
        codes=("1111", "2222"),
        members=(_member("1111"), _member("2222")),
    )

    decision = finalize_planned_op(
        _ambig_decision(5), pick, reference=reference, bridge_hash=_BRIDGE_HASH
    )

    assert decision.reason == "over_ceiling"
    assert decision.is_over is True
    assert decision.msbos == "G/M"
    assert decision.recommended_units == 2
    assert decision.planned_op is not None
    assert decision.planned_op.gate == ""
    assert decision.planned_op.ceiling_token == "G/M"
    assert decision.planned_op.ceiling_units == 2
    assert decision.planned_op.ceiling_codes == "1111,2222"


def test_cluster_within_ceiling_stays_exempt_annotated() -> None:
    # Reserved 2 is within the most permissive G/M 2 (shadow-over vs the T/S
    # member, but within under the ceiling reading).
    reference = _reference([("1111", "G/M", 2), ("2222", "T/S", 0)])
    pick = _cluster_pick(
        codes=("1111", "2222"),
        members=(_member("1111"), _member("2222")),
    )

    decision = finalize_planned_op(
        _ambig_decision(2), pick, reference=reference, bridge_hash=_BRIDGE_HASH
    )

    assert decision.reason == "within_ceiling"
    assert decision.is_over is False
    assert decision.planned_op is not None
    assert decision.planned_op.ceiling_token == "G/M"
    assert decision.planned_op.ceiling_units == 2


def test_cluster_all_ts_ceiling_is_over_when_reserved() -> None:
    reference = _reference([("1111", "T/S", 0), ("2222", "T/S", 0)])
    pick = _cluster_pick(
        codes=("1111", "2222"),
        members=(_member("1111"), _member("2222")),
    )

    decision = finalize_planned_op(
        _ambig_decision(1), pick, reference=reference, bridge_hash=_BRIDGE_HASH
    )

    assert decision.reason == "over_ceiling"
    assert decision.is_over is True
    assert decision.msbos == "T/S"


# --- dominance validity precondition -----------------------------------------


def test_dominance_voided_when_a_member_not_eligible_keeps_ambiguous() -> None:
    reference = _reference([("1111", "G/M", 2), ("2222", "T/S", 0)])
    pick = _cluster_pick(
        codes=("1111", "2222", "9999"),
        members=(_member("1111"), _member("2222"), _member("9999")),
        all_eligible=False,  # 9999 is not in MSBOS
    )

    decision = finalize_planned_op(
        _ambig_decision(9), pick, reference=reference, bridge_hash=_BRIDGE_HASH
    )

    assert decision.reason == "ambiguous_planned_op"
    assert decision.is_over is False
    assert decision.planned_op is not None
    assert decision.planned_op.ceiling_token == ""


def test_single_distinct_code_cluster_does_not_apply_ceiling() -> None:
    reference = _reference([("1111", "G/M", 2)])
    pick = _cluster_pick(codes=("1111",), members=(_member("1111"),))

    decision = finalize_planned_op(
        _ambig_decision(9), pick, reference=reference, bridge_hash=_BRIDGE_HASH
    )

    assert decision.reason == "ambiguous_planned_op"


# --- single-code ambiguous_code ceiling --------------------------------------


def test_single_code_ambiguous_over_ceiling() -> None:
    reference = _reference([("1234", "G/M", 2), ("1234", "T/S", 0)])
    decision_in = evaluate_reservation(
        reserved_units=5, planned_icd9_nodot="1234", reference=reference
    )
    assert decision_in.reason == "ambiguous_code"
    pick = _selected_pick(resolved="1234")

    decision = finalize_planned_op(
        decision_in, pick, reference=reference, bridge_hash=_BRIDGE_HASH
    )

    assert decision.reason == "over_ceiling"
    assert decision.is_over is True
    assert decision.planned_op is not None
    assert decision.planned_op.ceiling_codes == "1234"


def test_single_code_ambiguous_within_ceiling() -> None:
    reference = _reference([("1234", "G/M", 2), ("1234", "T/S", 0)])
    decision_in = evaluate_reservation(
        reserved_units=2, planned_icd9_nodot="1234", reference=reference
    )
    pick = _selected_pick(resolved="1234")

    decision = finalize_planned_op(
        decision_in, pick, reference=reference, bridge_hash=_BRIDGE_HASH
    )

    assert decision.reason == "within_ceiling"
    assert decision.is_over is False


# --- per-member verdict gate on a ceiling over -------------------------------


def test_cluster_over_gate_unconfirmed_bridge_member_routes_to_review() -> None:
    reference = _reference([("1111", "G/M", 2), ("2222", "T/S", 0)])
    pick = _cluster_pick(
        codes=("1111", "2222"),
        members=(
            _member("1111"),
            _member(
                "2222", source="incpt_bridge", score=0.80, human_agreed=True
            ),  # below threshold
        ),
    )

    decision = finalize_planned_op(
        _ambig_decision(5), pick, reference=reference, bridge_hash=_BRIDGE_HASH
    )

    assert decision.reason == "over_ceiling"
    assert decision.is_over is True  # raw over preserved for audit
    assert decision.planned_op is not None
    assert decision.planned_op.gate == "bridge_over_unconfirmed"


def test_cluster_over_gate_confirmed_bridge_member_keeps_hard() -> None:
    reference = _reference([("1111", "G/M", 2), ("2222", "T/S", 0)])
    pick = _cluster_pick(
        codes=("1111", "2222"),
        members=(
            _member("1111"),
            _member(
                "2222",
                source="incpt_bridge",
                score=BRIDGE_HARD_VERDICT_MIN_SCORE,
                human_agreed=True,
            ),
        ),
    )

    decision = finalize_planned_op(
        _ambig_decision(5), pick, reference=reference, bridge_hash=_BRIDGE_HASH
    )

    assert decision.reason == "over_ceiling"
    assert decision.planned_op is not None
    assert decision.planned_op.gate == ""


def test_cluster_gate_disagreement_outranks_over_gate() -> None:
    # A bridge member whose First-Choice and human codes differ and one hits MSBOS.
    reference = _reference([("1111", "G/M", 2), ("2222", "T/S", 0)])
    pick = _cluster_pick(
        codes=("1111", "2222"),
        members=(
            _member("1111"),
            _member(
                "2222",
                source="incpt_bridge",
                score=0.50,
                human_agreed=False,
                human_icd9="1111",  # differs from resolved 2222, hits MSBOS
            ),
        ),
    )

    decision = finalize_planned_op(
        _ambig_decision(5), pick, reference=reference, bridge_hash=_BRIDGE_HASH
    )

    assert decision.reason == "over_ceiling"
    assert decision.planned_op is not None
    assert decision.planned_op.gate == "bridge_disagreement"


def test_within_ceiling_never_gated_even_with_unconfirmed_member() -> None:
    # within_ceiling is not an over, so the over gate never fires; it stays exempt.
    reference = _reference([("1111", "G/M", 2), ("2222", "T/S", 0)])
    pick = _cluster_pick(
        codes=("1111", "2222"),
        members=(
            _member("1111"),
            _member("2222", source="incpt_bridge", score=0.10, human_agreed=False),
        ),
    )

    decision = finalize_planned_op(
        _ambig_decision(2), pick, reference=reference, bridge_hash=_BRIDGE_HASH
    )

    assert decision.reason == "within_ceiling"
    assert decision.planned_op is not None
    assert decision.planned_op.gate == ""


# --- non-ceiling passthrough + platelet no-op --------------------------------


def test_non_ceiling_decision_falls_through_to_attach() -> None:
    reference = _reference([("8151", "G/M", 2)])
    decision_in = evaluate_reservation(
        reserved_units=1, planned_icd9_nodot="8151", reference=reference
    )
    assert decision_in.reason == "within_recommendation"
    pick = _selected_pick(resolved="8151")

    decision = finalize_planned_op(
        decision_in, pick, reference=reference, bridge_hash=_BRIDGE_HASH
    )

    assert decision.reason == "within_recommendation"
    assert decision.planned_op is not None
    assert decision.planned_op.ceiling_token == ""
    assert decision.planned_op.source_code == "PX001"


def test_platelet_ambiguous_pick_gets_no_ceiling() -> None:
    reference = _reference([("1111", "G/M", 2), ("2222", "T/S", 0)])
    platelet_in = PlateletReservationDecision(
        reserved_units=5, reason="ambiguous_planned_op", reference_hash=_REF_HASH
    )
    pick = _cluster_pick(
        codes=("1111", "2222"),
        members=(_member("1111"), _member("2222")),
    )

    decision = finalize_planned_op(
        platelet_in, pick, reference=reference, bridge_hash=_BRIDGE_HASH
    )

    assert decision.reason == "ambiguous_planned_op"
    assert decision.is_over is False
    assert decision.planned_op is not None
    assert decision.planned_op.ceiling_token == ""
