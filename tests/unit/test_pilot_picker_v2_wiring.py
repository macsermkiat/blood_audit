"""Pilot wiring tests for the planned-op picker v2 seam (ticket #199).

Covers the 2x2 flag/schema matrix, the +opbound2 CODE_VERSION token, the
declared-row verdict overlays (RBC + platelet), and the det-leg MSBOS helpers
running with a synthetic bridge. Import-level only: never calls
run_llm_leg.main() and never touches the network.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

from bba.cohort_detector.models import OperativeEvent
from bba.component_map import ComponentFamily
from bba.preop_reservation.bridge import _bridge_from_rows
from bba.preop_reservation.reference import _reference_from_rows

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"

_ORDER_DT = datetime(2026, 3, 1, 8, 0, 0, tzinfo=UTC)


def _load_pilot_module(filename: str, module_name: str) -> ModuleType:
    if str(PILOT_DIR) not in sys.path:
        sys.path.insert(0, str(PILOT_DIR))
    spec = importlib.util.spec_from_file_location(module_name, PILOT_DIR / filename)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _reference():
    rows = [
        {
            "icd9_code_nodot": "8151",
            "msbos": "G/M",
            "recommended_units": "2",
            "operation": "Synthetic hip replacement",
            "procedure_group": "Synthetic group",
        },
        {
            "icd9_code_nodot": "4573",
            "msbos": "T/S",
            "recommended_units": "0",
            "operation": "Synthetic open procedure",
            "procedure_group": "Synthetic group",
        },
    ]
    return _reference_from_rows(rows, content_hash="e" * 64)


def _bridge(
    *,
    icd9: str = "8151",
    score: str = "0.99",
    human_index: str = "0",
    human_icd9: str = "",
) -> object:
    return _bridge_from_rows(
        [
            {
                "oprtact": "PX001",
                "icd9": icd9,
                "icd9_nodot": icd9.replace(".", ""),
                "score": score,
                "human_index": human_index,
                "human_agreed": "true" if human_index == "0" else "false",
                "human_icd9": human_icd9,
                "name": "Synthetic op PX001",
            }
        ],
        content_hash="f" * 64,
    )


def _event(icd9: str, *, hours: float = 1.0) -> OperativeEvent:
    return OperativeEvent(
        icd9=icd9,
        or_flag=False,
        operative_datetime=_ORDER_DT + timedelta(hours=hours),
        name=None,
    )


# --- 2x2 flag / schema matrix -------------------------------------------------


@pytest.mark.parametrize(
    ("msbos_env", "picker_env", "expect_columns"),
    [
        ("1", "1", True),
        ("1", "0", False),
        ("0", "1", False),
        ("0", "0", False),
    ],
)
def test_report_fieldnames_picker_columns_gated_on_both_seams(
    monkeypatch: pytest.MonkeyPatch,
    msbos_env: str,
    picker_env: str,
    expect_columns: bool,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", msbos_env)
    monkeypatch.setenv("BBA_PILOT_MSBOS_PLANNED_OP_PICKER_V2", picker_env)
    module = _load_pilot_module(
        "run_pipeline.py", f"pilot_run_pipeline_p2_{msbos_env}{picker_env}"
    )

    fieldnames = module._report_fieldnames()

    for column in module.MSBOS_PICKER_V2_FIELDNAMES:
        assert (column in fieldnames) is expect_columns
    assert "_bridge_gate" not in fieldnames


@pytest.mark.parametrize(
    ("msbos_env", "picker_env", "expect_token"),
    [
        ("1", "1", True),
        ("1", "0", False),
        ("0", "1", False),
        ("0", "0", False),
    ],
)
def test_llm_leg_opbound_token_gated_on_both_seams(
    monkeypatch: pytest.MonkeyPatch,
    msbos_env: str,
    picker_env: str,
    expect_token: bool,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", msbos_env)
    monkeypatch.setenv("BBA_PILOT_MSBOS_PLANNED_OP_PICKER_V2", picker_env)
    module = _load_pilot_module(
        "run_llm_leg.py", f"pilot_run_llm_leg_p2_{msbos_env}{picker_env}"
    )

    # Exact-suffix pin (#210): the token is +opbound2, NOT +opbound. Checking
    # for "+opbound2" fails on a stale "+opbound" (which is not a substring of
    # the surrounding "...+opbound+..."), unlike the old loose substring check.
    assert ("+opbound2" in module.CODE_VERSION) is expect_token
    if msbos_env == "1" and picker_env == "0":
        assert "+msbos5" in module.CODE_VERSION
        assert "+opbound" not in module.CODE_VERSION
    if expect_token:
        assert "+msbos5+opbound2" in module.CODE_VERSION


def test_picker_default_is_on_with_env_escape_hatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Go-live 2026-07-19: the picker seam defaults ON; the env override
    # remains the operator escape hatch back to the legacy picker.
    monkeypatch.delenv("BBA_PILOT_MSBOS_PLANNED_OP_PICKER_V2", raising=False)
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "1")
    module = _load_pilot_module("run_pipeline.py", "pilot_run_pipeline_p2_default")

    assert module.MSBOS_PLANNED_OP_PICKER_V2_PILOT_ENABLED is True
    assert all(
        c in module._report_fieldnames() for c in module.MSBOS_PICKER_V2_FIELDNAMES
    )

    monkeypatch.setenv("BBA_PILOT_MSBOS_PLANNED_OP_PICKER_V2", "0")
    forced_off = _load_pilot_module(
        "run_pipeline.py", "pilot_run_pipeline_p2_forced_off"
    )
    assert forced_off.MSBOS_PLANNED_OP_PICKER_V2_PILOT_ENABLED is False
    assert not any(
        c in forced_off._report_fieldnames()
        for c in forced_off.MSBOS_PICKER_V2_FIELDNAMES
    )


# --- declared-row verdict overlays --------------------------------------------


@pytest.fixture(scope="module")
def pipeline() -> ModuleType:
    return _load_pilot_module("run_pipeline.py", "pilot_run_pipeline_p2_overlay")


def test_rbc_overlay_precedence_matrix(pipeline: ModuleType) -> None:
    overlay = pipeline._declared_msbos_overlay

    # Ambiguity (reason + provenance present) outranks everything.
    assert overlay(
        {
            "msbos_reason": "ambiguous_planned_op",
            "msbos_op_pick_status": "ambiguous_top_rank",
            "_bridge_gate": "bridge_disagreement",
            "msbos_is_over": True,
        }
    ) == ("NEEDS_REVIEW", "ambiguous_planned_op")
    # Legacy \x00AMBIG row under picker-OFF: reason set but NO provenance ->
    # stays declared-exempt (forced-off parity).
    assert overlay({"msbos_reason": "ambiguous_planned_op"}) is None
    # All-candidates-excluded declared surgery with reserved units -> review; a
    # zero-unit (screen-only) reservation has nothing to judge -> stays exempt.
    assert overlay(
        {"msbos_op_pick_status": "all_candidates_excluded", "msbos_reserved_units": 1}
    ) == ("NEEDS_REVIEW", "all_candidates_excluded")
    assert (
        overlay(
            {
                "msbos_op_pick_status": "all_candidates_excluded",
                "msbos_reserved_units": 0,
            }
        )
        is None
    )
    # A ceiling over is no longer "ambiguous_planned_op"; it routes as an over
    # even though the pick_status stays ambiguous_top_rank.
    assert overlay(
        {
            "msbos_reason": "over_ceiling",
            "msbos_op_pick_status": "ambiguous_top_rank",
            "msbos_is_over": True,
        }
    ) == ("PREOP_OVER_RESERVATION", "preop_over_reservation")
    # A gated ceiling over -> review.
    assert overlay(
        {
            "msbos_reason": "over_ceiling",
            "msbos_op_pick_status": "ambiguous_top_rank",
            "_bridge_gate": "bridge_over_unconfirmed",
            "msbos_is_over": True,
        }
    ) == ("NEEDS_REVIEW", "preop_over_reservation_bridge_unconfirmed")
    # within_ceiling stays declared-exempt (annotated).
    assert (
        overlay(
            {
                "msbos_reason": "within_ceiling",
                "msbos_op_pick_status": "ambiguous_top_rank",
                "msbos_is_over": False,
            }
        )
        is None
    )
    # Disagreement fires before the score gate.
    assert overlay({"_bridge_gate": "bridge_disagreement", "msbos_is_over": True}) == (
        "NEEDS_REVIEW",
        "preop_reservation_bridge_disagreement",
    )
    # Gated over -> review; ungated over -> hard.
    assert overlay(
        {"_bridge_gate": "bridge_over_unconfirmed", "msbos_is_over": True}
    ) == (
        "NEEDS_REVIEW",
        "preop_over_reservation_bridge_unconfirmed",
    )
    assert overlay({"msbos_is_over": True}) == (
        "PREOP_OVER_RESERVATION",
        "preop_over_reservation",
    )
    # Legacy branches unchanged.
    assert overlay({"msbos_reason": "operation_unresolved"}) == (
        "NEEDS_REVIEW",
        "operation_unresolved",
    )
    assert overlay({"msbos_reason": "unresolved_code"}) is None


def test_platelet_overlay_precedence_matrix(pipeline: ModuleType) -> None:
    overlay = pipeline._declared_platelet_overlay

    assert overlay({"_bridge_gate": "bridge_disagreement", "msbos_is_over": True}) == (
        "NEEDS_REVIEW",
        "preop_reservation_bridge_disagreement",
    )
    assert overlay(
        {"_bridge_gate": "bridge_over_unconfirmed", "msbos_is_over": True}
    ) == (
        "NEEDS_REVIEW",
        "preop_over_reservation_bridge_unconfirmed",
    )
    assert overlay({"msbos_is_over": True}) == (
        "PREOP_OVER_RESERVATION",
        "preop_over_reservation",
    )
    # Pick ambiguity flows through the platelet review reasons.
    assert overlay({"msbos_reason": "ambiguous_planned_op"}) == (
        "NEEDS_REVIEW",
        "platelet_reservation_review",
    )
    assert overlay({"msbos_reason": "within_major_non_neuraxial"}) is None


# --- det helper with a synthetic bridge ---------------------------------------


def test_rbc_columns_with_bridge_carry_provenance_and_gate(
    pipeline: ModuleType,
) -> None:
    reference = _reference()
    bridge = _bridge(icd9="1733", human_index="1", human_icd9="4573")

    columns = pipeline._msbos_reservation_columns(
        classification="PERIOP_TRANSFUSION_EXEMPT",
        hn="HN1",
        reqno="REQ1",
        op_events=[_event("INCPT:PX001")],
        order_datetime=_ORDER_DT,
        reserved_units_map={("HN1", "REQ1", ComponentFamily.RED_CELL): 1},
        msbos_reference=reference,
        oprtact_bridge=bridge,
    )

    # P0752 class: First (1733) misses MSBOS, human (4573) hits -> the row
    # would silently stay exempt as unresolved_code without the guard.
    assert columns["msbos_reason"] == "unresolved_code"
    assert columns["msbos_source_code"] == "PX001"
    assert columns["msbos_bridge_icd9"] == "1733"
    assert columns["msbos_bridge_human_index"] == "1"
    assert columns["msbos_bridge_human_agreed"] is False
    assert columns["msbos_op_pick_status"] == "selected"
    assert columns["msbos_candidate_count"] == 1
    assert columns["msbos_tie_count"] == 1
    assert columns["msbos_bridge_hash"] == "f" * 64
    assert columns["_bridge_gate"] == "bridge_disagreement"
    assert pipeline._declared_msbos_overlay(columns) == (
        "NEEDS_REVIEW",
        "preop_reservation_bridge_disagreement",
    )


def test_rbc_columns_confirmed_bridge_over_stays_hard(pipeline: ModuleType) -> None:
    reference = _reference()
    bridge = _bridge(icd9="8151", score="0.99", human_index="0")

    columns = pipeline._msbos_reservation_columns(
        classification="PERIOP_TRANSFUSION_EXEMPT",
        hn="HN1",
        reqno="REQ1",
        op_events=[_event("INCPT:PX001")],
        order_datetime=_ORDER_DT,
        reserved_units_map={("HN1", "REQ1", ComponentFamily.RED_CELL): 5},
        msbos_reference=reference,
        oprtact_bridge=bridge,
    )

    assert columns["msbos_reason"] == "over_gm_excess"
    assert columns["msbos_is_over"] is True
    assert columns["_bridge_gate"] == ""
    assert pipeline._declared_msbos_overlay(columns) == (
        "PREOP_OVER_RESERVATION",
        "preop_over_reservation",
    )


def test_rbc_columns_unconfirmed_bridge_over_reviews(pipeline: ModuleType) -> None:
    reference = _reference()
    bridge = _bridge(icd9="8151", score="0.80", human_index="0")

    columns = pipeline._msbos_reservation_columns(
        classification="PERIOP_TRANSFUSION_EXEMPT",
        hn="HN1",
        reqno="REQ1",
        op_events=[_event("INCPT:PX001")],
        order_datetime=_ORDER_DT,
        reserved_units_map={("HN1", "REQ1", ComponentFamily.RED_CELL): 5},
        msbos_reference=reference,
        oprtact_bridge=bridge,
    )

    assert columns["msbos_is_over"] is True
    assert columns["_bridge_gate"] == "bridge_over_unconfirmed"
    assert pipeline._declared_msbos_overlay(columns) == (
        "NEEDS_REVIEW",
        "preop_over_reservation_bridge_unconfirmed",
    )


def test_rbc_columns_without_bridge_have_no_picker_columns(
    pipeline: ModuleType,
) -> None:
    reference = _reference()

    columns = pipeline._msbos_reservation_columns(
        classification="PERIOP_TRANSFUSION_EXEMPT",
        hn="HN1",
        reqno="REQ1",
        op_events=[_event("8151")],
        order_datetime=_ORDER_DT,
        reserved_units_map={("HN1", "REQ1", ComponentFamily.RED_CELL): 1},
        msbos_reference=reference,
        oprtact_bridge=None,
    )

    assert "msbos_op_pick_status" not in columns
    assert "_bridge_gate" not in columns
    assert columns["msbos_reason"] == "within_recommendation"


# --- dominance ceiling routing through the det helper (#210/#213) --------------


def _ambiguous_code_reference():
    rows = [
        {
            "icd9_code_nodot": "1234",
            "msbos": "G/M",
            "recommended_units": "2",
            "operation": "Synthetic op a",
            "procedure_group": "Synthetic group",
        },
        {
            "icd9_code_nodot": "1234",
            "msbos": "T/S",
            "recommended_units": "0",
            "operation": "Synthetic op b",
            "procedure_group": "Synthetic group",
        },
    ]
    return _reference_from_rows(rows, content_hash="e" * 64)


def test_rbc_cluster_over_ceiling_routes_hard(pipeline: ModuleType) -> None:
    # Two same-time distinct MSBOS ops (8151 G/M 2, 4573 T/S 0) -> ambiguity set;
    # reserved 5 exceeds the G/M 2 ceiling -> hard over_ceiling.
    reference = _reference()

    columns = pipeline._msbos_reservation_columns(
        classification="PERIOP_TRANSFUSION_EXEMPT",
        hn="HN1",
        reqno="REQ1",
        op_events=[_event("8151"), _event("4573")],
        order_datetime=_ORDER_DT,
        reserved_units_map={("HN1", "REQ1", ComponentFamily.RED_CELL): 5},
        msbos_reference=reference,
        oprtact_bridge=_bridge(),
    )

    assert columns["msbos_reason"] == "over_ceiling"
    assert columns["msbos_is_over"] is True
    assert columns["msbos_op_pick_status"] == "ambiguous_top_rank"
    assert columns["msbos_ceiling_basis"] == "G/M 2 (4573,8151)"
    assert columns["_bridge_gate"] == ""
    assert pipeline._declared_msbos_overlay(columns) == (
        "PREOP_OVER_RESERVATION",
        "preop_over_reservation",
    )


def test_rbc_cluster_within_ceiling_stays_exempt(pipeline: ModuleType) -> None:
    reference = _reference()

    columns = pipeline._msbos_reservation_columns(
        classification="PERIOP_TRANSFUSION_EXEMPT",
        hn="HN1",
        reqno="REQ1",
        op_events=[_event("8151"), _event("4573")],
        order_datetime=_ORDER_DT,
        reserved_units_map={("HN1", "REQ1", ComponentFamily.RED_CELL): 2},
        msbos_reference=reference,
        oprtact_bridge=_bridge(),
    )

    assert columns["msbos_reason"] == "within_ceiling"
    assert columns["msbos_is_over"] is False
    assert columns["msbos_ceiling_basis"] == "G/M 2 (4573,8151)"
    assert pipeline._declared_msbos_overlay(columns) is None


def test_rbc_single_code_ambiguous_ceiling_routes(pipeline: ModuleType) -> None:
    # One code resolving to two tariffs (G/M 2 + T/S 0); reserved 5 over the
    # G/M 2 ceiling -> over_ceiling with the confirmed exact/bridge gate.
    reference = _ambiguous_code_reference()
    bridge = _bridge(icd9="1234", score="0.99", human_index="0")

    columns = pipeline._msbos_reservation_columns(
        classification="PERIOP_TRANSFUSION_EXEMPT",
        hn="HN1",
        reqno="REQ1",
        op_events=[_event("INCPT:PX001")],
        order_datetime=_ORDER_DT,
        reserved_units_map={("HN1", "REQ1", ComponentFamily.RED_CELL): 5},
        msbos_reference=reference,
        oprtact_bridge=bridge,
    )

    assert columns["msbos_reason"] == "over_ceiling"
    assert columns["msbos_is_over"] is True
    assert columns["msbos_ceiling_basis"] == "G/M 2 (1234)"
    assert columns["_bridge_gate"] == ""
    assert pipeline._declared_msbos_overlay(columns) == (
        "PREOP_OVER_RESERVATION",
        "preop_over_reservation",
    )


def test_rbc_all_candidates_excluded_routes_review(pipeline: ModuleType) -> None:
    # A single in-window denylisted charge (AS058) with reserved units + declared
    # surgery -> all_candidates_excluded -> NEEDS_REVIEW, not silent exempt.
    reference = _reference()

    columns = pipeline._msbos_reservation_columns(
        classification="PERIOP_TRANSFUSION_EXEMPT",
        hn="HN1",
        reqno="REQ1",
        op_events=[_event("INCPT:AS058")],
        order_datetime=_ORDER_DT,
        reserved_units_map={("HN1", "REQ1", ComponentFamily.RED_CELL): 2},
        msbos_reference=reference,
        oprtact_bridge=_bridge(),
    )

    assert columns["msbos_op_pick_status"] == "all_candidates_excluded"
    assert pipeline._declared_msbos_overlay(columns) == (
        "NEEDS_REVIEW",
        "all_candidates_excluded",
    )


def test_platelet_columns_with_bridge_carry_provenance(pipeline: ModuleType) -> None:
    reference = _reference()
    bridge = _bridge(icd9="1733", human_index="1", human_icd9="4573")

    columns = pipeline._msbos_platelet_columns(
        classification="PERIOP_TRANSFUSION_EXEMPT",
        hn="HN1",
        reqno="REQ1",
        op_events=[_event("INCPT:PX001")],
        order_datetime=_ORDER_DT,
        pre_op_count_k_ul=250.0,
        reserved_units_map={("HN1", "REQ1", ComponentFamily.PLATELET): 4},
        msbos_reference=reference,
        oprtact_bridge=bridge,
    )

    assert columns["msbos_source_code"] == "PX001"
    assert columns["_bridge_gate"] == "bridge_disagreement"
    assert pipeline._declared_platelet_overlay(columns) == (
        "NEEDS_REVIEW",
        "preop_reservation_bridge_disagreement",
    )


# --- LLM-leg twin seam resolution ---------------------------------------------


def test_llm_leg_flag_resolution_matches_det_leg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "1")
    monkeypatch.setenv("BBA_PILOT_MSBOS_PLANNED_OP_PICKER_V2", "1")
    det = _load_pilot_module("run_pipeline.py", "pilot_run_pipeline_p2_twin")
    llm = _load_pilot_module("run_llm_leg.py", "pilot_run_llm_leg_p2_twin")

    assert det.MSBOS_PLANNED_OP_PICKER_V2_PILOT_ENABLED is True
    assert llm.MSBOS_PLANNED_OP_PICKER_V2_PILOT_ENABLED is True
    # Both legs share the library picker + finalize seams (twin parity is
    # structural: one implementation, two dispatch sites).
    from bba.preop_reservation.planned_op import (
        finalize_planned_op,
        planned_op_v2_for_events,
    )

    assert det.planned_op_v2_for_events is planned_op_v2_for_events
    assert llm.planned_op_v2_for_events is planned_op_v2_for_events
    assert det.finalize_planned_op is finalize_planned_op
    assert llm.finalize_planned_op is finalize_planned_op
