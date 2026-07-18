"""Pilot wiring tests for the planned-op picker v2 seam (ticket #199).

Covers the 2x2 flag/schema matrix, the +opbound CODE_VERSION token, the
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

    assert ("+opbound" in module.CODE_VERSION) is expect_token
    if msbos_env == "1" and picker_env == "0":
        assert "+msbos5" in module.CODE_VERSION
    if expect_token:
        assert "+msbos5+opbound" in module.CODE_VERSION


def test_picker_default_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BBA_PILOT_MSBOS_PLANNED_OP_PICKER_V2", raising=False)
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "1")
    module = _load_pilot_module("run_pipeline.py", "pilot_run_pipeline_p2_default")

    assert module.MSBOS_PLANNED_OP_PICKER_V2_PILOT_ENABLED is False
    assert not any(
        c in module._report_fieldnames() for c in module.MSBOS_PICKER_V2_FIELDNAMES
    )


# --- declared-row verdict overlays --------------------------------------------


@pytest.fixture(scope="module")
def pipeline() -> ModuleType:
    return _load_pilot_module("run_pipeline.py", "pilot_run_pipeline_p2_overlay")


def test_rbc_overlay_precedence_matrix(pipeline: ModuleType) -> None:
    overlay = pipeline._declared_msbos_overlay

    # Ambiguity outranks everything.
    assert overlay(
        {
            "msbos_op_pick_status": "ambiguous_top_rank",
            "_bridge_gate": "bridge_disagreement",
            "msbos_is_over": True,
        }
    ) == ("NEEDS_REVIEW", "ambiguous_planned_op")
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
    # Both legs share the library picker + gate seams (twin parity is
    # structural: one implementation, two dispatch sites).
    from bba.preop_reservation.planned_op import (
        attach_planned_op,
        planned_op_v2_for_events,
    )

    assert det.planned_op_v2_for_events is planned_op_v2_for_events
    assert llm.planned_op_v2_for_events is planned_op_v2_for_events
    assert det.attach_planned_op is attach_planned_op
    assert llm.attach_planned_op is attach_planned_op
