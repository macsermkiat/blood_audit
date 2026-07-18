"""Focused pilot seam and byte-parity tests for ticket #162."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from bba import feature_flags
from bba.audit_orders import AuditOrder, FilterResult
from bba.attribution.outputs import RANKING_CSV_COLUMNS
from bba.cohort_detector import CohortAssignment, CohortLabel, OperativeEvent
from bba.deterministic_classifier import BypassReason, ClassifierResult
from bba.hb_lookup import HbLookupResult
from bba.platelet_lookup import PlateletLookupResult
from bba.vitals_extractor import VitalsNote


PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"


def _load_pilot_module(filename: str, module_name: str) -> ModuleType:
    if str(PILOT_DIR) not in sys.path:
        sys.path.insert(0, str(PILOT_DIR))
    spec = importlib.util.spec_from_file_location(module_name, PILOT_DIR / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_run_llm_leg(module_name: str) -> ModuleType:
    return _load_pilot_module("run_llm_leg.py", module_name)


def _load_run_pipeline(module_name: str) -> ModuleType:
    return _load_pilot_module("run_pipeline.py", module_name)


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("1", True), ("0", False), ("anything-else", False)],
)
def test_run_pipeline_msbos_env_override(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", env_value)

    module = _load_run_pipeline(f"pilot_run_pipeline_msbos_{env_value}")

    assert module.MSBOS_RESERVATION_PILOT_ENABLED is expected
    assert module.CODE_VERSION == "pilot-mini"


def test_run_pipeline_msbos_unset_uses_library_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BBA_PILOT_MSBOS_RESERVATION", raising=False)

    module = _load_run_pipeline("pilot_run_pipeline_msbos_default")

    assert module.MSBOS_RESERVATION_PILOT_ENABLED is True
    assert (
        module.MSBOS_RESERVATION_PILOT_ENABLED
        is feature_flags.MSBOS_RESERVATION_ENABLED
    )
    assert module.CODE_VERSION == "pilot-mini"


@pytest.mark.parametrize("env_value", ["1", "0"])
def test_run_pipeline_msbos_report_fieldnames_are_exact(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", env_value)
    module = _load_run_pipeline(f"pilot_run_pipeline_msbos_fields_{env_value}")
    flag_off = (
        module.REPORT_FIELDNAMES
        + module.RETURNS_LEDGER_FIELDNAMES
        + module.DECLARED_USETYPE_FIELDNAMES
    )
    expected = (
        flag_off
        + module.MSBOS_RESERVATION_FIELDNAMES
        + module.MSBOS_PLATELET_FIELDNAMES
        if env_value == "1"
        else flag_off
    )

    assert module._report_fieldnames() == expected
    assert module.MSBOS_PLATELET_FIELDNAMES == [
        "msbos_plt_category",
        "msbos_plt_count_k_ul",
        "msbos_plt_over_above_per_ul",
        "msbos_plt_clinician_signed",
    ]
    if env_value == "1":
        assert all(
            name in module._report_fieldnames()
            for name in (
                module.MSBOS_RESERVATION_FIELDNAMES + module.MSBOS_PLATELET_FIELDNAMES
            )
        )
    else:
        assert not any(
            name in module._report_fieldnames()
            for name in (
                module.MSBOS_RESERVATION_FIELDNAMES + module.MSBOS_PLATELET_FIELDNAMES
            )
        )


def test_run_pipeline_msbos_fieldnames_are_disjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "1")
    module = _load_run_pipeline("pilot_run_pipeline_msbos_disjoint")
    base_names = (
        module.REPORT_FIELDNAMES
        + module.RETURNS_LEDGER_FIELDNAMES
        + module.DECLARED_USETYPE_FIELDNAMES
    )

    msbos_names = module.MSBOS_RESERVATION_FIELDNAMES + module.MSBOS_PLATELET_FIELDNAMES
    assert len(msbos_names) == len(set(msbos_names))
    assert set(msbos_names).isdisjoint(base_names)
    assert not any(name.startswith("msbos_") for name in base_names)
    # Freeze the LITERAL schema (order + names). Q1: NO msbos_note_resolved column.
    assert module.MSBOS_RESERVATION_FIELDNAMES == [
        "msbos_reserved_units",
        "msbos_token",
        "msbos_recommended_units",
        "msbos_reason",
        "msbos_is_over",
        "msbos_resolved_icd9",
        "msbos_reference_hash",
    ]
    assert module.MSBOS_PLATELET_FIELDNAMES == [
        "msbos_plt_category",
        "msbos_plt_count_k_ul",
        "msbos_plt_over_above_per_ul",
        "msbos_plt_clinician_signed",
    ]


def test_run_pipeline_planned_op_icd9_nearest_ambiguous_and_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "1")
    module = _load_run_pipeline("pilot_run_pipeline_msbos_planned_op")
    order_datetime = datetime(2026, 7, 16, 4, 0, tzinfo=UTC)

    def event(code: str, offset_hours: int) -> OperativeEvent:
        return OperativeEvent(
            icd9=code,
            or_flag=True,
            operative_datetime=order_datetime + timedelta(hours=offset_hours),
        )

    assert (
        module._planned_op_icd9(
            [event("past", -1), event("later", 12), event("nearest", 4)],
            order_datetime,
        )
        == "nearest"
    )
    assert (
        module._planned_op_icd9(
            [event("1000", 4), event("2000", 4), event("later", 12)],
            order_datetime,
        )
        == "\x00AMBIG"
    )
    assert module._planned_op_icd9([event("past", -1)], order_datetime) == ""


def test_run_pipeline_msbos_reservation_columns_cover_locked_verdicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "1")
    module = _load_run_pipeline("pilot_run_pipeline_msbos_columns")
    reference = module.load_msbos_reference()
    order_datetime = datetime(2026, 7, 16, 4, 0, tzinfo=UTC)
    gm_code = "0139"
    ts_code = "0080"
    gm_row = reference.resolve(gm_code)
    ts_row = reference.resolve(ts_code)
    assert gm_row not in (None, "ambiguous") and gm_row.msbos == "G/M"
    assert ts_row not in (None, "ambiguous") and ts_row.msbos == "T/S"

    def event(code: str, offset_hours: int = 4) -> OperativeEvent:
        return OperativeEvent(
            icd9=code,
            or_flag=True,
            operative_datetime=order_datetime + timedelta(hours=offset_hours),
        )

    def columns(
        *,
        classification: str = "RETURNED_NOT_TRANSFUSED",
        hn: str = "HN1",
        reqno: str = "REQ1",
        events: list[OperativeEvent] | None = None,
        reserved: int = 0,
        include_key: bool = True,
    ) -> dict[str, object]:
        reservations = (
            {(hn.strip(), reqno.strip(), module.ComponentFamily.RED_CELL): reserved}
            if include_key
            else {}
        )
        return module._msbos_reservation_columns(
            classification=classification,
            hn=hn,
            reqno=reqno,
            op_events=events if events is not None else [event(gm_code)],
            order_datetime=order_datetime,
            reserved_units_map=reservations,
            msbos_reference=reference,
        )

    assert columns(classification="APPROPRIATE", reserved=5) == {}

    gm_over = columns(reserved=gm_row.recommended_units + 1)
    assert gm_over["msbos_reason"] == "over_gm_excess"
    assert gm_over["msbos_is_over"] is True
    assert gm_over["msbos_token"] == "G/M"

    gm_within = columns(reserved=gm_row.recommended_units)
    assert gm_within["msbos_reason"] == "within_recommendation"
    assert gm_within["msbos_is_over"] is False

    ts_screen = columns(events=[event(ts_code)], reserved=0)
    assert ts_screen["msbos_reason"] == "type_and_screen_screen_only"
    ts_over = columns(events=[event(ts_code)], reserved=1)
    assert ts_over["msbos_reason"] == "over_type_and_screen_crossmatched"
    assert ts_over["msbos_is_over"] is True

    ambiguous = columns(events=[event(gm_code), event(ts_code)], reserved=1)
    assert ambiguous["msbos_reason"] == "ambiguous_planned_op"
    no_upcoming = columns(events=[event(gm_code, -1)], reserved=1)
    assert no_upcoming["msbos_reason"] == "no_planned_op"

    absent = columns(include_key=False)
    blank_hn = columns(hn=" ", include_key=False)
    genuine_zero = columns(events=[event(ts_code)], reserved=0)
    assert absent["msbos_reason"] == "reservation_lookup_miss"
    assert blank_hn["msbos_reason"] == "reservation_lookup_miss"
    assert genuine_zero["msbos_reason"] == "type_and_screen_screen_only"
    assert absent["msbos_reason"] == blank_hn["msbos_reason"]
    assert genuine_zero["msbos_reason"] != absent["msbos_reason"]


def test_run_pipeline_msbos_platelet_columns_cover_locked_verdicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "1")
    module = _load_run_pipeline("pilot_run_pipeline_msbos_platelet_columns")
    order_datetime = datetime(2026, 7, 16, 4, 0, tzinfo=UTC)
    group_by_code = {
        "major": ("Arthroplasty",),
        "neuro": ("C Spine",),
        "cardiac": ("ศัลยกรรมหัวใจและทรวงอก",),
        "uncat": ("Unknown group",),
        "ambiguous-category": ("Arthroplasty", "C Spine"),
    }
    reference = SimpleNamespace(
        content_hash="platelet-reference-hash",
        groups_for=lambda code: group_by_code.get(code, ()),
    )

    def event(code: str, offset_hours: int = 4) -> OperativeEvent:
        return OperativeEvent(
            icd9=code,
            or_flag=True,
            operative_datetime=order_datetime + timedelta(hours=offset_hours),
        )

    def columns(
        *,
        classification: str = "RETURNED_NOT_TRANSFUSED",
        hn: str = "HN1",
        reqno: str = "REQ1",
        events: list[OperativeEvent] | None = None,
        count: float | None = 120.0,
        reserved: int = 2,
        include_key: bool = True,
    ) -> dict[str, object]:
        reservations = (
            {(hn.strip(), reqno.strip(), module.ComponentFamily.PLATELET): reserved}
            if include_key
            else {}
        )
        return module._msbos_platelet_columns(
            classification=classification,
            hn=hn,
            reqno=reqno,
            op_events=events if events is not None else [event("neuro")],
            order_datetime=order_datetime,
            pre_op_count_k_ul=count,
            reserved_units_map=reservations,
            msbos_reference=reference,
        )

    assert columns(classification="APPROPRIATE") == {}

    over = columns(events=[event("neuro")], count=120.5)
    assert over["msbos_reason"] == "over_neuraxial"
    assert over["msbos_is_over"] is True
    assert over["msbos_plt_count_k_ul"] == 120.5
    assert over["msbos_plt_over_above_per_ul"] == 100_000
    assert over["msbos_plt_category"] == "neuraxial"

    within = columns(events=[event("major")], count=80.0)
    assert within["msbos_reason"] == "within_major_non_neuraxial"
    assert within["msbos_is_over"] is False
    assert within["msbos_plt_count_k_ul"] == 80.0
    assert within["msbos_plt_over_above_per_ul"] == 80_000

    missing_count = columns(events=[event("cardiac")], count=None)
    assert missing_count["msbos_reason"] == "missing_pre_op_count"
    assert missing_count["msbos_is_over"] is False
    assert missing_count["msbos_plt_count_k_ul"] == ""
    assert missing_count["msbos_plt_category"] == "cardiac_cpb"
    assert missing_count["msbos_plt_over_above_per_ul"] == 100_000

    genuine_zero = columns(events=[event("major")], count=120.0, reserved=0)
    assert genuine_zero["msbos_reason"] == "no_reserved_units"
    assert genuine_zero["msbos_is_over"] is False
    assert genuine_zero["msbos_plt_count_k_ul"] == 120.0
    assert genuine_zero["msbos_plt_category"] == ""
    assert genuine_zero["msbos_plt_over_above_per_ul"] == ""

    unresolved_cases = {
        "uncategorised_procedure": [event("uncat")],
        "ambiguous_category": [event("ambiguous-category")],
        "no_planned_op": [],
        "ambiguous_planned_op": [event("major"), event("neuro")],
    }
    for expected_reason, events in unresolved_cases.items():
        unresolved = columns(events=events, count=55.5)
        assert unresolved["msbos_reason"] == expected_reason
        assert unresolved["msbos_is_over"] is False
        assert unresolved["msbos_plt_count_k_ul"] == 55.5
        assert unresolved["msbos_plt_over_above_per_ul"] == ""

    join_misses = [
        columns(hn=" "),
        columns(reqno=" "),
        columns(include_key=False),
    ]
    for join_miss in join_misses:
        assert join_miss["msbos_reason"] == "reservation_lookup_miss"
        assert join_miss["msbos_is_over"] is False
        assert join_miss["msbos_plt_count_k_ul"] == ""
    assert genuine_zero["msbos_reason"] != join_misses[0]["msbos_reason"]

    assert set(module.MSBOS_PLATELET_FIELDNAMES) <= set(module._report_fieldnames())
    shared_names = set(module.MSBOS_RESERVATION_FIELDNAMES) - {
        "msbos_token",
        "msbos_recommended_units",
    }
    assert shared_names <= set(over)
    assert set(module.MSBOS_PLATELET_FIELDNAMES) <= set(over)
    assert "msbos_token" not in over
    assert "msbos_recommended_units" not in over
    rbc = module._msbos_reservation_columns(
        classification="RETURNED_NOT_TRANSFUSED",
        hn="HN1",
        reqno="REQ1",
        op_events=[event("major")],
        order_datetime=order_datetime,
        reserved_units_map={("HN1", "REQ1", module.ComponentFamily.RED_CELL): 1},
        msbos_reference=module.load_msbos_reference(),
    )
    assert not set(module.MSBOS_PLATELET_FIELDNAMES).intersection(rbc)


def _configure_minimal_run_pipeline_platelet(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> AuditOrder:
    order_datetime = datetime(2026, 7, 16, 4, 0, tzinfo=UTC)
    order = AuditOrder(
        audit_id="audit-pilot-msbos-platelet-main",
        hn="HN1",
        an="AN1",
        reqno="REQ1",
        order_datetime=order_datetime,
        anchor_imputed=False,
        products_ordered=("PLT-POOL",),
        diagnosis_codes=(),
        component="platelet",
    )
    operation = OperativeEvent(
        icd9="major",
        or_flag=True,
        operative_datetime=order_datetime + timedelta(hours=4),
    )
    platelet_result = PlateletLookupResult(
        value_k_ul=120.0,
        datetime_utc=order_datetime,
        source="HEMATOLOGY",
        freshness="fresh",
    )
    returns_result = ClassifierResult(
        classification="RETURNED_NOT_TRANSFUSED",
        bypass_reason=BypassReason.RETURNED_NOT_TRANSFUSED,
        cohort_threshold=None,
        rationale="returned_not_transfused",
    )
    reference = SimpleNamespace(
        content_hash="platelet-reference-hash",
        groups_for=lambda code: ("Arthroplasty",) if code == "major" else (),
    )
    reserved_units_map = {
        ("HN1", "REQ1", module.ComponentFamily.PLATELET): 2,
    }

    work = tmp_path / "pilot"
    bundle = work / "bundle"
    bundle.mkdir(parents=True)
    monkeypatch.setattr(module, "WORK", work)
    monkeypatch.setattr(module, "BUNDLE", bundle)
    # Register the shared flags with monkeypatch even though this pilot leg only
    # reads import-time constants, so the integration seam cannot leak state.
    monkeypatch.setattr(
        feature_flags,
        "RESERVE_AHEAD_ROUTER_ENABLED",
        feature_flags.RESERVE_AHEAD_ROUTER_ENABLED,
    )
    monkeypatch.setattr(
        feature_flags,
        "DECLARED_USETYPE_ENABLED",
        feature_flags.DECLARED_USETYPE_ENABLED,
    )
    monkeypatch.setattr(
        feature_flags,
        "MSBOS_RESERVATION_ENABLED",
        feature_flags.MSBOS_RESERVATION_ENABLED,
    )
    monkeypatch.setattr(module, "_read_csv", lambda _name: [])
    monkeypatch.setattr(module, "load_bdvsttrans_rows", lambda _bundle: [])
    monkeypatch.setattr(module, "load_msbos_reference", lambda: reference)
    monkeypatch.setattr(
        module,
        "reserved_units_by_component",
        lambda _rows: reserved_units_map,
    )
    monkeypatch.setattr(
        module,
        "build_audit_orders",
        lambda _inputs, _config: FilterResult(included=(order,), excluded=()),
    )
    monkeypatch.setattr(module, "lookup_platelet", lambda **_kwargs: platelet_result)
    monkeypatch.setattr(module, "_build_op_events", lambda *_args: (operation,))
    monkeypatch.setattr(
        module,
        "_platelet_returns_result",
        lambda **_kwargs: returns_result,
    )
    return order


def test_run_pipeline_main_serializes_flag_on_platelet_msbos_columns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "1")
    module = _load_run_pipeline("pilot_run_pipeline_msbos_platelet_main_on")
    order = _configure_minimal_run_pipeline_platelet(module, monkeypatch, tmp_path)

    module.main()

    with (module.WORK / "report.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert row["reqno"] == order.reqno
    assert row["component"] == "platelet"
    assert row["classification"] == "RETURNED_NOT_TRANSFUSED"
    assert {
        "msbos_plt_category": row["msbos_plt_category"],
        "msbos_plt_count_k_ul": row["msbos_plt_count_k_ul"],
        "msbos_plt_over_above_per_ul": row["msbos_plt_over_above_per_ul"],
        "msbos_plt_clinician_signed": row["msbos_plt_clinician_signed"],
    } == {
        "msbos_plt_category": "major_non_neuraxial",
        "msbos_plt_count_k_ul": "120.0",
        "msbos_plt_over_above_per_ul": "80000",
        "msbos_plt_clinician_signed": "True",
    }
    assert {
        "msbos_reserved_units": row["msbos_reserved_units"],
        "msbos_reason": row["msbos_reason"],
        "msbos_is_over": row["msbos_is_over"],
        "msbos_reference_hash": row["msbos_reference_hash"],
        "msbos_resolved_icd9": row["msbos_resolved_icd9"],
    } == {
        "msbos_reserved_units": "2",
        "msbos_reason": "over_major_non_neuraxial",
        "msbos_is_over": "True",
        "msbos_reference_hash": "platelet-reference-hash",
        "msbos_resolved_icd9": "major",
    }
    assert row["msbos_token"] == ""
    assert row["msbos_recommended_units"] == ""


def test_run_pipeline_main_flag_off_keeps_msbos_schema_frozen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "0")
    module = _load_run_pipeline("pilot_run_pipeline_msbos_platelet_main_off")
    _configure_minimal_run_pipeline_platelet(module, monkeypatch, tmp_path)

    module.main()

    with (module.WORK / "report.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert not any(name.startswith("msbos_") for name in rows[0])


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("1", True), ("0", False), ("anything-else", False)],
)
def test_run_llm_leg_msbos_env_override(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", env_value)

    module = _load_run_llm_leg(f"pilot_run_llm_leg_msbos_{env_value}")

    assert module.MSBOS_RESERVATION_PILOT_ENABLED is expected
    tokens = module.CODE_VERSION.split("+")
    assert ("msbos5" in tokens) is expected, (
        "flag-on runs need a fresh reservation cache identity"
    )
    assert "msbos" not in tokens, (
        "the retired T2 token must not survive as a standalone cache identity"
    )


def test_run_llm_leg_msbos_unset_uses_library_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With no env override the pilot leg follows the library default, which is
    # ON since the MSBOS go-live (#167): a fresh reservation cache identity, no
    # standalone retired-T2 token.
    monkeypatch.delenv("BBA_PILOT_MSBOS_RESERVATION", raising=False)

    module = _load_run_llm_leg("pilot_run_llm_leg_msbos_default")

    assert module.MSBOS_RESERVATION_PILOT_ENABLED is True
    assert "msbos5" in module.CODE_VERSION.split("+")
    assert "msbos" not in module.CODE_VERSION.split("+")
    assert (
        module.MSBOS_RESERVATION_PILOT_ENABLED
        is feature_flags.MSBOS_RESERVATION_ENABLED
    )


# Flag-off is now an explicit opt-out (BBA_PILOT_MSBOS_RESERVATION=0); the unset
# default is ON post-go-live (#167), so it is no longer a flag-off case.
@pytest.mark.parametrize("env_value", ["0"])
def test_msbos_flag_off_serialized_schemas_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", env_value)
    llm_module = _load_run_llm_leg(f"pilot_run_llm_leg_msbos_schema_{env_value}")
    pipeline_module = _load_run_pipeline(f"pilot_run_pipeline_msbos_schema_{env_value}")

    serialized_columns = (
        tuple(pipeline_module.REPORT_FIELDNAMES)
        + tuple(pipeline_module.RETURNS_LEDGER_FIELDNAMES)
        + tuple(pipeline_module.DECLARED_USETYPE_FIELDNAMES)
        + RANKING_CSV_COLUMNS
    )
    assert llm_module.MSBOS_RESERVATION_PILOT_ENABLED is False
    assert "msbos5" not in llm_module.CODE_VERSION.split("+")
    assert not any(
        marker in name
        for name in serialized_columns
        for marker in ("over_reservation", "operation_unresolved", "note_resolved")
    ), "T3's in-memory fields must not alter any flag-off serialized schema"


def test_msbos_deterministic_final_vocabulary_is_inert_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "0")
    module = _load_run_llm_leg("pilot_run_llm_leg_msbos_vocab")

    assert "PREOP_OVER_RESERVATION" in module.DETERMINISTIC_FINAL


@pytest.mark.parametrize("env_value", ["0"])
def test_msbos_flag_off_never_loads_reference_or_reservations(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", env_value)
    module = _load_run_llm_leg(f"pilot_run_llm_leg_msbos_no_load_{env_value}")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("flag-off MSBOS loader/accessor was invoked")

    monkeypatch.setattr(module, "load_msbos_reference", forbidden)
    monkeypatch.setattr(module, "reserved_units_by_component", forbidden)
    monkeypatch.setattr(module, "_read_csv", lambda _filename: [])
    monkeypatch.setattr(module, "_read_optional_csv", lambda _filename: [])
    monkeypatch.setattr(
        module, "_read_preferred_optional_csv", lambda _preferred, _fallback: []
    )
    monkeypatch.setattr(module, "load_bdvsttrans_rows", lambda _bundle: [])

    built = module._build_inputs()

    assert built[-2] is None
    assert built[-1] == {}


def test_planned_op_icd9_uses_nearest_upcoming_and_detects_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "0")
    module = _load_run_llm_leg("pilot_run_llm_leg_msbos_planned_op")
    order_datetime = datetime(2026, 7, 16, 4, 0, tzinfo=UTC)

    def event(code: str, offset_hours: int) -> OperativeEvent:
        return OperativeEvent(
            icd9=code,
            or_flag=True,
            operative_datetime=order_datetime + timedelta(hours=offset_hours),
        )

    assert (
        module._planned_op_icd9(
            [event("past", -1), event("later", 12), event("nearest", 4)],
            order_datetime,
        )
        == "nearest"
    )
    assert (
        module._planned_op_icd9(
            [event("1000", 4), event("2000", 4), event("later", 12)],
            order_datetime,
        )
        == "\x00AMBIG"
    )
    assert module._planned_op_icd9([event("past", -1)], order_datetime) == ""


def test_planned_op_icd9_keeps_nearer_minor_code_when_later_major_drives_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "1")
    module = _load_run_llm_leg("pilot_run_llm_leg_msbos_minor_lock")
    order_datetime = datetime(2026, 7, 16, 4, 0, tzinfo=UTC)
    nearer_minor = OperativeEvent(
        icd9="minor-code",
        or_flag=True,
        operative_datetime=order_datetime + timedelta(hours=2),
    )
    later_major = OperativeEvent(
        icd9="major-code",
        or_flag=True,
        operative_datetime=order_datetime + timedelta(hours=8),
    )

    assert (
        module._planned_op_icd9([later_major, nearer_minor], order_datetime)
        == "minor-code"
    ), (
        "T3 must lock the pre-existing all-upcoming selection even though the "
        "reserve-ahead router may be triggered by the later major operation"
    )


def _configure_minimal_msbos_pilot(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    shipped_note: str,
    classifier_classification: str = "POTENTIALLY_INAPPROPRIATE",
) -> AuditOrder:
    order_datetime = datetime(2026, 7, 16, 4, 0, tzinfo=UTC)
    order = AuditOrder(
        audit_id="audit-pilot-msbos-t3",
        hn="HN1",
        an="AN1",
        reqno="REQ1",
        order_datetime=order_datetime,
        anchor_imputed=False,
        products_ordered=("LPRC",),
        diagnosis_codes=(),
    )
    operation = OperativeEvent(
        icd9="0124",
        or_flag=True,
        operative_datetime=order_datetime + timedelta(hours=4),
    )
    hb = HbLookupResult(
        value_g_dl=9.0,
        datetime_utc=order_datetime,
        source="HEMATOLOGY",
        freshness="fresh",
        delta_hb_bypass=False,
        delta_hb_windows=(),
        needs_review_single_low_hb=False,
    )
    cohort = CohortAssignment(
        label=CohortLabel.DEFAULT,
        threshold=7.0,
        evidence_code=None,
        evidence_name=None,
    )
    bundle = SimpleNamespace(
        items=(
            SimpleNamespace(
                id="E1",
                source="IPDNRFOCUSDT",
                timestamp_utc=order_datetime,
                payload={"text": shipped_note},
            ),
        ),
        bundle_hash="d" * 64,
        periop_summary=None,
        administration_summary=None,
    )
    reference = module.load_msbos_reference()
    built_inputs = (
        [],
        [],
        [],
        [],
        [],
        [],
        {},
        {},
        [],
        [],
        {},
        {},
        {},
        {},
        {},
        None,
        reference,
        {("HN1", "REQ1", module.ComponentFamily.RED_CELL): 5},
    )

    work = tmp_path / "pilot"
    bundle_dir = work / "bundle"
    bundle_dir.mkdir(parents=True)
    monkeypatch.setattr(module, "WORK", work)
    monkeypatch.setattr(module, "BUNDLE", bundle_dir)
    monkeypatch.setattr(module, "AUDIT_STORE_ROOT", work / "data" / "audit_store")
    monkeypatch.setattr(module, "RUN_ID", "run-pilot-msbos-t3")
    monkeypatch.setattr(module, "ONLY_REQNOS", frozenset())
    # main() deliberately binds these process-wide flags from its import-time
    # pilot constants. Register their current values with monkeypatch so this
    # integration seam cannot leak a flag-on state into later unit tests.
    monkeypatch.setattr(
        feature_flags,
        "RESERVE_AHEAD_ROUTER_ENABLED",
        feature_flags.RESERVE_AHEAD_ROUTER_ENABLED,
    )
    monkeypatch.setattr(
        feature_flags,
        "DECLARED_USETYPE_ENABLED",
        feature_flags.DECLARED_USETYPE_ENABLED,
    )
    monkeypatch.setattr(
        feature_flags,
        "MSBOS_RESERVATION_ENABLED",
        feature_flags.MSBOS_RESERVATION_ENABLED,
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(module, "_build_inputs", lambda: built_inputs)
    monkeypatch.setattr(
        module,
        "build_audit_orders",
        lambda _inputs, _config: FilterResult(included=(order,), excluded=()),
    )
    monkeypatch.setattr(module, "_hb_observations", lambda _rows, _an: ())
    monkeypatch.setattr(
        module,
        "resolve_hb_with_fallback",
        lambda **_kwargs: (hb, "", "order_datetime"),
    )
    monkeypatch.setattr(module, "_op_events", lambda *_args: [operation])
    monkeypatch.setattr(module, "_med_events", lambda *_args: [])
    monkeypatch.setattr(module, "_latest_anc", lambda *_args: None)
    monkeypatch.setattr(module, "assign_cohort", lambda _inputs: cohort)
    monkeypatch.setattr(module, "is_blood_requiring_procedure", lambda _code: True)
    monkeypatch.setattr(
        module,
        "vitals_notes_for",
        lambda *_args: (
            VitalsNote(
                timestamp=order_datetime,
                text="Craniotomy (tumor) appears only in the unwindowed source",
                source="IPDNRFOCUSDT",
            ),
        ),
    )
    monkeypatch.setattr(module, "build_evidence_bundle", lambda **_kwargs: bundle)
    monkeypatch.setattr(
        module,
        "classify",
        lambda _inputs: ClassifierResult(
            classification=classifier_classification,
            bypass_reason=BypassReason.NONE,
            cohort_threshold=7.0,
            rationale="preop_defer_llm",
        ),
    )
    return order


@pytest.mark.parametrize(
    ("shipped_note", "expected_classification", "expected_reason"),
    [
        (
            "No operation name survived the evidence window.",
            "NEEDS_REVIEW",
            "operation_unresolved",
        ),
        (
            "Scheduled Craniotomy (tumor) tomorrow.",
            "PREOP_OVER_RESERVATION",
            "preop_over_reservation",
        ),
    ],
    ids=["conflict-unresolved-included-in-report", "note-resolved-over-reservation"],
)
def test_flag_on_pilot_dispatches_conflicting_codes_from_shipped_notes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shipped_note: str,
    expected_classification: str,
    expected_reason: str,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "1")
    module = _load_run_llm_leg(
        f"pilot_run_llm_leg_t3_{expected_classification.lower()}"
    )
    order = _configure_minimal_msbos_pilot(
        module, monkeypatch, tmp_path, shipped_note=shipped_note
    )

    module.main()

    report = json.loads((module.WORK / "llm_report.json").read_text())
    assert [record["audit_id"] for record in report] == [order.audit_id], (
        "a locally persisted conflicting-code terminal must remain visible in the report"
    )
    assert report[0]["llm_final"]["final_classification"] == expected_classification
    assert report[0]["llm_final"]["review_reason"] == expected_reason


def test_flag_off_pilot_never_reaches_note_disambiguation_or_writes_t3_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "0")
    module = _load_run_llm_leg("pilot_run_llm_leg_t3_flag_off")
    _configure_minimal_msbos_pilot(
        module,
        monkeypatch,
        tmp_path,
        shipped_note="Craniotomy (tumor)",
        classifier_classification="APPROPRIATE",
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("flag-off pilot reached note disambiguation")

    monkeypatch.setattr(module, "evaluate_reservation_with_notes", forbidden)

    with pytest.raises(SystemExit, match="nothing to submit"):
        module.main()

    store = module.AuditStore(
        module.AuditStoreConfig(
            root_dir=module.AUDIT_STORE_ROOT, code_version=module.CODE_VERSION
        )
    )
    assert store.read_audit_results(run_id=module.RUN_ID) == (), (
        "flag-off execution must produce no operation-unresolved or note-resolved rows"
    )
