"""Focused pilot seam and byte-parity tests for ticket #162."""

from __future__ import annotations

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
def test_run_llm_leg_msbos_env_override(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", env_value)

    module = _load_run_llm_leg(f"pilot_run_llm_leg_msbos_{env_value}")

    assert module.MSBOS_RESERVATION_PILOT_ENABLED is expected
    tokens = module.CODE_VERSION.split("+")
    assert ("msbos4" in tokens) is expected, (
        "flag-on runs need a fresh T4 cache identity"
    )
    assert "msbos" not in tokens, (
        "the retired T2 token must not survive as a standalone cache identity"
    )


def test_run_llm_leg_msbos_unset_uses_library_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BBA_PILOT_MSBOS_RESERVATION", raising=False)

    module = _load_run_llm_leg("pilot_run_llm_leg_msbos_default")

    assert module.MSBOS_RESERVATION_PILOT_ENABLED is False
    assert "msbos4" not in module.CODE_VERSION.split("+")
    assert "msbos" not in module.CODE_VERSION.split("+")
    assert (
        module.MSBOS_RESERVATION_PILOT_ENABLED
        is feature_flags.MSBOS_RESERVATION_ENABLED
    )


@pytest.mark.parametrize("env_value", [None, "0"])
def test_msbos_flag_off_serialized_schemas_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
) -> None:
    if env_value is None:
        monkeypatch.delenv("BBA_PILOT_MSBOS_RESERVATION", raising=False)
    else:
        monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", env_value)
    llm_module = _load_run_llm_leg(
        f"pilot_run_llm_leg_msbos_schema_{env_value or 'default'}"
    )
    pipeline_module = _load_run_pipeline(
        f"pilot_run_pipeline_msbos_schema_{env_value or 'default'}"
    )

    serialized_columns = (
        tuple(pipeline_module.REPORT_FIELDNAMES)
        + tuple(pipeline_module.RETURNS_LEDGER_FIELDNAMES)
        + tuple(pipeline_module.DECLARED_USETYPE_FIELDNAMES)
        + RANKING_CSV_COLUMNS
    )
    assert llm_module.MSBOS_RESERVATION_PILOT_ENABLED is False
    assert "msbos4" not in llm_module.CODE_VERSION.split("+")
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


@pytest.mark.parametrize("env_value", [None, "0"])
def test_msbos_flag_off_never_loads_reference_or_reservations(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str | None,
) -> None:
    if env_value is None:
        monkeypatch.delenv("BBA_PILOT_MSBOS_RESERVATION", raising=False)
    else:
        monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", env_value)
    module = _load_run_llm_leg(
        f"pilot_run_llm_leg_msbos_no_load_{env_value or 'default'}"
    )

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
