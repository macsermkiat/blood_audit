"""Committee-report tests for MSBOS reservation pilot ticket #167."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import bba.feature_flags as feature_flags
import pytest

from bba.audit_store import AuditRow, LlmCall
from bba.preop_reservation.pilot_report import (
    PilotReportError,
    build_pilot_report,
    reconcile_returns,
)

_NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)
_RUN_ID = "run-msbos-report"
_CODE_VERSION = "pilot+msbos4"
_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "pilot"
    / "msbos_reservation_pilot_report.py"
)


def _row(
    audit_id: str,
    final_classification: str,
    *,
    component: str = "red_cell",
) -> AuditRow:
    values: dict[str, object] = {
        "audit_id": audit_id,
        "run_id": _RUN_ID,
        "run_timestamp": _NOW,
        "hn_hash": f"hn-{audit_id}",
        "an_hash": f"an-{audit_id}",
        "reqno": f"REQ-{audit_id}",
        "order_datetime": _NOW,
        "products_ordered": ("LPRC",),
        "hb_value": 8.0,
        "hb_datetime": _NOW,
        "hb_freshness": "fresh",
        "hb_source": "LAB",
        "vitals_sbp": None,
        "vitals_hr": None,
        "vitals_timestamp": None,
        "vitals_source": None,
        "prior_rbc_units_24h": 0,
        "prior_rbc_units_7d": 0,
        "cohort_threshold": 7.0,
        "delta_hb_window_results": (),
        "rule_classification": "NEEDS_REVIEW",
        "final_classification": final_classification,
        "cohort_applied": "general_medical",
        "indications_json": (),
        "negative_evidence_json": (),
        "confidence": 1.0,
        "reasoning_summary_thai": "",
        "reasoning_summary_en": "Persisted reservation terminal.",
        "needs_human_review": final_classification == "NEEDS_REVIEW",
        "review_reason": None,
        "model_id": "msbos-reservation",
        "prompt_hash": "prompt",
        "evidence_bundle_hash": "bundle",
        "redactor_version": "test",
        "redactor_model_sha": "sha",
        "policy_version": "test",
        "verifier_pass": True,
        "verifier_retries": 0,
        "escalated_to_opus": False,
        "component": component,
    }
    return AuditRow.model_validate(values)


def _payload(key: str, audit_id: str, **overrides: object) -> dict[str, object]:
    if key in {"over_reservation", "operation_unresolved"}:
        payload: dict[str, object] = {
            key: True,
            "audit_id": audit_id,
            "reason": (
                "over_none" if key == "over_reservation" else "operation_unresolved"
            ),
            "resolved_icd9": "1000",
            "note_resolved": False,
        }
    else:
        payload = {
            key: True,
            "audit_id": audit_id,
            "reason": (
                "over_major_non_neuraxial"
                if key == "platelet_over_reservation"
                else "uncategorised_procedure"
            ),
            "resolved_icd9": "0613",
            "category": "major_non_neuraxial",
            "pre_op_count_k_ul": 120.0,
            "reserved_units": 2,
            "clinician_signed": True,
        }
    payload.update(overrides)
    return payload


def _call(
    audit_id: str,
    key: str,
    *,
    call_id: str | None = None,
    model_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> LlmCall:
    is_platelet = key.startswith("platelet_")
    return LlmCall.model_validate(
        {
            "call_id": call_id or f"call-{audit_id}",
            "audit_id": audit_id,
            "run_id": _RUN_ID,
            "model_id": model_id
            or ("msbos-platelet-reservation" if is_platelet else "msbos-reservation"),
            "anthropic_version": "deterministic",
            "prompt_cache_id": None,
            "request_json": payload or _payload(key, audit_id),
            "response_json": {},
            "request_timestamp": _NOW,
            "latency_ms": 0,
            "extended_thinking_blocks": None,
            "cold_storage_uri": None,
        }
    )


def _build(rows: list[AuditRow], calls: list[LlmCall]):
    return build_pilot_report(rows, calls, run_id=_RUN_ID, code_version=_CODE_VERSION)


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location("msbos_report_cli_test", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rbc_coverage_counts_only_conflicting_code_terminals() -> None:
    # Arrange
    rows = [
        _row("resolved", "PREOP_OVER_RESERVATION"),
        _row("unresolved", "NEEDS_REVIEW"),
        _row("plain-none", "PREOP_OVER_RESERVATION"),
        _row("plain-gm", "PREOP_OVER_RESERVATION"),
    ]
    calls = [
        _call(
            "resolved",
            "over_reservation",
            payload=_payload("over_reservation", "resolved", note_resolved=True),
        ),
        _call("unresolved", "operation_unresolved"),
        _call("plain-none", "over_reservation"),
        _call(
            "plain-gm",
            "over_reservation",
            payload=_payload("over_reservation", "plain-gm", reason="over_gm_excess"),
        ),
    ]

    # Act
    coverage = _build(rows, calls).coverage.rbc_note_resolution_rate

    # Assert
    assert coverage.resolved == 1
    assert coverage.unresolved == 1
    assert coverage.denominator == 2
    assert coverage.rate == 0.5


def test_platelet_coverage_separates_non_category_review_reasons() -> None:
    # Arrange
    reasons = [
        "uncategorised_procedure",
        "ambiguous_category",
        "missing_pre_op_count",
        "no_planned_op",
        "ambiguous_planned_op",
    ]
    rows = [_row("over", "PREOP_OVER_RESERVATION", component="platelet")]
    rows.extend(
        _row(f"review-{index}", "NEEDS_REVIEW", component="platelet")
        for index in range(len(reasons))
    )
    calls = [_call("over", "platelet_over_reservation")]
    calls.extend(
        _call(
            f"review-{index}",
            "platelet_reservation_review",
            payload=_payload(
                "platelet_reservation_review",
                f"review-{index}",
                reason=reason,
            ),
        )
        for index, reason in enumerate(reasons)
    )

    # Act
    coverage = _build(rows, calls).coverage

    # Assert
    rate = coverage.platelet_category_resolution_rate
    assert (rate.resolved, rate.category_unresolved, rate.denominator) == (1, 2, 3)
    assert rate.rate == pytest.approx(1 / 3)
    assert coverage.platelet_other_review_reasons.model_dump() == {
        "missing_pre_op_count": 1,
        "no_planned_op": 1,
        "ambiguous_planned_op": 1,
    }


def test_precision_is_pending_with_non_additive_assertion_denominators() -> None:
    # Arrange
    specifications = [
        ("none-note", "over_none", True, "red_cell"),
        ("gm", "over_gm_excess", False, "red_cell"),
        (
            "ts",
            "over_type_and_screen_crossmatched",
            False,
            "red_cell",
        ),
        ("ceiling", "over_ceiling", False, "red_cell"),
        ("plt-major", "over_major_non_neuraxial", False, "platelet"),
        ("plt-neuraxial", "over_neuraxial", False, "platelet"),
        ("plt-cardiac", "over_cardiac_cpb", False, "platelet"),
    ]
    rows = [
        _row(audit_id, "PREOP_OVER_RESERVATION", component=component)
        for audit_id, _reason, _note, component in specifications
    ]
    calls = []
    for audit_id, reason, note_resolved, component in specifications:
        key = (
            "platelet_over_reservation"
            if component == "platelet"
            else "over_reservation"
        )
        calls.append(
            _call(
                audit_id,
                key,
                payload=_payload(
                    key,
                    audit_id,
                    reason=reason,
                    **(
                        {"note_resolved": note_resolved}
                        if component == "red_cell"
                        else {}
                    ),
                ),
            )
        )

    # Act
    precision = _build(rows, calls).precision

    # Assert
    assert precision.status == "PENDING_CLINICIAN_VALIDATED_SAMPLE"
    assert precision.rbc_assertion_denominators.model_dump() == {
        "none_bucket_over_assertions": 1,
        "note_resolved_over_assertions": 1,
        "over_gm_excess": 1,
        "over_type_and_screen_crossmatched": 1,
        "over_ceiling": 1,
    }
    assert precision.platelet_assertion_denominators.model_dump() == {
        "over_major_non_neuraxial": 1,
        "over_neuraxial": 1,
        "over_cardiac_cpb": 1,
    }
    assert precision.rbc_none_and_note_resolved_non_additive is True
    assert "precision" not in type(precision).model_fields
    assert "ppv" not in type(precision).model_fields


def test_reconcile_returns_disjoint_sets_pass() -> None:
    # Arrange / Act
    result = reconcile_returns({"over-1", "over-2"}, {"return-1"})

    # Assert
    assert result.status == "PASS"
    assert result.double_fire_count == 0
    assert result.double_fire_ids == ()


def test_reconcile_returns_overlap_fails_and_lists_ids() -> None:
    # Arrange / Act
    result = reconcile_returns({"shared", "over-only"}, {"shared", "return-only"})

    # Assert — this is the non-tautological failure proof required by Rule 9.
    assert result.status == "FAIL"
    assert result.double_fire_count == 1
    assert result.double_fire_ids == ("shared",)


@pytest.mark.parametrize(
    "returns_terminal",
    ["RETURNED_NOT_TRANSFUSED", "PERIOP_TRANSFUSION_EXEMPT"],
)
def test_over_marker_on_returns_terminal_row_is_reported_not_aborted(
    returns_terminal: str,
) -> None:
    # Arrange — the exact double-fire the reconciliation guards: an over-marker
    # whose committed row is a returns terminal. It must reach reconcile_returns
    # and be REPORTED as FAIL, not abort the whole gating report (Codex #174 P2).
    rows = [
        _row("double-fire", returns_terminal),
        _row("clean-over", "PREOP_OVER_RESERVATION"),
    ]
    calls = [
        _call("double-fire", "over_reservation"),
        _call("clean-over", "over_reservation"),
    ]

    # Act
    reconciliation = _build(rows, calls).reconciliation

    # Assert
    assert reconciliation.status == "FAIL"
    assert reconciliation.double_fire_ids == ("double-fire",)
    assert reconciliation.returns_terminal_count == 1


def test_over_marker_on_non_returns_final_still_fails_loud() -> None:
    # Arrange — a non-returns final-classification mismatch is still corruption.
    row = _row("wrong-final", "APPROPRIATE")
    call = _call("wrong-final", "over_reservation")

    # Act / Assert
    with pytest.raises(PilotReportError, match="final_classification"):
        _build([row], [call])


def test_orphan_marker_fails_loud_and_lists_all_ids() -> None:
    # Arrange
    calls = [
        _call("orphan-b", "over_reservation"),
        _call("orphan-a", "over_reservation"),
    ]

    # Act / Assert
    with pytest.raises(PilotReportError, match=r"orphan-a.*orphan-b"):
        _build([], calls)


def test_duplicate_marker_for_one_audit_id_fails_loud() -> None:
    # Arrange
    row = _row("duplicate", "PREOP_OVER_RESERVATION")
    calls = [
        _call("duplicate", "over_reservation", call_id="call-one"),
        _call("duplicate", "over_reservation", call_id="call-two"),
    ]

    # Act / Assert
    with pytest.raises(PilotReportError, match="duplicate reservation marker"):
        _build([row], calls)


def test_duplicate_scoped_audit_id_fails_loud() -> None:
    # Arrange
    rows = [
        _row("duplicate-row", "APPROPRIATE"),
        _row("duplicate-row", "NEEDS_REVIEW"),
    ]

    # Act / Assert
    with pytest.raises(PilotReportError, match="duplicate scoped audit row"):
        _build(rows, [])


@pytest.mark.parametrize(
    "payload",
    [
        _payload(
            "over_reservation",
            "bad-key",
            operation_unresolved=True,
        ),
        {"audit_id": "bad-key", "reason": "over_none"},
    ],
)
def test_marker_requires_exactly_one_true_boolean_key(
    payload: dict[str, object],
) -> None:
    # Arrange
    row = _row("bad-key", "PREOP_OVER_RESERVATION")
    call = _call("bad-key", "over_reservation", payload=payload)

    # Act / Assert
    with pytest.raises(PilotReportError, match="exactly one true marker key"):
        _build([row], [call])


def test_review_markers_are_skipped_not_tallied_or_failed() -> None:
    # Picker-v2 NEEDS_REVIEW terminals (bridge disagreement / all-candidates-
    # excluded) carry a review-only request key, so the committee report skips
    # them rather than tripping the exactly-one-marker-key validation (#196/#210).
    rows = [
        _row("over", "PREOP_OVER_RESERVATION"),
        _row("disagree", "NEEDS_REVIEW"),
        _row("excluded", "NEEDS_REVIEW"),
        _row("plt-disagree", "NEEDS_REVIEW", component="platelet"),
    ]
    calls = [
        _call("over", "over_reservation"),
        _call(
            "disagree",
            "bridge_disagreement",
            payload={
                "bridge_disagreement": True,
                "audit_id": "disagree",
                "reason": "within_recommendation",
                "resolved_icd9": "1000",
                "note_resolved": False,
            },
        ),
        _call(
            "excluded",
            "all_candidates_excluded",
            payload={
                "all_candidates_excluded": True,
                "audit_id": "excluded",
                "reason": "no_planned_op",
                "resolved_icd9": "",
                "note_resolved": False,
            },
        ),
        _call(
            "plt-disagree",
            "platelet_bridge_disagreement",
            model_id="msbos-platelet-reservation",
            payload={
                "platelet_bridge_disagreement": True,
                "audit_id": "plt-disagree",
                "reason": "within_major_non_neuraxial",
                "resolved_icd9": "0613",
            },
        ),
    ]

    report = _build(rows, calls)

    # Only the real over-assertion is tallied; both review markers are ignored.
    assert report.provenance.total_reservation_markers == 1
    assert report.reconciliation.over_marker_count == 1
    assert report.precision.rbc_assertion_denominators.none_bucket_over_assertions == 1


def test_marker_key_must_match_model_family() -> None:
    # Arrange
    row = _row("wrong-family", "PREOP_OVER_RESERVATION")
    call = _call(
        "wrong-family",
        "over_reservation",
        model_id="msbos-platelet-reservation",
    )

    # Act / Assert
    with pytest.raises(PilotReportError, match="does not match model_id"):
        _build([row], [call])


def test_payload_audit_id_must_match_call_audit_id() -> None:
    # Arrange
    row = _row("payload-mismatch", "PREOP_OVER_RESERVATION")
    call = _call(
        "payload-mismatch",
        "over_reservation",
        payload=_payload("over_reservation", "different-id"),
    )

    # Act / Assert
    with pytest.raises(PilotReportError, match="payload audit_id"):
        _build([row], [call])


def test_marker_final_classification_must_match_terminal_kind() -> None:
    # Arrange
    row = _row("wrong-final", "NEEDS_REVIEW")
    call = _call("wrong-final", "over_reservation")

    # Act / Assert
    with pytest.raises(PilotReportError, match="final_classification"):
        _build([row], [call])


def test_marker_component_must_match_model_family() -> None:
    # Arrange
    row = _row("wrong-component", "PREOP_OVER_RESERVATION", component="platelet")
    call = _call("wrong-component", "over_reservation")

    # Act / Assert
    with pytest.raises(PilotReportError, match="requires component"):
        _build([row], [call])


@pytest.mark.parametrize(
    ("key", "component", "unknown_reason"),
    [
        ("over_reservation", "red_cell", "not_an_rbc_reason"),
        (
            "platelet_over_reservation",
            "platelet",
            "not_a_platelet_reason",
        ),
    ],
)
def test_unknown_reason_fails_loud(
    key: str, component: str, unknown_reason: str
) -> None:
    # Arrange
    row = _row("unknown-reason", "PREOP_OVER_RESERVATION", component=component)
    call = _call(
        "unknown-reason",
        key,
        payload=_payload(key, "unknown-reason", reason=unknown_reason),
    )

    # Act / Assert
    with pytest.raises(PilotReportError, match="unknown .* reservation reason"):
        _build([row], [call])


@pytest.mark.parametrize(
    ("key", "component", "field", "bad_value"),
    [
        ("over_reservation", "red_cell", "reason", None),
        ("over_reservation", "red_cell", "resolved_icd9", None),
        ("over_reservation", "red_cell", "note_resolved", 1),
        ("platelet_over_reservation", "platelet", "category", None),
        ("platelet_over_reservation", "platelet", "pre_op_count_k_ul", 120),
        ("platelet_over_reservation", "platelet", "reserved_units", True),
        ("platelet_over_reservation", "platelet", "clinician_signed", 1),
    ],
)
def test_missing_or_mistyped_required_marker_field_fails_loud(
    key: str,
    component: str,
    field: str,
    bad_value: object,
) -> None:
    # Arrange
    row = _row("bad-field", "PREOP_OVER_RESERVATION", component=component)
    payload = _payload(key, "bad-field")
    if bad_value is None:
        payload.pop(field)
    else:
        payload[field] = bad_value
    call = _call("bad-field", key, payload=payload)

    # Act / Assert
    with pytest.raises(PilotReportError, match=field):
        _build([row], [call])


def test_empty_inputs_return_well_formed_zero_marker_report() -> None:
    # Arrange / Act
    report = _build([], [])

    # Assert
    assert report.provenance.scoped_audit_rows == 0
    assert report.provenance.total_reservation_markers == 0
    assert report.coverage.rbc_note_resolution_rate.rate is None
    assert report.coverage.platelet_category_resolution_rate.rate is None
    assert report.reconciliation.status == "PASS"


def test_each_coverage_rate_carries_within_invisible_limitation() -> None:
    # Arrange / Act
    coverage = _build([], []).coverage

    # Assert
    for limitation in (
        coverage.rbc_note_resolution_rate.limitation,
        coverage.platelet_category_resolution_rate.limitation,
    ):
        assert (
            "Resolution rate among reservation-terminal rows, not population coverage"
            in limitation
        )
        assert "resolved-to-within" in limitation


@pytest.mark.parametrize("missing_name", ["BBA_RUN_ID", "BBA_CODE_VERSION"])
def test_cli_requires_run_id_and_code_version(
    missing_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    module = _load_cli()
    monkeypatch.setenv("BBA_AUDIT_STORE_DIR", str(tmp_path))
    monkeypatch.setenv("BBA_RUN_ID", _RUN_ID)
    monkeypatch.setenv("BBA_CODE_VERSION", _CODE_VERSION)
    monkeypatch.delenv(missing_name)

    # Act
    exit_code = module.main()

    # Assert
    assert exit_code == 1
    assert missing_name in capsys.readouterr().err


def test_cli_rejects_zero_reservation_markers_after_scoped_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    module = _load_cli()
    seen: list[tuple[str | None, str | None]] = []

    class FakeStore:
        def __init__(self, _config: object) -> None:
            pass

        def read_audit_results(
            self, run_id: str | None, code_version: str | None
        ) -> tuple[AuditRow, ...]:
            seen.append((run_id, code_version))
            return (_row("ordinary", "APPROPRIATE"),)

        def read_llm_calls(
            self, run_id: str | None, code_version: str | None
        ) -> tuple[LlmCall, ...]:
            seen.append((run_id, code_version))
            return ()

    monkeypatch.setattr(module, "AuditStore", FakeStore)
    monkeypatch.setenv("BBA_AUDIT_STORE_DIR", str(tmp_path))
    monkeypatch.setenv("BBA_RUN_ID", _RUN_ID)
    monkeypatch.setenv("BBA_CODE_VERSION", _CODE_VERSION)

    # Act
    exit_code = module.main()

    # Assert
    assert exit_code == 1
    assert seen == [(_RUN_ID, _CODE_VERSION), (_RUN_ID, _CODE_VERSION)]
    assert "no reservation activity" in capsys.readouterr().err


def test_cli_writes_deterministic_json_without_mutating_feature_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    module = _load_cli()
    row = _row("reported", "PREOP_OVER_RESERVATION")
    call = _call("reported", "over_reservation")
    original_flag = feature_flags.MSBOS_RESERVATION_ENABLED

    class FakeStore:
        def __init__(self, _config: object) -> None:
            pass

        def read_audit_results(
            self, run_id: str | None, code_version: str | None
        ) -> tuple[AuditRow, ...]:
            return (row,)

        def read_llm_calls(
            self, run_id: str | None, code_version: str | None
        ) -> tuple[LlmCall, ...]:
            return (call,)

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    output = tmp_path / "report.json"
    monkeypatch.setattr(module, "AuditStore", FakeStore)
    monkeypatch.setenv("BBA_AUDIT_STORE_DIR", str(store_dir))
    monkeypatch.setenv("BBA_RUN_ID", _RUN_ID)
    monkeypatch.setenv("BBA_CODE_VERSION", _CODE_VERSION)
    monkeypatch.setenv("BBA_MSBOS_REPORT_OUT", str(output))

    # Act
    exit_code = module.main()
    artifact = json.loads(output.read_text(encoding="utf-8"))
    stdout = capsys.readouterr().out

    # Assert
    assert exit_code == 0
    assert feature_flags.MSBOS_RESERVATION_ENABLED is original_flag
    assert artifact["provenance"]["total_reservation_markers"] == 1
    assert "timestamp" not in artifact["provenance"]
    assert stdout.count("not population coverage") == 2
    # Post-go-live (#167): the blanket HOLD is gone; a clean reconciliation drives
    # a PASS recommendation with precision framed as a tracked follow-up.
    assert "RECOMMENDATION: Returns reconciliation PASS" in stdout
    assert "tracked post-go-live follow-up" in stdout
    assert "HOLD" not in stdout
    assert output.read_text(encoding="utf-8") == json.dumps(
        artifact, indent=2, sort_keys=True, ensure_ascii=False
    )


def test_cli_recommends_investigate_on_returns_double_fire(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange — an over-marker whose committed row is a returns terminal is a
    # double-fire; the report must recommend INVESTIGATE, not PASS.
    module = _load_cli()
    row = _row("double-fire", "RETURNED_NOT_TRANSFUSED")
    call = _call("double-fire", "over_reservation")

    class FakeStore:
        def __init__(self, _config: object) -> None:
            pass

        def read_audit_results(
            self, run_id: str | None, code_version: str | None
        ) -> tuple[AuditRow, ...]:
            return (row,)

        def read_llm_calls(
            self, run_id: str | None, code_version: str | None
        ) -> tuple[LlmCall, ...]:
            return (call,)

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    monkeypatch.setattr(module, "AuditStore", FakeStore)
    monkeypatch.setenv("BBA_AUDIT_STORE_DIR", str(store_dir))
    monkeypatch.setenv("BBA_RUN_ID", _RUN_ID)
    monkeypatch.setenv("BBA_CODE_VERSION", _CODE_VERSION)
    monkeypatch.setenv("BBA_MSBOS_REPORT_OUT", str(tmp_path / "report.json"))

    # Act
    exit_code = module.main()
    stdout = capsys.readouterr().out

    # Assert
    assert exit_code == 0
    assert "RECOMMENDATION: INVESTIGATE — returns double-fire detected" in stdout
    assert "double-fire" in stdout
    assert "PASS" not in stdout


def test_cli_refuses_output_inside_audit_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange — an output path resolving inside the store must be rejected before
    # any read/write so write_text can never clobber a persisted parquet/marker.
    module = _load_cli()
    store_dir = tmp_path / "audit_store"
    store_dir.mkdir()
    sneaky = store_dir / "nested" / "report.json"
    monkeypatch.setenv("BBA_AUDIT_STORE_DIR", str(store_dir))
    monkeypatch.setenv("BBA_RUN_ID", _RUN_ID)
    monkeypatch.setenv("BBA_CODE_VERSION", _CODE_VERSION)
    monkeypatch.setenv("BBA_MSBOS_REPORT_OUT", str(sneaky))

    # Act
    exit_code = module.main()

    # Assert
    assert exit_code == 1
    assert "inside the audit store" in capsys.readouterr().err
    assert not sneaky.exists()


def test_cli_creates_missing_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange — a valid scoped run must create a missing output directory rather
    # than fail with ENOENT.
    module = _load_cli()
    row = _row("reported", "PREOP_OVER_RESERVATION")
    call = _call("reported", "over_reservation")

    class FakeStore:
        def __init__(self, _config: object) -> None:
            pass

        def read_audit_results(
            self, run_id: str | None, code_version: str | None
        ) -> tuple[AuditRow, ...]:
            return (row,)

        def read_llm_calls(
            self, run_id: str | None, code_version: str | None
        ) -> tuple[LlmCall, ...]:
            return (call,)

    store_dir = tmp_path / "audit_store"
    store_dir.mkdir()
    nested = tmp_path / "reports" / "deep" / "report.json"
    monkeypatch.setattr(module, "AuditStore", FakeStore)
    monkeypatch.setenv("BBA_AUDIT_STORE_DIR", str(store_dir))
    monkeypatch.setenv("BBA_RUN_ID", _RUN_ID)
    monkeypatch.setenv("BBA_CODE_VERSION", _CODE_VERSION)
    monkeypatch.setenv("BBA_MSBOS_REPORT_OUT", str(nested))

    # Act
    exit_code = module.main()

    # Assert
    assert exit_code == 0
    assert nested.is_file()
    _ = capsys.readouterr()
