from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_build_review() -> ModuleType:
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "pilot" / "build_review.py"
    )
    spec = importlib.util.spec_from_file_location("pilot_build_review_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render_empty_review(
    module: ModuleType,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    returns_enabled: bool,
    msbos_enabled: bool = False,
) -> bytes:
    bundle = root / "bundle"
    bundle.mkdir(parents=True)
    for name in (
        "BDVST.csv",
        "BDVSTDT.csv",
        "Diagnosis.csv",
        "Lab.csv",
        "Med.csv",
        "IPTSUMOPRT.csv",
        "ICD9CM.csv",
        "BDVSTST.csv",
        "IPDADMPROGRESS.csv",
        "IPDNRFOCUSDT.csv",
    ):
        (bundle / name).write_text("", encoding="utf-8")
    manifest = root / "sample_manifest.csv"
    report = root / "report.csv"
    output = root / "review.html"
    manifest.write_text("", encoding="utf-8")
    report.write_text("", encoding="utf-8")

    monkeypatch.setattr(module, "WORK", root)
    monkeypatch.setattr(module, "BUNDLE", bundle)
    monkeypatch.setattr(module, "MANIFEST", manifest)
    monkeypatch.setattr(module, "DET_REPORT", report)
    monkeypatch.setattr(module, "LLM_REPORT", root / "missing-llm.json")
    monkeypatch.setattr(module, "ICD10_DICT_CSV", root / "missing-icd10.csv")
    monkeypatch.setattr(module, "OUT", output)
    monkeypatch.setattr(module, "RETURNS_LEDGER_ENABLED", returns_enabled)
    monkeypatch.setattr(module, "MSBOS_RESERVATION_PILOT_ENABLED", msbos_enabled)
    module.main()
    return output.read_bytes()


def _render_review_with_rows(
    module: ModuleType,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest_csv: str,
    report_csv: str,
    llm_json: str,
    msbos_enabled: bool = False,
) -> bytes:
    bundle = root / "bundle"
    bundle.mkdir(parents=True)
    for name in (
        "BDVST.csv",
        "BDVSTDT.csv",
        "Diagnosis.csv",
        "Lab.csv",
        "Med.csv",
        "IPTSUMOPRT.csv",
        "ICD9CM.csv",
        "BDVSTST.csv",
        "IPDADMPROGRESS.csv",
        "IPDNRFOCUSDT.csv",
    ):
        (bundle / name).write_text("", encoding="utf-8")
    manifest = root / "sample_manifest.csv"
    report = root / "report.csv"
    llm_report = root / "llm_report.json"
    output = root / "review.html"
    manifest.write_text(manifest_csv, encoding="utf-8")
    report.write_text(report_csv, encoding="utf-8")
    llm_report.write_text(llm_json, encoding="utf-8")
    # A clean response audit, as audit_llm_responses.py leaves it (#239 gate).
    response_audit = root / "llm_response_audit.json"
    response_audit.write_text(
        json.dumps(
            {
                "judge": "all",
                "report_sha256": hashlib.sha256(llm_json.encode("utf-8")).hexdigest(),
                "findings": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "RESPONSE_AUDIT", response_audit)

    monkeypatch.setattr(module, "WORK", root)
    monkeypatch.setattr(module, "BUNDLE", bundle)
    monkeypatch.setattr(module, "MANIFEST", manifest)
    monkeypatch.setattr(module, "DET_REPORT", report)
    monkeypatch.setattr(module, "LLM_REPORT", llm_report)
    monkeypatch.setattr(module, "ICD10_DICT_CSV", root / "missing-icd10.csv")
    monkeypatch.setattr(module, "OUT", output)
    monkeypatch.setattr(module, "RETURNS_LEDGER_ENABLED", True)
    monkeypatch.setattr(module, "MSBOS_RESERVATION_PILOT_ENABLED", msbos_enabled)
    module.main()
    return output.read_bytes()


def test_flag_off_review_omits_returns_presentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_build_review()
    rendered = _render_empty_review(
        module, tmp_path, monkeypatch, returns_enabled=False
    ).decode()
    assert "cls-returned_not_transfused" not in rendered
    assert "<dt>RETURNED_NOT_TRANSFUSED</dt>" not in rendered
    assert "Returned \u2014 not transfused (excluded)" not in rendered
    assert "cls-periop_transfusion_exempt" not in rendered
    assert "<dt>PERIOP_TRANSFUSION_EXEMPT</dt>" not in rendered
    assert "Peri-op transfusion \u2014 exempt (excluded)" not in rendered


def test_flag_on_review_includes_returns_presentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_build_review()
    rendered = _render_empty_review(
        module, tmp_path, monkeypatch, returns_enabled=True
    ).decode()
    assert "cls-returned_not_transfused" in rendered
    assert "<dt>RETURNED_NOT_TRANSFUSED</dt>" in rendered
    assert "Returned \u2014 not transfused (excluded)" in rendered
    assert "cls-periop_transfusion_exempt" in rendered
    assert "<dt>PERIOP_TRANSFUSION_EXEMPT</dt>" in rendered
    assert "Peri-op transfusion \u2014 exempt (excluded)" in rendered


def test_flag_off_review_omits_operation_unresolved_glossary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Flag-off byte parity: the T3 glossary entry must not leak into review.html
    # when the MSBOS reservation pilot is disabled.
    module = _load_build_review()
    rendered = _render_empty_review(
        module, tmp_path, monkeypatch, returns_enabled=True, msbos_enabled=False
    ).decode()
    assert "<dt>operation_unresolved</dt>" not in rendered


def test_flag_on_review_includes_operation_unresolved_glossary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_build_review()
    rendered = _render_empty_review(
        module, tmp_path, monkeypatch, returns_enabled=True, msbos_enabled=True
    ).decode()
    assert "<dt>operation_unresolved</dt>" in rendered


def test_preop_over_reservation_pill_defined_unconditional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_build_review()
    rendered = _render_empty_review(
        module, tmp_path, monkeypatch, returns_enabled=False
    ).decode()
    assert ".cls-preop_over_reservation" in rendered
    assert "var(--err-bg)" in rendered


def test_returns_pill_classes_defined_when_returns_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_build_review()
    rendered = _render_empty_review(
        module, tmp_path, monkeypatch, returns_enabled=True
    ).decode()
    assert ".cls-returned_not_transfused" in rendered
    assert ".cls-periop_transfusion_exempt" in rendered
    assert "var(--neu-bg)" in rendered


def test_summary_pill_wrapping_js_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_build_review()
    rendered = _render_empty_review(
        module, tmp_path, monkeypatch, returns_enabled=True
    ).decode()
    assert "document.querySelector('table')" not in rendered
    assert "var sentinels" not in rendered


def test_summary_table_focus_overflow_and_kbd_styles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_build_review()
    rendered = _render_empty_review(
        module, tmp_path, monkeypatch, returns_enabled=True
    ).decode()
    assert ":focus-visible" in rendered
    assert "outline-offset: 2px" in rendered
    assert ".table-scroll" in rendered
    assert "overflow-x: auto" in rendered
    assert "kbd-dismiss" in rendered
    assert "0.7rem" not in rendered


def test_summary_pills_and_mismatch_rendered_server_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_build_review()
    rendered = _render_review_with_rows(
        module,
        tmp_path,
        monkeypatch,
        manifest_csv="HN,REQNO,AN\nA<B,R1,AN1\nHN2,R2,AN2\n",
        report_csv=(
            "reqno,classification\nR1,POTENTIALLY_INAPPROPRIATE\nR2,APPROPRIATE\n"
        ),
        llm_json=json.dumps(
            [
                {
                    "reqno": "R1",
                    "llm_final": {
                        "final_classification": "APPROPRIATE",
                        "confidence": 0.91,
                        "model": "test",
                        "review_reason": "model_verdict",
                        "indications": [],
                        "negative_evidence": [],
                        "reasoning_en": "x",
                        "reasoning_th": "x",
                    },
                }
            ]
        ),
    ).decode()
    # R1 is a major mismatch and is shaded server-side; R2 has no LLM row,
    # so it is not shaded and its LLM cell remains plain sentinel text.
    assert "<span class='cls cls-potentially_inappropriate'>" in rendered
    assert "<span class='cls cls-appropriate'>" in rendered
    assert rendered.count("<tr class='verdict-mismatch'>") == 1
    assert "A&lt;B" in rendered
    assert "(LLM not run)" in rendered


def test_msbos_flag_off_report_columns_do_not_change_review_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_build_review()
    manifest_csv = "HN,REQNO,AN\nA<B,R1,AN1\n"
    without_columns = _render_review_with_rows(
        module,
        tmp_path / "without",
        monkeypatch,
        manifest_csv=manifest_csv,
        report_csv="reqno,classification,component\nR1,RETURNED_NOT_TRANSFUSED,platelet\n",
        llm_json="[]",
    )
    with_columns = _render_review_with_rows(
        module,
        tmp_path / "with",
        monkeypatch,
        manifest_csv=manifest_csv,
        report_csv=(
            "reqno,classification,component,msbos_reserved_units,msbos_token,"
            "msbos_recommended_units,msbos_reason,msbos_is_over,"
            "msbos_resolved_icd9,msbos_reference_hash,msbos_plt_category,"
            "msbos_plt_count_k_ul,msbos_plt_over_above_per_ul,"
            "msbos_plt_clinician_signed\n"
            "R1,RETURNED_NOT_TRANSFUSED,platelet,2,,,over_neuraxial,True,"
            "0199,hash,neuraxial,120.0,100000,True\n"
        ),
        llm_json="[]",
    )

    assert with_columns == without_columns
    rendered = with_columns.decode()
    assert "MSBOS" not in rendered
    assert "cls-msbos-" not in rendered
    assert "above tariff" not in rendered
    assert "<dt>above</dt>" not in rendered
    assert "PLT 120 > 100" not in rendered
    assert "msbos-counts" not in rendered


def test_msbos_flag_on_renders_summary_cases_counts_glossary_and_css(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_build_review()
    rendered = _render_review_with_rows(
        module,
        tmp_path,
        monkeypatch,
        manifest_csv=(
            "HN,REQNO,AN\n"
            "A<B,R1,AN1\n"
            "HN2,R2,AN2\n"
            "HN3,R3,AN3\n"
            "HN4,R4,AN4\n"
            "HN5,R5,AN5\n"
            "HN6,R6,AN6\n"
        ),
        report_csv=(
            "reqno,classification,component,msbos_reserved_units,msbos_token,"
            "msbos_recommended_units,msbos_reason,msbos_is_over,"
            "msbos_resolved_icd9,msbos_reference_hash,msbos_plt_category,"
            "msbos_plt_count_k_ul,msbos_plt_over_above_per_ul,"
            "msbos_plt_clinician_signed,returns_units_transfused,"
            "returns_units_returned,returns_units_total\n"
            "R1,RETURNED_NOT_TRANSFUSED,red_cell,3,G/M,2,over_gm_excess,True,"
            "0139,hash,,,,,,\n"
            "R2,RETURNED_NOT_TRANSFUSED,red_cell,1,G/M,2,within_recommendation,"
            "False,0139,hash,,,,,,\n"
            "R3,APPROPRIATE,red_cell,3,G/M,2,over_gm_excess,True,0139,hash,,,,,,\n"
            "R4,RETURNED_NOT_TRANSFUSED,platelet,2,,,over_neuraxial,True,0199,"
            "hash,neuraxial,120.0,100000,True,,\n"
            "R5,RETURNED_NOT_TRANSFUSED,red_cell,0,,0,reservation_lookup_miss,"
            "False,,hash,,,,,,\n"
            "R6,PERIOP_TRANSFUSION_EXEMPT,red_cell,2,G/M,2,"
            "within_recommendation,False,0139,hash,,,,,0,1,2\n"
        ),
        llm_json="[]",
        msbos_enabled=True,
    ).decode()

    assert "<th>MSBOS</th>" in rendered
    assert "<span class='cls cls-msbos-warn'>3 vs G/M 2</span>" in rendered
    assert "<span class='cls cls-msbos-ok'>within</span>" in rendered
    assert (
        "<td><span class='cls cls-appropriate'>Appropriate</span></td>"
        "<td>(LLM not run)</td><td>—</td>" in rendered
    )
    assert "MSBOS reservation: Reserved 3; MSBOS tariff G/M 2" in rendered
    assert "<span class='cls cls-msbos-warn'>unlinked</span>" in rendered
    assert "Reservation detail lines not linked (unlinked)" in rendered
    # The ledger never records status 5 (transfused), so the page states
    # returned units out of the total instead of a transfused count that is
    # always 0 (review of platelet case 68012561, 2026-09-23).
    assert "Reserved 2; MSBOS tariff G/M 2; 1 of 2 units returned" in rendered
    assert " transfused, " not in rendered
    assert "<span class='cls cls-msbos-warn'>PLT 120 > 100</span>" in rendered
    assert (
        "Reserved 2u platelets; pre-op count 120k/uL > neuraxial cutoff 100k/uL"
        in rendered
    )
    # R4 is an annotated platelet over row and is counted alongside RBC returns.
    assert (
        "Returned (4): 2 above / 1 within / 0 within-ceiling / 1 unresolved" in rendered
    )
    assert (
        "Peri-op exempt (1): 0 above / 1 within / 0 within-ceiling / 0 unresolved"
        in rendered
    )
    assert "MSBOS reservation: </div>" not in rendered
    assert "<dt>above</dt>" in rendered
    assert "INFORMATIONAL" in rendered
    # Corrected glossary (#201): on declared pre-op rows MSBOS screening CAN
    # reclassify (spec #194/#196); the old "never changes classification"
    # claim is gone.
    assert "MSBOS screening CAN change the classification" in rendered
    assert "anticipated hemorrhage, case cancellation, or emergency status" in rendered
    assert ".cls-msbos-warn" in rendered
    assert ".cls-msbos-ok" in rendered
    assert "A&lt;B" in rendered


def test_msbos_flag_on_rejects_duplicate_reqno_component_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_build_review()
    with pytest.raises(
        ValueError,
        match=r"duplicate REQNO in report scope.*'R1'",
    ):
        _render_review_with_rows(
            module,
            tmp_path,
            monkeypatch,
            manifest_csv="HN,REQNO,AN\n",
            report_csv=(
                "reqno,classification,component\n"
                "R1,RETURNED_NOT_TRANSFUSED,red_cell\n"
                "R1,RETURNED_NOT_TRANSFUSED,platelet\n"
            ),
            llm_json="[]",
            msbos_enabled=True,
        )


def _returns_det(reason: str, **extra: str) -> dict[str, str]:
    return {
        "classification": "RETURNED_NOT_TRANSFUSED",
        "msbos_reason": reason,
        "msbos_reserved_units": "3",
        "msbos_recommended_units": "2",
        "msbos_token": "G/M",
        **extra,
    }


def test_msbos_summary_pill_maps_every_reachable_reason() -> None:
    # Pin the full reason -> (text, color) mapping so no reachable plain-evaluator
    # reason silently falls through to em-dash / the wrong colour role.
    module = _load_build_review()
    cases = {
        "over_gm_excess": ("3 vs G/M 2", "cls-msbos-warn"),
        "over_none": ("3 vs none 0", "cls-msbos-warn"),
        "over_type_and_screen_crossmatched": ("T/S; 3u reserved", "cls-msbos-warn"),
        "within_recommendation": ("within", "cls-msbos-ok"),
        "type_and_screen_screen_only": ("within", "cls-msbos-ok"),
        "ambiguous_code": ("code unresolved", "cls-msbos-warn"),
        "unresolved_code": ("code unresolved", "cls-msbos-warn"),
        "ambiguous_planned_op": ("op unresolved", "cls-msbos-warn"),
        "no_planned_op": ("op unresolved", "cls-msbos-warn"),
        "operation_unresolved": ("op unresolved", "cls-msbos-warn"),
        "reservation_lookup_miss": ("unlinked", "cls-msbos-warn"),
    }
    for reason, (text, pill_class) in cases.items():
        pill = module._msbos_summary_pill(_returns_det(reason))
        assert pill == f"<span class='cls {pill_class}'>{text}</span>", reason

    # Blank reason and non-returns rows collapse to a plain em-dash (no pill).
    assert module._msbos_summary_pill(_returns_det("")) == "—"
    assert (
        module._msbos_summary_pill(
            {"classification": "APPROPRIATE", "msbos_reason": "over_gm_excess"}
        )
        == "—"
    )


def test_msbos_case_line_maps_every_reachable_reason() -> None:
    module = _load_build_review()
    expected = {
        "over_gm_excess": "Reserved 3; MSBOS tariff G/M 2",
        "over_none": "Reserved 3; MSBOS tariff none 0",
        "over_type_and_screen_crossmatched": "Reserved 3; MSBOS tariff T/S",
        "within_recommendation": "Reserved 3; MSBOS tariff G/M 2",
        "type_and_screen_screen_only": "Reserved 3; MSBOS tariff T/S",
        "ambiguous_code": "MSBOS operation code unresolved",
        "unresolved_code": "MSBOS operation code unresolved",
        "ambiguous_planned_op": "MSBOS planned operation unresolved",
        "no_planned_op": "MSBOS planned operation unresolved",
        "operation_unresolved": "MSBOS planned operation unresolved",
        "reservation_lookup_miss": "Reservation detail lines not linked (unlinked)",
    }
    for reason, text in expected.items():
        line = module._msbos_case_line(_returns_det(reason), "RETURNED_NOT_TRANSFUSED")
        assert line == text, reason

    # Exempt rows append transfused/returned counts from the returns columns.
    exempt = module._msbos_case_line(
        _returns_det(
            "over_gm_excess",
            returns_units_transfused="0",
            returns_units_returned="2",
            returns_units_total="3",
        ),
        "PERIOP_TRANSFUSION_EXEMPT",
    )
    assert exempt == "Reserved 3; MSBOS tariff G/M 2; 2 of 3 units returned"


def _platelet_returns_det(reason: str, **extra: str) -> dict[str, str]:
    return {
        "classification": "RETURNED_NOT_TRANSFUSED",
        "component": "platelet",
        "msbos_reason": reason,
        "msbos_reserved_units": "2",
        "msbos_plt_category": "neuraxial",
        "msbos_plt_count_k_ul": "120.0",
        "msbos_plt_over_above_per_ul": "100000",
        **extra,
    }


def test_fmt_plt_k_preserves_non_integral_values() -> None:
    module = _load_build_review()

    assert module._fmt_plt_k(120.0) == "120"
    assert module._fmt_plt_k(120.5) == "120.5"
    assert module._fmt_plt_k("") == ""
    assert module._fmt_plt_k(None) == ""
    assert module._fmt_plt_k("x") == ""


def test_fmt_plt_cutoff_k_is_crash_safe() -> None:
    module = _load_build_review()

    assert module._fmt_plt_cutoff_k(100000) == "100"
    assert module._fmt_plt_cutoff_k(80000) == "80"
    assert module._fmt_plt_cutoff_k("") == ""
    assert module._fmt_plt_cutoff_k(None) == ""
    assert module._fmt_plt_cutoff_k("x") == ""


def test_msbos_platelet_summary_pill_maps_every_reason_without_rbc_fallthrough() -> (
    None
):
    module = _load_build_review()
    cases = {
        "over_major_non_neuraxial": ("PLT 120 > 100", "cls-msbos-warn"),
        "over_neuraxial": ("PLT 120 > 100", "cls-msbos-warn"),
        "over_cardiac_cpb": ("PLT 120 > 100", "cls-msbos-warn"),
        "within_major_non_neuraxial": ("within", "cls-msbos-ok"),
        "within_neuraxial": ("within", "cls-msbos-ok"),
        "within_cardiac_cpb": ("within", "cls-msbos-ok"),
        "no_reserved_units": ("within", "cls-msbos-ok"),
        "missing_pre_op_count": ("count missing", "cls-msbos-warn"),
        "uncategorised_procedure": ("op uncategorised", "cls-msbos-warn"),
        "ambiguous_category": ("category ambiguous", "cls-msbos-warn"),
        "no_planned_op": ("op unresolved", "cls-msbos-warn"),
        "ambiguous_planned_op": ("op unresolved", "cls-msbos-warn"),
        "reservation_lookup_miss": ("unlinked", "cls-msbos-warn"),
    }
    for reason, (text, pill_class) in cases.items():
        pill = module._msbos_summary_pill(_platelet_returns_det(reason))
        assert pill == f"<span class='cls {pill_class}'>{text}</span>", reason
        expected_bucket = (
            "above"
            if reason.startswith("over_")
            else "within"
            if reason.startswith("within_") or reason == "no_reserved_units"
            else "unresolved"
        )
        assert module._msbos_reason_bucket(reason) == expected_bucket

    assert (
        module._msbos_summary_pill(_platelet_returns_det("within_recommendation"))
        == "—"
    )

    category_cases = {
        "over_major_non_neuraxial": (
            "major_non_neuraxial",
            "80000",
            "PLT 120 > 80",
        ),
        "over_cardiac_cpb": ("cardiac_cpb", "100000", "PLT 120 > 100"),
    }
    for reason, (category, cutoff, text) in category_cases.items():
        pill = module._msbos_summary_pill(
            _platelet_returns_det(
                reason,
                msbos_plt_category=category,
                msbos_plt_over_above_per_ul=cutoff,
            )
        )
        assert pill == f"<span class='cls cls-msbos-warn'>{text}</span>", reason


def test_msbos_platelet_case_line_maps_every_reason() -> None:
    module = _load_build_review()
    expected = {
        "over_neuraxial": (
            "Reserved 2u platelets; pre-op count 120k/uL > neuraxial cutoff 100k/uL"
        ),
        "within_neuraxial": (
            "Reserved 2u platelets; pre-op count 120k/uL within neuraxial cutoff 100k/uL"
        ),
        "no_reserved_units": "No platelet units reserved",
        "missing_pre_op_count": "Platelet pre-op count missing",
        "uncategorised_procedure": "MSBOS platelet category could not be resolved",
        "ambiguous_category": "MSBOS platelet category ambiguous",
        "no_planned_op": "MSBOS planned operation unresolved",
        "ambiguous_planned_op": "MSBOS planned operation unresolved",
        "reservation_lookup_miss": "Reservation detail lines not linked (unlinked)",
    }
    for reason, text in expected.items():
        line = module._msbos_case_line(
            _platelet_returns_det(reason), "RETURNED_NOT_TRANSFUSED"
        )
        assert line == text, reason

    category_cases = {
        "over_major_non_neuraxial": (
            "major_non_neuraxial",
            "80000",
            "Reserved 2u platelets; pre-op count 120k/uL > "
            "major-non-neuraxial cutoff 80k/uL",
        ),
        "over_cardiac_cpb": (
            "cardiac_cpb",
            "100000",
            "Reserved 2u platelets; pre-op count 120k/uL > cardiac-CPB cutoff 100k/uL",
        ),
    }
    for reason, (category, cutoff, text) in category_cases.items():
        line = module._msbos_case_line(
            _platelet_returns_det(
                reason,
                msbos_plt_category=category,
                msbos_plt_over_above_per_ul=cutoff,
            ),
            "RETURNED_NOT_TRANSFUSED",
        )
        assert line == text, reason

    exempt = module._msbos_case_line(
        _platelet_returns_det(
            "within_neuraxial",
            returns_units_transfused="0",
            returns_units_returned="1",
            returns_units_total="2",
        ),
        "PERIOP_TRANSFUSION_EXEMPT",
    )
    assert exempt.endswith("; 1 of 2 units returned")


# ---------------------------------------------------------------------------
# Component-aware rendering (platelet orders on the shared review page)
# ---------------------------------------------------------------------------

_PLT_REPORT = (
    "reqno,classification,rationale,component,platelet_count_k_ul,"
    "platelet_freshness,hb_value_g_dl,cohort_label,cohort_threshold\n"
    "P1,NEEDS_REVIEW,plt_defer_llm,platelet,11.0,fresh,,,\n"
)
_RBC_REPORT = (
    "reqno,classification,rationale,component,platelet_count_k_ul,"
    "platelet_freshness,hb_value_g_dl,cohort_label,cohort_threshold\n"
    "R1,APPROPRIATE,hb_lt_7_universal,red_cell,,,6.5,cohort_unknown,7.0\n"
)


def test_platelet_case_shows_the_count_not_the_hb_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A reviewer judging a platelet order needs the trigger count; the Hb value,
    # Hb lookup anchor and Hb cohort threshold are RBC concepts that read as
    # missing data ("—", "n/a") on a platelet case.
    module = _load_build_review()
    rendered = _render_review_with_rows(
        module,
        tmp_path,
        monkeypatch,
        manifest_csv="HN,REQNO,AN,component\nHN1,P1,AN1,platelet\n",
        report_csv=_PLT_REPORT,
        llm_json="[]",
    ).decode()

    assert "Platelet count @ order:" in rendered
    assert "11,000 /µL" in rendered
    assert "(fresh)" in rendered
    assert "Hb @ anchor" not in rendered
    assert "Hb lookup anchor" not in rendered
    assert "Platelet count history" in rendered
    assert "Hb history" not in rendered
    assert "data-component='platelet'" in rendered


def test_page_title_is_component_neutral_when_platelets_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_build_review()
    rendered = _render_review_with_rows(
        module,
        tmp_path,
        monkeypatch,
        manifest_csv=("HN,REQNO,AN,component\nHN1,P1,AN1,platelet\nHN2,R1,AN2,rbc\n"),
        report_csv=_PLT_REPORT + _RBC_REPORT.split("\n", 1)[1],
        llm_json="[]",
    ).decode()

    assert "KCMH RBC Order Appropriateness Audit" not in rendered
    assert "Blood Component Order Appropriateness Audit" in rendered
    assert "1 RBC, 1 platelet" in rendered
    # Mixed pages get a component column, a count column and a filter.
    assert "<th>Comp</th>" in rendered and "<th>Plt (k/µL)</th>" in rendered
    assert "id='filter-component'" in rendered
    # The RBC case on the same page keeps its Hb strip.
    assert "Hb @ anchor" in rendered


def test_rbc_only_page_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # RBC-only reviews already in clinicians' hands must not change shape.
    module = _load_build_review()
    rendered = _render_review_with_rows(
        module,
        tmp_path,
        monkeypatch,
        manifest_csv="HN,REQNO,AN,component\nHN2,R1,AN2,rbc\n",
        report_csv=_RBC_REPORT,
        llm_json="[]",
    ).decode()

    assert "KCMH RBC Order Appropriateness Audit — Human Review" in rendered
    assert "Hb @ anchor" in rendered and "Hb history" in rendered
    assert "<th>Comp</th>" not in rendered
    # No component CONTROL on an RBC-only page (the shared filter script may
    # still name the element id).
    assert "id='filter-component'" not in rendered
    assert "data-component='" not in rendered


def test_platelet_history_window_matches_the_gate_bounds() -> None:
    # lookup_platelet() and the LLM bundle's _filter_platelet() use a STRICT
    # 7-day lower bound and an inclusive order-time upper bound. The page says
    # it shows the same window, so a count exactly 7 days old (which the gate
    # ignores) must not be presented to the reviewer as decision evidence.
    from datetime import datetime, timedelta

    module = _load_build_review()
    anchor = datetime(2025, 3, 8, 9, 0, tzinfo=module.TZ_LOCAL)

    assert module._in_platelet_window(anchor, anchor) is True
    assert (
        module._in_platelet_window(
            anchor - timedelta(days=7) + timedelta(minutes=1), anchor
        )
        is True
    )
    assert module._in_platelet_window(anchor - timedelta(days=7), anchor) is False
    assert module._in_platelet_window(anchor + timedelta(minutes=1), anchor) is False
    assert module._in_platelet_window(None, anchor) is False
    assert module._in_platelet_window(anchor, None) is False


def test_keyboard_navigation_skips_filtered_out_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Codex P2 on #236: with a filter active (component or mismatches), j/k
    # stepped through the full case list and scrolled to display:none sections,
    # so navigation appeared to stall on interleaved pages.
    module = _load_build_review()
    rendered = _render_review_with_rows(
        module,
        tmp_path,
        monkeypatch,
        manifest_csv="HN,REQNO,AN,component\nHN2,R1,AN2,rbc\n",
        report_csv=_RBC_REPORT,
        llm_json="[]",
    ).decode()

    assert "caseEls[next].style.display === 'none'" in rendered
    assert "Math.min(activeIdx + 1, caseEls.length - 1)" not in rendered


def test_mismatch_filter_reads_the_flag_from_the_nav_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Codex P2 on #236 (and a latent bug on main): the [!] / [!!] mismatch flag
    # is rendered only inside the nav link, so a filter that looked for it
    # inside the case section hid EVERY case when "Mismatches only" was ticked.
    module = _load_build_review()
    rendered = _render_review_with_rows(
        module,
        tmp_path,
        monkeypatch,
        manifest_csv="HN,REQNO,AN,component\nHN2,R1,AN2,rbc\n",
        report_csv=_RBC_REPORT,
        llm_json="[]",
    ).decode()

    assert "sec.querySelector('.nav-flag')" not in rendered
    assert "link.querySelector('.nav-flag')" in rendered
    # One implementation serves both controls.
    assert "window.filterMismatches = applyFilters;" in rendered
    assert "window.filterComponent = applyFilters;" in rendered


def test_excluded_platelet_order_keeps_its_component_from_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Codex P2 on #236: an order excluded by build_audit_orders has a sparse
    # report row with no component column value. Without the manifest it was
    # rendered and counted as RBC, so the platelet filter hid it and the page
    # header miscounted (308 RBC instead of 304 on the hematology sandbox).
    module = _load_build_review()
    rendered = _render_review_with_rows(
        module,
        tmp_path,
        monkeypatch,
        manifest_csv="HN,REQNO,AN,component\nHN1,P9,AN1,platelet\n",
        report_csv=(
            "reqno,classification,rationale,component\nP9,excluded,hemoglobinopathy,\n"
        ),
        llm_json="[]",
    ).decode()

    assert "data-component='platelet'" in rendered
    assert "(0 RBC, 1 platelet)" in rendered
    assert "Platelet count history" in rendered


def test_platelet_page_explains_platelet_codes_and_verdicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Codex P2 on #236: the glossary defined every verdict in Hb terms and left
    # plt_* rationale codes as raw slugs, which is wrong or unreadable for the
    # platelet cases the page now presents.
    module = _load_build_review()
    rendered = _render_review_with_rows(
        module,
        tmp_path,
        monkeypatch,
        manifest_csv="HN,REQNO,AN,component\nHN1,P1,AN1,platelet\n",
        report_csv=_PLT_REPORT,
        llm_json="[]",
    ).decode()

    assert "<dt>plt_defer_llm</dt>" in rendered
    assert "<dt>plt_lt_10_heme_prophylaxis</dt>" in rendered
    assert "Platelet: count below the policy threshold" in rendered
    assert "non-RBC product" not in rendered
    # The verdict box resolves the code to its label instead of a bare slug.
    assert "indication judged by the LLM" in rendered


def test_platelet_page_explains_the_count_trend_review_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Issue #237: a reviewer who opens a row floored by the count-trend
    # guardrail must learn WHY it is in their queue (the projection did not
    # support "expected below 10,000"), not read a bare slug.
    module = _load_build_review()
    rendered = _render_review_with_rows(
        module,
        tmp_path,
        monkeypatch,
        manifest_csv="HN,REQNO,AN,component\nHN1,P1,AN1,platelet\n",
        report_csv=_PLT_REPORT,
        llm_json="[]",
    ).decode()

    assert (
        "straight-line projection"
        in module._REVIEW_REASON_LABELS["platelet_trend_unsupported"]
    )
    assert rendered.count("<dt>platelet_trend_unsupported</dt>") == 1
    assert rendered.count("<dt>platelet_llm_overclear_suspect</dt>") == 1


def test_initial_keyboard_lookup_ignores_hidden_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Codex P2 on #236: before any case is active, findActiveByScroll scanned
    # hidden sections too; a display:none section has a zero rect and was picked.
    module = _load_build_review()
    rendered = _render_review_with_rows(
        module,
        tmp_path,
        monkeypatch,
        manifest_csv="HN,REQNO,AN,component\nHN2,R1,AN2,rbc\n",
        report_csv=_RBC_REPORT,
        llm_json="[]",
    ).decode()

    assert "if (caseEls[i].style.display === 'none') continue;" in rendered
    # With every case filtered out the lookup must not fall back to case 0,
    # or `e` / `x` would act on a hidden case.
    assert "return firstVisible;" in rendered
    assert "firstVisible < 0 ? 0" not in rendered


def test_applying_a_filter_drops_the_active_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Codex P2 on #236: a case active before filtering stayed active after it
    # was hidden, so `e` marked a hidden case reviewed and `x` expanded it.
    module = _load_build_review()
    rendered = _render_review_with_rows(
        module,
        tmp_path,
        monkeypatch,
        manifest_csv="HN,REQNO,AN,component\nHN2,R1,AN2,rbc\n",
        report_csv=_RBC_REPORT,
        llm_json="[]",
    ).decode()

    apply_body = rendered.split("function applyFilters()", 1)[1].split(
        "window.filterMismatches", 1
    )[0]
    assert "activeIdx = -1;" in apply_body
    # Same function scope: the handler state is declared in the script that
    # defines applyFilters, not in a separate block.
    assert rendered.count("var activeIdx = -1;") == 1


# --- Issue #239: the page is not built from unaudited LLM responses ------------
# A clinician found a self-contradicting answer on case 1 of a rendered page.
# audit_llm_responses.py catches those, but only if it actually runs: the page
# builder refuses LLM verdicts that have no clean, full audit OF THIS REPORT.

_LLM_ENTRY = (
    '[{"reqno": "R1", "audit_id": "a1", "llm_final": '
    '{"final_classification": "APPROPRIATE", "model": "claude-sonnet-5"}}]'
)


def _audit_json(report_json: str, *, judge: str = "all", findings: str = "[]") -> str:
    digest = hashlib.sha256(report_json.encode("utf-8")).hexdigest()
    return (
        f'{{"judge": "{judge}", "report_sha256": "{digest}", "findings": {findings}}}'
    )


def _gate_paths(
    tmp_path: Path, llm_json: str, audit_json: str | None
) -> tuple[Path, Path]:
    report = tmp_path / "llm_report.json"
    audit = tmp_path / "llm_response_audit.json"
    report.write_text(llm_json, encoding="utf-8")
    if audit_json is not None:
        audit.write_text(audit_json, encoding="utf-8")
    return report, audit


def test_llm_verdicts_without_an_audit_block_the_page(tmp_path: Path) -> None:
    module = _load_build_review()
    report, audit = _gate_paths(tmp_path, _LLM_ENTRY, None)

    blocker = module.response_audit_blocker(report, audit)

    assert blocker is not None
    assert "audit_llm_responses.py" in blocker


def test_high_finding_blocks_the_page(tmp_path: Path) -> None:
    module = _load_build_review()
    high = '[{"reqno": "R1", "code": "consortium_label_contradiction", "severity": "HIGH"}]'
    report, audit = _gate_paths(
        tmp_path, _LLM_ENTRY, _audit_json(_LLM_ENTRY, findings=high)
    )

    blocker = module.response_audit_blocker(report, audit)

    assert blocker is not None
    assert "1 HIGH" in blocker


def test_audit_of_a_different_report_does_not_unlock_the_page(tmp_path: Path) -> None:
    # A re-run merges fresh records into llm_report.json. An audit that read the
    # OLD report says nothing about them, even if it finished later (an mtime
    # check would accept it), so the audit is bound to the report's contents.
    module = _load_build_review()
    rerun = _LLM_ENTRY.replace("APPROPRIATE", "INAPPROPRIATE")
    report, audit = _gate_paths(tmp_path, rerun, _audit_json(_LLM_ENTRY))

    blocker = module.response_audit_blocker(report, audit)

    assert blocker is not None
    assert "different llm_report.json" in blocker


def test_clean_full_audit_of_this_report_lets_the_page_build(tmp_path: Path) -> None:
    module = _load_build_review()
    medium = '[{"reqno": "R1", "code": "label_before_reasoning", "severity": "MEDIUM"}]'
    report, audit = _gate_paths(
        tmp_path, _LLM_ENTRY, _audit_json(_LLM_ENTRY, findings=medium)
    )

    assert module.response_audit_blocker(report, audit) is None


def test_runs_without_model_responses_need_no_audit(tmp_path: Path) -> None:
    # No LLM leg, or only deterministic rows the LLM leg persists under
    # llm_final (MSBOS over-reservation, injection filter): nothing to audit,
    # and those runs rendered before this gate existed.
    module = _load_build_review()
    deterministic = (
        '[{"reqno": "R1", "audit_id": "a1", "llm_final": '
        '{"final_classification": "PREOP_OVER_RESERVATION", "model": "msbos-reservation"}}]'
    )
    for llm_json in ("[]", deterministic):
        report, audit = _gate_paths(tmp_path, llm_json, None)
        assert module.response_audit_blocker(report, audit) is None
    assert module.response_audit_blocker(tmp_path / "absent.json", audit) is None


def test_a_missing_llm_result_still_needs_the_audit(tmp_path: Path) -> None:
    module = _load_build_review()
    dropped = '[{"reqno": "R1", "audit_id": "a1", "llm_final": null}]'
    report, audit = _gate_paths(tmp_path, dropped, None)

    assert module.response_audit_blocker(report, audit) is not None


def test_main_refuses_to_render_and_the_override_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_build_review()
    report, audit = _gate_paths(tmp_path, _LLM_ENTRY, None)
    monkeypatch.setattr(module, "LLM_REPORT", report)
    monkeypatch.setattr(module, "RESPONSE_AUDIT", audit)
    monkeypatch.delenv("BBA_PILOT_ALLOW_UNAUDITED", raising=False)

    with pytest.raises(SystemExit) as refused:
        module.enforce_response_audit()
    assert "BBA_PILOT_ALLOW_UNAUDITED=1" in str(refused.value)

    monkeypatch.setenv("BBA_PILOT_ALLOW_UNAUDITED", "1")
    module.enforce_response_audit()  # explicit operator override: no exit


@pytest.mark.parametrize("judge", ["off", "candidates", "all"])
def test_the_code_check_audit_is_what_unlocks_the_page(
    tmp_path: Path, judge: str
) -> None:
    # User ruling 2026-09-21: the model consortium is a dev-test review run from
    # Claude Code, not a production step, so the page does not wait for it. The
    # gate is the code audit: it exists, it read THIS report, it has no HIGH.
    module = _load_build_review()
    report, audit = _gate_paths(
        tmp_path, _LLM_ENTRY, _audit_json(_LLM_ENTRY, judge=judge)
    )

    assert module.response_audit_blocker(report, audit) is None


def test_an_audit_file_from_before_the_report_digest_does_not_unlock_the_page(
    tmp_path: Path,
) -> None:
    module = _load_build_review()
    report, audit = _gate_paths(tmp_path, _LLM_ENTRY, "[]")

    assert module.response_audit_blocker(report, audit) is not None


def test_the_page_renders_the_report_the_gate_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Local Codex review of #241: the gate read llm_report.json, then main()
    # read it again later. A re-run landing between the two reads was rendered
    # without ever being audited. main() now renders what the gate returned.
    module = _load_build_review()
    validated = [
        {
            "reqno": "R1",
            "llm_final": {
                "final_classification": "INAPPROPRIATE",
                "confidence": 0.9,
                "model": "claude-sonnet-5",
                "review_reason": None,
                "indications": [],
                "negative_evidence": [],
                "reasoning_en": "VALIDATED-REPORT-REASONING",
                "reasoning_th": "x",
            },
        }
    ]
    monkeypatch.setattr(module, "enforce_response_audit", lambda: validated)

    rendered = _render_review_with_rows(
        module,
        tmp_path,
        monkeypatch,
        manifest_csv="HN,REQNO,AN\nHN1,R1,AN1\n",
        report_csv="reqno,classification\nR1,NEEDS_REVIEW\n",
        llm_json=json.dumps(
            [
                {
                    "reqno": "R1",
                    "llm_final": {
                        **validated[0]["llm_final"],
                        "reasoning_en": "UNAUDITED-RERUN",
                    },
                }
            ]
        ),
    ).decode()

    assert "VALIDATED-REPORT-REASONING" in rendered
    assert "UNAUDITED-RERUN" not in rendered


def test_the_gate_hands_back_the_entries_it_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_build_review()
    report, audit = _gate_paths(tmp_path, _LLM_ENTRY, _audit_json(_LLM_ENTRY))
    monkeypatch.setattr(module, "LLM_REPORT", report)
    monkeypatch.setattr(module, "RESPONSE_AUDIT", audit)

    assert module.enforce_response_audit() == json.loads(_LLM_ENTRY)

    monkeypatch.setattr(module, "LLM_REPORT", tmp_path / "absent.json")
    assert module.enforce_response_audit() == []


def test_returned_cell_states_units_returned_out_of_units_issued() -> None:
    # Case 68012561: one bag came back at 10:05, then five more were issued and
    # given. A bare return time read as "this order was returned".
    module = _load_build_review()
    det = {
        "returned_blood_datetime_local": "2025-02-26 10:05:24",
        "returns_units_returned": "1",
        "returns_units_total": "6",
    }

    assert (
        module._returned_display(det, [])
        == "2025-02-26 10:05:24 (1 of 6 units returned)"
    )


def test_returned_cell_without_ledger_counts_keeps_the_time_alone() -> None:
    module = _load_build_review()
    det = {"returned_blood_datetime_local": "2025-02-26 10:05:24"}

    assert module._returned_display(det, []) == "2025-02-26 10:05:24"
    assert module._returned_display({}, []) == "—"


def test_platelet_page_explains_the_specialist_target_review_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ruling 2026-09-23: the reviewer must see that the order failed policy but
    # followed a consultant's documented target, not a bare slug.
    module = _load_build_review()
    rendered = _render_review_with_rows(
        module,
        tmp_path,
        monkeypatch,
        manifest_csv="HN,REQNO,AN,component\nHN1,P1,AN1,platelet\n",
        report_csv=_PLT_REPORT,
        llm_json="[]",
    ).decode()

    assert "specialist" in module._REVIEW_REASON_LABELS["platelet_specialist_target"]
    assert rendered.count("<dt>platelet_specialist_target</dt>") == 1


def test_nursing_notes_sort_by_clock_time_not_by_the_raw_digits() -> None:
    # HOSxP drops leading zeros from PROGRESSTIME, so 02:33 arrives as "23300"
    # and 16:54 as "165400". A text sort puts 16:54 first; a reviewer reading
    # the shift sequence around the order needs the night note before the
    # afternoon one.
    module = _load_build_review()
    afternoon = {"PROGRESSDATE": "2025-01-02 00:00:00.000", "PROGRESSTIME": "165400"}
    night = {"PROGRESSDATE": "2025-01-02 00:00:00.000", "PROGRESSTIME": "23300"}
    next_day = {"PROGRESSDATE": "2025-01-03 00:00:00.000", "PROGRESSTIME": "5"}

    ordered = sorted([next_day, afternoon, night], key=module.focus_note_sort_key)

    assert ordered == [night, afternoon, next_day]


def test_same_day_progress_notes_sort_by_entry_time() -> None:
    # PROGDATE carries no time (always 00:00), so notes written on one day
    # keep export order unless the entry timestamp breaks the tie.
    module = _load_build_review()
    evening = {"PROGDATE": "2024-04-02 00:00:00.000", "FIRSTDATE": "2024-04-02 20:17:30.000", "ITEMNO": "1"}
    morning = {"PROGDATE": "2024-04-02 00:00:00.000", "FIRSTDATE": "2024-04-02 08:05:00.000", "ITEMNO": "2"}
    day_before = {"PROGDATE": "2024-04-01 00:00:00.000", "FIRSTDATE": "2024-04-03 09:00:00.000", "ITEMNO": "1"}

    ordered = sorted([evening, morning, day_before], key=module.progress_note_sort_key)

    assert ordered == [day_before, morning, evening]


def test_progress_notes_without_entry_time_fall_back_to_note_number() -> None:
    # FIRSTDATE is not a required ingest column; without it PROGNO (the
    # per-day note number) orders the day numerically, so note 10 does not
    # land before note 2.
    module = _load_build_review()
    note_10 = {"PROGDATE": "2024-04-02 00:00:00.000", "ITEMNO": "1", "PROGNO": "10"}
    note_2 = {"PROGDATE": "2024-04-02 00:00:00.000", "ITEMNO": "1", "PROGNO": "2"}

    ordered = sorted([note_10, note_2], key=module.progress_note_sort_key)

    assert ordered == [note_2, note_10]


def test_every_progress_note_of_a_day_survives_dedup() -> None:
    # HOSxP exports ITEMNO=1 on every progress row; the notes of one day differ
    # only by PROGNO. Deduping on (date, ITEMNO) showed the reviewer one note
    # per day while the LLM saw all of them.
    module = _load_build_review()
    day = "2024-07-26 00:00:00.000"
    morning = {"PROGDATE": day, "ITEMNO": "1", "PROGNO": "1"}
    evening = {"PROGDATE": day, "ITEMNO": "1", "PROGNO": "3"}

    assert module.progress_note_key(morning) != module.progress_note_key(evening)
    # The same row reached twice (first-N of AN and the window) is one note.
    assert module.progress_note_key(morning) == module.progress_note_key(dict(morning))


def test_progress_note_header_shows_when_the_note_was_entered() -> None:
    module = _load_build_review()

    with_entry = module.progress_note_when(
        {"PROGDATE": "2024-04-02 00:00:00.000", "FIRSTDATE": "2024-04-02 16:18:16.000"}
    )
    without_entry = module.progress_note_when({"PROGDATE": "2024-04-02 00:00:00.000"})

    assert with_entry == "2024-04-02 (entered 2024-04-02 16:18:16)"
    assert without_entry == "2024-04-02"
