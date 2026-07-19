from __future__ import annotations

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
            "returns_units_returned\n"
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
            "within_recommendation,False,0139,hash,,,,,1,1\n"
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
    assert "Reserved 2; MSBOS tariff G/M 2; 1 transfused, 1 returned" in rendered
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
            returns_units_transfused="1",
            returns_units_returned="2",
        ),
        "PERIOP_TRANSFUSION_EXEMPT",
    )
    assert exempt == "Reserved 3; MSBOS tariff G/M 2; 1 transfused, 2 returned"


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
            returns_units_transfused="1",
            returns_units_returned="1",
        ),
        "PERIOP_TRANSFUSION_EXEMPT",
    )
    assert exempt.endswith("; 1 transfused, 1 returned")
