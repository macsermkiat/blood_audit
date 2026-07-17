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
    monkeypatch.setattr(module, "MSBOS_RESERVATION_PILOT_ENABLED", False)
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
