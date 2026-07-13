from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_build_review() -> ModuleType:
    script = Path(__file__).resolve().parents[2] / "scripts" / "pilot" / "build_review.py"
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
