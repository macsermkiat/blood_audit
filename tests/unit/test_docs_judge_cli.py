"""Smoke test for scripts/docs_judge/__main__.py in dry-run mode.

WHY this test exists:
- The CLI is the only path that wires splitter, judge, rollup, parity, and
  report together. A dry run over a tiny corpus proves the wiring without
  the network and pins the output files a later humanize pass reads.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from docs_judge import __main__ as cli  # noqa: E402

EN = """---
title: T
---

## Intro

This is an English paragraph long enough to be judged by the tool.
"""
TH = """---
title: ท
---

## บทนำ

นี่คือย่อหน้าภาษาไทยที่ยาวพอสำหรับการตัดสินโดยเครื่องมือนี้
"""


def test_dry_run_writes_report_with_pages_parity_and_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = tmp_path / "docs/src/content/docs"
    (docs / "en/clinical").mkdir(parents=True)
    (docs / "th/clinical").mkdir(parents=True)
    (docs / "en/clinical/p.mdx").write_text(EN, encoding="utf-8")
    (docs / "th/clinical/p.mdx").write_text(TH, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"

    assert cli.main(["run", "--out", str(out), "--dry-run"]) == 0

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert {p["language"] for p in report["pages"]} == {"en", "th"}
    assert len(report["judgments"]) == 2
    assert len(report["parity"]) == 1
    assert report["calibration"]["th_slop"]["n"] == 10
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "## Pages ranked by rewrite count" in md
    assert (out / "jev_cache.jsonl").exists()


def test_missing_api_key_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="TYPESAFE_API_KEY"):
        cli.main(["calibrate", "--out", str(tmp_path / "o")])
