"""Salvage tag-leaked bilingual reasoning in a pilot llm_report.json.

Pilot run 2026-07-06 (165 rows, claude-sonnet-5): on 131 rows the model
serialized BOTH reasoning summaries into ``reasoning_en`` — separated by
fragments of its internal tool-call tag syntax — and returned an empty
``reasoning_th``. The pipeline salvage now lives in
:func:`bba.audit_pipeline.replay.split_leaked_summaries`; this script
applies the same split retroactively to an already-written report, plus
the empty-reasoning guardrail (rows with no reasoning in either language
are floored to NEEDS_REVIEW / needs_human_review=True, matching what the
fixed pipeline would have produced).

The original file is preserved as ``llm_report.json.bak`` (never
overwritten on re-runs). Re-running on an already-fixed report is a
no-op.

Usage:
    python scripts/pilot/fix_reasoning_split.py            # uses $BBA_PILOT_WORK_DIR or /tmp/bba_mini
    python scripts/pilot/build_review.py                   # then regenerate review.html
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from bba.audit_pipeline.replay import (
    EMPTY_REASONING_REVIEW_REASON,
    split_leaked_summaries,
)

REVIEWABLE = "NEEDS_REVIEW"


def fix_record(record: dict[str, Any]) -> tuple[dict[str, Any], bool, bool]:
    """Return (fixed_record, salvaged, flagged_empty) without mutating input."""
    llm_final = record.get("llm_final")
    if not isinstance(llm_final, dict):
        return record, False, False

    en = llm_final.get("reasoning_en") or ""
    th = llm_final.get("reasoning_th") or ""
    clean_en, clean_th = split_leaked_summaries(en, th)
    salvaged = (clean_en, clean_th) != (en, th)

    fixed_final = {**llm_final, "reasoning_en": clean_en, "reasoning_th": clean_th}

    flagged_empty = False
    if not clean_en.strip() and not clean_th.strip():
        flagged_empty = fixed_final.get(
            "final_classification"
        ) != REVIEWABLE or not fixed_final.get("needs_human_review")
        fixed_final = {
            **fixed_final,
            "final_classification": REVIEWABLE,
            "needs_human_review": True,
            "review_reason": fixed_final.get("review_reason")
            or EMPTY_REASONING_REVIEW_REASON,
        }

    if not salvaged and not flagged_empty:
        return record, False, False
    return {**record, "llm_final": fixed_final}, salvaged, flagged_empty


def main() -> None:
    default_report = (
        Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini")) / "llm_report.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=default_report)
    args = parser.parse_args()

    report_path: Path = args.report
    if not report_path.exists():
        sys.exit(f"report not found: {report_path}")

    records = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        sys.exit(f"expected a JSON list in {report_path}")

    fixed_records: list[dict[str, Any]] = []
    salvaged_count = 0
    flagged_count = 0
    for record in records:
        fixed, salvaged, flagged = fix_record(record)
        fixed_records.append(fixed)
        salvaged_count += salvaged
        flagged_count += flagged

    if salvaged_count == 0 and flagged_count == 0:
        print(f"{report_path}: nothing to fix ({len(records)} records clean)")
        return

    backup = report_path.with_suffix(".json.bak")
    if not backup.exists():
        shutil.copy2(report_path, backup)
        print(f"original preserved: {backup}")

    report_path.write_text(
        json.dumps(fixed_records, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(
        f"{report_path}: {len(records)} records — "
        f"{salvaged_count} bilingual leaks split, "
        f"{flagged_count} empty-reasoning rows floored to {REVIEWABLE}"
    )


if __name__ == "__main__":
    main()
