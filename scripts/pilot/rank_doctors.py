"""Rank top-10 ordering doctors / departments by blood-order
appropriateness (Feature 2 pilot runner).

Thin glue over :mod:`bba.attribution` — the loaders, aggregation,
ranking, and writers are the integrated, tested module; this script only
resolves paths and reports the reconciliation summary.

Verdict source for THIS build: the 300-case human review workbook
(Sheet1 col J). The next build swaps in full-cohort pipeline verdicts
via the same :data:`bba.attribution.VerdictSource` seam, gated on the
peri-op classifier fix.

Environment variables:

* ``BBA_REVIEW_XLSX`` — the human-review workbook (default:
  ``~/Downloads/Review การใช้เลือด.xlsx``).
* ``BBA_BDVST_CSV`` — BDVST export with REQNO + DCTREQ (default:
  ``../Bloodbank/data/encrypted/BDVST.csv`` relative to the repo root).
* ``BBA_DCT_CSV`` — DCT.csv doctor registry (default:
  ``../Bloodbank/data/raw/DCT.csv``).
* ``BBA_PILOT_WORK_DIR`` — output directory (default: ``/tmp/bba_mini``).

Outputs (in the work dir): ``doctor_ranking.csv``,
``department_ranking.csv``, ``doctor_rankings.html``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from bba.attribution import (
    build_rankings,
    human_label_verdict_source,
    load_dct_registry,
    load_reqno_to_doctor,
    write_ranking_csv,
    write_rankings_html,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BLOODBANK = _REPO_ROOT.parent / "Bloodbank" / "data"

REVIEW_XLSX = Path(
    os.environ.get(
        "BBA_REVIEW_XLSX",
        str(Path.home() / "Downloads" / "Review การใช้เลือด.xlsx"),
    )
)
BDVST_CSV = Path(
    os.environ.get("BBA_BDVST_CSV", str(_BLOODBANK / "encrypted" / "BDVST.csv"))
)
DCT_CSV = Path(os.environ.get("BBA_DCT_CSV", str(_BLOODBANK / "raw" / "DCT.csv")))
WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))


def main() -> int:
    for path, label in (
        (REVIEW_XLSX, "review workbook (BBA_REVIEW_XLSX)"),
        (BDVST_CSV, "BDVST export (BBA_BDVST_CSV)"),
        (DCT_CSV, "DCT registry (BBA_DCT_CSV)"),
    ):
        if not path.exists():
            print(f"missing {label}: {path}", file=sys.stderr)
            return 1
    WORK.mkdir(parents=True, exist_ok=True)

    verdicts = human_label_verdict_source(REVIEW_XLSX)()
    result = build_rankings(
        verdicts=verdicts,
        reqno_to_doctor=load_reqno_to_doctor(BDVST_CSV),
        dct_registry=load_dct_registry(DCT_CSV),
    )

    doctor_csv = write_ranking_csv(result.doctors.rows, WORK / "doctor_ranking.csv")
    dept_csv = write_ranking_csv(
        result.departments.rows, WORK / "department_ranking.csv"
    )
    html_path = write_rankings_html(
        result,
        WORK / "doctor_rankings.html",
        verdict_source_label=(
            f"{result.totals.total}-case human review "
            f"({REVIEW_XLSX.name}, Sheet1 col J)"
        ),
    )

    t = result.totals
    print(
        f"verdicts: {t.total} "
        f"(appropriate {t.appropriate} / inappropriate {t.inappropriate} / "
        f"unresolved {t.unresolved})"
    )
    print(f"ranked bucket: {result.doctors.bucket}; min N: {result.doctors.min_orders}")
    print(f"doctor table rows: {len(result.doctors.rows)} -> {doctor_csv}")
    print(f"department table rows: {len(result.departments.rows)} -> {dept_csv}")
    print(f"html: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
