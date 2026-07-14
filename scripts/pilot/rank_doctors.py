"""Rank top-10 ordering doctors / departments by blood-order
appropriateness (Feature 2 pilot runner).

Thin glue over :mod:`bba.attribution` — the loaders, aggregation,
ranking, and writers are the integrated, tested module; this script only
resolves paths and reports the reconciliation summary.

Verdict source is selectable via ``BBA_VERDICT_SOURCE`` (spec #119 follow-up):

* ``human`` (default) — the 300-case human review workbook (Sheet1 col J).
* ``pipeline`` — the application's own full-cohort per-order verdicts read from
  the audit store, via the same :data:`bba.attribution.VerdictSource` seam. This
  is what a live scorecard re-baseline uses once a run has committed audit rows;
  it pools the store-only ``POTENTIALLY_INAPPROPRIATE`` /
  ``PREOP_RESERVATION_UNCONFIRMED`` into Unresolved and passes the excluded
  returns terminals (``RETURNED_NOT_TRANSFUSED`` / ``PERIOP_TRANSFUSION_EXEMPT``)
  through for the ranking layer to hold apart (matches
  ``reconcile_verdict_sources.py`` and the shipped attribution exclusion).

Environment variables:

* ``BBA_VERDICT_SOURCE`` — ``human`` (default) or ``pipeline``.
* ``BBA_REVIEW_XLSX`` — the human-review workbook (human source; default:
  ``~/Downloads/Review การใช้เลือด.xlsx``).
* ``BBA_AUDIT_STORE_DIR`` — audit-store root (pipeline source; default:
  ``$BBA_DATA_DIR/audit_store``).
* ``BBA_RUN_ID`` — scope the pipeline read to one run so each REQNO resolves to a
  single verdict (recommended for the pipeline source).
* ``BBA_CODE_VERSION`` — optional; scope the pipeline read to one code version.
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
from collections.abc import Mapping
from pathlib import Path

from bba.attribution import (
    build_rankings,
    human_label_verdict_source,
    load_dct_registry,
    load_reqno_to_doctor,
    needs_review_verdict_projector,
    pipeline_verdict_source,
    write_ranking_csv,
    write_rankings_html,
)
from bba.audit_store import AuditStore, AuditStoreConfig
from bba.cli.identity import code_version

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BLOODBANK = _REPO_ROOT.parent / "Bloodbank" / "data"

VERDICT_SOURCE = os.environ.get("BBA_VERDICT_SOURCE", "human").strip().lower()
REVIEW_XLSX = Path(
    os.environ.get(
        "BBA_REVIEW_XLSX",
        str(Path.home() / "Downloads" / "Review การใช้เลือด.xlsx"),
    )
)
_DATA_DIR = os.environ.get("BBA_DATA_DIR")
_AUDIT_STORE_RAW = os.environ.get("BBA_AUDIT_STORE_DIR") or (
    str(Path(_DATA_DIR) / "audit_store") if _DATA_DIR else ""
)
RUN_ID = os.environ.get("BBA_RUN_ID") or None
CODE_VERSION_FILTER = os.environ.get("BBA_CODE_VERSION") or None
BDVST_CSV = Path(
    os.environ.get("BBA_BDVST_CSV", str(_BLOODBANK / "encrypted" / "BDVST.csv"))
)
DCT_CSV = Path(os.environ.get("BBA_DCT_CSV", str(_BLOODBANK / "raw" / "DCT.csv")))
WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))


def _resolve_verdicts() -> tuple[Mapping[str, str], str]:
    """Return ``(verdicts, source_label)`` for the selected verdict source.

    Fail-loud on a missing/empty source so an empty ranking can never present
    as a real re-baseline (mirrors ``reconcile_verdict_sources.py``).
    """
    if VERDICT_SOURCE == "pipeline":
        if not _AUDIT_STORE_RAW:
            raise SystemExit(
                "BBA_VERDICT_SOURCE=pipeline requires BBA_AUDIT_STORE_DIR or "
                "BBA_DATA_DIR to locate the audit store"
            )
        # Require an explicit single run scope. Without BBA_RUN_ID the store read
        # spans every committed run/version, and pipeline_verdict_source only
        # rejects a REQNO carrying *different* verdicts — disjoint partial reruns
        # (or duplicate same-verdict rows) would silently merge into one ranking.
        # The scorecard must be built from one run, mirroring the report builder's
        # single-run/version contract.
        if not RUN_ID:
            raise SystemExit(
                "BBA_VERDICT_SOURCE=pipeline requires BBA_RUN_ID to scope the "
                "read to a single run; ranking across mixed runs would silently "
                "merge disjoint verdicts"
            )
        audit_store_dir = Path(_AUDIT_STORE_RAW)
        if not audit_store_dir.exists():
            raise SystemExit(
                f"missing audit store (BBA_AUDIT_STORE_DIR): {audit_store_dir}"
            )
        store = AuditStore(
            AuditStoreConfig(root_dir=audit_store_dir, code_version=str(code_version()))
        )
        rows = store.read_audit_results(run_id=RUN_ID, code_version=CODE_VERSION_FILTER)
        # POTENTIALLY_INAPPROPRIATE / PREOP_RESERVATION_UNCONFIRMED -> Unresolved;
        # the excluded returns terminals pass through for the ranking layer to
        # hold apart from the scorable denominator.
        verdicts = pipeline_verdict_source(
            rows, projector=needs_review_verdict_projector
        )()
        if not verdicts:
            raise SystemExit(
                "audit store returned no committed audit rows for the requested "
                f"scope (run_id={RUN_ID!r}, code_version={CODE_VERSION_FILTER!r}); "
                "run the pipeline for this run before re-baselining scorecards"
            )
        label = (
            f"{len(verdicts)}-order pipeline verdicts "
            f"(audit store, run_id={RUN_ID or 'ALL'}, "
            f"code_version={CODE_VERSION_FILTER or 'ALL'})"
        )
        return verdicts, label

    if VERDICT_SOURCE != "human":
        raise SystemExit(
            f"unknown BBA_VERDICT_SOURCE {VERDICT_SOURCE!r}; expected "
            "'human' or 'pipeline'"
        )
    if not REVIEW_XLSX.exists():
        raise SystemExit(f"missing review workbook (BBA_REVIEW_XLSX): {REVIEW_XLSX}")
    verdicts = human_label_verdict_source(REVIEW_XLSX)()
    return verdicts, f"human review ({REVIEW_XLSX.name}, Sheet1 col J)"


def main() -> int:
    for path, label in (
        (BDVST_CSV, "BDVST export (BBA_BDVST_CSV)"),
        (DCT_CSV, "DCT registry (BBA_DCT_CSV)"),
    ):
        if not path.exists():
            print(f"missing {label}: {path}", file=sys.stderr)
            return 1
    WORK.mkdir(parents=True, exist_ok=True)

    verdicts, source_label = _resolve_verdicts()
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
        verdict_source_label=f"{result.totals.total}-order ranked; {source_label}",
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
