"""Validate the pipeline verdict source against the 300-case human review.

The pre-swap cross-check for Feature 2: before
:func:`bba.attribution.pipeline_verdict_source` (the application's own
per-order verdicts) replaces the human-review workbook as the ranking
input, this script reads BOTH sources and prints their bucket totals plus,
over the REQNOs they share, how often they agree and how many orders the
pipeline *cleared* that the human reviewer called inappropriate — the
peri-op over-clear signal that must be ~0 to trust the swap.

Read-only. Writes nothing. Fail-loud (non-zero exit) if either source is
missing or the audit store has no committed rows for the requested scope.

Environment variables:

* ``BBA_REVIEW_XLSX`` — the human-review workbook (default:
  ``~/Downloads/Review การใช้เลือด.xlsx``).
* ``BBA_AUDIT_STORE_DIR`` — root of the audit-store Parquet layout
  (default: ``$BBA_DATA_DIR/audit_store``).
* ``BBA_RUN_ID`` — scope the pipeline read to one run so each REQNO
  resolves to a single verdict (recommended). Without it, every committed
  run/version is read and a REQNO with conflicting verdicts fails loud.
* ``BBA_CODE_VERSION`` — optional; scope the pipeline read to one code
  version (default: read every committed version).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

from bba.attribution import (
    human_label_verdict_source,
    needs_review_verdict_projector,
    pipeline_verdict_source,
    reconcile_verdict_sources,
)
from bba.audit_store import AuditStore, AuditStoreConfig
from bba.cli.identity import code_version
from bba.report_generator.models import Classification

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


def _buckets(verdicts: Mapping[str, Classification]) -> tuple[int, int, int, int]:
    """(appropriate, inappropriate, unresolved, total) — unresolved folds
    NEEDS_REVIEW and INSUFFICIENT_EVIDENCE together, matching the report's
    3-bucket convention (see ``bba.attribution.pipeline._bucket_totals``)."""
    appropriate = sum(1 for c in verdicts.values() if c == "APPROPRIATE")
    inappropriate = sum(1 for c in verdicts.values() if c == "INAPPROPRIATE")
    total = len(verdicts)
    return appropriate, inappropriate, total - appropriate - inappropriate, total


def main() -> int:
    if not _AUDIT_STORE_RAW:
        print(
            "missing audit store: set BBA_AUDIT_STORE_DIR or BBA_DATA_DIR",
            file=sys.stderr,
        )
        return 1
    audit_store_dir = Path(_AUDIT_STORE_RAW)
    if not audit_store_dir.exists():
        print(
            f"missing audit store (BBA_AUDIT_STORE_DIR): {audit_store_dir}",
            file=sys.stderr,
        )
        return 1
    if not REVIEW_XLSX.exists():
        print(
            f"missing review workbook (BBA_REVIEW_XLSX): {REVIEW_XLSX}",
            file=sys.stderr,
        )
        return 1

    store = AuditStore(
        AuditStoreConfig(root_dir=audit_store_dir, code_version=str(code_version()))
    )
    rows = store.read_audit_results(run_id=RUN_ID, code_version=CODE_VERSION_FILTER)
    # POTENTIALLY_INAPPROPRIATE is a non-confident pre-verdict -> Unresolved,
    # consistent with bba.verification.models; never ranked as Inappropriate.
    pipeline = pipeline_verdict_source(rows, projector=needs_review_verdict_projector)()
    if not pipeline:
        print(
            "audit store returned no committed audit rows for the requested "
            f"scope (run_id={RUN_ID!r}, code_version={CODE_VERSION_FILTER!r})",
            file=sys.stderr,
        )
        return 1

    human = human_label_verdict_source(REVIEW_XLSX)()
    recon = reconcile_verdict_sources(pipeline, human)

    p_appr, p_inappr, p_unres, p_total = _buckets(pipeline)
    h_appr, h_inappr, h_unres, h_total = _buckets(human)
    agree_pct = 100.0 * recon.agree / recon.overlap if recon.overlap else 0.0

    print("verdict-source reconciliation")
    print(
        f"  audit store : {audit_store_dir} "
        f"(run_id={RUN_ID or 'ALL'}, code_version={CODE_VERSION_FILTER or 'ALL'})"
    )
    print(f"  review xlsx : {REVIEW_XLSX}")
    print("")
    print(
        f"pipeline verdicts: {p_total} orders "
        f"(appropriate {p_appr} / inappropriate {p_inappr} / unresolved {p_unres})"
    )
    print(
        f"human review     : {h_total} cases  "
        f"(appropriate {h_appr} / inappropriate {h_inappr} / unresolved {h_unres})"
    )
    print("")
    print(f"overlap (REQNOs in both): {recon.overlap}")
    print(f"  agree    : {recon.agree} ({agree_pct:.1f}%)")
    print(f"  disagree : {recon.disagree}")
    print(
        "  pipeline over-clears (human INAPPROPRIATE -> pipeline APPROPRIATE): "
        f"{recon.pipeline_over_clears}   <- must be ~0 to trust the swap"
    )
    print(f"pipeline-only REQNOs: {recon.pipeline_only}")
    print(
        f"human-only REQNOs   : {recon.human_only} "
        "(human labels absent from the audited cohort)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
