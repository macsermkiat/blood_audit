"""Write the read-only committee report for the MSBOS reservation pilot.

Environment variables:

* ``BBA_AUDIT_STORE_DIR`` — audit-store root; overrides ``BBA_DATA_DIR``.
* ``BBA_DATA_DIR`` — audit-store parent (resolved as ``<value>/audit_store``).
* ``BBA_RUN_ID`` — required persisted run scope.
* ``BBA_CODE_VERSION`` — required persisted code-version scope.
* ``BBA_MSBOS_REPORT_OUT`` — JSON output path (default under the work dir).
* ``BBA_PILOT_WORK_DIR`` — output work dir (default ``/tmp/bba_mini``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from bba.audit_store import AuditStore, AuditStoreConfig
from bba.preop_reservation.pilot_report import (
    MsbosReservationPilotReport,
    PilotReportError,
    build_pilot_report,
)

_HOLD_RECOMMENDATION = (
    "HOLD — over-reservation precision pending a clinician-validated sample"
)


def _audit_store_path() -> Path | None:
    explicit = os.environ.get("BBA_AUDIT_STORE_DIR")
    if explicit:
        return Path(explicit)
    data_dir = os.environ.get("BBA_DATA_DIR")
    return Path(data_dir) / "audit_store" if data_dir else None


def _output_path() -> Path:
    work = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))
    return Path(
        os.environ.get(
            "BBA_MSBOS_REPORT_OUT",
            str(work / "msbos_reservation_pilot_report.json"),
        )
    )


def _format_rate(resolved: int, denominator: int, rate: float | None) -> str:
    if rate is None:
        return f"{resolved}/{denominator} (null)"
    return f"{resolved}/{denominator} ({rate:.1%})"


def _print_summary(report: MsbosReservationPilotReport, artifact: Path) -> None:
    rbc = report.coverage.rbc_note_resolution_rate
    platelet = report.coverage.platelet_category_resolution_rate
    other = report.coverage.platelet_other_review_reasons
    precision = report.precision
    reconciliation = report.reconciliation
    print("MSBOS pre-op reservation pilot report")
    print(
        f"Scope: run_id={report.provenance.run_id} "
        f"code_version={report.provenance.code_version} "
        f"rows={report.provenance.scoped_audit_rows} "
        f"reservation_markers={report.provenance.total_reservation_markers}"
    )
    print("Coverage")
    print(
        "  RBC note resolution: "
        f"{_format_rate(rbc.resolved, rbc.denominator, rbc.rate)}; "
        f"unresolved={rbc.unresolved}. Limitation: {rbc.limitation}"
    )
    print(
        "  Platelet category resolution: "
        f"{_format_rate(platelet.resolved, platelet.denominator, platelet.rate)}; "
        f"category_unresolved={platelet.category_unresolved}. "
        f"Limitation: {platelet.limitation}"
    )
    print(
        "  Other platelet reviews: "
        f"missing_pre_op_count={other.missing_pre_op_count}, "
        f"no_planned_op={other.no_planned_op}, "
        f"ambiguous_planned_op={other.ambiguous_planned_op}"
    )
    print(f"Precision/PPV: {precision.status}")
    print(f"  {precision.pending_note}")
    print(f"  {precision.overlap_note}")
    print(
        f"Returns reconciliation: {reconciliation.status} "
        f"(over_markers={reconciliation.over_marker_count}, "
        f"returns_terminals={reconciliation.returns_terminal_count}, "
        f"double_fires={reconciliation.double_fire_count})"
    )
    if reconciliation.double_fire_ids:
        print(f"  Double-fire audit_ids: {', '.join(reconciliation.double_fire_ids)}")
    print(f"RECOMMENDATION: {_HOLD_RECOMMENDATION}")
    print(f"JSON artifact: {artifact}")


def main() -> int:
    """Build the pinned report from persisted rows and marker calls."""
    run_id = os.environ.get("BBA_RUN_ID")
    code_version = os.environ.get("BBA_CODE_VERSION")
    if not run_id:
        print("missing required BBA_RUN_ID", file=sys.stderr)
        return 1
    if not code_version:
        print("missing required BBA_CODE_VERSION", file=sys.stderr)
        return 1
    store_dir = _audit_store_path()
    if store_dir is None:
        print(
            "missing audit store: set BBA_AUDIT_STORE_DIR or BBA_DATA_DIR",
            file=sys.stderr,
        )
        return 1
    if not store_dir.is_dir():
        print(f"missing audit store directory: {store_dir}", file=sys.stderr)
        return 1

    # Read-only guard: never let the output path resolve inside the audit-store
    # root (directly or via a symlink), which would let write_text clobber a
    # persisted parquet or commit marker and break the READ-ONLY contract.
    artifact = _output_path()
    store_root = store_dir.resolve()
    artifact_resolved = artifact.resolve()
    if artifact_resolved == store_root or store_root in artifact_resolved.parents:
        print(
            "refusing to write the report inside the audit store "
            f"({store_root}); set BBA_MSBOS_REPORT_OUT to a path outside it",
            file=sys.stderr,
        )
        return 1

    store = AuditStore(AuditStoreConfig(root_dir=store_dir, code_version=code_version))
    rows = store.read_audit_results(run_id=run_id, code_version=code_version)
    calls = store.read_llm_calls(run_id=run_id, code_version=code_version)
    if not rows:
        print(
            "audit store returned zero committed audit rows for "
            f"run_id={run_id} code_version={code_version}",
            file=sys.stderr,
        )
        return 1
    try:
        report = build_pilot_report(
            rows, calls, run_id=run_id, code_version=code_version
        )
    except PilotReportError as exc:
        print(f"MSBOS pilot report integrity failure: {exc}", file=sys.stderr)
        return 1
    if report.provenance.total_reservation_markers == 0:
        print(
            "no reservation activity for "
            f"run_id={run_id} code_version={code_version}; the producing run needs "
            "BBA_PILOT_MSBOS_RESERVATION=1 and MSBOS_RESERVATION_ENABLED",
            file=sys.stderr,
        )
        return 1

    try:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(
            json.dumps(
                report.model_dump(mode="json"),
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        print(
            f"failed to write MSBOS report artifact {artifact}: {exc}", file=sys.stderr
        )
        return 1
    _print_summary(report, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
