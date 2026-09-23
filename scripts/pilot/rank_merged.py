"""Doctor / department ranking from the pilot's MERGED verdicts, per component.

``rank_doctors.py`` with ``BBA_VERDICT_SOURCE=pipeline`` reads the audit store,
which holds LLM-leg rows only: every order the deterministic leg finalised
(APPROPRIATE auto-clears, returns, exemptions) is missing, so the denominators
are wrong. This script merges ``report.csv`` (deterministic final verdicts)
with ``llm_report.json`` (LLM final verdicts, which replace the deterministic
NEEDS_REVIEW / POTENTIALLY_INAPPROPRIATE they were routed from) and ranks one
component at a time, so the mean / min / max trigger columns are Hb for RBC
and platelet count for platelets. It replaces the scratch glue used for the
2026-09-19 hematology ranking, which was never committed.

    BBA_PILOT_WORK_DIR=/tmp/bba_hemato python scripts/pilot/rank_merged.py red_cell
    BBA_PILOT_WORK_DIR=/tmp/bba_hemato python scripts/pilot/rank_merged.py platelet

Outputs ``<work>/<component>_doctor_ranking.csv``,
``<component>_department_ranking.csv`` and ``<component>_doctor_rankings.html``.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from bba.attribution import (
    build_rankings,
    load_dct_registry,
    load_order_labs,
    load_reqno_to_doctor,
    write_ranking_csv,
    write_rankings_html,
)

WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))
_BLOODBANK = Path(__file__).resolve().parents[2].parent / "Bloodbank" / "data"
# The bundle's BDVST copy has no DCTREQ (ordering doctor); the full export does.
BDVST_CSV = Path(
    os.environ.get("BBA_BDVST_CSV", str(_BLOODBANK / "encrypted" / "BDVST.csv"))
)
DCT_CSV = Path(os.environ.get("BBA_DCT_CSV", str(_BLOODBANK / "raw" / "DCT.csv")))
COMPONENTS = ("red_cell", "platelet")
# Deterministic verdicts the LLM leg is dispatched from; every other verdict is
# a deterministic terminal.
ROUTED_TO_LLM = frozenset({"NEEDS_REVIEW", "POTENTIALLY_INAPPROPRIATE"})


def merged_verdicts(work: Path, component: str) -> Mapping[str, str]:
    """Final verdict per REQNO for one component: the LLM leg's where the
    deterministic leg routed the order to it, else the deterministic leg's.
    Excluded orders are left out. An LLM verdict for an order the report does
    not know is an error, never a silent drop."""
    with (work / "report.csv").open(encoding="utf-8") as f:
        det = {r["reqno"]: r for r in csv.DictReader(f)}
    llm = {
        r["reqno"]: r["llm_final"]["final_classification"]
        for r in json.loads((work / "llm_report.json").read_text(encoding="utf-8"))
        if r.get("llm_final")
    }
    missing = sorted(set(llm) - set(det))
    if missing:
        raise ValueError(
            f"{work / 'llm_report.json'} holds LLM verdicts for orders absent from "
            f"{work / 'report.csv'} (stale report?): {missing[:10]}"
        )
    out: dict[str, str] = {}
    for reqno, row in det.items():
        row_component = (row.get("component") or "red_cell").strip() or "red_cell"
        if row_component != component or row["classification"] == "excluded":
            continue
        # Only a verdict the deterministic leg ROUTED onward can be replaced by
        # the LLM leg's; a deterministic terminal (auto-clear, return, periop
        # exemption, MSBOS over-reservation) stands even if a stale LLM row
        # exists for the same REQNO.
        replaceable = row["classification"] in ROUTED_TO_LLM
        out[reqno] = (
            llm[reqno] if replaceable and reqno in llm else row["classification"]
        )
    return out


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in COMPONENTS:
        print(f"usage: rank_merged.py {{{'|'.join(COMPONENTS)}}}", file=sys.stderr)
        return 2
    component = args[0]
    for path, label in ((BDVST_CSV, "BDVST"), (DCT_CSV, "DCT registry")):
        if not path.exists():
            print(f"missing {label}: {path}", file=sys.stderr)
            return 1
    verdicts = merged_verdicts(WORK, component)
    if not verdicts:
        print(f"no {component} orders in {WORK / 'report.csv'}", file=sys.stderr)
        return 1
    result = build_rankings(
        verdicts=verdicts,
        reqno_to_doctor=load_reqno_to_doctor(BDVST_CSV),
        dct_registry=load_dct_registry(DCT_CSV),
        order_labs=load_order_labs(WORK / "report.csv"),
    )
    doctor_csv = write_ranking_csv(
        result.doctors.rows, WORK / f"{component}_doctor_ranking.csv"
    )
    dept_csv = write_ranking_csv(
        result.departments.rows, WORK / f"{component}_department_ranking.csv"
    )
    html = write_rankings_html(
        result,
        WORK / f"{component}_doctor_rankings.html",
        verdict_source_label=(
            f"{result.totals.total} {component} orders; deterministic verdicts "
            "with the LLM leg's final verdict where it ran"
        ),
    )
    t = result.totals
    print(
        f"{component}: {t.total} orders (appropriate {t.appropriate} / "
        f"inappropriate {t.inappropriate} / unresolved {t.unresolved})"
    )
    print(f"  {doctor_csv}\n  {dept_csv}\n  {html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
