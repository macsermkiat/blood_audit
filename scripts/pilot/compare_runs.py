"""Compare a REQNO-level run (v2) with an issue-event run (v3), counts only.

blood_audit #276, plan 4.3 and 6.2. v2 judged each REQNO once; v3 judges each
issue event. Rows join through ``parent_audit_id``: v3 writes it on every row,
and for v2 it is ``build_audit_id(HN, REQNO)`` from the v2 sample manifest.

    python scripts/pilot/compare_runs.py <v2_dir> <v3_dir>

Reads ``report.csv``, ``llm_report.json`` and ``sample_manifest.csv`` from
both folders and writes nothing into the v2 folder. Prints summary counts and
writes into the v3 folder:

* ``compare_v2_v3.csv``: per REQNO the v2 final verdict and Hb, the v3 event
  verdicts, Hb values and linkage tags;
* ``compare_linkage_audit_ids.txt``: audit_ids of the events whose linked
  order differs under the other linkage rule (``linked_reqno_latest_before``),
  the events to re-ask if the owner changes the default (plan 7 step 3).

Final verdicts are merged as ``rank_merged.py`` merges them: the LLM leg's
verdict replaces a deterministic verdict that was routed to it.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from bba.audit_orders import build_audit_id

from rank_merged import ROUTED_TO_LLM

Row = Mapping[str, str]
LINKAGE_LIST = "compare_linkage_audit_ids.txt"
PER_REQNO_CSV = "compare_v2_v3.csv"


@dataclass(frozen=True)
class Comparison:
    counts: dict[str, int]
    per_reqno: list[dict[str, str]]
    relink_audit_ids: list[str]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _llm_finals(path: Path, key: str) -> dict[str, str]:
    if not path.exists():
        return {}
    return {
        rec[key]: rec["llm_final"]["final_classification"]
        for rec in json.loads(path.read_text(encoding="utf-8"))
        if rec.get("llm_final")
    }


def _final(row: Row, llm: Mapping[str, str], key: str) -> str:
    det = row["classification"]
    return llm.get(row[key], det) if det in ROUTED_TO_LLM else det


def _hb(row: Row) -> str:
    return (row.get("hb_value_g_dl") or "").strip()


def compare(v2_dir: Path, v3_dir: Path) -> Comparison:
    """Count how v2 REQNO rows map onto v3 event and REQNO-level rows."""
    v2_rows = _read_csv(v2_dir / "report.csv")
    v2_hn = {r["REQNO"]: r["HN"] for r in _read_csv(v2_dir / "sample_manifest.csv")}
    v2_llm = _llm_finals(v2_dir / "llm_report.json", "reqno")
    v3_rows = _read_csv(v3_dir / "report.csv")
    v3_llm = _llm_finals(v3_dir / "llm_report.json", "audit_id")

    by_parent: dict[str, list[dict[str, str]]] = {}
    for row in v3_rows:
        by_parent.setdefault(row["parent_audit_id"], []).append(row)
    events = [r for r in v3_rows if (r.get("event_seq") or "").strip()]
    reqno_level = [r for r in v3_rows if not (r.get("event_seq") or "").strip()]
    reqnos_with_events = {r["reqno"] for r in events}

    counts: Counter[str] = Counter()
    counts["v2_rows"] = len(v2_rows)
    counts["v3_event_rows"] = len(events)
    counts["v3_reqnos_with_events"] = len(reqnos_with_events)
    counts["v3_reqno_level_rows"] = len(reqno_level)
    counts["v3_audit_ids_distinct"] = len({r["audit_id"] for r in v3_rows})
    for row in reqno_level:
        counts[f"v3_reqno_level:{row.get('rationale') or '(none)'}"] += 1
    for row in events:
        counts[f"linkage:{row.get('linkage_rule') or '(none)'}"] += 1
        counts[
            f"attributed_doctor_differs:{row.get('attributed_doctor_differs') or '(blank)'}"
        ] += 1
        counts[
            f"declared_use_applies:{row.get('declared_use_applies') or '(blank)'}"
        ] += 1
        if not (row.get("linked_reqno") or "").strip():
            counts["events_without_linked_order"] += 1
        if (
            row.get("linkage_rule") == "fresh_order"
            and row["linked_reqno"] in reqnos_with_events
        ):
            counts["fresh_links_to_reqno_with_own_events"] += 1
    per_reqno_events = Counter(r["reqno"] for r in events)
    for n in per_reqno_events.values():
        counts[f"events_per_reqno:{n if n < 5 else '5+'}"] += 1

    relink = sorted(
        r["audit_id"]
        for r in events
        if (r.get("linked_reqno_latest_before") or "").strip()
        and r["linked_reqno_latest_before"] != r["linked_reqno"]
    )
    counts["events_relinked_under_latest_before"] = len(relink)

    per_reqno: list[dict[str, str]] = []
    for v2 in v2_rows:
        reqno = v2["reqno"]
        hn = v2_hn.get(reqno)
        if hn is None:
            counts["v2_rows_without_manifest_hn"] += 1
            continue
        mapped = sorted(
            by_parent.get(build_audit_id(hn, reqno), []),
            key=lambda r: int(r["event_seq"] or 0),
        )
        mapped_events = [r for r in mapped if (r.get("event_seq") or "").strip()]
        if mapped_events:
            counts["v2_rows_mapped_to_events"] += 1
        elif mapped:
            counts["v2_rows_mapped_to_reqno_level_row"] += 1
        else:
            counts["v2_rows_unmapped"] += 1
        v2_final = _final(v2, v2_llm, "reqno")
        v3_finals = [_final(r, v3_llm, "audit_id") for r in mapped_events]
        if mapped_events:
            first_hb = _hb(mapped_events[0])
            if mapped_events[0].get("component") == "red_cell":
                counts[
                    "first_event_hb_equals_v2"
                    if first_hb and first_hb == _hb(v2)
                    else "first_event_hb_differs_or_missing"
                ] += 1
            counts[
                "first_event_verdict_equals_v2"
                if v3_finals[0] == v2_final
                else "first_event_verdict_differs"
            ] += 1
        per_reqno.append(
            {
                "reqno": reqno,
                "v2_final": v2_final,
                "v2_hb": _hb(v2),
                "v3_events": str(len(mapped_events)),
                "v3_finals": "|".join(v3_finals)
                or (mapped[0]["classification"] if mapped else ""),
                "v3_hbs": "|".join(_hb(r) for r in mapped_events),
                "v3_linkage": "|".join(
                    r.get("linkage_rule") or "" for r in mapped_events
                ),
            }
        )
    return Comparison(dict(sorted(counts.items())), per_reqno, relink)


def main(argv: Sequence[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 2:
        print("usage: compare_runs.py <v2_dir> <v3_dir>", file=sys.stderr)
        return 2
    v2_dir, v3_dir = (Path(a) for a in args)
    if v2_dir.resolve() == v3_dir.resolve():
        print("v2 and v3 must be different folders", file=sys.stderr)
        return 2
    result = compare(v2_dir, v3_dir)
    with (v3_dir / PER_REQNO_CSV).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result.per_reqno[0]))
        writer.writeheader()
        writer.writerows(result.per_reqno)
    (v3_dir / LINKAGE_LIST).write_text(",".join(result.relink_audit_ids))
    for key, value in result.counts.items():
        print(f"{key:<48} {value}")
    print(f"per-REQNO table -> {v3_dir / PER_REQNO_CSV}")
    print(f"re-ask list if the linkage default changes -> {v3_dir / LINKAGE_LIST}")
    return 1 if result.counts.get("v2_rows_unmapped") else 0


if __name__ == "__main__":
    raise SystemExit(main())
