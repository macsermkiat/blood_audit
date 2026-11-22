"""Check ``bba.issue_events`` against the full HOSxP export (blood_audit #276).

Local only: the export and the owner-held gold CSV are not in the repo, so CI
never runs this, and it exits 0 with a skip line when either is absent.

It clusters every eligible 2025 request (REQTYPE P, BDVSTST 4/5, no
CANCELDATE, HN and AN present, red-cell or platelet-only lines) into issue
events and compares:

* the plan's measured totals (per_issue_plan.md sections 0 and 1.1);
* per gold REQNO: event count, units per event, terminal statuses,
  all-returned flag, reserved-not-issued units and the event anchor in hours
  after the own order, against gold.csv.

Output is counts only. A gold mismatch line names the REQNO, the event and
the field, with the gold and computed unit counts or status codes.
Exit status 1 on any mismatch, or when the gold CSV is named but unreadable.

Environment variables:

* ``BBA_PILOT_RAW_DIR``: the HOSxP export (default ``../Bloodbank/data/encrypted``
  relative to the repo root, as ``sample_bundle.py``).
* ``BBA_ISSUE_EVENTS_GOLD_CSV``: the owner's gold.csv. Without it the gold
  comparison is skipped and only the totals are checked. Set to a missing
  or empty file, the check fails.

Usage: ``uv run python scripts/pilot/issue_events_gold_check.py``
"""

from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from _hosxp_dt import _combine, _parse_hosxp_date, _parse_time

from bba.audit_orders.rules import is_rbc_product
from bba.component_map import is_platelet_product
from bba.issue_events import IssueEvent, LedgerUnits, cluster_events, units_from_ledger

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_TABLES = ("BDVST.csv", "BDVSTDT.csv", "BDVSTTRANS.csv")
# Plan 1.2: refused (6), cancelled (7) and cancelled doctor orders (11, 12) never split an event.
NON_SPLITTING_STATUSES = frozenset({"6", "7", "11", "12"})
YEAR, JANUARY = "2025", "2025-01"

# per_issue_plan.md sections 0 and 1.1, measured over the same export.
EXPECTED_TOTALS = {
    "eligible_requests_2025": 13_740,
    "issued_units_2025": 28_245,
    "units_reserved_not_issued_2025": 681,
    "requests_without_issued_unit_2025": 40,
    "events_2025": 20_830,
    "events_2025_red_cell": 15_011,
    "events_2025_platelet": 5_819,
    "eligible_requests_january": 1_198,
    "requests_without_issued_unit_january": 5,
    "events_january": 1_830,
    "events_january_red_cell": 1_321,
    "events_january_platelet": 509,
}
ANCHOR_TOLERANCE_H = 0.006


@dataclass(frozen=True)
class Order:
    reqno: str
    hn: str
    an: str
    reqtype: str
    status: str
    cancelled: bool
    req_month: str
    family: str
    placed_at: datetime | None


def read_csv(path: Path) -> Iterator[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            yield {k.strip().upper(): (v or "").strip() for k, v in raw.items() if k}


def family_of(products: list[str]) -> str:
    if any(is_rbc_product(p) for p in products):
        return "red_cell"
    if products and all(is_platelet_product(p) for p in products):
        return "platelet"
    return "other"


def load_orders(export: Path) -> dict[str, Order]:
    products: dict[str, list[str]] = defaultdict(list)
    for r in read_csv(export / "BDVSTDT.csv"):
        products[r["REQNO"]].append(r["BDTYPE"])
    orders: dict[str, Order] = {}
    for r in read_csv(export / "BDVST.csv"):
        requested = _combine(_parse_hosxp_date(r["REQDATE"]), _parse_time(r["REQTIME"]))
        visit = _combine(_parse_hosxp_date(r["BDVSTDATE"]), _parse_time(r["BDVSTTIME"]))
        orders.setdefault(
            r["REQNO"],
            Order(
                reqno=r["REQNO"],
                hn=r["HN"],
                an=r["AN"],
                reqtype=r["REQTYPE"],
                status=r["BDVSTST"],
                cancelled=bool(r["CANCELDATE"]),
                req_month=r["REQDATE"][:7],
                family=family_of(products.get(r["REQNO"], [])),
                placed_at=requested or visit,
            ),
        )
    return orders


def eligible(order: Order) -> bool:
    return (
        order.req_month.startswith(YEAR)
        and order.reqtype == "P"
        and order.status in {"4", "5"}
        and not order.cancelled
        and bool(order.an)
        and bool(order.hn)
        and order.family != "other"
    )


def splitting_order_times(
    orders: dict[str, Order],
) -> dict[tuple[str, str], list[datetime]]:
    times: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    for o in orders.values():
        if o.an and o.placed_at is not None and o.status not in NON_SPLITTING_STATUSES:
            times[(o.an, o.family)].append(o.placed_at)
    return times


def ledger_by_reqno(export: Path, wanted: set[str]) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in read_csv(export / "BDVSTTRANS.csv"):
        if r["REQNO"] in wanted:
            rows[r["REQNO"]].append(r)
    return rows


def totals(
    orders: dict[str, Order],
    results: dict[str, tuple[LedgerUnits, tuple[IssueEvent, ...]]],
) -> Counter[str]:
    got: Counter[str] = Counter()
    for q, (ledger, events) in results.items():
        o = orders[q]
        if not eligible(o):
            continue
        scopes = ("2025", "january") if o.req_month == JANUARY else ("2025",)
        for scope in scopes:
            got[f"eligible_requests_{scope}"] += 1
            got[f"requests_without_issued_unit_{scope}"] += not ledger.issued
            got[f"events_{scope}"] += len(events)
            got[f"events_{scope}_{o.family}"] += len(events)
        got["issued_units_2025"] += len(ledger.issued)
        got["units_reserved_not_issued_2025"] += ledger.units_reserved_not_issued
    return got


def gold_mismatches(
    gold_rows: list[dict[str, str]],
    orders: dict[str, Order],
    results: dict[str, tuple[LedgerUnits, tuple[IssueEvent, ...]]],
) -> tuple[list[str], Counter[str]]:
    by_reqno: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in gold_rows:
        by_reqno[r["reqno"]].append(r)
    out: list[str] = []
    seen: Counter[str] = Counter()
    for q, rows in by_reqno.items():
        ledger, events = results[q]
        seen["reqnos"] += 1
        want_events = int(rows[0]["events_in_reqno"])
        if len(events) != want_events:
            out.append(f"{q} events_in_reqno: gold {want_events} got {len(events)}")
        if (
            int(rows[0]["units_reserved_not_issued"])
            != ledger.units_reserved_not_issued
        ):
            out.append(f"{q} units_reserved_not_issued differs")
        event_rows = [r for r in rows if r["row_type"] == "event"]
        if not event_rows:
            seen[f"reqno_rows_{rows[0]['row_type']}"] += 1
            continue
        if len(event_rows) != int(rows[0]["events_in_reqno"]):
            out.append(f"{q} gold event rows disagree with events_in_reqno")
        for r in event_rows:
            seq = int(r["event_seq"])
            if not 1 <= seq <= len(events):
                out.append(f"{q} ev{seq} not produced")
                continue
            event = events[seq - 1]
            seen["event_rows"] += 1
            seen["event_units"] += len(event.units)
            out += event_mismatches(q, r, event, orders[q].placed_at)
    return out, seen


def event_mismatches(
    q: str, gold: dict[str, str], event: IssueEvent, own_placed_at: datetime | None
) -> list[str]:
    tag = f"{q} ev{gold['event_seq']}"
    out = []
    if int(gold["issue_units"]) != len(event.units):
        out.append(
            f"{tag} issue_units: gold {gold['issue_units']} got {len(event.units)}"
        )
    statuses = "".join(u.status for u in event.units)
    if gold["unit_statuses"] != statuses:
        out.append(f"{tag} unit_statuses: gold {gold['unit_statuses']} got {statuses}")
    if (gold["all_returned"] == "true") != event.all_returned:
        out.append(f"{tag} all_returned differs")
    if own_placed_at is None:
        out.append(f"{tag} own order has no placement time")
    else:
        hours = (event.issue_datetime - own_placed_at).total_seconds() / 3600
        if abs(hours - float(gold["issue_h_after_own_order"])) > ANCHOR_TOLERANCE_H:
            out.append(f"{tag} issue_h_after_own_order differs")
    return out


def main() -> int:
    export = Path(
        os.environ.get("BBA_PILOT_RAW_DIR")
        or REPO_ROOT.parent / "Bloodbank/data/encrypted"
    )
    missing = [name for name in EXPORT_TABLES if not (export / name).is_file()]
    if missing:
        print(
            f"skip: HOSxP export not found ({len(missing)} of {len(EXPORT_TABLES)} tables missing)"
        )
        return 0
    gold_env = os.environ.get("BBA_ISSUE_EVENTS_GOLD_CSV", "")
    gold_rows = []
    if gold_env:
        gold_path = Path(gold_env)
        if gold_path.is_file():
            with gold_path.open(encoding="utf-8", newline="") as fh:
                gold_rows = list(csv.DictReader(fh))
        if not gold_rows:
            print(
                "BAD BBA_ISSUE_EVENTS_GOLD_CSV is set but the file is missing or empty"
            )
            return 1

    orders = load_orders(export)
    wanted = {q for q, o in orders.items() if eligible(o)} | {
        r["reqno"] for r in gold_rows
    }
    trans_rows = ledger_by_reqno(export, wanted)
    order_times = splitting_order_times(orders)
    results = {}
    for q in sorted(wanted):
        o = orders[q]
        units = units_from_ledger(trans_rows.get(q, []), o.an)
        results[q] = (
            units,
            cluster_events(units.issued, order_times[(o.an, o.family)]),
        )

    got = totals(orders, results)
    failures = 0
    for name, want in EXPECTED_TOTALS.items():
        ok = got[name] == want
        failures += not ok
        print(f"{'ok ' if ok else 'BAD'} {name}: expected {want:,} got {got[name]:,}")

    if not gold_rows:
        print("skip: gold comparison (BBA_ISSUE_EVENTS_GOLD_CSV unset)")
    else:
        mismatches, seen = gold_mismatches(gold_rows, orders, results)
        failures += len(mismatches)
        print(f"gold: {dict(seen)}; {len(mismatches)} mismatch(es)")
        for line in mismatches:
            print(f"BAD {line}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
