"""Check ``bba.issue_events`` against the full HOSxP export (blood_audit #276).

Local only: the export and the owner-held gold CSV are not in the repo, so CI
never runs this, and it exits 0 with a skip line when either is absent.

It clusters every eligible 2025 request (REQTYPE P, BDVSTST 4/5, no
CANCELDATE, HN and AN present, red-cell or platelet-only lines) into issue
events, links each event to its order, and compares:

* the plan's measured totals (per_issue_plan.md sections 0 and 1.1);
* per gold REQNO: event count, units per event, terminal statuses,
  all-returned flag, reserved-not-issued units and the event anchor in hours
  after the own order, against gold.csv;
* per gold event: link tag, linked REQNO and its status and placement time,
  the #276 latest-before REQNO, and whether the linked order's DCTREQ is the
  gold one (compared, never printed).

It also prints link counts over 2025 and January (plan 2.2), unasserted.

Output is counts only. A gold mismatch line names the REQNO, the event and
the field, with the gold and computed unit counts, status codes, link tags
or REQNOs.
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

from _hosxp_dt import _parse_hosxp_date, _parse_time

from bba.attribution import load_reqno_to_doctor
from bba.audit_orders import BloodOrderInput
from bba.issue_events import (
    CandidateOrder,
    EventLink,
    LedgerUnits,
    candidate_order,
    cluster_events,
    link_events,
    units_from_ledger,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_TABLES = ("BDVST.csv", "BDVSTDT.csv", "BDVSTTRANS.csv")
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
    candidate: CandidateOrder | None

    @property
    def placed_at(self) -> datetime | None:
        return self.candidate.placed_at if self.candidate else None


Results = dict[str, tuple[LedgerUnits, tuple[EventLink, ...]]]


def read_csv(path: Path) -> Iterator[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            yield {k.strip().upper(): (v or "").strip() for k, v in raw.items() if k}


def load_orders(export: Path) -> dict[str, Order]:
    products: dict[str, list[str]] = defaultdict(list)
    for r in read_csv(export / "BDVSTDT.csv"):
        products[r["REQNO"]].append(r["BDTYPE"])
    orders: dict[str, Order] = {}
    for r in read_csv(export / "BDVST.csv"):
        record = BloodOrderInput(
            hn=r["HN"],
            an=r["AN"] or None,
            reqno=r["REQNO"],
            bdvstst=r["BDVSTST"],
            reqtype=r["REQTYPE"],
            canceldate=r["CANCELDATE"] or None,
            req_date=_parse_hosxp_date(r["REQDATE"]),
            req_time=_parse_time(r["REQTIME"]),
            bdvst_date=_parse_hosxp_date(r["BDVSTDATE"]),
            bdvst_time=_parse_time(r["BDVSTTIME"]),
            products=tuple(products.get(r["REQNO"], [])),
            diagnosis_codes=(),
        )
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
                candidate=candidate_order(record),
            ),
        )
    return orders


def eligible(order: Order) -> bool:
    return (
        order.req_month.startswith(YEAR)
        and order.reqtype == "P"
        and order.status in {"4", "5"}
        and not order.cancelled
        and bool(order.hn)
        and order.candidate is not None
    )


def candidates_by_scope(
    orders: dict[str, Order],
) -> dict[tuple[str, str], list[CandidateOrder]]:
    scopes: dict[tuple[str, str], list[CandidateOrder]] = defaultdict(list)
    for o in orders.values():
        if o.candidate is not None:
            scopes[(o.candidate.an, o.candidate.component)].append(o.candidate)
    return scopes


def ledger_by_reqno(export: Path, wanted: set[str]) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in read_csv(export / "BDVSTTRANS.csv"):
        if r["REQNO"] in wanted:
            rows[r["REQNO"]].append(r)
    return rows


def scopes_of(order: Order) -> tuple[str, ...]:
    return ("2025", "january") if order.req_month == JANUARY else ("2025",)


def totals(orders: dict[str, Order], results: Results) -> Counter[str]:
    got: Counter[str] = Counter()
    for q, (ledger, links) in results.items():
        o = orders[q]
        if o.candidate is None or not eligible(o):
            continue
        for scope in scopes_of(o):
            got[f"eligible_requests_{scope}"] += 1
            got[f"requests_without_issued_unit_{scope}"] += not ledger.issued
            got[f"events_{scope}"] += len(links)
            got[f"events_{scope}_{o.candidate.component}"] += len(links)
        got["issued_units_2025"] += len(ledger.issued)
        got["units_reserved_not_issued_2025"] += ledger.units_reserved_not_issued
    return got


def link_counts(
    orders: dict[str, Order], results: Results, doctors: dict[str, str]
) -> Counter[str]:
    """Plan 2.2 link counts per scope, for the report; not asserted."""
    got: Counter[str] = Counter()
    for q, (_, links) in results.items():
        o = orders[q]
        if not eligible(o):
            continue
        for seq, link in enumerate(links, start=1):
            position = "first" if seq == 1 else "later"
            before = link.latest_before.reqno if link.latest_before else None
            fresh = link.tag == "fresh_order"
            facts = {
                link.tag: True,
                f"{position}_events": True,
                "order_after_issue": link.order_after_issue,
                f"{position}_differs_from_latest_before": link.linked.reqno != before,
                "latest_before_none": before is None,
                "fresh_order_status_1": fresh and link.linked.status == "1",
                "fresh_order_other_doctor": fresh
                and doctors.get(link.linked.reqno) != doctors.get(q),
            }
            for scope in scopes_of(o):
                for name, hit in facts.items():
                    got[f"links_{scope}_{name}"] += hit
    return got


def gold_mismatches(
    gold_rows: list[dict[str, str]],
    orders: dict[str, Order],
    results: Results,
    doctors: dict[str, str],
) -> tuple[list[str], Counter[str]]:
    by_reqno: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in gold_rows:
        by_reqno[r["reqno"]].append(r)
    out: list[str] = []
    seen: Counter[str] = Counter()
    for q, rows in by_reqno.items():
        if q not in results:
            out.append(f"{q} not linked")
            continue
        ledger, links = results[q]
        events = tuple(link.event for link in links)
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
            link = links[seq - 1]
            seen["event_rows"] += 1
            seen["event_units"] += len(link.event.units)
            seen[f"event_rows_{link.tag}"] += 1
            out += event_mismatches(q, r, link, orders[q].placed_at)
            out += link_mismatches(q, r, link, orders[q].placed_at, doctors)
    return out, seen


def link_mismatches(
    q: str,
    gold: dict[str, str],
    link: EventLink,
    own_placed_at: datetime | None,
    doctors: dict[str, str],
) -> list[str]:
    tag = f"{q} ev{gold['event_seq']}"
    before = link.latest_before.reqno if link.latest_before else ""
    out = [
        f"{tag} {field}: gold {want} got {got}"
        for field, want, got in (
            ("linkage_tag", gold["linkage_tag"], link.tag),
            ("linked_reqno", gold["linked_reqno"], link.linked.reqno),
            ("linked_order_status", gold["linked_order_status"], link.linked.status),
            ("linked_reqno_latest_before", gold["linked_reqno_latest_before"], before),
        )
        if want != got
    ]
    if doctors.get(link.linked.reqno, "") != gold["linked_dctreq"]:
        out.append(f"{tag} linked_dctreq differs")
    differs = doctors.get(link.linked.reqno) != doctors.get(q)
    if (gold["attributed_doctor_differs"] == "true") != differs:
        out.append(f"{tag} attributed_doctor_differs differs")
    if own_placed_at is not None:
        hours = (link.linked.placed_at - own_placed_at).total_seconds() / 3600
        want_h = float(gold["linked_order_h_after_own_order"])
        if abs(hours - want_h) > ANCHOR_TOLERANCE_H:
            out.append(f"{tag} linked_order_h_after_own_order differs")
    return out


def event_mismatches(
    q: str, gold: dict[str, str], link: EventLink, own_placed_at: datetime | None
) -> list[str]:
    tag = f"{q} ev{gold['event_seq']}"
    event = link.event
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
    scopes = candidates_by_scope(orders)
    doctors = dict(load_reqno_to_doctor(export / "BDVST.csv"))
    results: Results = {}
    failures = 0
    for q in sorted(wanted):
        o = orders[q]
        own = o.candidate
        if own is None:
            print(f"BAD {q} is not a linkage candidate (status, AN, family or time)")
            failures += 1
            continue
        scope = scopes[(own.an, own.component)]
        units = units_from_ledger(trans_rows.get(q, []), o.an)
        events = cluster_events(units.issued, [c.placed_at for c in scope])
        results[q] = (units, link_events(own, events, scope))

    got = totals(orders, results)
    for name, want in EXPECTED_TOTALS.items():
        ok = got[name] == want
        failures += not ok
        print(f"{'ok ' if ok else 'BAD'} {name}: expected {want:,} got {got[name]:,}")
    for name, count in sorted(link_counts(orders, results, doctors).items()):
        print(f"info {name}: {count:,}")

    if not gold_rows:
        print("skip: gold comparison (BBA_ISSUE_EVENTS_GOLD_CSV unset)")
    else:
        mismatches, seen = gold_mismatches(gold_rows, orders, results, doctors)
        failures += len(mismatches)
        print(f"gold: {dict(seen)}; {len(mismatches)} mismatch(es)")
        for line in mismatches:
            print(f"BAD {line}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
