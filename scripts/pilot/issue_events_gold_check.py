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

* the audit records (plan 4.1-4.2, PR 3): audit_orders runs over the
  eligible requests with their diagnoses, and ``build_audit_events`` turns
  them into event records and REQNO-level excluded records. Per gold REQNO:
  row type and reason, event count and sequence, distinct event audit_ids
  under the REQNO's parent id, linkage fields, the linked USETYPE,
  ``unit_family_mismatch`` and ``declared_use_applies`` (operations are
  IPTSUMOPRT records, INCPT surgery fees, INCGRP 111, and general
  anaesthesia charges, AS056-AS059: owner decision 8.15 (c)). The January
  record counts of plan 6.2 are asserted; 2025 counts are printed.

* per gold event (PR 4, ``_gold_check_hb.py``): the #249 cutoff event and
  the Hb read at the event anchor with it, against gold.csv's
  ``prior_event_cutoff_*`` and ``hb_*`` columns. Plan 1.3's #249 prior
  counts are asserted on its own pool; the same counts on the audit records,
  and how many events get a cutoff, are printed.

* per gold event (PR 5, ``_gold_check_returns.py``): the event returns
  summary (disposition, unit, returned, incompatible and transfused counts)
  against gold.csv's ``all_returned``, ``expected_class`` and
  ``unit_statuses``. For every clustered request, its events plus its
  reserved-not-issued units must reproduce the REQNO-level totals. Plan 1.3's
  all-returned counts are asserted on its own pool; the same counts on the
  audit records are printed.

It also prints link counts over 2025 and January (plan 2.2), unasserted.

Output is counts only. A gold mismatch line names the REQNO, the event and
the field, with the gold and computed unit counts, status codes, link tags,
REQNOs, USETYPE codes or flags; never an HN, AN, DCTREQ or audit_id.
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
import logging
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import run_pipeline as rp
from _gold_check_hb import cutoff_counts, gold_hb_mismatches
from _gold_check_returns import (
    gold_returns_mismatches,
    reqno_total_mismatches,
    returns_counts,
)
from _hosxp_dt import _parse_hosxp_date, _parse_time
from _periop_operations import operations_by_an

from bba.attribution import load_reqno_to_doctor
from bba.audit_orders import (
    AuditOrdersConfig,
    BloodOrderInput,
    build_audit_id,
    build_audit_orders,
)
from bba.declared_use import collapse_usetype
from bba.issue_events import (
    AuditEvent,
    AuditEvents,
    CandidateOrder,
    EventLink,
    ExcludedReqno,
    LedgerUnits,
    build_audit_events,
    candidate_order,
    cluster_events,
    link_events,
    units_from_ledger,
)
from bba.issue_events.audit_event import unit_family

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_TABLES = (
    "BDVST.csv",
    "BDVSTDT.csv",
    "BDVSTTRANS.csv",
    "Diagnosis.csv",
    "IPTSUMOPRT.csv",
    "INCPT_OPRTACT.csv",
    "Lab.csv",
)
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
    "units_family_mismatch_2025": 1,
    # Owner decision 8.15 (c): red-cell later events linked to a fresh order
    # declaring USETYPE 2/3, over all eligible requests (measure6.py).
    "fresh_declaring_red_cell_events_2025": 258,
    # Plan 6.2 January invariants for the audit records.
    "audit_event_records_january": 1_812,
    "audit_event_reqnos_january": 1_178,
    "audit_no_issue_records_january": 5,
    "audit_excluded_records_january": 15,
    "audit_excluded_measured_events_january": 18,
    # Plan 1.3 #249 priors over every eligible red-cell event (measure3.py).
    "cutoff_plan13_pool_any_prior_within_24h_2025": 3_828,
    "cutoff_plan13_pool_latest_prior_all_returned_2025": 208,
    # Per event, the library cutoff equals an inline re-derivation over the
    # same records; and "not all returned" equals "has a status 2 or 5 unit".
    "cutoff_library_differs_from_inline_2025": 0,
    "cutoff_transfused_differs_from_status_2_or_5_2025": 0,
    # Plan 1.3 returned events: every eligible request, plain 4 h window.
    "returns_plan13_pool_all_returned_2025_red_cell": 1_928,
    "returns_plan13_pool_all_returned_2025_platelet": 80,
    "returns_plan13_pool_mixed_2025": 1_170,
    # Owner decision 1.3 (2026-11-23): an all-returned event is
    # RETURNED_NOT_TRANSFUSED even when its REQNO issued fewer units than
    # ordered, so the event summary agrees with the #249 cutoff's flag.
    "returns_audit_not_transfused_2025_red_cell": 1_902,
    "returns_audit_not_transfused_2025_platelet": 78,
    "returns_audit_disposition_differs_from_all_returned_2025": 0,
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
    record: BloodOrderInput
    usetype: str | None
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
    usetypes: dict[str, list[str]] = defaultdict(list)
    for r in read_csv(export / "BDVSTDT.csv"):
        products[r["REQNO"]].append(r["BDTYPE"])
        usetypes[r["REQNO"]].append(r["USETYPE"])
    diagnoses: dict[str, list[str]] = defaultdict(list)
    for r in read_csv(export / "Diagnosis.csv"):
        diagnoses[r["AN"]].append(r["ICD10"])
    # Mixed USETYPE detail lines collapse to no declaration and log a warning.
    logging.getLogger("bba.declared_use").setLevel(logging.ERROR)
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
            diagnosis_codes=rp._icd_codes(*diagnoses.get(r["AN"], []), r["ICD10"]),
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
                record=record,
                usetype=collapse_usetype(usetypes.get(r["REQNO"], [])),
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
        got["units_family_mismatch_2025"] += sum(
            unit_family(u.bdtype) != o.candidate.component for u in ledger.issued
        )
        got["fresh_declaring_red_cell_events_2025"] += sum(
            link.tag == "fresh_order"
            and o.candidate.component == "red_cell"
            and orders[link.linked.reqno].usetype in {"2", "3"}
            for link in links
        )
    return got


def operation_times(export: Path, ans: set[str]) -> dict[str, list[datetime]]:
    """Operation times per admission under owner decision 8.15 (c)."""
    operations = operations_by_an(
        rp._normalize_iptsumoprt(
            [r for r in read_csv(export / "IPTSUMOPRT.csv") if r["AN"] in ans]
        ),
        rp._normalize_incpt(
            [r for r in read_csv(export / "INCPT_OPRTACT.csv") if r["AN"] in ans]
        ),
    )
    return {an: [op.at for op in operations.get(an, [])] for an in ans}


def audit_records(
    export: Path, orders: dict[str, Order], audited: set[str], results: Results
) -> AuditEvents:
    """Plan PR 3: audit_orders, then one record per event or excluded REQNO."""
    filtered = build_audit_orders(
        [orders[q].record for q in sorted(audited)],
        AuditOrdersConfig(code_version="issue-events-gold-check"),
    )
    return build_audit_events(
        filtered,
        {q: ledger for q, (ledger, _) in results.items()},
        [o.candidate for o in orders.values() if o.candidate is not None],
        usetypes={q: o.usetype for q, o in orders.items()},
        operations=operation_times(export, {orders[q].an for q in audited}),
    )


def audit_totals(orders: dict[str, Order], audit: AuditEvents) -> Counter[str]:
    """Plan 6.2 record counts per scope; January is asserted, 2025 printed."""
    got: Counter[str] = Counter()
    for scope in ("2025", "january"):
        reqnos = {
            e.order.reqno
            for e in audit.events
            if scope in scopes_of(orders[e.order.reqno])
        }
        got[f"audit_event_reqnos_{scope}"] = len(reqnos)
    for e in audit.events:
        for scope in scopes_of(orders[e.order.reqno]):
            got[f"audit_event_records_{scope}"] += 1
            got[f"audit_events_unit_family_mismatch_{scope}"] += e.unit_family_mismatch
            got[f"audit_declared_use_applies_{scope}_{e.linkage_rule}"] += (
                e.declared_use_applies
            )
            if e.linkage_rule == "fresh_order" and e.linked_usetype in {"2", "3"}:
                got[f"audit_fresh_declaring_{scope}_{e.order.component}"] += 1
                got[f"audit_fresh_declaring_applies_{scope}_{e.order.component}"] += (
                    e.declared_use_applies
                )
    for r in audit.excluded:
        for scope in scopes_of(orders[r.reqno]):
            kind = "no_issue" if r.reason == "no_issue_record" else "excluded"
            got[f"audit_{kind}_records_{scope}"] += 1
            if kind == "excluded":
                got[f"audit_excluded_measured_events_{scope}"] += r.events_in_reqno
                got[f"audit_excluded_reason_{scope}_{r.reason}"] += 1
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


def audit_mismatches(
    gold_rows: list[dict[str, str]], orders: dict[str, Order], audit: AuditEvents
) -> tuple[list[str], Counter[str]]:
    """Compare the PR 3 audit records of each gold REQNO with gold.csv."""
    events: dict[str, list[AuditEvent]] = defaultdict(list)
    for e in audit.events:
        events[e.order.reqno].append(e)
    excluded = {r.reqno: r for r in audit.excluded}
    by_reqno: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in gold_rows:
        by_reqno[r["reqno"]].append(r)
    out: list[str] = []
    seen: Counter[str] = Counter()
    for q, rows in by_reqno.items():
        parent = build_audit_id(orders[q].hn, q)
        got, record = events.get(q, []), excluded.get(q)
        kind = rows[0]["row_type"]
        if kind != "event":
            seen[f"audit_reqno_records_{kind}"] += record is not None
            out += reqno_record_mismatches(q, rows[0], got, record, parent)
            continue
        if record is not None:
            out.append(f"{q} row_type: gold event got {record.reason}")
        want_seqs = sorted(int(r["event_seq"]) for r in rows)
        if [e.event_seq for e in got] != want_seqs:
            out.append(
                f"{q} event_seq: gold {want_seqs} got {[e.event_seq for e in got]}"
            )
            continue
        ids = {e.audit_id for e in got}
        if len(ids) != len(got) or parent in ids:
            out.append(
                f"{q} event audit_ids are not disjoint from each other and the parent"
            )
        if {e.parent_audit_id for e in got} != {parent}:
            out.append(f"{q} parent_audit_id is not the REQNO's legacy audit_id")
        for r in rows:
            seen["audit_event_records"] += 1
            seen[f"audit_declared_use_applies_{r['declared_use_applies']}"] += 1
            out += audit_event_mismatches(q, r, got[int(r["event_seq"]) - 1])
    return out, seen


def audit_event_mismatches(q: str, gold: dict[str, str], e: AuditEvent) -> list[str]:
    tag = f"{q} ev{gold['event_seq']}"
    before = e.linked_reqno_latest_before or ""
    # gold.csv has no unit_family_mismatch column; no gold unit crosses
    # families (plan 1.1 measured 1 such unit in 2025, outside the gold set).
    fields = (
        ("events_in_reqno", gold["events_in_reqno"], str(e.events_in_reqno)),
        ("issue_units", gold["issue_units"], str(e.issue_units)),
        ("unit_statuses", gold["unit_statuses"], "".join(e.issue_unit_statuses)),
        (
            "units_reserved_not_issued",
            gold["units_reserved_not_issued"],
            str(e.units_reserved_not_issued),
        ),
        ("linkage_tag", gold["linkage_tag"], e.linkage_rule),
        ("linked_reqno", gold["linked_reqno"], e.linked_reqno),
        ("linked_order_status", gold["linked_order_status"], e.linked_order_status),
        ("linked_reqno_latest_before", gold["linked_reqno_latest_before"], before),
        ("linked_usetype", gold["linked_usetype"], e.linked_usetype or ""),
        ("unit_family_mismatch", "false", str(e.unit_family_mismatch).lower()),
        (
            "declared_use_applies",
            gold["declared_use_applies"],
            str(e.declared_use_applies).lower(),
        ),
    )
    out = [
        f"{tag} {name}: gold {want} got {got}"
        for name, want, got in fields
        if want != got
    ]
    if (
        abs(e.order_to_issue_hours - float(gold["order_to_issue_h"]))
        > ANCHOR_TOLERANCE_H
    ):
        out.append(f"{tag} order_to_issue_h differs")
    return out


def reqno_record_mismatches(
    q: str,
    gold: dict[str, str],
    got: list[AuditEvent],
    record: object,
    parent: str,
) -> list[str]:
    if got:
        return [f"{q} gold {gold['row_type']} row got {len(got)} event records"]
    if record is None:
        return [f"{q} gold {gold['row_type']} row got no REQNO-level record"]
    assert isinstance(record, ExcludedReqno)
    out = [
        f"{q} {name}: gold {want} got {have}"
        for name, want, have in (
            ("reason", gold["reason"], record.reason),
            ("events_in_reqno", gold["events_in_reqno"], str(record.events_in_reqno)),
            (
                "units_reserved_not_issued",
                gold["units_reserved_not_issued"],
                str(record.units_reserved_not_issued),
            ),
            ("event_seq", gold["event_seq"], ""),
        )
        if want != have
    ]
    if not record.audit_id == record.parent_audit_id == parent:
        out.append(f"{q} REQNO-level audit_id is not the legacy audit_id")
    if record.event_seq is not None:
        out.append(f"{q} REQNO-level record has an event_seq")
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

    audited = {q for q in results if eligible(orders[q])} | {
        r["reqno"] for r in gold_rows
    }
    audit = audit_records(export, orders, audited, results)
    red_cell_events = [
        (o.an, q, link.event)
        for q, (_, links) in results.items()
        if eligible(o := orders[q])
        and o.candidate
        and o.candidate.component == "red_cell"
        for link in links
    ]
    cutoffs = cutoff_counts(audit, red_cell_events)
    ans = {q: orders[q].an for q in results}
    returns = returns_counts(
        audit,
        results,
        {q: o.candidate.component for q, o in orders.items() if o.candidate},
        lambda q: eligible(orders[q]),
    )
    total_out, total_seen = reqno_total_mismatches(ans, trans_rows, results)
    failures += len(total_out)
    print(
        f"reqno totals = events + reserved: {dict(total_seen)}; {len(total_out)} mismatch(es)"
    )
    for line in total_out:
        print(f"BAD {line}")
    got = totals(orders, results) + audit_totals(orders, audit) + cutoffs + returns
    for name, want in EXPECTED_TOTALS.items():
        ok = got[name] == want
        failures += not ok
        print(f"{'ok ' if ok else 'BAD'} {name}: expected {want:,} got {got[name]:,}")
    for name, count in sorted(link_counts(orders, results, doctors).items()):
        print(f"info {name}: {count:,}")
    for name, count in sorted(audit_totals(orders, audit).items()):
        if name not in EXPECTED_TOTALS:
            print(f"info {name}: {count:,}")
    for name, count in sorted((cutoffs + returns).items()):
        if name not in EXPECTED_TOTALS:
            print(f"info {name}: {count:,}")

    if not gold_rows:
        print("skip: gold comparison (BBA_ISSUE_EVENTS_GOLD_CSV unset)")
    else:
        mismatches, seen = gold_mismatches(gold_rows, orders, results, doctors)
        audit_out, audit_seen = audit_mismatches(gold_rows, orders, audit)
        hb_out, hb_seen = gold_hb_mismatches(gold_rows, export, audit)
        returns_out, returns_seen = gold_returns_mismatches(gold_rows, audit)
        mismatches += audit_out + hb_out + returns_out
        seen += audit_seen + hb_seen + returns_seen
        failures += len(mismatches)
        print(f"gold: {dict(seen)}; {len(mismatches)} mismatch(es)")
        for line in mismatches:
            print(f"BAD {line}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
