"""Per-event returns for ``issue_events_gold_check.py`` (plan PR 5).

Each audit event record gets ``summarize_event_returns`` over its units.
Against gold.csv, an event screens ``not_transfused`` exactly when gold marks
it all-returned and expects RETURNED_NOT_TRANSFUSED, and its unit, returned
and incompatible counts match gold's statuses.

``reqno_total_mismatches`` checks, for every request the script clustered,
that its events plus its reserved-not-issued units add up to the REQNO
counts from ``summarize_returns`` over the admission-scoped rows. Reserved
units are found here from the raw PAYDATE/PAYTIME, not from
``units_from_ledger``.

``returns_counts`` gives the 2025 all-returned and mixed event counts on the
audit records, how many disagree with the event's ``all_returned`` flag (the
#249 cutoff's test, owner decision 1.3), and the same counts on plan 1.3's
pool (every eligible request, plain 4 h window, no order split).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping

from _hosxp_dt import _parse_hosxp_date, _parse_time

from bba.issue_events import (
    AuditEvent,
    AuditEvents,
    EventLink,
    IssueEvent,
    LedgerUnits,
    cluster_events,
)
from bba.returns_ledger import (
    ReturnsSummary,
    physical_unit_rows,
    rows_for_admission,
    summarize_event_returns,
    summarize_returns,
    terminal_status,
)

COUNTED = ("units_total", "units_returned", "units_transfused", "units_incompat")


def _event_summary(event: IssueEvent) -> ReturnsSummary:
    return summarize_event_returns([u.status for u in event.units])


def _reserved_statuses(rows: list[dict[str, str]]) -> list[str]:
    """Terminal statuses of units with no readable PAYDATE/PAYTIME on any row."""
    return [
        terminal_status([r["UNITSTAT"] for r in group])
        for group in physical_unit_rows(rows)
        if not any(
            _parse_hosxp_date(r["PAYDATE"]) and _parse_time(r["PAYTIME"]) for r in group
        )
    ]


def reqno_total_mismatches(
    ans: Mapping[str, str],
    trans_rows: Mapping[str, list[dict[str, str]]],
    results: Mapping[str, tuple[LedgerUnits, tuple[EventLink, ...]]],
) -> tuple[list[str], Counter[str]]:
    """Events plus reserved units reproduce each REQNO's counts."""
    out: list[str] = []
    seen: Counter[str] = Counter()
    for q, (ledger, links) in results.items():
        scoped = rows_for_admission(trans_rows.get(q, []), ans[q])
        # Only the unit counts are compared, and they do not read UNITAMT.
        reqno = summarize_returns(scoped, [])
        events = [_event_summary(link.event) for link in links]
        reserved = _reserved_statuses(scoped)
        parts = [*events, summarize_event_returns(reserved)] if reserved else events
        seen["reqnos"] += 1
        seen["events"] += len(events)
        seen["reserved_units"] += len(reserved)
        if len(reserved) != ledger.units_reserved_not_issued:
            out.append(f"{q} reserved units: raw {len(reserved)} ledger differs")
        for field in COUNTED:
            total = sum(getattr(p, field) for p in parts)
            if total != getattr(reqno, field):
                out.append(
                    f"{q} {field}: REQNO {getattr(reqno, field)} events+reserved {total}"
                )
        if any(s not in {"1", "3", "7"} for s in reserved):
            out.append(f"{q} reserved unit with status {sorted(set(reserved))}")
    return out, seen


def _plain_window_events(
    results: Mapping[str, tuple[LedgerUnits, tuple[EventLink, ...]]],
    eligible: Callable[[str], bool],
) -> Iterable[tuple[str, IssueEvent]]:
    for q, (ledger, _) in results.items():
        if eligible(q):
            for event in cluster_events(ledger.issued, []):
                yield q, event


def returns_counts(
    audit: AuditEvents,
    results: Mapping[str, tuple[LedgerUnits, tuple[EventLink, ...]]],
    component: Mapping[str, str],
    in_2025: Callable[[str], bool],
) -> Counter[str]:
    """2025 all-returned and mixed events, audit records and plan 1.3 pool."""
    got: Counter[str] = Counter()
    for e in audit.events:
        if not in_2025(e.order.reqno):
            continue
        s = summarize_event_returns(e.issue_unit_statuses)
        family = e.order.component
        got[f"returns_audit_events_2025_{family}"] += 1
        got[f"returns_audit_{s.disposition}_2025_{family}"] += 1
        got[f"returns_audit_mixed_2025_{family}"] += (
            s.disposition == "transfused" and s.units_returned > 0
        )
        all_returned = e.link.event.all_returned
        got[f"returns_audit_all_returned_flag_2025_{family}"] += all_returned
        got[f"returns_audit_mixed_flag_2025_{family}"] += (
            not all_returned and s.units_returned > 0
        )
        got["returns_audit_disposition_differs_from_all_returned_2025"] += (
            s.disposition == "not_transfused"
        ) != all_returned
    for q, event in _plain_window_events(results, in_2025):
        family = component[q]
        got[f"returns_plan13_pool_events_2025_{family}"] += 1
        got[f"returns_plan13_pool_all_returned_2025_{family}"] += event.all_returned
        got["returns_plan13_pool_mixed_2025"] += (
            not event.all_returned and event.units_returned > 0
        )
    return got


def gold_returns_mismatches(
    gold_rows: list[dict[str, str]],
    audit: AuditEvents,
) -> tuple[list[str], Counter[str]]:
    events: dict[tuple[str, str], AuditEvent] = {
        (e.order.reqno, str(e.event_seq)): e for e in audit.events
    }
    out: list[str] = []
    seen: Counter[str] = Counter()
    for gold in gold_rows:
        if gold["row_type"] != "event":
            continue
        q, seq = gold["reqno"], gold["event_seq"]
        e = events.get((q, seq))
        if e is None:
            out.append(f"{q} ev{seq} returns: no audit event record")
            continue
        s = summarize_event_returns(e.issue_unit_statuses)
        statuses = gold["unit_statuses"]
        want_not_transfused = gold["all_returned"] == "true"
        if want_not_transfused != (gold["expected_class"] == "RETURNED_NOT_TRANSFUSED"):
            out.append(f"{q} ev{seq} gold all_returned and expected_class disagree")
        seen[f"returns_gold_events_{s.disposition}"] += 1
        fields = (
            (
                "returns_disposition_not_transfused",
                want_not_transfused,
                s.disposition == "not_transfused",
            ),
            ("returns_units_total", int(gold["issue_units"]), s.units_total),
            ("returns_units_returned", statuses.count("3"), s.units_returned),
            ("returns_units_incompat", statuses.count("7"), s.units_incompat),
            ("returns_units_transfused", statuses.count("5"), s.units_transfused),
        )
        out += [
            f"{q} ev{seq} {name}: gold {want} got {have}"
            for name, want, have in fields
            if want != have
        ]
    return out, seen
