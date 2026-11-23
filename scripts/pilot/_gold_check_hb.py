"""Event Hb and the #249 cutoff for ``issue_events_gold_check.py`` (plan PR 4).

Each gold red-cell event's Hb is read at ``resolve_event_anchors`` through the
production lookup (``resolve_hb_with_fallback``, blood-bank visit fallback
only, ``not_before_utc`` from ``prior_transfusion_cutoff``) and compared with
gold.csv's cutoff and ``hb_*`` columns. The cutoff pool is every audit event
record, so a prior under another REQNO on the admission counts.

``cutoff_counts`` reports, over 2025 red-cell events, how many have a prior
red-cell event on the admission within 24 h, how many have a cutoff, and how
many have an all-returned most recent prior, on the final audit records and
on plan 1.3's pool (every eligible request, audit_orders exclusions not
applied).
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

import run_pipeline as rp
from _anchor_candidates import build_anchor_candidates

from bba.hb_lookup import (
    PRIOR_TRANSFUSION_WINDOW,
    AnchorCandidate,
    HbObservation,
    prior_transfusion_cutoff,
    resolve_event_anchors,
    resolve_hb_with_fallback,
)
from bba.issue_events import AuditEvent, AuditEvents, IssueEvent

HB_TOLERANCE = 0.005
HOURS_TOLERANCE = 0.006
TRANSFUSING_STATUSES = frozenset({"2", "5"})


def _rows(path: Path, keep: str, wanted: set[str]) -> Iterator[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            if (raw.get(keep) or "").strip() in wanted:
                yield {
                    k.strip().upper(): (v or "").strip() for k, v in raw.items() if k
                }


def _by_an(events: Iterable[AuditEvent]) -> dict[str, list[AuditEvent]]:
    pool: dict[str, list[AuditEvent]] = defaultdict(list)
    for e in events:
        pool[e.order.an].append(e)
    return pool


def _hours(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds() / 3600


def cutoff_counts(
    audit: AuditEvents, eligible_events: list[tuple[str, str, IssueEvent]]
) -> Counter[str]:
    """Plan 1.3 counts; ``eligible_events`` is (AN, REQNO, event), red cell."""
    got: Counter[str] = Counter()
    pool = _by_an(audit.events)
    for e in audit.events:
        if e.order.component != "red_cell":
            continue
        priors = [
            p
            for p in pool[e.order.an]
            if p.order.component == "red_cell"
            and p.issue_datetime < e.issue_datetime
            and e.issue_datetime - p.issue_datetime <= PRIOR_TRANSFUSION_WINDOW
        ]
        cutoff = prior_transfusion_cutoff(pool[e.order.an], e)
        latest = max(
            priors,
            key=lambda p: (p.issue_datetime, p.order.reqno, p.event_seq),
            default=None,
        )
        skipped = latest is not None and latest.link.event.all_returned
        got["cutoff_red_cell_events_2025"] += 1
        got["cutoff_any_prior_within_24h_2025"] += latest is not None
        got["cutoff_has_cutoff_2025"] += cutoff is not None
        got["cutoff_latest_prior_all_returned_2025"] += skipped
        got["cutoff_latest_prior_all_returned_earlier_transfusion_2025"] += (
            skipped and cutoff is not None
        )
        got["cutoff_any_prior_but_no_cutoff_2025"] += (
            latest is not None and cutoff is None
        )
        got["cutoff_transfused_differs_from_status_2_or_5_2025"] += (
            not e.link.event.all_returned
        ) != any(u.status in TRANSFUSING_STATUSES for u in e.link.event.units)
        got["cutoff_under_another_reqno_2025"] += (
            cutoff is not None and cutoff.order.reqno != e.order.reqno
        )
    plain: dict[str, list[tuple[datetime, bool, str]]] = defaultdict(list)
    for an, reqno, ev in eligible_events:
        plain[an].append((ev.issue_datetime, ev.all_returned, reqno))
    library = {
        (e.order.reqno, e.issue_datetime): prior_transfusion_cutoff(pool[e.order.an], e)
        for e in audit.events
        if e.order.component == "red_cell"
    }
    for lst in plain.values():
        lst.sort()
        for i, (t, _, reqno) in enumerate(lst):
            prior = [
                x for x in lst[:i] if x[0] < t and t - x[0] <= PRIOR_TRANSFUSION_WINDOW
            ]
            if (reqno, t) in library:
                c = library[(reqno, t)]
                transfused = [x for x in prior if not x[1]]
                naive = (transfused[-1][0], transfused[-1][2]) if transfused else None
                mine = (c.issue_datetime, c.order.reqno) if c else None
                got["cutoff_library_differs_from_inline_2025"] += naive != mine
                got["cutoff_library_compared_inline_2025"] += 1
            got["cutoff_plan13_pool_any_prior_within_24h_2025"] += bool(prior)
            got["cutoff_plan13_pool_latest_prior_all_returned_2025"] += (
                bool(prior) and prior[-1][1]
            )
    return got


def gold_hb_mismatches(
    gold_rows: list[dict[str, str]], export: Path, audit: AuditEvents
) -> tuple[list[str], Counter[str]]:
    """Compare each gold event's cutoff and Hb with gold.csv."""
    events = {(e.order.reqno, e.event_seq): e for e in audit.events}
    gold = [r for r in gold_rows if r["row_type"] == "event"]
    reqnos = {r["reqno"] for r in gold}
    ans = {e.order.an for (q, _), e in events.items() if q in reqnos}
    lab = list(_rows(export / "Lab.csv", "AN", ans))
    bdvst = {r["REQNO"]: r for r in _rows(export / "BDVST.csv", "REQNO", reqnos)}
    bdvstdt = list(_rows(export / "BDVSTDT.csv", "REQNO", reqnos))
    fallback = {
        q: [c for c in cands if c.reason == "blood_bank_visit_fallback"]
        for q, cands in build_anchor_candidates(
            bdvstdt_rows=bdvstdt, bdvst_by_reqno=bdvst
        ).items()
    }
    pool = _by_an(audit.events)
    observations = {an: rp._build_hb_observations(lab, an) for an in ans}
    out: list[str] = []
    seen: Counter[str] = Counter()
    for r in gold:
        e = events.get((r["reqno"], int(r["event_seq"])))
        if e is None:
            out.append(f"{r['reqno']} ev{r['event_seq']} has no audit record")
            continue
        seen["hb_gold_events"] += 1
        cutoff = prior_transfusion_cutoff(pool[e.order.an], e)
        seen["hb_gold_events_with_cutoff"] += cutoff is not None
        out += _cutoff_mismatches(r, e, cutoff)
        if e.order.component == "red_cell":
            out += _hb_mismatches(
                r, e, cutoff, observations[e.order.an], fallback.get(e.order.reqno, [])
            )
    return out, seen


def _cutoff_mismatches(
    gold: dict[str, str], e: AuditEvent, cutoff: AuditEvent | None
) -> list[str]:
    tag = f"{gold['reqno']} ev{gold['event_seq']}"
    got_reqno = cutoff.order.reqno if cutoff else ""
    got_seq = str(cutoff.event_seq) if cutoff else ""
    out = [
        f"{tag} {name}: gold {want or '-'} got {have or '-'}"
        for name, want, have in (
            ("prior_event_cutoff_reqno", gold["prior_event_cutoff_reqno"], got_reqno),
            ("prior_event_cutoff_seq", gold["prior_event_cutoff_seq"], got_seq),
        )
        if want != have
    ]
    want_h = gold["prior_event_cutoff_h_after_own_order"]
    if cutoff is not None and want_h:
        got_h = _hours(cutoff.issue_datetime, e.order.order_datetime)
        if abs(got_h - float(want_h)) > HOURS_TOLERANCE:
            out.append(
                f"{tag} prior_event_cutoff_h_after_own_order: gold {want_h} got {got_h:.2f}"
            )
    elif bool(want_h) != (cutoff is not None):
        out.append(f"{tag} prior_event_cutoff_h_after_own_order presence differs")
    return out


def _hb_mismatches(
    gold: dict[str, str],
    e: AuditEvent,
    cutoff: AuditEvent | None,
    observations: list[HbObservation],
    fallback: list[AnchorCandidate],
) -> list[str]:
    tag = f"{gold['reqno']} ev{gold['event_seq']}"
    anchor = resolve_event_anchors(e)
    hb, _display, reason = resolve_hb_with_fallback(
        observations=observations,
        order_datetime=anchor.anchor_utc,
        candidates=fallback,
        not_before_utc=cutoff.issue_datetime if cutoff else None,
    )
    out = [
        f"{tag} {name}: gold {want or '-'} got {have or '-'}"
        for name, want, have in (
            ("hb_source", gold["hb_source"], hb.source or ""),
            ("hb_freshness", gold["hb_freshness"], hb.freshness),
            (
                "hb_anchor_reason",
                gold["hb_anchor_reason"],
                "issue_datetime" if reason == "order_datetime" else reason,
            ),
        )
        if want != have
    ]
    age = _hours(anchor.anchor_utc, hb.datetime_utc) if hb.datetime_utc else None
    for name, want, have, tolerance in (
        ("hb_g_dl", gold["hb_g_dl"], hb.value_g_dl, HB_TOLERANCE),
        ("hb_min_24h_g_dl", gold["hb_min_24h_g_dl"], hb.min_24h_g_dl, HB_TOLERANCE),
        ("hb_age_h", gold["hb_age_h"], age, HOURS_TOLERANCE),
    ):
        if (not want) != (have is None) or (
            have is not None and abs(have - float(want)) > tolerance
        ):
            out.append(
                f"{tag} {name}: gold {want or '-'} got {'-' if have is None else f'{have:.2f}'}"
            )
    return out
