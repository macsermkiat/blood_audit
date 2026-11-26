"""Issue events on the review page (blood_audit #276, plan 4.3).

The page shows one case per issue event, numbered by the REQNO's place in the
sample manifest and then by event_seq. Every case carries the REQNO's unit
timeline: each issued unit with its issue time, status, return time, the event
it belongs to and that event's linked order and flags. Units are rebuilt from
the bundle's ledger with the same parser the pipeline uses
(``units_from_ledger``) and assigned to events by the events' issue times; a
count that disagrees with the report's ``issue_units`` fails loud.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

from bba.issue_events import IssuedUnit, units_from_ledger

from _hosxp_dt import TZ_LOCAL, _fmt_utc_local

ReportRow = Mapping[str, str]


def report_by_audit_id(rows: Iterable[ReportRow]) -> dict[str, ReportRow]:
    """report.csv rows keyed by audit_id. A repeated audit_id would let one
    case render another event's verdict, so it fails loud."""
    by_id: dict[str, ReportRow] = {}
    for row in rows:
        audit_id = (row.get("audit_id") or "").strip()
        if not audit_id:
            raise ValueError(
                "report.csv has a row without audit_id (a REQNO-level report "
                "from before issue events?); re-run run_pipeline.py"
            )
        if audit_id in by_id:
            raise ValueError(f"duplicate audit_id in report scope: {audit_id!r}")
        by_id[audit_id] = row
    return by_id


def _seq(row: ReportRow) -> int:
    return int(row["event_seq"])


def events_by_reqno(report: Mapping[str, ReportRow]) -> dict[str, list[ReportRow]]:
    """Each REQNO's event rows in event_seq order (REQNO-level rows left out)."""
    grouped: dict[str, list[ReportRow]] = defaultdict(list)
    for row in report.values():
        if (row.get("event_seq") or "").strip():
            grouped[row["reqno"]].append(row)
    return {reqno: sorted(rows, key=_seq) for reqno, rows in grouped.items()}


def review_cases(
    sample_rows: Sequence[ReportRow],
    event_rows: Sequence[ReportRow],
    report: Mapping[str, ReportRow],
) -> list[dict[str, str]]:
    """The page's cases in order: each sampled REQNO's events by event_seq, or
    its one REQNO-level row (excluded, no_issue_record) when it has none.

    Every event in ``events_manifest.csv`` must belong to a sampled REQNO and
    every sampled REQNO must have events or a REQNO-level row; anything else
    means the manifests and the report come from different runs.
    """
    events: dict[str, list[ReportRow]] = defaultdict(list)
    for row in event_rows:
        events[row["REQNO"]].append(row)
    reqno_level = {
        row["reqno"]: audit_id
        for audit_id, row in report.items()
        if not (row.get("event_seq") or "").strip()
    }
    sampled = [row["REQNO"] for row in sample_rows]
    unsampled = sorted(set(events) - set(sampled))
    if unsampled:
        raise ValueError(
            f"events_manifest.csv has events for {len(unsampled)} REQNO(s) "
            f"missing from sample_manifest.csv (e.g. {unsampled[:5]})"
        )
    duplicated = sorted(r for r, n in Counter(sampled).items() if n > 1)
    if duplicated:
        raise ValueError(f"sample_manifest.csv lists REQNO(s) twice: {duplicated}")
    cases: list[dict[str, str]] = []
    for sample in sample_rows:
        reqno = sample["REQNO"]
        if events.get(reqno):
            for event in sorted(events[reqno], key=lambda r: int(r["event_seq"])):
                cases.append({**sample, **event})
        elif reqno in reqno_level:
            cases.append({**sample, "audit_id": reqno_level[reqno], "event_seq": ""})
        else:
            raise ValueError(
                f"REQNO {reqno} has neither events nor a REQNO-level row in "
                "report.csv; the report and the manifests are from different runs"
            )
    missing = sorted(c["audit_id"] for c in cases if c["audit_id"] not in report)
    if missing:
        raise ValueError(
            f"{len(missing)} event(s) in events_manifest.csv have no report.csv "
            f"row (e.g. {missing[:5]})"
        )
    return cases


def _upper(row: Mapping[str, str]) -> dict[str, str]:
    return {key.upper(): value for key, value in row.items()}


def _local(raw: str) -> datetime:
    return datetime.fromisoformat(raw).replace(tzinfo=ZoneInfo(TZ_LOCAL))


def _flag(raw: str | None) -> str:
    return {"True": "yes", "False": "no"}.get((raw or "").strip(), "—")


def unit_timeline(
    trans_rows: Sequence[Mapping[str, str]],
    an: str,
    reqno_events: Sequence[ReportRow],
) -> tuple[list[dict[str, str]], int]:
    """The REQNO's issued units as timeline rows, and its reserved-not-issued
    count.

    A unit belongs to the latest event whose issue time is at or before the
    unit's: clusters are contiguous in issue order and each event starts at
    its first unit's issue time (plan 1.1-1.2).
    """
    ledger = units_from_ledger([_upper(r) for r in trans_rows], an)
    starts = [(_local(row["issue_datetime_local"]), row) for row in reqno_events]
    members: dict[str, list[IssuedUnit]] = defaultdict(list)
    for unit in ledger.issued:
        owners = [row for start, row in starts if start <= unit.issue_at]
        if not owners:
            raise ValueError(
                f"an issued unit of REQNO {reqno_events[0]['reqno']} precedes its "
                "first event; the bundle ledger and report.csv disagree"
            )
        members[owners[-1]["audit_id"]].append(unit)
    for row in reqno_events:
        if len(members[row["audit_id"]]) != int(row["issue_units"]):
            raise ValueError(
                f"REQNO {row['reqno']} event {row['event_seq']}: the bundle ledger "
                f"gives {len(members[row['audit_id']])} unit(s), report.csv says "
                f"{row['issue_units']}"
            )
    rows: list[dict[str, str]] = []
    for row in reqno_events:
        for unit in members[row["audit_id"]]:
            rows.append(
                {
                    "Unit": f"{len(rows) + 1} ({unit.bdtype})",
                    "Issued": _fmt_utc_local(unit.issue_at),
                    "Status": unit.status,
                    "Returned": (
                        _fmt_utc_local(unit.returned_at)
                        if unit.returned and unit.returned_at is not None
                        else ""
                    ),
                    "Event": f"{row['event_seq']} of {row['events_in_reqno']}",
                    "Linkage": row.get("linkage_rule") or "",
                    "Linked REQNO": row.get("linked_reqno") or "",
                    "DCTREQ differs": _flag(row.get("attributed_doctor_differs")),
                    "Declared use applies": _flag(row.get("declared_use_applies")),
                    "Declared use": _declared_use(row),
                    "_seq": row["event_seq"],
                }
            )
    return rows, ledger.units_reserved_not_issued


def _declared_use(row: ReportRow) -> str:
    code = (row.get("declared_use_code") or "").strip()
    label = (row.get("declared_use_label") or "").strip()
    return f"{code} {label}".strip() or "—"
