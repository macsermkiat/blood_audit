"""Issue events as both pilot legs build and judge them (blood_audit #276).

``run_pipeline.py`` (deterministic leg) and ``run_llm_leg.py`` (LLM leg) must
judge the same events: the same orders split and link them, the same pool
holds the #249 prior transfusions, and the same notes window feeds the
all-returned guard. Each piece lives here once so the legs cannot drift.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime

from bba.audit_orders import (
    AuditOrdersConfig,
    BloodOrderInput,
    FilterResult,
    build_audit_orders,
)
from bba.deterministic_classifier import ClassifierResult, classify
from bba.deterministic_classifier.models import ClassifierInputs
from bba.hb_lookup import event_notes_window
from bba.issue_events import (
    AuditEvent,
    AuditEvents,
    CandidateOrder,
    build_audit_events,
    candidate_order,
    units_from_ledger,
)
from bba.returns_ledger import ReturnsSummary
from bba.vitals_extractor import PeriopSummary, VitalsNote, scan_periop

from _hosxp_dt import _parse_hosxp_date, _parse_time
from _periop_operations import operations_by_an

# Plan 4.1 base token: the 4 h issue-event window and the own-REQNO linkage
# rule are part of the code identity. Changing either, or the event facts the
# LLM prompt carries, needs a new token and a fresh BBA_PILOT_RUN_ID.
EVENT_CODE_VERSION = "pilot-event-1+gap4h+link-own"


def _icd_codes(*raws: str | None) -> tuple[str, ...]:
    out: list[str] = []
    for raw in raws:
        if not raw:
            continue
        for chunk in raw.replace(";", ",").split(","):
            code = chunk.strip()
            if code:
                out.append(code)
    return tuple(out)


def blood_order_input(
    r: Mapping[str, str],
    products_by_reqno: Mapping[str, Sequence[str]],
    diag_by_an: Mapping[str, Sequence[str]],
) -> BloodOrderInput:
    """One BDVST row as an audit_orders input."""
    reqno = r["REQNO"]
    an = (r.get("AN") or "").strip() or None
    return BloodOrderInput(
        hn=r["HN"],
        an=an,
        reqno=reqno,
        bdvstst=(r.get("BDVSTST") or "").strip(),
        reqtype=(r.get("REQTYPE") or "").strip(),
        canceldate=(r.get("CANCELDATE") or "").strip() or None,
        req_date=_parse_hosxp_date(r.get("REQDATE") or ""),
        req_time=_parse_time(r.get("REQTIME") or ""),
        bdvst_date=_parse_hosxp_date(r.get("BDVSTDATE") or ""),
        bdvst_time=_parse_time(r.get("BDVSTTIME") or ""),
        products=tuple(products_by_reqno.get(reqno, [])),
        diagnosis_codes=_icd_codes(
            *([*diag_by_an.get(an or "", [])] + [r.get("ICD10")])
        ),
    )


def related_orders(
    bdvst_related: Sequence[Mapping[str, str]],
    bdvstdt_related: Sequence[Mapping[str, str]],
    diag_by_an: Mapping[str, Sequence[str]],
    *,
    audited_reqnos: set[str],
    collapse_usetype: Callable[[list[str]], str | None],
) -> tuple[dict[str, BloodOrderInput], dict[str, str | None]]:
    """Every order on the run's admissions, any status and month, by REQNO.

    These orders split and link issue events and hold the #249 cutoff pool.
    The second map is each order's collapsed USETYPE, joined on (HN, REQNO):
    a REQNO recurs across admissions, so a bare REQNO join could attach
    another patient's declaration. Exits when the bundle was cut without
    status-1 (pending) orders, lists a REQNO twice, or misses an audited one.
    """
    if not any((r.get("BDVSTST") or "").strip() == "1" for r in bdvst_related):
        sys.exit(
            "BDVST_RELATED.csv has no status-1 (pending) order: the bundle was "
            "cut without them, so fresh orders would silently link as standing "
            "orders; rebuild it with sample_bundle.py"
        )
    products: dict[str, list[str]] = defaultdict(list)
    usetype_values: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in bdvstdt_related:
        products[r["REQNO"]].append((r.get("BDTYPE") or "").strip())
        usetype_values[((r.get("HN") or "").strip(), r["REQNO"])].append(
            (r.get("USETYPE") or "").strip()
        )
    related: dict[str, BloodOrderInput] = {}
    for r in bdvst_related:
        if r["REQNO"] in related:
            sys.exit(f"BDVST_RELATED.csv lists REQNO {r['REQNO']} twice")
        related[r["REQNO"]] = blood_order_input(r, products, diag_by_an)
    missing = sorted(audited_reqnos - set(related))
    if missing:
        sys.exit(f"audited REQNOs missing from BDVST_RELATED.csv: {missing[:10]}")
    usetypes = {
        reqno: collapse_usetype(usetype_values.get((record.hn.strip(), reqno), []))
        for reqno, record in related.items()
    }
    return related, usetypes


def operation_times(
    iptsumoprt: Sequence[Mapping[str, object]],
    incpt: Sequence[Mapping[str, object]],
) -> dict[str, list[datetime]]:
    """Each admission's operation times under owner decision 8.15 (c)."""
    return {
        an: [op.at for op in ops]
        for an, ops in operations_by_an(iptsumoprt, incpt).items()
    }


def build_issue_events(
    audited: FilterResult,
    related: Mapping[str, BloodOrderInput],
    bdvsttrans: Sequence[Mapping[str, str]],
    *,
    usetypes: Mapping[str, str | None],
    operations: Mapping[str, Sequence[datetime]],
) -> tuple[AuditEvents, AuditEvents]:
    """This run's audit records, and the bundle's other audit records.

    ``related`` holds every order on the run's admissions (BDVST_RELATED,
    any status, any month). Each one that can have caused an issue splits
    and links events. The eligible ones outside this run (another REQDATE
    month, or not sampled) become audit records too; they are never
    reported, only searched for a #249 prior transfusion (plan revision log
    PR 4: a per-month pool silently loses cross-month cutoffs).
    """
    candidates: dict[str, CandidateOrder] = {
        reqno: c
        for reqno, record in related.items()
        if (c := candidate_order(record)) is not None
    }
    run_reqnos = {o.reqno for o in audited.included} | {
        x.reqno for x in audited.excluded
    }
    others = FilterResult(
        included=build_audit_orders(
            [related[q] for q in candidates if q not in run_reqnos],
            AuditOrdersConfig(code_version=EVENT_CODE_VERSION),
        ).included,
        excluded=(),
    )
    trans_by_reqno: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for r in bdvsttrans:
        trans_by_reqno[r.get("REQNO") or ""].append(r)

    def records(orders: FilterResult) -> AuditEvents:
        reqnos = [o.reqno for o in orders.included] + [x.reqno for x in orders.excluded]
        ledgers = {
            q: units_from_ledger(trans_by_reqno.get(q, []), candidates[q].an)
            for q in reqnos
            if q in candidates
        }
        return build_audit_events(
            orders,
            ledgers,
            candidates.values(),
            usetypes=usetypes,
            operations=operations,
        )

    return records(audited), records(others)


def events_by_admission(*records: AuditEvents) -> dict[str, list[AuditEvent]]:
    """Every event of ``records`` per admission: the #249 and notes-window pool."""
    by_an: dict[str, list[AuditEvent]] = defaultdict(list)
    for group in records:
        for event in group.events:
            by_an[event.order.an].append(event)
    return by_an


def event_window_periop(
    notes: Sequence[VitalsNote],
    events_on_an: Sequence[AuditEvent],
    event: AuditEvent,
    returns_summary: ReturnsSummary | None,
) -> PeriopSummary | None:
    """Peri-op findings in an all-returned event's own notes window, else None."""
    if returns_summary is None or returns_summary.disposition != "not_transfused":
        return None
    start, end = event_notes_window(events_on_an, event)
    return scan_periop([n for n in notes if start <= n.timestamp < end])


def classify_with_event_notes_guard(
    inputs: ClassifierInputs, event_periop: PeriopSummary | None
) -> ClassifierResult:
    """``classify``, with the all-returned notes guard reading ``event_periop``.

    Owner decision 2026-11-24: only notes charted in the event's own window
    (:func:`event_notes_window`) can keep an all-returned event from
    RETURNED_NOT_TRANSFUSED. The guard is the classifier's first rule, so a
    release returns before any other rule reads the notes; a hold re-runs on
    ``inputs``, so every later rule still reads the leg's own peri-op.
    """
    if event_periop is not None:
        released = classify(
            inputs.model_copy(
                update={
                    "periop_intraop_transfusion": event_periop.intraop_transfusion,
                    "periop_blood_loss_ml": event_periop.blood_loss_ml,
                }
            )
        )
        if released.classification == "RETURNED_NOT_TRANSFUSED":
            return released
    return classify(inputs)
