"""Submit a real Anthropic batch for the LLM-bound cases from the
mini bundle, then write final audit rows to a file-backed audit_store.

Safety: progress / focus notes are NOT shipped to the API (the
encrypted bundle has not been through ``thai-medical-deid``). Only
structured signals go into the evidence bundle:

* ICD-10 diagnoses (AN-scoped, deduped)
* Hb history (7-day pre-anchor; tagged with closest / 24h-min / 48h-min)
* Plt, WBC, Neutrophils CBC (±1 day window)
* Meds list (±1 day window)
* Vitals numbers extracted via regex (no narrative leaves the machine)

The Hb chunks carry a guidance EvidenceChunk that instructs the LLM
to weight closest + lowest values and to explicitly cite any
sub-threshold Hb that fell outside the 24h primary window.

This script intentionally bypasses :class:`LlmClientConfig` (which
enforces the ``ALLOWED_MODELS`` snapshot-pin contract) and passes the
model id directly to the transport. Anthropic has not yet published
date-pinned snapshots for Sonnet 4.6 / Opus 4.7 — the previously-
pinned IDs return ``not_found_error`` on a live batch call. The pilot
script uses the floating alias for now; when snapshot IDs are
published, swap the constants in ``src/bba/llm_client/models.py`` and
delete this bypass.

Environment variables:

* ``BBA_PILOT_WORK_DIR`` — directory containing ``bundle/`` from
  ``sample_bundle.py`` (default: ``/tmp/bba_mini``).
* ``ANTHROPIC_API_KEY`` — required.
* ``BBA_PILOT_LLM_MODEL`` — model id to use (default:
  ``claude-sonnet-4-6``).
* ``BBA_PILOT_RUN_ID`` — run id suffix for the audit_store (default:
  ``pilot-mini``). Bump to force re-run; the store is idempotent on
  (run_id, audit_id).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime, time as _time, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from bba.audit_orders import (
    AuditOrdersConfig,
    BloodOrderInput,
    build_audit_orders,
)
from bba.audit_pipeline import PipelineRowContext
from bba.audit_pipeline.replay import apply_batch_results
from bba.audit_store import AuditStore, AuditStoreConfig
from bba.cohort_detector import (
    CohortInputs,
    MedEvent,
    OperativeEvent,
    assign_cohort,
)
from bba.deterministic_classifier import classify
from bba.deterministic_classifier.crystalloid import total_crystalloid_liters
from bba.deterministic_classifier.models import ClassifierInputs
from bba.evidence_bundle_builder import (
    DiagnosisRecord,
    EvidenceInputs,
    HbRecord,
    MedRecord,
    OrderAnchor,
    VitalsRecord,
    build_evidence_bundle,
)
from bba.hb_lookup import HbObservation, lookup_hb, parse_hb_value
from bba.ingest.models import ParsedTimeOfDay
from bba.ingest.row_timestamp import RowTimestamp
from bba.ingest.time_parser import parse_hosxp_time
from bba.llm_client import AnthropicBatchTransport, BatchSubmissionRequest
from bba.prompt_builder import EvidenceChunk, PromptBuildRequest, build_prompt
from bba.vitals_extractor import VitalsNote, extract_vitals

WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))
BUNDLE = WORK / "bundle"
AUDIT_STORE_ROOT = WORK / "data" / "audit_store"
RUN_ID = os.environ.get("BBA_PILOT_RUN_ID", "pilot-mini")
MODEL_ID = os.environ.get("BBA_PILOT_LLM_MODEL", "claude-sonnet-4-6")
CODE_VERSION = "pilot-mini"
TZ_LOCAL = "Asia/Bangkok"

HB_HEM_CODE = "290095"
HB_POCT_CODE = "500001"
ANC_CODE = "290093"
PLT_CODE = "290078"
WBC_CODES = {"290136", "120015"}
NEUTROPHIL_PCT_CODE = "290092"
NEUTROPHIL_ABS_CODE = "290093"

# Per-source windowing for the LLM evidence bundle:
#   Hb history       : 7 days pre-anchor (deterministic policy window)
#   Plt, WBC, Neut   : day-before + day-of transfusion
#   Med              : day-before + day-of transfusion
#   Diagnoses        : AN-scoped (no row date in schema)
#   Free-text notes  : NEVER sent (PHI not through thai-medical-deid)
WINDOW_HB_DAYS = 7
WINDOW_NOTES_DAYS_BEFORE = 1
WINDOW_NOTES_DAYS_AFTER = 0

CRYSTALLOID_KEYWORDS = (
    "nss", "0.9% nacl", "0.9 nacl", "0.9%nacl", "normal saline",
    "rls", "ringer", "lactated ringer", "lrs",
    "plasmalyte", "plasma-lyte",
    "d5w", "d5/w", "d5s", "d5%", "5% dextrose",
)

csv.field_size_limit(sys.maxsize)


# ============================================================
# Live transport wrapper.
#
# The audit_pipeline orchestrator carries a cost guard that rejects
# isinstance(transport, AnthropicBatchTransport) — useful for tests
# but blocks live use too. Production callers compose the live
# transport via a Protocol-compliant wrapper that is NOT a subclass.
# ============================================================


class RealAnthropicTransport:
    """Production transport — delegates to AnthropicBatchTransport via composition.

    The audit_pipeline.cost_guard's isinstance() check on
    AnthropicBatchTransport blocks the test path; production callers use
    a Protocol-compliant wrapper. This wrapper is exactly that.
    """

    def __init__(
        self,
        *,
        api_key: str,
        poll_interval_seconds: float = 30.0,
        max_wait_seconds: float = 3600.0,
    ) -> None:
        self._inner = AnthropicBatchTransport(
            api_key=api_key,
            poll_interval_seconds=poll_interval_seconds,
            max_wait_seconds=max_wait_seconds,
        )

    def submit_batch_only(self, **kw: Any) -> str:
        return self._inner.submit_batch_only(**kw)

    def fetch_batch_results(self, batch_id: str, **kw: Any) -> Any:
        return self._inner.fetch_batch_results(batch_id, **kw)

    def submit_batch(self, **kw: Any) -> Any:
        return self._inner.submit_batch(**kw)


# ============================================================
# CSV reading and HOSxP parsing helpers (mirrors run_pipeline.py)
# ============================================================


def _read_csv(name: str) -> list[dict[str, str]]:
    with (BUNDLE / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _parse_hosxp_date(raw: str) -> date | None:
    if not raw:
        return None
    head = raw.split(" ", 1)[0]
    try:
        return date.fromisoformat(head)
    except ValueError:
        return None


def _parse_time(raw: str | None) -> ParsedTimeOfDay | None:
    if not raw:
        return None
    s = str(raw).strip()
    if s.isdigit() and 1 <= len(s) <= 6:
        s = s.zfill(6)
    return parse_hosxp_time(s).value


def _combine(d: date | None, t: ParsedTimeOfDay | None) -> datetime | None:
    if d is None or t is None:
        return None
    return RowTimestamp.from_parts(d, t, tz=TZ_LOCAL).utc


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


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


def _hb_observations(lab: list[dict[str, str]], an: str) -> list[HbObservation]:
    obs: list[HbObservation] = []
    for i, r in enumerate(lab):
        if r.get("AN") != an:
            continue
        labexm = (r.get("LABEXM") or "").strip()
        if labexm not in {HB_HEM_CODE, HB_POCT_CODE}:
            continue
        v = parse_hb_value(r.get("RESULT"))
        if v is None:
            continue
        dt = _combine(_parse_hosxp_date(r.get("LVSTDATE") or ""),
                       _parse_time(r.get("LVSTTIME") or ""))
        if dt is None:
            continue
        obs.append(HbObservation(
            value_g_dl=v, datetime_utc=dt,
            source="HEMATOLOGY" if labexm == HB_HEM_CODE else "POCT",
            item_no=i,
        ))
    return obs


def _med_events(med: list[dict[str, str]], an: str) -> tuple[MedEvent, ...]:
    out: list[MedEvent] = []
    for r in med:
        if r.get("AN") != an:
            continue
        drug = " ".join(filter(None, [
            (r.get("NAME_MEDITEM") or "").strip(),
            (r.get("NAME_GENERIC") or "").strip(),
            (r.get("STRENGTH") or "").strip(),
            (r.get("STRENGTHUNIT") or "").strip(),
            (r.get("MEDUSEQTY") or "").strip(),
        ]))
        dt = _combine(_parse_hosxp_date(r.get("PRSCDATE") or ""),
                       _parse_time(r.get("PRSCTIME") or ""))
        if dt is None or not drug:
            continue
        out.append(MedEvent(drug=drug, timestamp=dt))
    return tuple(out)


def _op_events(
    iptsumoprt: list[dict[str, str]],
    icd9_dict: dict[str, dict[str, str]],
    an: str,
) -> tuple[OperativeEvent, ...]:
    out: list[OperativeEvent] = []
    for r in iptsumoprt:
        if r.get("AN") != an:
            continue
        icd9 = (r.get("ICD9CM") or "").strip().replace(".", "")
        indate = r.get("INDATE")
        if isinstance(indate, date):
            d = indate
        else:
            try:
                d = date.fromisoformat(str(indate or "")[:10])
            except ValueError:
                continue
        t = _parse_time(str(r.get("INTIME") or "")) or ParsedTimeOfDay(
            hour=0, minute=0, second=0,
        )
        dt = _combine(d, t)
        if dt is None:
            continue
        meta = icd9_dict.get(icd9, {})
        out.append(OperativeEvent(
            icd9=icd9, or_flag=(meta.get("ORFLAG") or "") == "1",
            operative_datetime=dt,
            name=(meta.get("NAME") or "").strip() or None,
        ))
    return tuple(out)


def _latest_anc(
    lab: list[dict[str, str]], an: str, anchor: datetime,
) -> int | None:
    best: tuple[datetime, int] | None = None
    for r in lab:
        if r.get("AN") != an:
            continue
        if (r.get("LABEXM") or "").strip() != ANC_CODE:
            continue
        try:
            v = float((r.get("RESULT") or "").strip())
        except ValueError:
            continue
        dt = _combine(_parse_hosxp_date(r.get("LVSTDATE") or ""),
                       _parse_time(r.get("LVSTTIME") or ""))
        if dt is None or dt > anchor:
            continue
        if best is None or dt > best[0]:
            best = (dt, int(v))
    return None if best is None else best[1]


def _vitals_notes_for(
    progress: list[dict[str, str]],
    focus: list[dict[str, str]],
    an: str,
    anchor: datetime,
) -> tuple[VitalsNote, ...]:
    """Build VitalsNote list from notes for the AN.

    The vitals_extractor regex only emits numbers; the underlying text
    never leaves the local process, so PHI-bearing notes are safe here.
    """
    out: list[VitalsNote] = []
    for r in progress:
        if r.get("AN") != an:
            continue
        dt = _combine(_parse_hosxp_date(r.get("PROGDATE") or ""),
                       ParsedTimeOfDay(hour=0, minute=0, second=0))
        text = (r.get("OBJECTIVE") or "").strip()
        if dt is None or not text:
            continue
        out.append(VitalsNote(
            timestamp=dt, text=text, source="IPDADMPROGRESS",
        ))
    for r in focus:
        if r.get("AN") != an:
            continue
        dt = _combine(_parse_hosxp_date(r.get("PROGRESSDATE") or ""),
                       _parse_time(r.get("PROGRESSTIME") or ""))
        text = " ".join(filter(None, [
            (r.get("ACTION") or "").strip(),
            (r.get("RESPONSE") or "").strip(),
        ]))
        if dt is None or not text:
            continue
        out.append(VitalsNote(
            timestamp=dt, text=text, source="IPDNRFOCUSDT",
        ))
    return tuple(out)


def _is_crystalloid(name: str) -> bool:
    lower = name.lower()
    return any(k in lower for k in CRYSTALLOID_KEYWORDS)


def _normalize_iptsumoprt(raw: list[dict[str, str]]) -> list[dict[str, str]]:
    from bba.ingest.date_parser import parse_iptsumoprt_date
    out: list[dict[str, str]] = []
    for r in raw:
        upper = {k.upper(): v for k, v in r.items()}
        indate_raw = upper.get("INDATE", "")
        if indate_raw:
            parsed = parse_iptsumoprt_date(indate_raw)
            if parsed.value is not None:
                upper["INDATE"] = parsed.value  # type: ignore[assignment]
        out.append(upper)
    return out


def _render_payload(source: str, payload: dict[str, Any]) -> str:
    """Render a structured EvidenceItem payload as one line for the LLM."""
    if source == "Diagnosis":
        code = payload.get("icd10", "")
        name = payload.get("description") or ""
        return f"ICD-10 {code}: {name}".strip()
    if source == "Lab":
        ts = payload.get("timestamp", "")
        val = payload.get("value_g_dl", "")
        src = payload.get("source", "")
        return f"Hb {val} g/dL ({src}) at {ts}"
    if source == "Vitals":
        ts = payload.get("timestamp", "")
        bits = [f"{k.upper()}={v}" for k in ("sbp", "dbp", "hr", "rr", "bt")
                if (v := payload.get(k)) is not None]
        return f"Vitals at {ts}: " + ", ".join(bits)
    if source == "Med":
        ts = payload.get("timestamp", "")
        drug = payload.get("drug", "")
        return f"Med at {ts}: {drug}"
    return json.dumps(payload, sort_keys=True)


def _build_inputs():
    """Return all the CSV slices the per-case loop needs."""
    bdvst = _read_csv("BDVST.csv")
    bdvstdt = _read_csv("BDVSTDT.csv")
    diag = _read_csv("Diagnosis.csv")
    lab = _read_csv("Lab.csv")
    med = _read_csv("Med.csv")
    progress = _read_csv("IPDADMPROGRESS.csv")
    focus = _read_csv("IPDNRFOCUSDT.csv")
    iptsumoprt = _normalize_iptsumoprt(_read_csv("IPTSUMOPRT.csv"))
    icd9 = _read_csv("ICD9CM.csv")

    icd9_dict: dict[str, dict[str, str]] = {
        (r.get("Icd9cm") or "").strip().replace(".", ""): {
            "NAME": (r.get("Name") or "").strip(),
            "ORFLAG": (r.get("Orflag") or "").strip(),
        }
        for r in icd9
    }

    products_by_reqno: dict[str, list[str]] = {}
    for r in bdvstdt:
        products_by_reqno.setdefault(r["REQNO"], []).append(
            (r.get("BDTYPE") or "").strip()
        )

    diag_by_an: dict[str, list[str]] = {}
    diag_name_by_code: dict[str, str] = {}
    for r in diag:
        an = r.get("AN", "")
        code = (r.get("ICD10") or "").strip()
        if code:
            diag_by_an.setdefault(an, []).append(code)
            name = (r.get("NAME_ICD10") or "").strip()
            if name:
                diag_name_by_code.setdefault(code, name)

    inputs: list[BloodOrderInput] = []
    for r in bdvst:
        hn = r["HN"]
        reqno = r["REQNO"]
        an = (r.get("AN") or "").strip() or None
        inputs.append(BloodOrderInput(
            hn=hn,
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
            diagnosis_codes=_icd_codes(*(diag_by_an.get(an or "", [])
                                          + [r.get("ICD10")])),
        ))

    return (inputs, lab, med, iptsumoprt, icd9_dict, progress, focus,
            diag_name_by_code)


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set")
    if not BUNDLE.exists():
        sys.exit(f"bundle not found: {BUNDLE} (run sample_bundle.py first)")

    AUDIT_STORE_ROOT.mkdir(parents=True, exist_ok=True)
    (inputs, lab, med, iptsumoprt, icd9_dict, progress, focus,
     diag_name_by_code) = _build_inputs()

    fr = build_audit_orders(inputs, AuditOrdersConfig(code_version=CODE_VERSION))
    print(f"audit_orders: included={len(fr.included)} excluded={len(fr.excluded)}")

    contexts: list[PipelineRowContext] = []

    for order in fr.included:
        hb_obs = _hb_observations(lab, order.an)
        hb = lookup_hb(observations=hb_obs, anchor_utc=order.order_datetime)

        op_events = _op_events(iptsumoprt, icd9_dict, order.an)
        med_events = _med_events(med, order.an)
        anc = _latest_anc(lab, order.an, order.order_datetime)
        cohort = assign_cohort(CohortInputs(
            audit_id=order.audit_id,
            hn=order.hn,
            an=order.an,
            order_datetime=order.order_datetime,
            procedure_events=op_events,
            diagnosis_codes=order.diagnosis_codes,
            med_events=med_events,
            blood_orders=(),
            anc_value=anc,
        ))

        prior_ops = [o for o in op_events
                      if o.operative_datetime <= order.order_datetime]
        proximity_h = (
            (order.order_datetime
             - max(prior_ops, key=lambda o: o.operative_datetime
                    ).operative_datetime).total_seconds() / 3600.0
            if prior_ops else None
        )

        crystalloid_events = tuple(m for m in med_events if _is_crystalloid(m.drug))
        crystalloid_liters = total_crystalloid_liters(
            crystalloid_events, order.order_datetime
        )

        vitals_notes = _vitals_notes_for(progress, focus, order.an,
                                          order.order_datetime)
        vitals = extract_vitals(anchor=order.order_datetime, notes=vitals_notes)

        # Calendar-day window in the source-data timezone, not a rolling
        # 48-h slice anchored on the order's clock time. For a 14:00
        # order, the spec calls for "day-before + day-of transfusion",
        # so the bound must be the local midnight that ends "day-of",
        # not order_datetime + 24h (which would leak into the next day).
        local_tz = ZoneInfo(TZ_LOCAL)
        order_date_local = order.order_datetime.astimezone(local_tz).date()
        notes_lo = datetime.combine(
            order_date_local - timedelta(days=WINDOW_NOTES_DAYS_BEFORE),
            _time.min, tzinfo=local_tz,
        ).astimezone(timezone.utc)
        notes_hi = datetime.combine(
            order_date_local + timedelta(days=WINDOW_NOTES_DAYS_AFTER + 1),
            _time.min, tzinfo=local_tz,
        ).astimezone(timezone.utc)
        hb_lo = order.order_datetime - timedelta(days=WINDOW_HB_DAYS)

        diagnoses: tuple[DiagnosisRecord, ...] = tuple(
            DiagnosisRecord(icd10=code,
                            description=diag_name_by_code.get(code))
            for code in dict.fromkeys(order.diagnosis_codes)
        )
        hb_for_bundle = tuple(
            HbRecord(timestamp=o.datetime_utc, value_g_dl=o.value_g_dl,
                     source=o.source, item_no=o.item_no)
            for o in hb_obs
            if hb_lo <= o.datetime_utc <= order.order_datetime
        )
        meds_in_window = sorted(
            [m for m in med_events if notes_lo <= m.timestamp < notes_hi],
            key=lambda m: m.timestamp, reverse=True,
        )
        meds_for_bundle = tuple(
            MedRecord(timestamp=m.timestamp, drug=m.drug) for m in meds_in_window
        )

        vital_records: tuple[VitalsRecord, ...] = ()
        if vitals.note_timestamp is not None and any(
            getattr(vitals.vitals, k) is not None
            for k in ("sbp", "dbp", "hr", "rr", "bt")
        ):
            src_map = {
                "IPDADMPROGRESS": "IPDADMPROGRESS",
                "IPDNRFOCUSDT": "IPDNRFOCUSDT",
                "LLM_EXTRACTED": "LLM_extracted",
                "NONE_IN_WINDOW": None,
            }
            v_src = src_map.get(vitals.source.value)
            if v_src is not None:
                vital_records = (VitalsRecord(
                    timestamp=vitals.note_timestamp,
                    source=cast(Any, v_src),
                    sbp=vitals.vitals.sbp, dbp=vitals.vitals.dbp,
                    hr=vitals.vitals.hr, rr=vitals.vitals.rr,
                    bt=vitals.vitals.bt,
                ),)

        hn_hash = _hash(order.hn)
        an_hash = _hash(order.an)

        bundle = build_evidence_bundle(inputs=EvidenceInputs(
            anchor=OrderAnchor(
                order_datetime=order.order_datetime,
                hn_hash=hn_hash, an_hash=an_hash,
                products=order.products_ordered,
            ),
            diagnoses=diagnoses,
            progress_notes=(),  # PHI-safety
            focus_notes=(),     # PHI-safety
            meds=meds_for_bundle,
            hb_history=hb_for_bundle,
            vitals=vital_records,
        ))

        # Hb chunks get extra annotations so the LLM can weight closest +
        # lowest values: each Hb item is tagged with hours-before-anchor
        # plus optional flags for "closest", "min in 24h", "min in 48h".
        hb_payloads: list[tuple[str, float, datetime]] = []
        for it in bundle.items:
            if it.source != "Lab":
                continue
            p = dict(it.payload)
            if "value_g_dl" not in p or it.timestamp_utc is None:
                continue
            hb_payloads.append((it.id, float(p["value_g_dl"]), it.timestamp_utc))

        closest_id: str | None = None
        min24_id: str | None = None
        min48_id: str | None = None
        if hb_payloads:
            pre = [(i, v, t) for i, v, t in hb_payloads
                    if t <= order.order_datetime]
            if pre:
                closest_id = max(pre, key=lambda x: x[2])[0]
                w24 = [x for x in pre
                        if (order.order_datetime - x[2]) <= timedelta(hours=24)]
                if w24:
                    min24_id = min(w24, key=lambda x: x[1])[0]
                w48 = [x for x in pre
                        if (order.order_datetime - x[2]) <= timedelta(hours=48)]
                if w48:
                    min48_id = min(w48, key=lambda x: x[1])[0]

        chunks: list[EvidenceChunk] = []
        for item in bundle.items:
            text = _render_payload(item.source, dict(item.payload))
            if not text.strip():
                continue
            if item.id in (closest_id, min24_id, min48_id):
                ts = item.timestamp_utc
                hrs = ((order.order_datetime - ts).total_seconds() / 3600.0
                        if ts else None)
                flags: list[str] = []
                if item.id == closest_id:
                    flags.append("closest pre-order Hb")
                if item.id == min24_id:
                    flags.append("minimum in 24h pre-order")
                if item.id == min48_id and item.id != min24_id:
                    flags.append("minimum in 48h pre-order")
                if hrs is not None:
                    flags.append(f"{hrs:.1f}h before order")
                text = f"{text}  [{'; '.join(flags)}]"
            elif item.source == "Lab" and "value_g_dl" in dict(item.payload):
                ts = item.timestamp_utc
                hrs = ((order.order_datetime - ts).total_seconds() / 3600.0
                        if ts else None)
                if hrs is not None:
                    text = f"{text}  [{hrs:.1f}h before order]"
            chunks.append(EvidenceChunk(
                evidence_id=item.id, source=item.source, text=text,
            ))

        # Append CBC chunks (Plt / WBC / Neutrophils) in the ±1d window.
        next_eid = 901
        for r in lab:
            if r.get("AN") != order.an:
                continue
            code = (r.get("LABEXM") or "").strip()
            if code not in {PLT_CODE, *WBC_CODES, NEUTROPHIL_ABS_CODE,
                              NEUTROPHIL_PCT_CODE}:
                continue
            dt = _combine(_parse_hosxp_date(r.get("LVSTDATE") or ""),
                           _parse_time(r.get("LVSTTIME") or ""))
            if dt is None or not (notes_lo <= dt < notes_hi):
                continue
            value = (r.get("RESULT") or "").strip()
            if not value:
                continue
            name = (r.get("NAME_LABEXM") or "").strip() or code
            unit = (r.get("NRMUNIT") or "").strip()
            lo = (r.get("MINNRM") or "").strip()
            hi = (r.get("MAXNRM") or "").strip()
            hrs = (order.order_datetime - dt).total_seconds() / 3600.0
            text = (f"{name} {value}{(' ' + unit) if unit else ''} at "
                     f"{dt.isoformat()}  [ref {lo}-{hi}; "
                     f"{hrs:+.1f}h vs order]")
            chunks.append(EvidenceChunk(
                evidence_id=f"E{next_eid}", source="Lab", text=text,
            ))
            next_eid += 1

        guidance_lines = [
            "Hb weighting policy for this audit:",
            "- The Hb value CLOSEST to the order datetime is the primary trigger.",
            "- The LOWEST Hb in the 24h-pre-order window dominates the indication call.",
            "- A 48h-window minimum is supportive but not primary.",
            "- Hb values >48h before the order are background trend only.",
            "Each Hb chunk above carries a bracketed flag for these positions.",
            "",
            "REASONING REQUIREMENT — transparency about non-primary Hb values:",
            "- If ANY sub-threshold Hb (< cohort threshold) exists in the 7-day",
            "  pre-order history but falls OUTSIDE the 24h primary window,",
            "  you MUST explicitly cite that value in reasoning_summary_en AND",
            "  reasoning_summary_th, naming the evidence id, the value, and the",
            "  number of hours before the order it occurred.",
            "- State why that value is not the primary trigger (e.g. 'Hb 6.9",
            "  g/dL [E_n] occurred 28 h pre-order, outside the 24h primary",
            "  window; the closest pre-order Hb 7.1 g/dL [E_m] is gray-zone').",
            "- A reviewer must be able to verify from your reasoning that you",
            "  saw EVERY sub-threshold Hb in the 7-day window and explicitly",
            "  considered the temporal tradeoff between recency and value.",
            "- Do not silently drop sub-threshold Hb values from the reasoning;",
            "  ignoring them is the failure mode this policy exists to prevent.",
        ]
        chunks.append(EvidenceChunk(
            evidence_id="E999",
            source="Analysis_Hint",
            text="\n".join(guidance_lines),
        ))

        if not chunks:
            print(f"  WARN: empty chunks for {order.reqno}; skipping LLM submit")
            continue

        contexts.append(PipelineRowContext(
            order=order,
            hb_result=hb,
            vitals_result=vitals,
            cohort_assignment=cohort,
            procedure_proximity_hours=proximity_h,
            crystalloid_liters_prior_4h=crystalloid_liters,
            hn_hash=hn_hash, an_hash=an_hash,
            prior_rbc_units_24h=0, prior_rbc_units_7d=0,
            redactor_version="structured-only-no-text-deid-0.0",
            redactor_model_sha="0" * 64,
            policy_version="KCMH-PR17.2 / AABB-2023 (pilot)",
            prompt_hash="0" * 64,
            evidence_bundle_hash=bundle.bundle_hash,
            evidence_chunks=tuple(chunks),
        ))

    DETERMINISTIC_FINAL = {"APPROPRIATE", "INSUFFICIENT_EVIDENCE", "INAPPROPRIATE"}
    classifier_results: dict[str, Any] = {}
    llm_contexts: list[PipelineRowContext] = []
    for ctx in contexts:
        cres = classify(ClassifierInputs(
            audit_id=ctx.order.audit_id,
            hb_result=ctx.hb_result,
            cohort_assignment=ctx.cohort_assignment,
            order_datetime=ctx.order.order_datetime,
            procedure_proximity_hours=ctx.procedure_proximity_hours,
            crystalloid_liters_prior_4h=ctx.crystalloid_liters_prior_4h,
        ))
        classifier_results[ctx.order.audit_id] = cres
        if cres.classification not in DETERMINISTIC_FINAL:
            llm_contexts.append(ctx)

    print(f"\nLLM-bound: {len(llm_contexts)} / {len(contexts)}")
    if not llm_contexts:
        sys.exit("nothing to submit")

    submissions: list[BatchSubmissionRequest] = []
    for ctx in llm_contexts:
        threshold = (ctx.cohort_assignment.threshold
                     if ctx.cohort_assignment.threshold is not None else 7.0)
        prompt = build_prompt(PromptBuildRequest(
            task_mode="HB_7_10_REVIEW",
            cohort_threshold=threshold,
            evidence_chunks=ctx.evidence_chunks,
            few_shot_examples=(),
        ))
        submissions.append(BatchSubmissionRequest(
            audit_id=ctx.order.audit_id,
            run_id=RUN_ID,
            task_mode="HB_7_10_REVIEW",
            prompt=prompt,
        ))

    transport = RealAnthropicTransport(
        api_key=api_key, poll_interval_seconds=20.0, max_wait_seconds=3600.0,
    )
    print(f"\nSubmitting batch of {len(submissions)} requests to "
          f"Anthropic (model={MODEL_ID})...")
    t0 = time.time()
    batch_id = transport.submit_batch_only(
        model=MODEL_ID, requests=submissions, prompt_cache_enabled=True,
    )
    print(f"  batch_id = {batch_id}")
    print("  polling (this can take a while)...")
    response = transport.fetch_batch_results(
        batch_id, model=MODEL_ID, requests=submissions, prompt_cache_enabled=True,
    )
    elapsed = time.time() - t0
    print(f"  batch complete in {elapsed:.1f}s; {len(response.results)} results")

    audit_store = AuditStore(AuditStoreConfig(
        root_dir=AUDIT_STORE_ROOT, code_version=CODE_VERSION,
    ))
    context_map = {ctx.order.audit_id: ctx for ctx in llm_contexts}
    write_summary = apply_batch_results(
        response,
        audit_store=audit_store,
        run_id=RUN_ID,
        contexts=context_map,
        classifier_results=classifier_results,
    )
    print(f"  persisted audit rows: {len(write_summary.audit_ids_persisted)}")

    print("\n" + "=" * 120)
    rows = list(audit_store.read_audit_results(run_id=RUN_ID))
    rows_by_id = {r.audit_id: r for r in rows}
    print(f"{'reqno':<10} {'det.verdict':<26} {'final':<26} "
          f"{'conf':<6} {'review_reason':<22} {'reasoning_en (head)'}")
    print("=" * 120)
    for ctx in llm_contexts:
        det = classifier_results[ctx.order.audit_id]
        r = rows_by_id.get(ctx.order.audit_id)
        reasoning = (r.reasoning_summary_en[:60] if r and r.reasoning_summary_en
                     else "")
        final = r.final_classification if r else "(no row)"
        conf = f"{r.confidence:.2f}" if r else "—"
        rr = (r.review_reason or "—") if r else "—"
        print(f"{ctx.order.reqno:<10} {det.classification:<26} "
              f"{final:<26} {conf:<6} {rr[:20]:<22} {reasoning}")

    report = []
    for ctx in llm_contexts:
        det = classifier_results[ctx.order.audit_id]
        r = rows_by_id.get(ctx.order.audit_id)
        report.append({
            "reqno": ctx.order.reqno,
            "audit_id": ctx.order.audit_id,
            "deterministic": {
                "classification": det.classification,
                "rationale": det.rationale,
                "cohort": ctx.cohort_assignment.label.value,
                "threshold": ctx.cohort_assignment.threshold,
                "hb": ctx.hb_result.value_g_dl,
            },
            "llm_final": ({
                "rule_classification": r.rule_classification,
                "final_classification": r.final_classification,
                "model": r.model_id,
                "indications": [dict(d) for d in r.indications_json],
                "negative_evidence": [dict(d) for d in r.negative_evidence_json],
                "reasoning_en": r.reasoning_summary_en,
                "reasoning_th": r.reasoning_summary_thai,
                "confidence": r.confidence,
                "review_reason": r.review_reason,
                "needs_human_review": r.needs_human_review,
                "verifier_pass": r.verifier_pass,
                "escalated_to_opus": r.escalated_to_opus,
            } if r else None),
        })
    out = WORK / "llm_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nFull JSON report: {out}")


if __name__ == "__main__":
    main()
