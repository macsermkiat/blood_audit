"""Drive the deterministic audit pipeline on a mini bundle.

Composes audit_orders -> hb_lookup -> cohort_detector ->
deterministic_classifier on the sampled cases and prints a per-case
report.

The CLI's ``bba audit`` only ships the ingest leg today; this script
demonstrates the deterministic classification path. For cases that
route to the LLM (POTENTIALLY_INAPPROPRIATE / NEEDS_REVIEW), see the
companion ``run_llm_leg.py``.

Environment variables:

* ``BBA_PILOT_WORK_DIR`` — directory containing the ``bundle/``
  subdirectory written by ``sample_bundle.py`` (default: ``/tmp/bba_mini``).
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from bba.audit_orders import (
    AuditOrdersConfig,
    BloodOrderInput,
    build_audit_orders,
)
from bba.cohort_detector import (
    CohortInputs,
    MedEvent,
    OperativeEvent,
    assign_cohort,
)
from bba.deterministic_classifier import classify
from bba.deterministic_classifier.crystalloid import total_crystalloid_liters
from bba.deterministic_classifier.models import ClassifierInputs
from bba.hb_lookup import HbObservation, lookup_hb, parse_hb_value
from bba.ingest.date_parser import parse_kcmh_english_date
from bba.ingest.models import ParsedTimeOfDay
from bba.ingest.row_timestamp import RowTimestamp
from bba.ingest.time_parser import parse_hosxp_time

WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))
BUNDLE = WORK / "bundle"
TZ_LOCAL = "Asia/Bangkok"
HB_HEM_CODE = "290095"
HB_POCT_CODE = "500001"
ANC_CODE = "290093"
CODE_VERSION = "pilot-mini"
INCPT_OPERATION_GROUPS = {"110", "111"}

CRYSTALLOID_KEYWORDS = (
    "nss",
    "0.9% nacl",
    "0.9 nacl",
    "0.9%nacl",
    "normal saline",
    "rls",
    "ringer",
    "lactated ringer",
    "lrs",
    "plasmalyte",
    "plasma-lyte",
    "d5w",
    "d5/w",
    "d5s",
    "d5%",
    "5% dextrose",
)

# Stable schema for report.csv. Used both for writing per-case rows and
# for emitting an empty header-only file when every sampled case was
# excluded — build_review.py then reads zero rows instead of crashing
# on a missing file (Codex review P2 on PR #67).
REPORT_FIELDNAMES = [
    "reqno",
    "an",
    "order_datetime_utc",
    "anchor_imputed",
    "products_ordered",
    "diagnosis_codes_n",
    "hb_value_g_dl",
    "hb_freshness",
    "hb_source",
    "hb_delta_bypass",
    "hb_needs_review_single_low",
    "cohort_label",
    "cohort_threshold",
    "cohort_evidence_code",
    "cohort_evidence_name",
    "procedure_proximity_hours",
    "upcoming_procedure_hours",
    "crystalloid_liters_prior_4h",
    "anc_value",
    "transfusion_datetime_local",
    "classification",
    "rationale",
    "bypass_reason",
]

csv.field_size_limit(sys.maxsize)


def _read_csv(name: str) -> list[dict[str, str]]:
    with (BUNDLE / name).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_optional_csv(name: str) -> list[dict[str, str]]:
    path = BUNDLE / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
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
    stripped = str(raw).strip()
    if stripped.isdigit() and 1 <= len(stripped) <= 6:
        stripped = stripped.zfill(6)
    return parse_hosxp_time(stripped).value


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


def _build_hb_observations(
    lab_rows: list[dict[str, str]], an: str
) -> list[HbObservation]:
    obs: list[HbObservation] = []
    for i, r in enumerate(lab_rows):
        if r.get("AN") != an:
            continue
        labexm = (r.get("LABEXM") or "").strip()
        if labexm not in {HB_HEM_CODE, HB_POCT_CODE}:
            continue
        v = parse_hb_value(r.get("RESULT"))
        if v is None:
            continue
        dt = _combine(
            _parse_hosxp_date(r.get("LVSTDATE") or ""),
            _parse_time(r.get("LVSTTIME") or ""),
        )
        if dt is None:
            continue
        obs.append(
            HbObservation(
                value_g_dl=v,
                datetime_utc=dt,
                source="HEMATOLOGY" if labexm == HB_HEM_CODE else "POCT",
                item_no=i,
            )
        )
    return obs


def _latest_anc(
    lab_rows: list[dict[str, str]], an: str, anchor: datetime
) -> int | None:
    best: tuple[datetime, int] | None = None
    for r in lab_rows:
        if r.get("AN") != an:
            continue
        if (r.get("LABEXM") or "").strip() != ANC_CODE:
            continue
        raw = (r.get("RESULT") or "").strip()
        try:
            v = float(raw)
        except ValueError:
            continue
        dt = _combine(
            _parse_hosxp_date(r.get("LVSTDATE") or ""),
            _parse_time(r.get("LVSTTIME") or ""),
        )
        if dt is None or dt > anchor:
            continue
        if best is None or dt > best[0]:
            best = (dt, int(v))
    return None if best is None else best[1]


def _build_op_events(
    iptsumoprt: list[dict[str, str]],
    ipddchsumoprt: list[dict[str, str]],
    incpt: list[dict[str, str]],
    optract_dict: dict[str, dict[str, str]],
    icd9_dict: dict[str, dict[str, str]],
    an: str,
) -> tuple[OperativeEvent, ...]:
    out: list[OperativeEvent] = []
    for r in [*iptsumoprt, *ipddchsumoprt]:
        if r.get("AN") != an:
            continue
        icd9 = (r.get("ICD9CM") or "").strip().replace(".", "")
        indate = r.get("INDATE")
        if isinstance(indate, date):
            d = indate
        else:
            date_iso = (indate or "").strip()
            try:
                d = date.fromisoformat(date_iso[:10])
            except ValueError:
                continue
        time_raw = str(r.get("INTIME") or "").strip()
        t = _parse_time(time_raw) or ParsedTimeOfDay(hour=0, minute=0, second=0)
        dt = _combine(d, t)
        if dt is None:
            continue
        meta = icd9_dict.get(icd9, {})
        text_name = (r.get("OPRTTEXT") or "").strip()
        out.append(
            OperativeEvent(
                icd9=icd9,
                or_flag=(
                    (r.get("ORFLAG") or "").strip() == "1"
                    or (meta.get("ORFLAG") or "").strip() == "1"
                ),
                operative_datetime=dt,
                name=text_name or (meta.get("NAME") or "").strip() or None,
            )
        )
    for r in incpt:
        if r.get("AN") != an:
            continue
        if (r.get("CANCELDATE") or "").strip():
            continue
        if (r.get("INCGRP") or "").strip() not in INCPT_OPERATION_GROUPS:
            continue
        code = (r.get("INCOME") or r.get("ORDERCODE") or "").strip()
        source_code = code or "UNMAPPED"
        optract = optract_dict.get(source_code, {})
        optract_codes = tuple(
            c
            for c in (
                (optract.get("ICD9CM") or "").strip().replace(".", ""),
                (optract.get("ICD9CMADD1") or "").strip().replace(".", ""),
                (optract.get("ICD9CMADD2") or "").strip().replace(".", ""),
            )
            if c
        )
        incdate = r.get("INCDATE")
        if isinstance(incdate, date):
            d = incdate
        else:
            try:
                d = date.fromisoformat(str(incdate or "")[:10])
            except ValueError:
                continue
        t = _parse_time(str(r.get("INCTIME") or "")) or ParsedTimeOfDay(
            hour=0,
            minute=0,
            second=0,
        )
        dt = _combine(d, t)
        if dt is None:
            continue
        group = (r.get("INCGRP") or "").strip()
        fallback_name = f"INCPT charge group {group}" if group else "INCPT charge"
        optract_name = (
            (optract.get("NAME EN") or "").strip()
            or (optract.get("NAME") or "").strip()
            or None
        )
        if optract_codes:
            for optract_code in optract_codes:
                meta = icd9_dict.get(optract_code, {})
                out.append(
                    OperativeEvent(
                        icd9=optract_code,
                        or_flag=(meta.get("ORFLAG") or "").strip() == "1",
                        operative_datetime=dt,
                        name=optract_name or (meta.get("NAME") or "").strip() or None,
                    )
                )
            continue
        out.append(
            OperativeEvent(
                # INCPT without an OPRTACT ICD-9 bridge still carries useful
                # operation timing, but remains ineligible for ICD-9 prefix
                # cohort rules.
                icd9=f"INCPT:{source_code}",
                or_flag=False,
                operative_datetime=dt,
                name=optract_name or fallback_name,
            )
        )
    return tuple(out)


def _build_med_events(med_rows: list[dict[str, str]], an: str) -> tuple[MedEvent, ...]:
    out: list[MedEvent] = []
    for r in med_rows:
        if r.get("AN") != an:
            continue
        drug = " ".join(
            filter(
                None,
                [
                    (r.get("NAME_MEDITEM") or "").strip(),
                    (r.get("NAME_GENERIC") or "").strip(),
                    (r.get("STRENGTH") or "").strip(),
                    (r.get("STRENGTHUNIT") or "").strip(),
                    (r.get("MEDUSEQTY") or "").strip(),
                ],
            )
        )
        dt = _combine(
            _parse_hosxp_date(r.get("PRSCDATE") or ""),
            _parse_time(r.get("PRSCTIME") or ""),
        )
        if dt is None or not drug:
            continue
        out.append(MedEvent(drug=drug, timestamp=dt))
    return tuple(out)


def _is_crystalloid(name: str) -> bool:
    lower = name.lower()
    return any(k in lower for k in CRYSTALLOID_KEYWORDS)


def _normalize_iptsumoprt(raw: list[dict[str, str]]) -> list[dict[str, str]]:
    """Mirror the ingest normalize: uppercase column names + parse INDATE."""
    from bba.ingest.date_parser import parse_iptsumoprt_date

    out: list[dict[str, str]] = []
    for r in raw:
        upper = {k.upper(): v for k, v in r.items()}
        indate_raw = upper.get("INDATE", "")
        if indate_raw:
            parsed = parse_iptsumoprt_date(indate_raw)
            if parsed.value is not None:
                upper["INDATE"] = parsed.value
        out.append(upper)
    return out


def _normalize_incpt(raw: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize INCPT column names and date cells for operation lookup."""
    out: list[dict[str, str]] = []
    for r in raw:
        upper = {k.upper(): v for k, v in r.items()}
        incdate_raw = upper.get("INCDATE", "")
        if incdate_raw:
            parsed = parse_kcmh_english_date(incdate_raw)
            if parsed.value is not None:
                upper["INCDATE"] = parsed.value.isoformat()
        out.append(upper)
    return out


def _normalize_optract(raw: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for r in raw:
        upper = {k.upper(): v for k, v in r.items()}
        code = (upper.get("OPRTACT") or "").strip()
        if code:
            out[code] = upper
    return out


def main() -> None:
    if not BUNDLE.exists():
        sys.exit(f"bundle not found: {BUNDLE} (run sample_bundle.py first)")

    bdvst = _read_csv("BDVST.csv")
    bdvstdt = _read_csv("BDVSTDT.csv")
    diag = _read_csv("Diagnosis.csv")
    lab = _read_csv("Lab.csv")
    med = _read_csv("Med.csv")
    iptsumoprt = _normalize_iptsumoprt(_read_csv("IPTSUMOPRT.csv"))
    ipddchsumoprt = _normalize_iptsumoprt(_read_optional_csv("IPDDCHSUMOPRT.csv"))
    incpt = _normalize_incpt(_read_optional_csv("INCPT.csv"))
    optract_dict = _normalize_optract(_read_optional_csv("OPRTACT.csv"))
    icd9 = _read_csv("ICD9CM.csv")
    icd9_dict = {
        (r.get("Icd9cm") or "").strip().replace(".", ""): {
            "NAME": (r.get("Name") or "").strip(),
            "ORFLAG": (r.get("Orflag") or "").strip(),
        }
        for r in icd9
    }

    products_by_reqno: dict[str, list[str]] = {}
    # Earliest USEDATE+USETIME per REQNO — the moment the unit was issued
    # from the blood bank (proxy for transfusion start; stored as local time).
    use_dt_by_reqno: dict[str, str] = {}
    for r in bdvstdt:
        reqno = r["REQNO"]
        products_by_reqno.setdefault(reqno, []).append((r.get("BDTYPE") or "").strip())
        use_date = (r.get("USEDATE") or "").strip().split(" ")[0]
        use_time_raw = (r.get("USETIME") or "").strip()
        if use_date and use_time_raw:
            use_time = (
                f"{use_time_raw[:2]}:{use_time_raw[2:4]}:{use_time_raw[4:6]}"
                if len(use_time_raw) == 6
                else use_time_raw
            )
            candidate = f"{use_date} {use_time}"
            if reqno not in use_dt_by_reqno or candidate < use_dt_by_reqno[reqno]:
                use_dt_by_reqno[reqno] = candidate

    diag_by_an: dict[str, list[str]] = {}
    for r in diag:
        diag_by_an.setdefault(r.get("AN", ""), []).append(
            (r.get("ICD10") or "").strip()
        )

    inputs: list[BloodOrderInput] = []
    for r in bdvst:
        hn = r["HN"]
        reqno = r["REQNO"]
        an = (r.get("AN") or "").strip() or None
        inputs.append(
            BloodOrderInput(
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
                diagnosis_codes=_icd_codes(
                    *(diag_by_an.get(an or "", []) + [r.get("ICD10")])
                ),
            )
        )

    filter_result = build_audit_orders(
        inputs,
        AuditOrdersConfig(code_version=CODE_VERSION),
    )

    print(
        f"\naudit_orders: included={len(filter_result.included)} "
        f"excluded={len(filter_result.excluded)}"
    )
    if filter_result.excluded:
        print("excluded:")
        for ex in filter_result.excluded:
            print(f"  reqno={ex.reqno} reason={ex.reason} detail={ex.detail}")

    print("\n" + "=" * 100)
    print(
        f"{'reqno':<10} {'an':<22} {'cohort':<24} {'hb':<6} {'fresh':<14} "
        f"{'thr':<5} {'classification':<26} {'rationale'}"
    )
    print("=" * 100)

    rows: list[dict[str, Any]] = []
    for order in filter_result.included:
        hb_obs = _build_hb_observations(lab, order.an)
        hb = lookup_hb(observations=hb_obs, anchor_utc=order.order_datetime)
        op_events = _build_op_events(
            iptsumoprt, ipddchsumoprt, incpt, optract_dict, icd9_dict, order.an
        )
        med_events = _build_med_events(med, order.an)
        crystalloid_events = tuple(m for m in med_events if _is_crystalloid(m.drug))
        crystalloid_liters = total_crystalloid_liters(
            crystalloid_events, order.order_datetime
        )
        anc = _latest_anc(lab, order.an, order.order_datetime)

        cohort = assign_cohort(
            CohortInputs(
                audit_id=order.audit_id,
                hn=order.hn,
                an=order.an,
                order_datetime=order.order_datetime,
                procedure_events=op_events,
                diagnosis_codes=order.diagnosis_codes,
                med_events=med_events,
                blood_orders=(),
                anc_value=anc,
            )
        )

        prior_ops = [
            o for o in op_events if o.operative_datetime <= order.order_datetime
        ]
        proximity_h = (
            (
                order.order_datetime
                - max(prior_ops, key=lambda o: o.operative_datetime).operative_datetime
            ).total_seconds()
            / 3600.0
            if prior_ops
            else None
        )
        upcoming_ops = [
            o for o in op_events if o.operative_datetime >= order.order_datetime
        ]
        upcoming_h = (
            (
                min(upcoming_ops, key=lambda o: o.operative_datetime).operative_datetime
                - order.order_datetime
            ).total_seconds()
            / 3600.0
            if upcoming_ops
            else None
        )

        clf = classify(
            ClassifierInputs(
                audit_id=order.audit_id,
                hb_result=hb,
                cohort_assignment=cohort,
                order_datetime=order.order_datetime,
                procedure_proximity_hours=proximity_h,
                upcoming_procedure_hours=upcoming_h,
                crystalloid_liters_prior_4h=crystalloid_liters,
            )
        )

        hb_disp = f"{hb.value_g_dl:.1f}" if hb.value_g_dl is not None else "----"
        thr_disp = f"{cohort.threshold:.1f}" if cohort.threshold is not None else "n/a"
        print(
            f"{order.reqno:<10} {order.an[:20]:<22} "
            f"{cohort.label.value:<24} "
            f"{hb_disp:<6} {hb.freshness:<14} {thr_disp:<5} "
            f"{clf.classification:<26} {clf.rationale}"
        )

        rows.append(
            {
                "reqno": order.reqno,
                "an": order.an,
                "order_datetime_utc": order.order_datetime.isoformat(),
                "anchor_imputed": order.anchor_imputed,
                "products_ordered": "|".join(order.products_ordered),
                "diagnosis_codes_n": len(order.diagnosis_codes),
                "hb_value_g_dl": hb.value_g_dl,
                "hb_freshness": hb.freshness,
                "hb_source": hb.source,
                "hb_delta_bypass": hb.delta_hb_bypass,
                "hb_needs_review_single_low": hb.needs_review_single_low_hb,
                "cohort_label": cohort.label.value,
                "cohort_threshold": cohort.threshold,
                "cohort_evidence_code": cohort.evidence_code,
                "cohort_evidence_name": cohort.evidence_name,
                "procedure_proximity_hours": proximity_h,
                "upcoming_procedure_hours": upcoming_h,
                "crystalloid_liters_prior_4h": crystalloid_liters,
                "anc_value": anc,
                "transfusion_datetime_local": use_dt_by_reqno.get(order.reqno, ""),
                "classification": clf.classification,
                "rationale": clf.rationale,
                "bypass_reason": clf.bypass_reason.value,
            }
        )

    # Append excluded cases as sparse rows so build_review.py can surface
    # the exclusion reason (e.g. "obstetric") instead of showing "—".
    for ex in filter_result.excluded:
        rows.append(
            {"reqno": ex.reqno, "classification": "excluded", "rationale": ex.reason}
        )

    out_csv = WORK / "report.csv"
    # Always write — even an all-excluded sample needs a valid (header-
    # only) report.csv so build_review.py can open it. Stable fieldnames
    # also keep the schema consistent across runs (one column won't
    # disappear if every case happens to have e.g. anc_value=None).
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REPORT_FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nFull report written to {out_csv} ({len(rows)} rows)")

    print("\nClassification summary:")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1
    for k in sorted(counts):
        print(f"  {k:<26} {counts[k]}")

    routed_to_llm = sum(
        1
        for r in rows
        if r["classification"] in {"POTENTIALLY_INAPPROPRIATE", "NEEDS_REVIEW"}
    )
    if routed_to_llm:
        print(
            f"\n{routed_to_llm} case(s) would route to the LLM_REVIEW leg. "
            "Run scripts/pilot/run_llm_leg.py to submit them."
        )


if __name__ == "__main__":
    main()
