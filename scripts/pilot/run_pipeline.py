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
    CohortAssignment,
    CohortInputs,
    CohortLabel,
    MedEvent,
    OperativeEvent,
    assign_cohort,
)
from bba.deterministic_classifier import (
    ClassifierResult,
    classify,
    is_blood_requiring_procedure,
    periop_envelope,
)
from bba.deterministic_classifier.crystalloid import total_crystalloid_liters
from bba.deterministic_classifier.models import ClassifierInputs
from bba.feature_flags import RETURNS_LEDGER_ENABLED
from bba.hb_lookup import (
    HbLookupResult,
    HbObservation,
    parse_hb_value,
    resolve_evidence_anchor,
    resolve_hb_with_fallback,
)
from bba.ingest.date_parser import parse_kcmh_english_date
from bba.ingest.models import ParsedTimeOfDay
from bba.platelet_classifier import classify_platelet
from bba.platelet_classifier.models import PlateletClassifierInputs
from bba.platelet_lookup import (
    PLATELET_LABEXM,
    PlateletObservation,
    lookup_platelet,
    parse_platelet_count,
)
from bba.returns_ledger import ReturnsSummary, rows_for_admission, summarize_returns
from bba.vitals_extractor import PeriopSummary, scan_periop

from _anchor_candidates import build_anchor_candidates
from _hosxp_dt import (
    _combine,
    _fmt_hosxp_time,
    _fmt_local_datetime,
    _parse_hosxp_date,
    _parse_time,
)
from _periop_notes import vitals_notes_for

WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))
BUNDLE = WORK / "bundle"
# Operator opt-in for the missing-Hb positive-evidence pre-check (MTP /
# peri-procedural auto-APPROPRIATE on no documented Hb). Defaults off because
# the policy is SEED pending clinical sign-off — see ClassifierInputs and
# docs/CONTEXT.md §"Missing-Hb positive-evidence pre-check".
ENABLE_MISSING_HB_POSITIVE_EVIDENCE = os.environ.get(
    "BBA_PILOT_ENABLE_MISSING_HB_POSITIVE_EVIDENCE", ""
).strip().lower() in ("1", "true", "yes", "on")
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
    "evidence_anchor_reason",
    "evidence_anchor_datetime_local",
    "reanchor_gap_hours",
    "products_ordered",
    "diagnosis_codes_n",
    "hb_anchor_datetime_local",
    "hb_anchor_reason",
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
    "dispense_datetime_local",
    "use_datetime_local",
    "returned_blood_datetime_local",
    "classification",
    "rationale",
    "bypass_reason",
    # Phase 2 platelet columns — empty for red_cell rows.
    "component",
    "platelet_count_k_ul",
    "platelet_freshness",
]

# Returns-ledger disposition columns (spec #119, ticket #120). Appended to the
# report schema ONLY when RETURNS_LEDGER_ENABLED is on, so a flag-off run
# reproduces the base schema — and today's report.csv — byte-for-byte.
RETURNS_LEDGER_FIELDNAMES = [
    "returns_disposition",
    "returns_units_total",
    "returns_units_returned",
    "returns_units_transfused",
    "returns_ordered_unit_amount",
    "returns_ledger_complete",
]


def _returns_disposition_for_classifier(
    returns_summary: ReturnsSummary | None,
) -> str:
    """Return the gated disposition passed into the pure classifier."""
    if RETURNS_LEDGER_ENABLED and returns_summary is not None:
        return returns_summary.disposition
    return "inconclusive"


def _returns_periop_context_for_classifier(
    returns_summary: ReturnsSummary | None,
    *,
    surgical_context: bool,
    intraop_transfusion: bool,
    procedure_proximity_hours: float | None,
    upcoming_procedure_hours: float | None,
) -> bool:
    """Return the gated peri-op envelope passed into the pure classifier (#123).

    Mirrors :func:`_returns_disposition_for_classifier` — off, or with no
    ledger coverage, the classifier sees ``False`` so the exemption cannot
    fire and today's output is unchanged.
    """
    if RETURNS_LEDGER_ENABLED and returns_summary is not None:
        return periop_envelope(
            surgical_context=surgical_context,
            intraop_transfusion=intraop_transfusion,
            procedure_proximity_hours=procedure_proximity_hours,
            upcoming_procedure_hours=upcoming_procedure_hours,
        )
    return False


# Returns-ledger terminals that short-circuit the platelet gate, mirroring
# ``bba.audit_pipeline.pipeline._RETURNS_TERMINAL_CLASSIFICATIONS`` (spec #119).
# Returns are component-agnostic, so an all-returned platelet order skips the
# platelet gate exactly like the RBC path. Kept as a local mirror (not an
# import) for the same reason the leg mirrors the classifier-input helpers.
_RETURNS_TERMINAL_CLASSIFICATIONS = frozenset(
    {"RETURNED_NOT_TRANSFUSED", "PERIOP_TRANSFUSION_EXEMPT"}
)

# Inert Hb / cohort sentinels for the platelet returns-terminal check. A
# platelet order has no Hb or cohort, but ``classify`` requires both on
# :class:`ClassifierInputs`; its returns branch reads neither value (only
# ``cohort.threshold`` for the discarded result field), so these placeholders —
# copied from :meth:`PipelineRowContext.for_platelet` — never affect the
# outcome. See :func:`_platelet_returns_result`.
_PLATELET_SENTINEL_HB = HbLookupResult(
    value_g_dl=None,
    datetime_utc=None,
    source=None,
    freshness="missing",
    delta_hb_bypass=False,
    delta_hb_windows=(),
    needs_review_single_low_hb=False,
)
_PLATELET_SENTINEL_COHORT = CohortAssignment(
    label=CohortLabel.UNKNOWN,
    threshold=None,
    evidence_code=None,
    evidence_name=None,
)


def _platelet_returns_result(
    *,
    audit_id: str,
    order_datetime: datetime,
    returns_summary: ReturnsSummary | None,
    periop: PeriopSummary | None,
) -> ClassifierResult | None:
    """Returns-ledger short-circuit for a platelet order (spec #119).

    Mirrors ``pipeline.run_pipeline``'s platelet branch: run the RBC classifier
    on the order's returns disposition + peri-op envelope and, if it yields a
    returns terminal, use that instead of the platelet gate. Peri-op IS fed —
    matching production's tested contract (``test_platelet_dispatch`` feeds a
    ``periop_summary`` and expects ``PERIOP_TRANSFUSION_EXEMPT``) — so BOTH
    terminals are reachable and, crucially, the hard intra-op/EBL contradiction
    guard stays active: an all-returned platelet whose notes chart an intra-op
    transfusion or EBL >= PERIOP_MIN_EBL_ML falls through instead of being
    falsely cleared. ``procedure_proximity_hours``/``upcoming_procedure_hours``
    are ``None`` (matching :meth:`PipelineRowContext.for_platelet`), so the
    envelope rests only on surgical_context / intra-op transfusion.

    Returns the :class:`ClassifierResult` iff it is a returns terminal, else
    ``None``. Off, or with no ledger coverage, returns ``None`` so the platelet
    path is byte-identical to today.
    """
    if not (RETURNS_LEDGER_ENABLED and returns_summary is not None):
        return None
    surgical_context = periop.surgical_context if periop else False
    intraop_transfusion = periop.intraop_transfusion if periop else False
    result = classify(
        ClassifierInputs(
            audit_id=audit_id,
            hb_result=_PLATELET_SENTINEL_HB,
            cohort_assignment=_PLATELET_SENTINEL_COHORT,
            order_datetime=order_datetime,
            procedure_proximity_hours=None,
            upcoming_procedure_hours=None,
            crystalloid_liters_prior_4h=0.0,
            periop_blood_loss_ml=periop.blood_loss_ml if periop else None,
            periop_intraop_transfusion=intraop_transfusion,
            periop_surgical_context=surgical_context,
            returns_disposition=_returns_disposition_for_classifier(returns_summary),
            returns_periop_context=_returns_periop_context_for_classifier(
                returns_summary,
                surgical_context=surgical_context,
                intraop_transfusion=intraop_transfusion,
                procedure_proximity_hours=None,
                upcoming_procedure_hours=None,
            ),
        )
    )
    if result.classification in _RETURNS_TERMINAL_CLASSIFICATIONS:
        return result
    return None


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


def _read_preferred_optional_csv(*names: str) -> list[dict[str, str]]:
    for name in names:
        path = BUNDLE / name
        if path.exists():
            with path.open(encoding="utf-8", newline="") as fh:
                return list(csv.DictReader(fh))
    return []


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


def _build_platelet_observations(
    lab_rows: list[dict[str, str]], an: str
) -> list[PlateletObservation]:
    obs: list[PlateletObservation] = []
    for i, r in enumerate(lab_rows):
        if r.get("AN") != an:
            continue
        if (r.get("LABEXM") or "").strip() != PLATELET_LABEXM:
            continue
        v = parse_platelet_count(r.get("RESULT"))
        if v is None:
            continue
        dt = _combine(
            _parse_hosxp_date(r.get("LVSTDATE") or ""),
            _parse_time(r.get("LVSTTIME") or ""),
        )
        if dt is None:
            continue
        obs.append(
            PlateletObservation(
                value_k_ul=v,
                datetime_utc=dt,
                source="HEMATOLOGY",
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
        code = (
            r.get("O__OPRTACT") or r.get("INCOME") or r.get("ORDERCODE") or ""
        ).strip()
        source_code = code or "UNMAPPED"
        row_optract = {
            "ICD9CM": (r.get("O__ICD9CM") or "").strip(),
            "ICD9CMADD1": (r.get("O__ICD9CMADD1") or "").strip(),
            "ICD9CMADD2": (r.get("O__ICD9CMADD2") or "").strip(),
            "NAME EN": (r.get("O__NAME_EN") or r.get("O__NAME EN") or "").strip(),
            "NAME": (r.get("O__NAME") or "").strip(),
        }
        optract = (
            row_optract
            if any(row_optract.values())
            else optract_dict.get(source_code, {})
        )
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


def _normalize_bdvsttrans(raw: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize optional BDVSTTRANS rows to uppercase keys."""
    return [{k.upper(): v for k, v in r.items()} for r in raw]


def _earliest_return_datetime_local(rows: list[dict[str, str]]) -> str:
    """Return earliest BDVSTTRANS return timestamp, or empty string."""
    candidates: list[str] = []
    for r in rows:
        returned_at = _fmt_local_datetime(r.get("RTNDATE"), r.get("RTNTIME"))
        if returned_at:
            candidates.append(returned_at)
    return min(candidates) if candidates else ""


def main() -> None:
    if not BUNDLE.exists():
        sys.exit(f"bundle not found: {BUNDLE} (run sample_bundle.py first)")

    bdvst = _read_csv("BDVST.csv")
    bdvstdt = _read_csv("BDVSTDT.csv")
    diag = _read_csv("Diagnosis.csv")
    lab = _read_csv("Lab.csv")
    med = _read_csv("Med.csv")
    # Free-text notes feed the missing-Hb peri-op pre-pass (scan_periop).
    # Optional so the deterministic leg still runs on a bundle without them.
    progress = _read_optional_csv("IPDADMPROGRESS.csv")
    focus = _read_optional_csv("IPDNRFOCUSDT.csv")
    iptsumoprt = _normalize_iptsumoprt(_read_csv("IPTSUMOPRT.csv"))
    ipddchsumoprt = _normalize_iptsumoprt(_read_optional_csv("IPDDCHSUMOPRT.csv"))
    incpt = _normalize_incpt(
        _read_preferred_optional_csv("INCPT_OPRTACT.csv", "INCPT.csv")
    )
    optract_dict = _normalize_optract(_read_optional_csv("OPRTACT.csv"))
    bdvsttrans = _normalize_bdvsttrans(_read_optional_csv("BDVSTTRANS.csv"))
    icd9 = _read_csv("ICD9CM.csv")
    icd9_dict = {
        (r.get("Icd9cm") or "").strip().replace(".", ""): {
            "NAME": (r.get("Name") or "").strip(),
            "ORFLAG": (r.get("Orflag") or "").strip(),
        }
        for r in icd9
    }

    bdvst_by_reqno = {r["REQNO"]: r for r in bdvst}

    # Returns-ledger index: BDVSTTRANS joins audited orders by REQNO exactly
    # (spec #119). One row per dispensed physical unit. Consumed only on the
    # RETURNS_LEDGER_ENABLED path below; inert when the flag is off.
    bdvsttrans_by_reqno: dict[str, list[dict[str, str]]] = {}
    for r in bdvsttrans:
        # Key on the raw REQNO to match order.reqno and every other REQNO index
        # in this script (bdvst_by_reqno, unitamt_lines_by_reqno); a one-sided
        # strip here would silently miss the join.
        bdvsttrans_by_reqno.setdefault(r.get("REQNO") or "", []).append(r)

    products_by_reqno: dict[str, list[str]] = {}
    # Ordered unit amount per REQNO, one raw UNITAMT string per BDVSTDT detail
    # line. summarize_returns parses these fail-closed (spec #119).
    unitamt_lines_by_reqno: dict[str, list[str]] = {}
    # Separate dispense and use displays per REQNO. Dispense comes from the
    # parent BDVST PICKDATE/PICKTIME; use comes from the earliest full BDVSTDT
    # USEDATE/USETIME, with a date-only marker when no full datetime exists.
    dispense_dt_by_reqno: dict[str, str] = {}
    use_dt_by_reqno: dict[str, str] = {}
    exact_use_dt_by_reqno: dict[str, datetime] = {}
    for r in bdvstdt:
        reqno = r["REQNO"]
        products_by_reqno.setdefault(reqno, []).append((r.get("BDTYPE") or "").strip())
        unitamt_lines_by_reqno.setdefault(reqno, []).append(
            (r.get("UNITAMT") or "").strip()
        )

        parent = bdvst_by_reqno.get(reqno, {})
        pick_date = (parent.get("PICKDATE") or "").strip().split(" ")[0]
        pick_time_raw = (parent.get("PICKTIME") or "").strip()
        if reqno not in dispense_dt_by_reqno:
            dispense_dt_by_reqno[reqno] = (
                _fmt_local_datetime(pick_date, pick_time_raw)
                if pick_date and pick_time_raw
                else ""
            )

        use_date = (r.get("USEDATE") or "").strip().split(" ")[0]
        use_time_raw = (r.get("USETIME") or "").strip()
        if use_date and use_time_raw:
            use_time = _fmt_hosxp_time(use_time_raw)
            candidate = f"{use_date} {use_time}"
            use_dt = _combine(_parse_hosxp_date(use_date), _parse_time(use_time_raw))
            if use_dt is not None and (
                reqno not in exact_use_dt_by_reqno
                or use_dt < exact_use_dt_by_reqno[reqno]
            ):
                exact_use_dt_by_reqno[reqno] = use_dt
                use_dt_by_reqno[reqno] = candidate
        if (
            use_date
            and reqno not in exact_use_dt_by_reqno
            and reqno not in use_dt_by_reqno
        ):
            use_dt_by_reqno[reqno] = f"{use_date} (time missing)"

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

    candidates_by_reqno = build_anchor_candidates(
        bdvstdt_rows=bdvstdt, bdvst_by_reqno=bdvst_by_reqno
    )

    rows: list[dict[str, Any]] = []
    for order in filter_result.included:
        # --- Platelet path (Phase 2, component="platelet") ---
        if order.component == "platelet":
            plt_obs = _build_platelet_observations(lab, order.an)
            plt_result = lookup_platelet(
                observations=plt_obs,
                anchor_utc=order.order_datetime,
            )
            # Returns-ledger short-circuit FIRST (mirror pipeline.run_pipeline's
            # platelet branch): an all-returned platelet order skips the platelet
            # gate exactly like the RBC path. Peri-op is scanned admission-wide
            # (same as this leg's RBC path — the accepted #123 Risk #3), so the
            # hard intra-op/EBL contradiction guard stays active; the model leg
            # uses the windowed bundle summary instead. Gated on
            # RETURNS_LEDGER_ENABLED, so a flag-off run never reads the ledger and
            # this row is byte-identical.
            plt_returns_summary: ReturnsSummary | None = None
            plt_periop: PeriopSummary | None = None
            if RETURNS_LEDGER_ENABLED:
                plt_returns_summary = summarize_returns(
                    rows_for_admission(
                        bdvsttrans_by_reqno.get(order.reqno, []), order.an
                    ),
                    unitamt_lines_by_reqno.get(order.reqno, []),
                )
                plt_periop = scan_periop(
                    vitals_notes_for(progress, focus, order.an, order.order_datetime)
                )
            plt_returns_result = _platelet_returns_result(
                audit_id=order.audit_id,
                order_datetime=order.order_datetime,
                returns_summary=plt_returns_summary,
                periop=plt_periop,
            )
            if plt_returns_result is not None:
                plt_classification = plt_returns_result.classification
                plt_rationale = plt_returns_result.rationale
            else:
                plt_clf = classify_platelet(
                    PlateletClassifierInputs(
                        audit_id=order.audit_id,
                        platelet_count=plt_result.value_k_ul,
                    )
                )
                plt_classification = plt_clf.classification
                plt_rationale = plt_clf.rationale
            plt_disp = (
                f"{plt_result.value_k_ul:.0f}"
                if plt_result.value_k_ul is not None
                else "----"
            )
            print(
                f"{order.reqno:<10} {order.an[:20]:<22} "
                f"{'platelet':<24} "
                f"PLT={plt_disp:<5} {plt_result.freshness:<14} "
                f"{'n/a':<5} {plt_classification:<26} {plt_rationale}"
            )
            plt_row: dict[str, Any] = {
                "reqno": order.reqno,
                "an": order.an,
                "order_datetime_utc": order.order_datetime.isoformat(),
                "anchor_imputed": order.anchor_imputed,
                "products_ordered": "|".join(order.products_ordered),
                "diagnosis_codes_n": len(order.diagnosis_codes),
                "component": "platelet",
                "platelet_count_k_ul": plt_result.value_k_ul,
                "platelet_freshness": plt_result.freshness,
                "classification": plt_classification,
                "rationale": plt_rationale,
            }
            if RETURNS_LEDGER_ENABLED and plt_returns_summary is not None:
                plt_row.update(
                    {
                        "returns_disposition": plt_returns_summary.disposition,
                        "returns_units_total": plt_returns_summary.units_total,
                        "returns_units_returned": plt_returns_summary.units_returned,
                        "returns_units_transfused": (
                            plt_returns_summary.units_transfused
                        ),
                        "returns_ordered_unit_amount": (
                            plt_returns_summary.ordered_unit_amount
                            if plt_returns_summary.ordered_unit_amount is not None
                            else ""
                        ),
                        "returns_ledger_complete": plt_returns_summary.ledger_complete,
                    }
                )
            rows.append(plt_row)
            continue

        # --- Red-cell path (Phase 1, component="red_cell") ---
        # Reserve-ahead elective orders are crossmatched days before transfusion;
        # re-anchor the Hb lookback (and the LLM gate's evidence windows) onto
        # the issue datetime so the op-day Hb is what the order is judged on.
        # Cohort, procedure proximity and the classifier itself keep the REQ
        # order anchor — those encode the order-decision context, not evidence.
        evidence_anchor = resolve_evidence_anchor(
            order_datetime=order.order_datetime,
            candidates=candidates_by_reqno.get(order.reqno, []),
        )
        hb_obs = _build_hb_observations(lab, order.an)
        hb, hb_anchor_display, hb_anchor_reason = resolve_hb_with_fallback(
            observations=hb_obs,
            order_datetime=evidence_anchor.anchor_utc,
            candidates=candidates_by_reqno.get(order.reqno, []),
        )
        op_events = _build_op_events(
            iptsumoprt, ipddchsumoprt, incpt, optract_dict, icd9_dict, order.an
        )
        med_events = _build_med_events(med, order.an)
        crystalloid_events = tuple(m for m in med_events if _is_crystalloid(m.drug))
        crystalloid_liters = total_crystalloid_liters(
            crystalloid_events, order.order_datetime
        )
        anc = _latest_anc(lab, order.an, order.order_datetime)

        # Returns-ledger read path (spec #119, ticket #120), behind the flag.
        # Off -> no ledger read, returned datetime stays "" and the returns
        # columns are omitted, so report.csv is byte-identical to today.
        returns_summary: ReturnsSummary | None = None
        returned_dt = ""
        if RETURNS_LEDGER_ENABLED:
            # Scope to THIS order's admission: a REQNO can recur across
            # admissions in the complete export, so an unscoped read could feed a
            # foreign admission's returned units and false-clear the order.
            trans_rows = rows_for_admission(
                bdvsttrans_by_reqno.get(order.reqno, []), order.an
            )
            returns_summary = summarize_returns(
                trans_rows, unitamt_lines_by_reqno.get(order.reqno, [])
            )
            returned_dt = _earliest_return_datetime_local(trans_rows)

        cohort = assign_cohort(
            CohortInputs(
                audit_id=order.audit_id,
                hn=order.hn,
                an=order.an,
                order_datetime=order.order_datetime,
                procedure_events=op_events,
                diagnosis_codes=order.diagnosis_codes,
                med_events=med_events,
                # MTP cluster arm is unfed in the pilot: BDVSTTRANS now carries
                # REQNO (used above for the returns ledger), but this ticket
                # does not build per-order BloodOrderEvent records from it, so
                # detect_mtp_pattern still never fires here. See README "MTP arm
                # is unfed".
                blood_orders=(),
                anc_value=anc,
            )
        )

        # Minor bedside / diagnostic procedures (perm cath, tracheostomy,
        # lumbar puncture, taps, arterial/central lines) are dropped BEFORE
        # deriving proximity — they never justify a transfusion, so they must
        # not fire a peri-procedural / pre-op crossmatch signal. op_events
        # stays unfiltered where it feeds assign_cohort above; the cohort
        # allow-lists gate on OR-flag + surgical prefixes and never match
        # these anyway.
        prior_ops = [
            o
            for o in op_events
            if o.operative_datetime <= order.order_datetime
            and is_blood_requiring_procedure(o.icd9)
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
            o
            for o in op_events
            if o.operative_datetime >= order.order_datetime
            and is_blood_requiring_procedure(o.icd9)
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

        # Peri-op pre-pass evidence (Case 107 extractor). Scanned from the
        # admission's free-text notes; the windowed authority for production
        # is the bundle's periop_summary (audit_pipeline + run_llm_leg path).
        # Here the deterministic-only leg scans the loaded notes so its
        # missing-Hb report mirrors the same hard signals (intra-op
        # transfusion / EBL >= PERIOP_MIN_EBL_ML) the LLM leg would see.
        periop = scan_periop(
            vitals_notes_for(progress, focus, order.an, evidence_anchor.anchor_utc)
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
                enable_missing_hb_positive_evidence=ENABLE_MISSING_HB_POSITIVE_EVIDENCE,
                periop_blood_loss_ml=periop.blood_loss_ml,
                periop_intraop_transfusion=periop.intraop_transfusion,
                periop_surgical_context=periop.surgical_context,
                returns_disposition=_returns_disposition_for_classifier(
                    returns_summary
                ),
                returns_periop_context=_returns_periop_context_for_classifier(
                    returns_summary,
                    surgical_context=periop.surgical_context,
                    intraop_transfusion=periop.intraop_transfusion,
                    procedure_proximity_hours=proximity_h,
                    upcoming_procedure_hours=upcoming_h,
                ),
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

        row: dict[str, Any] = {
            "reqno": order.reqno,
            "an": order.an,
            "order_datetime_utc": order.order_datetime.isoformat(),
            "anchor_imputed": order.anchor_imputed,
            "evidence_anchor_reason": evidence_anchor.reason,
            "evidence_anchor_datetime_local": evidence_anchor.display,
            "reanchor_gap_hours": (
                f"{evidence_anchor.gap_hours:.1f}"
                if evidence_anchor.reason == "issue_reanchor"
                else ""
            ),
            "products_ordered": "|".join(order.products_ordered),
            "diagnosis_codes_n": len(order.diagnosis_codes),
            "hb_anchor_datetime_local": hb_anchor_display,
            "hb_anchor_reason": hb_anchor_reason,
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
            "dispense_datetime_local": dispense_dt_by_reqno.get(order.reqno, ""),
            "use_datetime_local": use_dt_by_reqno.get(order.reqno, ""),
            "returned_blood_datetime_local": returned_dt,
            "classification": clf.classification,
            "rationale": clf.rationale,
            "bypass_reason": clf.bypass_reason.value,
            "component": "red_cell",
        }
        if RETURNS_LEDGER_ENABLED and returns_summary is not None:
            row.update(
                {
                    "returns_disposition": returns_summary.disposition,
                    "returns_units_total": returns_summary.units_total,
                    "returns_units_returned": returns_summary.units_returned,
                    "returns_units_transfused": returns_summary.units_transfused,
                    "returns_ordered_unit_amount": (
                        returns_summary.ordered_unit_amount
                        if returns_summary.ordered_unit_amount is not None
                        else ""
                    ),
                    "returns_ledger_complete": returns_summary.ledger_complete,
                }
            )
        rows.append(row)

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
    fieldnames = REPORT_FIELDNAMES + (
        RETURNS_LEDGER_FIELDNAMES if RETURNS_LEDGER_ENABLED else []
    )
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
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
