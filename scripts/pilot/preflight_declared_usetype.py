"""Read-only go-live preflight for BDVSTDT.USETYPE declared intent.

The report compares the deterministic RBC leg with declared use absent and
present, using the same per-order builders and orchestration as
``run_pipeline.py``.  It writes one JSON evidence artifact and stdout only.  It
does not mutate either declared-use feature flag and never enables the feature.

Environment variables:

* ``BBA_PILOT_WORK_DIR`` — directory containing ``bundle/`` (default
  ``/tmp/bba_mini``).
* ``BBA_PREFLIGHT_OUT`` — JSON artifact path (default
  ``<work>/preflight_declared_usetype.json``).
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bba import feature_flags
from bba.audit_orders import AuditOrdersConfig, BloodOrderInput, build_audit_orders
from bba.cohort_detector import CohortInputs, OperativeEvent, assign_cohort
from bba.declared_use import (
    DECLARED_SURGICAL_LABELS,
    DeclaredUseLabel,
    collapse_usetype,
    label_for,
)
from bba.deterministic_classifier import (
    PRE_OP_CROSSMATCH_WINDOW_HOURS,
    classify,
    is_blood_requiring_procedure,
    periop_envelope,
)
from bba.deterministic_classifier.crystalloid import total_crystalloid_liters
from bba.deterministic_classifier.models import ClassifierInputs, ClassifierResult
from bba.hb_lookup import resolve_evidence_anchor, resolve_hb_with_fallback
from bba.returns_ledger import (
    Disposition,
    ReturnsSummary,
    rows_for_admission,
    summarize_returns,
)
from bba.vitals_extractor import scan_periop

from _anchor_candidates import build_anchor_candidates
from _hosxp_dt import _parse_hosxp_date, _parse_time
from _periop_notes import vitals_notes_for
from preflight_returns_validation import (
    _load_bdvsttrans_by_reqno,
    _resolve_bdvsttrans_path,
)
from run_pipeline import (
    CODE_VERSION,
    ENABLE_MISSING_HB_POSITIVE_EVIDENCE,
    _build_hb_observations,
    _build_med_events,
    _build_op_events,
    _icd_codes,
    _is_crystalloid,
    _latest_anc,
    _normalize_incpt,
    _normalize_iptsumoprt,
    _normalize_optract,
)

csv.field_size_limit(sys.maxsize)

WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))
BUNDLE = WORK / "bundle"
_RETURNS_EXITS = frozenset({"RETURNED_NOT_TRANSFUSED", "PERIOP_TRANSFUSION_EXEMPT"})
_EXPECTED_ON = ("NEEDS_REVIEW", "preop_defer_llm_declared")
_EXPECTED_OFF_TO_BUCKET = {
    ("NEEDS_REVIEW", "hb_7_to_10"): "bucket_rationale_rename",
    ("POTENTIALLY_INAPPROPRIATE", "hb_ge_10"): "bucket_highhb_to_defer",
    ("APPROPRIATE", "bypass_delta_hb"): "bucket_deltahb_preempt",
}
_REMAINING_REQUIREMENTS = (
    "Remaining go-live requirements: a flag-on LLM-leg comparison, an "
    "attribution/dashboard delta summary, and clinician sign-off on the "
    "hb_ge_10→defer and bypass_delta_hb→defer buckets. The library default "
    "stays OFF regardless."
)

OrderKey = tuple[str, str]


@dataclass(frozen=True)
class MixedUseTypeOrder:
    hn: str
    reqno: str
    codes: tuple[str, ...]


@dataclass(frozen=True)
class UseTypeSummary:
    distribution: dict[str, int]
    raw_code_frequency: dict[str, int]
    mixed_orders: list[MixedUseTypeOrder]


@dataclass(frozen=True)
class CrossHNCollision:
    reqno: str
    hns: tuple[str, ...]


@dataclass(frozen=True)
class OrderDeclaredUse:
    hn: str
    reqno: str
    an: str
    label: DeclaredUseLabel | None


@dataclass(frozen=True)
class FlipFinding:
    hn: str
    reqno: str
    declared_use: DeclaredUseLabel | None
    from_classification: str
    from_rationale: str
    to_classification: str
    to_rationale: str
    bucket: str
    structured_upcoming: bool
    unexpected_reason: str | None = None


@dataclass(frozen=True)
class PreflightResult:
    orders_total: int
    orders_audited: int
    orders_excluded: int
    orders_red_cell: int
    declared_surgical_orders: int
    usetype_distribution: dict[str, int]
    raw_code_frequency: dict[str, int]
    mixed_hn_reqno_count: int
    mixed_orders: list[MixedUseTypeOrder]
    cross_hn_reqno_collisions: list[CrossHNCollision]
    returns_exit_count: int
    returns_exit_reqnos: list[str]
    flip_count: int
    flip_bucket_counts: dict[str, int]
    flips: list[FlipFinding]
    unexpected_flips: list[FlipFinding]
    incremental_signal_count: int
    incremental_signal_reqnos: list[str]
    recommendation: str
    invariant_violations: list[FlipFinding] = field(default_factory=list)


def summarize_usetype(
    *,
    audited_keys: Sequence[OrderKey],
    values_by_hn_reqno: Mapping[OrderKey, Sequence[str]],
) -> UseTypeSummary:
    """Collapse and tally USETYPE values for the audited ``(HN, REQNO)`` set."""
    distribution: Counter[str] = Counter()
    raw_codes: Counter[str] = Counter()
    mixed: list[MixedUseTypeOrder] = []
    for hn, reqno in audited_keys:
        values = values_by_hn_reqno.get((hn.strip(), reqno), ())
        codes = tuple(sorted({value.strip() for value in values if value.strip()}))
        raw_codes.update(value.strip() for value in values if value.strip())
        if len(codes) > 1:
            mixed.append(MixedUseTypeOrder(hn=hn.strip(), reqno=reqno, codes=codes))
        collapsed = collapse_usetype(values)
        distribution[label_for(collapsed) if collapsed else "none"] += 1
    return UseTypeSummary(
        distribution=dict(sorted(distribution.items())),
        raw_code_frequency=dict(sorted(raw_codes.items())),
        mixed_orders=sorted(mixed, key=lambda item: (item.reqno, item.hn)),
    )


def cross_hn_collisions(
    bdvstdt_rows: Sequence[Mapping[str, str]],
) -> list[CrossHNCollision]:
    """Report REQNOs that occur under more than one non-blank HN."""
    hns_by_reqno: dict[str, set[str]] = {}
    for row in bdvstdt_rows:
        reqno = (row.get("REQNO") or "").strip()
        hn = (row.get("HN") or "").strip()
        if reqno and hn:
            hns_by_reqno.setdefault(reqno, set()).add(hn)
    return [
        CrossHNCollision(reqno=reqno, hns=tuple(sorted(hns)))
        for reqno, hns in sorted(hns_by_reqno.items())
        if len(hns) > 1
    ]


def incremental_signal_reqnos(
    orders: Sequence[OrderDeclaredUse],
    op_events_by_an: Mapping[str, Sequence[OperativeEvent]],
) -> list[str]:
    """Declared-surgical audited orders with no blood-requiring op row at all."""
    reqnos: list[str] = []
    for order in orders:
        if order.label not in DECLARED_SURGICAL_LABELS:
            continue
        has_blood_requiring_op = any(
            is_blood_requiring_procedure(event.icd9)
            for event in op_events_by_an.get(order.an, ())
        )
        if not has_blood_requiring_op:
            reqnos.append(order.reqno)
    return sorted(reqnos)


def bucket_flip(
    *,
    reqno: str,
    hn: str,
    label: DeclaredUseLabel | None,
    structured_upcoming: bool,
    res_off: ClassifierResult,
    res_on: ClassifierResult,
) -> FlipFinding | None:
    """Place a real off/on transition in one expected bucket or ``unexpected``."""
    off_state = (res_off.classification, res_off.rationale)
    on_state = (res_on.classification, res_on.rationale)
    if off_state == on_state:
        return None

    reasons: list[str] = []
    bucket = _EXPECTED_OFF_TO_BUCKET.get(off_state)
    if bucket is None:
        reasons.append(f"off state {off_state!r} is outside the three expected buckets")
    if on_state != _EXPECTED_ON:
        reasons.append(f"on state {on_state!r} is not {_EXPECTED_ON!r}")
    if label not in DECLARED_SURGICAL_LABELS:
        reasons.append(f"flip carried non-surgical label {label!r}")
    if structured_upcoming:
        reasons.append("flip had a structured upcoming procedure in-window")
    if reasons:
        bucket = "unexpected"
    assert bucket is not None
    return FlipFinding(
        hn=hn,
        reqno=reqno,
        declared_use=label,
        from_classification=res_off.classification,
        from_rationale=res_off.rationale,
        to_classification=res_on.classification,
        to_rationale=res_on.rationale,
        bucket=bucket,
        structured_upcoming=structured_upcoming,
        unexpected_reason="; ".join(reasons) or None,
    )


def recommendation(
    *,
    audited_orders: int,
    declared_surgical_orders: int,
    mixed_count: int,
    unexpected_flip_count: int,
) -> str:
    """Return a deterministic GO/HOLD decision plus outstanding go-live work."""
    if audited_orders == 0:
        decision = "HOLD — zero audited orders; there is nothing to assess."
    elif declared_surgical_orders == 0:
        decision = "HOLD — no declared-surgical orders; there is nothing to assess."
    elif mixed_count:
        decision = f"HOLD — {mixed_count} mixed-(HN,REQNO) order(s) found."
    elif unexpected_flip_count:
        decision = f"HOLD — {unexpected_flip_count} unexpected flip(s) found."
    else:
        decision = "GO — deterministic preflight checks passed."
    return f"{decision} {_REMAINING_REQUIREMENTS}"


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
            return _read_optional_csv(name)
    return []


def _build_audit_inputs(
    bdvst: Sequence[Mapping[str, str]],
    products_by_reqno: Mapping[str, list[str]],
    diag_by_an: Mapping[str, list[str]],
) -> list[BloodOrderInput]:
    inputs: list[BloodOrderInput] = []
    for row in bdvst:
        an = (row.get("AN") or "").strip() or None
        reqno = row["REQNO"]
        inputs.append(
            BloodOrderInput(
                hn=row["HN"],
                an=an,
                reqno=reqno,
                bdvstst=(row.get("BDVSTST") or "").strip(),
                reqtype=(row.get("REQTYPE") or "").strip(),
                canceldate=(row.get("CANCELDATE") or "").strip() or None,
                req_date=_parse_hosxp_date(row.get("REQDATE") or ""),
                req_time=_parse_time(row.get("REQTIME") or ""),
                bdvst_date=_parse_hosxp_date(row.get("BDVSTDATE") or ""),
                bdvst_time=_parse_time(row.get("BDVSTTIME") or ""),
                products=tuple(products_by_reqno.get(reqno, ())),
                diagnosis_codes=_icd_codes(
                    *(diag_by_an.get(an or "", []) + [row.get("ICD10")])
                ),
            )
        )
    return inputs


def _returns_inputs(
    summary: ReturnsSummary | None,
    *,
    surgical_context: bool,
    intraop_transfusion: bool,
    procedure_proximity_hours: float | None,
    upcoming_procedure_hours: float | None,
) -> tuple[Disposition, bool]:
    """Mirror the deterministic leg's returns flag gate without mutating it."""
    if feature_flags.RETURNS_LEDGER_ENABLED and summary is not None:
        return (
            summary.disposition,
            periop_envelope(
                surgical_context=surgical_context,
                intraop_transfusion=intraop_transfusion,
                procedure_proximity_hours=procedure_proximity_hours,
                upcoming_procedure_hours=upcoming_procedure_hours,
            ),
        )
    return "inconclusive", False


def run_preflight() -> PreflightResult:
    """Run the offline evidence report over the configured pilot bundle."""
    if not BUNDLE.exists():
        sys.exit(f"bundle not found: {BUNDLE} (run sample_bundle.py first)")

    bdvst = _read_csv("BDVST.csv")
    bdvstdt = _read_csv("BDVSTDT.csv")
    diag = _read_csv("Diagnosis.csv")
    lab = _read_csv("Lab.csv")
    med = _read_csv("Med.csv")
    progress = _read_optional_csv("IPDADMPROGRESS.csv")
    focus = _read_optional_csv("IPDNRFOCUSDT.csv")
    iptsumoprt = _normalize_iptsumoprt(_read_csv("IPTSUMOPRT.csv"))
    ipddchsumoprt = _normalize_iptsumoprt(_read_optional_csv("IPDDCHSUMOPRT.csv"))
    incpt = _normalize_incpt(
        _read_preferred_optional_csv("INCPT_OPRTACT.csv", "INCPT.csv")
    )
    optract_dict = _normalize_optract(_read_optional_csv("OPRTACT.csv"))
    icd9 = _read_csv("ICD9CM.csv")
    icd9_dict = {
        (row.get("Icd9cm") or row.get("ICD9CM") or "").strip().replace(".", ""): {
            "NAME": (row.get("Name") or row.get("NAME") or "").strip(),
            "ORFLAG": (row.get("Orflag") or row.get("ORFLAG") or "").strip(),
        }
        for row in icd9
    }

    bdvst_by_reqno = {row["REQNO"]: row for row in bdvst}
    products_by_reqno: dict[str, list[str]] = {}
    unitamt_lines_by_reqno: dict[str, list[str]] = {}
    usetype_values_by_hn_reqno: dict[OrderKey, list[str]] = {}
    for row in bdvstdt:
        reqno = row["REQNO"]
        hn = (row.get("HN") or "").strip()
        products_by_reqno.setdefault(reqno, []).append(
            (row.get("BDTYPE") or "").strip()
        )
        unitamt_lines_by_reqno.setdefault(reqno, []).append(
            (row.get("UNITAMT") or "").strip()
        )
        usetype_values_by_hn_reqno.setdefault((hn, reqno), []).append(
            (row.get("USETYPE") or "").strip()
        )

    diag_by_an: dict[str, list[str]] = {}
    for row in diag:
        diag_by_an.setdefault(row.get("AN", ""), []).append(
            (row.get("ICD10") or "").strip()
        )
    filter_result = build_audit_orders(
        _build_audit_inputs(bdvst, products_by_reqno, diag_by_an),
        AuditOrdersConfig(code_version=CODE_VERSION),
    )
    audited_keys = [
        ((order.hn or "").strip(), order.reqno) for order in filter_result.included
    ]
    usetype_summary = summarize_usetype(
        audited_keys=audited_keys,
        values_by_hn_reqno=usetype_values_by_hn_reqno,
    )

    trans_by_reqno: dict[str, list[dict[str, str]]] = {}
    if feature_flags.RETURNS_LEDGER_ENABLED:
        ledger_path = _resolve_bdvsttrans_path()
        if not ledger_path.exists():
            sys.exit(f"BDVSTTRANS ledger not found: {ledger_path}")
        trans_by_reqno = _load_bdvsttrans_by_reqno(ledger_path)

    candidates_by_reqno = build_anchor_candidates(
        bdvstdt_rows=bdvstdt, bdvst_by_reqno=bdvst_by_reqno
    )
    declared_orders: list[OrderDeclaredUse] = []
    op_events_by_an: dict[str, tuple[OperativeEvent, ...]] = {}
    flips: list[FlipFinding] = []
    returns_exit_reqnos: list[str] = []
    orders_red_cell = 0

    for order in filter_result.included:
        key = ((order.hn or "").strip(), order.reqno)
        collapsed = collapse_usetype(usetype_values_by_hn_reqno.get(key, ()))
        label = label_for(collapsed) if collapsed else None
        declared_orders.append(
            OrderDeclaredUse(hn=key[0], reqno=order.reqno, an=order.an, label=label)
        )

        op_events = _build_op_events(
            iptsumoprt,
            ipddchsumoprt,
            incpt,
            optract_dict,
            icd9_dict,
            order.an,
        )
        op_events_by_an[order.an] = op_events
        if order.component != "red_cell":
            continue
        orders_red_cell += 1

        evidence_anchor = resolve_evidence_anchor(
            order_datetime=order.order_datetime,
            candidates=candidates_by_reqno.get(order.reqno, ()),
        )
        hb, _hb_anchor_display, _hb_anchor_reason = resolve_hb_with_fallback(
            observations=_build_hb_observations(lab, order.an),
            order_datetime=evidence_anchor.anchor_utc,
            candidates=candidates_by_reqno.get(order.reqno, ()),
        )
        med_events = _build_med_events(med, order.an)
        crystalloid_liters = total_crystalloid_liters(
            tuple(event for event in med_events if _is_crystalloid(event.drug)),
            order.order_datetime,
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
            event
            for event in op_events
            if event.operative_datetime <= order.order_datetime
            and is_blood_requiring_procedure(event.icd9)
        ]
        proximity_h = (
            (
                order.order_datetime
                - max(
                    prior_ops, key=lambda event: event.operative_datetime
                ).operative_datetime
            ).total_seconds()
            / 3600.0
            if prior_ops
            else None
        )
        upcoming_ops = [
            event
            for event in op_events
            if event.operative_datetime >= order.order_datetime
            and is_blood_requiring_procedure(event.icd9)
        ]
        upcoming_h = (
            (
                min(
                    upcoming_ops, key=lambda event: event.operative_datetime
                ).operative_datetime
                - order.order_datetime
            ).total_seconds()
            / 3600.0
            if upcoming_ops
            else None
        )
        periop = scan_periop(
            vitals_notes_for(progress, focus, order.an, evidence_anchor.anchor_utc)
        )

        returns_summary: ReturnsSummary | None = None
        if feature_flags.RETURNS_LEDGER_ENABLED:
            returns_summary = summarize_returns(
                rows_for_admission(trans_by_reqno.get(order.reqno, []), order.an),
                unitamt_lines_by_reqno.get(order.reqno, []),
            )
        returns_disposition, returns_periop_context = _returns_inputs(
            returns_summary,
            surgical_context=periop.surgical_context,
            intraop_transfusion=periop.intraop_transfusion,
            procedure_proximity_hours=proximity_h,
            upcoming_procedure_hours=upcoming_h,
        )
        inputs = ClassifierInputs(
            audit_id=order.audit_id,
            hb_result=hb,
            cohort_assignment=cohort,
            order_datetime=order.order_datetime,
            procedure_proximity_hours=proximity_h,
            upcoming_procedure_hours=upcoming_h,
            crystalloid_liters_prior_4h=crystalloid_liters,
            enable_missing_hb_positive_evidence=(ENABLE_MISSING_HB_POSITIVE_EVIDENCE),
            periop_blood_loss_ml=periop.blood_loss_ml,
            periop_intraop_transfusion=periop.intraop_transfusion,
            periop_surgical_context=periop.surgical_context,
            returns_disposition=returns_disposition,
            returns_periop_context=returns_periop_context,
            declared_use=None,
        )
        res_off = classify(inputs)
        res_on = classify(inputs.model_copy(update={"declared_use": label}))
        if res_off.classification in _RETURNS_EXITS:
            returns_exit_reqnos.append(order.reqno)
            continue
        finding = bucket_flip(
            reqno=order.reqno,
            hn=key[0],
            label=label,
            structured_upcoming=(
                upcoming_h is not None and upcoming_h <= PRE_OP_CROSSMATCH_WINDOW_HOURS
            ),
            res_off=res_off,
            res_on=res_on,
        )
        if finding is not None:
            flips.append(finding)

    incremental_reqnos = incremental_signal_reqnos(declared_orders, op_events_by_an)
    unexpected = [finding for finding in flips if finding.bucket == "unexpected"]
    bucket_counts = Counter(
        finding.bucket for finding in flips if finding.bucket != "unexpected"
    )
    declared_surgical_orders = sum(
        order.label in DECLARED_SURGICAL_LABELS for order in declared_orders
    )
    decision = recommendation(
        audited_orders=len(filter_result.included),
        declared_surgical_orders=declared_surgical_orders,
        mixed_count=len(usetype_summary.mixed_orders),
        unexpected_flip_count=len(unexpected),
    )
    return PreflightResult(
        orders_total=len(bdvst),
        orders_audited=len(filter_result.included),
        orders_excluded=len(filter_result.excluded),
        orders_red_cell=orders_red_cell,
        declared_surgical_orders=declared_surgical_orders,
        usetype_distribution=usetype_summary.distribution,
        raw_code_frequency=usetype_summary.raw_code_frequency,
        mixed_hn_reqno_count=len(usetype_summary.mixed_orders),
        mixed_orders=usetype_summary.mixed_orders,
        cross_hn_reqno_collisions=cross_hn_collisions(bdvstdt),
        returns_exit_count=len(returns_exit_reqnos),
        returns_exit_reqnos=sorted(returns_exit_reqnos),
        flip_count=len(flips),
        flip_bucket_counts=dict(sorted(bucket_counts.items())),
        flips=flips,
        unexpected_flips=unexpected,
        incremental_signal_count=len(incremental_reqnos),
        incremental_signal_reqnos=incremental_reqnos,
        recommendation=decision,
        invariant_violations=unexpected,
    )


def print_report(result: PreflightResult) -> None:
    """Print the human-readable declared-use evidence report."""
    line = "=" * 78
    print(line)
    print("DECLARED-USETYPE GO-LIVE PREFLIGHT (ticket #152)")
    print("READ-ONLY. This report does NOT enable or mutate any feature flag.")
    print(line)
    print(f"bundle                       : {BUNDLE}")
    print(f"orders in bundle             : {result.orders_total}")
    print(
        f"audited orders               : {result.orders_audited} "
        f"(excluded {result.orders_excluded}; RBC {result.orders_red_cell})"
    )
    print(f"declared-surgical orders     : {result.declared_surgical_orders}")
    print(f"collapsed label distribution: {result.usetype_distribution}")
    print(f"raw code frequency           : {result.raw_code_frequency}")
    print(f"mixed (HN,REQNO)             : {result.mixed_hn_reqno_count}")
    for mixed_order in result.mixed_orders:
        print(
            f"  reqno={mixed_order.reqno} hn={mixed_order.hn} "
            f"codes={list(mixed_order.codes)}"
        )
    print(f"cross-HN REQNO collisions    : {len(result.cross_hn_reqno_collisions)}")
    for collision in result.cross_hn_reqno_collisions:
        print(f"  reqno={collision.reqno} hns={list(collision.hns)}")

    print("\n-- Deterministic RBC flip matrix --")
    print(f"returns exits (excluded)     : {result.returns_exit_count}")
    print(f"flips                        : {result.flip_count}")
    print(f"expected bucket counts       : {result.flip_bucket_counts}")
    print(f"UNEXPECTED flips             : {len(result.unexpected_flips)}")
    for flip in result.flips:
        print(
            f"  reqno={flip.reqno} label={flip.declared_use} "
            f"{flip.from_classification}/{flip.from_rationale} -> "
            f"{flip.to_classification}/{flip.to_rationale} "
            f"bucket={flip.bucket}"
        )
        if flip.unexpected_reason:
            print(f"    UNEXPECTED: {flip.unexpected_reason}")

    print("\n-- Incremental signal --")
    print(
        "declared surgery/type_screen with no blood-requiring op row anywhere: "
        f"{result.incremental_signal_count}"
    )
    if result.incremental_signal_reqnos:
        print(f"  reqnos={result.incremental_signal_reqnos}")
    print("\n" + line)
    print("GO / HOLD SUMMARY")
    print(result.recommendation)
    print(line)


def _write_artifact(result: PreflightResult) -> Path:
    out = Path(
        os.environ.get(
            "BBA_PREFLIGHT_OUT", str(WORK / "preflight_declared_usetype.json")
        )
    )
    out.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return out


def main() -> None:
    result = run_preflight()
    print_report(result)
    artifact = _write_artifact(result)
    print(f"\nMachine-readable artifact written to {artifact}")


if __name__ == "__main__":
    main()
