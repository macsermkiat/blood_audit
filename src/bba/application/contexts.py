"""Compose canonical row contexts without placing raw narrative on a transport."""

from __future__ import annotations

import hashlib
import hmac
from collections import defaultdict
from datetime import timedelta

from bba import feature_flags
from bba.application.config import ApplicationConfig
from bba.application.hosxp import (
    Tables,
    anc,
    details,
    hb_observations,
    medications,
    moment,
    notes,
    operations,
    orders,
    platelet_observations,
    scoped,
)
from bba.audit_orders import AuditOrder
from bba.audit_pipeline import PipelineRowContext
from bba.cohort_detector import BloodOrderEvent, CohortInputs, assign_cohort
from bba.component_map import ComponentFamily, component_of_code
from bba.declared_use import DeclaredUse, collapse_usetype
from bba.deid_redactor.admission import PrivacyArtifact
from bba.deterministic_classifier import (
    NON_BLOOD_PROCEDURE_ICD9,
    is_blood_requiring_procedure,
    total_crystalloid_liters,
)
from bba.evidence_bundle_builder import (
    DiagnosisRecord,
    EvidenceInputs,
    HbRecord,
    MedRecord,
    OrderAnchor,
    PlateletRecord,
    build_evidence_bundle,
)
from bba.hb_lookup import (
    AnchorCandidate,
    resolve_evidence_anchor,
    resolve_hb_with_fallback,
)
from bba.platelet_lookup import lookup_platelet, project_24h
from bba.preop_reservation import (
    ReservationDecision,
    evaluate_platelet_reservation,
    evaluate_reservation,
    load_msbos_reference,
    reserved_units_by_component,
)
from bba.preop_reservation.bridge import load_oprtact_bridge
from bba.preop_reservation.planned_op import (
    AMBIGUOUS_PLANNED_OP_SENTINEL,
    finalize_planned_op,
    planned_op_v2_for_events,
)
from bba.returns_ledger import rows_for_admission, summarize_returns
from bba.vitals_extractor import VitalsNote, extract_vitals


def _identity(key: bytes, namespace: str, raw: str) -> str:
    return hmac.new(key, f"{namespace}:{raw}".encode(), hashlib.sha256).hexdigest()


def _blood_history(
    tables: Tables, included: tuple[AuditOrder, ...]
) -> tuple[BloodOrderEvent, ...]:
    result = []
    for prior in included:
        if prior.component != "red_cell":
            continue
        lines = details(tables, prior.hn, prior.reqno)
        amounts = []
        for line in lines:
            if component_of_code(line["BDTYPE"]) == ComponentFamily.RED_CELL:
                try:
                    amount = int(line["UNITAMT"])
                except ValueError:
                    continue
                if amount > 0:
                    amounts.append(amount)
        families = {component_of_code(line["BDTYPE"]) for line in lines}
        result.append(
            BloodOrderEvent(
                timestamp=prior.order_datetime,
                rbc_units=sum(amounts),
                co_ordered_with_ffp=ComponentFamily.FFP in families,
                co_ordered_with_platelets=ComponentFamily.PLATELET in families,
            )
        )
    return tuple(result)


def build_contexts(
    tables: Tables,
    *,
    config: ApplicationConfig,
    version: str,
    identity_key: bytes,
    artifacts: dict[str, PrivacyArtifact],
) -> tuple[PipelineRowContext, ...]:
    included = orders(tables, version)
    reserved = reserved_units_by_component(tables["BDVSTDT"])
    reference = (
        load_msbos_reference() if feature_flags.MSBOS_RESERVATION_ENABLED else None
    )
    bridge = (
        load_oprtact_bridge()
        if feature_flags.MSBOS_RESERVATION_ENABLED
        and feature_flags.MSBOS_PLANNED_OP_PICKER_V2_ENABLED
        else None
    )
    by_admission: dict[tuple[str, str], list[AuditOrder]] = defaultdict(list)
    for order in included:
        by_admission[(order.hn, order.an)].append(order)
    histories = {
        key: _blood_history(tables, tuple(group)) for key, group in by_admission.items()
    }
    contexts = []
    for order in included:
        lines = details(tables, order.hn, order.reqno)
        candidates = tuple(
            AnchorCandidate(
                anchor_utc=min(times), display=min(times).isoformat(), reason=reason
            )
            for date_col, time_col, reason in (
                ("USEDATE", "USETIME", "issue_datetime"),
                ("BDVSTDATE", "BDVSTTIME", "blood_bank_visit_fallback"),
            )
            if (
                times := tuple(
                    at
                    for row in lines
                    if row[date_col] and row[time_col]
                    if (at := moment(row[date_col], row[time_col])) is not None
                )
            )
        )
        anchor = resolve_evidence_anchor(
            order_datetime=order.order_datetime, candidates=candidates
        ).anchor_utc
        labs = scoped(tables, "Lab", order)
        hb_obs = hb_observations(labs)
        plt_obs = platelet_observations(labs)
        hb, _, _ = resolve_hb_with_fallback(
            observations=hb_obs, order_datetime=anchor, candidates=candidates
        )
        plt = lookup_platelet(observations=plt_obs, anchor_utc=anchor)
        ops = operations(tables, order)
        meds = medications(scoped(tables, "Med", order))
        history = histories[(order.hn, order.an)]
        cohort = assign_cohort(
            CohortInputs(
                audit_id=order.audit_id,
                hn=order.hn,
                an=order.an,
                order_datetime=order.order_datetime,
                procedure_events=ops,
                diagnosis_codes=order.diagnosis_codes,
                med_events=meds,
                blood_orders=history,
                anc_value=anc(labs, order.order_datetime),
            )
        )
        progress, focus = notes(tables, order)
        vital_notes = tuple(
            VitalsNote(source="IPDADMPROGRESS", timestamp=n.timestamp, text=n.text)
            for n in progress
        ) + tuple(
            VitalsNote(source="IPDNRFOCUSDT", timestamp=n.timestamp, text=n.text)
            for n in focus
        )
        vitals = extract_vitals(anchor=anchor, notes=vital_notes)
        code = (
            collapse_usetype(row["USETYPE"] for row in lines)
            if feature_flags.DECLARED_USETYPE_ENABLED
            else None
        )
        declared = DeclaredUse.from_code(code) if code else None
        hn_hash = _identity(identity_key, "hn", order.hn)
        an_hash = _identity(identity_key, "an", order.an)
        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(
                anchor=OrderAnchor(
                    order_datetime=order.order_datetime,
                    hn_hash=hn_hash,
                    an_hash=an_hash,
                    products=order.products_ordered,
                    hb_anchor=hb.datetime_utc,
                    window_anchor=anchor,
                ),
                component=order.component,
                diagnoses=tuple(
                    DiagnosisRecord(icd10=c) for c in order.diagnosis_codes
                ),
                progress_notes=progress,
                focus_notes=focus,
                meds=tuple(MedRecord(timestamp=m.timestamp, drug=m.drug) for m in meds),
                hb_history=tuple(
                    HbRecord(
                        timestamp=h.datetime_utc,
                        value_g_dl=h.value_g_dl,
                        source=h.source,
                        item_no=h.item_no,
                    )
                    for h in hb_obs
                ),
                platelet_history=tuple(
                    PlateletRecord(
                        timestamp=p.datetime_utc,
                        value_k_ul=p.value_k_ul,
                        source=p.source,
                        item_no=p.item_no,
                    )
                    for p in plt_obs
                )
                if order.component == "platelet"
                else (),
                declared_use=declared,
            )
        )
        valid_ops = tuple(o for o in ops if is_blood_requiring_procedure(o.icd9))
        prior_ops = tuple(
            (order.order_datetime - o.operative_datetime).total_seconds() / 3600
            for o in valid_ops
            if o.operative_datetime <= order.order_datetime
        )
        future_ops = tuple(
            (o.operative_datetime - order.order_datetime).total_seconds() / 3600
            for o in valid_ops
            if o.operative_datetime >= order.order_datetime
        )
        ledger = tuple(
            r
            for r in tables.order_rows("BDVSTTRANS", order.hn, order.reqno)
            if r["REQNO"] == order.reqno
            and r["HN"] == order.hn
            and component_of_code(r["BDTYPE"]).value == order.component
        )
        returns = (
            summarize_returns(
                rows_for_admission(ledger, order.an),
                [
                    r["UNITAMT"]
                    for r in lines
                    if component_of_code(r["BDTYPE"]).value == order.component
                ],
            )
            if feature_flags.RETURNS_LEDGER_ENABLED
            else None
        )
        reservation = None
        platelet_reservation = None
        if reference is not None:
            key = (
                order.hn.strip(),
                order.reqno.strip(),
                ComponentFamily(order.component),
            )
            if key in reserved:
                pick = None
                if bridge is not None:
                    planned, pick = planned_op_v2_for_events(
                        ops,
                        order.order_datetime,
                        bridge=bridge,
                        msbos_codes=reference.codes(),
                        approved_non_blood_codes=NON_BLOOD_PROCEDURE_ICD9,
                    )
                else:
                    upcoming = tuple(
                        o for o in ops if o.operative_datetime >= order.order_datetime
                    )
                    nearest = min(
                        (o.operative_datetime for o in upcoming), default=None
                    )
                    codes = frozenset(
                        o.icd9 for o in upcoming if o.operative_datetime == nearest
                    )
                    planned = (
                        next(iter(codes))
                        if len(codes) == 1
                        else AMBIGUOUS_PLANNED_OP_SENTINEL
                        if codes
                        else ""
                    )
                if order.component == "platelet":
                    platelet_reservation = evaluate_platelet_reservation(
                        reserved_units=reserved[key],
                        pre_op_count_k_ul=plt.value_k_ul,
                        planned_icd9_nodot=planned,
                        procedure_groups=reference.groups_for(planned),
                        reference_hash=reference.content_hash,
                    )
                    if pick is not None and bridge is not None:
                        platelet_reservation = finalize_planned_op(
                            platelet_reservation,
                            pick,
                            reference=reference,
                            bridge_hash=bridge.content_hash,
                        )
                else:
                    reservation = (
                        ReservationDecision(
                            reserved_units=reserved[key],
                            reason="ambiguous_planned_op",
                            reference_hash=reference.content_hash,
                        )
                        if planned == AMBIGUOUS_PLANNED_OP_SENTINEL
                        else evaluate_reservation(
                            reserved_units=reserved[key],
                            planned_icd9_nodot=planned,
                            reference=reference,
                        )
                    )
                    if pick is not None and bridge is not None:
                        reservation = finalize_planned_op(
                            reservation,
                            pick,
                            reference=reference,
                            bridge_hash=bridge.content_hash,
                        )
        artifact = artifacts.get(order.audit_id)
        proof = artifact.attestation if artifact else None
        version_meta = proof.redactor_version if proof else None
        context = PipelineRowContext(
            order=order,
            hb_result=hb,
            vitals_result=vitals,
            cohort_assignment=cohort,
            procedure_proximity_hours=min(prior_ops) if prior_ops else None,
            upcoming_procedure_hours=min(future_ops) if future_ops else None,
            crystalloid_liters_prior_4h=total_crystalloid_liters(
                tuple(
                    m
                    for m in meds
                    if any(
                        k in m.drug.lower()
                        for k in (
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
                    )
                ),
                order.order_datetime,
            ),
            hn_hash=hn_hash,
            an_hash=an_hash,
            prior_rbc_units_24h=sum(
                e.rbc_units
                for e in history
                if timedelta(0)
                < order.order_datetime - e.timestamp
                <= timedelta(hours=24)
            ),
            prior_rbc_units_7d=sum(
                e.rbc_units
                for e in history
                if timedelta(0)
                < order.order_datetime - e.timestamp
                <= timedelta(days=7)
            ),
            redactor_version=version_meta.version if version_meta else "not-applied",
            redactor_model_sha=version_meta.model_sha
            if version_meta
            else "unavailable",
            policy_version=config.policy_version,
            prompt_hash="not-submitted",
            evidence_bundle_hash=proof.redaction_hash if proof else bundle.bundle_hash,
            evidence_chunks=artifact.evidence_chunks if artifact else (),
            privacy_attestation=proof,
            periop_summary=bundle.periop_summary,
            administration_summary=bundle.administration_summary,
            returns_summary=returns,
            reservation_decision=reservation,
            platelet_reservation_decision=platelet_reservation,
            declared_use=declared.label if declared else None,
            enable_missing_hb_positive_evidence=config.enable_missing_hb_positive_evidence,
            component=order.component,
            platelet_result=plt if order.component == "platelet" else None,
            platelet_projected_24h_k_ul=project_24h(
                observations=plt_obs, anchor_utc=anchor
            )
            if order.component == "platelet"
            else None,
            platelet_projection_computed=order.component == "platelet",
            platelet_mtp_suppressed=order.component == "platelet"
            and cohort.label.value == "mtp",
        )
        contexts.append(context)
    return tuple(contexts)
