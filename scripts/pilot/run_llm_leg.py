"""Submit a real Anthropic batch for the LLM-bound cases from the
mini bundle, then write final audit rows to a file-backed audit_store.

De-identification posture (issue #76): de-identification is a FIRST GATE
run OUTSIDE this script — the pilot input is treated as already
de-identified, so this leg does not re-run ``thai-medical-deid`` in
process. The evidence bundle therefore now ships the narrative notes
themselves (progress + focus), not just the regex-extracted vitals
numbers, because Case 2 / REQNO 68012352 showed the LLM was starved of the
MAP / vasopressor evidence that lived in the narrative. Signals shipped:

* ICD-10 diagnoses (AN-scoped, deduped)
* IPDADMPROGRESS / IPDNRFOCUSDT narrative notes (per-source windowed)
* Hemodynamic summary (MAP nadir + vasopressor agent/dose), pinned E1,
  synthesized fact-only from the SAME shipped notes (issue #76)
* Hb history (7-day pre-anchor; tagged with closest / 24h-min / 48h-min)
* Plt, WBC, Neutrophils CBC (±1 day window)
* Meds list (±1 day window)
* Vitals numbers extracted via regex

The Hb chunks carry a guidance EvidenceChunk that instructs the LLM
to weight closest + lowest values and to explicitly cite any
sub-threshold Hb that fell outside the 24h primary window.

This script intentionally bypasses :class:`LlmClientConfig` and passes
the model id directly to the transport. The production allow-set
(``ALLOWED_MODELS``) is pinned to the bare aliases the live API returns
(``claude-sonnet-5`` / ``claude-opus-4-8``), so the echoed model_id
validates natively at row construction — no runtime allow-set patch is
needed here.

Environment variables:

* ``BBA_PILOT_WORK_DIR`` — directory containing ``bundle/`` from
  ``sample_bundle.py`` (default: ``/tmp/bba_mini``).
* ``ANTHROPIC_API_KEY`` — required.
* ``BBA_PILOT_LLM_MODEL`` — model id to use (default:
  ``claude-sonnet-5``).
* ``BBA_PILOT_RUN_ID`` — run id suffix for the audit_store (default:
  ``pilot-mini``). Bump to force re-run; the store is idempotent on
  (run_id, audit_id).
* ``BBA_PILOT_ONLY_REQNO`` — comma-separated REQNOs; process/submit only
  those cases and MERGE their fresh records into the existing
  ``llm_report.json``. Always pair with a fresh ``BBA_PILOT_RUN_ID``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from datetime import date, datetime, time as _time, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import bba.feature_flags as feature_flags
from bba.audit_orders import (
    AuditOrdersConfig,
    BloodOrderInput,
    build_audit_orders,
)
from bba.audit_pipeline import PipelineRowContext
from bba.audit_pipeline.pipeline import (
    _persist_injection_flagged_row,
    _persist_over_reservation_row,
    rbc_task_mode,
)
from bba.audit_pipeline.replay import apply_batch_results, is_over_reservation
from bba.audit_store import AuditStore, AuditStoreConfig
from bba.cohort_detector import (
    CohortAssignment,
    CohortInputs,
    CohortLabel,
    MedEvent,
    OperativeEvent,
    assign_cohort,
)
from bba.component_map import ComponentFamily
from bba.deterministic_classifier import (
    ClassifierResult,
    classify,
    is_blood_requiring_procedure,
    periop_envelope,
)
from bba.deterministic_classifier.crystalloid import total_crystalloid_liters
from bba.deterministic_classifier.models import ClassifierInputs
from bba.declared_use import (
    DeclaredUse,
    DeclaredUseLabel,
    collapse_usetype,
    label_for,
)
from bba.feature_flags import RETURNS_LEDGER_ENABLED
from bba.returns_ledger import ReturnsSummary, rows_for_admission, summarize_returns
from bba.evidence_bundle_builder import (
    DiagnosisRecord,
    EvidenceInputs,
    FocusNote,
    HbRecord,
    MedRecord,
    OrderAnchor,
    PlateletRecord,
    ProgressNote,
    VitalsRecord,
    build_evidence_bundle,
)
from bba.hb_lookup import (
    EvidenceAnchor,
    HbLookupResult,
    HbObservation,
    parse_hb_value,
    resolve_evidence_anchor,
    resolve_hb_with_fallback,
)
from bba.platelet_classifier import classify_platelet
from bba.platelet_classifier.models import PlateletClassifierInputs
from bba.platelet_lookup import (
    PLATELET_LABEXM,
    PlateletObservation,
    lookup_platelet,
    parse_platelet_count,
)
from bba.preop_reservation import (
    ReservationDecision,
    evaluate_reservation,
    load_msbos_reference,
    reserved_units_by_component,
)
from bba.ingest.date_parser import parse_kcmh_english_date
from bba.ingest.models import ParsedTimeOfDay
from bba.llm_client import AnthropicBatchTransport, BatchSubmissionRequest
from bba.prompt_builder import EvidenceChunk, PromptBuildRequest, build_prompt
from bba.vitals_extractor import PeriopSummary, extract_vitals

from _anchor_candidates import build_anchor_candidates
from _bdvsttrans_source import load_bdvsttrans_rows
from _hosxp_dt import _combine, _parse_hosxp_date, _parse_time
from _periop_notes import vitals_notes_for

WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))
BUNDLE = WORK / "bundle"
AUDIT_STORE_ROOT = WORK / "data" / "audit_store"
RUN_ID = os.environ.get("BBA_PILOT_RUN_ID", "pilot-mini")
MODEL_ID = os.environ.get("BBA_PILOT_LLM_MODEL", "claude-sonnet-5")
# Single-case iteration knob: comma-separated REQNOs. When set, only the
# matching orders are processed/submitted, and the fresh records are MERGED
# into the existing llm_report.json (all other cases keep their records)
# instead of overwriting it wholesale. Pair with a fresh BBA_PILOT_RUN_ID —
# the store is idempotent on (run_id, audit_id), so re-using a run id that
# already holds this case would keep the stale row.
ONLY_REQNOS = frozenset(
    v.strip()
    for v in os.environ.get("BBA_PILOT_ONLY_REQNO", "").split(",")
    if v.strip()
)
# Operator opt-in for the missing-Hb positive-evidence pre-check (MTP /
# peri-procedural auto-APPROPRIATE on no documented Hb). Defaults off because
# the policy is SEED pending clinical sign-off — see ClassifierInputs and
# docs/CONTEXT.md §"Missing-Hb positive-evidence pre-check".
ENABLE_MISSING_HB_POSITIVE_EVIDENCE = os.environ.get(
    "BBA_PILOT_ENABLE_MISSING_HB_POSITIVE_EVIDENCE", ""
).strip().lower() in ("1", "true", "yes", "on")
# Declared-use pilot seam (spec #147, ticket #151; go-live 2026-07-15), read at
# import so it can fold into CODE_VERSION below; main() sets
# feature_flags.DECLARED_USETYPE_ENABLED from this same constant. Defaults to the
# library flag (now ON) when BBA_PILOT_DECLARED_USETYPE is unset; the env var
# overrides ("1" forces on, anything else forces off). Read into a plain module
# constant here (the main() assignment is the only feature_flags mutation).
_declared_env = os.environ.get("BBA_PILOT_DECLARED_USETYPE")
DECLARED_USETYPE_PILOT_ENABLED = (
    _declared_env == "1"
    if _declared_env is not None
    else feature_flags.DECLARED_USETYPE_ENABLED
)
_msbos_env = os.environ.get("BBA_PILOT_MSBOS_RESERVATION")
MSBOS_RESERVATION_PILOT_ENABLED = (
    _msbos_env == "1"
    if _msbos_env is not None
    else feature_flags.MSBOS_RESERVATION_ENABLED
)
# Run/code identity (spec #119 §G, ticket #124). The audit_store is idempotent
# on (run_id, audit_id, code_version), so enabling a seam that changes verdicts
# must not silently reuse a flag-off run's committed rows. Folding each seam into
# CODE_VERSION makes enabling it a DISTINCT code identity, so a re-run recomputes
# every affected verdict instead of keeping stale pre-feature rows (AC "reusing a
# stale identity does not leave pre-feature rows in place"). Declared-use flips
# some orders to NEEDS_REVIEW, changing the submission set and verdicts, so it is
# folded in the same way (Codex P2, PR #156). Flag-off keeps the original
# "pilot-mini" identity, so a flag-off run is byte-identical to today. The seams
# are captured once at import and never toggled mid-run, so they are constant
# across this process's batch submit and result apply. A changed BDVSTTRANS
# ledger still needs a fresh BBA_PILOT_RUN_ID (it does not alter the code
# identity); production folds the ledger into run identity separately (#121).
CODE_VERSION = "pilot-mini"
if RETURNS_LEDGER_ENABLED:
    CODE_VERSION += "+returns"
if DECLARED_USETYPE_PILOT_ENABLED:
    CODE_VERSION += "+declared"
if MSBOS_RESERVATION_PILOT_ENABLED:
    CODE_VERSION += "+msbos"
TZ_LOCAL = "Asia/Bangkok"
INCPT_OPERATION_GROUPS = {"110", "111"}

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
#   Free-text notes  : shipped (de-id is a first gate run OUTSIDE this
#                      script; builder re-windows to its own ±24h)
WINDOW_HB_DAYS = 7
WINDOW_NOTES_DAYS_BEFORE = 1
WINDOW_NOTES_DAYS_AFTER = 0
# Re-anchored reserve-ahead orders only: the transfusion plays out across the
# op day, so the intra-op nadir and the morning-after Hb are drawn AFTER the
# USE issue time. Extend the Hb history upper bound this many calendar days
# past the op day so the LLM sees the full peri-transfusion trajectory, not
# just the stale value at issue time. Non-reanchored orders keep the strict
# pre-order upper bound (no forward extension).
WINDOW_HB_REANCHOR_DAYS_AFTER = 1

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

# Deterministic-final classifications: the pilot LLM leg never submits these to
# the model — they are terminal at the classifier. Mirrors
# ``bba.audit_pipeline.pipeline._DETERMINISTIC_FINAL_CLASSIFICATIONS`` (the
# production composer) so the model leg stays in lockstep with the live pipeline
# (spec #119, ticket #124). RETURNED_NOT_TRANSFUSED / PERIOP_TRANSFUSION_EXEMPT
# are the two returns-ledger terminals; without them here a returned/exempt row
# would be appended to ``llm_contexts`` and wrongly earn an LLM verdict.
DETERMINISTIC_FINAL = frozenset(
    {
        "APPROPRIATE",
        "INSUFFICIENT_EVIDENCE",
        "INAPPROPRIATE",
        "RETURNED_NOT_TRANSFUSED",
        "PERIOP_TRANSFUSION_EXEMPT",
        "PREOP_OVER_RESERVATION",
    }
)

# Mirror bba.audit_pipeline.pipeline._RESERVE_AHEAD_RATIONALES so declared-only
# deferrals (rationale "preop_defer_llm_declared") dispatch to
# RESERVE_AHEAD_REVIEW here too. Kept a local mirror for the same reason the
# leg mirrors the classifier-input helpers.
_RESERVE_AHEAD_RATIONALES = frozenset({"preop_defer_llm", "preop_defer_llm_declared"})


def _planned_op_icd9(
    op_events: Sequence[OperativeEvent], order_datetime: datetime
) -> str:
    """Select the nearest upcoming planned ICD-9, failing closed on a tie."""
    upcoming = sorted(
        (o for o in op_events if o.operative_datetime >= order_datetime),
        key=lambda o: (o.operative_datetime, o.icd9),
    )
    if not upcoming:
        return ""
    nearest_dt = upcoming[0].operative_datetime
    nearest = {o.icd9.strip() for o in upcoming if o.operative_datetime == nearest_dt}
    return upcoming[0].icd9.strip() if len(nearest) == 1 else "\x00AMBIG"


def _returns_disposition_for_classifier(returns_summary: ReturnsSummary | None) -> str:
    """Return the gated disposition passed into the pure classifier.

    Mirrors ``run_pipeline._returns_disposition_for_classifier`` and
    ``pipeline._classifier_inputs_for`` so all four classifier-input sites stay
    in lockstep (spec #119, ticket #124). Off, or with no ledger coverage, the
    classifier sees ``"inconclusive"`` and today's output is unchanged.
    """
    if RETURNS_LEDGER_ENABLED and returns_summary is not None:
        return returns_summary.disposition
    return "inconclusive"


def _declared_use_label_for_classifier(
    collapsed_code: str | None,
) -> DeclaredUseLabel | None:
    """Declared-use LABEL for the pure classifier (gated on the library flag,
    set from the pilot env seam in main())."""
    if feature_flags.DECLARED_USETYPE_ENABLED and collapsed_code is not None:
        return label_for(collapsed_code)
    return None


def _collapsed_usetype_for(values: list[str]) -> str | None:
    """Collapse an order's USETYPE detail lines, but only when the seam is on.

    ``collapse_usetype`` logs a warning on mixed nonblank codes; skipping the
    call when the seam is off keeps a flag-off run byte-identical — no new log
    output even over a mixed-code order.
    """
    if not feature_flags.DECLARED_USETYPE_ENABLED:
        return None
    return collapse_usetype(values)


def _declared_use_record(collapsed_code: str | None) -> DeclaredUse | None:
    """Declared-use RECORD for the evidence bundle — MAPPED codes only.

    Off, no code, or an unknown code (incl "5") → None, so the bundle stays
    byte-identical and the LLM never sees an uninterpreted code. The label
    still reaches the report columns via label_for(); only the bundle fact is
    withheld for unknown codes.
    """
    if not feature_flags.DECLARED_USETYPE_ENABLED or collapsed_code is None:
        return None
    if label_for(collapsed_code) == "unknown":
        return None
    return DeclaredUse.from_code(collapsed_code)


def _returns_periop_context_for_classifier(
    returns_summary: ReturnsSummary | None,
    *,
    surgical_context: bool,
    intraop_transfusion: bool,
    procedure_proximity_hours: float | None,
    upcoming_procedure_hours: float | None,
) -> bool:
    """Return the gated peri-op envelope passed into the pure classifier (#123/#124).

    Reuses :func:`bba.deterministic_classifier.periop_envelope` with its own
    6h/72h window constants — identical to the deterministic leg and the
    production composer — so a remote surgery cannot exempt an unrelated
    transfusion. Off, or with no ledger coverage, returns ``False`` so the
    exemption cannot fire and today's output is unchanged.
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
# Returns are component-agnostic, so an all-returned platelet order is
# deterministic-final and must NOT be LLM-submitted — kept in lockstep with the
# deterministic leg (``run_pipeline.py``) so both legs screen the same orders.
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

    Mirrors ``pipeline.run_pipeline``'s platelet branch AND
    ``run_pipeline._platelet_returns_result`` (the deterministic leg): the same
    pure ``classify()`` decision, so both legs agree given the same peri-op.
    Peri-op IS fed — matching production's tested contract
    (``test_platelet_dispatch`` feeds a ``periop_summary`` and expects
    ``PERIOP_TRANSFUSION_EXEMPT``) — so BOTH terminals are reachable and the hard
    intra-op/EBL contradiction guard stays active (an all-returned platelet whose
    notes chart an intra-op transfusion or EBL >= PERIOP_MIN_EBL_ML falls through
    instead of being falsely cleared). ``procedure_proximity_hours``/
    ``upcoming_procedure_hours`` are ``None`` (matching
    :meth:`PipelineRowContext.for_platelet`).

    This leg's caller passes the bundle's WINDOWED ``periop_summary`` (mirroring
    production and the RBC LLM path), so a remote same-admission surgery cannot
    exempt an unrelated transfusion. The deterministic leg scans peri-op
    admission-wide (the accepted #123 Risk #3); the two legs can therefore differ
    only on that same documented det/model split.

    Returns the :class:`ClassifierResult` iff it is a returns terminal, else
    ``None``. Off, or with no ledger coverage, returns ``None`` so the leg
    submits the identical set to today.
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


def _plt_observations(lab: list[dict[str, str]], an: str) -> list[PlateletObservation]:
    obs: list[PlateletObservation] = []
    for i, r in enumerate(lab):
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


def _med_events(med: list[dict[str, str]], an: str) -> tuple[MedEvent, ...]:
    out: list[MedEvent] = []
    for r in med:
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


def _op_events(
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
            try:
                d = date.fromisoformat(str(indate or "")[:10])
            except ValueError:
                continue
        t = _parse_time(str(r.get("INTIME") or "")) or ParsedTimeOfDay(
            hour=0,
            minute=0,
            second=0,
        )
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


def _latest_anc(
    lab: list[dict[str, str]],
    an: str,
    anchor: datetime,
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
        dt = _combine(
            _parse_hosxp_date(r.get("LVSTDATE") or ""),
            _parse_time(r.get("LVSTTIME") or ""),
        )
        if dt is None or dt > anchor:
            continue
        if best is None or dt > best[0]:
            best = (dt, int(v))
    return None if best is None else best[1]


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


def _fmt_lag(lag_min: int) -> str:
    """Render a signed lag-in-minutes as a phrase relative to the order anchor."""
    if lag_min < 0:
        return f"{abs(lag_min)} min before order"
    if lag_min > 0:
        return f"{lag_min} min after order"
    return "at order time"


def _render_hemodynamic(payload: dict[str, Any]) -> str:
    """Render the pinned hemodynamic summary as one fact-only line (issue #76).

    FACT-ONLY by contract (mirrors the builder's payload guardrail): the MAP
    nadir with its provenance and the vasopressor agents/doses, and nothing
    else. No 'instability' / 'refractory' / appropriateness wording — the LLM
    weighs hemodynamic status, the summary never pre-judges it."""
    parts: list[str] = []
    nadir = payload.get("map_nadir")
    if nadir is not None:
        prov_bits = [str(payload.get("map_nadir_source") or "")]
        lag = payload.get("map_nadir_lag_min")
        if lag is not None:
            prov_bits.append(_fmt_lag(int(lag)))
        prov = ", ".join(b for b in prov_bits if b)
        parts.append(f"MAP nadir {nadir} mmHg" + (f" ({prov})" if prov else ""))
    for v in payload.get("vasopressors") or ():
        seg_bits = [str(v.get("agent") or "")]
        dose = v.get("dose")
        if dose:
            seg_bits.append(str(dose))
        prov_bits = [str(v.get("source") or "")]
        lag = v.get("lag_min")
        if lag is not None:
            prov_bits.append(_fmt_lag(int(lag)))
        prov = ", ".join(b for b in prov_bits if b)
        seg = " ".join(b for b in seg_bits if b)
        parts.append(seg + (f" ({prov})" if prov else ""))
    return "Hemodynamics: " + "; ".join(parts) if parts else ""


def _render_periop(payload: dict[str, Any]) -> str:
    """Render the pinned peri-operative summary as fact-only lines (Case 107).

    FACT-ONLY by contract (mirrors the builder's payload guardrail): the
    surgical-context flag, the EBL in millilitres, the intra-op-transfusion
    flag, and the verbatim provenance snippets — nothing else. No
    'appropriate' / 'indicated' wording; the LLM weighs peri-op context, the
    summary never pre-judges it. The leading 'PERI-OP SIGNALS:' label makes the
    item un-skippable — Case 107's failure was the model ignoring this very
    evidence when it sat only in free-text prose."""
    parts: list[str] = []
    if payload.get("surgical_context"):
        parts.append("surgery=YES")
    ebl = payload.get("blood_loss_ml")
    if ebl is not None:
        parts.append(f"blood_loss={ebl} ml")
    if payload.get("intraop_transfusion"):
        parts.append("intra-op transfusion=YES")
    # Note-scan portion renders exactly as before: when no surgical / EBL /
    # intra-op part fired, it stays empty (old contract: ``if not parts:
    # return ""``), so evidence quotes never render without a signal part. This
    # keeps flag-off output byte-identical structurally, not merely by the
    # scan_periop findings<->flag invariant.
    signals_line = ""
    if parts:
        signals_line = "PERI-OP SIGNALS: " + ", ".join(parts)
        quotes: list[str] = []
        for f in payload.get("findings") or ():
            snippet = (f.get("snippet") or "").strip()
            if not snippet:
                continue
            prov_bits = [str(f.get("source") or "")]
            lag = f.get("lag_min")
            if lag is not None:
                prov_bits.append(_fmt_lag(int(lag)))
            prov = ", ".join(b for b in prov_bits if b)
            quotes.append(f'"{snippet}"' + (f" ({prov})" if prov else ""))
        if quotes:
            signals_line += " | evidence: " + "; ".join(quotes)
    declared = payload.get("declared_use")
    declared_line = ""
    if declared:
        declared_line = (
            "DECLARED INTENT AT ORDER TIME (not confirmation surgery occurred): "
            f"clinician coded use = {declared['label']} "
            f"(BDVSTDT.USETYPE, code {declared['code']})"
        )
    rendered = [s for s in (signals_line, declared_line) if s]
    return "\n".join(rendered)


def _has_note_derived_periop(payload: dict[str, Any]) -> bool:
    """True iff the Periop payload carries note-scan evidence (not a
    declared-use-only item). Keys the quote-or-deny hint so a declared-only
    item never claims to satisfy the peri-operative indication."""
    return bool(
        payload.get("surgical_context")
        or payload.get("blood_loss_ml") is not None
        or payload.get("intraop_transfusion")
        or payload.get("findings")
    )


def _render_administration(payload: dict[str, Any]) -> str:
    """Render affirmative administration facts without appropriateness wording.

    The high-salience line exists so the LLM cannot skim past documented
    administration facts. It confirms only what the snippets affirm and never
    represents absence as non-administration.
    """
    if not payload.get("has_affirmative_marker"):
        return ""
    line = "ADMINISTRATION EVIDENCE: documented=YES"
    quotes: list[str] = []
    for finding in payload.get("findings") or ():
        snippet = (finding.get("snippet") or "").strip()
        if not snippet:
            continue
        prov_bits = [str(finding.get("source") or "")]
        lag = finding.get("lag_min")
        if lag is not None:
            prov_bits.append(_fmt_lag(int(lag)))
        prov = ", ".join(bit for bit in prov_bits if bit)
        quotes.append(f'"{snippet}"' + (f" ({prov})" if prov else ""))
    if quotes:
        line += " | evidence: " + "; ".join(quotes)
    return line


def _render_payload(source: str, payload: dict[str, Any]) -> str:
    """Render a structured EvidenceItem payload as one line for the LLM."""
    if source == "Hemodynamic":
        return _render_hemodynamic(payload)
    if source == "Periop":
        return _render_periop(payload)
    if source == "Administration":
        return _render_administration(payload)
    if source == "IPDADMPROGRESS":
        sections = payload.get("sections") or ()
        rendered = "; ".join(
            f"{s.get('label', '')}: {s.get('text', '')}".strip()
            for s in sections
            if (s.get("text") or "").strip()
        )
        return f"Progress note: {rendered}" if rendered else ""
    if source == "IPDNRFOCUSDT":
        text = (payload.get("text") or "").strip()
        return f"Focus note: {text}" if text else ""
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
        bits = [
            f"{k.upper()}={v}"
            for k in ("sbp", "dbp", "hr", "rr", "bt")
            if (v := payload.get(k)) is not None
        ]
        return f"Vitals at {ts}: " + ", ".join(bits)
    if source == "Med":
        ts = payload.get("timestamp", "")
        drug = payload.get("drug", "")
        return f"Med at {ts}: {drug}"
    return json.dumps(payload, sort_keys=True)


def _incpt_evidence_chunks(
    incpt: list[dict[str, str]],
    optract_dict: dict[str, dict[str, str]],
    *,
    an: str,
    anchor: datetime,
    anchor_label: str = "order",
    window_start: datetime,
    window_end: datetime,
    start_eid: int,
) -> tuple[tuple[EvidenceChunk, ...], int]:
    """Render INCPT operation-charge rows for LLM judgment by description.

    INCPT codes are charge/income codes, not ICD-9-CM procedure codes. OPRTACT
    bridges some of them to ICD-9-CM; unmapped rows still ask the LLM to judge
    operation type from names and descriptions.
    """
    chunks: list[EvidenceChunk] = []
    next_eid = start_eid
    for r in incpt:
        if r.get("AN") != an:
            continue
        if (r.get("CANCELDATE") or "").strip():
            continue
        incgrp = (r.get("INCGRP") or "").strip()
        if incgrp not in INCPT_OPERATION_GROUPS:
            continue
        t = _parse_time(str(r.get("INCTIME") or "")) or ParsedTimeOfDay(
            hour=0,
            minute=0,
            second=0,
        )
        dt = _combine(_parse_hosxp_date(str(r.get("INCDATE") or "")), t)
        if dt is None or not (window_start <= dt < window_end):
            continue
        income = (r.get("INCOME") or "").strip()
        ordercode = (r.get("ORDERCODE") or "").strip()
        optract_code = (r.get("O__OPRTACT") or "").strip()
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
            else optract_dict.get(optract_code)
            or optract_dict.get(income)
            or optract_dict.get(ordercode)
            or {}
        )
        optract_codes = [
            c
            for c in (
                (optract.get("ICD9CM") or "").strip().replace(".", ""),
                (optract.get("ICD9CMADD1") or "").strip().replace(".", ""),
                (optract.get("ICD9CMADD2") or "").strip().replace(".", ""),
            )
            if c
        ]
        optract_name = (optract.get("NAME EN") or "").strip() or (
            optract.get("NAME") or ""
        ).strip()
        group_name = (r.get("INCGRP → NAME") or r.get("INCGRP__NAME") or "").strip()
        hours = (dt - anchor).total_seconds() / 3600.0
        text = (
            f"INCPT operation/procedure charge at {dt.isoformat()}: "
            f"INCGRP={incgrp}"
            f"{(' ' + group_name) if group_name else ''}; "
            f"INCOME={income or '(blank)'}; ORDERCODE={ordercode or '(blank)'}. "
            f"OPRTACT name={optract_name or '(unmapped)'}; "
            f"OPRTACT ICD9CM={','.join(optract_codes) if optract_codes else '(unmapped)'}. "
            "INCPT codes are charge codes; use OPRTACT ICD9CM when mapped, "
            "otherwise judge operation type from group/name/description. "
            f"[{hours:+.1f}h vs {anchor_label}]"
        )
        chunks.append(
            EvidenceChunk(
                evidence_id=f"E{next_eid}",
                source="INCPT",
                text=text,
            )
        )
        next_eid += 1
    return tuple(chunks), next_eid


def _build_inputs():
    """Return all the CSV slices the per-case loop needs."""
    bdvst = _read_csv("BDVST.csv")
    bdvstdt = _read_csv("BDVSTDT.csv")
    msbos_reference = (
        load_msbos_reference() if MSBOS_RESERVATION_PILOT_ENABLED else None
    )
    reserved_units_map = (
        reserved_units_by_component(bdvstdt) if MSBOS_RESERVATION_PILOT_ENABLED else {}
    )
    diag = _read_csv("Diagnosis.csv")
    lab = _read_csv("Lab.csv")
    med = _read_csv("Med.csv")
    progress = _read_csv("IPDADMPROGRESS.csv")
    focus = _read_csv("IPDNRFOCUSDT.csv")
    iptsumoprt = _normalize_iptsumoprt(_read_csv("IPTSUMOPRT.csv"))
    ipddchsumoprt = _normalize_iptsumoprt(_read_optional_csv("IPDDCHSUMOPRT.csv"))
    incpt = _normalize_incpt(
        _read_preferred_optional_csv("INCPT_OPRTACT.csv", "INCPT.csv")
    )
    optract_dict = _normalize_optract(_read_optional_csv("OPRTACT.csv"))
    icd9 = _read_csv("ICD9CM.csv")

    icd9_dict: dict[str, dict[str, str]] = {
        (r.get("Icd9cm") or "").strip().replace(".", ""): {
            "NAME": (r.get("Name") or "").strip(),
            "ORFLAG": (r.get("Orflag") or "").strip(),
        }
        for r in icd9
    }

    products_by_reqno: dict[str, list[str]] = {}
    # Ordered unit amount per REQNO, one raw UNITAMT string per BDVSTDT detail
    # line; summarize_returns parses these fail-closed (spec #119, ticket #124).
    unitamt_lines_by_reqno: dict[str, list[str]] = {}
    # Join key is (HN, REQNO), never bare REQNO: REQNO recurs across admissions,
    # so a declaration from another HN must not attach to the audited order.
    usetype_values_by_hn_reqno: dict[tuple[str, str], list[str]] = {}
    for r in bdvstdt:
        reqno = r["REQNO"]
        products_by_reqno.setdefault(reqno, []).append((r.get("BDTYPE") or "").strip())
        unitamt_lines_by_reqno.setdefault(reqno, []).append(
            (r.get("UNITAMT") or "").strip()
        )
        usetype_values_by_hn_reqno.setdefault(
            ((r.get("HN") or "").strip(), reqno), []
        ).append((r.get("USETYPE") or "").strip())

    # Returns-ledger index: BDVSTTRANS joins audited orders by REQNO exactly
    # (spec #119, ticket #124). One row per dispensed physical unit. Read ONLY
    # when RETURNS_LEDGER_ENABLED, so a flag-off run never opens the optional
    # ledger and stays byte-identical to today even if the file is malformed.
    # Keys are uppercased so summarize_returns reads UNITSTAT; the index keys on
    # the raw REQNO to match order.reqno and every other REQNO index (a one-sided
    # strip would silently miss the join).
    # Source: the canonical export at $BBA_BDVSTTRANS_CSV when set, else the
    # bundle copy (rows already UPPERCASE-keyed by the shared loader).
    bdvsttrans_by_reqno: dict[str, list[dict[str, str]]] = {}
    if RETURNS_LEDGER_ENABLED:
        for row in load_bdvsttrans_rows(BUNDLE):
            bdvsttrans_by_reqno.setdefault(row.get("REQNO") or "", []).append(row)

    bdvst_by_reqno = {r["REQNO"]: r for r in bdvst}
    candidates_by_reqno = build_anchor_candidates(
        bdvstdt_rows=bdvstdt, bdvst_by_reqno=bdvst_by_reqno
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

    return (
        inputs,
        lab,
        med,
        iptsumoprt,
        ipddchsumoprt,
        incpt,
        optract_dict,
        icd9_dict,
        progress,
        focus,
        diag_name_by_code,
        candidates_by_reqno,
        bdvsttrans_by_reqno,
        unitamt_lines_by_reqno,
        usetype_values_by_hn_reqno,
        msbos_reference,
        reserved_units_map,
    )


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set")
    if not BUNDLE.exists():
        sys.exit(f"bundle not found: {BUNDLE} (run sample_bundle.py first)")

    # Phase 5 (#110): the pilot LLM leg runs with the reserve-ahead router
    # enabled so preop_defer_llm cases dispatch to RESERVE_AHEAD_REVIEW and the
    # asymmetric administration-confirmation gate synthesizes
    # PREOP_RESERVATION_UNCONFIRMED. This is the first flagged-on run anywhere;
    # the live pipeline/resume default in bba.feature_flags stays OFF. Because
    # this process does both batch submit and result application, the flag is
    # constant across the two (the operational no-toggle constraint holds).
    # An operator can force an A/B flag-off run via
    # BBA_PILOT_RESERVE_AHEAD_ROUTER=0.
    feature_flags.RESERVE_AHEAD_ROUTER_ENABLED = (
        os.environ.get("BBA_PILOT_RESERVE_AHEAD_ROUTER", "1") != "0"
    )
    # Declared-use pilot seam (spec #147, ticket #151). Set the library flag from
    # the import-time constant (single source of truth; also folded into
    # CODE_VERSION) so the classifier twins / _classify_from_context fallback
    # agree with this leg's direct classify() call.
    feature_flags.DECLARED_USETYPE_ENABLED = DECLARED_USETYPE_PILOT_ENABLED
    feature_flags.MSBOS_RESERVATION_ENABLED = MSBOS_RESERVATION_PILOT_ENABLED

    AUDIT_STORE_ROOT.mkdir(parents=True, exist_ok=True)
    (
        inputs,
        lab,
        med,
        iptsumoprt,
        ipddchsumoprt,
        incpt,
        optract_dict,
        icd9_dict,
        progress,
        focus,
        diag_name_by_code,
        candidates_by_reqno,
        bdvsttrans_by_reqno,
        unitamt_lines_by_reqno,
        usetype_values_by_hn_reqno,
        msbos_reference,
        reserved_units_map,
    ) = _build_inputs()

    fr = build_audit_orders(inputs, AuditOrdersConfig(code_version=CODE_VERSION))
    print(f"audit_orders: included={len(fr.included)} excluded={len(fr.excluded)}")

    contexts: list[PipelineRowContext] = []
    # Re-anchor provenance per order, keyed by audit_id. Kept out of the shared
    # PipelineRowContext model so the schema is untouched; consumed only when
    # emitting the JSON report below.
    anchor_by_id: dict[str, EvidenceAnchor] = {}

    for order in fr.included:
        if ONLY_REQNOS and order.reqno not in ONLY_REQNOS:
            continue
        # --- Platelet path (Phase 2, component="platelet") ---
        # Only active when feature_flags.PLATELET_LLM_ENABLED is True.
        # With the flag off, non-terminal platelet verdicts orphan intentionally
        # (matching the pipeline.py Stage C2 gate); INSUFFICIENT_EVIDENCE rows
        # were already persisted by the deterministic leg.
        if order.component == "platelet":
            if not feature_flags.PLATELET_LLM_ENABLED:
                continue
            plt_obs = _plt_observations(lab, order.an)
            plt_result = lookup_platelet(
                observations=plt_obs,
                anchor_utc=order.order_datetime,
            )
            # The evidence bundle (below) is built for EVERY platelet order,
            # including those with no usable count, so the returns short-circuit
            # can screen returns-first with the bundle's windowed peri-op — the
            # INSUFFICIENT_EVIDENCE gate now runs AFTER it (see below), matching
            # production / the deterministic leg's returns-before-gate precedence.
            hn_hash = _hash(order.hn)
            an_hash = _hash(order.an)
            anchor_by_id[order.audit_id] = EvidenceAnchor(
                anchor_utc=order.order_datetime,
                display="",
                reason="order",
                gap_hours=0.0,
            )
            plt_diagnoses: tuple[DiagnosisRecord, ...] = tuple(
                DiagnosisRecord(icd10=code, description=diag_name_by_code.get(code))
                for code in dict.fromkeys(order.diagnosis_codes)
            )
            # Platelet count trend for the evidence bundle (Stage C2).
            # PlateletRecord objects for all valid counts for this AN.
            plt_records = tuple(
                PlateletRecord(
                    timestamp=obs.datetime_utc,
                    value_k_ul=obs.value_k_ul,
                    source="HEMATOLOGY",
                    item_no=obs.item_no,
                )
                for obs in plt_obs
            )
            # --- Platelet evidence: notes / meds / procedures / vitals ---
            # Fix 3 (Codex P2): the platelet hard signals (active_bleeding,
            # procedure_indication, prophylactic_marrow_failure) are grounded
            # in narrative notes and procedure records. Without this evidence
            # the LLM cannot ground any signal and everything floors to review.
            # Reuse the same evidence-gathering helpers as the RBC path; swap
            # Hb history for platelet history and keep everything else.
            plt_ev_anchor = order.order_datetime  # no re-anchoring for platelets
            plt_local_tz = ZoneInfo(TZ_LOCAL)
            plt_order_date_local = plt_ev_anchor.astimezone(plt_local_tz).date()
            plt_notes_lo = datetime.combine(
                plt_order_date_local - timedelta(days=WINDOW_NOTES_DAYS_BEFORE),
                _time.min,
                tzinfo=plt_local_tz,
            ).astimezone(timezone.utc)
            plt_notes_hi = datetime.combine(
                plt_order_date_local + timedelta(days=WINDOW_NOTES_DAYS_AFTER + 1),
                _time.min,
                tzinfo=plt_local_tz,
            ).astimezone(timezone.utc)
            # op_events is used in the RBC path for cohort + proximity only;
            # platelet rows have no cohort and no proximity requirement, so
            # procedure evidence surfaces through narrative notes (same as RBC).
            plt_med_events = _med_events(med, order.an)
            plt_meds_in_window = sorted(
                [
                    m
                    for m in plt_med_events
                    if plt_notes_lo <= m.timestamp < plt_notes_hi
                ],
                key=lambda m: m.timestamp,
                reverse=True,
            )
            plt_meds_for_bundle = tuple(
                MedRecord(timestamp=m.timestamp, drug=m.drug)
                for m in plt_meds_in_window
            )
            plt_vitals_notes = vitals_notes_for(
                progress, focus, order.an, plt_ev_anchor
            )
            plt_vitals = extract_vitals(anchor=plt_ev_anchor, notes=plt_vitals_notes)
            plt_progress_for_bundle = tuple(
                ProgressNote(timestamp=n.timestamp, text=n.text)
                for n in plt_vitals_notes
                if n.source == "IPDADMPROGRESS"
            )
            plt_focus_for_bundle = tuple(
                FocusNote(timestamp=n.timestamp, text=n.text)
                for n in plt_vitals_notes
                if n.source == "IPDNRFOCUSDT"
            )
            plt_vital_records: tuple[VitalsRecord, ...] = ()
            if plt_vitals.note_timestamp is not None and any(
                getattr(plt_vitals.vitals, k) is not None
                for k in ("sbp", "dbp", "hr", "rr", "bt")
            ):
                _plt_src_map = {
                    "IPDADMPROGRESS": "IPDADMPROGRESS",
                    "IPDNRFOCUSDT": "IPDNRFOCUSDT",
                    "LLM_EXTRACTED": "LLM_extracted",
                    "NONE_IN_WINDOW": None,
                }
                _plt_v_src = _plt_src_map.get(plt_vitals.source.value)
                if _plt_v_src is not None:
                    plt_vital_records = (
                        VitalsRecord(
                            timestamp=plt_vitals.note_timestamp,
                            source=cast(Any, _plt_v_src),
                            sbp=plt_vitals.vitals.sbp,
                            dbp=plt_vitals.vitals.dbp,
                            hr=plt_vitals.vitals.hr,
                            rr=plt_vitals.vitals.rr,
                            bt=plt_vitals.vitals.bt,
                        ),
                    )
            bundle = build_evidence_bundle(
                inputs=EvidenceInputs(
                    anchor=OrderAnchor(
                        order_datetime=order.order_datetime,
                        hn_hash=hn_hash,
                        an_hash=an_hash,
                        products=order.products_ordered,
                    ),
                    component="platelet",
                    diagnoses=plt_diagnoses,
                    progress_notes=plt_progress_for_bundle,
                    focus_notes=plt_focus_for_bundle,
                    meds=plt_meds_for_bundle,
                    platelet_history=plt_records,
                    vitals=plt_vital_records,
                )
            )
            # Returns-ledger short-circuit FIRST — BEFORE the platelet gate below
            # (mirror pipeline.run_pipeline's platelet branch: the returns terminal
            # precedes classify_platelet). An all-returned (or peri-op-exempt)
            # platelet order is deterministic-final and must NOT be LLM-submitted;
            # the deterministic leg (run_pipeline.py) persists its terminal row,
            # here we skip submission. Because this runs before the gate, an
            # all-returned platelet with NO usable count is still screened by the
            # ledger (the Codex P2 fix). Uses the bundle's WINDOWED periop_summary
            # (not admission-wide), so a remote same-admission surgery cannot
            # exempt an unrelated transfusion — mirroring production and the RBC
            # LLM path. The deterministic leg scans peri-op admission-wide (the
            # accepted #123 Risk #3), so the two legs can differ only on that same
            # documented split. Gated on RETURNS_LEDGER_ENABLED so a flag-off run
            # submits the identical set.
            if RETURNS_LEDGER_ENABLED and (
                _platelet_returns_result(
                    audit_id=order.audit_id,
                    order_datetime=order.order_datetime,
                    returns_summary=summarize_returns(
                        rows_for_admission(
                            bdvsttrans_by_reqno.get(order.reqno, []), order.an
                        ),
                        unitamt_lines_by_reqno.get(order.reqno, []),
                    ),
                    periop=bundle.periop_summary,
                )
                is not None
            ):
                continue
            # Deterministic platelet gate: INSUFFICIENT_EVIDENCE is terminal (same
            # contract as the pipeline library). POTENTIALLY_INAPPROPRIATE and
            # NEEDS_REVIEW route onward to the LLM. Runs AFTER the returns check so
            # a returns terminal wins over the platelet gate (production precedence).
            plt_clf = classify_platelet(
                PlateletClassifierInputs(
                    audit_id=order.audit_id,
                    platelet_count=plt_result.value_k_ul,
                )
            )
            if plt_clf.classification == "INSUFFICIENT_EVIDENCE":
                # Terminal: no LLM submission; deterministic-final row would be
                # persisted by the pipeline library but is out-of-scope here.
                continue
            plt_chunks: list[EvidenceChunk] = []
            for item in bundle.items:
                text = _render_payload(item.source, dict(item.payload))
                if text.strip():
                    plt_chunks.append(
                        EvidenceChunk(
                            evidence_id=item.id,
                            source=item.source,
                            text=text,
                        )
                    )
            if not plt_chunks:
                print(
                    f"  WARN: empty chunks for platelet order {order.reqno}; "
                    "skipping LLM submit"
                )
                continue
            contexts.append(
                PipelineRowContext.for_platelet(
                    order=order,
                    platelet_result=plt_result,
                    hn_hash=hn_hash,
                    an_hash=an_hash,
                    redactor_version="external-deid-gate-narrative-0.1",
                    redactor_model_sha="0" * 64,
                    policy_version="KCMH-PR17.2 / AABB-2023 (pilot)",
                    prompt_hash="0" * 64,
                    evidence_bundle_hash=bundle.bundle_hash,
                    evidence_chunks=tuple(plt_chunks),
                )
            )
            continue

        # --- Red-cell path (Phase 1, component="red_cell") ---
        # Guarded so a flag-off run never calls collapse_usetype (which warns on
        # mixed codes) — keeps flag-off byte-identical, no new log output.
        collapsed_usetype = _collapsed_usetype_for(
            usetype_values_by_hn_reqno.get(((order.hn or "").strip(), order.reqno), [])
        )
        # Reserve-ahead elective orders are crossmatched days before they are
        # transfused; re-anchor the evidence windows (Hb lookback, notes, CBC,
        # meds, vitals, INCPT) onto the issue datetime so the model adjudicates
        # the op-day context instead of the reservation-day window. Cohort and
        # procedure proximity keep the REQ order anchor (the order-decision
        # context). evidence_anchor == order.order_datetime when not re-anchored,
        # so same-day orders are byte-identical to before. See
        # bba.hb_lookup.resolve_evidence_anchor.
        evidence_anchor = resolve_evidence_anchor(
            order_datetime=order.order_datetime,
            candidates=candidates_by_reqno.get(order.reqno, []),
        )
        ev_anchor = evidence_anchor.anchor_utc
        reanchored = evidence_anchor.reason == "issue_reanchor"
        anchor_by_id[order.audit_id] = evidence_anchor
        # Display noun for hours-before-anchor flags. For re-anchored orders the
        # windows hang off the transfusion datetime, so "before order" would be
        # off by the reservation->transfusion gap (days). Non-reanchored cases
        # keep "order", so their evidence text is byte-identical to before.
        anchor_label = "transfusion" if reanchored else "order"

        hb_obs = _hb_observations(lab, order.an)
        hb, _hb_anchor_display, _hb_anchor_reason = resolve_hb_with_fallback(
            observations=hb_obs,
            order_datetime=ev_anchor,
            candidates=candidates_by_reqno.get(order.reqno, []),
        )
        # When the resolver anchored on a post-anchor draw (the fallback
        # ladder), that draw is the Hb that routed this case to the LLM. Carry
        # its timestamp so the evidence bundle's Hb upper bound includes it;
        # otherwise the model adjudicates without the triggering value. None
        # for the order-time path keeps the original pre-anchor-only window.
        hb_bundle_anchor = (
            hb.datetime_utc
            if hb.datetime_utc is not None and hb.datetime_utc > ev_anchor
            else None
        )

        op_events = _op_events(
            iptsumoprt, ipddchsumoprt, incpt, optract_dict, icd9_dict, order.an
        )
        med_events = _med_events(med, order.an)
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
                # MTP cluster arm is unfed in the pilot: BDVSTTRANS has no
                # REQNO, so there is no precise per-order RBC-unit history to
                # build BloodOrderEvent records from. See README "MTP arm is
                # unfed". detect_mtp_pattern therefore never fires here.
                blood_orders=(),
                anc_value=anc,
            )
        )

        # Minor bedside / diagnostic procedures (perm cath, tracheostomy,
        # lumbar puncture, taps, arterial/central lines) are dropped BEFORE
        # deriving proximity — they never justify a transfusion, so they must
        # not fire a peri-procedural / pre-op crossmatch signal. op_events
        # stays unfiltered where it feeds assign_cohort above; kept in sync
        # with run_pipeline.py's deterministic leg.
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

        crystalloid_events = tuple(m for m in med_events if _is_crystalloid(m.drug))
        crystalloid_liters = total_crystalloid_liters(
            crystalloid_events, order.order_datetime
        )

        vitals_notes = vitals_notes_for(progress, focus, order.an, ev_anchor)
        vitals = extract_vitals(anchor=ev_anchor, notes=vitals_notes)

        # Issue #76: ship the narrative notes themselves (not just the regex
        # numbers) so the LLM sees the MAP / vasopressor evidence that Case 2
        # was starved of. Reusing vitals_notes keeps the bundle narrative,
        # the hemodynamic scan input, and the shipped items the SAME note set,
        # so the synthesized Hemodynamic summary is always a strict subset of
        # evidence the LLM also reads in full. The builder re-windows these to
        # its own per-source windows; PHI-safety is handled OUTSIDE this script
        # (the pilot input is treated as already de-identified — see header).
        progress_for_bundle = tuple(
            ProgressNote(timestamp=n.timestamp, text=n.text)
            for n in vitals_notes
            if n.source == "IPDADMPROGRESS"
        )
        focus_for_bundle = tuple(
            FocusNote(timestamp=n.timestamp, text=n.text)
            for n in vitals_notes
            if n.source == "IPDNRFOCUSDT"
        )

        # Calendar-day window in the source-data timezone, not a rolling
        # 48-h slice anchored on the order's clock time. For a 14:00
        # order, the spec calls for "day-before + day-of transfusion",
        # so the bound must be the local midnight that ends "day-of",
        # not order_datetime + 24h (which would leak into the next day).
        local_tz = ZoneInfo(TZ_LOCAL)
        order_date_local = ev_anchor.astimezone(local_tz).date()
        notes_lo = datetime.combine(
            order_date_local - timedelta(days=WINDOW_NOTES_DAYS_BEFORE),
            _time.min,
            tzinfo=local_tz,
        ).astimezone(timezone.utc)
        notes_hi = datetime.combine(
            order_date_local + timedelta(days=WINDOW_NOTES_DAYS_AFTER + 1),
            _time.min,
            tzinfo=local_tz,
        ).astimezone(timezone.utc)
        hb_lo = ev_anchor - timedelta(days=WINDOW_HB_DAYS)
        # Re-anchored reserve-ahead orders: the intra-op nadir and the
        # morning-after Hb are drawn AFTER the USE issue time, so a strict
        # backward upper bound (ev_anchor) hides the peri-transfusion drop the
        # case is actually about. Widen the upper bound to the end of op-day +N
        # so the LLM sees the full trajectory. The builder mirrors this via
        # OrderAnchor.hb_anchor (passed below). Non-reanchored orders keep the
        # backward-only bound, so their bundles stay byte-identical.
        if reanchored:
            hb_op_day_hi = datetime.combine(
                order_date_local + timedelta(days=WINDOW_HB_REANCHOR_DAYS_AFTER + 1),
                _time.min,
                tzinfo=local_tz,
            ).astimezone(timezone.utc)
            hb_bundle_anchor = (
                max(hb_bundle_anchor, hb_op_day_hi)
                if hb_bundle_anchor is not None
                else hb_op_day_hi
            )

        diagnoses: tuple[DiagnosisRecord, ...] = tuple(
            DiagnosisRecord(icd10=code, description=diag_name_by_code.get(code))
            for code in dict.fromkeys(order.diagnosis_codes)
        )
        hb_hi = hb_bundle_anchor if hb_bundle_anchor is not None else ev_anchor
        hb_for_bundle = tuple(
            HbRecord(
                timestamp=o.datetime_utc,
                value_g_dl=o.value_g_dl,
                source=o.source,
                item_no=o.item_no,
            )
            for o in hb_obs
            if hb_lo <= o.datetime_utc <= hb_hi
        )
        meds_in_window = sorted(
            [m for m in med_events if notes_lo <= m.timestamp < notes_hi],
            key=lambda m: m.timestamp,
            reverse=True,
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
                vital_records = (
                    VitalsRecord(
                        timestamp=vitals.note_timestamp,
                        source=cast(Any, v_src),
                        sbp=vitals.vitals.sbp,
                        dbp=vitals.vitals.dbp,
                        hr=vitals.vitals.hr,
                        rr=vitals.vitals.rr,
                        bt=vitals.vitals.bt,
                    ),
                )

        hn_hash = _hash(order.hn)
        an_hash = _hash(order.an)

        bundle = build_evidence_bundle(
            inputs=EvidenceInputs(
                anchor=OrderAnchor(
                    order_datetime=order.order_datetime,
                    hn_hash=hn_hash,
                    an_hash=an_hash,
                    products=order.products_ordered,
                    hb_anchor=hb_bundle_anchor,
                    window_anchor=ev_anchor if reanchored else None,
                ),
                diagnoses=diagnoses,
                progress_notes=progress_for_bundle,
                focus_notes=focus_for_bundle,
                meds=meds_for_bundle,
                hb_history=hb_for_bundle,
                vitals=vital_records,
                declared_use=_declared_use_record(collapsed_usetype),
            )
        )

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
            pre = [(i, v, t) for i, v, t in hb_payloads if t <= ev_anchor]
            if pre:
                closest_id = max(pre, key=lambda x: x[2])[0]
                w24 = [x for x in pre if (ev_anchor - x[2]) <= timedelta(hours=24)]
                if w24:
                    min24_id = min(w24, key=lambda x: x[1])[0]
                w48 = [x for x in pre if (ev_anchor - x[2]) <= timedelta(hours=48)]
                if w48:
                    min48_id = min(w48, key=lambda x: x[1])[0]

        chunks: list[EvidenceChunk] = []
        for item in bundle.items:
            text = _render_payload(item.source, dict(item.payload))
            if not text.strip():
                continue
            if item.id in (closest_id, min24_id, min48_id):
                ts = item.timestamp_utc
                hrs = (ev_anchor - ts).total_seconds() / 3600.0 if ts else None
                flags: list[str] = []
                if item.id == closest_id:
                    flags.append(f"closest pre-{anchor_label} Hb")
                if item.id == min24_id:
                    flags.append(f"minimum in 24h pre-{anchor_label}")
                if item.id == min48_id and item.id != min24_id:
                    flags.append(f"minimum in 48h pre-{anchor_label}")
                if hrs is not None:
                    flags.append(f"{hrs:.1f}h before {anchor_label}")
                text = f"{text}  [{'; '.join(flags)}]"
            elif item.source == "Lab" and "value_g_dl" in dict(item.payload):
                ts = item.timestamp_utc
                hrs = (ev_anchor - ts).total_seconds() / 3600.0 if ts else None
                if hrs is not None:
                    # Re-anchored bundles include op-day draws AFTER the
                    # transfusion (intra-op nadir, morning-after); render those
                    # as "after" instead of a negative "before".
                    when = "before" if hrs >= 0 else "after"
                    text = f"{text}  [{abs(hrs):.1f}h {when} {anchor_label}]"
            chunks.append(
                EvidenceChunk(
                    evidence_id=item.id,
                    source=item.source,
                    text=text,
                )
            )

        next_eid = 801
        incpt_chunks, next_eid = _incpt_evidence_chunks(
            incpt,
            optract_dict,
            an=order.an,
            anchor=ev_anchor,
            anchor_label=anchor_label,
            window_start=notes_lo,
            window_end=notes_hi,
            start_eid=next_eid,
        )
        chunks.extend(incpt_chunks)

        # Append CBC chunks (Plt / WBC / Neutrophils) in the ±1d window.
        for r in lab:
            if r.get("AN") != order.an:
                continue
            code = (r.get("LABEXM") or "").strip()
            if code not in {
                PLT_CODE,
                *WBC_CODES,
                NEUTROPHIL_ABS_CODE,
                NEUTROPHIL_PCT_CODE,
            }:
                continue
            dt = _combine(
                _parse_hosxp_date(r.get("LVSTDATE") or ""),
                _parse_time(r.get("LVSTTIME") or ""),
            )
            if dt is None or not (notes_lo <= dt < notes_hi):
                continue
            value = (r.get("RESULT") or "").strip()
            if not value:
                continue
            name = (r.get("NAME_LABEXM") or "").strip() or code
            unit = (r.get("NRMUNIT") or "").strip()
            lo = (r.get("MINNRM") or "").strip()
            hi = (r.get("MAXNRM") or "").strip()
            hrs = (ev_anchor - dt).total_seconds() / 3600.0
            text = (
                f"{name} {value}{(' ' + unit) if unit else ''} at "
                f"{dt.isoformat()}  [ref {lo}-{hi}; "
                f"{hrs:+.1f}h vs {anchor_label}]"
            )
            chunks.append(
                EvidenceChunk(
                    evidence_id=f"E{next_eid}",
                    source="Lab",
                    text=text,
                )
            )
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
        if any(
            it.source == "Periop" and _has_note_derived_periop(dict(it.payload))
            for it in bundle.items
        ):
            guidance_lines += [
                "",
                "PERI-OPERATIVE EVIDENCE — quote-or-deny requirement:",
                "- A pinned 'PERI-OP SIGNALS' item is present above (surgical",
                "  context / estimated blood loss / intra-op transfusion",
                "  recovered from the free-text narrative).",
                "- A surgery, EBL, or intra-op transfusion documented ONLY in a",
                "  free-text note SATISFIES the peri-operative indication. Empty",
                "  structured procedure rows do NOT negate it — do NOT write",
                "  'no operative procedure documented' when the narrative shows one.",
                "- For EACH peri-op fact in that item you MUST either quote the",
                "  supporting snippet (naming its evidence id) in",
                "  reasoning_summary_en AND reasoning_summary_th, OR explicitly",
                "  state why you reject it.",
                "- Silently ignoring the peri-op signal is the failure mode this",
                "  requirement exists to prevent (Case 107 / REQNO 68074627).",
            ]
        chunks.append(
            EvidenceChunk(
                evidence_id=f"E{next_eid}",
                source="Analysis_Hint",
                text="\n".join(guidance_lines),
            )
        )
        next_eid += 1

        if not chunks:
            print(f"  WARN: empty chunks for {order.reqno}; skipping LLM submit")
            continue

        # Returns-ledger read path (spec #119, ticket #124), behind the flag.
        # Off -> no ledger read, returns_summary stays None and the classifier
        # sees "inconclusive", so the LLM leg's submission set is byte-identical
        # to today. Joined by REQNO exactly, mirroring the deterministic leg.
        returns_summary: ReturnsSummary | None = None
        if RETURNS_LEDGER_ENABLED:
            returns_summary = summarize_returns(
                rows_for_admission(bdvsttrans_by_reqno.get(order.reqno, []), order.an),
                unitamt_lines_by_reqno.get(order.reqno, []),
            )

        reservation_decision = None
        if (
            MSBOS_RESERVATION_PILOT_ENABLED
            and order.component == "red_cell"
            and msbos_reference
        ):
            planned = _planned_op_icd9(op_events, order.order_datetime)
            reserved = reserved_units_map.get(
                (order.hn.strip(), order.reqno.strip(), ComponentFamily.RED_CELL), 0
            )
            if planned == "\x00AMBIG":
                reservation_decision = ReservationDecision(
                    reserved_units=reserved,
                    is_over=False,
                    reason="ambiguous_planned_op",
                    reference_hash=msbos_reference.content_hash,
                )
            else:
                reservation_decision = evaluate_reservation(
                    reserved_units=reserved,
                    planned_icd9_nodot=planned,
                    reference=msbos_reference,
                )

        contexts.append(
            PipelineRowContext(
                order=order,
                hb_result=hb,
                vitals_result=vitals,
                cohort_assignment=cohort,
                procedure_proximity_hours=proximity_h,
                upcoming_procedure_hours=upcoming_h,
                crystalloid_liters_prior_4h=crystalloid_liters,
                hn_hash=hn_hash,
                an_hash=an_hash,
                prior_rbc_units_24h=0,
                prior_rbc_units_7d=0,
                redactor_version="external-deid-gate-narrative-0.1",
                redactor_model_sha="0" * 64,
                policy_version="KCMH-PR17.2 / AABB-2023 (pilot)",
                prompt_hash="0" * 64,
                evidence_bundle_hash=bundle.bundle_hash,
                evidence_chunks=tuple(chunks),
                periop_summary=bundle.periop_summary,
                administration_summary=bundle.administration_summary,
                enable_missing_hb_positive_evidence=ENABLE_MISSING_HB_POSITIVE_EVIDENCE,
                returns_summary=returns_summary,
                reservation_decision=reservation_decision,
                declared_use=_declared_use_label_for_classifier(collapsed_usetype),
            )
        )

    # Two deliberately separate maps (Codex round-6 P1):
    #   * ``classifier_results`` holds ONLY RBC ``ClassifierResult`` entries and
    #     is handed to ``apply_batch_results``. Platelet contexts are excluded so
    #     the replay path re-derives ``rule_classification`` from the platelet
    #     gate (``_platelet_gate_result``) instead of reading ``.cohort_threshold``
    #     off a ``PlateletClassifierResult`` (which has no such attribute and
    #     would crash). This mirrors the main pipeline, which submits platelet
    #     batches with ``classifier_results={}``.
    #   * ``report_classifier_results`` holds BOTH RBC and platelet deterministic
    #     results and is consumed only by the summary + JSON-report loops below,
    #     so the round-4 KeyError fix's intent is preserved without leaking a
    #     ``PlateletClassifierResult`` into the RBC replay classifier map.
    classifier_results: dict[str, Any] = {}
    report_classifier_results: dict[str, Any] = {}
    llm_contexts: list[PipelineRowContext] = []
    over_reserved_ctxs: list[PipelineRowContext] = []
    audit_store = AuditStore(
        AuditStoreConfig(
            root_dir=AUDIT_STORE_ROOT,
            code_version=CODE_VERSION,
        )
    )
    for ctx in contexts:
        if ctx.component == "platelet":
            # Platelet contexts were already classified in the order loop above
            # (INSUFFICIENT_EVIDENCE was skipped; anything reaching here routes
            # to the LLM). No RBC classify() call — sentinel Hb values must not
            # drive a classification decision.
            # Store the platelet classifier result in the REPORT map only so the
            # downstream summary and JSON-report loops can look it up without a
            # KeyError. It is intentionally kept OUT of ``classifier_results``
            # (the replay map) so ``apply_batch_results`` never receives a
            # ``PlateletClassifierResult``. Use the deterministic gate (same
            # inputs as the order loop) rather than re-reading the pre-computed
            # value that is local to the order loop.
            plt_count = (
                ctx.platelet_result.value_k_ul
                if ctx.platelet_result is not None
                else None
            )
            report_classifier_results[ctx.order.audit_id] = classify_platelet(
                PlateletClassifierInputs(
                    audit_id=ctx.order.audit_id,
                    platelet_count=plt_count,
                )
            )
            llm_contexts.append(ctx)
            continue
        periop = ctx.periop_summary
        cres = classify(
            ClassifierInputs(
                audit_id=ctx.order.audit_id,
                hb_result=ctx.hb_result,
                cohort_assignment=ctx.cohort_assignment,
                order_datetime=ctx.order.order_datetime,
                procedure_proximity_hours=ctx.procedure_proximity_hours,
                upcoming_procedure_hours=ctx.upcoming_procedure_hours,
                crystalloid_liters_prior_4h=ctx.crystalloid_liters_prior_4h,
                enable_missing_hb_positive_evidence=ctx.enable_missing_hb_positive_evidence,
                periop_blood_loss_ml=periop.blood_loss_ml if periop else None,
                periop_intraop_transfusion=periop.intraop_transfusion
                if periop
                else False,
                periop_surgical_context=periop.surgical_context if periop else False,
                returns_disposition=_returns_disposition_for_classifier(
                    ctx.returns_summary
                ),
                returns_periop_context=_returns_periop_context_for_classifier(
                    ctx.returns_summary,
                    surgical_context=periop.surgical_context if periop else False,
                    intraop_transfusion=periop.intraop_transfusion if periop else False,
                    procedure_proximity_hours=ctx.procedure_proximity_hours,
                    upcoming_procedure_hours=ctx.upcoming_procedure_hours,
                ),
                declared_use=(
                    ctx.declared_use if feature_flags.DECLARED_USETYPE_ENABLED else None
                ),
            )
        )
        classifier_results[ctx.order.audit_id] = cres
        report_classifier_results[ctx.order.audit_id] = cres
        reserve_ahead = (
            feature_flags.RESERVE_AHEAD_ROUTER_ENABLED
            and cres.rationale in _RESERVE_AHEAD_RATIONALES
        )
        if (
            MSBOS_RESERVATION_PILOT_ENABLED
            and reserve_ahead
            and is_over_reservation(classifier_result=cres, context=ctx)
        ):
            _persist_over_reservation_row(
                ctx,
                classifier_result=cres,
                audit_store=audit_store,
                run_id=RUN_ID,
            )
            over_reserved_ctxs.append(ctx)
            continue
        if cres.classification not in DETERMINISTIC_FINAL:
            llm_contexts.append(ctx)

    print(
        "  over-reserved (not submitted, PREOP_OVER_RESERVATION): "
        f"{len(over_reserved_ctxs)}"
    )
    print(f"\nLLM-bound: {len(llm_contexts)} / {len(contexts)}")
    # Over-reserved rows are persisted but not submitted; a run filtered to only
    # such REQNOs still has verdicts to report, so it must not exit here (#163).
    if not llm_contexts and not over_reserved_ctxs:
        sys.exit("nothing to submit")

    submissions: list[BatchSubmissionRequest] = []
    # Codex round-5 P2 (security): honor injection routing in the pilot leg. A
    # prompt whose scanner raised route_to_needs_review must NEVER be submitted
    # to Anthropic; it is quarantined to a local NEEDS_REVIEW row below,
    # mirroring the batch pipeline's _persist_injection_flagged_row.
    injection_flagged: list[PipelineRowContext] = []
    for ctx in llm_contexts:
        if ctx.component == "platelet":
            # Platelet review uses the dedicated PLATELET_REVIEW task mode
            # and system prompt (no Hb cohort threshold).
            prompt = build_prompt(
                PromptBuildRequest(
                    task_mode="PLATELET_REVIEW",
                    cohort_threshold=None,
                    evidence_chunks=ctx.evidence_chunks,
                    few_shot_examples=(),
                )
            )
            if prompt.route_to_needs_review:
                injection_flagged.append(ctx)
                continue
            submissions.append(
                BatchSubmissionRequest(
                    audit_id=ctx.order.audit_id,
                    run_id=RUN_ID,
                    task_mode="PLATELET_REVIEW",
                    prompt=prompt,
                )
            )
            continue
        threshold = (
            ctx.cohort_assignment.threshold
            if ctx.cohort_assignment.threshold is not None
            else 7.0
        )
        cres = classifier_results[ctx.order.audit_id]
        reserve_ahead = (
            feature_flags.RESERVE_AHEAD_ROUTER_ENABLED
            and cres.rationale in _RESERVE_AHEAD_RATIONALES
        )
        task_mode = rbc_task_mode(ctx.hb_result.value_g_dl, reserve_ahead=reserve_ahead)
        prompt = build_prompt(
            PromptBuildRequest(
                task_mode=task_mode,
                cohort_threshold=threshold,
                evidence_chunks=ctx.evidence_chunks,
                few_shot_examples=(),
            )
        )
        if prompt.route_to_needs_review:
            injection_flagged.append(ctx)
            continue
        submissions.append(
            BatchSubmissionRequest(
                audit_id=ctx.order.audit_id,
                run_id=RUN_ID,
                task_mode=task_mode,
                prompt=prompt,
            )
        )

    # Codex round-5 P2 (security): persist injection-flagged rows locally as
    # NEEDS_REVIEW — they were never sent to Anthropic. Mirrors the batch
    # pipeline's quarantine: an RBC context carries its real deterministic
    # ClassifierResult; a platelet context (classifier_result=None) re-derives
    # rule_classification from the platelet gate inside
    # _persist_injection_flagged_row. The pilot's platelet gate runs with
    # defer off (the order loop above), so the default flag matches here.
    for inj_ctx in injection_flagged:
        _persist_injection_flagged_row(
            inj_ctx,
            classifier_result=(
                None
                if inj_ctx.component == "platelet"
                else classifier_results.get(inj_ctx.order.audit_id)
            ),
            audit_store=audit_store,
            run_id=RUN_ID,
        )
    if injection_flagged:
        print(
            "  injection-flagged (not submitted, routed to NEEDS_REVIEW): "
            f"{len(injection_flagged)}"
        )

    if not submissions:
        print("  no clean submissions remaining after the injection filter")
    else:
        transport = RealAnthropicTransport(
            api_key=api_key,
            poll_interval_seconds=20.0,
            # Anthropic batches have a 24h SLA; the old 1h cap gave up while
            # the batch was still processing. Env-overridable for a shorter
            # wait when iterating locally.
            max_wait_seconds=float(os.environ.get("BBA_PILOT_BATCH_MAX_WAIT", "86400")),
        )
        t0 = time.time()
        # Resume path: if BBA_PILOT_BATCH_ID is set, re-attach to an already-
        # submitted batch instead of creating a new one. The deterministic
        # context build above is reproducible, so ``submissions`` matches the
        # original submission set and fetch_batch_results can rebuild each
        # result's request_json. Use this after a Ctrl+C during polling so the
        # in-flight batch is not abandoned and re-billed.
        resume_batch_id = os.environ.get("BBA_PILOT_BATCH_ID")
        if resume_batch_id:
            batch_id = resume_batch_id
            print(f"  resuming existing batch_id = {batch_id} (skipping submit)")
        else:
            print(
                f"\nSubmitting batch of {len(submissions)} requests to "
                f"Anthropic (model={MODEL_ID})..."
            )
            batch_id = transport.submit_batch_only(
                model=MODEL_ID,
                requests=submissions,
                prompt_cache_enabled=True,
            )
            print(f"  batch_id = {batch_id}")
            print(f"  (resume with: BBA_PILOT_BATCH_ID={batch_id})")
        print("  polling (this can take a while)...")
        response = transport.fetch_batch_results(
            batch_id,
            model=MODEL_ID,
            requests=submissions,
            prompt_cache_enabled=True,
        )
        elapsed = time.time() - t0
        print(f"  batch complete in {elapsed:.1f}s; {len(response.results)} results")

        context_map = {ctx.order.audit_id: ctx for ctx in llm_contexts}
        # ``classifier_results`` holds RBC ``ClassifierResult`` entries only;
        # platelet contexts are absent, so the replay path re-derives their
        # ``rule_classification`` from the platelet gate rather than reading
        # ``.cohort_threshold`` off a ``PlateletClassifierResult``.
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
    print(
        f"{'reqno':<10} {'det.verdict':<26} {'final':<26} "
        f"{'conf':<6} {'review_reason':<22} {'reasoning_en (head)'}"
    )
    print("=" * 120)
    # Over-reserved rows were persisted (PREOP_OVER_RESERVATION) but skipped
    # LLM submission; surface them in the console table and JSON report so the
    # deterministic verdict is not invisible to the review page (#163). Empty
    # when the flag is off, so a flag-off run is byte-identical.
    reported_ctxs = [*llm_contexts, *over_reserved_ctxs]
    for ctx in reported_ctxs:
        det = report_classifier_results[ctx.order.audit_id]
        r = rows_by_id.get(ctx.order.audit_id)
        reasoning = r.reasoning_summary_en[:60] if r and r.reasoning_summary_en else ""
        final = r.final_classification if r else "(no row)"
        conf = f"{r.confidence:.2f}" if r else "—"
        rr = (r.review_reason or "—") if r else "—"
        print(
            f"{ctx.order.reqno:<10} {det.classification:<26} "
            f"{final:<26} {conf:<6} {rr[:20]:<22} {reasoning}"
        )

    report = []
    for ctx in reported_ctxs:
        det = report_classifier_results[ctx.order.audit_id]
        r = rows_by_id.get(ctx.order.audit_id)
        ev = anchor_by_id.get(ctx.order.audit_id)
        report.append(
            {
                "reqno": ctx.order.reqno,
                "audit_id": ctx.order.audit_id,
                "evidence_anchor": (
                    {
                        "reason": ev.reason,
                        "datetime_local": ev.display,
                        "gap_hours": round(ev.gap_hours, 1),
                    }
                    if ev is not None
                    else None
                ),
                "deterministic": {
                    "classification": det.classification,
                    "rationale": det.rationale,
                    "cohort": ctx.cohort_assignment.label.value,
                    "threshold": ctx.cohort_assignment.threshold,
                    "hb": ctx.hb_result.value_g_dl,
                },
                "llm_final": (
                    {
                        "rule_classification": r.rule_classification,
                        "final_classification": r.final_classification,
                        "model": r.model_id,
                        "indications": [dict(d) for d in r.indications_json],
                        "negative_evidence": [
                            dict(d) for d in r.negative_evidence_json
                        ],
                        "reasoning_en": r.reasoning_summary_en,
                        "reasoning_th": r.reasoning_summary_thai,
                        "confidence": r.confidence,
                        "review_reason": r.review_reason,
                        "needs_human_review": r.needs_human_review,
                        "verifier_pass": r.verifier_pass,
                        "escalated_to_opus": r.escalated_to_opus,
                    }
                    if r
                    else None
                ),
            }
        )
    out = WORK / "llm_report.json"
    if ONLY_REQNOS and out.exists():
        # Filtered run: splice the fresh records into the existing report so
        # the other cases (and the review page built from it) are preserved.
        existing = json.loads(out.read_text())
        fresh_by_reqno = {rec["reqno"]: rec for rec in report}
        existing_reqnos = {rec["reqno"] for rec in existing}
        report = [fresh_by_reqno.get(rec["reqno"], rec) for rec in existing] + [
            rec for rec in report if rec["reqno"] not in existing_reqnos
        ]
        print(f"  merged {len(fresh_by_reqno)} fresh record(s) into existing report")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nFull JSON report: {out}")


if __name__ == "__main__":
    main()
