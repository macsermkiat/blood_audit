"""Assemble a single HTML page bundling all source data + audit
verdicts for human review.

For each sampled (HN, REQNO, AN) key: render identity + anchor, the
BDVST row, BDVSTDT line items, AN-scoped Diagnoses deduped by ICD-10
(with the global ICD-10 description dictionary), Hb history (7-day
pre-order), ANC history (±1d), CBC subset (Hb/Plt/WBC/Neutrophils),
procedures (±1 week, AN-scoped, joined to ICD9CM names), windowed
meds, deduped progress + focus notes (raw, collapsible), the
deterministic verdict, and the LLM verdict (indications, negative
evidence, EN+TH reasoning, confidence).

Environment variables:

* ``BBA_PILOT_WORK_DIR`` — input/output directory (default:
  ``/tmp/bba_mini``). Must contain ``bundle/``, ``report.csv``,
  ``llm_report.json``, ``sample_manifest.csv``.
* ``BBA_PILOT_ICD10_CSV`` — path to the HOSxP ICD-10 master dictionary
  (default: ``../Bloodbank/data/raw/ICD10.csv`` relative to the repo
  root). Missing-file is OK; the HTML falls back to the bundle's
  ``NAME_ICD10`` column for any codes that appear there.
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from bba.feature_flags import (
    MSBOS_PLANNED_OP_PICKER_V2_ENABLED,
    MSBOS_RESERVATION_ENABLED,
    RETURNS_LEDGER_ENABLED,
)

WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))
_msbos_env = os.environ.get("BBA_PILOT_MSBOS_RESERVATION")
MSBOS_RESERVATION_PILOT_ENABLED = (
    _msbos_env == "1" if _msbos_env is not None else MSBOS_RESERVATION_ENABLED
)
_picker_v2_env = os.environ.get("BBA_PILOT_MSBOS_PLANNED_OP_PICKER_V2")
MSBOS_PLANNED_OP_PICKER_V2_PILOT_ENABLED = (
    _picker_v2_env == "1"
    if _picker_v2_env is not None
    else MSBOS_PLANNED_OP_PICKER_V2_ENABLED
)
BUNDLE = WORK / "bundle"
LLM_REPORT = WORK / "llm_report.json"
DET_REPORT = WORK / "report.csv"
MANIFEST = WORK / "sample_manifest.csv"
OUT = WORK / "review.html"
TZ_LOCAL = ZoneInfo("Asia/Bangkok")

_DEFAULT_ICD10 = (
    Path(__file__).resolve().parents[2].parent
    / "Bloodbank"
    / "data"
    / "raw"
    / "ICD10.csv"
)
ICD10_DICT_CSV = Path(os.environ.get("BBA_PILOT_ICD10_CSV", str(_DEFAULT_ICD10)))

# Per-source windowing rules around the evidence anchor.
WINDOW_PROC_DAYS = 7
WINDOW_HB_DAYS = 7
# Re-anchored reserve-ahead orders only: extend the Hb history upper bound this
# many calendar days past the op day so the human reviewer sees the same
# intra-op nadir + morning-after draws the LLM gate sees (both drawn AFTER the
# USE issue time). Mirrors run_llm_leg.WINDOW_HB_REANCHOR_DAYS_AFTER.
WINDOW_HB_REANCHOR_DAYS_AFTER = 1
WINDOW_NOTES_DAYS_BEFORE = 1
WINDOW_NOTES_DAYS_AFTER = 1
WINDOW_PERIOP_NOTES_DAYS_AFTER = 2
PROGRESS_FIRST_N = 3

# Clinical-reviewer spec: only show CBC subset in the "All labs" section.
CBC_LABEXM_CODES: frozenset[str] = frozenset(
    {
        "290095",  # Hemoglobin (HEMATOLOGY)
        "500001",  # Hemoglobin (POCT)
        "290078",  # Platelets
        "290136",  # WBC
        "120015",  # WBC (alt source)
        "290092",  # Neutrophils %
        "290093",  # Neutrophils # (ANC)
    }
)

# REQTYPE dictionary per docs/ingest-mapping.md.
REQTYPE_LABELS: dict[str, str] = {
    "P": "ผู้ป่วยขอ / patient request",
    "H": "โรงพยาบาลอื่นขอ / other-hospital referral",
}

UNITSTAT_LABELS: dict[str, str] = {
    "1": "จอง / reserved",
    "2": "จ่าย / issued",
    "3": "คืน / returned",
    "4": "ฝากแช่ / stored",
    "5": "ให้เลือด / transfused",
    "6": "เปลี่ยนประเภท / changed type",
    "7": "Incompat",
    "8": "เก็บ / kept",
    "9": "compat",
    "10": "รวมถุง / pooled",
}

# Diagnosis-type precedence — best-keeps wins on duplicate ICD-10 rows.
_TYPE_RANK = {
    "Principal Diagnosis": 0,
    "Complication": 1,
    "Comorbidity": 2,
}

_EBL_RE = re.compile(
    r"\b(?:EBL|blood\s*loss)\s*[:=.]?\s*([0-9][0-9,]*(?:\.\d+)?)\s*(mL|ml|cc|L|liter|liters)\b",
    re.IGNORECASE,
)

_RATIONALE_LABELS: dict[str, str] = {
    "hb_lt_7_universal": "Hb < 7.0 g/dL — below the universal threshold; no bypass required",
    "hb_lt_threshold": "Hb below the cohort-specific threshold",
    "hb_7_to_10": "Hb 7.0–10.0 g/dL — gray zone; requires a documented clinical indication",
    "hb_ge_10": "Hb ≥ 10.0 g/dL — above threshold; no qualifying bypass was found",
    "hb_missing": "No Hb result in the 7-day pre-anchor window",
    "cohort_unknown": "Patient cohort could not be determined",
    "cohort_non_threshold": "Cohort has no fixed Hb threshold (e.g., active haematological malignancy)",
    "bypass_mtp": "Massive Transfusion Protocol cohort — auto-classified APPROPRIATE",
    "bypass_mtp_hb_missing": "MTP cohort with no Hb available — auto-classified APPROPRIATE",
    "bypass_peri_procedural": "[legacy/flag-off] Procedure within 6 h before order — peri-procedural bypass",
    "bypass_peri_procedural_hb_missing": "Peri-procedural bypass (Hb missing) — auto-classified APPROPRIATE",
    "preop_defer_llm": (
        "[legacy/flag-off] Upcoming procedure within 72 h — routed to LLM review, not auto-cleared "
        "(a crossmatch reservation is not a transfusion indication). Minor "
        "procedures (perm cath, tracheostomy, lumbar puncture, thoracocentesis, "
        "paracentesis, arthrocentesis, arterial/central line) are excluded from "
        "this signal."
    ),
    "preop_defer_llm_declared": (
        "[legacy/flag-off] Declared surgical intent (BDVSTDT.USETYPE = surgery/type-screen) "
        "routed the order to LLM review with no structured op row; a "
        "declaration is not a transfusion indication."
    ),
    "bypass_pre_op_crossmatch": (
        "[legacy] Upcoming procedure within 72 h — auto-classified APPROPRIATE. "
        "Superseded by preop_defer_llm (now routes to LLM review); retained so "
        "historical reports still render."
    ),
    "bypass_delta_hb": "Rapid Hb drop (delta-Hb trigger) fired — auto-classified APPROPRIATE",
    "bypass_hemodilution": "Haemodilution pattern flagged — Hb unreliable; sent to NEEDS_REVIEW",
    "single_low_hb_no_trend": "Single Hb below threshold with no supporting trend — NEEDS_REVIEW",
}

_BYPASS_LABELS: dict[str, str] = {
    "none": "No bypass pathway applied; classification is purely Hb-tier based",
    "mtp": "Massive Transfusion Protocol — volume-based bypass, Hb thresholds suspended",
    "peri_procedural_6h": "Peri-procedural: active procedure within the 6 h window before order",
    "pre_op_crossmatch": (
        "[legacy] Pre-operative crossmatch: scheduled procedure within 72 h after "
        "order. Superseded — pre-op orders now route to LLM review with "
        "bypass=none; retained for historical rows."
    ),
    "delta_hb": "Delta-Hb: ≥ 2 g/dL drop in Hb in the 24 h pre-anchor window",
    "hemodilution_flagged": "Haemodilution suspected: Hb rise after IV fluid consistent with dilution artifact",
}

_REVIEW_REASON_LABELS: dict[str, str] = {
    "model_verdict": (
        "No guardrail action — the final classification is the model's own verdict"
    ),
    "llm_overclear_asserted_inappropriate": (
        "Guardrail-asserted INAPPROPRIATE: the LLM cleared a withheld "
        "gray-zone/high-Hb order with no genuine hard indication, so the "
        "over-clear guardrail asserted the final verdict (not the model's "
        "own label); the human-review flag is cleared"
    ),
    "llm_native_review_asserted_inappropriate": (
        "Guardrail-converted INAPPROPRIATE: the model itself returned "
        "NEEDS_REVIEW with reasoning but no hard signal and no qualified "
        "bleed, so the verdict was converted to INAPPROPRIATE; the "
        "human-review flag is cleared"
    ),
    "llm_overclear_suspect": (
        "Over-clear floored to NEEDS_REVIEW: historical rows from before the "
        "assert guardrail, plus the live paths where asserting is unsafe — "
        "a shape-drifted tool payload (missing or garbled "
        "indications/negative_evidence), or a grounded high-confidence "
        "citation of a hard indication the structured system cannot "
        "dismiss (ACS; documented shock/pressors the vitals snapshot "
        "cannot see; a structurally-true sub-floor Hb withheld as "
        "unreliable)"
    ),
    "periop_signal_contradiction": (
        "Peri-operative hard signal contradicts the LLM verdict — kept "
        "NEEDS_REVIEW for a human (the intended residual)"
    ),
    "preop_reservation_unconfirmed": (
        "Reserve-ahead pre-op crossmatch with no affirmative administration "
        "evidence — terminal, excluded from transfusion attribution; not queued "
        "for human review"
    ),
    "operation_unresolved": (
        "Conflicting MSBOS operation code could not be uniquely resolved from "
        "the windowed clinical notes — escalated to human review"
    ),
    "administration_signal_contradiction": (
        "Structured intra-op/EBL evidence indicates administration but the model "
        "did not claim it — escalated to human review"
    ),
    "hallucination_suspect": (
        "Quote verifier rejected every attempt — the cited quotes did not "
        "ground in the evidence bundle"
    ),
    "empty_reasoning": (
        "Final verdict carried empty reasoning — floored to NEEDS_REVIEW "
        "(a verdict with no rationale is never asserted)"
    ),
    "platelet_llm_overclear_suspect": (
        "Platelet-leg over-clear floored to NEEDS_REVIEW (platelet "
        "guardrail; the RBC assert path does not apply to platelets)"
    ),
    "malformed_json": "Parse failure — the response was not valid JSON",
    "schema_mismatch": (
        "Parse failure — the tool payload did not match the expected schema"
    ),
    "classification_out_of_set": (
        "Parse failure — the classification label is outside the allowed vocabulary"
    ),
    "empty_response": "Parse failure — the model returned an empty response",
    "tool_use_missing": ("Parse failure — no tool-use block was found in the response"),
}


def _code_abbr(code: str, labels: dict[str, str]) -> str:
    """Wrap a rationale/bypass slug in <abbr> with a plain-language tooltip."""
    label = labels.get(code)
    if label:
        return f"<abbr title='{esc(label)}'><code>{esc(code)}</code></abbr>"
    return f"<code>{esc(code)}</code>"


csv.field_size_limit(sys.maxsize)


def esc(s: Any) -> str:
    return html.escape(str(s)) if s is not None else ""


_CLS_DISPLAY: dict[str, str] = {
    "APPROPRIATE": "Appropriate",
    "NEEDS_REVIEW": "Needs review",
    "POTENTIALLY_INAPPROPRIATE": "Potentially inappropriate",
    "PREOP_RESERVATION_UNCONFIRMED": (
        "Administration unconfirmed (pre-op reservation)"
    ),
    "PREOP_OVER_RESERVATION": "Pre-op over-reservation",
    "INAPPROPRIATE": "Inappropriate",
    "INSUFFICIENT_EVIDENCE": "Insufficient evidence",
    "EXCLUDED": "Excluded",
}

_RETURNS_TERMINALS = frozenset({"RETURNED_NOT_TRANSFUSED", "PERIOP_TRANSFUSION_EXEMPT"})
# Classes whose rows may carry MSBOS reservation detail (#201): the returns
# terminals where the annotation is informational, PLUS the post-flip verdict
# classes a declared row can reach once MSBOS screening reclassifies it
# (over -> PREOP_OVER_RESERVATION; unresolved/gated/ambiguous -> NEEDS_REVIEW,
# spec #194/#196). Rows without msbos_* data still render nothing: every
# consumer also requires a non-blank msbos_reason.
_MSBOS_RENDER_CLASSES = _RETURNS_TERMINALS | frozenset(
    {"PREOP_OVER_RESERVATION", "NEEDS_REVIEW"}
)
_MSBOS_ABOVE_REASONS = frozenset(
    {
        "over_gm_excess",
        "over_none",
        "over_type_and_screen_crossmatched",
        "over_ceiling",
        "over_major_non_neuraxial",
        "over_neuraxial",
        "over_cardiac_cpb",
    }
)
_MSBOS_WITHIN_REASONS = frozenset(
    {
        "within_recommendation",
        "type_and_screen_screen_only",
        "within_ceiling",
        "within_major_non_neuraxial",
        "within_neuraxial",
        "within_cardiac_cpb",
        "no_reserved_units",
    }
)
# unresolved = everything else with a nonblank reason: ambiguous_code, unresolved_code,
# ambiguous_planned_op, no_planned_op, operation_unresolved, reservation_lookup_miss,
# uncategorised_procedure, ambiguous_category, missing_pre_op_count.


def _display_cls(cls: str | None, rationale: str | None = None) -> str:
    normalized = (cls or "").upper()
    if RETURNS_LEDGER_ENABLED and normalized == "RETURNED_NOT_TRANSFUSED":
        return "Returned — not transfused (excluded)"
    if RETURNS_LEDGER_ENABLED and normalized == "PERIOP_TRANSFUSION_EXEMPT":
        if rationale == "preop_declared_exempt":
            return "Declared pre-op order — exempt (excluded)"
        return "Peri-op transfusion — exempt (excluded)"
    return _CLS_DISPLAY.get(normalized, cls or "Excluded")


def _msbos_count_breakdown(label: str, counts: dict[str, int]) -> str:
    """One MSBOS bucket-count line; within-ceiling is surfaced separately (#214)."""
    return (
        f"{label} ({counts['denominator']}): "
        f"{counts['above']} above / "
        f"{counts['within']} within / "
        f"{counts['within_ceiling']} within-ceiling / "
        f"{counts['unresolved']} unresolved"
    )


def _ceiling_basis_short(det: dict[str, str]) -> str:
    """The token+units part of msbos_ceiling_basis, dropping the code list.

    ``"G/M 2 (8101,8103)"`` -> ``"G/M 2"``; blank stays blank. Escaped for HTML.
    """
    return esc((det.get("msbos_ceiling_basis") or "").split(" (")[0])


def _msbos_reason_bucket(reason: str) -> str | None:
    normalized = reason.strip()
    if not normalized:
        return None
    # within_ceiling is its own scannable bucket (the shadow-over exposure), not
    # folded into "within" (#210/#214). over_ceiling folds into "above" (a real
    # over under the most permissive tariff).
    if normalized == "within_ceiling":
        return "within_ceiling"
    if normalized in _MSBOS_ABOVE_REASONS:
        return "above"
    if normalized in _MSBOS_WITHIN_REASONS:
        return "within"
    return "unresolved"


def _fmt_plt_k(value: object) -> str:
    """Render a k/µL count/cutoff: integral -> no decimals, non-integral preserved, blank/bad -> ''.

    120.0 -> "120", 120.5 -> "120.5", "" / None / non-numeric -> "". Never truncates.
    """
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(f):
        return ""
    return str(int(f)) if f == int(f) else str(f)


def _fmt_plt_cutoff_k(raw: object) -> str:
    """Category cutoff stored per-µL -> displayed in k/µL; blank/missing/malformed -> ''.

    Defensive mirror of the RBC path's .get()-based rendering: never raises on a malformed row.
    """
    try:
        f = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(f):
        return ""
    return _fmt_plt_k(f / 1000)


def _msbos_platelet_summary_pill(det: dict[str, str]) -> str:
    reason = (det.get("msbos_reason") or "").strip()
    if reason in {
        "over_major_non_neuraxial",
        "over_neuraxial",
        "over_cardiac_cpb",
    }:
        count = esc(_fmt_plt_k(det.get("msbos_plt_count_k_ul", "")))
        cutoff = esc(_fmt_plt_cutoff_k(det.get("msbos_plt_over_above_per_ul", "")))
        text = f"PLT {count} > {cutoff}"
        pill_class = "cls-msbos-warn"
    elif reason in {
        "within_major_non_neuraxial",
        "within_neuraxial",
        "within_cardiac_cpb",
        "no_reserved_units",
    }:
        text = "within"
        pill_class = "cls-msbos-ok"
    elif reason == "missing_pre_op_count":
        text = "count missing"
        pill_class = "cls-msbos-warn"
    elif reason == "uncategorised_procedure":
        text = "op uncategorised"
        pill_class = "cls-msbos-warn"
    elif reason == "ambiguous_category":
        text = "category ambiguous"
        pill_class = "cls-msbos-warn"
    elif reason in {"no_planned_op", "ambiguous_planned_op"}:
        text = "op unresolved"
        pill_class = "cls-msbos-warn"
    elif reason == "reservation_lookup_miss":
        text = "unlinked"
        pill_class = "cls-msbos-warn"
    else:
        return "—"
    return f"<span class='cls {pill_class}'>{text}</span>"


def _msbos_summary_pill(det: dict[str, str]) -> str:
    det_class = (det.get("classification") or "").upper()
    reason = (det.get("msbos_reason") or "").strip()
    if det_class not in _MSBOS_RENDER_CLASSES or not reason:
        return "—"
    if (det.get("component") or "") == "platelet":
        return _msbos_platelet_summary_pill(det)

    reserved = esc(det.get("msbos_reserved_units", ""))
    recommended = esc(det.get("msbos_recommended_units", ""))
    if reason == "over_gm_excess":
        text = f"{reserved} vs G/M {recommended}"
        pill_class = "cls-msbos-warn"
    elif reason == "over_none":
        text = f"{reserved} vs none 0"
        pill_class = "cls-msbos-warn"
    elif reason == "over_type_and_screen_crossmatched":
        text = f"T/S; {reserved}u reserved"
        pill_class = "cls-msbos-warn"
    elif reason == "over_ceiling":
        text = f"over ceiling {_ceiling_basis_short(det)}"
        pill_class = "cls-msbos-warn"
    elif reason == "within_ceiling":
        text = f"within ceiling {_ceiling_basis_short(det)}"
        pill_class = "cls-msbos-ok"
    elif reason in _MSBOS_WITHIN_REASONS:
        text = "within"
        pill_class = "cls-msbos-ok"
    elif reason in {"ambiguous_code", "unresolved_code"}:
        text = "code unresolved"
        pill_class = "cls-msbos-warn"
    elif reason in {
        "ambiguous_planned_op",
        "no_planned_op",
        "operation_unresolved",
    }:
        text = "op unresolved"
        pill_class = "cls-msbos-warn"
    elif reason == "reservation_lookup_miss":
        text = "unlinked"
        pill_class = "cls-msbos-warn"
    else:
        return "—"
    return f"<span class='cls {pill_class}'>{text}</span>"


def _msbos_platelet_case_line(det: dict[str, str], det_class: str) -> str:
    reason = (det.get("msbos_reason") or "").strip()
    category_labels = {
        "major_non_neuraxial": "major-non-neuraxial",
        "neuraxial": "neuraxial",
        "cardiac_cpb": "cardiac-CPB",
    }
    if reason in {
        "over_major_non_neuraxial",
        "over_neuraxial",
        "over_cardiac_cpb",
    }:
        reserved = esc(det.get("msbos_reserved_units", ""))
        count = esc(_fmt_plt_k(det.get("msbos_plt_count_k_ul", "")))
        cutoff = esc(_fmt_plt_cutoff_k(det.get("msbos_plt_over_above_per_ul", "")))
        label = esc(category_labels.get(det.get("msbos_plt_category", ""), ""))
        text = (
            f"Reserved {reserved}u platelets; pre-op count {count}k/uL > "
            f"{label} cutoff {cutoff}k/uL"
        )
    elif reason in {
        "within_major_non_neuraxial",
        "within_neuraxial",
        "within_cardiac_cpb",
    }:
        reserved = esc(det.get("msbos_reserved_units", ""))
        count = esc(_fmt_plt_k(det.get("msbos_plt_count_k_ul", "")))
        cutoff = esc(_fmt_plt_cutoff_k(det.get("msbos_plt_over_above_per_ul", "")))
        label = esc(category_labels.get(det.get("msbos_plt_category", ""), ""))
        text = (
            f"Reserved {reserved}u platelets; pre-op count {count}k/uL within "
            f"{label} cutoff {cutoff}k/uL"
        )
    elif reason == "no_reserved_units":
        text = "No platelet units reserved"
    elif reason == "missing_pre_op_count":
        text = "Platelet pre-op count missing"
    elif reason == "uncategorised_procedure":
        text = "MSBOS platelet category could not be resolved"
    elif reason == "ambiguous_category":
        text = "MSBOS platelet category ambiguous"
    elif reason in {"no_planned_op", "ambiguous_planned_op"}:
        text = "MSBOS planned operation unresolved"
    elif reason == "reservation_lookup_miss":
        text = "Reservation detail lines not linked (unlinked)"
    else:
        text = ""

    has_returns_summary = any(
        (det.get(key) or "").strip()
        for key in (
            "returns_disposition",
            "returns_units_total",
            "returns_units_transfused",
            "returns_units_returned",
        )
    )
    if det_class == "PERIOP_TRANSFUSION_EXEMPT" and has_returns_summary:
        text += (
            f"; {esc(det.get('returns_units_transfused', ''))} transfused, "
            f"{esc(det.get('returns_units_returned', ''))} returned"
        )
    return text + _msbos_pick_detail(det)


def _msbos_pick_detail(det: dict[str, str]) -> str:
    """Picker-v2 provenance fragment for a case line; "" when the picker
    seam did not run (no msbos_op_pick_status column value)."""
    pick_status = (det.get("msbos_op_pick_status") or "").strip()
    if not pick_status:
        return ""
    detail = f"; pick {esc(pick_status)}"
    source_code = (det.get("msbos_source_code") or "").strip()
    if source_code:
        detail += f" via {esc(source_code)}"
    bridge_icd9 = (det.get("msbos_bridge_icd9") or "").strip()
    if bridge_icd9:
        agreed = (det.get("msbos_bridge_human_agreed") or "").strip()
        agreement = "human-agreed" if agreed == "True" else "human-disagreed"
        detail += (
            f" -> {esc(bridge_icd9)} "
            f"(score {esc(det.get('msbos_bridge_score', ''))}, {agreement})"
        )
    return detail


def _msbos_case_line(det: dict[str, str], det_class: str) -> str:
    if (det.get("component") or "") == "platelet":
        return _msbos_platelet_case_line(det, det_class)

    reason = (det.get("msbos_reason") or "").strip()
    reserved = esc(det.get("msbos_reserved_units", ""))
    recommended = esc(det.get("msbos_recommended_units", ""))
    token = esc(det.get("msbos_token", ""))
    if reason == "over_gm_excess":
        text = f"Reserved {reserved}; MSBOS tariff G/M {recommended}"
    elif reason == "over_none":
        text = f"Reserved {reserved}; MSBOS tariff none 0"
    elif reason == "over_type_and_screen_crossmatched":
        text = f"Reserved {reserved}; MSBOS tariff T/S"
    elif reason == "within_recommendation":
        text = f"Reserved {reserved}; MSBOS tariff {token} {recommended}"
    elif reason == "type_and_screen_screen_only":
        text = f"Reserved {reserved}; MSBOS tariff T/S"
    elif reason == "over_ceiling":
        text = (
            f"Reserved {reserved}; over ceiling "
            f"{esc(det.get('msbos_ceiling_basis', ''))}"
        )
    elif reason == "within_ceiling":
        text = (
            f"Reserved {reserved}; within ceiling "
            f"{esc(det.get('msbos_ceiling_basis', ''))}"
        )
    elif reason in {"ambiguous_code", "unresolved_code"}:
        text = "MSBOS operation code unresolved"
    elif reason in {
        "ambiguous_planned_op",
        "no_planned_op",
        "operation_unresolved",
    }:
        text = "MSBOS planned operation unresolved"
    elif reason == "reservation_lookup_miss":
        text = "Reservation detail lines not linked (unlinked)"
    else:
        text = ""

    has_returns_summary = any(
        (det.get(key) or "").strip()
        for key in (
            "returns_disposition",
            "returns_units_total",
            "returns_units_transfused",
            "returns_units_returned",
        )
    )
    if det_class == "PERIOP_TRANSFUSION_EXEMPT" and has_returns_summary:
        text += (
            f"; {esc(det.get('returns_units_transfused', ''))} transfused, "
            f"{esc(det.get('returns_units_returned', ''))} returned"
        )
    return text + _msbos_pick_detail(det)


def fmt_time(raw: Any) -> str:
    """Render a HOSxP int-typed time column as HH:MM:SS.

    HOSxP exports integer times with leading zeros dropped, so a raw
    ``"44103"`` means 04:41:03 (HHMMSS). Anything that doesn't fit the
    1–6 digit shape falls through as-is.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    if s.isdigit() and 1 <= len(s) <= 6:
        s = s.zfill(6)
        return f"{s[0:2]}:{s[2:4]}:{s[4:6]}"
    return s


def fmt_dt(date_raw: Any, time_raw: Any) -> str:
    d = str(date_raw or "").split(" ")[0]
    t = fmt_time(time_raw)
    if d and t:
        return f"{d} {t}"
    return d or t or ""


def fmt_reqtype(code: Any) -> str:
    c = str(code or "").strip()
    if not c:
        return ""
    label = REQTYPE_LABELS.get(c)
    return f"{c} — {label}" if label else c


def fmt_status(code: Any, table: dict[str, str]) -> str:
    c = str(code or "").strip()
    if not c:
        return ""
    label = table.get(c)
    return f"{c} — {label}" if label else c


def fmt_unitstat(code: Any) -> str:
    c = str(code or "").strip()
    if not c:
        return ""
    label = UNITSTAT_LABELS.get(c)
    return f"{c} — {label}" if label else c


def fmt_upcoming_procedure(raw: Any) -> str | None:
    """Render the nearest upcoming procedure as days + hours.

    Returns ``None`` when the procedure is more than 7 days ahead (not
    clinically relevant to this order) so the caller can omit the line.
    """
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        hours = float(s)
    except ValueError:
        return esc(s)
    if hours > 24 * 7:
        return None
    sign = "-" if hours < 0 else ""
    total = abs(hours)
    days = int(total // 24)
    rem_h = total - days * 24
    if days:
        rem = round(rem_h)
        disp = f"{days} d" if rem == 0 else f"{days} d {rem} h"
    else:
        disp = f"{rem_h:.1f} h"
    return sign + disp


def parse_hosxp_date(raw: Any) -> date | None:
    if raw is None:
        return None
    head = str(raw).split(" ", 1)[0]
    try:
        return date.fromisoformat(head)
    except ValueError:
        return None


def parse_hosxp_datetime(date_raw: Any, time_raw: Any) -> datetime | None:
    d = parse_hosxp_date(date_raw)
    if d is None:
        return None
    t = fmt_time(time_raw)
    if not t:
        return None
    try:
        parts = [int(part) for part in t.split(":")]
    except ValueError:
        return None
    if len(parts) == 2:
        hh, mm = parts
        ss = 0
    elif len(parts) == 3:
        hh, mm, ss = parts
    else:
        return None
    try:
        return datetime(d.year, d.month, d.day, hh, mm, ss, tzinfo=TZ_LOCAL)
    except ValueError:
        return None


def parse_report_datetime(raw: Any) -> datetime | None:
    s = str(raw or "").strip()
    if not s or "(" in s:
        return None
    try:
        parsed = datetime.fromisoformat(s)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=TZ_LOCAL)
    return parsed.astimezone(TZ_LOCAL)


def parse_iptsumoprt_date(raw: Any) -> date | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        return date.fromisoformat(s.split(" ")[0])
    except ValueError:
        pass
    try:
        return datetime.strptime(s, "%B %d, %Y, %I:%M %p").date()
    except ValueError:
        pass
    try:
        return datetime.strptime(s, "%B %d, %Y").date()
    except ValueError:
        return None


def load_icd10_dict() -> dict[str, str]:
    """Build {code -> English name} from the HOSxP ICD10 dictionary."""
    out: dict[str, str] = {}
    if not ICD10_DICT_CSV.exists():
        return out
    with ICD10_DICT_CSV.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            code = (r.get("Icd10") or "").strip()
            who = (r.get("Icd10who") or "").strip()
            name = (r.get("Name") or "").strip()
            if not name:
                continue
            base = code.split("/", 1)[0] if "/" in code else code
            if base and base not in out:
                out[base] = name
            if who and who not in out:
                out[who] = name
    return out


def _trim_med_row(r: dict[str, str]) -> dict[str, str]:
    return {
        "PRSCDATE": (r.get("PRSCDATE") or "").split(" ")[0],
        "PRSCTIME": fmt_time(r.get("PRSCTIME")),
        "DRUG": " ".join(
            filter(
                None,
                [
                    (r.get("NAME_MEDITEM") or "").strip(),
                    (r.get("NAME_GENERIC") or "").strip(),
                ],
            )
        ),
        "STRENGTH": " ".join(
            filter(
                None,
                [
                    (r.get("STRENGTH") or "").strip(),
                    (r.get("STRENGTHUNIT") or "").strip(),
                ],
            )
        ),
        "DOSE": (r.get("MEDUSEQTY") or "").strip(),
    }


def _hb_row(r: dict[str, str]) -> dict[str, str]:
    return {
        "datetime": fmt_dt(r.get("LVSTDATE"), r.get("LVSTTIME")),
        "test": r.get("NAME_LABEXM") or "",
        "value": r.get("RESULT") or "",
        "min": (r.get("MINNRM") or "").strip(),
        "max": (r.get("MAXNRM") or "").strip(),
        "unit": r.get("NRMUNIT") or "",
    }


def _lab_row(r: dict[str, str]) -> dict[str, str]:
    return {
        "datetime": fmt_dt(r.get("LVSTDATE"), r.get("LVSTTIME")),
        "group": r.get("NAME_LABGRP") or "",
        "test": r.get("NAME_LABEXM") or "",
        "code": r.get("LABEXM") or "",
        "value": r.get("RESULT") or "",
        "min": (r.get("MINNRM") or "").strip(),
        "max": (r.get("MAXNRM") or "").strip(),
        "unit": r.get("NRMUNIT") or "",
    }


def _note_snippet(text: str, start: int, end: int, radius: int = 90) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(text) else ""
    return prefix + " ".join(text[lo:hi].split()) + suffix


def _format_ebl_amount(raw: str) -> str:
    amount = raw.replace(",", "")
    if not amount:
        return raw
    if "." not in amount:
        try:
            return f"{int(amount):,}"
        except ValueError:
            return raw

    whole, fraction = amount.split(".", 1)
    fraction = fraction.rstrip("0")
    try:
        whole_fmt = f"{int(whole or '0'):,}"
    except ValueError:
        return raw
    return whole_fmt if not fraction else f"{whole_fmt}.{fraction}"


def _extract_ebl_rows(
    *,
    source: str,
    when: str,
    item: str,
    text: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for match in _EBL_RE.finditer(text):
        amount = _format_ebl_amount(match.group(1))
        rows.append(
            {
                "datetime": when,
                "source": source,
                "item": item,
                "finding": f"EBL {amount} {match.group(2)}",
                "quote": _note_snippet(text, match.start(), match.end()),
            }
        )
    return rows


def _finding_key(finding: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", "", finding).lower()


def render_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return "<p class='empty'>(no rows)</p>"
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(r.get(c, ''))}</td>" for c in cols) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


_SUMMARY_DATA_COLS = ["#", "REQNO", "HN", "AN", "Hb", "Cohort", "Threshold", "Returned"]


def render_summary_table(
    rows: list[dict[str, Any]], *, show_msbos: bool = False
) -> str:
    """Render the case-summary table.

    Unlike ``render_table``, the Deterministic and LLM cells carry pre-built
    trusted pill markup (``_det_html`` / ``_llm_html``) and a row may carry a
    ``verdict-mismatch`` class (``_mismatch``). Only that generated pill span
    and the row class bypass escaping; every data cell value is HTML-escaped
    exactly as ``render_table`` would.
    """
    if not rows:
        return "<p class='empty'>(no rows)</p>"
    headers = _SUMMARY_DATA_COLS + ["Deterministic", "LLM"]
    if show_msbos:
        headers.append("MSBOS")
    head = "".join(f"<th>{esc(c)}</th>" for c in headers)
    body_parts: list[str] = []
    for r in rows:
        data_cells = "".join(
            f"<td>{esc(r.get(c, ''))}</td>" for c in _SUMMARY_DATA_COLS
        )
        verdict_cells = f"<td>{r['_det_html']}</td><td>{r['_llm_html']}</td>"
        if show_msbos:
            verdict_cells += f"<td>{r['_msbos_html']}</td>"
        tr_open = "<tr class='verdict-mismatch'>" if r.get("_mismatch") else "<tr>"
        body_parts.append(tr_open + data_cells + verdict_cells + "</tr>")
    body = "".join(body_parts)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_indications(items: list[dict[str, Any]]) -> str:
    if not items:
        return "<p class='empty'>(none)</p>"
    rows: list[str] = []
    for it in items:
        conf = it.get("confidence", "")
        conf_s = f"{conf:.2f}" if isinstance(conf, float) else esc(conf)
        rows.append(
            "<tr>"
            f"<td><b>{esc(it.get('code', ''))}</b></td>"
            f"<td><code>{esc(it.get('source_id', ''))}</code></td>"
            f"<td>{conf_s}</td>"
            f"<td>{esc(it.get('quote', ''))}</td>"
            "</tr>"
        )
    return (
        "<table class='ind'><thead><tr><th>Indication</th>"
        "<th>Source IDs</th><th>Conf</th><th>Quote</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def render_negative(items: list[dict[str, Any]]) -> str:
    if not items:
        return "<p class='empty'>(none)</p>"
    lis = "".join(f"<li>{esc(it.get('text', ''))}</li>" for it in items)
    return f"<ul>{lis}</ul>"


def _inline_md(text: str) -> str:
    """Escape, then apply minimal inline markdown (``**bold**``)."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc(text))


def _format_line(line: str) -> str:
    """Format a body line: bold a leading ``LABEL:`` prefix if present,
    then apply inline markdown."""
    m = re.match(r"^([^:\n*]{1,70}):\s+(\S.*)$", line)
    if m:
        return (
            f"<b>{_inline_md(m.group(1).strip())}:</b> {_inline_md(m.group(2).strip())}"
        )
    return _inline_md(line)


def render_reasoning(text: Any) -> str:
    """Render LLM reasoning prose as readable, structured HTML.

    The model emits ``LABEL:`` / ``**markdown**`` section leads, numbered
    and bulleted findings, and blank-line-separated paragraphs in one
    string. Escaping it into a single ``<p>`` collapses everything into a
    wall of text, so split it into headings, lists, and paragraphs.
    """
    body = str(text or "").strip()
    if not body:
        return "<p class='empty'>(none)</p>"
    out: list[str] = []
    open_tag: str | None = None  # "ol" | "ul" while a list is open

    def close_list() -> None:
        nonlocal open_tag
        if open_tag:
            out.append(f"</{open_tag}>")
            open_tag = None

    def open_list(tag: str) -> None:
        nonlocal open_tag
        if open_tag != tag:
            close_list()
            cls = "reasoning-list" if tag == "ol" else "reasoning-bullets"
            out.append(f"<{tag} class='{cls}'>")
            open_tag = tag

    for block in re.split(r"\n\s*\n", body):
        for raw in block.split("\n"):
            line = raw.strip()
            if not line:
                continue
            num = re.match(r"^\d+[.)]\s+(.*)$", line)
            bullet = re.match(r"^([-•]|\*(?!\*))\s+(.*)$", line)
            if num:
                open_list("ol")
                out.append(f"<li>{_format_line(num.group(1))}</li>")
            elif bullet:
                open_list("ul")
                out.append(f"<li>{_format_line(bullet.group(2))}</li>")
            else:
                close_list()
                md_hdr = re.match(r"^\*\*(.+?)\*\*:?$", line)
                if md_hdr:
                    out.append(
                        f"<p class='reasoning-h'>{esc(md_hdr.group(1).rstrip(':'))}</p>"
                    )
                elif line.endswith(":") and len(line) <= 80:
                    out.append(f"<p class='reasoning-h'>{esc(line[:-1])}</p>")
                else:
                    out.append(f"<p>{_format_line(line)}</p>")
        close_list()
    return f"<div class='reasoning'>{''.join(out)}</div>"


def _short(s: str, n: int = 16) -> str:
    return s[:n] + "..." if len(s) > n else s


def _field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return ""


def _returned_blood_datetime(rows: list[dict[str, str]]) -> str:
    candidates: list[str] = []
    for r in rows:
        returned_at = fmt_dt(
            _field(r, "RTNDATE", "Rtndate"),
            _field(r, "RTNTIME", "Rtntime"),
        )
        if returned_at:
            candidates.append(returned_at)
    return min(candidates) if candidates else ""


def main() -> None:
    if not BUNDLE.exists():
        sys.exit(f"bundle not found: {BUNDLE} (run sample_bundle.py first)")
    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST}")

    def _read(name: str) -> list[dict[str, str]]:
        with (BUNDLE / name).open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    def _read_optional(name: str) -> list[dict[str, str]]:
        path = BUNDLE / name
        if not path.exists():
            return []
        with path.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    def _read_preferred_optional(*names: str) -> list[dict[str, str]]:
        for name in names:
            path = BUNDLE / name
            if path.exists():
                with path.open(encoding="utf-8", newline="") as fh:
                    return list(csv.DictReader(fh))
        return []

    bdvst = _read("BDVST.csv")
    bdvstdt = _read("BDVSTDT.csv")
    related_bdvst = _read_optional("BDVST_RELATED.csv")
    related_bdvstdt = _read_optional("BDVSTDT_RELATED.csv")
    bdvsttrans = _read_optional("BDVSTTRANS.csv")
    diag = _read("Diagnosis.csv")
    lab = _read("Lab.csv")
    med = _read("Med.csv")
    iptsumoprt = _read("IPTSUMOPRT.csv")
    ipddchsumoprt = _read_optional("IPDDCHSUMOPRT.csv")
    icd9 = _read("ICD9CM.csv")
    bdvstst_dict_rows = _read("BDVSTST.csv")
    progress = _read("IPDADMPROGRESS.csv")
    focus = _read("IPDNRFOCUSDT.csv")
    incpt = _read_preferred_optional("INCPT_OPRTACT.csv", "INCPT.csv")
    optract = _read_optional("OPRTACT.csv")

    icd9_dict = {
        (r.get("Icd9cm") or "").strip().replace(".", ""): (r.get("Name") or "").strip()
        for r in icd9
    }
    bdvstst_dict: dict[str, str] = {
        (r.get("BDVSTST") or "").strip(): (r.get("NAME") or "").strip()
        for r in bdvstst_dict_rows
    }
    optract_dict = {
        _field(r, "Oprtact", "OPRTACT").strip(): r
        for r in optract
        if _field(r, "Oprtact", "OPRTACT").strip()
    }

    icd10_dict = load_icd10_dict()
    for r in diag:
        code = (r.get("ICD10") or "").strip()
        name = (r.get("NAME_ICD10") or "").strip()
        if code and name and code not in icd10_dict:
            icd10_dict[code] = name
    print(f"ICD-10 dictionary: {len(icd10_dict)} codes")

    det_rows = list(csv.DictReader(DET_REPORT.open(encoding="utf-8")))
    det_by_reqno = {r["reqno"]: r for r in det_rows}
    if MSBOS_RESERVATION_PILOT_ENABLED:
        # det_by_reqno joins on the bare REQNO, so ANY duplicate REQNO in the report
        # scope (a REQNO reused across admissions, or an RBC+platelet split) would let
        # one row silently overwrite another and mis-render its verdict/annotation. Fail
        # loud instead (plan #176: "fail loud on duplicate REQNOs in the report scope").
        reqno_counts = Counter((r.get("reqno") or "").strip() for r in det_rows)
        dups = sorted(reqno for reqno, count in reqno_counts.items() if count > 1)
        if dups:
            raise ValueError(
                "duplicate REQNO in report scope, cannot safely join "
                f"MSBOS annotations: {dups}"
            )

    llm_report = (
        json.loads(LLM_REPORT.read_text(encoding="utf-8"))
        if LLM_REPORT.exists()
        else []
    )
    llm_by_reqno = {r["reqno"]: r for r in llm_report}

    manifest_rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    bdvst_by_reqno = {r["REQNO"]: r for r in bdvst}
    related_bdvst_by_reqno = {
        **bdvst_by_reqno,
        **{r["REQNO"]: r for r in related_bdvst},
    }
    related_bdvstdt_rows = related_bdvstdt or bdvstdt

    case_html_parts: list[str] = []
    summary_rows: list[dict[str, str]] = []
    case_mismatch_tags: list[str] = []
    msbos_counts: dict[str, dict[str, int]] = {
        render_class: {
            "denominator": 0,
            "above": 0,
            "within": 0,
            "within_ceiling": 0,
            "unresolved": 0,
        }
        for render_class in _MSBOS_RENDER_CLASSES
    }

    for i, m in enumerate(manifest_rows, start=1):
        hn = m["HN"]
        reqno = m["REQNO"]
        an = m["AN"]
        bdv = bdvst_by_reqno.get(reqno, {})
        det = det_by_reqno.get(reqno) or {}

        anchor_date = (bdv.get("REQDATE") or "").split(" ")[0]
        anchor_time = fmt_time(bdv.get("REQTIME"))
        anchor_dt = parse_hosxp_datetime(
            bdv.get("REQDATE"), bdv.get("REQTIME")
        ) or parse_hosxp_datetime(bdv.get("BDVSTDATE"), bdv.get("BDVSTTIME"))
        # Reserve-ahead elective orders re-window all evidence (Hb, notes, CBC,
        # meds, procedures) onto the issue (PICK/USE) datetime; the displayed
        # order anchor above stays REQTIME. Mirror the pipeline's re-anchor so
        # the highlighted in-window rows match what the gate actually scored.
        # Non-reanchored cases keep window_anchor == order anchor unchanged.
        evidence_anchor_reason = (
            det.get("evidence_anchor_reason") or "order_datetime"
        ).strip()
        reanchored = evidence_anchor_reason == "issue_reanchor"
        evidence_anchor_dt = parse_report_datetime(
            det.get("evidence_anchor_datetime_local")
        )
        window_anchor_dt = (
            evidence_anchor_dt if (reanchored and evidence_anchor_dt) else anchor_dt
        )
        window_anchor_d = window_anchor_dt.date() if window_anchor_dt else None
        hb_anchor_dt = parse_report_datetime(det.get("hb_anchor_datetime_local"))
        hb_anchor_dt = hb_anchor_dt or window_anchor_dt
        hb_anchor_reason = (det.get("hb_anchor_reason") or "order_datetime").strip()
        try:
            upcoming_h = float(det.get("upcoming_procedure_hours") or "")
        except ValueError:
            upcoming_h = None
        notes_hi_d = (
            (window_anchor_d + timedelta(days=WINDOW_NOTES_DAYS_AFTER))
            if window_anchor_d
            else None
        )
        periop_notes_hi_d = notes_hi_d
        if anchor_dt and upcoming_h is not None and 0 <= upcoming_h <= 72:
            periop_hi = (
                anchor_dt
                + timedelta(hours=upcoming_h)
                + timedelta(days=WINDOW_PERIOP_NOTES_DAYS_AFTER)
            ).date()
            periop_notes_hi_d = (
                max(periop_notes_hi_d, periop_hi) if periop_notes_hi_d else periop_hi
            )
        hb_lo_dt = (
            (hb_anchor_dt - timedelta(days=WINDOW_HB_DAYS)) if hb_anchor_dt else None
        )
        # Hb upper bound. Re-anchored orders extend it past the USE issue time to
        # the end of op-day +N (matching the LLM gate) so the intra-op nadir and
        # the morning-after draw are visible in the Hb table; otherwise it is the
        # Hb lookup anchor itself (backward-only).
        hb_hi_dt = hb_anchor_dt
        if reanchored and window_anchor_d is not None:
            hb_hi_dt = datetime.combine(
                window_anchor_d + timedelta(days=WINDOW_HB_REANCHOR_DAYS_AFTER + 1),
                datetime.min.time(),
                tzinfo=TZ_LOCAL,
            )
        notes_lo = (
            (window_anchor_d - timedelta(days=WINDOW_NOTES_DAYS_BEFORE))
            if window_anchor_d
            else None
        )
        notes_hi = notes_hi_d
        periop_notes_hi = periop_notes_hi_d
        proc_lo = (
            (window_anchor_d - timedelta(days=WINDOW_PROC_DAYS))
            if window_anchor_d
            else None
        )
        proc_hi = (
            (window_anchor_d + timedelta(days=WINDOW_PROC_DAYS))
            if window_anchor_d
            else None
        )

        def _in_notes_window(d: date | None) -> bool:
            return (
                d is not None
                and notes_lo is not None
                and notes_hi is not None
                and notes_lo <= d <= notes_hi
            )

        def _in_periop_notes_window(d: date | None) -> bool:
            return (
                d is not None
                and notes_lo is not None
                and periop_notes_hi is not None
                and notes_lo <= d <= periop_notes_hi
            )

        def _in_proc_window(d: date | None) -> bool:
            return d is None or proc_lo is None or (proc_lo <= d <= proc_hi)

        def _in_hb_window_dt(d: datetime | None) -> bool:
            return (
                d is not None
                and hb_lo_dt is not None
                and hb_hi_dt is not None
                and hb_lo_dt <= d <= hb_hi_dt
            )

        line_items = [r for r in bdvstdt if r.get("REQNO") == reqno]
        related_line_items = [
            r
            for r in related_bdvstdt_rows
            if r.get("REQNO") != reqno
            and r.get("HN") == hn
            and (related_bdvst_by_reqno.get(r.get("REQNO", ""), {}).get("AN") == an)
        ]
        ordered_products = {
            (r.get("BDTYPE") or "").strip().upper()
            for r in line_items
            if r.get("BDTYPE")
        }
        if RETURNS_LEDGER_ENABLED:
            # BDVSTTRANS joins by REQNO exactly (spec #119), so a multi-order
            # admission shows only this order's unit rows.
            trans_rows = [r for r in bdvsttrans if _field(r, "REQNO", "Reqno") == reqno]
        else:
            trans_rows = [
                r
                for r in bdvsttrans
                if (_field(r, "AN", "An") == an or _field(r, "HN", "Hn") == hn)
                and (
                    not ordered_products
                    or _field(r, "BDTYPE", "Bdtype").strip().upper() in ordered_products
                )
            ]
        trans_rows.sort(
            key=lambda r: (
                fmt_dt(
                    _field(r, "PAYDATE", "Paydate"),
                    _field(r, "PAYTIME", "Paytime"),
                ),
                fmt_dt(
                    _field(r, "GIVEDATE", "Givedate"),
                    _field(r, "GIVETIME", "Givetime"),
                ),
                fmt_dt(
                    _field(r, "RTNDATE", "Rtndate"),
                    _field(r, "RTNTIME", "Rtntime"),
                ),
            )
        )

        # Diagnoses, deduped by ICD-10, type-precedence-preserving.
        diag_by_code: dict[str, dict[str, str]] = {}
        for r in diag:
            if r.get("AN") != an:
                continue
            code = (r.get("ICD10") or "").strip()
            if not code:
                continue
            desc = icd10_dict.get(code) or (r.get("NAME_ICD10") or "").strip()
            row = {
                "ICD10": code,
                "Description": desc,
                "Type": r.get("NAME_DIAGTYPE") or "",
                "ICD10WHO": r.get("ICD10WHO") or "",
            }
            prev = diag_by_code.get(code)
            if prev is None or _TYPE_RANK.get(row["Type"], 9) < _TYPE_RANK.get(
                prev["Type"], 9
            ):
                diag_by_code[code] = row
        diag_rows = list(diag_by_code.values())
        diag_rows.sort(key=lambda d: (d["Type"] != "Principal Diagnosis", d["ICD10"]))

        hb_rows = [
            _hb_row(r)
            for r in lab
            if r.get("AN") == an
            and (r.get("LABEXM") or "").strip() in {"290095", "500001"}
            and _in_hb_window_dt(
                parse_hosxp_datetime(r.get("LVSTDATE"), r.get("LVSTTIME"))
            )
        ]
        hb_rows.sort(key=lambda r: r["datetime"], reverse=True)

        anc_rows = [
            _hb_row(r)
            for r in lab
            if r.get("AN") == an
            and (r.get("LABEXM") or "").strip() == "290093"
            and _in_notes_window(parse_hosxp_date(r.get("LVSTDATE")))
        ]
        anc_rows.sort(key=lambda r: r["datetime"], reverse=True)

        cbc_rows = [
            _lab_row(r)
            for r in lab
            if r.get("AN") == an
            and (r.get("LABEXM") or "").strip() in CBC_LABEXM_CODES
            and _in_notes_window(parse_hosxp_date(r.get("LVSTDATE")))
        ]
        cbc_rows.sort(key=lambda r: r["datetime"], reverse=True)

        med_rows = [
            _trim_med_row(r)
            for r in med
            if r.get("AN") == an
            and _in_notes_window(parse_hosxp_date(r.get("PRSCDATE")))
        ]
        med_rows.sort(key=lambda r: (r["PRSCDATE"], r["PRSCTIME"]), reverse=True)

        proc_rows: list[dict[str, str]] = []
        for source, rows in (
            ("IPTSUMOPRT", iptsumoprt),
            ("IPDDCHSUMOPRT", ipddchsumoprt),
        ):
            for r in rows:
                if _field(r, "An", "AN") != an:
                    continue
                pdate = parse_iptsumoprt_date(_field(r, "Indate", "INDATE"))
                if not _in_proc_window(pdate):
                    continue
                code = _field(r, "Icd9cm", "ICD9CM").strip().replace(".", "")
                proc_text = _field(r, "Oprttext", "OPRTTEXT").strip()
                proc_rows.append(
                    {
                        "Source": source,
                        "ICD9CM": code,
                        "OPRTACT": "",
                        "Name": proc_text or icd9_dict.get(code, ""),
                        "INDATE": _field(r, "Indate", "INDATE"),
                        "INTIME": fmt_time(_field(r, "Intime", "INTIME")),
                        "OUTDATE": _field(r, "Outdate", "OUTDATE"),
                        "OUTTIME": fmt_time(_field(r, "Outtime", "OUTTIME")),
                        "ORFLAG": _field(r, "Orflag", "ORFLAG"),
                        "INCOME": "",
                        "INCGRP": "",
                    }
                )
        for r in incpt:
            if _field(r, "An", "AN") != an:
                continue
            if _field(r, "Canceldate", "CANCELDATE").strip():
                continue
            incgrp = _field(r, "Incgrp", "INCGRP").strip()
            if incgrp not in {"110", "111"}:
                continue
            pdate = parse_iptsumoprt_date(_field(r, "Incdate", "INCDATE"))
            if not _in_proc_window(pdate):
                continue
            income = _field(r, "Income", "INCOME").strip()
            ordercode = _field(r, "Ordercode", "ORDERCODE").strip()
            optract_code = _field(r, "O__Oprtact", "O__OPRTACT").strip()
            row_op_map = {
                "Icd9cm": _field(r, "O__Icd9cm", "O__ICD9CM").strip(),
                "Icd9cmadd1": _field(r, "O__Icd9cmadd1", "O__ICD9CMADD1").strip(),
                "Icd9cmadd2": _field(r, "O__Icd9cmadd2", "O__ICD9CMADD2").strip(),
                "Name En": _field(r, "O__Name En", "O__NAME_EN", "O__NAME EN").strip(),
                "Name": _field(r, "O__Name", "O__NAME").strip(),
            }
            op_map = (
                row_op_map
                if any(row_op_map.values())
                else optract_dict.get(optract_code)
                or optract_dict.get(income)
                or optract_dict.get(ordercode)
                or {}
            )
            op_name = (
                _field(op_map, "Name En", "NAME EN").strip()
                or _field(op_map, "Name", "NAME").strip()
                or _field(r, "Incgrp → Name", "INCGRP → NAME", "INCGRP__NAME").strip()
            )
            op_codes = [
                c.strip().replace(".", "")
                for c in (
                    _field(op_map, "Icd9cm", "ICD9CM"),
                    _field(op_map, "Icd9cmadd1", "ICD9CMADD1"),
                    _field(op_map, "Icd9cmadd2", "ICD9CMADD2"),
                )
                if c.strip()
            ]
            proc_rows.append(
                {
                    "Source": "INCPT",
                    "ICD9CM": ",".join(op_codes),
                    "OPRTACT": optract_code or income or ordercode,
                    "Name": op_name,
                    "INDATE": _field(r, "Incdate", "INCDATE"),
                    "INTIME": fmt_time(_field(r, "Inctime", "INCTIME")),
                    "OUTDATE": "",
                    "OUTTIME": "",
                    "ORFLAG": "",
                    "INCOME": income,
                    "INCGRP": incgrp,
                }
            )

        # Progress notes: first N of AN + window. Dedup by (date, item).
        an_progress = [r for r in progress if r.get("AN") == an]
        an_progress.sort(key=lambda r: r.get("PROGDATE") or "")
        first_n = an_progress[:PROGRESS_FIRST_N]
        near_anchor = [
            r
            for r in an_progress
            if _in_notes_window(parse_hosxp_date(r.get("PROGDATE")))
        ]
        periop_progress_notes = [
            r
            for r in an_progress
            if _in_periop_notes_window(parse_hosxp_date(r.get("PROGDATE")))
        ]
        seen: set[tuple[str, str]] = set()
        prog_notes: list[dict[str, str]] = []
        for r in first_n + near_anchor:
            key = ((r.get("PROGDATE") or ""), (r.get("ITEMNO") or ""))
            if key in seen:
                continue
            seen.add(key)
            prog_notes.append(r)
        prog_notes.sort(key=lambda r: r.get("PROGDATE") or "")

        focus_notes = [
            r
            for r in focus
            if r.get("AN") == an
            and _in_notes_window(parse_hosxp_date(r.get("PROGRESSDATE")))
        ]
        focus_notes.sort(
            key=lambda r: (r.get("PROGRESSDATE") or "", r.get("PROGRESSTIME") or "")
        )
        periop_focus_notes = [
            r
            for r in focus
            if r.get("AN") == an
            and _in_periop_notes_window(parse_hosxp_date(r.get("PROGRESSDATE")))
        ]
        periop_focus_notes.sort(
            key=lambda r: (r.get("PROGRESSDATE") or "", r.get("PROGRESSTIME") or "")
        )

        periop_evidence: list[dict[str, str]] = []
        seen_periop: set[tuple[str, str, str, str]] = set()
        for n in periop_progress_notes:
            prog_date = parse_hosxp_date(n.get("PROGDATE"))
            if not _in_periop_notes_window(prog_date):
                continue
            when = (n.get("PROGDATE") or "").split(" ")[0]
            item = n.get("ITEMNO") or ""
            text = "\n".join(
                part
                for part in (
                    n.get("PROGDESC") or "",
                    n.get("PROGLIST") or "",
                    n.get("SUBJECTIVE") or "",
                    n.get("OBJECTIVE") or "",
                    n.get("ASSESSMENT") or "",
                    n.get("PLAN") or "",
                )
                if part
            )
            for row in _extract_ebl_rows(
                source="IPDADMPROGRESS", when=when, item=item, text=text
            ):
                key = (
                    row["source"],
                    row["datetime"],
                    row["item"],
                    _finding_key(row["finding"]),
                )
                if key not in seen_periop:
                    seen_periop.add(key)
                    periop_evidence.append(row)
        for n in periop_focus_notes:
            when = fmt_dt(n.get("PROGRESSDATE"), n.get("PROGRESSTIME"))
            item = n.get("ITEMNO") or ""
            text = "\n".join(
                part
                for part in (
                    n.get("FOCUS") or "",
                    n.get("ACTION") or "",
                    n.get("RESPONSE") or "",
                )
                if part
            )
            for row in _extract_ebl_rows(
                source="IPDNRFOCUSDT", when=when, item=item, text=text
            ):
                key = (
                    row["source"],
                    row["datetime"],
                    row["item"],
                    _finding_key(row["finding"]),
                )
                if key not in seen_periop:
                    seen_periop.add(key)
                    periop_evidence.append(row)

        llm = llm_by_reqno.get(reqno)
        det_class = det.get("classification") or "excluded"
        det_class_upper = det_class.upper()
        # Annotated RBC and platelet returns rows are counted together. A blank
        # msbos_reason -> _msbos_reason_bucket is None; skip it so the bucket tallies
        # always sum to the denominator (no "silent" uncounted rows).
        msbos_bucket = (
            _msbos_reason_bucket(det.get("msbos_reason", ""))
            if MSBOS_RESERVATION_PILOT_ENABLED
            and det_class_upper in _MSBOS_RENDER_CLASSES
            else None
        )
        if msbos_bucket is not None:
            msbos_counts[det_class_upper]["denominator"] += 1
            msbos_counts[det_class_upper][msbos_bucket] += 1
        # ``llm_final`` may be explicitly null when run_llm_leg.py persisted
        # a partial result (missing batch row); chaining ``.get`` on None
        # would crash, so coalesce twice.
        llm_final_obj = (llm or {}).get("llm_final") or {}
        if not llm:
            llm_final = "(LLM not run)"
        elif not llm_final_obj:
            llm_final = "(LLM data missing)"
        else:
            llm_final = llm_final_obj.get("final_classification") or "—"

        _has_llm = bool(llm and llm_final_obj)
        if not _has_llm:
            _nav_tag = ""
        elif det_class.upper() == (llm_final or "").upper():
            _nav_tag = ""
        # This store-only terminal verdict makes no transfusion claim, so a
        # deterministic-vs-LLM transfusion disagreement is not meaningful.
        elif (llm_final or "").upper() == "PREOP_RESERVATION_UNCONFIRMED":
            _nav_tag = ""
        elif {"POTENTIALLY_INAPPROPRIATE", "APPROPRIATE"} <= {
            det_class.upper(),
            (llm_final or "").upper(),
        }:
            _nav_tag = " <b class='nav-flag nav-flag--major'>[!!]</b>"
        else:
            _nav_tag = " <b class='nav-flag'>[!]</b>"
        case_mismatch_tags.append(_nav_tag)

        det_pill = (
            f"<span class='cls cls-{esc(det_class).lower()}'>"
            f"{esc(_display_cls(det_class, det.get('rationale')))}</span>"
        )
        if _has_llm:
            llm_cell = (
                f"<span class='cls cls-{esc(llm_final).lower()}'>"
                f"{esc(_display_cls(llm_final))}</span>"
            )
        else:
            llm_cell = esc(_display_cls(llm_final))

        summary_rows.append(
            {
                "#": str(i),
                "REQNO": reqno,
                "HN": _short(hn),
                "AN": _short(an),
                "Hb": det.get("hb_value_g_dl", "") or "—",
                "Cohort": det.get("cohort_label", "") or "—",
                "Threshold": det.get("cohort_threshold", "") or "—",
                "Returned": det.get("returned_blood_datetime_local")
                or _returned_blood_datetime(trans_rows)
                or "—",
                "_det_html": det_pill,
                "_llm_html": llm_cell,
                "_msbos_html": _msbos_summary_pill(det),
                "_mismatch": bool(_nav_tag),
            }
        )

        admission_date = (bdv.get("BDVSTDATE") or "").split(" ")[0] or "—"
        ebl_summary = (
            "; ".join(dict.fromkeys(r["finding"] for r in periop_evidence))
            if periop_evidence
            else "—"
        )

        # ----- Section assembly -----
        returned_blood_dt = (
            det.get("returned_blood_datetime_local")
            or _returned_blood_datetime(trans_rows)
            or "—"
        )
        upcoming_disp = fmt_upcoming_procedure(det.get("upcoming_procedure_hours"))
        # The deterministic gate resolves Hb backward, so its reason stays
        # "order_datetime" even on re-anchored orders; relabel it to match the
        # issue anchor the Hb lookup datetime above actually reflects.
        hb_lookup_reason_disp = det.get("hb_anchor_reason") or "order_datetime"
        if reanchored and hb_lookup_reason_disp == "order_datetime":
            hb_lookup_reason_disp = "issue_reanchor"
        meta_items = [
            f"<div><b>HN:</b> <code>{esc(hn)}</code></div>",
            f"<div><b>AN:</b> <code>{esc(an)}</code></div>",
            f"<div><b>Order anchor:</b> {esc(anchor_date)} {esc(anchor_time)}</div>",
            f"<div><b>Dispense datetime (BDVST PICK):</b> {esc(det.get('dispense_datetime_local') or '—')}</div>",
            f"<div><b>Use datetime (BDVSTDT USE):</b> {esc(det.get('use_datetime_local') or '—')}</div>",
            *(
                [
                    "<div><b><abbr title='Blood reserved this many hours before it was issued; all evidence windows (Hb, notes, CBC, meds, procedures) below are re-anchored onto the issue datetime instead of the reservation order anchor'>Evidence re-anchored to issue</abbr>:</b> "
                    f"{esc(det.get('evidence_anchor_datetime_local') or '—')} "
                    f"(+{esc(det.get('reanchor_gap_hours') or '—')}h after order)</div>"
                ]
                if reanchored
                else []
            ),
            f"<div><b>Blood return datetime:</b> {esc(returned_blood_dt)}</div>",
            f"<div><b>Products ordered:</b> {esc(det.get('products_ordered') or '—')}</div>",
            f"<div><b>Hb @ anchor:</b> {esc(det.get('hb_value_g_dl') or '—')} g/dL "
            f"({esc(det.get('hb_freshness') or '—')}, source {esc(det.get('hb_source') or '—')})</div>",
            f"<div><b>Hb lookup anchor:</b> "
            f"{esc(det.get('hb_anchor_datetime_local') or (det.get('evidence_anchor_datetime_local') if reanchored else None) or f'{anchor_date} {anchor_time}'.strip() or '—')} "
            f"({esc(hb_lookup_reason_disp)})</div>",
            f"<div><b><abbr title='Patient clinical cohort used to determine the Hb threshold for this order'>Cohort</abbr>:</b> "
            f"{esc(det.get('cohort_label') or '—')} "
            f"(threshold {esc(det.get('cohort_threshold') or 'n/a')})</div>",
        ]
        if upcoming_disp is not None:
            meta_items.append(f"<div><b>Upcoming procedure:</b> {upcoming_disp}</div>")
        meta_items.append(f"<div><b>EBL evidence:</b> {esc(ebl_summary)}</div>")
        parts: list[str] = [
            f"<section class='case' id='case-{i}'>",
            "<div class='case-hd'>",
            f"<h3>Case {i} — REQNO {esc(reqno)}</h3>",
            f"<button class='mark-reviewed' data-case='{i}'"
            " onclick='toggleReviewed(this)'>Mark reviewed</button>",
            "</div>",
            "<div class='meta'>",
            *meta_items,
            "</div>",
        ]

        # Verdict
        parts.append("<div class='verdict'>")
        parts.append("<div class='vbox det'>")
        parts.append("<h4>Deterministic verdict</h4>")
        parts.append(
            f"<div class='cls cls-{esc(det_class).lower()}'>"
            f"{esc(_display_cls(det_class, det.get('rationale')))}</div>"
        )
        parts.append(
            f"<div class='rationale'>rationale: "
            f"{_code_abbr(det.get('rationale') or '—', _RATIONALE_LABELS)}</div>"
        )
        parts.append(
            f"<div class='rationale'>bypass: "
            f"{_code_abbr(det.get('bypass_reason') or 'none', _BYPASS_LABELS)}</div>"
        )
        if (
            MSBOS_RESERVATION_PILOT_ENABLED
            and det_class_upper in _MSBOS_RENDER_CLASSES
            and (det.get("msbos_reason") or "").strip()
        ):
            parts.append(
                "<div class='rationale'>MSBOS reservation: "
                f"{_msbos_case_line(det, det_class_upper)}</div>"
            )
        parts.append("</div>")

        parts.append("<div class='vbox llm'>")
        parts.append("<h4>LLM verdict</h4>")
        # ``llm_final`` may be explicitly None when run_llm_leg.py persisted
        # a partial run (e.g. the batch row was missing). Treat that the
        # same as "no LLM row at all" so the page renders instead of
        # crashing.
        llm_block = (llm or {}).get("llm_final") if llm else None
        if llm_block:
            fc = llm_block["final_classification"]
            conf = llm_block["confidence"]
            model = llm_block.get("model", "")
            parts.append(
                f"<div class='cls cls-{esc(fc).lower()}'>"
                f"{esc(_display_cls(fc))} <span class='conf'>(conf {conf:.2f}; "
                f"{esc(model)})</span></div>"
            )
            review_reason = llm_block.get("review_reason") or "model_verdict"
            parts.append(
                f"<div class='rationale'>provenance: "
                f"{_code_abbr(review_reason, _REVIEW_REASON_LABELS)}</div>"
            )
            parts.append("<h5>Indications</h5>")
            parts.append(render_indications(llm_block["indications"]))
            parts.append("<h5>Negative evidence</h5>")
            parts.append(render_negative(llm_block["negative_evidence"]))
            parts.append("<details><summary><b>Reasoning — English</b></summary>")
            parts.append(render_reasoning(llm_block["reasoning_en"]))
            parts.append("</details>")
            parts.append(
                "<details lang='th'><summary><b>Reasoning — ภาษาไทย</b></summary>"
            )
            parts.append(render_reasoning(llm_block["reasoning_th"]))
            parts.append("</details>")
        elif llm:
            parts.append(
                "<p class='empty'>(LLM verdict missing — batch row dropped or unparsable)</p>"
            )
        else:
            parts.append(
                "<p class='empty'>(LLM not run — deterministic verdict is final)</p>"
            )
        parts.append("</div>")
        parts.append("</div>")

        # Source data
        parts.append(
            "<div class='src-hd'><h4>Source data (real rows)</h4>"
            "<button class='expand-toggle'"
            " onclick='toggleSrcDetails(this)'>Expand all</button></div>"
        )

        order_icd10 = (bdv.get("ICD10") or "").strip()
        order_icd10_name = icd10_dict.get(order_icd10, "")
        order_icd10_disp = (
            f"{order_icd10} — {order_icd10_name}"
            if order_icd10 and order_icd10_name
            else order_icd10
        )

        parts.append("<details><summary>Order — BDVST + BDVSTDT</summary>")
        parts.append(
            render_table(
                [
                    {
                        "REQNO": bdv.get("REQNO", ""),
                        "REQDATE": (bdv.get("REQDATE") or "").split(" ")[0],
                        "REQTIME": fmt_time(bdv.get("REQTIME")),
                        "BDVSTDATE": (bdv.get("BDVSTDATE") or "").split(" ")[0],
                        "BDVSTTIME": fmt_time(bdv.get("BDVSTTIME")),
                        "PICKDATE": (bdv.get("PICKDATE") or "").split(" ")[0],
                        "PICKTIME": fmt_time(bdv.get("PICKTIME")),
                        "BDVSTSTATUS": fmt_status(bdv.get("BDVSTST"), bdvstst_dict),
                        "REQTYPE": fmt_reqtype(bdv.get("REQTYPE")),
                        "ICD10 (order reason)": order_icd10_disp,
                        "DIAGNOSIS (order reason text)": bdv.get("DIAGNOSIS", ""),
                    }
                ],
                [
                    "REQNO",
                    "REQDATE",
                    "REQTIME",
                    "BDVSTDATE",
                    "BDVSTTIME",
                    "PICKDATE",
                    "PICKTIME",
                    "BDVSTSTATUS",
                    "REQTYPE",
                    "ICD10 (order reason)",
                    "DIAGNOSIS (order reason text)",
                ],
            )
        )
        parts.append("<p><b>Line items (BDVSTDT):</b></p>")
        parts.append(
            render_table(
                [
                    {
                        "BDTYPE (product)": r.get("BDTYPE", ""),
                        "UNITAMT": r.get("UNITAMT", ""),
                        "USEDATE": (r.get("USEDATE") or "").split(" ")[0],
                        "USETIME": fmt_time(r.get("USETIME")),
                    }
                    for r in line_items
                ],
                ["BDTYPE (product)", "UNITAMT", "USEDATE", "USETIME"],
            )
        )
        if related_line_items:
            parts.append("<p><b>Related same-admission blood products:</b></p>")
            parts.append(
                render_table(
                    [
                        {
                            "REQNO": r.get("REQNO", ""),
                            "BDTYPE (product)": r.get("BDTYPE", ""),
                            "UNITAMT": r.get("UNITAMT", ""),
                            "USEDATE": (r.get("USEDATE") or "").split(" ")[0],
                            "USETIME": fmt_time(r.get("USETIME")),
                        }
                        for r in related_line_items
                    ],
                    [
                        "REQNO",
                        "BDTYPE (product)",
                        "UNITAMT",
                        "USEDATE",
                        "USETIME",
                    ],
                )
            )
        parts.append("<p><b>Blood bank transactions (BDVSTTRANS):</b></p>")
        parts.append(
            render_table(
                [
                    {
                        "BDTYPE": _field(r, "BDTYPE", "Bdtype"),
                        "UNITSTAT": fmt_unitstat(_field(r, "UNITSTAT", "Unitstat")),
                        "PAY datetime": fmt_dt(
                            _field(r, "PAYDATE", "Paydate"),
                            _field(r, "PAYTIME", "Paytime"),
                        ),
                        "GIVE datetime": fmt_dt(
                            _field(r, "GIVEDATE", "Givedate"),
                            _field(r, "GIVETIME", "Givetime"),
                        ),
                        "RETURN datetime": fmt_dt(
                            _field(r, "RTNDATE", "Rtndate"),
                            _field(r, "RTNTIME", "Rtntime"),
                        ),
                        "QTYUSE": _field(r, "QTYUSE", "Qtyuse"),
                        "PAYCOMM": _field(r, "PAYCOMM", "Paycomm"),
                    }
                    for r in trans_rows
                ],
                [
                    "BDTYPE",
                    "UNITSTAT",
                    "PAY datetime",
                    "GIVE datetime",
                    "RETURN datetime",
                    "QTYUSE",
                    "PAYCOMM",
                ],
            )
        )
        parts.append("</details>")

        parts.append(
            f"<details><summary>Diagnoses ({len(diag_rows)} rows, "
            f"AN-scoped, all charted on admission {admission_date})</summary>"
        )
        parts.append(
            "<p class='empty'>Per-row charting date is not recoverable from "
            "the source bundle (HOSxP <code>V_DATE</code> is Excel-corrupted "
            "to <code>00:00.0</code> in every row); the admission date shown "
            "in the section title above is the order's admission date, used "
            "as a temporal proxy.</p>"
        )
        parts.append(
            render_table(
                diag_rows,
                ["ICD10", "Description", "Type", "ICD10WHO"],
            )
        )
        parts.append("</details>")

        hb_window_str = (
            f"{hb_lo_dt.isoformat(sep=' ')} … {hb_hi_dt.isoformat(sep=' ')}"
            if hb_lo_dt and hb_hi_dt
            else "(no anchor)"
        )
        hb_anchor_note = (
            f"; Hb lookup anchor: {hb_anchor_reason}"
            if hb_anchor_reason and hb_anchor_reason != "order_datetime"
            else ""
        )
        parts.append(
            f"<details><summary>Hb history ({len(hb_rows)} rows, "
            f"7-day pre-anchor window {hb_window_str}{hb_anchor_note})</summary>"
        )
        parts.append(
            render_table(hb_rows, ["datetime", "test", "value", "min", "max", "unit"])
        )
        parts.append("</details>")

        window_str = (
            f"{notes_lo.isoformat()} … {notes_hi.isoformat()}"
            if notes_lo and notes_hi
            else "(no anchor)"
        )
        periop_window_str = (
            f"{notes_lo.isoformat()} … {periop_notes_hi.isoformat()}"
            if notes_lo and periop_notes_hi
            else "(no anchor)"
        )
        if anc_rows:
            parts.append(
                f"<details><summary>ANC — Absolute Neutrophil Count "
                f"(infection / neutropenia / heme-malignancy signal) "
                f"({len(anc_rows)} rows, window {window_str})</summary>"
            )
            parts.append(
                render_table(
                    anc_rows, ["datetime", "test", "value", "min", "max", "unit"]
                )
            )
            parts.append("</details>")

        parts.append(
            f"<details><summary>CBC — Hb/Plt/WBC/Neutrophils "
            f"({len(cbc_rows)} rows, window {window_str})</summary>"
        )
        parts.append(
            render_table(
                cbc_rows,
                ["datetime", "group", "test", "code", "value", "min", "max", "unit"],
            )
        )
        parts.append("</details>")

        parts.append(
            f"<details><summary>Perioperative blood-loss evidence "
            f"({len(periop_evidence)} rows, note window {periop_window_str})</summary>"
        )
        parts.append(
            render_table(
                periop_evidence,
                ["datetime", "source", "finding", "quote"],
            )
        )
        parts.append("</details>")

        proc_window_str = (
            f"{proc_lo.isoformat()} … {proc_hi.isoformat()}"
            if proc_lo and proc_hi
            else "(no anchor)"
        )
        parts.append(
            f"<details><summary>Procedures — IPTSUMOPRT + IPDDCHSUMOPRT + INCPT "
            f"({len(proc_rows)} rows, AN-scoped, ±1 week window "
            f"{proc_window_str})</summary>"
        )
        parts.append(
            render_table(
                proc_rows,
                [
                    "Source",
                    "ICD9CM",
                    "OPRTACT",
                    "INCOME",
                    "Name",
                    "INDATE",
                    "INTIME",
                    "OUTDATE",
                    "OUTTIME",
                    "ORFLAG",
                    "INCGRP",
                ],
            )
        )
        parts.append("</details>")

        parts.append(
            f"<details><summary>Meds ({len(med_rows)} rows, "
            f"window {window_str})</summary>"
        )
        parts.append(
            render_table(med_rows, ["PRSCDATE", "PRSCTIME", "DRUG", "STRENGTH", "DOSE"])
        )
        parts.append("</details>")

        parts.append(
            f"<details><summary>Progress notes — IPDADMPROGRESS "
            f"({len(prog_notes)} rows: first {PROGRESS_FIRST_N} of AN + "
            f"window {window_str})</summary>"
        )
        if prog_notes:
            for n in prog_notes:
                d = (n.get("PROGDATE") or "").split(" ")[0]
                itemno = n.get("ITEMNO") or ""
                progno = n.get("PROGNO") or ""
                progdesc = (n.get("PROGDESC") or "").strip()
                proglist = (n.get("PROGLIST") or "").strip()
                dentdct = (n.get("DENTDCT") or "").strip()
                subj = (n.get("SUBJECTIVE") or "").strip()
                obj = (n.get("OBJECTIVE") or "").strip()
                asmt = (n.get("ASSESSMENT") or "").strip()
                plan = (n.get("PLAN") or "").strip()
                parts.append(
                    f"<div class='note'><div class='note-date'>{esc(d)}"
                    f" — item {esc(itemno)} / progno {esc(progno)}</div>"
                )
                if progdesc:
                    parts.append(f"<pre><b>desc:</b> {esc(progdesc)}</pre>")
                if proglist:
                    parts.append(f"<pre><b>list:</b> {esc(proglist)}</pre>")
                if dentdct:
                    parts.append(f"<pre><b>dent:</b> {esc(dentdct)}</pre>")
                if subj:
                    parts.append(f"<pre><b>S:</b>\n{esc(subj)}</pre>")
                if obj:
                    parts.append(f"<pre><b>O:</b>\n{esc(obj)}</pre>")
                if asmt:
                    parts.append(f"<pre><b>A:</b>\n{esc(asmt)}</pre>")
                if plan:
                    parts.append(f"<pre><b>P:</b>\n{esc(plan)}</pre>")
                if not (progdesc or proglist or dentdct or subj or obj or asmt or plan):
                    parts.append("<p class='empty'>(empty row)</p>")
                parts.append("</div>")
        else:
            parts.append("<p class='empty'>(none)</p>")
        parts.append("</details>")

        parts.append(
            f"<details><summary>Nursing focus notes — IPDNRFOCUSDT "
            f"({len(focus_notes)} rows, window {window_str})</summary>"
        )
        if focus_notes:
            for n in focus_notes:
                d = (n.get("PROGRESSDATE") or "").split(" ")[0]
                t = fmt_time(n.get("PROGRESSTIME"))
                itemno = n.get("ITEMNO") or ""
                focus_label = (n.get("FOCUS") or "").strip()
                action = (n.get("ACTION") or "").strip()
                resp = (n.get("RESPONSE") or "").strip()
                parts.append(
                    f"<div class='note'><div class='note-date'>"
                    f"{esc(d)} {esc(t)} — item {esc(itemno)}</div>"
                )
                if focus_label:
                    parts.append(
                        f"<pre lang='th'><b>focus:</b> {esc(focus_label)}</pre>"
                    )
                if action:
                    parts.append(f"<pre lang='th'><b>D/A:</b>\n{esc(action)}</pre>")
                if resp:
                    parts.append(f"<pre lang='th'><b>R:</b>\n{esc(resp)}</pre>")
                if not (focus_label or action or resp):
                    parts.append("<p class='empty'>(empty row)</p>")
                parts.append("</div>")
        else:
            parts.append("<p class='empty'>(none)</p>")
        parts.append("</details>")

        parts.append("</section>")
        case_html_parts.append("\n".join(parts))

    summary_html = render_summary_table(
        summary_rows, show_msbos=MSBOS_RESERVATION_PILOT_ENABLED
    )
    if MSBOS_RESERVATION_PILOT_ENABLED:
        returned_counts = msbos_counts["RETURNED_NOT_TRANSFUSED"]
        exempt_counts = msbos_counts["PERIOP_TRANSFUSION_EXEMPT"]
        msbos_counts_html = (
            "<div class='msbos-counts'>"
            f"{_msbos_count_breakdown('Returned', returned_counts)}<br>"
            f"{_msbos_count_breakdown('Peri-op exempt', exempt_counts)}"
            "</div>"
        )
        # Post-flip verdict classes (#201): a declared row MSBOS reclassified
        # keeps its annotation, so its bucket tallies surface here too. Only
        # rendered when non-empty (a picker-off run leaves both at zero and
        # the block absent, keeping the legacy layout).
        over_counts = msbos_counts["PREOP_OVER_RESERVATION"]
        review_counts = msbos_counts["NEEDS_REVIEW"]
        if over_counts["denominator"] or review_counts["denominator"]:
            msbos_counts_html = msbos_counts_html.replace(
                "</div>",
                "<br>"
                f"{_msbos_count_breakdown('Over-reserved', over_counts)}<br>"
                f"{_msbos_count_breakdown('MSBOS review', review_counts)}"
                "</div>",
                1,
            )
    else:
        msbos_counts_html = ""

    css = """
    :root {
        --s-bg:           oklch(99% 0.005 28);
        --s-bg-raised:    oklch(97% 0.008 28);
        --s-bg-muted:     oklch(94% 0.01  28);
        --s-border:       oklch(90% 0.012 28);
        --s-border-strong:oklch(80% 0.02  28);
        --s-ink:          oklch(22% 0.015 28);
        --s-muted:        oklch(48% 0.02  28);
        --accent:         oklch(43% 0.18  28);
        --ok-bg:   oklch(94% 0.04 150); --ok-fg:   oklch(34% 0.12 150);
        --warn-bg: oklch(94% 0.05  70); --warn-fg:  oklch(40% 0.14  70);
        --err-bg:  oklch(94% 0.05  28); --err-fg:   oklch(38% 0.16  28);
        --neu-bg:  oklch(92% 0.005 28); --neu-fg:   oklch(35% 0.01  28);
        --info-bg: oklch(94% 0.04 280); --info-fg:  oklch(38% 0.14 280);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI',
                     'Noto Sans Thai', sans-serif;
        line-height: 1.5;
        color: var(--s-ink);
        background: var(--s-bg);
    }
    body { max-width: 1200px; margin: 0 auto; padding: 20px; }
    h1 { border-bottom: 2px solid var(--accent); padding-bottom: 8px;
         font-size: 1.953rem; font-weight: 700; }
    h2 { font-size: 1.563rem; font-weight: 700;
         border-bottom: 2px solid var(--s-border-strong);
         padding-bottom: 6px; margin-top: 40px; }
    h2#summary { margin-top: 0; }
    h3 { font-size: 1.25rem; font-weight: 600; color: var(--s-ink); margin-top: 18px; }
    h4 { font-size: 1rem; font-weight: 600; margin-top: 12px; }
    h5 { font-size: 0.875rem; font-weight: 600; margin-top: 10px; color: var(--s-muted); }
    .case { margin-bottom: 60px; scroll-margin-top: 60px; }
    .case > h3 { border-bottom: 1px solid var(--s-border); padding-bottom: 4px; margin-top: 32px; }
    .meta { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 18px;
            background: var(--s-bg-raised); padding: 12px; font-size: 0.875rem;
            border: 1px solid var(--s-border); border-radius: 4px; }
    .verdict { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin: 16px 0; }
    .vbox { padding: 14px; border: 1px solid var(--s-border); border-radius: 6px; }
    .vbox.det { background: var(--s-bg-raised); }
    .vbox.llm { background: var(--info-bg); border-color: oklch(80% 0.08 280); }
    .cls { font-size: 1rem; font-weight: 700; padding: 5px 12px; border-radius: 20px;
           display: inline-block; margin-bottom: 8px; letter-spacing: 0.02em; }
    .cls-appropriate { background: var(--ok-bg);   color: var(--ok-fg); }
    .cls-inappropriate,
    .cls-potentially_inappropriate { background: var(--err-bg);  color: var(--err-fg); }
    .cls-needs_review { background: var(--warn-bg); color: var(--warn-fg); }
    .cls-insufficient_evidence { background: var(--neu-bg);  color: var(--neu-fg); }
    .cls-preop_reservation_unconfirmed { background: var(--info-bg); color: var(--info-fg); }
    .cls-excluded { background: var(--info-bg); color: var(--info-fg); }
    .cls-preop_over_reservation { background: var(--err-bg); color: var(--err-fg); }
    .conf { font-size: 0.75rem; font-weight: 400; color: var(--s-muted); }
    .rationale { font-size: 0.8125rem; color: var(--s-muted); margin-top: 4px; }
    table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 0.8125rem; }
    .table-scroll { overflow-x: auto; }
    th, td { border: 1px solid var(--s-border); padding: 4px 8px; text-align: left;
             vertical-align: top; }
    th { background: var(--s-bg-muted); color: var(--s-ink); font-weight: 600; }
    td code { font-size: 0.75rem;
              font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace; }
    table.ind td { font-size: 0.8125rem; }
    table.ind td:nth-child(3), table.ind td:nth-child(4) {
        font-size: 0.75rem; color: var(--s-muted);
        font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace; }

    details { margin: 8px 0; border: 1px solid var(--s-border); border-radius: 4px;
              padding: 6px 12px; background: var(--s-bg-raised); }
    details > summary { cursor: pointer; font-weight: 600; padding: 4px 0;
                        list-style: disclosure-closed; }
    details[open] > summary { list-style: disclosure-open; }
    .note { border: 1px solid var(--s-border); border-radius: 4px;
            padding: 6px 10px; margin: 6px 0; background: var(--s-bg-raised); }
    .note-date { font-weight: 700; color: var(--s-muted); font-size: 0.75rem; }
    pre { white-space: pre-wrap; word-wrap: break-word; font-size: 0.8125rem; line-height: 1.6;
          max-width: 80ch; background: var(--s-bg-muted); padding: 8px; margin: 4px 0;
          border: 1px solid var(--s-border); border-radius: 4px;
          font-family: 'JetBrains Mono', ui-monospace, 'SF Mono', Menlo, Consolas, monospace; }
    .note pre { font-family: 'Noto Sans Thai', 'Inter', -apple-system, BlinkMacSystemFont,
                'Segoe UI', sans-serif; line-height: 1.65; max-width: none; }
    .empty { color: var(--s-muted); font-style: italic; }
    .reasoning { font-size: 0.8125rem; line-height: 1.65; }
    .reasoning p { margin: 6px 0; }
    .reasoning-h { font-weight: 700; color: var(--s-ink); margin: 12px 0 4px; }
    .reasoning-list, .reasoning-bullets { margin: 6px 0; padding-left: 24px; }
    .reasoning-list li, .reasoning-bullets li { margin: 4px 0; }
    ul li { margin: 2px 0; }
    .legend { display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
              margin: 10px 0 18px; font-size: 0.8125rem; }
    .legend .cls { margin-bottom: 0; font-size: 0.8125rem; padding: 3px 10px; }
    table .cls { margin-bottom: 0; }
    @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
    .nav-flag { font-size: 0.75rem; color: var(--warn-fg); }
    .nav-flag--major { color: var(--err-fg); }
    tr.verdict-mismatch > td { background: oklch(98% 0.015 70); }
    /* Case header */
    .case-hd { display: flex; align-items: baseline; gap: 14px; }
    .case-hd h3 { margin: 0; }
    .case.is-reviewed { opacity: 0.6; }
    .case.is-reviewed .case-hd h3::after {
        content: ' ✓'; color: var(--ok-fg); font-size: 0.875rem; font-weight: 400; }
    /* Source data header */
    .src-hd { display: flex; align-items: center; gap: 10px; margin-top: 16px; }
    .src-hd h4 { margin: 0; }
    /* Buttons */
    button { font: inherit; cursor: pointer; border: 1px solid var(--s-border-strong);
             border-radius: 4px; padding: 3px 10px; background: var(--s-bg-raised);
             color: var(--s-ink); font-size: 0.8125rem; transition: background 0.12s; }
    button:hover { background: var(--s-bg-muted); }
    button:focus-visible, a:focus-visible, input:focus-visible {
        outline: 2px solid var(--accent); outline-offset: 2px; }
    .mark-reviewed { font-size: 0.75rem; padding: 2px 8px; }
    .case.is-reviewed .mark-reviewed { background: var(--ok-bg); color: var(--ok-fg);
                                        border-color: oklch(70% 0.08 150); }
    .expand-toggle { font-size: 0.75rem; padding: 2px 8px; }
    /* Abbr tooltips on code slugs */
    abbr[title] { text-decoration: underline dotted var(--s-muted);
                  cursor: help; text-underline-offset: 2px; }
    /* Nav redesign for 100-case scale */
    nav { position: sticky; top: 0; z-index: 10; background: var(--s-bg-raised);
          padding: 8px 14px; border: 1px solid var(--s-border);
          border-bottom: 2px solid var(--s-border-strong); margin-bottom: 20px;
          box-shadow: 0 2px 6px oklch(0% 0 0 / 0.04); }
    .nav-controls { display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
                    font-size: 0.875rem; }
    .nav-home { font-weight: 700; color: var(--accent); text-decoration: none; }
    .nav-home:hover { text-decoration: underline; }
    .nav-controls label { display: flex; align-items: center; gap: 5px; cursor: pointer; }
    .nav-controls input[type=number] { width: 4.5rem; padding: 2px 6px; font: inherit;
        font-size: 0.8125rem; border: 1px solid var(--s-border-strong); border-radius: 4px;
        background: var(--s-bg); color: var(--s-ink); }
    #reviewed-counter { color: var(--s-muted); font-size: 0.8125rem; margin-left: auto; }
    details.nav-cases { border: none; padding: 0; margin: 0; background: none; }
    details.nav-cases > summary { font-size: 0.8125rem; color: var(--s-muted);
        cursor: pointer; padding: 2px 0; margin-top: 4px; }
    .nav-links { display: flex; flex-wrap: wrap; gap: 4px 10px; padding: 6px 0 2px; }
    .nav-links a { color: var(--accent); text-decoration: none; font-size: 0.8125rem; }
    .nav-links a:hover { text-decoration: underline; }
    .nav-links a.is-reviewed { color: var(--ok-fg); }
    .nav-links a.is-hidden { display: none; }
    .nav-links a.is-active { font-weight: 700; color: var(--s-ink); }
    #filter-status { font-size: 0.75rem; color: var(--warn-fg); }
    /* Glossary */
    .glossary-body { column-count: 2; column-gap: 24px; font-size: 0.8125rem; }
    .glossary-body dt { font-weight: 600; margin-top: 8px; }
    .glossary-body dd { margin-left: 0; color: var(--s-muted); }
    @media (max-width: 800px) { .glossary-body { column-count: 1; } }
    /* Keyboard shortcut hint */
    .kbd-hint { font-size: 0.75rem; color: var(--s-muted); margin-bottom: 8px; }
    .kbd-dismiss { font-size: 0.75rem; padding: 1px 6px; margin-left: 8px; }
    kbd { font-size: 0.75rem; background: var(--s-bg-muted); border: 1px solid var(--s-border-strong);
          border-radius: 3px; padding: 1px 5px; font-family: inherit; }
    @media print { nav { display: none; } details { display: block; }
                   details > summary { display: none; } button { display: none; } }
    """
    n_cases = len(manifest_rows)
    _FLAG_TITLE = {
        " <b class='nav-flag nav-flag--major'>[!!]</b>": " <b class='nav-flag nav-flag--major' title='Major mismatch: Appropriate vs Potentially inappropriate'>[!!]</b>",
        " <b class='nav-flag'>[!]</b>": " <b class='nav-flag' title='Verdict mismatch between deterministic and LLM classifiers'>[!]</b>",
    }
    nav_links_items = "".join(
        f"<a href='#case-{i + 1}' data-case='{i + 1}'>"
        f"#{i + 1} {m['REQNO']}{_FLAG_TITLE.get(tag, tag)}</a>"
        for i, (m, tag) in enumerate(zip(manifest_rows, case_mismatch_tags))
    )
    n_mismatches = sum(1 for t in case_mismatch_tags if t)
    returns_legend_html = (
        '  <span class="cls cls-returned_not_transfused">'
        "Returned — not transfused (excluded)</span>\n"
        '  <span class="cls cls-periop_transfusion_exempt">'
        "Peri-op transfusion — exempt (excluded)</span>\n"
        if RETURNS_LEDGER_ENABLED
        else ""
    )
    returns_pill_css = (
        "\n    .cls-returned_not_transfused,\n"
        "    .cls-periop_transfusion_exempt "
        "{ background: var(--neu-bg); color: var(--neu-fg); }\n"
        if RETURNS_LEDGER_ENABLED
        else ""
    )
    msbos_annotation_css = (
        "\n    .cls-msbos-warn { background: var(--warn-bg); color: var(--warn-fg); }\n"
        "    .cls-msbos-ok   { background: var(--ok-bg);   color: var(--ok-fg); }\n"
        "    .msbos-counts { font-size: 0.8125rem; color: var(--s-muted); "
        "margin: 8px 0 0; }\n"
        if MSBOS_RESERVATION_PILOT_ENABLED
        else ""
    )
    returns_glossary_html = (
        "<dt>RETURNED_NOT_TRANSFUSED</dt><dd>All dispensed units were returned; "
        "excluded from scoring and review.</dd>\n"
        "<dt>PERIOP_TRANSFUSION_EXEMPT</dt><dd>Declared pre-op order (USETYPE "
        "M/G or T/S); exempt from transfusion judgment but still screened for "
        "reservation appropriateness. Legacy periop_transfusion_exempt "
        "rows mean confirmed transfusion in a surgical/procedural context and "
        "retain that legacy interpretation.</dd>\n"
        if RETURNS_LEDGER_ENABLED
        else ""
    )
    msbos_glossary_html = (
        "<dt>operation_unresolved</dt><dd>Conflicting MSBOS operation code "
        "could not be uniquely resolved from the windowed clinical notes — "
        "escalated to human review.</dd>\n"
        "<dt>above</dt><dd>reserved units exceeded the elective MSBOS tariff for "
        "the resolved planned procedure code (RBC), OR the pre-op platelet count "
        "exceeded the clinician-signed cutoff for the resolved procedure category "
        "(platelet: major-non-neuraxial 80k/µL; neuraxial and cardiac-CPB 100k/µL). "
        "On factual returns rows this annotation is INFORMATIONAL. On DECLARED "
        "pre-op rows (USETYPE surgery/type-screen) MSBOS screening CAN change the "
        "classification: an over-reservation reclassifies the row to "
        "PREOP_OVER_RESERVATION and an unresolved, ambiguous, or bridge-gated "
        "pick to NEEDS_REVIEW. It still does not account for anticipated "
        "hemorrhage, case cancellation, or emergency status (the schedule data "
        "cannot see these), and it judges ordering QUANTITY only, "
        "never the transfusion decision (the anaesthesiologist's call — the reason "
        "the bucket is exempt).</dd>\n"
        if MSBOS_RESERVATION_PILOT_ENABLED
        else ""
    )
    picker_glossary_html = (
        "<dt>preop_reservation_bridge_disagreement</dt><dd>The billing-code "
        "bridge's First-Choice ICD-9 and the human reviewer's selected code "
        "disagree and at least one of them carries an MSBOS recommendation — "
        "the procedure identity is contested, so the row routes to human "
        "review instead of any automatic verdict.</dd>\n"
        "<dt>preop_over_reservation_bridge_unconfirmed</dt><dd>The reservation "
        "exceeds the MSBOS recommendation for the bridge-resolved procedure, "
        "but the mapping lacks the confidence (score >= 0.95) plus human "
        "agreement required for a hard PREOP_OVER_RESERVATION — routed to "
        "review, never asserted.</dd>\n"
        "<dt>over ceiling</dt><dd>The planned operation was ambiguous (a "
        "near-simultaneous cluster or one code with several tariffs), but EVERY "
        "candidate resolves into MSBOS, so the reservation is judged against the "
        "MOST PERMISSIVE tariff in the set (the ceiling). Reserved units exceed "
        "even that ceiling, so the row is over under every reading — a hard "
        "PREOP_OVER_RESERVATION when the set is exact/gate-confirmed, otherwise "
        "review.</dd>\n"
        "<dt>within ceiling</dt><dd>An ambiguous-but-all-eligible reservation "
        "that is within the most permissive tariff (the ceiling). It stays "
        "declared-exempt and annotated; it may still be over under the set's "
        "least-permissive member (a shadow-over surfaced as its own count).</dd>\n"
        if MSBOS_RESERVATION_PILOT_ENABLED and MSBOS_PLANNED_OP_PICKER_V2_PILOT_ENABLED
        else ""
    )
    head = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KCMH RBC order appropriateness audit — human review</title>
<style>{css}{returns_pill_css}{msbos_annotation_css}</style></head><body>
<h1>KCMH RBC Order Appropriateness Audit — Human Review</h1>
<p>{n_cases} RBC orders. Deterministic: local rule-based verdict.
LLM: Anthropic Batch classification on structured evidence only.
{n_mismatches} verdict mismatches flagged.</p>
<p class='kbd-hint' id='kbd-hint'>Keyboard: <kbd>j</kbd> next · <kbd>k</kbd> prev · <kbd>e</kbd> mark reviewed · <kbd>f</kbd> filter · <kbd>E</kbd> expand all · <kbd>x</kbd> expand case <button class='kbd-dismiss' onclick='dismissKbdHint()'>Dismiss</button></p>
<nav>
  <div class='nav-controls'>
    <a href='#summary' class='nav-home'>Summary</a>
    <label><input type='checkbox' id='filter-mismatches' onchange='filterMismatches(this)'> Mismatches only</label>
    <span id='filter-status'></span>
    <label>Jump: <input type='number' id='jump-input' min='1' max='{n_cases}' placeholder='#'
      onchange='jumpToCase(this.value)' oninput='if(this.value>={n_cases})this.value={n_cases}'></label>
    <span id='reviewed-counter'>Reviewed: 0 / {n_cases}</span>
    <button id='reset-reviewed' onclick='resetReviewed()' style='display:none'>Reset all</button>
  </div>
  <details class='nav-cases'><summary>All cases ({n_cases}) — click to browse</summary>
  <div class='nav-links' id='nav-links'>{nav_links_items}</div>
  </details>
</nav>
<h2 id='summary'>Summary</h2>
<div class="legend">
  <b>Key:</b>
  <span class="cls cls-appropriate">Appropriate</span>
  <span class="cls cls-needs_review">Needs review</span>
  <span class="cls cls-potentially_inappropriate">Potentially inappropriate</span>
  <span class="cls cls-preop_reservation_unconfirmed">Administration unconfirmed (pre-op reservation)</span>
{returns_legend_html}  <span class="cls cls-insufficient_evidence">Insufficient evidence</span>
  <span class="cls cls-excluded">Excluded</span>
</div>
<div class='table-scroll'>{summary_html}</div>{msbos_counts_html}
<details style='margin:16px 0;'><summary><b>Glossary — verdict classes and classifier codes</b></summary>
<div class='glossary-body'>
<dl>
<dt>APPROPRIATE</dt><dd>Order meets evidence-based indications (Hb below threshold or qualifying bypass).</dd>
<dt>POTENTIALLY_INAPPROPRIATE</dt><dd>Hb above threshold and no qualifying bypass was identified; warrants clinician review.</dd>
<dt>NEEDS_REVIEW</dt><dd>Classifier could not confidently classify; manual review required (e.g. haemodilution, single borderline Hb).</dd>
<dt>INSUFFICIENT_EVIDENCE</dt><dd>LLM could not find enough structured evidence to classify.</dd>
{returns_glossary_html}<dt>EXCLUDED</dt><dd>Case excluded from audit scope (e.g. paediatric, non-RBC product).</dd>
<dt style='margin-top:14px;font-style:italic;'>Rationale codes</dt><dd></dd>
<dt>hb_lt_7_universal</dt><dd>Hb &lt; 7.0 g/dL — below the universal threshold; no bypass required.</dd>
<dt>hb_lt_threshold</dt><dd>Hb below the cohort-specific threshold.</dd>
<dt>hb_7_to_10</dt><dd>Hb 7.0–10.0 g/dL — gray zone; requires a documented clinical indication.</dd>
<dt>hb_ge_10</dt><dd>Hb ≥ 10.0 g/dL — above threshold; no qualifying bypass was found.</dd>
<dt>hb_missing</dt><dd>No Hb result in the 7-day pre-anchor window.</dd>
<dt>cohort_unknown</dt><dd>Patient cohort could not be determined.</dd>
<dt>cohort_non_threshold</dt><dd>Cohort has no fixed Hb threshold (e.g. active haematological malignancy).</dd>
<dt>bypass_mtp</dt><dd>Massive Transfusion Protocol cohort — auto-classified APPROPRIATE.</dd>
<dt>bypass_peri_procedural</dt><dd>[legacy/flag-off] Procedure within 6 h before order — peri-procedural bypass.</dd>
<dt>preop_defer_llm</dt><dd>[legacy/flag-off] Upcoming procedure within 72 h — routed to LLM review, not auto-cleared (a reservation is not an indication). Minor procedures (perm cath, tracheostomy, lumbar puncture, thoracocentesis, paracentesis, arthrocentesis, arterial/central line) are excluded from the peri-op signal.</dd>
<dt>preop_defer_llm_declared</dt><dd>[legacy/flag-off] Declared surgical intent (BDVSTDT.USETYPE = surgery/type-screen) routed the order to LLM review with no structured op row; a declaration is not a transfusion indication.</dd>
<dt>bypass_pre_op_crossmatch</dt><dd>[legacy] Upcoming procedure within 72 h — auto-classified APPROPRIATE. Superseded by preop_defer_llm; retained for historical reports.</dd>
<dt>bypass_delta_hb</dt><dd>Rapid Hb drop (delta-Hb trigger) fired — auto-classified APPROPRIATE.</dd>
<dt>bypass_hemodilution</dt><dd>Haemodilution pattern flagged — Hb unreliable; sent to NEEDS_REVIEW.</dd>
<dt>single_low_hb_no_trend</dt><dd>Single Hb below threshold with no supporting trend — NEEDS_REVIEW.</dd>
<dt style='margin-top:14px;font-style:italic;'>Bypass codes</dt><dd></dd>
<dt>none</dt><dd>No bypass applied; classification is purely Hb-tier based.</dd>
<dt>mtp</dt><dd>Massive Transfusion Protocol — volume-based bypass, Hb thresholds suspended.</dd>
<dt>peri_procedural_6h</dt><dd>Peri-procedural: active procedure within 6 h before order.</dd>
<dt>pre_op_crossmatch</dt><dd>[legacy] Pre-operative crossmatch bypass — superseded; pre-op orders now route to LLM review with bypass=none.</dd>
<dt>delta_hb</dt><dd>Delta-Hb: ≥ 2 g/dL drop in 24 h pre-anchor window.</dd>
<dt>hemodilution_flagged</dt><dd>Haemodilution suspected: Hb rise after IV fluid consistent with dilution artifact.</dd>
<dt style='margin-top:14px;font-style:italic;'>LLM provenance codes (review_reason)</dt><dd></dd>
<dt>model_verdict</dt><dd>No guardrail action — the final classification is the model's own verdict.</dd>
<dt>llm_overclear_asserted_inappropriate</dt><dd>Guardrail-asserted INAPPROPRIATE: the LLM cleared a withheld gray-zone/high-Hb order with no genuine hard indication, so the over-clear guardrail asserted the final verdict (not the model's own label); the human-review flag is cleared.</dd>
<dt>llm_native_review_asserted_inappropriate</dt><dd>Guardrail-converted INAPPROPRIATE: the model itself returned NEEDS_REVIEW with reasoning but no hard signal and no qualified bleed, so the verdict was converted to INAPPROPRIATE; the human-review flag is cleared.</dd>
<dt>llm_overclear_suspect</dt><dd>Over-clear floored to NEEDS_REVIEW: historical rows from before the assert guardrail, plus the live paths where asserting is unsafe — a shape-drifted tool payload (missing or garbled indications/negative_evidence), or a grounded high-confidence citation of a hard indication the structured system cannot dismiss (ACS; documented shock/pressors the vitals snapshot cannot see; a structurally-true sub-floor Hb withheld as unreliable).</dd>
<dt>periop_signal_contradiction</dt><dd>Peri-operative hard signal contradicts the LLM verdict — kept NEEDS_REVIEW for a human (the intended residual).</dd>
{msbos_glossary_html}{picker_glossary_html}<dt>hallucination_suspect</dt><dd>Quote verifier rejected every attempt — the cited quotes did not ground in the evidence bundle.</dd>
<dt>empty_reasoning</dt><dd>Final verdict carried empty reasoning — floored to NEEDS_REVIEW (a verdict with no rationale is never asserted).</dd>
<dt>platelet_llm_overclear_suspect</dt><dd>Platelet-leg over-clear floored to NEEDS_REVIEW (platelet guardrail; the RBC assert path does not apply to platelets).</dd>
<dt>malformed_json</dt><dd>Parse failure — the response was not valid JSON.</dd>
<dt>schema_mismatch</dt><dd>Parse failure — the tool payload did not match the expected schema.</dd>
<dt>classification_out_of_set</dt><dd>Parse failure — the classification label is outside the allowed vocabulary.</dd>
<dt>empty_response</dt><dd>Parse failure — the model returned an empty response.</dd>
<dt>tool_use_missing</dt><dd>Parse failure — no tool-use block was found in the response.</dd>
</dl>
</div>
</details>
"""
    body = "\n".join(case_html_parts)
    foot = "</body></html>"
    script = """<script>
(function(){
  /* ── Mark-reviewed with localStorage persistence ── */
  var STORE_KEY = 'bba_reviewed';
  function loadReviewed() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY) || '[]'); }
    catch(e) { return []; }
  }
  function saveReviewed(arr) {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(arr)); } catch(e){}
  }
  function updateCounter() {
    var total = document.querySelectorAll('.case').length;
    var done  = document.querySelectorAll('.case.is-reviewed').length;
    var el = document.getElementById('reviewed-counter');
    if (el) el.textContent = 'Reviewed: ' + done + ' / ' + total;
  }
  function applyReviewedState(caseNum, reviewed) {
    var sec = document.getElementById('case-' + caseNum);
    var navLink = document.querySelector('#nav-links a[data-case="' + caseNum + '"]');
    if (sec) sec.classList.toggle('is-reviewed', reviewed);
    if (navLink) navLink.classList.toggle('is-reviewed', reviewed);
    var btn = sec ? sec.querySelector('.mark-reviewed') : null;
    if (btn) btn.textContent = reviewed ? 'Reviewed' : 'Mark reviewed';
  }
  /* Restore state on load */
  loadReviewed().forEach(function(n){ applyReviewedState(n, true); });
  updateCounter();
  /* Hide kbd-hint if previously dismissed */
  try { if (localStorage.getItem('bba_kbd_dismissed')) {
    var kh = document.getElementById('kbd-hint');
    if (kh) kh.style.display = 'none';
  }} catch(e){}
  /* Show/hide reset button based on reviewed count */
  function _syncResetBtn() {
    var btn = document.getElementById('reset-reviewed');
    if (btn) btn.style.display = loadReviewed().length > 0 ? '' : 'none';
  }
  _syncResetBtn();

  window.toggleReviewed = function(btn) {
    var caseNum = parseInt(btn.dataset.case, 10);
    var arr = loadReviewed();
    var idx = arr.indexOf(caseNum);
    if (idx === -1) { arr.push(caseNum); }
    else            { arr.splice(idx, 1); }
    saveReviewed(arr);
    applyReviewedState(caseNum, idx === -1);
    updateCounter();
    _syncResetBtn();
  };
  window.resetReviewed = function() {
    saveReviewed([]);
    document.querySelectorAll('.case').forEach(function(sec) {
      var n = parseInt(sec.id.replace('case-', ''), 10);
      applyReviewedState(n, false);
    });
    updateCounter();
    _syncResetBtn();
  };
  window.dismissKbdHint = function() {
    var kh = document.getElementById('kbd-hint');
    if (kh) kh.style.display = 'none';
    try { localStorage.setItem('bba_kbd_dismissed', '1'); } catch(e){}
  };

  /* ── Source data: expand / collapse all <details> in a case ── */
  window.toggleSrcDetails = function(btn) {
    var sec = btn.closest('.case');
    if (!sec) return;
    var dets = sec.querySelectorAll('details');
    var anyOpen = Array.prototype.some.call(dets, function(d){ return d.open; });
    dets.forEach(function(d){ d.open = !anyOpen; });
    btn.textContent = anyOpen ? 'Expand all' : 'Collapse all';
  };

  /* ── Jump to case ── */
  window.jumpToCase = function(val) {
    var n = parseInt(val, 10);
    if (!n) return;
    var sec = document.getElementById('case-' + n);
    if (sec) sec.scrollIntoView({behavior:'smooth', block:'start'});
  };

  /* ── Filter: mismatches only ── */
  window.filterMismatches = function(cb) {
    var cases = document.querySelectorAll('.case');
    var hidden = 0;
    cases.forEach(function(sec) {
      var hasMismatch = sec.querySelector('.nav-flag') !== null;
      var hide = cb.checked && !hasMismatch;
      sec.style.display = hide ? 'none' : '';
      if (hide) hidden++;
    });
    var navLinks = document.querySelectorAll('#nav-links a');
    navLinks.forEach(function(a) {
      var hasMismatch = a.querySelector('.nav-flag') !== null;
      a.classList.toggle('is-hidden', cb.checked && !hasMismatch);
    });
    var st = document.getElementById('filter-status');
    if (st) st.textContent = (cb.checked && hidden > 0) ? hidden + ' cases hidden' : '';
  };

  /* ── Keyboard navigation: j/k/e/f/E/x ── */
  var caseEls = Array.prototype.slice.call(document.querySelectorAll('.case'));
  var activeIdx = -1;
  function findActiveByScroll() {
    var mid = window.innerHeight / 2;
    for (var i = caseEls.length - 1; i >= 0; i--) {
      var r = caseEls[i].getBoundingClientRect();
      if (r.top <= mid) return i;
    }
    return 0;
  }
  /* ── IntersectionObserver scrollspy ── */
  var _activeNavLink = null;
  if ('IntersectionObserver' in window) {
    var obs = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        var caseNum = entry.target.id.replace('case-', '');
        var link = document.querySelector('#nav-links a[data-case="' + caseNum + '"]');
        if (!link) return;
        if (entry.isIntersecting) {
          if (_activeNavLink) _activeNavLink.classList.remove('is-active');
          link.classList.add('is-active');
          _activeNavLink = link;
          activeIdx = parseInt(caseNum, 10) - 1;
        }
      });
    }, {threshold: 0.15});
    caseEls.forEach(function(sec) { obs.observe(sec); });
  }
  document.addEventListener('keydown', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'j' || e.key === 'k') {
      if (activeIdx < 0) activeIdx = findActiveByScroll();
      activeIdx = e.key === 'j'
        ? Math.min(activeIdx + 1, caseEls.length - 1)
        : Math.max(activeIdx - 1, 0);
      caseEls[activeIdx].scrollIntoView({behavior:'smooth', block:'start'});
    } else if (e.key === 'e') {
      if (activeIdx < 0) activeIdx = findActiveByScroll();
      var sec = caseEls[activeIdx];
      if (!sec) return;
      var btn = sec.querySelector('.mark-reviewed');
      if (btn) window.toggleReviewed(btn);
    } else if (e.key === 'x') {
      if (activeIdx < 0) activeIdx = findActiveByScroll();
      var sec2 = caseEls[activeIdx];
      if (!sec2) return;
      var toggleBtn = sec2.querySelector('.expand-toggle');
      if (toggleBtn) window.toggleSrcDetails(toggleBtn);
    } else if (e.key === 'f') {
      var cb = document.getElementById('filter-mismatches');
      if (cb) { cb.checked = !cb.checked; window.filterMismatches(cb); }
    } else if (e.key === 'E') {
      var allDets = document.querySelectorAll('.case details');
      var anyOpen = Array.prototype.some.call(allDets, function(d){ return d.open; });
      allDets.forEach(function(d){ d.open = !anyOpen; });
      document.querySelectorAll('.expand-toggle').forEach(function(btn){
        btn.textContent = anyOpen ? 'Expand all' : 'Collapse all';
      });
    }
  });
})();
</script>"""
    OUT.write_text(head + body + script + foot, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
