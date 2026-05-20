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
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))
BUNDLE = WORK / "bundle"
LLM_REPORT = WORK / "llm_report.json"
DET_REPORT = WORK / "report.csv"
MANIFEST = WORK / "sample_manifest.csv"
OUT = WORK / "review.html"

_DEFAULT_ICD10 = (
    Path(__file__).resolve().parents[2].parent
    / "Bloodbank"
    / "data"
    / "raw"
    / "ICD10.csv"
)
ICD10_DICT_CSV = Path(os.environ.get("BBA_PILOT_ICD10_CSV", str(_DEFAULT_ICD10)))

# Per-source windowing rules around the transfusion anchor.
WINDOW_PROC_DAYS = 7
WINDOW_HB_DAYS = 7
WINDOW_NOTES_DAYS_BEFORE = 1
WINDOW_NOTES_DAYS_AFTER = 0
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

# Diagnosis-type precedence — best-keeps wins on duplicate ICD-10 rows.
_TYPE_RANK = {
    "Principal Diagnosis": 0,
    "Complication": 1,
    "Comorbidity": 2,
}

csv.field_size_limit(sys.maxsize)


def esc(s: Any) -> str:
    return html.escape(str(s)) if s is not None else ""


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


def parse_hosxp_date(raw: Any) -> date | None:
    if raw is None:
        return None
    head = str(raw).split(" ", 1)[0]
    try:
        return date.fromisoformat(head)
    except ValueError:
        return None


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


def render_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return "<p class='empty'>(no rows)</p>"
    head = "".join(f"<th>{esc(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(r.get(c, ''))}</td>" for c in cols) + "</tr>"
        for r in rows
    )
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


def _short(s: str, n: int = 16) -> str:
    return s[:n] + "..." if len(s) > n else s


def _field(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return ""


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

    bdvst = _read("BDVST.csv")
    bdvstdt = _read("BDVSTDT.csv")
    diag = _read("Diagnosis.csv")
    lab = _read("Lab.csv")
    med = _read("Med.csv")
    iptsumoprt = _read("IPTSUMOPRT.csv")
    icd9 = _read("ICD9CM.csv")
    bdvstst_dict_rows = _read("BDVSTST.csv")
    progress = _read("IPDADMPROGRESS.csv")
    focus = _read("IPDNRFOCUSDT.csv")
    incpt = _read_optional("INCPT.csv")

    icd9_dict = {
        (r.get("Icd9cm") or "").strip().replace(".", ""): (r.get("Name") or "").strip()
        for r in icd9
    }
    bdvstst_dict: dict[str, str] = {
        (r.get("BDVSTST") or "").strip(): (r.get("NAME") or "").strip()
        for r in bdvstst_dict_rows
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

    llm_report = (
        json.loads(LLM_REPORT.read_text(encoding="utf-8"))
        if LLM_REPORT.exists()
        else []
    )
    llm_by_reqno = {r["reqno"]: r for r in llm_report}

    manifest_rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    bdvst_by_reqno = {r["REQNO"]: r for r in bdvst}

    case_html_parts: list[str] = []
    summary_rows: list[dict[str, str]] = []

    for i, m in enumerate(manifest_rows, start=1):
        hn = m["HN"]
        reqno = m["REQNO"]
        an = m["AN"]
        bdv = bdvst_by_reqno.get(reqno, {})

        anchor_date = (bdv.get("REQDATE") or "").split(" ")[0]
        anchor_time = fmt_time(bdv.get("REQTIME"))
        anchor_d = parse_hosxp_date(bdv.get("REQDATE")) or parse_hosxp_date(
            bdv.get("BDVSTDATE")
        )
        notes_lo = (
            (anchor_d - timedelta(days=WINDOW_NOTES_DAYS_BEFORE)) if anchor_d else None
        )
        notes_hi = (
            (anchor_d + timedelta(days=WINDOW_NOTES_DAYS_AFTER)) if anchor_d else None
        )
        proc_lo = (anchor_d - timedelta(days=WINDOW_PROC_DAYS)) if anchor_d else None
        proc_hi = (anchor_d + timedelta(days=WINDOW_PROC_DAYS)) if anchor_d else None
        hb_lo = (anchor_d - timedelta(days=WINDOW_HB_DAYS)) if anchor_d else None

        def _in_notes_window(d: date | None) -> bool:
            return d is not None and notes_lo is not None and notes_lo <= d <= notes_hi

        def _in_proc_window(d: date | None) -> bool:
            return d is None or proc_lo is None or (proc_lo <= d <= proc_hi)

        def _in_hb_window(d: date | None) -> bool:
            return (
                d is not None
                and hb_lo is not None
                and anchor_d is not None
                and hb_lo <= d <= anchor_d
            )

        line_items = [r for r in bdvstdt if r.get("REQNO") == reqno]

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
            and _in_hb_window(parse_hosxp_date(r.get("LVSTDATE")))
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
        for r in iptsumoprt:
            if _field(r, "An", "AN") != an:
                continue
            pdate = parse_iptsumoprt_date(_field(r, "Indate", "INDATE"))
            if not _in_proc_window(pdate):
                continue
            code = _field(r, "Icd9cm", "ICD9CM").strip().replace(".", "")
            proc_rows.append(
                {
                    "Source": "IPTSUMOPRT",
                    "ICD9CM": code,
                    "Name": icd9_dict.get(code, ""),
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
            proc_rows.append(
                {
                    "Source": "INCPT",
                    "ICD9CM": "",
                    "Name": _field(r, "Incgrp → Name", "INCGRP → NAME").strip(),
                    "INDATE": _field(r, "Incdate", "INCDATE"),
                    "INTIME": fmt_time(_field(r, "Inctime", "INCTIME")),
                    "OUTDATE": "",
                    "OUTTIME": "",
                    "ORFLAG": "",
                    "INCOME": _field(r, "Income", "INCOME").strip(),
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

        det = det_by_reqno.get(reqno) or {}
        llm = llm_by_reqno.get(reqno)
        det_class = det.get("classification") or "excluded"
        # ``llm_final`` may be explicitly null when run_llm_leg.py persisted
        # a partial result (missing batch row); chaining ``.get`` on None
        # would crash, so coalesce twice.
        llm_final_obj = (llm or {}).get("llm_final") or {}
        if not llm:
            llm_final = "(deterministic-final)"
        elif not llm_final_obj:
            llm_final = "(missing)"
        else:
            llm_final = llm_final_obj.get("final_classification") or "—"

        summary_rows.append(
            {
                "#": str(i),
                "REQNO": reqno,
                "HN": _short(hn),
                "AN": _short(an),
                "Hb": det.get("hb_value_g_dl", "") or "—",
                "Cohort": det.get("cohort_label", "") or "—",
                "Threshold": det.get("cohort_threshold", "") or "—",
                "Deterministic": det_class or "EXCLUDED",
                "LLM": llm_final or "—",
            }
        )

        admission_date = (bdv.get("BDVSTDATE") or "").split(" ")[0] or "—"
        for d in diag_rows:
            d["Admission date"] = admission_date

        # ----- Section assembly -----
        parts: list[str] = [
            f"<section class='case' id='case-{i}'>",
            f"<h2>Case {i} — REQNO {esc(reqno)}</h2>",
            "<div class='meta'>",
            f"<div><b>HN:</b> <code>{esc(hn)}</code></div>",
            f"<div><b>AN:</b> <code>{esc(an)}</code></div>",
            f"<div><b>Order anchor:</b> {esc(anchor_date)} {esc(anchor_time)}</div>",
            f"<div><b>Products ordered:</b> {esc(det.get('products_ordered') or '—')}</div>",
            f"<div><b>Hb @ anchor:</b> {esc(det.get('hb_value_g_dl') or '—')} g/dL "
            f"({esc(det.get('hb_freshness') or '—')}, source {esc(det.get('hb_source') or '—')})</div>",
            f"<div><b>Cohort:</b> {esc(det.get('cohort_label') or '—')} "
            f"(threshold {esc(det.get('cohort_threshold') or 'n/a')})</div>",
            "</div>",
        ]

        # Verdict
        parts.append("<div class='verdict'>")
        parts.append("<div class='vbox det'>")
        parts.append("<h3>Deterministic verdict</h3>")
        parts.append(
            f"<div class='cls cls-{esc(det_class).lower()}'>"
            f"{esc(det_class or 'EXCLUDED')}</div>"
        )
        parts.append(
            f"<div class='rationale'>rationale: <code>"
            f"{esc(det.get('rationale') or '—')}</code></div>"
        )
        parts.append(
            f"<div class='rationale'>bypass: <code>"
            f"{esc(det.get('bypass_reason') or 'none')}</code></div>"
        )
        parts.append("</div>")

        parts.append("<div class='vbox llm'>")
        parts.append("<h3>LLM verdict</h3>")
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
                f"{esc(fc)} <span class='conf'>(conf {conf:.2f}; "
                f"{esc(model)})</span></div>"
            )
            parts.append("<h4>Indications</h4>")
            parts.append(render_indications(llm_block["indications"]))
            parts.append("<h4>Negative evidence</h4>")
            parts.append(render_negative(llm_block["negative_evidence"]))
            parts.append("<details><summary><b>Reasoning — English</b></summary>")
            parts.append(f"<p>{esc(llm_block['reasoning_en'])}</p>")
            parts.append("</details>")
            parts.append("<details><summary><b>Reasoning — ภาษาไทย</b></summary>")
            parts.append(f"<p>{esc(llm_block['reasoning_th'])}</p>")
            parts.append("</details>")
        elif llm:
            parts.append(
                "<p class='empty'>(LLM submission recorded but final "
                "verdict missing — batch row dropped or unparsable)</p>"
            )
        else:
            parts.append(
                "<p class='empty'>(deterministic-final; LLM leg not invoked)</p>"
            )
        parts.append("</div>")
        parts.append("</div>")

        # Source data
        parts.append("<h3>Source data (real rows)</h3>")

        order_icd10 = (bdv.get("ICD10") or "").strip()
        order_icd10_name = icd10_dict.get(order_icd10, "")
        order_icd10_disp = (
            f"{order_icd10} — {order_icd10_name}"
            if order_icd10 and order_icd10_name
            else order_icd10
        )

        parts.append("<details open><summary>Order — BDVST + BDVSTDT</summary>")
        parts.append(
            render_table(
                [
                    {
                        "REQNO": bdv.get("REQNO", ""),
                        "REQDATE": (bdv.get("REQDATE") or "").split(" ")[0],
                        "REQTIME": fmt_time(bdv.get("REQTIME")),
                        "BDVSTDATE": (bdv.get("BDVSTDATE") or "").split(" ")[0],
                        "BDVSTTIME": fmt_time(bdv.get("BDVSTTIME")),
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
                        "ITEMNO": r.get("ITEMNO", ""),
                        "UNITAMT": r.get("UNITAMT", ""),
                        "USEDATE": (r.get("USEDATE") or "").split(" ")[0],
                        "USETIME": fmt_time(r.get("USETIME")),
                    }
                    for r in line_items
                ],
                ["BDTYPE (product)", "ITEMNO", "UNITAMT", "USEDATE", "USETIME"],
            )
        )
        parts.append("</details>")

        parts.append(
            f"<details open><summary>Diagnoses ({len(diag_rows)} rows, "
            f"AN-scoped, all charted on admission {admission_date})</summary>"
        )
        parts.append(
            "<p class='empty'>Per-row charting date is not recoverable from "
            "the source bundle (HOSxP <code>V_DATE</code> is Excel-corrupted "
            "to <code>00:00.0</code> in every row). The Admission date column "
            "is the order's admission date as a temporal proxy.</p>"
        )
        parts.append(
            render_table(
                diag_rows,
                ["ICD10", "Description", "Type", "Admission date", "ICD10WHO"],
            )
        )
        parts.append("</details>")

        hb_window_str = (
            f"{hb_lo.isoformat()} … {anchor_date}"
            if hb_lo and anchor_date
            else "(no anchor)"
        )
        parts.append(
            f"<details open><summary>Hb history ({len(hb_rows)} rows, "
            f"7-day pre-order window {hb_window_str})</summary>"
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
        if anc_rows:
            parts.append(
                f"<details open><summary>ANC — Absolute Neutrophil Count "
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

        proc_window_str = (
            f"{proc_lo.isoformat()} … {proc_hi.isoformat()}"
            if proc_lo and proc_hi
            else "(no anchor)"
        )
        parts.append(
            f"<details open><summary>Procedures — IPTSUMOPRT + INCPT "
            f"({len(proc_rows)} rows, AN-scoped, ±1 week window "
            f"{proc_window_str})</summary>"
        )
        parts.append(
            render_table(
                proc_rows,
                [
                    "Source",
                    "ICD9CM",
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
            f"<details open><summary>Progress notes — IPDADMPROGRESS "
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
            f"<details open><summary>Nursing focus notes — IPDNRFOCUSDT "
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
                    parts.append(f"<pre><b>focus:</b> {esc(focus_label)}</pre>")
                if action:
                    parts.append(f"<pre><b>D/A:</b>\n{esc(action)}</pre>")
                if resp:
                    parts.append(f"<pre><b>R:</b>\n{esc(resp)}</pre>")
                if not (focus_label or action or resp):
                    parts.append("<p class='empty'>(empty row)</p>")
                parts.append("</div>")
        else:
            parts.append("<p class='empty'>(none)</p>")
        parts.append("</details>")

        parts.append("</section>")
        case_html_parts.append("\n".join(parts))

    summary_html = render_table(
        summary_rows,
        ["#", "REQNO", "HN", "AN", "Hb", "Cohort", "Threshold", "Deterministic", "LLM"],
    )

    css = """
    :root { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.45; color: #1a1a1a; }
    body { max-width: 1200px; margin: 0 auto; padding: 20px; }
    h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
    h2 { border-top: 4px solid #333; padding-top: 18px; margin-top: 36px;
         background: #f4f4f4; padding-left: 8px; }
    h3 { color: #2a2a2a; margin-top: 18px; }
    .case { margin-bottom: 60px; }
    .meta { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 18px;
            background: #fafafa; padding: 12px; border-left: 4px solid #888;
            font-size: 14px; }
    .verdict { display: grid; grid-template-columns: 1fr 1fr; gap: 18px;
               margin: 16px 0; }
    .vbox { padding: 14px; border: 1px solid #ccc; border-radius: 6px; }
    .vbox.det { background: #f8f8ff; }
    .vbox.llm { background: #fff8f0; }
    .cls { font-size: 18px; font-weight: 700; padding: 6px 10px; border-radius: 4px;
           display: inline-block; margin-bottom: 8px; }
    .cls-appropriate { background: #d8f4d8; color: #115511; }
    .cls-inappropriate, .cls-potentially_inappropriate { background: #f8d8d8; color: #771111; }
    .cls-needs_review { background: #fff4cc; color: #886600; }
    .cls-insufficient_evidence { background: #e0e0e0; color: #333; }
    .cls-excluded { background: #e8e8ff; color: #444; }
    .conf { font-size: 12px; font-weight: normal; }
    .rationale { font-size: 13px; color: #555; }
    table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 13px; }
    th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: left;
             vertical-align: top; }
    th { background: #eee; }
    td code { font-size: 12px; }
    table.ind td { font-size: 13px; }
    details { margin: 8px 0; border: 1px solid #e0e0e0; border-radius: 4px;
              padding: 6px 12px; }
    details > summary { cursor: pointer; font-weight: 600; padding: 4px 0; }
    .note { border-left: 3px solid #aaa; padding: 4px 10px; margin: 6px 0;
            background: #fafafa; }
    .note-date { font-weight: bold; color: #444; font-size: 12px; }
    pre { white-space: pre-wrap; word-wrap: break-word; font-size: 12px;
          background: #fff; padding: 6px; margin: 4px 0; border: 1px solid #eee; }
    .empty { color: #888; font-style: italic; }
    ul li { margin: 2px 0; }
    nav { background: #fafafa; padding: 12px; border: 1px solid #ddd;
          margin-bottom: 20px; }
    nav a { margin-right: 12px; }
    """
    nav_links = " · ".join(
        f"<a href='#case-{i + 1}'>#{i + 1} {m['REQNO']}</a>"
        for i, m in enumerate(manifest_rows)
    )
    head = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>KCMH RBC audit — human review</title>
<style>{css}</style></head><body>
<h1>KCMH RBC Transfusion Audit — Human Review</h1>
<p>Single bundle of source data + audit verdicts for the sampled RBC orders.
Deterministic verdict comes from the local pipeline composition; LLM
verdict comes from a live Anthropic Batch call on a structured-only
evidence payload (no free-text notes sent — those weren't through
thai-medical-deid).</p>
<nav><b>Jump to:</b> {nav_links}</nav>
<h2 style='border-top:none; background:none; padding:0; margin-top:0;'>Summary</h2>
{summary_html}
"""
    body = "\n".join(case_html_parts)
    foot = "</body></html>"
    OUT.write_text(head + body + foot, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
