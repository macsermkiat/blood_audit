# Ingest mapping — encrypted HOSxP bundle → `bba.ingest` schemas

**Status:** locked 2026-05-20. 12 tables mapped 1:1. The previous "schema gap" blocker (memory `bloodbank_ingest_schema_gap.md`, 2026-05-17) is resolved.

This document is the authoritative reference for what the ingest pipeline reads from `data/encrypted/`. It captures three things:

1. **Per-table column spec** — exactly which columns each schema declares.
2. **Normalize-layer rules** — per-table preprocessing (column projection, dedupe, case-normalize, year-filter) that runs *before* `validate_header`.
3. **Encryption-pipeline prerequisites** — what `encrypt_phi.py` and `thai-medical-deid` must do before a bundle is safe to ingest.

The companion code lives in `src/bba/ingest/schemas.py` (`_REGISTRY_V1`). Changing this doc requires bumping the schema version (v1 → v2) so `run_id` regenerates and prior outputs are not silently mutated.

---

## Bundle contents

The encrypted bundle at `data/encrypted/` contains **12 CSV files**. Each maps 1:1 to a schema entry by file stem.

| File | Schema key | Source |
|---|---|---|
| `BDVST.csv` | `BDVST` | direct HOSxP export (pseudonymized) |
| `BDVSTDT.csv` | `BDVSTDT` | direct HOSxP export |
| `BDVSTST.csv` | `BDVSTST` | HOSxP status code dictionary |
| `BDTYPE.csv` | `BDTYPE` | HOSxP product code dictionary |
| `Diagnosis.csv` | `Diagnosis` | IT-joined: IPTSUMDIAG + ICD10 + DIAGTYPE, 2025-pre-filtered |
| `Lab.csv` | `Lab` | IT-joined: LVSTEXM + LABGRP + LABEXM, 2025-pre-filtered |
| `Med.csv` | `Med` | IT-joined: PRSC + PRSCDT + MEDITEMDIS + MEDGENERIC, MEDTYPE=10, 2025-pre-filtered |
| `IPDADMPROGRESS.csv` | `IPDADMPROGRESS` | direct HOSxP export (large; needs deid) |
| `IPDNRFOCUSDT.csv` | `IPDNRFOCUSDT` | direct HOSxP export (large; needs deid) |
| `IPTSUMOPRT.csv` | `IPTSUMOPRT` | direct HOSxP export (procedure encounters) |
| `INCPT.csv` | `INCPT` | direct HOSxP export (procedure / operation charges) |
| `ICD9CM.csv` | `ICD9CM` | HOSxP procedure code dictionary |

**Dropped from the schema:** `UnUSE_Patient_Background` — the bundle's file with that name is an obstetric/delivery record table, not patient demographics. The real `PT` (patient registration) table is not in the bundle. Audit removes `age_years` and `sex` from `AuditOrder` to compensate; pediatric and obstetric exclusions adapt accordingly (see *Downstream impacts*).

---

## Per-table column spec

### BDVST — 12 cols (order spine)

`HN` (NOT NULL), `REQNO` (NOT NULL), `AN`, `BDVSTST`, `REQTYPE`, `CANCELDATE`, `REQDATE`, `REQTIME`, `BDVSTDATE`, `BDVSTTIME`, `ICD10`, `DIAGNOSIS`.

- Primary key: `(HN, REQNO)`.
- `BDVSTST` is FK to the `BDVSTST` dictionary.
- `REQTYPE`: `P` = ผู้ป่วยขอ (patient), `H` = โรงพยาบาลอื่นขอ (other hospital) — the audit's `inter_hospital` exclusion fires on `REQTYPE == "H"`.
- `CANCELDATE` is non-null when an order is cancelled — KCMH extension, not in HOSxP dictionary.
- `ICD10` and `DIAGNOSIS` are the order-level reason (structured code + freetext). Independent — many rows have only one or the other. Distinct from admission-level Diagnosis.csv.
- `REQDATE`+`REQTIME` is the primary anchor; `BDVSTDATE`+`BDVSTTIME` is the fallback (`anchor_imputed=True`).

### BDVSTDT — 9 cols (order line items)

`REQNO` (NOT NULL), `HN`, `BDVSTDATE`, `BDVSTTIME`, `USEDATE`, `USETIME`, `BDTYPE`, `ITEMNO`, `UNITAMT`.

- One row per requested unit (keyed by `(REQNO, ITEMNO)`).
- `BDTYPE` is the per-unit product code (e.g. `LPRC`); FK to `BDTYPE` dictionary.
- `USEDATE`+`USETIME` is the requested transfusion moment (not necessarily the actual transfusion).
- The audit's RBC product gate matches `BDTYPE ∈ {LPRC, LDPRC, SDR}` (see `audit_orders/models.py:RBCProduct`).

### BDVSTST — 2 cols (status dictionary)

`BDVSTST` (NOT NULL), `NAME`.

- 13 status rows. `NAME` is the Thai status label (e.g. `1 = ขอ`, `2 = รับใบสั่ง`).
- BDVSTST is not in the HOSxP data dictionary (`column_description.csv` has zero rows for it) — the 13 file rows are the only definition of the codes.

### BDTYPE — 2 cols (product dictionary)

`BDTYPE` (NOT NULL), `NAME`.

- 138 product rows. `NAME` is the English full label (e.g. `LPPCFI = Leukocyte poor pooled platelet conc. Filter + Irradiate`).
- `NAME_EN` from the file is dropped because samples show it is almost always blank.

### Diagnosis — 6 cols (admission diagnosis, IT-joined)

`HN` (NOT NULL), `AN` (NOT NULL), `NAME_DIAGTYPE`, `ICD10`, `ICD10WHO`, `NAME_ICD10`.

- IT-joined from IPTSUMDIAG + ICD10 + DIAGTYPE.
- Pre-filtered upstream: 2025-only, age > 15, non-cancelled, `AN IS NOT NULL`.
- `NAME_DIAGTYPE` ∈ {`Comorbidity`, `Complication`, `Principal Diagnosis`, `External Cause`, `ICDO-M`}.
- `V_DATE` from the file is **dropped** — Excel-mangled to `"00:00.0"` in every row; recoverable via `(HN, AN) → BDVST.BDVSTDATE`.
- `AN` is NOT NULL because the upstream SQL enforces it.

### Lab — 12 cols (lab results, IT-joined)

`HN` (NOT NULL), `AN` (NOT NULL), `LVSTDATE`, `LVSTTIME`, `LABGRP`, `NAME_LABGRP`, `LABEXM`, `NAME_LABEXM`, `RESULT`, `MINNRM`, `MAXNRM`, `NRMUNIT`.

- IT-joined from LVSTEXM + LABGRP + LABEXM.
- Pre-filtered upstream: 2025-only, `RESULT IS NOT NULL`, plus the same cohort filter as Diagnosis.
- `MINNRM` and `MAXNRM` may have leading/trailing whitespace — downstream must `.strip()`.
- `LVSTTIME` is HOSxP-format int (`81142` = 08:11:42); parse via `parse_hosxp_time`.

### Med — 12 cols (medication prescriptions, IT-joined)

`HN` (NOT NULL), `AN` (NOT NULL), `PRSCDATE`, `PRSCTIME`, `MEDITEM`, `NAME_MEDITEM`, `GENERIC`, `NAME_GENERIC`, `STRENGTH`, `STRENGTHUNIT`, `MEDUSETYPE`, `MEDUSEQTY`.

- IT-joined from PRSC + PRSCDT + MEDITEMDIS + MEDGENERIC (+ dictionary lookups).
- Pre-filtered upstream: 2025-only, **`MEDTYPE = 10`** (= medications; `MEDTYPE = 20` is non-medications, excluded by IT).
- All anticoagulants, iron supplements, EPO, antibiotics, etc. are in scope per `MEDTYPE = 10`.
- `STRENGTHUNIT`, `MEDUSETYPE`, `MEDUSEQTY` are frequently blank — declare nullable.

### IPDADMPROGRESS — 7 cols (SOAP progress notes)

`HN` (NOT NULL), `AN` (NOT NULL), `PROGDATE`, `SUBJECTIVE`, `OBJECTIVE`, `ASSESSMENT`, `PLAN`.

- 2.7M rows; raw export, not pre-filtered.
- The file has **duplicate column names**: `HN` at positions 1 and 30, `AN` at positions 2 and 3. Normalize must read positionally; see *Normalize-layer rules*.
- All four SOAP fields are freetext Thai/English clinical narrative; PHI is *not* pre-scrubbed in encrypt_phi.py — see *Encryption pipeline prerequisites*.
- `PROGNO` is dropped — multiple notes per day per admission are aggregated rather than ordered. Re-add if audit later needs per-note granularity.

### IPDNRFOCUSDT — 6 cols (DAR nursing notes)

`HN` (NOT NULL), `AN` (NOT NULL), `PROGRESSDATE`, `PROGRESSTIME`, `ACTION`, `RESPONSE`.

- 16.9M rows (largest file, 2.1 GB); raw export.
- `FOCUS` is **always empty** in the bundle (0 of 200k sampled rows populated) — dropped.
- `ACTION` is DAR-format `D:`/`A:` narrative; `RESPONSE` is `R:` narrative.
- Freetext PHI needs deid (same as IPDADMPROGRESS).

### IPTSUMOPRT — 4 cols (procedure encounters)

`AN` (NOT NULL), `ICD9CM`, `INDATE`, `INTIME`.

- 45 841 rows.
- **No `HN` column** in the file — derive via `An → BDVST.AN → BDVST.HN` join.
- File uses Title-Case column names (`An`, `Icd9cm`, `Indate`, `Intime`) — normalize must uppercase before validate_header.
- `INDATE` is **Excel-locale-formatted** in the file (e.g. `"June 7, 2025, 12:00 AM"`) — needs a dedicated parser (`parse_iptsumoprt_date`) to convert to ISO 8601 before validate_header.
- Audit-relevant signal: procedure start moment is the time anchor for "acute blood loss" override. End time (`OUTDATE`/`OUTTIME`) and OR-only flag (`Orflag`) are **not** declared; audit treats all procedures equally.

### INCPT — 8 cols (procedure / operation charges)

`HN` (NOT NULL), `AN` (NOT NULL), `INCDATE`, `INCTIME`, `ORDERCODE`, `INCOME`, `CANCELDATE`, `INCGRP`.

- Support / fallback source for operation lookup when IPTSUMOPRT is incomplete.
- File uses Title-Case column names (`Hn`, `Incdate`, `Inctime`, `An`, etc.) — normalize must uppercase before validate_header.
- `INCDATE` is English-locale date-only text (e.g. `"January 9, 2025"`) — normalize parses it to ISO 8601.
- Pilot operation lookup treats non-cancelled rows in `INCGRP` 110 / 111 as procedure evidence for peri-procedural proximity only. `INCPT` charge / income codes are not ICD-9-CM procedure codes and must not drive deterministic cardiac / orthopedic cohort matching unless a future explicit ICD9 mapping is added.
- The LLM pilot prompt includes nearby `INCPT` rows with group/name context so operation type can be judged from descriptions rather than numeric charge-code prefixes.

### ICD9CM — 3 cols (procedure code dictionary)

`ICD9CM` (NOT NULL), `NAME`, `ORFLAG`.

- 5 023 rows.
- File uses Title-Case (`Icd9cm`, `Name`, `Orflag`) — normalize uppercases.
- `NAME` is the English procedure name (100% populated).
- `Thainame` is **always empty** (0 of 5 023 rows) — dropped.
- `ORFLAG` is the per-code surgical flag (`1 = ผ่าตัด`, 51.4% populated; nearly all populated rows are `1`). Kept as the OR-filtering escape hatch since IPTSUMOPRT.Orflag was dropped.

---

## Normalize-layer rules

The normalize layer runs **before** `validate_header` and produces a DataFrame whose columns are a subset of those declared in `_REGISTRY_V1`. Per policy *(a)*, dropped columns are logged to the run's audit log so reviewers can diff across runs and detect newly-arrived columns.

| Table | Rule |
|---|---|
| **all 12** | Project the CSV to the schema's declared columns; log the names of all dropped columns to the run audit. |
| `IPDADMPROGRESS` | Read positionally (not by header name); drop duplicate `HN` (position 30) and `AN` (position 3); then project. Also filter `PROGDATE` to the 2025 cohort year. |
| `IPDNRFOCUSDT` | Filter `PROGRESSDATE` to 2025. |
| `IPTSUMOPRT` | Case-normalize column names to ALL-CAPS (`An` → `AN`, `Icd9cm` → `ICD9CM`, `Indate` → `INDATE`, `Intime` → `INTIME`) *before* projection. Parse `INDATE` values via `parse_iptsumoprt_date` to ISO 8601. |
| `INCPT` | Case-normalize column names to ALL-CAPS (`Hn` → `HN`, `An` → `AN`, `Incdate` → `INCDATE`, `Inctime` → `INCTIME`) *before* projection. Parse `INCDATE` values via `parse_kcmh_english_date` to ISO 8601. |
| `ICD9CM` | Case-normalize column names to ALL-CAPS. |

---

## Encryption pipeline prerequisites

Before a bundle reaches `data/encrypted/`, two pipelines must run:

1. **`encrypt_phi.py`** — pseudonymizes structured HN/AN columns. Already handles BDVST/BDVSTDT/Diagnosis/Lab/Med/IPDADMPROGRESS/IPDNRFOCUSDT/IPTSUMOPRT. Verify per release.
2. **[`thai-medical-deid`](https://github.com/macsermkiat/thai-medical-deid)** — three-layer Thai medical deidentification (regex + PyThaiNLP gazetteer + Thai NER). Runs on the freetext fields:
   - `IPDADMPROGRESS`: `SUBJECTIVE`, `OBJECTIVE`, `ASSESSMENT`, `PLAN`
   - `IPDNRFOCUSDT`: `ACTION`, `RESPONSE`
   - Tokens to expect in the output: `[PERSON]`, `[PHONE]`, `[EMAIL]`, `[ADDRESS]`, `[NATIONAL_ID]`, `[HOSPITAL_IDS]`. DATE values are deliberately preserved to keep the clinical timeline intact.

**Bundle-acceptance invariant:** spot-check 100 random rows of `IPDADMPROGRESS` and `IPDNRFOCUSDT` post-deid for residual Thai/English name fragments, phone numbers, MRN-format integers. If any leak, the bundle is rejected.

---

## Downstream impacts (audit-layer changes that follow from this spec)

These are intentional consequences of the locked spec. They are tracked here so future code changes know *why* the model shape is what it is.

1. **`bba.audit_orders.models.BloodOrderInput`** — `birthdate: date | None` and `sex: str | None` are removed. The bundle has no patient demographics table (`UnUSE_Patient_Background` is obstetric records; `PT` is not exported). Cohort age-gating is fully upstream in the IT pre-filter (`age > 15`).
2. **`bba.audit_orders.models.AuditOrder`** — `age_years: int` and `sex: str | None` are removed for the same reason.
3. **`bba.audit_orders.models.ExclusionReason = "pediatric"`** — the enum value is preserved for back-compat (so prior runs deserialize), but the audit pipeline never fires it. Upstream IT filter (`age > 15`) handles the gate.
4. **`bba.audit_orders.models.ExclusionReason = "obstetric"`** — fires on ICD-10 obstetric codes (O-block) only; no sex-gate. A male patient with an O-code is treated as a data-quality issue, not a false positive.
5. **`bba.audit_orders.models.AuditOrder`** — *should* gain `request_reason_icd10: str | None` and `request_reason_text: str | None` to carry `BDVST.ICD10` and `BDVST.DIAGNOSIS` (the clinician's order-level reason, distinct from admission-level `Diagnosis.csv`).
6. **`schema_fingerprint()` changes** — `run_id` regenerates on the next ingest. Any prior on-disk `run_id` markers are orphaned (acceptable in RED phase).

---

## Cohort assumptions baked into the IT-joined files

`Diagnosis`, `Lab`, and `Med` are pre-filtered by KCMH IT. Their SQL applies:

- `BDVSTDATE BETWEEN '2025-01-01' AND '2025-12-31'` — 2025 cohort
- `TRUNC(MONTHS_BETWEEN(BDVSTDATE, BRTHDATE)/12) > 15` — age > 15 years at order time
- `CANCELDATE IS NULL` — non-cancelled orders
- `AN IS NOT NULL` — IPD encounters only (OPD excluded)

**Consequence:** a re-run with a different cohort year or age threshold requires KCMH IT to regenerate Diagnosis.csv, Lab.csv, and Med.csv. The audit cannot widen these on its own.

`IPDADMPROGRESS` and `IPDNRFOCUSDT` are *not* pre-filtered upstream — they are raw exports. The normalize layer's year-filter rule compensates.

`IPTSUMOPRT` is also a raw export; cohort scoping happens by `An → BDVST.AN` membership at audit time.

---

## Procedure-family date format

`IPTSUMOPRT.INDATE` and `ICD9CM.Firstdate`/`Lastdate` (the latter dropped from schema) use Excel-locale text:

```
"June 7, 2025, 12:00 AM"
"January 1, 2014, 12:00 AM"
```

The `parse_iptsumoprt_date` parser (to be added under `src/bba/ingest/`) converts these to `"YYYY-MM-DD"` strings. Other tables use the standard HOSxP format `"2025-06-07 00:00:00.000"`, which `parse_hosxp_time` already handles.

---

## Open work (tracked, not blocking)

1. **`PT` table export.** If KCMH IT delivers `PT.csv` (patient registration with `brthdate`, `male`), the audit can restore per-row age and sex without re-running the upstream pre-filter. Schema entry would be `{HN: NOT NULL, BRTHDATE, MALE}` (3 cols).
2. **Raw `IPTSUMDIAG.csv` export.** Currently `Diagnosis.csv` is the IT-joined product. If raw `IPTSUMDIAG` is delivered separately, audit could perform its own join logic against ICD-10 / DIAGTYPE dictionaries. Not on the critical path.
3. **OR-only procedure filter.** Audit can opt into "OR procedures only" via `ICD9CM.ORFLAG == "1"` lookup. The capability exists; no code wires it yet.
4. **Per-note ordering in IPDADMPROGRESS.** Re-add `PROGNO` if audit ever wants to distinguish or count notes within a day. Currently aggregated.
