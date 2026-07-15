# blood_audit — KCMH Transfusion Audit Pipeline

Post-hoc audit of inpatient adult transfusion appropriateness at Chulalongkorn Hospital (KCMH), against PR 17.2 + AABB 2023 guidelines. Phase 1 audits red blood cell (RBC) orders; Phase 2 extends the same pipeline to platelet orders.

> **Status:** 26 modules under `src/bba/`, version `0.1.0`.
> - **Phase 1 (RBC)** is feature-complete at module level. `bba audit` runs the ingest leg today; the analysis leg (deterministic classifier → LLM review → quote grounding) runs via the pilot scripts and is wired into the CLI in Phase 1.5.
> - **Phase 2 (platelet auditor)** is **merged** (PR #85). The deterministic leg is live; the platelet LLM leg is behind a default-off flag (`feature_flags.PLATELET_LLM_ENABLED`). The RBC path is byte-identical throughout.
> - Pilot validation against one month of historical data is pending clinical co-lead sign-off (M8). Phase B operative-table re-encryption is pending KCMH IT (#2).

Full requirements: **[PRD — issue #1](https://github.com/macsermkiat/blood_audit/issues/1)**.

## What this is (and isn't)

- **Is:** a post-hoc Quality Improvement pipeline. Reads finished HOSxP exports, routes each blood-bank order to its component family, then classifies each RBC unit against the 3-tier Hb policy (cohort-aware thresholds + deterministic bypasses) and each platelet unit against the §5.1 count gate. Verdicts are `APPROPRIATE` / `NEEDS_REVIEW` / `INSUFFICIENT_EVIDENCE` / `POTENTIALLY_INAPPROPRIATE` / `INAPPROPRIATE`.
- **Is not:** a real-time clinical decision support tool. Nothing in this codebase is intended to influence a transfusion order at the point of care.

## Quickstart

Requires Python ≥ 3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:macsermkiat/blood_audit.git
cd blood_audit
uv sync
```

The pipeline reads a HOSxP CSV bundle. No sample data ships in-repo (PHI exposure risk) — see [Expected input bundle](#expected-input-bundle). Configuration is via environment variables; **each run path below needs a different subset**, so set them per path rather than all at once:

| Variable | Needed by | Purpose |
|----------|-----------|---------|
| `BBA_DATA_DIR` | path B (`bba` CLI) | run-state directory (Parquet + DuckDB planned) |
| `BBA_DB_URL` | audit-store + dashboard | Postgres DSN (`postgresql://user@host/db`) |
| `ANTHROPIC_API_KEY` | path A (LLM leg) | Anthropic API key (`sk-ant-...`) |
| `BBA_PILOT_WORK_DIR` | paths A, C (pilot scripts) | where outputs land (e.g. `/tmp/bba_mini`) |
| `BBA_REVIEW_XLSX`, `BBA_BDVST_CSV`, `BBA_DCT_CSV` | path C (ranking) | verdict source + attribution inputs |

## Run the pipeline

Pick the path that matches your goal. Full operator walkthrough:
[docs site → Operators](docs/src/content/docs/en/operators/run-pipeline.mdx).

| I want to… | Path | Entry point |
|------------|------|-------------|
| Run **deterministic + live LLM** together on a small sample (RBC end-to-end; platelet LLM off by default) and get a review HTML | **A** | pilot scripts |
| **Ingest + validate** a full HOSxP export into the store (the supported CLI) | **B** | `bba` CLI |
| **Rank ordering doctors / departments** by appropriateness | **C** | ranking script |

> Every path reads real HOSxP data and writes **real PHI**. Never commit, email, or share any output.

### Path A — full audit on a sample (pilot scripts)

The closest thing to an end-to-end audit today — it runs the deterministic
classifier **and** the live LLM leg together on one sample. It samples ~10 cases
from an encrypted bundle, walks them through four scripts, and produces a review
HTML. A worked example of how the modules compose, not a supported entry point.
(Details: [`scripts/pilot/README.md`](scripts/pilot/README.md).)

The live LLM leg covers RBC orders. While `PLATELET_LLM_ENABLED` is off (the
default), `run_llm_leg.py` skips present-count platelet orders — they get a
deterministic verdict but orphan instead of an LLM review — so a sample that
includes platelet orders is not audited end-to-end until the flag is enabled.

```bash
export ANTHROPIC_API_KEY=sk-ant-...          # the LLM leg
export BBA_PILOT_WORK_DIR=/tmp/bba_mini      # all outputs land here

uv run python scripts/pilot/sample_bundle.py   # 1. sample ~10 cases
uv run python scripts/pilot/run_pipeline.py    # 2. deterministic verdicts  → report.csv
uv run python scripts/pilot/run_llm_leg.py     # 3. live Anthropic batch     → llm_report.json
uv run python scripts/pilot/build_review.py    # 4. assemble review          → review.html

open "$BBA_PILOT_WORK_DIR/review.html"
```

To iterate on a single case without re-running the whole batch, set
`BBA_PILOT_ONLY_REQNO` (with a fresh `BBA_PILOT_RUN_ID`) — the fresh record is
merged into `llm_report.json` and the other cases keep theirs. See
[`scripts/pilot/README.md`](scripts/pilot/README.md#re-running-a-single-case).

### Path B — ingest a full export (supported `bba` CLI)

Runs the wired `bba` CLI against a full 12-file HOSxP export.

```bash
export BBA_DATA_DIR=/path/to/persistent/data   # run-state today (Parquet + DuckDB planned)

# 1. Ingest the bundle: validate + normalize, write the run completion marker under $BBA_DATA_DIR
uv run bba ingest /path/to/hosxp_bundle/BDVST.csv

# 2. Run the audit ingest leg (validate + normalize, write the run completion marker; idempotent per run_id)
uv run bba audit --input /path/to/hosxp_bundle/BDVST.csv
```

**What `bba audit` does today:** the **ingest leg** only — it validates the
schema, normalizes rows, and writes a run completion marker under
`$BBA_DATA_DIR/audit/<run_id>/` (marker-only; DuckDB + Parquet payload writers
are the intended next storage). The analysis leg
(deterministic classifier → evidence bundle → de-id → prompt → LLM batch →
quote grounding → audit-store write) runs through the pilot scripts (Path A)
today; its CLI wiring lands in Phase 1.5. Re-running the same input + code
version is a no-op; pass `--force` to override (writes an `audit_log` row).

### Path C — rank doctors / departments (attribution)

Ranks the top-10 ordering doctors and departments by blood-order appropriateness
in three buckets (appropriate / inappropriate / unresolved). Thin glue over
`bba.attribution`. Full guide:
[docs site → Operators → Doctor & department ranking](docs/src/content/docs/en/operators/doctor-ranking.mdx).

```bash
export BBA_REVIEW_XLSX="$HOME/Downloads/Review การใช้เลือด.xlsx"  # verdict source (300-case review)
export BBA_BDVST_CSV=/path/to/BDVST.csv    # BDVST export with REQNO + DCTREQ
export BBA_DCT_CSV=/path/to/DCT.csv        # DCT.csv doctor registry
export BBA_PILOT_WORK_DIR=/tmp/bba_mini    # outputs land here

uv run python scripts/pilot/rank_doctors.py     # → doctor_ranking.csv, department_ranking.csv, doctor_rankings.html
open "$BBA_PILOT_WORK_DIR/doctor_rankings.html"
```

## CLI

| Subcommand | Status | Notes |
|------------|--------|-------|
| `bba ingest <csv> [--schema-version v1]` | Wired | Validates + hashes + normalizes the 12-table HOSxP bundle, then writes the run completion marker (marker-only today; Parquet loader is the intended next storage). `--schema-version` selects the ingest schema (default `v1`). |
| `bba audit --input <csv>` | Wired | Run-level idempotent; `--force` overrides with an `audit_log` row. |
| `bba evaluate --run-id <id>` | Integration seam | Underlying `bba.eval_harness` primitives ship and are tested; the CLI hand-off composes them against the deployment's `audit_store`. Raises `CliError` until wired. |
| `bba report --run-id <id> --format html\|pdf\|json` | Integration seam | Underlying `bba.report_generator` ships; CLI needs the deployment to source `ReportInputs` from the store. |
| `bba serve-dashboard --port 8000` | Integration seam | FastAPI app from `bba.dashboard.create_app` ships; CLI needs deployment-specific resolvers (auth, snapshot dir, PHI un-redactor). |
| `bba sentinel --weekly\|--quarterly` | Integration seam | `bba.monitoring` primitives ship; CLI needs the deployment's cadence dispatcher. |

The four integration seams **fail loud** with a `CliError` describing the missing glue — they do not fabricate defaults that would silently mis-configure the underlying module.

Examples (Path B; `BBA_DATA_DIR` must be set):

```bash
# Ingest + audit a bundle. The run_id is derived from the input + code version,
# so re-running the same bundle is a no-op.
uv run bba ingest /path/to/hosxp_bundle/BDVST.csv
uv run bba audit --input /path/to/hosxp_bundle/BDVST.csv

# Confirm a run already completed this session and print its row count (no-op).
# --run-id and --input are mutually exclusive; --run-id alone never re-runs the
# pipeline — it errors if that run was never completed:
uv run bba audit --run-id <run_id>

# Force a re-audit of an already-complete run — writes an audit_log override row
# instead of no-op'ing (needs --input; --force without --input does not re-run):
uv run bba audit --input /path/to/hosxp_bundle/BDVST.csv --force
```

## Expected input bundle

The HOSxP export directory must contain these 12 CSV files (header validation is strict; see `src/bba/ingest/schemas.py` for column lists):

```
BDVST.csv                       Blood-bank visit (one row per unit)
BDVSTDT.csv                     Blood-bank visit dates / times
BDTYPE.csv                      Blood-component-type dictionary
BDVSTST.csv                     Blood-bank visit status
Diagnosis.csv                   ICD-10 admission + discharge dx
Lab.csv                         All lab results (Hb queried here)
MED.csv                         Medication administration record
IPDADMPROGRESS.csv              Daily admission progress notes
IPDNRFOCUSDT.csv                Nursing focus-charting timestamps
IPTSUMOPRT.csv                  Operative table (Phase B — re-encrypted AN pending #2)
INCPT_OPRTACT.csv               Procedure charge support (INCPT joined to OPRTACT on Income, per #69)
ICD9CM.csv                      ICD-9-CM Vol 3 procedure code dictionary
```

A `run_id` is `sha256(input_csv_hash + schema_fingerprint + code_version)`. The same bundle + same code version always produces the same `run_id`, so re-running `bba audit` is a no-op unless `--force`.

## HOSxP join examples

These examples document how to pull the supporting HOSxP tables for the
adult inpatient transfusion subpopulation. Replace `DATE '<START_DATE>'`
and `DATE '<END_DATE>'` with the export window, for example the first and
last date of the audit month or year.

Common cohort pattern:

- Anchor on `CUH.BDVST` because it contains blood-bank visits/orders.
- Join `CUH.PT` on `HN` to compute adult age at `BDVSTDATE`.
- Keep inpatient rows only: `AN IS NOT NULL`.
- Exclude cancelled blood-bank visits: `CANCELDATE IS NULL`.
- Join most clinical evidence tables back to the subpopulation by `AN`.

```sql
/* LAB */
WITH SUBPOPULATION AS
(
 SELECT DISTINCT b.HN ,b.AN --,b.BDVSTDATE
 FROM CUH.BDVST b
  LEFT JOIN CUH.PT t ON t.HN = b.HN
 WHERE b.BDVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
     AND TRUNC(MONTHS_BETWEEN(b.BDVSTDATE,t.BRTHDATE)/12) > 15
     AND b.CANCELDATE IS NULL
     AND b.AN IS NOT NULL
)
SELECT DISTINCT sp.HN
    ,sp.AN
--    ,e.NAME AS SEX
--    ,FLOOR(MONTHS_BETWEEN(sp.BDVSTDATE,p.BRTHDATE)/12) AS AGE
--    ,p.BRTHDATE
--    ,OCCPTN
    ,l.LVSTDATE
    ,l.LVSTTIME
    ,l.LABGRP
    ,(SELECT NAME FROM CUH.LABGRP b WHERE b.LABGRP = l.LABGRP) AS NAME_LABGRP
    ,l.LABEXM
    ,(SELECT NAME FROM CUH.LABEXM x WHERE x.LABEXM = l.LABEXM) AS NAME_LABEXM
    ,l.RESULT
    ,l.MINNRM
    ,l.MAXNRM
    ,l.NRMUNIT
FROM SUBPOPULATION sp
--    LEFT JOIN CUH.PT p ON p.HN = sp.HN
--    LEFT JOIN CUH.MALE e ON e.MALE = p.MALE
    LEFT JOIN CUH.LVSTEXM l ON l.AN = sp.AN
WHERE l.LVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
    AND l.RESULT IS NOT NULL;
```

```sql
/* Diagnosis */
WITH SUBPOPULATION AS
(
 SELECT DISTINCT b.HN ,b.AN ,b.BDVSTDATE
 FROM CUH.BDVST b
  LEFT JOIN CUH.PT t ON t.HN = b.HN
 WHERE b.BDVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
     AND TRUNC(MONTHS_BETWEEN(b.BDVSTDATE,t.BRTHDATE)/12) > 15
     AND b.CANCELDATE IS NULL
     AND b.AN IS NOT NULL
)
SELECT DISTINCT
    sp.HN
    ,sp.AN
    ,sp.BDVSTDATE AS V_DATE
    ,d2.NAME AS NAME_DIAGTYPE
    ,d.ICD10
    ,c.ICD10WHO
    ,c.NAME AS NAME_ICD10
    ,'IN_HOSPITAL' AS DIAG_TYPE
FROM SUBPOPULATION sp
    INNER JOIN CUH.IPTSUMDIAG d ON d.AN = sp.AN
    LEFT JOIN CUH.ICD10 c ON c.ICD10 = d.ICD10
    LEFT JOIN CUH.DIAGTYPE d2 ON d2.DIAGTYPE = d.DIAGTYPE
WHERE sp.BDVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>';
--    AND d.DIAGTYPE = 3
--    AND c.ICD10WHO NOT IN ('O82','O820','O821','O842')
```

```sql
/* MED */
WITH SUBPOPULATION AS
(
 SELECT DISTINCT b.HN ,b.AN ,b.BDVSTDATE
 FROM CUH.BDVST b
  LEFT JOIN CUH.PT t ON t.HN = b.HN
 WHERE b.BDVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
     AND TRUNC(MONTHS_BETWEEN(b.BDVSTDATE,t.BRTHDATE)/12) > 15
     AND b.CANCELDATE IS NULL
     AND b.AN IS NOT NULL
)
SELECT DISTINCT sp.HN
    ,sp.AN
    ,p.PRSCDATE
    ,p.PRSCTIME
    ,p2.MEDITEM
    ,m.MEDNAME AS NAME_MEDITEM
    ,r.GENERIC
    ,r.NAME AS NAME_GENERIC
    ,m.STRENGTH
    ,(SELECT NAME FROM CUH.STRENGTHUNIT s WHERE s.STRENGTHUNIT = m.STRENGTHUNIT) AS STRENGTHUNIT
    ,(SELECT NAME FROM CUH.MEDUSETYPE m3 WHERE m3.MEDFORM = m.MEDFORM AND m3.MEDUSETYPE = m.MEDUSETYPE) AS MEDUSETYPE
    ,(SELECT NAME FROM CUH.MEDUSEQTY m2 WHERE m2.MEDUSEQTY = m.MEDUSEQTY) AS MEDUSEQTY
FROM SUBPOPULATION sp
    INNER JOIN CUH.PRSC p ON p.AN = sp.AN
    LEFT JOIN CUH.PRSCDT p2 ON p2.SPHMLCT = p.SPHMLCT AND p2.PRSCNO = p.PRSCNO
    LEFT JOIN CUH.MEDITEMDIS m ON m.MEDITEM = p2.MEDITEM
    LEFT JOIN CUH.MEDITEMGNR g ON g.MEDITEM = m.MEDITEM
    LEFT JOIN CUH.MEDGENERIC r ON r.GENERIC = g.GENERIC
WHERE p.PRSCDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
    AND m.MEDTYPE = 10;
```

```sql
/* Patient Background */
WITH SUBPOPULATION AS
(
 SELECT DISTINCT b.HN ,b.AN ,b.BDVSTDATE
 FROM CUH.BDVST b
  LEFT JOIN CUH.PT t ON t.HN = b.HN
 WHERE b.BDVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
     AND TRUNC(MONTHS_BETWEEN(b.BDVSTDATE,t.BRTHDATE)/12) > 15
     AND b.CANCELDATE IS NULL
     AND b.AN IS NOT NULL
)
SELECT DISTINCT sp.HN
    ,sp.AN
    ,d.DLVSTREQNO
    ,d.DLVSTRECDATE
    ,d.DLVSTRECTIME
    ,d.WARD
    ,d.ROOM
    ,d.BEDNO
    ,d.PTNTTYPE
    ,d.GRVDTY
    ,d.PARITY
    ,d.PARITY2
    ,d.PARITY3
    ,d.PARITY4
    ,d.ABRTN
    ,d.ABRTNFLAG
    ,d.CURTTFLAG
    ,d.ANCFLAG
    ,d.ANCPLACE
    ,(SELECT ANCPLACENM FROM CUH.ANCPLACE a WHERE a.ANCPLACE = d.ANCPLACE) AS NAME_ANCPLACE
    ,d.ANCPLACE2
    ,(SELECT ANCPLACENM FROM CUH.ANCPLACE a WHERE a.ANCPLACE = d.ANCPLACE2) AS NAME_ANCPLACE2
    ,d.NOATTEND
    ,d.NOATTEND2
    ,d.LMPDATE
    ,d.EDCDATE
    ,d.GSTTNAGE
    ,d.GSTTNDAY
    ,d.LABFLAG
    ,d.TKDATEBLDG
    ,d.BLDTEST
    ,d.RHTEST
    ,(SELECT NAME FROM CUH.RHTEST r WHERE r.RHTEST = d.RHTEST) AS NAME_RHTEST
    ,d.TKDATEHBSAG
    ,d.HBSAGTEST
--    ,(SELECT NAME FROM CUH.HBSAGTEST h WHERE h.HBSAGTEST = d.HBSAGTEST) AS NAME_HBSAGTEST
    ,d.TKDATEVDRL
    ,d.VDRLTEST
--    ,(SELECT NAME FROM CUH.VDRLTEST v WHERE v.VDRLTEST = d.VDRLTEST) AS NAME_VDRLTEST
    ,d.TKDATETPHA
    ,d.TPHATEST
    ,(SELECT NAME FROM CUH.TPHATEST t WHERE t.TPHATEST = d.TPHATEST) AS NAME_TPHATEST
    ,d.HIVTEST
    ,(SELECT NAME FROM CUH.HIVTEST h2 WHERE h2.HIVTEST = d.HIVTEST) AS NAME_HIVTEST
    ,d.TKDATEHCT
    ,d.HCTTEST
    ,d.TKDATECMIA
    ,d.CMIATEST
    ,(SELECT NAME FROM CUH.CMIATEST c WHERE c.CMIATEST = d.CMIATEST) AS NAME_CMIATEST
    ,d.LEADSIGN
--    ,(SELECT NAME FROM CUH.LEADSIGN l WHERE l.LEADSIGN = d.LEADSIGN) AS NAME_LEADSIGN
    ,d.LEADSIGNOTH
    ,d.HEADENGGMNT
    ,d.HEARTSOUND
    ,d.SOUNDDATE
    ,d.SOUNDTIME
    ,d.RECSTATUS
    ,d.KEYBACK
    ,d.TKDATEHIV
    ,d.FHRAMT
    ,d.DLVSTPLACE
    ,d.CMIATESTNEW
    ,d2.INDCTION
    ,d2.INDCTIONOTH
    ,d2.EPISTM
    ,d2.MTHDRP
    ,d2.PLCNTDL
    ,d3.ONSETDATE
    ,d3.ONSETTIME
    ,d3.FDLTTNDATE
    ,d3.FDLTTNTIME
    ,d3.DBRTHDATE
    ,d3.DBRTHTIME
    ,d3.PLCTDATE
    ,d3.PLCTTIME
    ,d3.RPTRTYPE
    ,d3.RPTRDATE
    ,d3.RPTRTIME
FROM SUBPOPULATION sp
    LEFT JOIN CUH.DLVSTEXT d ON d.AN = sp.AN
    LEFT JOIN CUH.DLVSTDESC d2 ON d2.DLVSTREQNO = d.DLVSTREQNO
    LEFT JOIN CUH.DLVST d3 ON d3.DLVSTREQNO = d.DLVSTREQNO
WHERE d.DLVSTRECDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>';
```

```sql
/* IPDNRFOCUSDT */
WITH SUBPOPULATION AS
(
 SELECT DISTINCT b.HN ,b.AN ,b.BDVSTDATE
 FROM CUH.BDVST b
  LEFT JOIN CUH.PT t ON t.HN = b.HN
 WHERE b.BDVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
     AND TRUNC(MONTHS_BETWEEN(b.BDVSTDATE,t.BRTHDATE)/12) > 15
     AND b.CANCELDATE IS NULL
     AND b.AN IS NOT NULL
)
SELECT DISTINCT sp.HN
    ,sp.AN
    ,i.ITEMNO
    ,i.ITEMNO2
    ,i.FOCUS
    ,i.ACTION
    ,i.RESPONSE
    ,i.PROGRESSDATE
    ,i.PROGRESSTIME
FROM SUBPOPULATION sp
    INNER JOIN CUH.IPDNRFOCUSDT i ON i.AN = sp.AN
WHERE i.CANCELDATE IS NULL;
```

```sql
/* IPDADMPROGRESS */
WITH SUBPOPULATION AS
(
 SELECT DISTINCT b.HN ,b.AN --,b.BDVSTDATE
 FROM CUH.BDVST b
  LEFT JOIN CUH.PT t ON t.HN = b.HN
 WHERE b.BDVSTDATE BETWEEN DATE '<START_DATE>' AND DATE '<END_DATE>'
     AND TRUNC(MONTHS_BETWEEN(b.BDVSTDATE,t.BRTHDATE)/12) > 15
     AND b.CANCELDATE IS NULL
     AND b.AN IS NOT NULL
)
SELECT DISTINCT sp.HN
    ,i.*
FROM SUBPOPULATION sp
    INNER JOIN CUH.IPDADMPROGRESS i ON i.AN = sp.AN
WHERE i.CANCELDATE IS NULL;
```

```sql
/* INCPT */
SELECT
  "KCMH_HIS"."INCPT"."hn" AS "hn",
  "KCMH_HIS"."INCPT"."incdate" AS "incdate",
  "KCMH_HIS"."INCPT"."inctime" AS "inctime",
  "KCMH_HIS"."INCPT"."ordercode" AS "ordercode",
  "KCMH_HIS"."INCPT"."income" AS "income",
  "KCMH_HIS"."INCPT"."an" AS "an",
  "KCMH_HIS"."INCPT"."canceldate" AS "canceldate",
  "KCMH_HIS"."INCPT"."incgrp" AS "incgrp",
  "Incgrp"."name" AS "Incgrp__name",
  "Incgrp"."incgrp" AS "Incgrp__incgrp",
  "Oprtact - Income"."oprtact" AS "Oprtact - Income__oprtact",
  "Oprtact - Income"."name_en" AS "Oprtact - Income__name_en",
  "Oprtact - Income"."name" AS "Oprtact - Income__name",
  "Oprtact - Income"."income" AS "Oprtact - Income__income",
  "Oprtact - Income"."maxamt" AS "Oprtact - Income__maxamt",
  "Oprtact - Income"."dftamt" AS "Oprtact - Income__dftamt",
  "Oprtact - Income"."rfndamt" AS "Oprtact - Income__rfndamt",
  "Oprtact - Income"."minamt" AS "Oprtact - Income__minamt",
  "Oprtact - Income"."spclty" AS "Oprtact - Income__spclty",
  "Oprtact - Income"."otamt" AS "Oprtact - Income__otamt",
  "Oprtact - Income"."otrfndamt" AS "Oprtact - Income__otrfndamt",
  "Oprtact - Income"."icd9cm" AS "Oprtact - Income__icd9cm",
  "Oprtact - Income"."icd10tm" AS "Oprtact - Income__icd10tm",
  "Oprtact - Income"."oprttype" AS "Oprtact - Income__oprttype",
  "Oprtact - Income"."cost" AS "Oprtact - Income__cost",
  "Oprtact - Income"."nhso" AS "Oprtact - Income__nhso",
  "Oprtact - Income"."drgmx" AS "Oprtact - Income__drgmx",
  "Oprtact - Income"."csmbs" AS "Oprtact - Income__csmbs",
  "Oprtact - Income"."sss" AS "Oprtact - Income__sss",
  "Oprtact - Income"."var" AS "Oprtact - Income__var",
  "Oprtact - Income"."minvalue" AS "Oprtact - Income__minvalue",
  "Oprtact - Income"."maxvalue" AS "Oprtact - Income__maxvalue",
  "Oprtact - Income"."dscper" AS "Oprtact - Income__dscper",
  "Oprtact - Income"."dscamt" AS "Oprtact - Income__dscamt",
  "Oprtact - Income"."prfact" AS "Oprtact - Income__prfact",
  "Oprtact - Income"."sumprfamt" AS "Oprtact - Income__sumprfamt",
  "Oprtact - Income"."oprtgrp" AS "Oprtact - Income__oprtgrp",
  "Oprtact - Income"."icdnum" AS "Oprtact - Income__icdnum",
  "Oprtact - Income"."icdtype" AS "Oprtact - Income__icdtype",
  "Oprtact - Income"."highcost" AS "Oprtact - Income__highcost",
  "Oprtact - Income"."major" AS "Oprtact - Income__major",
  "Oprtact - Income"."dz8" AS "Oprtact - Income__dz8",
  "Oprtact - Income"."vac" AS "Oprtact - Income__vac",
  "Oprtact - Income"."vcctype" AS "Oprtact - Income__vcctype",
  "Oprtact - Income"."timemake" AS "Oprtact - Income__timemake",
  "Oprtact - Income"."validcode" AS "Oprtact - Income__validcode",
  "Oprtact - Income"."canceldate" AS "Oprtact - Income__canceldate",
  "Oprtact - Income"."datasource" AS "Oprtact - Income__datasource",
  "Oprtact - Income"."firststf" AS "Oprtact - Income__firststf",
  "Oprtact - Income"."firstdate" AS "Oprtact - Income__firstdate",
  "Oprtact - Income"."laststf" AS "Oprtact - Income__laststf",
  "Oprtact - Income"."lastdate" AS "Oprtact - Income__lastdate",
  "Oprtact - Income"."incomeold" AS "Oprtact - Income__incomeold",
  "Oprtact - Income"."icd9cmadd1" AS "Oprtact - Income__icd9cmadd1",
  "Oprtact - Income"."icd9cmadd2" AS "Oprtact - Income__icd9cmadd2",
  "Oprtact - Income"."oprtactauto" AS "Oprtact - Income__oprtactauto",
  "Oprtact - Income"."perday" AS "Oprtact - Income__perday",
  "Oprtact - Income"."unit" AS "Oprtact - Income__unit",
  "Oprtact - Income"."note" AS "Oprtact - Income__note",
  "Oprtact - Income"."cancelstf" AS "Oprtact - Income__cancelstf",
  "Oprtact - Income"."cancelnote" AS "Oprtact - Income__cancelnote",
  "Oprtact - Income"."oprtactgrp" AS "Oprtact - Income__oprtactgrp",
  "Oprtact - Income"."thtype" AS "Oprtact - Income__thtype",
  "Oprtact - Income"."addper" AS "Oprtact - Income__addper",
  "Oprtact - Income"."remflag" AS "Oprtact - Income__remflag",
  "Oprtact - Income"."nodefchk" AS "Oprtact - Income__nodefchk"
FROM
  "KCMH_HIS"."INCPT"
 
LEFT JOIN "KCMH_HIS"."INCGRP" AS "Incgrp" ON "KCMH_HIS"."INCPT"."incgrp" = "Incgrp"."incgrp"
  LEFT JOIN "cleaned_ddc_internal"."OPRTACT" AS "Oprtact - Income" ON "KCMH_HIS"."INCPT"."income" = "Oprtact - Income"."oprtact"
WHERE
  "KCMH_HIS"."INCPT"."incgrp" BETWEEN 110
   AND 111
  AND (
    "KCMH_HIS"."INCPT"."incdate" >= timestamp '2025-01-01 00:00:00.000'
  )
  AND (
    "KCMH_HIS"."INCPT"."incdate" < timestamp '2026-01-01 00:00:00.000'
  )
  AND ("KCMH_HIS"."INCPT"."an" IS NOT NULL)
  AND (
    ("KCMH_HIS"."INCPT"."an" <> '')
   
    OR ("KCMH_HIS"."INCPT"."an" IS NULL)
  )
```


## Architecture

26 modules under `src/bba/`. See **[`docs/CONTEXT.md`](docs/CONTEXT.md)** for the module glossary — every public interface, invariant, and seam is documented there. Read it before touching anything.

Coarse dependency shape (the RBC path is Phase 1; the platelet path is Phase 2 and forks at the `component_map` intake gate):

```
ingest → audit_orders → component_map ─┬─ RED_CELL → hb_lookup, vitals_extractor, cohort_detector
                                       │                             ↘
                                       │                              deterministic_classifier
                                       │                                     ↘
                                       └─ PLATELET → platelet_lookup → platelet_classifier
                                                                             ↘ (platelet_guardrail)
evidence_bundle_builder → deid_redactor → prompt_builder → llm_client ─────────↘
                                                    ↘                    quote_grounder → confidence_calibrator
                                                                                                  ↘
                                                                                    audit_pipeline → audit_store
                                                                                                         ↑
      eval_harness, monitoring, review_actions, dashboard, report_generator, attribution, verification, cli
```

Modules added since the initial Phase 1 build: `component_map` (component-family intake gate), `platelet_lookup`, `platelet_classifier`, `platelet_guardrail` (Phase 2 platelet auditor); `attribution` (doctor/department appropriateness ranking); `verification` (before/after scoring for the peri-op fix).

## What's new since Phase 1

- **Phase 2 — platelet auditor (merged, PR #85).** `component_map` routes each blood-bank order to a component family (`RED_CELL`, `PLATELET`, `FFP`, `CRYO`, `WHOLE_BLOOD`, `UNKNOWN`) from the BDTYPE dictionary name. Platelet orders go through `platelet_lookup` (count value layer) and `platelet_classifier` (§5.1 gate). The gate **auto-clears nothing**: count ≥ 100 ×10³/µL → `POTENTIALLY_INAPPROPRIATE` (routes to the LLM review leg), count < 100 → `NEEDS_REVIEW` (defers to the LLM leg), missing count → `INSUFFICIENT_EVIDENCE` by default. The platelet LLM leg (`PLATELET_REVIEW` prompt + grounded-hard-signal over-clear guardrail) is behind default-off `feature_flags.PLATELET_LLM_ENABLED` — so today only missing-count `INSUFFICIENT_EVIDENCE` rows are written deterministic-final; present-count orders (`POTENTIALLY_INAPPROPRIATE` / `NEEDS_REVIEW`) are classified but orphan (skipped, **not reviewed**) until the flag is enabled.
- **Doctor / department attribution (Feature 2, PR #82).** `bba.attribution` ranks the top-10 ordering doctors and departments by blood-order appropriateness in three buckets (appropriate / inappropriate / unresolved). Attribution is `BDVST.DCTREQ` → `DCT.csv`; the verdict source is swappable (`VerdictSource`) — the current build ranks on the 300-case human review, and a full-cohort pipeline verdict source drops in without other changes. Driven by `scripts/pilot/rank_doctors.py` and surfaced through the `serve-dashboard` integration seam.
- **Cohort refinements.** Orthopedic surgery is now its own `ORTHO_SURGERY` cohort (Hb floor 8.0, PR #83), split out from the cardiac cohort. A `CARDIOPULMONARY_COMORBIDITY` cohort (heart-disease floor 8.0) and the peri-op defer-to-LLM behavior landed in PR #81 — pre-op crossmatch defers to the LLM, and hard peri-op evidence (`intraop_transfusion` or `EBL ≥ 500 mL`) bypasses.

## Safety & policy notes

- **No live PHI in tests.** Integration tests use mock-AN fixtures; LLM tests replay VCR cassettes from `#22`.
- **3-tier Hb classifier is authoritative.** Vitals are supporting evidence only. Deterministic bypasses require structured MTP, procedure-timing, pre-op crossmatch, delta-Hb, or hard peri-op evidence (`intraop_transfusion` or `EBL ≥ 500 mL`). See the docs page [3-tier Hb classifier](docs/src/content/docs/en/developers/three-tier-hb.mdx) for the decision-flow graphic.
- **Missing-Hb positive-evidence pre-pass is disabled by default.** The `enable_missing_hb_positive_evidence` flag (`BBA_PILOT_ENABLE_MISSING_HB_POSITIVE_EVIDENCE`) is `False` until the QI committee signs off. When `False`, missing Hb always returns `INSUFFICIENT_EVIDENCE` (original PRD spec); when `True`, the pre-pass auto-approves on hard peri-op evidence and defers everything else to the LLM rather than dead-ending.
- **Hemodynamic and peri-op summaries are supporting evidence only.** The `scan_hemodynamics` and `scan_periop` scans surface MAP/vasopressor/EBL facts as pinned, truncation-exempt evidence items. They never gate the deterministic classifier — all appropriateness weighting stays with the LLM and the auditor.
- **Quote-grounding is fail-closed.** The LLM_REVIEW leg's claims are checked against six anti-hallucination layers (NFC + substring + cited_id + within-doc uniqueness + ≥25 chars + numeric-tuple + medical-NLI). Failures route to `hallucination_suspect`, not to a result row.
- **Prose-trusted bleeding auto-clears are current-episode only.** The RBC over-clear guardrail's sole prose exemption (a grounded `ACTIVE_BLEEDING` quote, confidence ≥ 0.8, > 300 mL or a life-threatening marker) blanks quote spans charted under a `d/m/y` date anchor (Thai BE or CE) more than 7 days before the order, so a weeks-old quantified index bleed cannot clear today's order (case 68080335). Undated or unparseable text is untouched — the gate can only withhold the exemption, never widen the auto-clear surface. The RBC prompt states the same current-episode rule so the model's verdict agrees with the guardrail.
- **Platelet auditor auto-clears nothing.** `platelet_classifier` never emits `APPROPRIATE` or `INAPPROPRIATE` in v1: because the policies withhold platelets at very low counts for several exclusion populations (TTP/HIT transfusion is actively harmful), every present count routes onward to review or the LLM rather than dead-ending as deterministic-final. Pinned by a hypothesis property test (CR-M2).
- **Component families are isolated at intake.** `component_map` keeps platelet statistics out of RBC reporting and the dashboard, and excludes `FFP` / `CRYO` / `WHOLE_BLOOD` with a precise reason rather than guessing them into an auditable family.
- **Run-level idempotency is enforced at the store layer.** `bba audit` cannot accidentally produce two rows for the same `(run_id, encounter_id)` pair.

## How this was built (history)

Each module was developed against a child issue under the PRD using a three-step Claude Code workflow:

```
./scripts/ralph_ticket.sh <N> <module_path>
```

Generates:
1. `/tdd` — scaffold failing tests + interface (RED)
2. `/ralph-loop` — iterate red → green → refactor until `<promise>ISSUE-N-COMPLETE</promise>` fires
3. `/codex:review` — independent review on the diff

Then `gh issue close N` after review passes. Parallel work happened in `git worktree` siblings (`../blood_audit-<branch>/`).

### Build order

| Ticket | Module | Depends on |
|--------|--------|------------|
| #2 | M0 — KCMH IT procedure-table export request (manual) | (none) |
| #3 | `bba.ingest` | (none — foundation) |
| #4 | `bba.audit_orders` | #3 |
| #5 | `bba.hb_lookup` | #3 |
| #6 | `bba.vitals_extractor` | #3 |
| #16 | `bba.evidence_bundle_builder` | #3, #5, #6 |
| #19 | `bba.audit_store` | #3 |
| #17 | `bba.deid_redactor` | #16 |
| #7 | `bba.cohort_detector` | #3, #2 |
| #8 | `bba.deterministic_classifier` | #4, #5, #6, #7 |
| #21 | `bba.prompt_builder` | #17 |
| #22 | `bba.llm_client` | #21 |
| #20 | `bba.eval_harness` | #19 |
| #23 | `bba.confidence_calibrator` | #22 |
| #18 | `bba.quote_grounder` | (none — pure function) |
| #25 | `bba.review_actions` | #19 |
| #24 | `bba.audit_pipeline` | #8, #16, #17, #22, #18, #19 |
| #26 | `bba.dashboard` | #19, #25 |
| #27 | `bba.monitoring` | #20, #24 |
| #28 | `bba.report_generator` | #19 |
| #29 | `bba.cli` | #24, #20, #28, #26, #27 |

Modules without deps were built in parallel.

Phase 2 (platelet auditor) and the attribution/cohort features followed the same workflow on feature branches: `feat/periop-defer-llm` (#81, peri-op defer + cardiopulmonary cohort), `feat/doctor-dept-ranking` (#82, `attribution`), `feat/ortho-surgery-cohort` (#83, `ORTHO_SURGERY` split), `feat/platelet-auditor` (#84, deterministic core), and `feat/platelet-pipeline-dispatch` (#85, live-pipeline integration + platelet LLM leg).
