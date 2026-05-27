# blood_audit — KCMH RBC Transfusion Audit Pipeline (Phase 1)

Post-hoc audit of inpatient adult RBC transfusion appropriateness at Chulalongkorn Hospital, against KCMH PR 17.2 + AABB 2023 guidelines.

> **Status:** Phase 1 is feature-complete (20/20 modules, version `0.1.0`). Pilot run against one month of historical data is pending clinical co-lead sign-off (M8). Phase B operative-table data re-encryption is pending KCMH IT (#2).

Full requirements: **[PRD — issue #1](https://github.com/macsermkiat/blood_audit/issues/1)**.

## What this is (and isn't)

- **Is:** a post-hoc Quality Improvement pipeline. Reads finished HOSxP exports, classifies each RBC unit as `APPROPRIATE` / `NEEDS_REVIEW` / `INSUFFICIENT_EVIDENCE` / `POTENTIALLY_INAPPROPRIATE` against the 3-tier Hb policy with cohort-aware thresholds and deterministic bypasses.
- **Is not:** a real-time clinical decision support tool. Nothing in this codebase is intended to influence a transfusion order at the point of care.

## Quickstart

Requires Python ≥ 3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:macsermkiat/blood_audit.git
cd blood_audit
uv sync
```

Set the three environment variables:

```bash
export BBA_DATA_DIR=/path/to/persistent/data    # required — Parquet + DuckDB + run-state
export BBA_DB_URL=postgresql://user@host/db     # required for audit-store + dashboard
export ANTHROPIC_API_KEY=sk-ant-...             # required for the LLM_REVIEW leg
```

No sample data ships in-repo (PHI exposure risk). The pipeline reads a HOSxP CSV bundle — see [Expected input bundle](#expected-input-bundle).

### End-to-end on a real bundle

```bash
# 1. Ingest the HOSxP CSV bundle into DuckDB + Parquet under $BBA_DATA_DIR
uv run bba ingest /path/to/hosxp_bundle/BDVST.csv

# 2. Run the audit pipeline (deterministic classifier + LLM review route + quote grounding)
uv run bba audit --input /path/to/hosxp_bundle/BDVST.csv
# The pipeline is run-level idempotent: re-running on the same input is a no-op.
# Use --force to override (writes a compliance row to audit_log).
```

## CLI

| Subcommand | Status | Notes |
|------------|--------|-------|
| `bba ingest <csv>` | Wired | Validates + hashes + Parquet-loads the 12-table HOSxP bundle. |
| `bba audit --input <csv>` | Wired | Run-level idempotent; `--force` overrides with an `audit_log` row. |
| `bba evaluate --run-id <id>` | Integration seam | Underlying `bba.eval_harness` primitives ship and are tested; the CLI hand-off composes them against the deployment's `audit_store`. Raises `CliError` until wired. |
| `bba report --run-id <id> --format html\|pdf\|json` | Integration seam | Underlying `bba.report_generator` ships; CLI needs the deployment to source `ReportInputs` from the store. |
| `bba serve-dashboard --port 8000` | Integration seam | FastAPI app from `bba.dashboard.create_app` ships; CLI needs deployment-specific resolvers (auth, snapshot dir, PHI un-redactor). |
| `bba sentinel --weekly\|--quarterly` | Integration seam | `bba.monitoring` primitives ship; CLI needs the deployment's cadence dispatcher. |

The four integration seams **fail loud** with a `CliError` describing the missing glue — they do not fabricate defaults that would silently mis-configure the underlying module.

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
INCPT.csv                       Procedure / operation charge support table
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

20 modules under `src/bba/`. See **[`docs/CONTEXT.md`](docs/CONTEXT.md)** for the module glossary — every public interface, invariant, and seam is documented there. Read it before touching anything.

Coarse dependency shape:

```
ingest → audit_orders, hb_lookup, vitals_extractor, cohort_detector
                                          ↘
                                           deterministic_classifier
                                                  ↘
evidence_bundle_builder → deid_redactor → prompt_builder → llm_client
                                                                ↘
                                                   quote_grounder → confidence_calibrator
                                                                              ↘
                                                                       audit_pipeline → audit_store
                                                                                            ↑
                          eval_harness, monitoring, review_actions, dashboard, report_generator, cli
```

## Safety & policy notes

- **No live PHI in tests.** Integration tests use mock-AN fixtures; LLM tests replay VCR cassettes from `#22`.
- **3-tier Hb classifier is authoritative.** Vitals are supporting evidence only. Deterministic bypasses require structured MTP, procedure-timing, pre-op crossmatch, or delta-Hb evidence. See the docs page [3-tier Hb classifier](docs/src/content/docs/en/developers/three-tier-hb.mdx) for the decision-flow graphic.
- **Quote-grounding is fail-closed.** The LLM_REVIEW leg's claims are checked against six anti-hallucination layers (NFC + substring + cited_id + within-doc uniqueness + ≥25 chars + numeric-tuple + medical-NLI). Failures route to `hallucination_suspect`, not to a result row.
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
