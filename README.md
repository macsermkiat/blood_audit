# blood_audit — KCMH RBC Transfusion Audit Pipeline (Phase 1)

Post-hoc audit of inpatient adult RBC transfusion appropriateness at Chulalongkorn Hospital, against KCMH PR 17.2 + AABB 2023 guidelines.

> **Status:** Phase 1 is feature-complete (20/20 modules, version `0.1.0`). Pilot run against one month of historical data is pending clinical co-lead sign-off (M8). Phase B operative-table data re-encryption is pending KCMH IT (#2).

Full requirements: **[PRD — issue #1](https://github.com/macsermkiat/blood_audit/issues/1)**.

## What this is (and isn't)

- **Is:** a post-hoc Quality Improvement pipeline. Reads finished HOSxP exports, classifies each RBC unit as `APPROPRIATE` / `LLM_REVIEW` / `POTENTIALLY_INAPPROPRIATE` against the 3-tier Hb policy with cohort-aware thresholds and an acute-blood-loss override.
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

# 2. Run the audit pipeline (deterministic classifier + LLM_REVIEW + quote grounding)
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
UnUSE_Patient_Background.csv    Patient demographics + background
IPTSUMOPRT.csv                  Operative table (Phase B — re-encrypted AN pending #2)
ICD9CM.csv                      ICD-9-CM Vol 3 procedure code dictionary
```

A `run_id` is `sha256(input_csv_hash + schema_fingerprint + code_version)`. The same bundle + same code version always produces the same `run_id`, so re-running `bba audit` is a no-op unless `--force`.

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
- **3-tier Hb classifier is authoritative.** Vitals are supporting evidence; the acute-blood-loss override only bypasses the Hb threshold with documented delta-Hb (≥1.5/6h, ≥2/12h, ≥2.5/24h). See [`policy/policy.md`](policy/policy.md).
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
