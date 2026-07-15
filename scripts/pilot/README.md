# Pilot scripts — mini-bundle audit walkthrough

A four-step composition pipeline that runs the full audit end-to-end on
a small random sample drawn from an encrypted HOSxP bundle, then
materialises a single self-contained HTML for human clinical review.
Built for the M8 pilot-run milestone before clinical co-lead sign-off.

These scripts intentionally live outside the `bba.*` package — they are
deployment-shaped glue that the `bba.cli`'s integration seams (e.g.
`bba audit`'s LLM-leg) would otherwise own. Treat them as a worked
example of how to compose the modules, not as a supported entry point.

## Why these scripts exist

The CLI's `bba audit` ships only the ingest leg today. The LLM-driven
analysis leg (deterministic classifier → evidence bundle → de-id →
prompt → Anthropic Batch → quote ground → calibrate → audit store) is
documented as an integration seam — the modules ship, but the
orchestrator that composes them is left to the deployment. These
scripts are that composition, written against the encrypted bundle in
`../Bloodbank/data/encrypted/`.

## Pipeline

```
sample_bundle.py        →  bundle/        (11 sampled-down CSVs)
                        →  sample_manifest.csv

run_pipeline.py         →  report.csv     (per-case deterministic verdict)

preflight_declared_usetype.py
                        →  preflight_declared_usetype.json
                                            (read-only go-live evidence)

run_llm_leg.py          →  llm_report.json
                        →  data/audit_store/   (Parquet + markers)

build_review.py         →  review.html    (single page for human review)
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `BBA_PILOT_RAW_DIR` | `../Bloodbank/data/encrypted` | Source HOSxP CSVs (11 files) |
| `BBA_PILOT_WORK_DIR` | `/tmp/bba_mini` | All outputs land here |
| `BBA_PILOT_SAMPLE_N` | `10` | Number of orders to sample |
| `BBA_PILOT_SAMPLE_SEED` | `20260519` | RNG seed for reproducibility |
| `BBA_PILOT_ICD10_CSV` | `../Bloodbank/data/raw/ICD10.csv` | ICD-10 master dictionary |
| `BBA_PILOT_LLM_MODEL` | `claude-sonnet-5` | Anthropic model id |
| `BBA_PILOT_RUN_ID` | `pilot-mini` | run_id stamped on audit_store rows |
| `BBA_PILOT_ENABLE_MISSING_HB_POSITIVE_EVIDENCE` | `false` | Opt-in to the missing-Hb MTP / peri-procedural auto-APPROPRIATE pre-check (SEED — set `1`/`true` only after clinical sign-off) |
| `BBA_PILOT_DECLARED_USETYPE` | `0` (off) | When set to `1`, threads BDVSTDT.USETYPE declared intent into both pilot legs and appends `declared_use_code` / `declared_use_label` to `report.csv` |
| `BBA_PREFLIGHT_OUT` | `$BBA_PILOT_WORK_DIR/preflight_declared_usetype.json` | Output path for the declared-USETYPE preflight JSON artifact |
| `BBA_PILOT_ONLY_REQNO` | _(unset)_ | Comma-separated REQNOs: `run_llm_leg.py` processes/submits only those cases and MERGES the fresh records into the existing `llm_report.json` (other cases keep their records). Always pair with a fresh `BBA_PILOT_RUN_ID` — the store is idempotent on `(run_id, audit_id)`, so a reused run id keeps the stale row |
| `BBA_PILOT_BATCH_MAX_WAIT` | `86400` | Seconds to wait for the Anthropic batch (default = the 24h batch SLA) |
| `ANTHROPIC_API_KEY` | _(required)_ | Anthropic credentials |
| `BBA_DATA_DIR` | _(required for `bba ingest`)_ | Where ingest writes Parquet + markers |
| `BBA_DB_URL` | _(required for `bba ingest`)_ | DB URL — placeholder is fine for ingest-only |

## End-to-end

```bash
uv sync                                              # in repo root, once
export ANTHROPIC_API_KEY=sk-ant-…
export BBA_PILOT_WORK_DIR=/tmp/bba_mini

uv run python scripts/pilot/sample_bundle.py         # sample 10 cases
uv run python scripts/pilot/run_pipeline.py          # deterministic verdicts
uv run python scripts/pilot/preflight_declared_usetype.py
                                                      # read-only go-live evidence

# Optional: run the ingest leg via the CLI on the sampled bundle
export BBA_DATA_DIR="$BBA_PILOT_WORK_DIR/data"
export BBA_DB_URL=postgresql://localhost/bba_pilot
uv run bba ingest "$BBA_PILOT_WORK_DIR/bundle/BDVST.csv"

uv run python scripts/pilot/run_llm_leg.py           # live Anthropic batch
uv run python scripts/pilot/build_review.py          # assemble review.html
open "$BBA_PILOT_WORK_DIR/review.html"
```

## Re-running a single case

To iterate on one case (e.g. after a prompt or guardrail change)
without paying for a full batch, filter the LLM leg to that REQNO
under a fresh run id, then rebuild the review — the fresh record is
merged into `llm_report.json`; the other cases are untouched:

```bash
BBA_PILOT_ONLY_REQNO=68080335 \
BBA_PILOT_RUN_ID=pilot-mini-68080335-v2 \
uv run python scripts/pilot/run_llm_leg.py

uv run python scripts/pilot/build_review.py
```

## What the LLM sees

As of issue #76 the pilot forwards the **free-text SOAP narrative**
(IPDADMPROGRESS S/O/A/P and IPDNRFOCUSDT action/response) to the
Anthropic API, not just regex-extracted numbers. Case 2 / REQNO
68012352 showed the deterministic-only payload starved the LLM of the
MAP / vasopressor evidence that lived in the prose, so the narrative now
ships verbatim and a fact-only hemodynamic summary is synthesised from
the same notes.

> **PHI precondition — operator-owned gate.** De-identification is a
> **first gate run OUTSIDE these scripts**. Because the narrative is now
> sent off-machine, the bundle you point `BBA_PILOT_RAW_DIR` at MUST
> already have passed
> [`thai-medical-deid`](https://github.com/macsermkiat/thai-medical-deid)
> (or an equivalent redactor). The raw encrypted HOSxP export in
> `../Bloodbank/data/encrypted/` still contains raw PHI in the SOAP
> columns — running `run_llm_leg.py` against it directly will send PHI
> to Anthropic. The pilot does not re-run or verify de-identification in
> process; that is the operator's responsibility before invoking it.

The full LLM evidence payload:

- ICD-10 diagnoses (AN-scoped, deduped, with ICD-10-master descriptions)
- IPDADMPROGRESS / IPDNRFOCUSDT SOAP narrative (per-source windowed)
- Hemodynamic summary (MAP nadir + vasopressor agent/dose), synthesised
  fact-only from the shipped notes
- Hb history (7-day pre-anchor; tagged closest / 24h-min / 48h-min)
- Plt, WBC, Neutrophils CBC (±1 day window)
- Meds list (±1 day window)
- Vital-sign numbers extracted via regex (SBP, DBP, HR, RR, BT)

## Weighting policy injected into the LLM prompt

`run_llm_leg.py` appends a synthetic `EvidenceChunk` (E999) that tells
the LLM how to weight Hb values relative to the order anchor:

- The Hb value closest to the order time is the primary trigger.
- The minimum Hb in the 24h pre-order window dominates the indication call.
- A 48h-window minimum is supportive, not primary.
- Sub-threshold Hb values outside the 24h window must be explicitly cited
  in the reasoning so a human reviewer sees the temporal trade-off.

This is **not validated** against a clinical gold standard. n=7
LLM-bound cases in the demo is a smoke test, not statistical
generalization. The repo ships `bba.eval_harness` for the real
validation step (M8 milestone).

## Known limitations & caveats

- **Model-id bypass.** `run_llm_leg.py` calls `AnthropicBatchTransport`
  directly instead of through `LlmClientConfig`. The model ids pinned in
  `src/bba/llm_client/models.py` (`claude-sonnet-5`, `claude-opus-4-8`)
  are bare aliases the live Anthropic API accepts, so the echoed
  model_id validates against `ALLOWED_MODELS` natively — no runtime
  allow-set patch is needed.

- **Cost-guard bypass.** `bba.audit_pipeline.cost_guard.assert_test_safe_transport`
  rejects `isinstance(transport, AnthropicBatchTransport)` to keep the
  test path off the live API. The pilot wraps the transport in a
  composition-only `RealAnthropicTransport` that satisfies the
  `AnthropicTransport` Protocol without inheriting from the test-banned
  class. Production wiring would do the same.

- **PHI safety.** All outputs under `$BBA_PILOT_WORK_DIR/` (the
  sampled bundle, `review.html`, the audit_store) contain real PHI.
  They are **not** to be committed, emailed, or shared outside the
  approved clinical review channel. The repo's `.gitignore` blocks
  `/tmp/bba_mini` and any `*.csv` under tracked directories.

- **Diagnosis dates.** HOSxP `Diagnosis.V_DATE` is Excel-corrupted to
  `00:00.0` in every row of the encrypted bundle. The review HTML
  surfaces the order's admission date (`BDVST.BDVSTDATE`) as a temporal
  proxy with an explicit footnote — there is no way to recover the
  real charting date from this bundle.

- **Single-run validation only.** The deterministic + LLM verdicts in
  this pilot are unanchored to a clinical gold standard. Treat them as
  a demonstration of the composition path, not as audit evidence
  ready for committee review.

- **MTP arm is unfed.** Both pilot scripts pass `blood_orders=()` into
  `assign_cohort`, so the MTP cluster rule (`detect_mtp_pattern`: ≥4 RBC
  units, or RBC+FFP+platelet co-order, within a 1-h window) can never
  fire. `BDVSTTRANS` carries no `REQNO`, so the bundle has no precise
  per-order RBC-unit history to build `BloodOrderEvent` records from —
  the existing join is an admission-scoped *display* join for human
  review only. Consequently, even with
  `BBA_PILOT_ENABLE_MISSING_HB_POSITIVE_EVIDENCE` enabled, true
  active-MTP missing-Hb cases still surface as `INSUFFICIENT_EVIDENCE`
  (correctly parked for a reviewer); only the peri-procedural bypass
  arm fires. A reusable `BloodOrderEvent` builder over the joined
  blood-order tables is a separate, sign-off-gated feature — no
  production code constructs `BloodOrderEvent` today.
