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
| `BBA_PILOT_LLM_MODEL` | `claude-sonnet-4-6` | Anthropic model id |
| `BBA_PILOT_RUN_ID` | `pilot-mini` | run_id stamped on audit_store rows |
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

# Optional: run the ingest leg via the CLI on the sampled bundle
export BBA_DATA_DIR="$BBA_PILOT_WORK_DIR/data"
export BBA_DB_URL=postgresql://localhost/bba_pilot
uv run bba ingest "$BBA_PILOT_WORK_DIR/bundle/BDVST.csv"

uv run python scripts/pilot/run_llm_leg.py           # live Anthropic batch
uv run python scripts/pilot/build_review.py          # assemble review.html
open "$BBA_PILOT_WORK_DIR/review.html"
```

## What the LLM sees (and doesn't)

The pilot does **not** send free-text progress / focus notes to the
Anthropic API. The encrypted bundle has not been through
[`thai-medical-deid`](https://github.com/macsermkiat/thai-medical-deid)
and still contains raw PHI in the SOAP narrative. Until that
pre-processing step is wired into the bundle build, the LLM evidence
payload is restricted to structured signals:

- ICD-10 diagnoses (AN-scoped, deduped, with ICD-10-master descriptions)
- Hb history (7-day pre-anchor; tagged closest / 24h-min / 48h-min)
- Plt, WBC, Neutrophils CBC (±1 day window)
- Meds list (±1 day window)
- Vital-sign numbers extracted via regex from the notes locally

A `vitals_extractor` regex pass over the PHI-bearing notes runs in the
local process and emits only numbers (SBP, DBP, HR, RR, BT) into the
bundle, so the narrative never leaves the machine.

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
  directly instead of through `LlmClientConfig`, because the snapshot
  IDs pinned in `src/bba/llm_client/models.py`
  (`claude-sonnet-4-6-20251018`, `claude-opus-4-7-20251030`) are not
  exposed on the live Anthropic API today — they return
  `not_found_error`. The bypass lets the pilot use the floating alias
  `claude-sonnet-4-6`. When Anthropic publishes date-pinned snapshots,
  swap the constants in `bba.llm_client.models` and delete the bypass.

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
