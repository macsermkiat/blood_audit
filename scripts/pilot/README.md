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

## Auxiliary & maintenance scripts

Beyond the five-step walkthrough above, the pilot ships helper drivers for
iteration, go-live evidence, and retroactive repair. All read the same
`BBA_PILOT_WORK_DIR` and follow the same PHI handling rules.

### `offline_reapply.py` — re-score cached LLM responses, zero API spend

Rebuilds pilot contexts with the CURRENT cohorts, dispatch, and guardrails, then
replays the cached Anthropic responses through the audit transform without
calling the API. Use it to validate a cohort/guardrail change against an
existing batch instead of paying for a fresh one. `BBA_PILOT_RUN_ID` is required
and must differ from the source run id (a reused id makes every audit-store
write an idempotent no-op).

```bash
BBA_PILOT_SOURCE_RUN_ID=pilot-mini \
BBA_PILOT_RUN_ID=pilot-mini-reapply-v2 \
uv run python scripts/pilot/offline_reapply.py
```

### `preflight_returns_validation.py` — returns-ledger go-live evidence (#125)

READ-ONLY. Produces the evidence a clinician owner needs before flipping
`RETURNS_LEDGER_ENABLED` for a real run — reissue/partial-coverage prevalence,
administration-note recall, the no-non-returned-unit invariant, and a go/hold
summary. It never enables the feature and never changes pipeline behaviour.
`BBA_PREFLIGHT_BDVSTTRANS` defaults to the bundle's `BDVSTTRANS.csv` (else the
raw Bloodbank export).

```bash
uv run python scripts/pilot/preflight_returns_validation.py
# → $BBA_PILOT_WORK_DIR/preflight_returns_validation.json
```

### `reconcile_verdict_sources.py` — pre-swap cross-check for doctor ranking

The gate before Feature 2 (doctor/department ranking) swaps its input from the
human-review workbook to the pipeline's own verdicts. Reads BOTH sources and,
over the REQNOs they share, prints agreement plus the peri-op OVER-CLEAR count
(orders the pipeline cleared that the reviewer called inappropriate) — which
must be ~0 to trust the swap. Read-only and fail-loud (non-zero exit) if either
source is missing or the two share no REQNOs.

```bash
BBA_RUN_ID=pilot-mini \
BBA_AUDIT_STORE_DIR="$BBA_PILOT_WORK_DIR/data/audit_store" \
uv run python scripts/pilot/reconcile_verdict_sources.py
```

### `fix_reasoning_split.py` — repair tag-leaked reasoning in an old report

Retroactive fix for the 2026-07-06 `claude-sonnet-5` run, where the model
serialized both reasoning summaries into `reasoning_en` (separated by leaked
tool-call tags) and left `reasoning_th` empty. Applies the same split the
pipeline now does, plus the empty-reasoning guardrail (rows with no reasoning in
either language floor to `NEEDS_REVIEW` / `needs_human_review=True`). Preserves
`llm_report.json.bak`; re-running on a fixed report is a no-op. Run it before
rebuilding the review.

```bash
uv run python scripts/pilot/fix_reasoning_split.py   # uses $BBA_PILOT_WORK_DIR
uv run python scripts/pilot/build_review.py          # then regenerate review.html
```

### `notes_surgical_context.py` — surgical-context prototype (experimental)

PROTOTYPE, regex only, NO LLM, not wired into the pipeline. When the operative
tables (IPTSUMOPRT / INCPT) miss an order's index surgery, it recovers the
surgical context from nursing focus notes (IPDNRFOCUSDT) and computes a
notes-derived hours-to-surgery that could feed the same ≤72 h pre-op-crossmatch
bypass. Reads `report.csv` + `bundle/IPDNRFOCUSDT.csv` under the work dir.

```bash
uv run python scripts/pilot/notes_surgical_context.py        # summary + 7 target cases
uv run python scripts/pilot/notes_surgical_context.py --all  # every order
```

## Local-only PHI tools — do NOT publish or log

> **These two scripts print REAL PHI (raw HN / AN).** Run them only in your own
> terminal, never through an agent or CI step that logs output to a shared
> transcript, and never commit their output. They are intentionally kept off the
> public docs site. Both need `PHI_HMAC_KEY` — the same key `encrypt_phi.py`
> used to pseudonymize HN/AN as `PHI_<first 16 hex of HMAC-SHA256(key, value)>`.
> The truncated hash is one-way, so resolution is a forward scan: hash every raw
> HN/AN and match the tokens you ask for. Replace the placeholder paths below
> with your own local raw-export location.

### `reverse_lookup_phi.py` — resolve specific `PHI_*` tokens → HN/AN

Emits only the tokens you pass on the command line — no bulk dump.

```bash
export PHI_HMAC_KEY="$(cat <raw-parent-dir>/.phi_hmac_key)"
python scripts/pilot/reverse_lookup_phi.py \
    --raw-dir <path-to-raw-HOSxP-CSVs> \
    PHI_xxxxxxxxxxxxxxxx PHI_yyyyyyyyyyyyyyyy
```

### `html_decode_hn_an.py` — build a `(REQNO, HN, AN)` table from a review HTML

Extracts every case's `PHI_*` HN/AN tokens from a `review.html` and resolves
them to a CSV keyed by CaseNumber (= REQNO).

```bash
export PHI_HMAC_KEY="$(cat <raw-parent-dir>/.phi_hmac_key)"
python scripts/pilot/html_decode_hn_an.py \
    --html "$BBA_PILOT_WORK_DIR/review.html" \
    --raw-dir <path-to-raw-HOSxP-CSVs> \
    --out "$BBA_PILOT_WORK_DIR/hn_an_table.csv"
```
