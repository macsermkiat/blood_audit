# Pilot scripts — mini-bundle audit walkthrough

A four-step composition pipeline that runs the full audit end-to-end on
a small random sample drawn from an encrypted HOSxP bundle, then
materialises a single self-contained HTML for human clinical review.
Built for the M8 pilot-run milestone before clinical co-lead sign-off.

These scripts remain historical pilot examples outside the `bba.*` package. The supported application entry point is now `bba audit`. It composes canonical HOSxP inputs, deterministic decisions, trusted privacy admission, model review, and committed AuditRows. Use [the production CLI guide](../../docs/production-cli.md) for operator configuration and verification.

The pilot scripts retain dataset-specific sampling and review artifacts. They use the same privacy and request-identity boundaries when a model transport is selected.

## Pipeline

```
sample_bundle.py        →  bundle/        (11 sampled-down CSVs)
                        →  sample_manifest.csv

run_pipeline.py         →  report.csv     (deterministic verdict per issue event)
                        →  events_manifest.csv

run_llm_leg.py          →  llm_report.json
                        →  data/audit_store/   (Parquet + markers)

audit_llm_responses.py  →  llm_response_audit.{json,csv}
                        →  llm_response_audit_ids.txt
                                            (checks the LLM RESPONSES; exit 2 on a HIGH finding)

build_review.py         →  review.html    (single page for human review;
                                            refuses LLM verdicts without a fresh, clean audit)
```

## The audit unit is the issue event

Since #276 both legs judge one issue event per row, not one REQNO. An issue
event is the units one REQNO issued with no gap over 4 h and no new order for
the same admission and family in between. Each event is judged at its first
unit's issue time (`PAYDATE` + `PAYTIME`), against the order linked to it
(`own`, `fresh_order`, or `standing_order`), and credited to that order's
`DCTREQ`. `PICKTIME` is no longer an anchor. The
[issue-event design record](../../docs/issue-event-plan.md) gives the rules and
the measured scale. The [report schema](../../docs/report-schema.md#pilot-issue-event-outputs)
lists the per-event columns.

`scripts/pilot/_pilot_events.py` builds the events for both legs and holds
`EVENT_CODE_VERSION` (`pilot-event-1+gap4h+link-own`). Changing the 4 h
window, the linkage rule, or the event facts in the prompt needs a new base
token and a fresh `BBA_PILOT_RUN_ID`. Audit IDs from REQNO-level runs (v1, v2)
do not match event audit IDs; join them through `parent_audit_id`.

The bundle must include pending (status 1) orders in `BDVST_RELATED.csv`.
Without them, fresh orders would link as standing orders, so both legs exit.
Rebuild an older bundle with `sample_bundle.py`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `BBA_PILOT_RAW_DIR` | `../Bloodbank/data/encrypted` | Source HOSxP CSVs (11 files) |
| `BBA_PILOT_WORK_DIR` | `/tmp/bba_mini` | All outputs land here |
| `BBA_PILOT_SAMPLE_N` | `10` | Number of orders to sample |
| `BBA_PILOT_SAMPLE_SEED` | `20260519` | RNG seed for reproducibility |
| `BBA_PILOT_REQNO_FILE` | _(unset)_ | Text file with one REQNO per line (blank lines and lines starting with `#` are ignored). `sample_bundle.py` builds the bundle from exactly those RBC orders instead of a seeded draw, and ignores `BBA_PILOT_SAMPLE_N` / `BBA_PILOT_SAMPLE_SEED`. Stops without writing anything if a REQNO is missing, is not an eligible RBC order, matches more than one (HN, AN), or is listed twice. Use a fresh `BBA_PILOT_WORK_DIR`: a failed run leaves an earlier run's files in place. Use it to rebuild a reviewed sample: a seed reproduces a sample only on the exact BDVST snapshot it was drawn from |
| `BBA_PILOT_PLATELET_REQNO_FILE` | _(unset)_ | Same format as `BBA_PILOT_REQNO_FILE`, for platelet-ONLY orders: `sample_bundle.py` builds the platelet stream from exactly those orders and ignores `BBA_PILOT_PLATELET_SAMPLE_N` / `BBA_PILOT_PLATELET_SEED` (manifest seed = `list`). Stops without writing if a REQNO is not an eligible platelet-only order (a mixed RBC+platelet REQNO is rejected). Lets a doctor-restricted cohort carry its platelet orders |
| `BBA_PILOT_ICD10_CSV` | `../Bloodbank/data/raw/ICD10.csv` | ICD-10 master dictionary |
| `BBA_PILOT_LLM_MODEL` | `claude-sonnet-5-5` | Anthropic model id |
| `BBA_PILOT_RUN_ID` | `pilot-mini` | run_id stamped on audit_store rows |
| `BBA_PILOT_ENABLE_MISSING_HB_POSITIVE_EVIDENCE` | `false` | Opt-in to the missing-Hb MTP / peri-procedural auto-APPROPRIATE pre-check (SEED — set `1`/`true` only after clinical sign-off) |
| `BBA_PILOT_DECLARED_USETYPE` | the library flag `DECLARED_USETYPE_ENABLED` (on) | `1` forces on, any other value forces off. When on, both pilot legs thread BDVSTDT.USETYPE declared intent into the classifier and `report.csv` gets `declared_use_code` / `declared_use_label`. `compare_runs.py` reads `declared_use_code` to find standing-order events whose order declares use |
| `BBA_BDVST_CSV` | `../Bloodbank/data/encrypted/BDVST.csv` | BDVST export with DCTREQ; `run_pipeline.py` reads it only to fill `attributed_doctor_differs`, the rankings for attribution |
| `BBA_PILOT_ONLY_REQNO` | _(unset)_ | Comma-separated REQNOs: `run_llm_leg.py` processes/submits only every issue event of those REQNOs and MERGES the fresh records into the existing `llm_report.json` by `audit_id` (other events keep their records). Always pair with a fresh `BBA_PILOT_RUN_ID` — the store is idempotent on `(run_id, audit_id, code_version)`, so a reused run id keeps the stale row |
| `BBA_PILOT_ONLY_AUDIT_ID` | _(unset)_ | Comma-separated event `audit_id`s (the re-ask list): the same as `BBA_PILOT_ONLY_REQNO` for single issue events. An audit_id with no event in the bundle stops the run. The two filters combine as a union: an event runs when its REQNO is in `BBA_PILOT_ONLY_REQNO` OR its audit_id is in `BBA_PILOT_ONLY_AUDIT_ID`. Unset `BBA_PILOT_ONLY_REQNO` for an audit_id re-ask. An empty value is the same as unset, so an empty re-ask list runs every event |
| `BBA_PILOT_PRIVACY_PREPARE` | _(unset)_ | Path of a request file to write instead of submitting (prepare stage). See `issue_privacy_artifacts.py` for the prepare, issue, and llm order |
| `BBA_PILOT_ALLOW_UNAUDITED` | _(unset)_ | `1` lets `build_review.py` build a page from model responses whose audit is missing, stale, or has a HIGH finding. Setting it is the data owner's decision for each run |
| `BBA_PILOT_BATCH_MAX_WAIT` | `86400` | Seconds to wait for the Anthropic batch (default = the 24h batch SLA) |
| `BBA_PILOT_TRANSPORT` | `anthropic-batch` | `claude-cli` runs the LLM leg through the local `claude` CLI on the claude.ai subscription, one `claude -p --json-schema` call per case, answers checkpointed under `$BBA_PILOT_WORK_DIR/cli_batches/<batch_id>/` (resume with `BBA_PILOT_BATCH_ID`). For A/B runs that should not spend API credit; NOT the production path, rows carry `_transport: claude-cli`. No `ANTHROPIC_API_KEY` needed |
| `BBA_PILOT_CLI_WORKERS` | `3` | Concurrent `claude -p` processes when `BBA_PILOT_TRANSPORT=claude-cli` |
| `ANTHROPIC_API_KEY` | _(required for `anthropic-batch`)_ | Anthropic credentials |
| `BBA_PRIVACY_POLICY` | _(required before live submission)_ | Operator-controlled Ed25519 public-key policy and approved redactor identity |
| `BBA_PRIVACY_ARTIFACTS` | _(required before live submission)_ | Approved redacted evidence and signed attestations keyed by audit ID, bound to the exact prompt and run |
| `BBA_DATA_DIR` | _(required for `bba ingest`)_ | Where ingest writes Parquet + markers |
| `BBA_DB_URL` | _(required for `bba ingest`)_ | DB URL — placeholder is fine for ingest-only |

## End-to-end

```bash
uv sync --extra live                                 # in repo root, once
export ANTHROPIC_API_KEY=sk-ant-…
export BBA_PILOT_WORK_DIR=/tmp/bba_mini
export BBA_PRIVACY_POLICY=/path/to/trust-policy.json
export BBA_PRIVACY_ARTIFACTS=/path/to/approved-evidence.json

uv run python scripts/pilot/sample_bundle.py         # sample 10 cases
uv run python scripts/pilot/run_pipeline.py          # deterministic verdicts

# Optional: run the ingest leg via the CLI on the sampled bundle
export BBA_DATA_DIR="$BBA_PILOT_WORK_DIR/data"
export BBA_DB_URL=postgresql://localhost/bba_pilot
uv run bba ingest "$BBA_PILOT_WORK_DIR/bundle/BDVST.csv"

uv run --extra live python scripts/pilot/run_llm_leg.py # admitted Anthropic batch
uv run python scripts/pilot/audit_llm_responses.py   # audit the responses (code checks, no API)
uv run python scripts/pilot/build_review.py          # assemble review.html
open "$BBA_PILOT_WORK_DIR/review.html"
```

The model step requires artifacts issued by the approved local redaction
authority for this exact run and evidence. Missing or rejected proof routes to
local review before either API or subscription transport. Encryption alone is
insufficient. Follow [privacy admission](../../docs/production-cli.md#admit-model-evidence).

## Auditing the LLM responses before the page is built

`audit_llm_responses.py` runs between the LLM leg and `build_review.py`
(issue #239: a clinician found an answer labelled APPROPRIATE whose own
reasoning concluded INAPPROPRIATE on case 1 of a rendered page). It changes no
verdict and, by default, makes no API call: it runs the code checks (parse
failures, answers cut off at the output token limit, label vs the class the
reasoning appears to conclude, final verdict changed with no recorded reason,
label written before the reasoning, citations, trigger value, summary language)
and writes `llm_response_audit.{json,csv}` plus
`llm_response_audit_ids.txt`, the audit_ids of the issue events a re-run can
fix.

A record whose `review_reason` is `hallucination_suspect` is reported as a HIGH
`parse_failure`, like the other parse-failure reasons, and lands on the re-ask
list.

### Re-ask the flagged events

The loop: run the audit, fix the cause, re-ask the listed events under a fresh
run id, and run the audit again. A re-ask goes through the same three privacy
stages as the first pass, because the signed artifacts bind the run id.

1. Check that `llm_response_audit_ids.txt` is not empty. An empty
   `BBA_PILOT_ONLY_AUDIT_ID` means no filter, and the LLM leg would process
   every event.
2. Unset `BBA_PILOT_ONLY_REQNO`. The two filters combine as a union, so a
   leftover REQNO list re-asks every event of those REQNOs as well as the
   listed audit_ids.
3. Pick a fresh `BBA_PILOT_RUN_ID` and new file names for the request and
   artifact files, so the first pass's files stay in place. The month wrapper
   script calls this suffix `SFX` (for example `_r2`).
4. Run prepare, issue, and llm with the same run id, then audit again.

```bash
unset BBA_PILOT_ONLY_REQNO
export BBA_PILOT_ONLY_AUDIT_ID="$(cat "$BBA_PILOT_WORK_DIR/llm_response_audit_ids.txt")"
export BBA_PILOT_RUN_ID=pilot-mini-r2
SFX=_r2

# prepare: write the exact prompt requests (raw notes, mode 0600)
BBA_PILOT_PRIVACY_PREPARE="/secure/requests$SFX.json" \
uv run python scripts/pilot/run_llm_leg.py

# issue: redact and sign on the operator's machine (full command in issue_privacy_artifacts.py)
$ISSUER issue --requests "/secure/requests$SFX.json" ... --out "/secure/artifacts$SFX.json"

# llm: submit the admitted requests; fresh records merge into llm_report.json by audit_id
BBA_PRIVACY_POLICY=/secure/policy.json \
BBA_PRIVACY_ARTIFACTS="/secure/artifacts$SFX.json" \
uv run python scripts/pilot/run_llm_leg.py

uv run python scripts/pilot/audit_llm_responses.py
```

`build_review.py` exits with an error when `llm_report.json` holds model
responses and the audit is missing, read a different `llm_report.json` (the
audit stores the report's SHA-256, so re-audit after every re-run), or carries a
HIGH finding. Runs with no model responses need no audit.
`BBA_PILOT_ALLOW_UNAUDITED=1` builds the page anyway and prints the reason the
builder would have refused. Setting it is the data owner's decision for each
run, not an operator default.

### Dev-test consortium (Claude Code, not a pipeline step)

`dev_consortium/consortium.py`, `offline_reapply.py` and `fix_reasoning_split.py` predate issue events (#276) and still work per REQNO.

While a prompt, schema or evidence change is being tested, a consortium of
reviewers checks whether the answers the API returned are sound and correct. It
is NOT a production step and NOT a gate (user ruling 2026-09-21). It runs from a
Claude Code session on the subscription, so it makes no Anthropic API call.

```bash
# 1. review bundles from the audit store: evidence the model saw + its answer
uv run python scripts/pilot/dev_consortium/consortium.py build --reqno-file reqnos.txt
# 2. in Claude Code: one subagent per reviewer and batch (e.g. sonnet, opus,
#    haiku), each told to read dev_consortium/INSTRUCTIONS.md and write
#    dev_consortium/answers/<reviewer>/<reqno>.json
# 3. combine: code counts the votes, dissent is shown, nothing is averaged
uv run python scripts/pilot/dev_consortium/consortium.py combine
```

The bundles hold the de-identified notes the model saw. Sending them to a
reviewer outside Anthropic (for example the Codex CLI) is the data owner's
decision for each run. `audit_llm_responses.py --judge candidates|all` remains
as an opt-in API version of the narrow "which class does this summary
conclude?" check for unattended runs; it bills `ANTHROPIC_API_KEY`.

## Re-running a single case

To iterate on one case (e.g. after a prompt or guardrail change)
without paying for a full batch, filter the LLM leg under a fresh run id,
then rebuild the review. The fresh records merge into `llm_report.json` by
`audit_id`, and the other events keep their records.
`BBA_PILOT_ONLY_REQNO` re-runs every event of that REQNO;
`BBA_PILOT_ONLY_AUDIT_ID` re-runs one event. Set only one of them. Every
submission needs a signed privacy artifact for its run id. Without one, the
event is stored as a local privacy-failed row and no model call happens. Run
the prepare, issue, and llm stages as in the re-ask loop above. The block below
shows only the filter and run-id settings that the three stages share:

```bash
BBA_PILOT_ONLY_AUDIT_ID=<event audit_id> \
BBA_PILOT_RUN_ID=pilot-mini-one-event-v2 \
uv run python scripts/pilot/run_llm_leg.py

uv run python scripts/pilot/audit_llm_responses.py
uv run python scripts/pilot/build_review.py
```

## What the review page shows

`build_review.py` reads `report.csv`, `llm_report.json`, `sample_manifest.csv`,
and `events_manifest.csv`. It exits when `events_manifest.csv` is missing.

- **One case per issue event.** Cases follow the REQNO order of
  `sample_manifest.csv`, then `event_seq`. The header reads
  `Case N — REQNO … · event k of n`. Case numbers count events, so a v2 case
  number does not point at the same case on a v3 page.
- **REQNO-level cases.** A REQNO that `audit_orders` excluded, or that issued no
  unit (`no_issue_record`), keeps one case with no event label.
- **Unit timeline.** Every event case shows the REQNO's issued units: issue
  time, status, return time, the event each unit belongs to, and that event's
  linkage, linked REQNO, DCTREQ-differs flag, and declared-use flags. The
  timeline shows no DCTREQ code.
- **PICK is display only.** `PICKDATE` and `PICKTIME` stay in the BDVST table,
  labelled "not an anchor". Every evidence window reads from the event's issue
  time.
- **Joins fail loud.** The builder exits on a repeated `audit_id`, an event in
  `events_manifest.csv` with no `report.csv` row or the reverse, a sampled
  REQNO with neither events nor a REQNO-level row, or a timeline whose unit
  count disagrees with `issue_units`. Each means the files come from different
  runs.

The page shows more notes than the model saw. The model's bundle holds
`IPDNRFOCUSDT` notes charted in [issue - 24 h, issue + 24 h], at most 5 before
the issue and 5 at or after it (`WINDOW_FOCUS`, `CAP_FOCUS_BEFORE`,
`CAP_FOCUS_AFTER` in `src/bba/evidence_bundle_builder/builder.py`), and
`IPDADMPROGRESS` notes in the same 24 h window, at most 8. The page lists
notes by calendar day, from the day before the issue to the day after, with
no cap. Peri-operative note sections extend to two days after an operation
planned within 72 h of the issue. A note the reviewer sees on
the page may be absent from the model's evidence. Check the model's cited
evidence IDs before treating a missed note as a model error.

## Compare a v3 run with v2

`compare_runs.py` compares a REQNO-level run (v2) with an issue-event run (v3)
and prints counts only. It joins through `parent_audit_id`, computing the v2
side as `build_audit_id(HN, REQNO)` from the v2 `sample_manifest.csv`.

```bash
uv run python scripts/pilot/compare_runs.py <v2_dir> <v3_dir>
```

It reads `report.csv` and `llm_report.json` from both folders and
`sample_manifest.csv` from the v2 folder. It writes nothing into the v2 folder
and writes two files into the v3 folder.

- `compare_v2_v3.csv`: per v2 REQNO, the v2 final verdict and Hb, and the v3
  event verdicts, Hb values, and linkage tags.
- `compare_linkage_audit_ids.txt`: the re-ask list if the owner changes the
  linkage default. It holds the union of two sets of event audit_ids: events
  whose `linked_reqno_latest_before` is set and differs from `linked_reqno`,
  and `standing_order` events whose own order declares USETYPE 2 or 3
  (`declared_use_code`). The second set is empty when the declared-use seam
  is off, because `declared_use_code` is then absent.

Final verdicts merge as `rank_merged.py` merges them. A v2 row whose REQNO is
missing from the v2 `sample_manifest.csv` is skipped and only counted
(`v2_rows_without_manifest_hn`). The command exits 1 when any other v2 row maps
to no v3 row. That is expected for a subset run such as a gold
run, where most v2 rows lie outside the subset.

## Rank per event

`rank_merged.py red_cell` and `rank_merged.py platelet` merge `report.csv`
with `llm_report.json` by `audit_id` and rank one component at a time. Each
event counts once and is credited to the `DCTREQ` of its `linked_reqno`, so a
REQNO's later events can credit other doctors. `RETURNED_NOT_TRANSFUSED` and
`PERIOP_TRANSFUSION_EXEMPT` events are attributed and held out of the scored
denominator. REQNO-level `excluded` rows are not ranked.

`rank_doctors.py` defaults to `BBA_VERDICT_SOURCE=pipeline`, which reads one
run's red-cell rows from the audit store and joins them to `report.csv` by
`audit_id`. The store holds LLM-leg rows only, so prefer `rank_merged.py` for
complete denominators. `BBA_VERDICT_SOURCE=human` exits, because the review
workbook labels are per REQNO and the ranking is per event.

The doctor-profile tool kept outside this repository reads the same per-event
outputs. It needs this code (PR 8 or later) in the checkout that
`BBA_CODE_DIR` names.

Do not use `bba report` or `bba serve-dashboard` on a pilot audit store. They
attribute by REQNO and would credit fresh-order events to the wrong doctor
without an error ([module contracts](../../docs/CONTEXT.md#application-path-stays-per-reqno)).

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
  `src/bba/llm_client/models.py` (`claude-sonnet-5-5`, `claude-opus-5-5`)
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

### `preflight_*.py` — superseded

`preflight_declared_usetype.py` and `preflight_returns_validation.py` were
one-off per-REQNO go-live evidence for the declared-use (#151) and returns (#125)
seams. Since issue events (#276) they exit with a "superseded by issue events"
message; the last working versions are in git history.

### `reconcile_verdict_sources.py` — superseded

It compared the audit store's verdict per REQNO with the human-review
workbook's label per REQNO. Since issue events (#276) one REQNO can hold
several events with different verdicts, so the script exits with a
"superseded by issue events" message. The last working version is in git
history.

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
