#!/usr/bin/env bash
# Creates the 21 child issues for Phase 1 of blood_audit.
# Idempotent label creation; non-idempotent issue creation (run once).
# Requires: gh CLI authenticated.

set -euo pipefail

REPO="macsermkiat/blood_audit"

echo "===> Creating labels"
gh label create phase-1 --repo "$REPO" --color "5319e7" --description "Phase 1 RBC audit" 2>/dev/null || true
gh label create deep     --repo "$REPO" --color "0e8a16" --description "Deep module — TDD ≥80% unit coverage required" 2>/dev/null || true
gh label create thin     --repo "$REPO" --color "fbca04" --description "Thinner module — integration + golden-path E2E" 2>/dev/null || true
gh label create blocked  --repo "$REPO" --color "d93f0b" --description "Blocked on unresolved dependency or external" 2>/dev/null || true
gh label create prereq   --repo "$REPO" --color "bfd4f2" --description "Non-code prerequisite" 2>/dev/null || true

mk() {
  local title="$1"; shift
  local labels="$1"; shift
  local body="$1"; shift
  gh issue create --repo "$REPO" --title "$title" --label "$labels" --body "$body"
}

# ---- #2 — M0 prerequisite (manual, blocks #7) ----
mk "M0 — Request procedure / operative table export from KCMH IT" "phase-1,prereq,blocked" "$(cat <<'EOF'
Tracks the KCMH IT data-extension request that unblocks `bba.cohort_detector` (#7) and the peri-procedural 6h bypass rule in `bba.deterministic_classifier` (#8). This is a **manual ops task** — not a ralph-loop ticket.

## What we need

A new CSV in `data/encrypted/` linking admission numbers (`AN`) to procedure events with:

| Field | Type | Purpose |
|---|---|---|
| `HN` | varchar | Patient hospital number |
| `AN` | varchar | Admission number |
| `procedure_code` | varchar | ICD-10-PCS / ICD-9-CM Vol3 / local KCMH procedure code |
| `procedure_name` | varchar | Human-readable name |
| `procedure_datetime` | timestamp | When the procedure occurred (timezone Asia/Bangkok) |
| `or_or_ward_lct` | numeric | Location code (so we can distinguish OR vs ICU vs ward bedside) |
| `status` | varchar | scheduled / in-progress / completed / cancelled |

## Why

Per PRD #1 (Implementation Decisions, M5 `cohort_detector`): cardiac-surgery cohort threshold 7.5 g/dL and ortho+cardiac threshold 8.0 g/dL require deterministic detection from procedure data, not LLM inference. Without this table, both rules degrade to "cohort unknown → NEEDS_REVIEW," which pushes ~10–20% of orders to human review.

## Acceptance

- [ ] Request filed with KCMH IT
- [ ] Procedure-table CSV delivered to `data/encrypted/`
- [ ] Schema documented in `docs/data-dictionary.md` (one row per field, with source HOSxP table reference)
- [ ] Sample row sanity-checked against a known cardiac-surgery case from `BDVST.csv`
- [ ] Interim mitigation documented in `bba.cohort_detector` for cases where this table is unavailable (route to NEEDS_REVIEW)

## Definition of done

Close this issue once the CSV is in `data/encrypted/` and `docs/data-dictionary.md` is updated. **Do not start #7 until this is closed.**

References: PRD #1 → "Further Notes → Blocker: KCMH IT procedure / operative table export (E1.c)".
EOF
)"

# ---- #3 — bba.ingest ----
mk "bba.ingest — CSV → DuckDB+Parquet ingestion" "phase-1,deep" "$(cat <<'EOF'
Implements `bba.ingest` per PRD #1, Implementation Decisions §1.

## Scope

Ingests the 10 HOSxP CSVs in `data/encrypted/` into a DuckDB+Parquet store. Handles HOSxP quirks: time format variants (HHMMSS, HH:MM, decimal hour like `8.5`, Excel serial fractions, Buddhist-year prefixes, `0`/`9999`/`null` sentinels); tz `Asia/Bangkok` → UTC normalization; pandera schema validation with versioned `hosxp_schema_vN.py` modules; file-content hashing; `run_id = sha256(input_csv_hash + schema_fingerprint + code_version)`; idempotent re-runs (no-op if `run_id` already complete).

## Acceptance criteria

- [ ] `src/bba/ingest/` package with strict time parser, pandera schemas, DuckDB writer
- [ ] All 10 CSVs ingestable: BDVST, BDVSTDT, BDTYPE, BDVSTST, Diagnosis, Lab, MED, IPDADMPROGRESS, IPDNRFOCUSDT, UnUSE_Patient_Background
- [ ] Unrecognized time formats → `parse_warning` column (NULL value); never silently shifts
- [ ] All timestamps stored UTC; rendered Asia/Bangkok at boundaries
- [ ] Lint rule banning naive `datetime.now()`/`datetime.utcnow()` in this module
- [ ] `run_id` idempotency: same input → same `run_id` → no-op on re-run
- [ ] Property tests (hypothesis) on time parser covering: HHMMSS valid, HH:MM, decimal hour, Excel serial, Buddhist year, `0`, `9999`, `null`, empty, garbage
- [ ] Schema-drift detection: unknown column in CSV → fail loud with helpful error
- [ ] Coverage ≥ 80% on `src/bba/ingest/`
- [ ] `ruff check` + `mypy --strict` clean

## Dependencies

- None. Foundation for #4, #5, #6, #7, #9, #12.

## References

- PRD #1 → Implementation Decisions §1
- Round 2 fixes E29 (schema drift), E31 (tz), E32 (run_id idempotency), E35 (time parser allow-list)

Output `<promise>ISSUE-3-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #4 — bba.audit_orders ----
mk "bba.audit_orders — per-(HN, REQNO) RBC filter with hard-exclusions" "phase-1,deep" "$(cat <<'EOF'
Implements `bba.audit_orders` per PRD #1, Implementation Decisions §2.

## Scope

Produces the canonical `audit_orders` DuckDB table: one row per `(HN, REQNO)` for RBC orders (LPRC / LDPRC / SDR), with `BDVSTST ∈ {4, 5}`, `CANCELDATE IS NULL`, age ≥ 15, AN-scoped, `REQTYPE = 'P'`. Hard-excludes hemoglobinopathy (D55, D56, D57, D58), AIHA (D59.x), TMA cohorts. Anchor datetime = `REQDATE + REQTIME` with `BDVSTDATE + BDVSTTIME` fallback flagged.

## Acceptance criteria

- [ ] Implementation in `src/bba/audit_orders/`
- [ ] Golden-fixture tests per excluded subgroup: pediatric (<15), OB, OPD (AN=null), inter-hospital (REQTYPE='H'), hemoglobinopathy (each of D55/D56/D57/D58), AIHA (D59.x), TMA, refused (BDVSTST=6), cancelled (CANCELDATE not null)
- [ ] Anchor-imputation flag emitted when REQDATE/REQTIME is null
- [ ] Output schema matches PRD §"Output schema" identity + anchor fields
- [ ] Coverage ≥ 80%
- [ ] `ruff check` + `mypy --strict` clean

## Dependencies

- Depends on #3 (must be closed)

## References

- PRD #1 → Implementation Decisions §2
- Round 1 B1 (hemoglobinopathy hard-exclude); Round 2 clinical agent (G6PD D55 inclusion)

Output `<promise>ISSUE-4-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #5 — bba.hb_lookup ----
mk "bba.hb_lookup — most-recent Hb with freshness + tiered delta-Hb" "phase-1,deep" "$(cat <<'EOF'
Implements `bba.hb_lookup` per PRD #1, Implementation Decisions §3.

## Scope

Returns the most-recent Hb before the order anchor. `LABEXM 290095` (HEMATOLOGY Hemoglobin) preferred; `LABEXM 500001` (POCT) fallback. Numeric validation [2, 25] g/dL. Freshness tiers: `<24h fresh`, `24-72h stale_24_72h`, `72h-7d stale_3_7d`, `>7d missing`. **Tiered delta-Hb trigger** (per Round 2 E3): drop ≥ 1.5 g/dL in 6h, ≥ 2 g/dL in 12h, or ≥ 2.5 g/dL in 24h → flags `delta_hb_bypass = True` for `bba.deterministic_classifier`.

## Acceptance criteria

- [ ] Implementation in `src/bba/hb_lookup/`
- [ ] Freshness tier boundary tests
- [ ] LABEXM source preference test (HEMATOLOGY beats POCT when both in window)
- [ ] Numeric-range validation rejects out-of-bound and non-numeric `RESULT` strings
- [ ] Multi-Hb tie-breaking (same datetime → highest ITEMNO wins)
- [ ] Tiered delta-Hb trigger: ≥1.5/6h, ≥2/12h, ≥2.5/24h; all three thresholds tested
- [ ] Single-Hb case (no prior in window) flagged for NEEDS_REVIEW if Hb < 8 (per Round 2 clinical)
- [ ] Property tests on numeric parsing (hypothesis)
- [ ] Coverage ≥ 80%; ruff + mypy clean

## Dependencies

- Depends on #3

## References

- PRD #1 → Implementation Decisions §3
- Round 1 B3 (delta-Hb); Round 2 clinical (tiered timings)

Output `<promise>ISSUE-5-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #6 — bba.vitals_extractor ----
mk "bba.vitals_extractor — regex + LLM fallback over IPDADMPROGRESS / IPDNRFOCUSDT" "phase-1,deep" "$(cat <<'EOF'
Implements `bba.vitals_extractor` per PRD #1, Implementation Decisions §4.

## Scope

Extracts vital signs (SBP, DBP, HR, RR, BT) from free-text columns within ±6h of the order anchor. Sources: `IPDADMPROGRESS.OBJECTIVE` (preferred — cleaner SOAP format), `IPDNRFOCUSDT` FOCUS/ACTION/RESPONSE (fresher but noisier). Regex-first; LLM fallback only when regex finds nothing. Sanity bounds: SBP 60–220, DBP 30–150, HR 30–200, RR 5–50, BT 30–43.

## Acceptance criteria

- [ ] Implementation in `src/bba/vitals_extractor/`
- [ ] Regex patterns cover Thai + English variations: `BP 110/60`, `BP:118/63`, `PR108`, `HR 97`, `P 14`, `RR 20-23`, `BT 38.4`, `Temp 37`
- [ ] Sanity-bound enforcement; out-of-bound values discarded with `vitals_data_error` flag
- [ ] ±6h window selection from anchor; most-recent-before-order preferred; fallback to most-recent-within-+6h with `vitals_post_order` flag
- [ ] Source provenance recorded: `IPDADMPROGRESS` | `IPDNRFOCUSDT` | `LLM_extracted` | `none_in_window`
- [ ] Property tests for regex robustness against fuzzy formatting (hypothesis)
- [ ] LLM fallback boundary: only invoked when regex returns null SBP or HR
- [ ] Coverage ≥ 80%; ruff + mypy clean

## Dependencies

- Depends on #3

## References

- PRD #1 → Implementation Decisions §4
- Skill: `regex-vs-llm-structured-text` (regex first, LLM only on regex failure)

Output `<promise>ISSUE-6-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #7 — bba.cohort_detector ----
mk "bba.cohort_detector — deterministic cohort + threshold assignment" "phase-1,deep,blocked" "$(cat <<'EOF'
Implements `bba.cohort_detector` per PRD #1, Implementation Decisions §5.

## ⚠️ BLOCKED on #2

Cannot start until #2 (KCMH IT procedure-table export) is closed. Interim mitigation path is documented here but the full implementation depends on procedure data.

## Scope

Deterministic cohort assignment based on procedure codes + ward + diagnoses + active medications. Returns `(cohort_label, cohort_threshold)`:

| Cohort | Threshold | Detection signal |
|---|---|---|
| `cardiac_surgery` | 7.5 | Procedure code in cardiac-sx allow-list within 30 days |
| `ortho_cardiac` | 8.0 | Ortho procedure code + ICD-10 cardiac history (I20–I25, I50) |
| `esrd_epo` | 8.0 + symptom-required | ICD-10 N18.5/N18.6 + dialysis meds (heparin-for-HD, sevelamer, cinacalcet) |
| `mtp` | auto-APPROPRIATE bypass | ≥4 RBC units within 1h OR co-ordered with FFP + platelets in same window |
| `heme_malignancy_active` | T2 supportive (not a hard threshold) | ICD-10 C8x–C9x + recent chemo meds + ANC<500 |
| `default` | 7.0 | None of the above |

If procedure data unavailable, fall back to `cohort_unknown` → route to NEEDS_REVIEW (do not silently apply default 7.0).

## Acceptance criteria

- [ ] Implementation in `src/bba/cohort_detector/`
- [ ] One golden-fixture test per cohort class + one no-cohort case
- [ ] MTP detection: temporal-cluster rule tested at boundary (3 units vs 4 units; just-over 1h vs just-under)
- [ ] ESRD-EPO: requires both ICD-10 AND dialysis-med signal
- [ ] Cohort threshold returned as numeric (not enum); downstream classifier uses it directly
- [ ] Fallback to `cohort_unknown` when procedure data missing — explicit, never silent
- [ ] Coverage ≥ 80%; ruff + mypy clean

## Dependencies

- Depends on #3 (ingest)
- BLOCKED on #2 (procedure-table data)

## References

- PRD #1 → Implementation Decisions §5
- Round 1 C1 (deterministic, not LLM-judged); Round 2 N1 (ESRD), N2 (MTP), N3 (heme nadir)

Output `<promise>ISSUE-7-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #8 — bba.deterministic_classifier ----
mk "bba.deterministic_classifier — Hb-tier rule engine" "phase-1,deep" "$(cat <<'EOF'
Implements `bba.deterministic_classifier` per PRD #1, Implementation Decisions §6.

## Scope

The core deterministic rule engine. Given an audit row + Hb result + vitals + cohort + delta-Hb + procedure-proximity + crystalloid balance, returns `(rule_classification, bypass_reason)`:

- `Hb < cohort_threshold` → `APPROPRIATE`
- `cohort_threshold ≤ Hb < 10` → `NEEDS_LLM_REVIEW`
- `Hb ≥ 10` → `POTENTIALLY_INAPPROPRIATE` (LLM may override later)
- Hb missing → `INSUFFICIENT_EVIDENCE`
- Delta-Hb trigger fires → `APPROPRIATE` (`bypass_reason=delta_hb`)
- Procedure ≤ 6h before order → `APPROPRIATE` (`bypass_reason=peri_procedural`)
- MTP cohort → `APPROPRIATE` (`bypass_reason=mtp`)
- ≥ 2L crystalloid in prior 4h AND Hb < 7 → `NEEDS_REVIEW` (not auto-APPROPRIATE; hemodilution flag)
- B2 invariant: documentation absence MUST NOT produce `INAPPROPRIATE`. Positive evidence required for `INAPPROPRIATE`.

## Acceptance criteria

- [ ] Implementation in `src/bba/deterministic_classifier/`
- [ ] Exhaustive fixture table covering each (Hb-tier × cohort × bypass) combination
- [ ] B2 invariant test: a row with missing notes and Hb=8 must classify as `INSUFFICIENT_EVIDENCE` or `NEEDS_LLM_REVIEW`, never `INAPPROPRIATE`
- [ ] Hemodilution flag test: Hb=6.5 with ≥2L crystalloid in 4h → `NEEDS_REVIEW`, not `APPROPRIATE`
- [ ] Bypass-reason field populated and distinct per pathway
- [ ] Property test: monotonicity — increasing Hb never moves classification from `INAPPROPRIATE` toward `APPROPRIATE` (within same cohort)
- [ ] Coverage ≥ 80%; ruff + mypy clean

## Dependencies

- Depends on #4, #5, #6, #7

## References

- PRD #1 → Implementation Decisions §6
- Round 1 B2 (positive-evidence requirement), B3 (delta-Hb), B4 (peri-procedural), B5 (hemodilution); Round 2 E3 (tiered delta)

Output `<promise>ISSUE-8-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #9 — bba.evidence_bundle_builder ----
mk "bba.evidence_bundle_builder — note ranking + canonical JSON" "phase-1,deep" "$(cat <<'EOF'
Implements `bba.evidence_bundle_builder` per PRD #1, Implementation Decisions §7.

## Scope

Builds the structured JSON evidence bundle sent to the LLM. Per-source windows: Diagnosis (AN-scoped); IPDADMPROGRESS (±24h, cap 8 entries); IPDNRFOCUSDT (±24h, cap 10 entries — 5 before, 5 after order); MED (−72h / +24h); Lab Hb history (−7d); vitals (±6h). Stable evidence IDs (`E1`, `E2`, …). Canonical-JSON serialization (sorted keys, UTF-8 NFC, fixed indentation) → SHA-256 bundle hash.

## Acceptance criteria

- [ ] Implementation in `src/bba/evidence_bundle_builder/`
- [ ] Per-source window enforcement tested
- [ ] IPDADMPROGRESS section priority on truncation: ASSESSMENT + PLAN first, OBJECTIVE next, SUBJECTIVE last
- [ ] IPDNRFOCUSDT time-proximity ranking: closest to order time first; 5+5 split
- [ ] Stable evidence IDs (E1...EN) assigned deterministically
- [ ] Canonical JSON serialization: same input → byte-identical output → same hash
- [ ] Hash-stability property test: input reorderings produce same hash
- [ ] Cap enforcement: bundles never exceed token-budget proxy (8K char target)
- [ ] Coverage ≥ 80%; ruff + mypy clean

## Dependencies

- Depends on #3, #5, #6

## References

- PRD #1 → Implementation Decisions §7
- Round 2 E37 (extended-thinking storage policy belongs in #12, not here)

Output `<promise>ISSUE-9-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #10 — bba.deid_redactor ----
mk "bba.deid_redactor — thai-medical-deid wrapper with role tokens + k-anonymity" "phase-1,deep" "$(cat <<'EOF'
Implements `bba.deid_redactor` per PRD #1, Implementation Decisions §8.

## Scope

Pinned-version wrapper around `macsermkiat/thai-medical-deid`. Emits role-preserving tokens `[ATTENDING]` / `[NURSE]` / `[PATIENT]` / `[FAMILY]` (post-processing wrapper, not a fork of the redactor). k-anonymity gate (k ≥ 5 on `{ward, ICD-3char, age-band, sex, admission-month}`) — failing bundles route to NEEDS_REVIEW. Date-shift to relative offsets (Δ-days-from-admission). Age cap at 89. Semantic-degradation flag: if redacted note has > 4 `[PERSON]`-class tokens within 50 chars → NEEDS_REVIEW.

## Acceptance criteria

- [ ] Implementation in `src/bba/deid_redactor/`
- [ ] `thai-medical-deid==X.Y.Z` pinned in `pyproject.toml`; vendored model SHA + gazetteer version recorded
- [ ] Role-token wrapper tested on a hand-labeled corpus of ≥ 30 redacted notes
- [ ] k-anonymity gate enforced: bundles with quasi-identifier combos k<5 are routed away
- [ ] Date-shift correctness: dates inside notes remapped to Δ-days from admission
- [ ] Age cap 89 enforced
- [ ] Semantic-degradation flag fires at the documented threshold
- [ ] Bundle-hash stability test: same input bundle + same redactor version → same redacted output → same hash
- [ ] Coverage ≥ 80%; ruff + mypy clean

## Dependencies

- Depends on #9

## References

- PRD #1 → Implementation Decisions §8
- Round 1 C3 (version pin), C8 (k-anonymity), C12 (role tokens); Round 2 E18 (post-processing wrapper)
- External dep: github.com/macsermkiat/thai-medical-deid

Output `<promise>ISSUE-10-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #11 — bba.quote_grounder ----
mk "bba.quote_grounder — anti-hallucination verifier (6 layers)" "phase-1,deep" "$(cat <<'EOF'
Implements `bba.quote_grounder` per PRD #1, Implementation Decisions §9.

## Scope

The hallucination defense. Given an LLM output + the redacted evidence bundle, verifies every cited `evidence_quote` against the cited source. Six layers, all must pass:

1. **NFC normalization** on both sides (Thai NFD/NFC mixing)
2. **Contiguous substring match** in the cited source's redacted text
3. **Strict `cited_id` match** (not "any source in bundle")
4. **Within-document uniqueness** (quote does not appear in >1 evidence item from the same source-doc); scope to non-boilerplate spans
5. **Minimum character length ≥ 25** (Thai shorthand-aware, not 8-token)
6. **Numeric-tuple grounding** for lab citations: extract `(analyte, value, unit)` from both sides and match on tuple equality
7. **Medical-NLI semantic-entailment gate** (optional — use medical-domain NLI when available; XNLI-Thai general-domain is not adequate per Round 2 E8)

Failure path: each indication is `verified` or `rejected` with a reason. If ALL Tier-1 citations are rejected → caller retries/escalates.

## Acceptance criteria

- [ ] Implementation in `src/bba/quote_grounder/`
- [ ] All six layers tested separately + in combination
- [ ] Adversarial fixture set: NFC vs NFD, numeric paraphrase (`Hgb=8.3` vs `Hb 8.3 g/dL`), cross-source attribution, concatenated quotes (`a ... b`), short common phrases (`no bleeding` cited in unrelated context)
- [ ] Verifier-as-classifier: produce confusion-matrix output on 200 hand-labeled verdicts for #13 to consume
- [ ] Pure function (no I/O); zero dependencies on Anthropic SDK
- [ ] Property tests: any random insertion/deletion of 1 character in a valid quote → rejection
- [ ] Coverage ≥ 80%; ruff + mypy clean

## Dependencies

- None (pure function — can be developed in parallel with anything)

## References

- PRD #1 → Implementation Decisions §9
- Round 1 C9 (six layers); Round 2 E13 (custom_id mapping happens in #15), E14 (NFC), E15 (numeric tuple)

Output `<promise>ISSUE-11-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #12 — bba.audit_store ----
mk "bba.audit_store — append-only Parquet + transactional ordering" "phase-1,deep" "$(cat <<'EOF'
Implements `bba.audit_store` per PRD #1, Implementation Decisions §10.

## Scope

Persistence layer. Append-only Parquet for `audit_results` + `llm_calls`. Transactional ordering: write `llm_calls` first (safe append), then `audit_results` last as the **commit marker**. Reconciliation job finds orphaned `llm_calls` without a downstream `audit_results` row and re-emits or quarantines. `run_id` idempotency (writing a row with an existing `run_id` partition is a no-op unless `--force`). DuckDB read-only snapshot view (daily-rotated) for the dashboard so in-flight writes don't produce inconsistent reads. Cold-storage migration for Opus extended-thinking blocks after 90 days.

## Acceptance criteria

- [ ] Implementation in `src/bba/audit_store/`
- [ ] Write/read round-trip tested with the full audit-row schema from PRD §"Output schema"
- [ ] Transactional-ordering invariant: an `audit_results` row without a matching `llm_calls` row is a bug; test forces the failure mode and confirms reconciliation
- [ ] Idempotent re-run: same `run_id` writes once, returns cached result on second call
- [ ] Snapshot-view consistency: dashboard reads from snapshot N don't observe in-flight writes for snapshot N+1
- [ ] Cold-storage policy stub for extended-thinking blocks (actual migration can be Phase 1.5)
- [ ] Coverage ≥ 80%; ruff + mypy clean

## Dependencies

- Depends on #3

## References

- PRD #1 → Implementation Decisions §10
- Round 1 C2 (full response persist), C4 (split storage); Round 2 E30 (transactional boundary), E32 (run_id), E37 (cold storage)

Output `<promise>ISSUE-12-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #13 — bba.eval_harness ----
mk "bba.eval_harness — stratified gold standard + statistical metrics" "phase-1,deep" "$(cat <<'EOF'
Implements `bba.eval_harness` per PRD #1, Implementation Decisions §11.

## Scope

Stratified sampling with enrichment (target ~138 INAPPROPRIATE-positives per stratum); reviewer assignment + adjudication; metrics computation with Wilson CIs, cluster-robust SE (clustered on physician + ward), Horvitz-Thompson prevalence reweighting; per-stratum κ + Gwet's AC1 + PABAK; LOMO-CV when dataset < 12 months else blocked temporal split; pre-registered subgroup analysis with Bonferroni on primary endpoints + BH-FDR on exploratory subgroups; verifier-as-classifier confusion-matrix consumer from #11; outcome-anchored falsification analysis (24h Hb response, transfusion reactions, 30d mortality, "% flagged cases where physician amended documentation") on labeled subset.

## Acceptance criteria

- [ ] Implementation in `src/bba/eval_harness/`
- [ ] Sampling correctness: stratified-with-enrichment produces target n per stratum
- [ ] Reweighting math: Horvitz-Thompson against known references
- [ ] Wilson CI computation against scipy reference
- [ ] κ + Gwet's AC1 + PABAK against published example tables
- [ ] Cluster-robust SE on synthetic clustered data (correct vs naive comparison)
- [ ] LOMO-CV implementation; auto-fallback to blocked temporal if ≥ 12 months
- [ ] Subgroup-test multiple correction implemented (hierarchical FDR)
- [ ] Coverage ≥ 80%; ruff + mypy clean

## Dependencies

- Depends on #12

## References

- PRD #1 → Implementation Decisions §11
- Round 1 D1–D7; Round 2 E19–E27

Output `<promise>ISSUE-13-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #14 — bba.prompt_builder ----
mk "bba.prompt_builder — system + few-shot + payload assembly" "phase-1,thin" "$(cat <<'EOF'
Implements `bba.prompt_builder` per PRD #1, Implementation Decisions §12.

## Scope

Assembles system prompt + few-shot block + per-row user payload. Task-mode switch (`HB_7_10_REVIEW` / `HB_GT_10_OVERRIDE`). Injects `cohort_threshold` as a hard numeric input. Wraps each evidence chunk in `<evidence id="E1" untrusted="true">…</evidence>`. Pre-LLM injection scanner (regex + small classifier) for imperative verbs, fake-guideline patterns, bilingual Thai/EN jailbreaks; flagged rows route to NEEDS_REVIEW without LLM call. Prompt-caching markers on the system + few-shot block.

## Acceptance criteria

- [ ] Implementation in `src/bba/prompt_builder/`
- [ ] Task-mode switching tested
- [ ] Injection scanner test set: ≥20 known adversarial patterns (direct, indirect, Thai-EN code-switched, fake-guideline)
- [ ] Prompt-cache marker correctness verified (system + few-shot is single cached block)
- [ ] Integration test: a benign bundle round-trips through; an adversarial bundle is short-circuited
- [ ] Coverage ≥ 70% (thinner module); ruff + mypy clean

## Dependencies

- Depends on #10

## References

- PRD #1 → Implementation Decisions §12
- Round 1 C10 (injection defense); Round 2 E10 (additional injection coverage)

Output `<promise>ISSUE-14-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #15 — bba.llm_client ----
mk "bba.llm_client — Anthropic Batch API wrapper with escalation" "phase-1,thin" "$(cat <<'EOF'
Implements `bba.llm_client` per PRD #1, Implementation Decisions §13.

## Scope

Anthropic Batch API integration. Snapshot-pinned model IDs (e.g., `claude-sonnet-4-6-20XXXXXX`). `custom_id == audit_id` assertion on every result — never positional zip. Prompt caching engaged. Structured-output (tool-use) JSON shape with **fail-closed parsing** (malformed → NEEDS_REVIEW with `parse_failure` flag). Retry → Sonnet (≤ 2x) → escalate to Opus 4.7. Sonnet/Opus classification-disagreement detection → routes to NEEDS_REVIEW. Persists FULL response object (input msgs, output, usage, response headers including `anthropic-version` build identifiers) to `llm_calls` table via #12.

## Acceptance criteria

- [ ] Implementation in `src/bba/llm_client/`
- [ ] `custom_id` assertion: if any returned result's `custom_id` mismatches expected `audit_id`, abort batch with explicit error
- [ ] Betamax/VCR cassettes for offline replay of Anthropic API
- [ ] Retry → escalation test: simulate Sonnet failure twice → Opus invocation
- [ ] Sonnet/Opus disagreement test: synthetic responses with different classifications → NEEDS_REVIEW
- [ ] Malformed-JSON fail-closed test: garbage output → NEEDS_REVIEW with `parse_failure`
- [ ] Full response persistence: every call writes to `llm_calls` via `bba.audit_store`
- [ ] Coverage ≥ 70%; ruff + mypy clean

## Dependencies

- Depends on #14

## References

- PRD #1 → Implementation Decisions §13
- Round 1 C2 (persist response), C13 (fail-closed); Round 2 E13 (custom_id), E17 (disagreement)
- Skill: `claude-api` (prompt caching guidance)

Output `<promise>ISSUE-15-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #16 — bba.confidence_calibrator ----
mk "bba.confidence_calibrator — isotonic calibration + agreement-based confidence" "phase-1,thin" "$(cat <<'EOF'
Implements `bba.confidence_calibrator` per PRD #1, Implementation Decisions §14.

## Scope

Maps raw LLM-reported confidence to calibrated P(correct) via isotonic regression fit on the training-split predictions vs gold-standard labels. Optional **agreement-based confidence**: run Sonnet 3× with reshuffled few-shot ordering; `confidence = fraction agreeing`. Monthly recalibration check via ECE (Expected Calibration Error) on held-out 200.

## Acceptance criteria

- [ ] Implementation in `src/bba/confidence_calibrator/`
- [ ] Isotonic fit math verified against scikit-learn reference
- [ ] ECE computation tested against published example
- [ ] Agreement-based confidence: deterministic seed control for the 3× shuffle so testing is reproducible
- [ ] Calibration plot generator (reliability diagram) — output to docs/eval/
- [ ] Coverage ≥ 70%; ruff + mypy clean

## Dependencies

- Depends on #15

## References

- PRD #1 → Implementation Decisions §14
- Round 1 C11; Round 2 E10 (calibration set size)

Output `<promise>ISSUE-16-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #17 — bba.audit_pipeline ----
mk "bba.audit_pipeline — orchestration + row-level checkpointing" "phase-1,thin" "$(cat <<'EOF'
Implements `bba.audit_pipeline` per PRD #1, Implementation Decisions §15.

## Scope

Per-audit-row orchestration: deterministic engine → bundle → redact → LLM → verify → retry/escalate → persist. Row-level checkpointing in `batch_runs` with state machine (`pending` → `submitted` → `partial` → `complete` | `failed`). Resume on startup: reconciles any `submitted` batches by polling Anthropic, persists completed-but-unpersisted results. Unique constraint on `(audit_id, attempt_id)`; deterministic winning-attempt rule (last verifier-passed wins).

## Acceptance criteria

- [ ] Implementation in `src/bba/audit_pipeline/`
- [ ] State-machine transitions tested
- [ ] Resume scenario: kill process mid-batch, restart, verify no duplicate writes and no lost work
- [ ] Replay test: applying the same batch result twice is a no-op (property test)
- [ ] End-to-end smoke test: tiny synthetic dataset flows through full pipeline → audit_results table
- [ ] Coverage ≥ 70%; ruff + mypy clean

## Dependencies

- Depends on #8, #9, #10, #15, #11, #12

## References

- PRD #1 → Implementation Decisions §15
- Round 1 C5 (checkpointing); Round 2 E34 (retry dedup)

Output `<promise>ISSUE-17-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #18 — bba.review_actions ----
mk "bba.review_actions — Postgres mutable state + PHI-access log" "phase-1,thin" "$(cat <<'EOF'
Implements `bba.review_actions` per PRD #1, Implementation Decisions §16.

## Scope

Postgres-backed mutable state (chosen over SQLite for write-lock-free concurrency under multi-reviewer load). Append-only tables: `review_actions` (agree / override(reason) / escalate / use-as-few-shot-candidate) and `phi_access_log` (reviewer_id, timestamp, hn_hash, an_hash, break-glass justification). Immutable insert pattern (no updates, no deletes).

## Acceptance criteria

- [ ] Implementation in `src/bba/review_actions/`
- [ ] Append-only invariant enforced at DB level (revoke UPDATE/DELETE; trigger guard)
- [ ] PHI-access log completeness: every dashboard access of un-redacted text writes a row
- [ ] Concurrent-write test: 5 simulated reviewers, no lock contention
- [ ] Schema migration scripts in `migrations/` (alembic or equivalent)
- [ ] Coverage ≥ 70%; ruff + mypy clean

## Dependencies

- Depends on #12

## References

- PRD #1 → Implementation Decisions §16
- Round 1 C4 (split storage); Round 2 E28 (Postgres over SQLite)

Output `<promise>ISSUE-18-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #19 — bba.dashboard ----
mk "bba.dashboard — FastAPI + HTMX reviewer UI" "phase-1,thin" "$(cat <<'EOF'
Implements `bba.dashboard` per PRD #1, Implementation Decisions §17.

## Scope

FastAPI + HTMX + Tailwind reviewer dashboard. Views: NEEDS_REVIEW queue (sortable), case-detail view (un-redacted with care-team-of-record gating + break-glass justification capture), per-ward scorecard, per-physician own-view scorecard, pipeline-health dashboard. Every un-redacted access writes to `phi_access_log` (via #18). Daily DuckDB snapshot view for `audit_results` reads — no in-flight inconsistency.

## Acceptance criteria

- [ ] Implementation in `src/bba/dashboard/`
- [ ] All five views render with synthetic test data
- [ ] Break-glass flow: viewing un-redacted requires justification text → written to `phi_access_log`
- [ ] Per-physician own-view: route guard enforces requesting physician = subject physician
- [ ] DuckDB snapshot read path tested (no inconsistency under concurrent batch write)
- [ ] Smoke tests for all routes
- [ ] Coverage ≥ 60% (UI module); ruff + mypy clean

## Dependencies

- Depends on #12, #18

## References

- PRD #1 → Implementation Decisions §17
- Round 1 C4; Round 2 E28 (Postgres for mutable)

Output `<promise>ISSUE-19-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #20 — bba.monitoring ----
mk "bba.monitoring — SPRT/CUSUM drift detection + sentinel" "phase-1,thin" "$(cat <<'EOF'
Implements `bba.monitoring` per PRD #1, Implementation Decisions §18.

## Scope

Weekly 50–75-case clinical-reviewer sample. SPRT / O'Brien-Fleming α-spending on quote-grounding failure rate and NEEDS_REVIEW rate (target ARL₀ ≥ 500 → ≈ 1 false alarm/year). Quarterly 100-row golden-set re-run on pinned model snapshot for drift probe. Weekly intra-model κ on a fixed 200-case sentinel (alert if κ < 0.90).

## Acceptance criteria

- [ ] Implementation in `src/bba/monitoring/`
- [ ] SPRT trigger correctness: synthetic drift injected at known offset → alarm fires at expected sample
- [ ] Sentinel set construction is deterministic (fixed seed) so weekly κ is comparable
- [ ] Alerting integration stub (log + structured output; Slack/email integration optional Phase 1.5)
- [ ] Coverage ≥ 70%; ruff + mypy clean

## Dependencies

- Depends on #13, #17

## References

- PRD #1 → Implementation Decisions §18
- Round 1 D7; Round 2 E22 (α-spending)

Output `<promise>ISSUE-20-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #21 — bba.report_generator ----
mk "bba.report_generator — monthly CSV + PDF" "phase-1,thin" "$(cat <<'EOF'
Implements `bba.report_generator` per PRD #1, Implementation Decisions §19.

## Scope

Monthly CSV + PDF aggregations. Sections: hospital-wide trend; per-ward scorecard; per-physician own-view (own-data only); indication-distribution breakdown; cohort-exception breakdown; pipeline-health summary. Distribution: email link / SSO web portal. All reports include `policy_version`, `model_id`, `redactor_version` footer.

## Acceptance criteria

- [ ] Implementation in `src/bba/report_generator/`
- [ ] Snapshot-based template tests (golden output for fixed input)
- [ ] CSV schema documented in `docs/report-schema.md`
- [ ] PDF rendering tested with a sample run
- [ ] Coverage ≥ 60% (templating module); ruff + mypy clean

## Dependencies

- Depends on #12

## References

- PRD #1 → Implementation Decisions §19

Output `<promise>ISSUE-21-COMPLETE</promise>` when all criteria met.
EOF
)"

# ---- #22 — bba.cli ----
mk "bba.cli — click entrypoints + run-level idempotency" "phase-1,thin" "$(cat <<'EOF'
Implements `bba.cli` per PRD #1, Implementation Decisions §20.

## Scope

`click` CLI entrypoints: `bba ingest`, `bba audit`, `bba evaluate`, `bba report`, `bba serve-dashboard`, `bba sentinel`. Run-level idempotency: `bba audit` computes `run_id = sha256(input_csv_hash + schema_fingerprint + code_version)`; if `run_id` partition exists and is `complete` → no-op unless `--force`. Structured logging via `structlog`. Process-level exception scrubbing (`sys.excepthook` + `faulthandler` redirect into a redaction sidecar) so uncaught exceptions never leak `bundle` into journald.

## Acceptance criteria

- [ ] Implementation in `src/bba/cli.py`
- [ ] All 6 subcommands invokeable with `--help` returning usable text
- [ ] Run-level idempotency: `bba audit` twice on the same input is a no-op (verified)
- [ ] `--force` flag overrides idempotency with audit-log entry
- [ ] Exception scrubbing: an injected uncaught exception with `bundle` in scope produces a scrubbed traceback in logs
- [ ] Coverage ≥ 60% (orchestration glue); ruff + mypy clean

## Dependencies

- Depends on #17, #13, #21, #19, #20

## References

- PRD #1 → Implementation Decisions §20
- Round 2 E32 (run_id idempotency), E39 (sys.excepthook)

Output `<promise>ISSUE-22-COMPLETE</promise>` when all criteria met.
EOF
)"

echo ""
echo "===> All 21 child issues created. View at: https://github.com/$REPO/issues"
