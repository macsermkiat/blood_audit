# Missing-Hb elective cases: periop pre-pass + LLM deferral (NOT window extension)

> Handoff plan. Repo branch `feat/missing-hb-positive-evidence-bypass`. Data: `/tmp/bba_mini/`.

## Context

In `/tmp/bba_mini/review.html`, 39/300 cases are flagged `INSUFFICIENT_EVIDENCE` because Hb is "missing." The hypothesis was that these are elective surgeries whose pre-op Hb fell *outside* the lookback window, so extending the Hb window would fix them.

**The data contradicts that hypothesis.** Tracing all 39 back to the raw `Lab.csv`:

| Where the patient's Hb actually sits | Cases |
|---|---|
| Only Hb is drawn **after** the transfusion (post-tx check, e.g. 13.0 / 14.9 / 16.3 g/dL) | 29 |
| **No Hb anywhere** in the dataset | 7 |
| Pre-tx Hb exists but **30 days** before order (stale) | 1 |
| Pre-tx Hb exists, **re-anchor edge interaction** | 2 |

So the missingness is **real, not a windowing artifact**. A backward extension catches ~1 stale case; a forward extension would feed the deterministic gate a *post-transfusion* Hb (inflated by the transfusion itself) — a hindsight-bias trap that would mislabel. **Window extension is the wrong lever.**

The real situation: 27/39 are elective reserve-ahead orders with surgery ≤72h. The classifier already has a `PRE_OP_CROSSMATCH` bypass for exactly this, but it is **deliberately suppressed on missing Hb** (`src/bba/deterministic_classifier/classifier.py:182-191`, rationale: "transfusing pre-op with no Hb is what an audit should flag"). And critically: **a deterministic `INSUFFICIENT_EVIDENCE` is terminal** — `DETERMINISTIC_FINAL = {APPROPRIATE, INSUFFICIENT_EVIDENCE, INAPPROPRIATE}` (`scripts/pilot/run_llm_leg.py:1328`), so these 39 **never reach the LLM** and never benefit from the peri-op evidence extractor already built (`bba.vitals_extractor.periop`). When cases *do* reach the LLM it auto-resolves 61% with no human (33/54, mostly APPROPRIATE at median confidence 0.80).

**Goal (user, clinical lead):** maximum automation / least human review, with maximum accuracy. Chosen approach: **deterministic pre-pass, then LLM** — auto-approve only on *hard* documented peri-op evidence; defer everything else to the LLM (which auto-resolves the well-documented majority and floors the rest). Aligns with the project's clinical-correctness preference (judge the actual transfusion event with real context, not reservation-time conservatism).

## Supporting evidence (commands already run against /tmp/bba_mini)
- All 39 missing-Hb cases → `INSUFFICIENT_EVIDENCE`, `rationale="hb_missing"`, `bypass_reason="none"` in `report.csv`.
- Join validated: 216/216 `fresh` cases have their HN in `Lab.csv` (key = PHI_ token via `sample_manifest.csv` REQNO→HN).
- 34/39 have `upcoming_procedure_hours`; 27/39 have surgery ≤72h.
- `llm_report.json` has only 54 rows (deterministic NEEDS_REVIEW=50 + POTENTIALLY_INAPPROPRIATE=4). **0/39 missing cases reached the LLM.** Of the 54, 33 auto-resolved (needs_human_review=False), 21 → human.
- `replay.py:431`: `needs_human_review = final_classification == "NEEDS_REVIEW"` (so LLM NEEDS_REVIEW = human floor; LLM APPROPRIATE/INAPPROPRIATE/INSUFFICIENT resolve automatically).

## Design decision

For a **missing-Hb** order, replace the terminal `INSUFFICIENT_EVIDENCE` with a two-stage policy:

1. **Deterministic pre-pass (skips the LLM):** auto-classify `APPROPRIATE` **only** when *hard* peri-op evidence is present — `intraop_transfusion == True` OR `blood_loss_ml >= PERIOP_GUARDRAIL_MIN_EBL_ML` (500). Documented-blood-loss / documented-intra-op-transfusion cases where the surgical indication is unambiguous.
   - **Not** `surgical_context` alone and **not** `upcoming_procedure ≤72h` alone — those are the weak "surgery is scheduled" signals the design already refused to rubber-stamp on missing Hb. They defer (stage 2).
2. **Defer to LLM (route, don't dead-end):** every other missing-Hb case becomes `NEEDS_REVIEW` (rationale `hb_missing_defer_llm`) so the existing router sends it to the LLM **with** the peri-op block + op-day Hb context. The LLM reads the free-text prose (often holds blood-loss the structured row lacks), and:
   - clear surgical indication → `APPROPRIATE` (auto, grounded in cited evidence),
   - genuinely ambiguous → `NEEDS_REVIEW` → **human floor**,
   - truly no documentation → `INSUFFICIENT_EVIDENCE` (auto-recorded documentation gap).

**Accuracy invariant (unchanged):** the deterministic gate Hb value stays **backward-only**. A post-transfusion Hb is never the decision value; op-day/post-op Hb reaches the LLM as clearly-labeled *context* only.

## Implementation components

### 1. Thread peri-op hard signals into `ClassifierInputs`
- `src/bba/deterministic_classifier/models.py`: add to `ClassifierInputs` (frozen): `periop_blood_loss_ml: int | None = None`, `periop_intraop_transfusion: bool = False`, `periop_surgical_context: bool = False` (last for traceability). Document semantics mirroring existing fields.
- Reuse `bba.vitals_extractor.periop.scan_periop` (pure; returns `PeriopSummary{surgical_context, blood_loss_ml, intraop_transfusion, findings}`). Do **not** re-implement extraction.

### 2. New deterministic branch (`src/bba/deterministic_classifier/classifier.py`)
In the `hb.value_g_dl is None` block (lines 109-139), gated behind existing `enable_missing_hb_positive_evidence`, **after** MTP + peri-procedural≤6h and **before** the terminal return:
- New `BypassReason.PERIOP_EVIDENCE` (or reuse `PRE_OP_CROSSMATCH`) + rationale `bypass_periop_evidence_hb_missing`.
- Fire `APPROPRIATE` iff `periop_intraop_transfusion` OR (`periop_blood_loss_ml is not None and periop_blood_loss_ml >= PERIOP_GUARDRAIL_MIN_EBL_ML`).
- Terminal branch changes: instead of `INSUFFICIENT_EVIDENCE`, return `NEEDS_REVIEW` + rationale `hb_missing_defer_llm` so the router defers to the LLM.

### 3. Populate the new inputs at every classify site (keep the legs unified)
Three `ClassifierInputs` construction sites must all pass the peri-op signals (the legs-must-stay-unified rule):
- `scripts/pilot/run_pipeline.py:657` — **currently loads no notes**; add IPDNRFOCUSDT/`VitalsNote` loading + `scan_periop` ahead of `classify`. Factor the note-loading in `run_llm_leg._vitals_notes_for` (line 470) into a shared helper to avoid divergence.
- `scripts/pilot/run_llm_leg.py:1332` — periop already computed for the bundle (`bundle.periop_summary`); compute/scan before `classify` and pass it in.
- `src/bba/audit_pipeline/pipeline.py:_classifier_inputs_for` (line 530) — production path. Classifier runs **before** the bundle, so either compute `scan_periop` in the input builder from notes carried on `PipelineRowContext`, or reorder so the periop scan precedes classify. This is the generalizable (multicenter/production) source of truth; the pilot scripts mirror it.

### 4. LLM context for deferred cases (secondary / optional toggle)
Peri-op block already reaches the LLM (EXEMPT_FROM_DROP). Optionally surface op-day/post-op Hb for now-deferred non-re-anchored cases, **explicitly labeled "POST-transfusion"** so the model does not misread it as the trigger. Default conservative: rely on the peri-op block; treat op-day-Hb context as a follow-up toggle (risk: high post-tx Hb could bias toward INAPPROPRIATE).

### 5. Enable the flag for the pilot run
`ENABLE_MISSING_HB_POSITIVE_EVIDENCE` defaults `False` ("SEED pending clinical sign-off"). Clinical lead is signing off → set the env var on for the pilot re-run; document in `docs/CONTEXT.md`. Keep it a flag for multicenter rollout control.

## Files to modify
- `src/bba/deterministic_classifier/models.py` — new `ClassifierInputs` peri-op fields, new `BypassReason` member, docstrings.
- `src/bba/deterministic_classifier/classifier.py` — pre-pass branch + terminal `NEEDS_REVIEW` deferral; precedence docstring update.
- `src/bba/audit_pipeline/pipeline.py` — `_classifier_inputs_for` populates peri-op signals (production source of truth).
- `scripts/pilot/run_pipeline.py` + `scripts/pilot/run_llm_leg.py` — shared note-load + `scan_periop` ahead of classify; pass new inputs.
- (Maybe) shared constants module for `PERIOP_GUARDRAIL_MIN_EBL_ML` (currently `src/bba/audit_pipeline/replay.py:82`).
- Tests (TDD, write first): classifier branch table; update existing missing-Hb assertions.

## Verification
1. **Unit (TDD):** (intraop_tx | EBL≥500) + flag on → APPROPRIATE/`bypass_periop_evidence_hb_missing`; surgical_context-only / upcoming-only / empty → NEEDS_REVIEW/`hb_missing_defer_llm`; flag off → unchanged INSUFFICIENT_EVIDENCE. `mypy` clean; full suite (2035 baseline) green.
2. **Deterministic leg:** re-run `scripts/pilot/run_pipeline.py` on `/tmp/bba_mini/bundle` with flag on → in `report.csv` the 39 missing-Hb cases split into a small auto-APPROPRIATE set (hard periop evidence) + a NEEDS_REVIEW (defer) set; **zero** remain terminal `INSUFFICIENT_EVIDENCE`.
3. **LLM leg (billable, needs API key):** re-run `scripts/pilot/run_llm_leg.py` → deferred cases reach the LLM; well-documented majority auto-resolve to APPROPRIATE; only ambiguous land `NEEDS_REVIEW` (human). Compare human-review count before/after.
4. **Spot-check** reqnos 68069997, 68056191, 68062324, 68054627, 68068127, 68009325 end-to-end in regenerated `review.html`.

## Risks / open items
- **Heavily-tested pure classifier:** missing-Hb → INSUFFICIENT is locked by many tests; changing the terminal verdict is a deliberate behavior change — update tests to encode the new intent (why), not just outcomes.
- **`PERIOP_GUARDRAIL_MIN_EBL_ML` duplication:** classifier auto-approve bar and `replay.py` escalation bar should share one constant or they will drift.
- **2 re-anchor edge cases** (68049423: order→tx 14d gap re-anchored the 7-day window *past* the only pre-tx Hb; 68074507: tx "00:00:00" midnight/time-missing) are a **separate** small fix, not part of this change.
- **op-day-Hb-as-context (Component 4)** left as a follow-up toggle pending user call, given post-tx-Hb bias risk.
- **LLM leg is billable + needs API key**; regenerating report.csv/review.html does NOT re-run the LLM — verdicts lag until `run_llm_leg.py` is re-run.
