# Applying the Chula DRAFT transfusion policies to blood_audit

**Status:** Phase 1 IMPLEMENTED — PR #83, deploy-gated on the two §4 clinician
rulings (CR-H1, CR-M1). Phase 2 PLANNED (re-scoped per §8), not built.
**Reviews:** grill + adversarial passes complete 2026-07-07 (§8); verification
pass 2026-07-08 (§8, third subsection).
**Author:** drafted 2026-07-07
**Source policies:** `../Bloodbank/policy/recommendation/*.docx` (4 DRAFT guidelines,
Chulalongkorn Hospital)

---

## 1. Source material

Four DRAFT department guidelines (near-duplicates; policy 3 is the surgical
superset, policy 1/medicine is the outlier with rich platelet special-populations):

| # | Department | Distinctive RBC rule |
|---|------------|----------------------|
| 1 | Internal medicine (อายุรกรรม) | platelet special-populations; no surgical Hb rows |
| 2 | Orthopedic surgery | ortho surgery → 8 |
| 3 | General surgery | cardiac surgery → 7.5, ortho surgery → 8 |
| 4 | OB/GYN surgery (สูติกรรม) | curative-intent chemo → 8–10 |

All four are **DRAFT** and carry the disclaimer that thresholds are advisory
and clinician-adjustable. Treated here as **SEEDs pending clinical sign-off**,
consistent with every existing allow-list in the codebase.

## 2. Gap analysis — the app already encodes most RBC thresholds

The app is an **RBC-only** appropriateness auditor (Hb-tier + clinical cohort;
platelets/FFP appear only as MTP co-order *signals*, never audited).

| Policy rule (RBC) | Policy | App today | Status |
|---|---|---|---|
| Stable inpatient (universal) | 7 | `DEFAULT` = 7.0 | ✅ match |
| Heart-disease comorbidity | 8–10 | `CARDIOPULMONARY_COMORBIDITY` = 8.0 | ✅ match |
| Cardiac **surgery** | 7.5 | `CARDIAC_SURGERY` = 7.5 | ✅ match |
| Orthopedic **surgery** | 8 | `ORTHO_CARDIAC` = 8.0 **requires ortho AND cardiac history** | ⚠️ partial |
| Curative-intent chemotherapy | 8–10 | none (heme cohort defers to LLM) | ❌ missing |

Only **two** genuine RBC gaps. The entire platelet/FFP/PCC/Cryo side is a new
audit surface (the app audits no non-RBC component).

## 3. Resolved scope (decision tree, grilled 2026-07-07)

1. **Phased, not one deliverable.** Phase 1 (RBC gaps) ships first and
   independently; Phase 2 (platelet) is planned now but built after.
2. **Ortho split** — additive-deprecate: add emitted `ORTHO_SURGERY` (8.0,
   ortho-alone); keep `ORTHO_CARDIAC` in the enum + threshold map, deprecated
   and never emitted. (Verified safe: `cohort_applied` is persisted as a plain
   `str` and never coerced back into `CohortLabel`, so a rename would not
   crash — additive is chosen for eval-diff legibility + codebase precedent.)
3. **Ortho precedence** — `ORTHO_SURGERY` keeps `ORTHO_CARDIAC`'s slot (above
   `CARDIAC_SURGERY`), preserving "higher floor wins" when a patient has both
   an ortho and a cardiac operation. Pre-existing convention; flagged for
   sign-off, not silently changed.
4. **Curative-intent chemo → Phase 2 (LLM).** NOT a Phase 1 deterministic
   cohort: "curative intent" is not in structured data, and the only
   deterministic proxy ("any chemo → 8.0") would **over-clear** palliative-chemo
   transfusions at Hb ∈ [7,8) — the exact dangerous false-clear direction the
   pilot review flagged. Belongs with the LLM prompt/guardrail work.
5. **Phase 2 = platelet only.** FFP (no lab gate — pure context), PCC (policy
   says "ยังไม่สามารถเบิกจ่ายได้" / not yet reimbursable → likely not in use),
   and Cryo (narrow DIC/massive-transfusion fibrinogen edge) are out of Phase 2.
6. **Platelet architecture = thin deterministic gate + LLM for all context.**
7. **Extreme cutoffs — SUPERSEDED by §8/CR-C1, do NOT implement as written.**
   ~~plt < 10 → `APPROPRIATE`~~; plt ≥ 100 → `POTENTIALLY_INAPPROPRIATE`.
   The plt < 10 auto-clear was removed as a patient-safety defect (dengue-no-bleed
   / TTP / HIT / ITP / aplastic populations withhold platelets at very low
   counts). v1 auto-clears NOTHING — the live gate is §5.1.
8. **Shared pipeline, component dispatch** — platelet orders are first-class
   audit units routed by component family; RBC path stays byte-identical.
9. **Additive, component-tagged sampling** with a separate `PLATELET_SAMPLE_N`
   — the seed-04062025 RBC cohort is untouched.

---

## 4. Phase 1 — Ortho cohort split (deterministic, low-risk)

### Behavior change
An orthopedic operation raises the RBC floor to **8.0 on its own**, with no
cardiac-history requirement. Effect: an ortho-surgery patient without heart
disease moves from `DEFAULT` (7.0) to `ORTHO_SURGERY` (8.0) — Hb ∈ [7,8) flips
from `NEEDS_REVIEW` to auto-`APPROPRIATE` for those cases.

### Files
- `src/bba/cohort_detector/models.py` — add `ORTHO_SURGERY = "ortho_surgery"`
  to `CohortLabel`; document `ORTHO_CARDIAC` as deprecated. (9 members.)
- `src/bba/cohort_detector/rules.py` — add `ORTHO_SURGERY_THRESHOLD = 8.0`;
  add `CohortLabel.ORTHO_SURGERY` to `COHORT_THRESHOLDS`; mark
  `ORTHO_CARDIAC_THRESHOLD` deprecated (kept). Update `__all__`.
- `src/bba/cohort_detector/detector.py` — replace the `ortho_event AND
  cardiac_history → ORTHO_CARDIAC` block with `ortho_event is not None →
  ORTHO_SURGERY` at the same precedence slot (above `CARDIAC_SURGERY`); drop the
  now-unused `find_cardiac_history_diagnosis` import (function stays in rules.py,
  still exported/tested). Update precedence docstring.

### Tests (`tests/unit/test_cohort_detector.py`)
- `TestPublicAPI`: 8 → 9 members; add `ORTHO_SURGERY` to the name list.
- `TestThresholdNumericContract`: add `(ORTHO_SURGERY, 8.0)`; keep
  `(ORTHO_CARDIAC, 8.0)`.
- `TestThresholdMapTotality`: unchanged (both in map).
- Rewrite `TestCohortOrthoCardiac` → `TestCohortOrthoSurgery` encoding NEW
  intent: ortho-alone → `ORTHO_SURGERY` 8.0; ortho+cardiac-history →
  `ORTHO_SURGERY` 8.0; cardiac-history-alone → `CARDIOPULMONARY_COMORBIDITY`.
- `TestCohortPrecedence.test_ortho_cardiac_preferred_over_plain_cardiac` →
  ortho+cardiac-surgery both present → `ORTHO_SURGERY` 8.0 wins over 7.5.
- Add a regression test asserting `ORTHO_CARDIAC` is **never emitted** by
  `assign_cohort` (deprecation tripwire).
- Check `test_deterministic_classifier.py` / `test_prompt_builder.py` for
  `ORTHO_CARDIAC` references and update.

### Open clinical questions BEFORE Phase 1 build (from review — §8/CR-H1, CR-M1)
- **Pre-op vs post-op scope (CR-H1).** The policy says "ผู้ป่วยที่**จะ**ผ่าตัด"
  — *about to* operate (pre-op optimization target). But `find_recent_ortho_surgery`
  matches surgery ALREADY DONE in `[anchor-30d, anchor]`. So the policy's actual
  beneficiary (upcoming ortho op) hits `preop_defer_llm` (NEEDS_REVIEW), while a
  patient 2–3 wk post-op gets an 8.0 auto-clear. Removing the cardiac-history
  gate also drops the qualifier that happened to align 8.0 with its FOCUS/Carson
  evidence base (hip-fracture patients *with CV disease*). **Needs a clinician
  ruling: does ortho-8.0 apply pre-op, post-op (current 30-d window), or both —
  and unconditionally or only with a CV-risk qualifier?** This can change the
  implementation, so resolve before coding.
- **Dual ortho+cardiac-surgery precedence (CR-M1).** "Higher floor wins" (8.0
  over 7.5) over-clears the [7.5,8.0) band for a population cardiac surgery
  manages restrictively (TITRe2). Get an explicit clinician ruling, not an
  inherited convention.

### Acceptance
- Full unit suite green; `ORTHO_CARDIAC` never emitted by `assign_cohort`
  (deprecation tripwire test) — load-bearing since `find_cardiac_history_diagnosis`
  + `CARDIAC_HISTORY_ICD10_PREFIXES` go newly dead (§8/AR-M7).
- Re-run the seed-04062025 pilot; diff `report.csv` cohort column. Count BOTH
  DEFAULT→ORTHO_SURGERY (ortho-alone, no CV disease) AND the
  ortho+cardiac-surgery 7.5→8.0 shift (§8/AR-M7) — the former is the H1 over-clear
  subset the clinician needs to see quantified.

---

## 5. Phase 2 — Platelet appropriateness auditor (new subsystem)

### 5.1 Deterministic gate (REVISED after clinical review — see §8/CR-C1)

> **The original `plt < 10 → APPROPRIATE` auto-clear is REMOVED. It was a
> patient-safety defect** (§8/CR-C1): `APPROPRIATE` is a terminal
> deterministic-final class (`pipeline.py:68`) — no LLM, no human — but the
> policies explicitly WITHHOLD platelets at very low counts for dengue-no-bleed,
> TTP, HIT, ITP/autoimmune, aplastic-no-bleed, chronic marrow failure, and
> snakebite-no-bleed (TTP/HIT: transfusion is actively harmful). In a Thai
> dengue-endemic dataset this would systematically false-clear inappropriate
> transfusions. Unlike Hb<7 (≈no contraindication population), plt<10 has 6+
> named exclusions — the "Hb<7 analog" was invalid.

On a platelet order with count `plt` (×10³/µL, LABEXM 290078):
- **`plt ≥ 100` → `POTENTIALLY_INAPPROPRIATE`** — routes to LLM/review (NOT
  terminal). Safe because reviewed; the ">100k head-bleed don't-give" rule
  agrees. Add a hard test that platelet `POTENTIALLY_INAPPROPRIATE` never
  becomes deterministic-final (§8/CR-M2).
- **Everything else (`plt < 100`, incl. `plt < 10`) → `NEEDS_REVIEW` → LLM**,
  which reads notes for indication (prophylaxis / LP / CVC / major surgery /
  active bleeding / dept) AND the exclusion populations, then applies the
  policy threshold. **Nothing is deterministically auto-cleared** in v1 —
  platelet appropriateness is context-dependent and the deterministic layer
  cannot safely clear it.
- `plt` missing → mirror the RBC missing-Hb contract (defer per the opt-in flag).

Optional v2 (post-sign-off): a *conditional* low-count auto-clear — `plt < 10`
→ `APPROPRIATE` ONLY when a positive prophylactic indication is documented
(chemo/HSCT/consumptive) AND no TTP/HIT/ITP/dengue/snakebite-no-bleed/
aplastic-no-bleed exclusion — but "documented indication/exclusion" is free-text,
so v1 keeps this in the LLM leg.

### 5.2 Context deferred to the LLM (NOT deterministic)
All of these are free-text / dept-dependent and route through `NEEDS_REVIEW`:
LP 50k vs 80k (dept not in structured data), CVC 50k, major non-neuraxial
surgery 80k/100k, cardiac-surgery-incl-CPB "don't give", head-bleed >100k,
consumptive thrombocytopenia 10k, prophylaxis-not-indicated conditions
(aplastic anemia, chronic marrow failure, ITP, HIT, TTP), dengue, snakebite.
These become **LLM prompt rules**, encoded with the RBC threshold rules.

### 5.3 Build stages (each its own PR — REWRITTEN 2026-07-08 per §8 required revisions)

**Gating prerequisite (§8/AR-M8):** the `BDTYPE → component family` map
(red-cell / platelet / ffp / cryo, from the `GRPCAUSELABCBC` / ORDERCODE
grouping) must exist and be verified against the bundle dictionary BEFORE any
stage below starts. It is not an open item inside stage 1; nothing dispatches
without it.

1. **Intake + store schema foundations** (§8/AR-C1, AR-C2, AR-H6) — the seam
   originates at `audit_orders`, NOT `audit_pipeline`: a new inclusion path
   alongside the `RBC_PRODUCTS` allow-list (`rules.py:27`) with a widened
   product type (the `RBCProduct` Literal rejects platelet products today).
   Re-derive the store schema with a `component` axis: `AuditRow` is Hb-shaped
   (required non-null `hb_value`/`hb_datetime`/`cohort_threshold`), so
   `component` is a schema re-derive, not a field add; thread the component
   filter through `aggregate.py` / `builder.py` / `csv_writer.py` / dashboard
   so platelet rows never blend into RBC stats. RBC path byte-identical.
2. **Generic lookup core** (§8/AR-H5) — extract the shared engine from
   `hb_lookup` (source preference, freshness tiers, trend/staleness)
   parameterized by component config; platelet config = LABEXM 290078,
   unit ×10³/µL, its own validity range and staleness window. NOT a mirror:
   `hb_lookup`'s [2, 25] g/dL validation (`parse.py`, `models.py`) rejects
   every platelet count.
3. **Platelet classifier** — the §5.1 gate. Rationale slugs: `plt_ge_100`,
   `plt_defer_llm`, `plt_missing` (no `plt_lt_10` slug — the auto-clear is
   removed). Includes the CR-M2 invariant test: no platelet verdict is ever
   deterministic-final in v1 (`POTENTIALLY_INAPPROPRIATE` and `NEEDS_REVIEW`
   both route onward). MTP interaction (§8/CR-M3, AR-M9): platelet units
   inside an active MTP window are suppressed as independent audit units —
   the MTP co-order stays a *signal*, never a second audit row.
4. **Pipeline dispatch** (§8/AR-C3) — component branches in
   `_classifier_inputs_for` / `_deterministic_audit_row` / persist /
   submission builders, in BOTH `run_pipeline` and `replay.py` (the logic is
   duplicated); the raise-on-Hb=None path must not trip on platelet rows.
   Dispatch lands only after stages 1–3 exist (§8/AR-H6: stage order was
   inverted in the original plan).
5. **Platelet LLM subsystem** (§8/AR-H4, CR-M4, CR-C2) — new `TaskMode` +
   platelet prompt template + bypass of the Hb-only
   `ALLOWED_COHORT_THRESHOLDS` gate (`system_prompt.py:128`). Platelet
   evidence bundle: count trend + procedure/bleeding context. Prompt encodes
   the §5.2 rules incl. the "expected drop <10k in 24h" pre-emptive
   prophylaxis clause (CR-M4). Over-clear guardrail hardened as the RBC B1
   guardrail — **with NO blanket plt<10 exemption** (CR-C2: a blanket
   exemption would let an LLM over-clear of a TTP/HIT/dengue patient at
   plt<10 stand); any exemption must be conditional on a documented positive
   indication.
   **OPEN DECISION (verification pass 2026-07-08): B1 guardrail composition.**
   `llm_overclear_suspect` (`replay.py:237`) floors any LLM `APPROPRIATE`
   over a deterministic `NEEDS_REVIEW` unless a structured hard signal fires
   — and every hard signal is RBC-shaped (Hb < 7, hard peri-op, MTP,
   SBP/HR). Since the v1 platelet gate emits `NEEDS_REVIEW` for ALL
   plt < 100, reusing the guardrail unchanged means v1 can never finalize
   `APPROPRIATE` below 100k — only the plt ≥ 100 branch could clear. Before
   this stage: either define platelet-specific structured hard signals, or
   explicitly accept that v1's LLM leg cannot clear sub-100k counts
   (everything lands in human review). Do not decide silently.
6. **Sampling** — extend `sample_bundle.py` to sample platelet-family orders
   as a component-tagged stream; `BBA_PILOT_PLATELET_SAMPLE_N` (~100), own
   seed; RBC sampling byte-identical.

### 5.4 Open items to verify at Phase 2 kickoff
- Platelet order **volume** in the raw data (raw `BDVSTDT` column order differs
  from the bundle projection — recount with the correct `BDTYPE` column).
- Verify **LABEXM 290078** (platelet count, ×10³/µL) against the bundle's lab
  dictionary before wiring the lookup config.
- Confirm the platelet-count staleness window clinically (platelets move faster
  than Hb post-transfusion; the RBC re-anchor logic may need a tighter window).
- Confirm `plt ≥ 100` ceiling vs the 80–100k high-bleeding-risk surgical rule
  (is 100 the right ceiling, or should 80–100 also defer?).
- Resolve the **B1-guardrail composition decision** (§5.3 stage 5): platelet
  hard signals vs. accept-no-sub-100k-clears — a user/clinician call, not an
  implementation detail.

---

## 6. Out of scope (explicit)
- **FFP, PCC, Cryo auditing** — possible Phase 3; FFP/Cryo are LLM-context
  problems, PCC is flagged non-reimbursable.
- **Curative-intent chemo deterministic cohort** — intentionally deferred to the
  Phase 2 LLM prompt work (§3.4).
- **Department-aware thresholds** — department is not in the structured feed;
  dept-specific splits (LP 50k/80k) are resolved by the LLM from notes, not code.

## 7. Clinical sign-off gates
Every new threshold (ortho-alone 8.0; platelet 10/100; all §5.2 LLM rules) is a
SEED frozen before scoring, pending clinician sign-off — mirroring the existing
allow-list policy. No new threshold is represented as production-final.

## 8. Review (adversarial pass complete — 2026-07-07)

Two independent reviewers. **Net verdict: Phase 1 proceed AFTER two clinician
rulings; Phase 2 re-scoped — the plt<10 auto-clear is removed and the seam moves
up to `audit_orders`.**

### Clinical safety (healthcare-reviewer) — verdict BLOCK on Phase 2 as-was
- **CR-C1 (CRITICAL) — `plt<10 → APPROPRIATE` is an unsafe terminal auto-clear.**
  Contradicts §5.2's own deferral list: dengue-no-bleed, TTP, HIT, ITP, aplastic-
  no-bleed, chronic marrow failure, snakebite-no-bleed all withhold platelets at
  <10k (TTP/HIT: transfusion harmful). **FIXED in §5.1** — plt<10 now routes to
  NEEDS_REVIEW/LLM; nothing platelet auto-clears in v1. *This reverses the Q6/Q7
  grill decision "10k appropriate" — flagged to user for confirmation.*
- **CR-C2 (CRITICAL) — the plt<10 guardrail exemption repeats C1.** FIXED in
  §5.3.5 — no blanket plt<10 exemption; any exemption conditional on a documented
  indication.
- **CR-H1 (HIGH) — ortho 8.0 pre-op/post-op scope inversion.** See Phase 1 open
  questions. Policy = pre-op target; app = post-op 30-d window; removing the
  cardiac gate over-generalizes 8.0 beyond its CV-disease evidence base.
- **CR-M1** dual-surgery precedence needs explicit sign-off. **CR-M2** add a test
  that platelet `POTENTIALLY_INAPPROPRIATE` never becomes deterministic-final.
  **CR-M3** carry the MTP co-order bypass into the platelet path. **CR-M4** encode
  the "expected drop <10k in 24h" pre-emptive clause in the platelet prompt.
- **Affirmed:** the curative-chemo deferral (§3.4) is clinically sound; plt≥100→
  review and the FFP/PCC/Cryo out-of-scope calls are reasonable.

### Architecture (architect) — Phase 1 sound; Phase 2 seam underscoped
- **AR-C1 (CRITICAL)** — platelet orders are rejected before the pipeline by
  `audit_orders` (`RBC_PRODUCTS` allow-list + `RBCProduct` Literal type). The seam
  must originate at `audit_orders` (new inclusion path, widened product type), NOT
  in `audit_pipeline`. **Re-scopes §5.3 stage 1.**
- **AR-C2 (CRITICAL)** — the store `AuditRow` is Hb-shaped (required non-null
  `hb_value`/`cohort_threshold`/…); `component` is a schema re-derive, not an add.
  Report aggregation groups by `cohort_applied` with no component axis → platelet
  rows blend into RBC stats unless a `component` filter is threaded through
  `aggregate.py`/`builder.py`/`csv_writer.py`/dashboard.
- **AR-C3 (CRITICAL)** — `run_pipeline` calls the single Hb `classify()`
  unconditionally and `_deterministic_audit_row` RAISES on Hb=None outside the
  bypass set — a platelet row trips it. Needs component branches in
  `_classifier_inputs_for`/`_deterministic_audit_row`/`_persist…`/`_build_submission…`
  (duplicated in `replay.py`).
- **AR-H4** the LLM leg is Hb-templated (`TaskMode` literal, Hb prompts,
  `ALLOWED_COHORT_THRESHOLDS` gate) — platelets need a new TaskMode + template +
  gate bypass. **AR-H5** `hb_lookup` is Hb-specific (value range 2–25 rejects
  platelet counts; g/dL delta specs) — reframe `platelet_lookup` as a shared
  generic core + component config, NOT a mirror. **AR-H6** stage order inverted:
  component map + `audit_orders` widening + store schema must PRECEDE dispatch.
- **AR-M7** additive-deprecate `ORTHO_CARDIAC` verified safe (str, never coerced);
  caveats: dead-code tripwire + count the 7.5→8.0 shift. **AR-M8** the
  `BDTYPE→component` map is a gating prerequisite, not an open item. **AR-M9**
  define platelet-audit-unit vs MTP-signal interaction (suppress platelet units
  under active MTP).

### Required revisions (folded in)
- §5.1 platelet gate redesigned (CR-C1); §5.3.5 guardrail (CR-C2); Phase 1 open
  clinical questions added (CR-H1/M1).
- **Phase 2 re-scope: DONE 2026-07-08** — §5.3 rewritten as
  `component map (prerequisite)` → `intake + store schema` → `generic lookup
  core` → `platelet classifier` → `dispatch` → `platelet LLM subsystem` →
  `sampling`, with MTP-platelet suppression (CR-M3/AR-M9) and the
  platelet-final-class invariant test (CR-M2) folded into stage 3.

### Verification pass (2026-07-08, post-implementation review)

Phase 1 code verified against the plan and PR #83; Phase 2 constraints
re-verified against current code. Findings:

- **All four §8 architecture claims hold in code:** `RBC_PRODUCTS` allow-list
  (`audit_orders/rules.py:27`); `hb_lookup` [2, 25] g/dL validation
  (`parse.py:13-14`, `models.py:37`); `AuditRow` required non-null
  `hb_value`/`hb_datetime`/`cohort_threshold` (`audit_store/models.py`);
  `ALLOWED_COHORT_THRESHOLDS` raise (`system_prompt.py:128`).
- **M1 pilot count = 0** (artifact-verified, `/tmp/bba_mini`
  `report_oldcode.csv` vs `report.csv`): zero `cardiac_surgery →
  ortho_surgery` moves in the 300-case sample; the dual-surgery precedence
  question (CR-M1) has no live cases — it is a policy-only ruling. The two
  `default → ortho_surgery` moves and 0 verdict changes were re-confirmed.
- **NEW — B1 guardrail composition gap** (neither reviewer caught it):
  reusing `llm_overclear_suspect` unchanged makes v1 platelet unable to
  finalize `APPROPRIATE` below 100k. Recorded as an explicit OPEN DECISION in
  §5.3 stage 5 and §5.4.
- **Phase 1 test-intent gap fixed:** `test_deterministic_classifier.py`
  threshold parametrizations covered only the deprecated `ORTHO_CARDIAC`;
  `ORTHO_SURGERY` (the emitted label) added alongside it.
- **Doc rot fixed:** §3.7 annotated as superseded (CR-C1); §5.3 rewritten;
  status header updated.
- **H1 nuance recorded:** if the clinician rules "pre-op only", the change is
  a redesign, not a toggle — cohort detection is REQ-anchored over a
  *backward* 30-d window, so a pre-op ortho reservation hits
  `preop_defer_llm` with `{cohort_threshold}` = 7.0, and the restrictive
  prompt then steers the LLM *against* the policy's 8.0 pre-op target for
  exactly the policy's target population. Pre-op scope needs
  `upcoming_ops`-based detection or prompt-side encoding.

### Open decisions for the user
1. **Confirm the platelet-gate reversal** (plt<10 no longer auto-clears — reverses
   our grilled "10k appropriate"). Recommend: accept; it's a genuine safety fix.
2. **Two clinician rulings gate Phase 1 build:** ortho pre-op/post-op scope
   (CR-H1) and dual-surgery precedence (CR-M1).
