# `bba.source_mapping` — the EMR→domain seam (design + glossary)

> Status: design only (no code yet). Produced via `/improve-codebase-architecture`,
> candidate #1 (the Source-Mapping / EMR adapter seam). Goal steer from the user:
> "make it generalize, don't overfit — scale-up later for production usage and
> multicenter."

## Context

**Why this exists.** The `src/bba/` modules (ingest, hb_lookup, vitals_extractor,
cohort_detector, deterministic_classifier, evidence_bundle_builder, audit_pipeline,
…) are deep and well-tested. But the *composition root* that turns raw center
tables into the `PipelineRowContext` objects `audit_pipeline.run_pipeline` consumes
does **not** live in `src/`. It lives as ~4,200 lines of glue across three pilot
scripts (`scripts/pilot/run_pipeline.py`, `run_llm_leg.py`, `build_review.py`),
where:

- Every center-specific HOSxP fact is a hardcoded magic literal, duplicated
  across all three scripts (`HB_HEM_CODE="290095"`, `HB_POCT_CODE="500001"`,
  `ANC_CODE="290093"`, `INCPT_OPERATION_GROUPS={"110","111"}`,
  `CRYSTALLOID_KEYWORDS`, the `INCPT/OPRTACT` bridging rules, ICD9 normalization).
- The extraction helpers (`_build_hb_observations`, `_build_op_events`,
  `_build_med_events`, `_latest_anc`, `_read_csv`, `_icd_codes`, `_is_crystalloid`,
  `_normalize_iptsumoprt/_incpt/_optract`) are verbatim-triplicated.
- The scripts re-read raw CSVs directly, bypassing the tested `bba.ingest`
  normalize/drift layer.

This is the **multicenter obstacle**: onboarding center #2 today means editing
literals scattered across three 1,500-line scripts. The pilot README itself frames
the orchestrator as "an integration seam — the modules ship, but the orchestrator
that composes them is left to the deployment."

**Intended outcome.** Lift the composition root into one tested internal module
`bba.source_mapping` with a sharp seam at the **domain-model boundary**: the moment
a domain input model (`HbObservation`, `MedEvent`, `OperativeEvent`, `VitalsNote`,
`BloodOrderInput`) is constructed, all center-specific knowledge stops; everything
downstream is already pure and center-agnostic. Center facts collapse into one
frozen `SourceProfile` value. The result generalizes to multicenter **without**
speculative abstraction — per the forks already resolved with the user:
**consolidate now, no `Protocol` yet** (center #2 shape is still speculative; the
seam stays cuttable later), and the module returns **domain input models (sharp
seam)**, not raw rows.

This single move also retires architecture candidate #4 (orchestrator bypasses):
once the pilot calls `run_pipeline` through this assembler, the model-id /
cost-guard / MTP-arm glue has one home.

## Scope decisions already locked (do not re-litigate)

- **No `Protocol` / no `SourceReader` interface yet.** One concrete module. The
  domain-model boundary is the seam; a second center turns it into a real seam
  ("one adapter = hypothetical, two = real"). Until then, abstraction is overfitting.
- **Return type is domain models**, assembled into `SourceEncounter` bundles —
  not a generic row mapping.
- **Reuse `bba.ingest`, do not add a third HOSxP reader.** `source_mapping`
  consumes the existing `NormalizedRow` output of `bba.ingest.normalize`
  (drift-checked, deduped, projected), never raw `_read_csv`.
- **`source_mapping` is a pure transform (no I/O).** It takes already-normalized,
  name-keyed rows and returns domain bundles. Reading Parquet / sampled CSV stays
  the caller's job (pilot / CLI), so the module is unit-testable with in-memory rows.
- PHI / safety invariants are unchanged and out of scope here: de-id remains an
  operator-owned gate run *outside* this module before any LLM call;
  `enable_missing_hb_positive_evidence` stays `False` pending clinical sign-off.

## New glossary entries (to add to `docs/CONTEXT.md` during implementation)

**SourceEncounter** — An AN-scoped bundle of center-agnostic domain input models
for one inpatient admission: `hn_hash`, `an_hash`, and tuples of `BloodOrderInput`,
`HbObservation`, `MedEvent`, `OperativeEvent`, `VitalsNote`, plus diagnoses. It is
the **interface of the `source_mapping` seam**: the last place center-specific
(HOSxP) knowledge is allowed to appear is in *producing* a `SourceEncounter`;
everything that consumes one is pure and shared across centers. Windowing
(REQ→USE re-anchoring, per-order evidence windows) happens *downstream* of this
bundle in `hb_lookup` / `vitals_extractor` / `cohort_detector`, not inside it —
the bundle is the full AN-scoped material, not a per-order slice.

**SourceProfile** — A frozen value object isolating every center-specific fact the
mapping needs, gathered out of the scattered pilot literals into one place: lab
codes (`hb_hematology="290095"`, `hb_poct="500001"`, `anc="290093"`, and the
`run_llm_leg` additions `plt="290078"`, `wbc={"290136","120015"}`,
`neutrophil_pct="290092"`, `neutrophil_abs="290093"`), `incpt_operation_groups`
(`{"110","111"}`), `crystalloid_keywords`, evidence/proximity windows, and any
column-name / date-format quirks. The default instance is `KCMH_HOSXP_PROFILE`.
A new center is "a new `SourceProfile`," not a new code path. Its codes are
surfaced verbatim in a seed test (the existing `TestAllowListSeeds` pattern) so a
silent code drift fails loudly.

## Build order

All new code under `src/bba/source_mapping/`. Mirror the existing module layout
(`models.py`, focused transform files, `__init__.py` re-exporting the public API).
Keep files 200–400 lines.

1. **`docs/CONTEXT.md`** — add the two glossary entries above (first, so the
   vocabulary is authoritative before code references it).

2. **`src/bba/source_mapping/profile.py`** — `SourceProfile` frozen dataclass
   (`@dataclass(frozen=True, slots=True)` or Pydantic `frozen=True`, matching the
   ingest models' convention) + the `KCMH_HOSXP_PROFILE` default instance carrying
   every literal currently hardcoded in the pilot. Single source of truth for codes.

3. **`src/bba/source_mapping/models.py`** — `SourceEncounter` frozen model.
   Fields exactly as the glossary entry. Reuses the existing domain models
   (`BloodOrderInput`, `HbObservation` from `bba.hb_lookup.models`, `MedEvent`,
   `OperativeEvent`, `VitalsNote` from `bba.vitals_extractor` / `audit_orders` /
   `cohort_detector` models — confirm exact import path per
   `grep -rn "class HbObservation" src/bba`).

4. **`src/bba/source_mapping/extract.py`** — the center-specific extraction,
   ported (not rewritten) from the pilot's `_build_hb_observations`,
   `_build_op_events` (incl. the `INCPT/OPRTACT` bridging:
   `O__OPRTACT or INCOME or ORDERCODE`, `INCGRP in profile.incpt_operation_groups`,
   ICD9 `.replace(".","")`, `or_flag` derivation), `_build_med_events`,
   `_latest_anc`, `_is_crystalloid`. **Every literal becomes a `SourceProfile`
   field read.** Input is name-keyed normalized rows
   (`Mapping[CSVTable, Sequence[Mapping[str, str]]]`), produced by zipping
   `bba.ingest.normalize`'s `NormalizedHeader.header` with each
   `NormalizedRow.cells` — i.e. consuming the tested ingest layer, not raw CSV.

5. **`src/bba/source_mapping/encounter.py`** — `build_source_encounters(rows,
   profile) -> tuple[SourceEncounter, ...]`: groups the normalized rows by `AN`,
   runs the `extract.py` functions per admission, emits one `SourceEncounter` per AN.

6. **`src/bba/source_mapping/assemble.py`** — the **center-agnostic assembler**
   `assemble_contexts(encounter, *, policy_version, …) -> tuple[PipelineRowContext, ...]`.
   Fans out per `BloodOrderInput` in the encounter; for each order, windows the
   AN-scoped domain lists at that order's anchor by calling the *existing* pure
   modules (`hb_lookup`, `vitals_extractor`, `cohort_detector`,
   evidence/peri-op proximity) and fills every `PipelineRowContext` field
   (`hb_result`, `vitals_result`, `cohort_assignment`, `procedure_proximity_hours`,
   `crystalloid_liters_prior_4h`, hashes, `periop_summary`, …). This file contains
   **zero** center literals — that is the proof the seam is sharp.

7. **`src/bba/source_mapping/__init__.py`** — re-export `SourceProfile`,
   `KCMH_HOSXP_PROFILE`, `SourceEncounter`, `build_source_encounters`,
   `assemble_contexts`.

8. **Rewire the pilot (thin)** — `scripts/pilot/run_pipeline.py` collapses to:
   `ingest/normalize rows → build_source_encounters → assemble_contexts →
   audit_pipeline.run_pipeline`. Delete the now-duplicated extraction helpers and
   magic literals from all three pilot scripts; they import from
   `bba.source_mapping` instead. (If time-boxed, `run_pipeline.py` first, then the
   other two follow the same import swap.)

## Tests (TDD — write first, per repo convention)

- `tests/source_mapping/test_profile.py` — **`TestAllowListSeeds`-style** seed
  test asserting `KCMH_HOSXP_PROFILE`'s codes verbatim (`"290095"`, `"500001"`,
  `"290093"`, `{"110","111"}`, the WBC/PLT/neutrophil codes). Loud failure on drift.
- `tests/source_mapping/test_extract.py` — unit tests per extraction function
  feeding hand-built normalized rows, asserting the domain models produced. This is
  the payoff: extraction logic that was untestable glue is now testable **through
  the `SourceEncounter` interface**. Cover the gnarly `INCPT/OPRTACT` bridging and
  ICD9 normalization paths explicitly.
- `tests/source_mapping/test_encounter.py` — AN grouping: rows for two admissions
  produce two `SourceEncounter`s with correctly partitioned domain lists.
- `tests/source_mapping/test_assemble.py` — property/behavioral tests: per-order
  fan-out count, REQ→USE re-anchoring is applied, and assembled `PipelineRowContext`
  fields match a known fixture. Plus an **assembler-purity check**: import/grep-level
  assertion that `assemble.py` references no center literal (cheap seam regression
  guard).
- Maintain ≥80% coverage on the new package.

## Verification (end-to-end)

1. `pytest tests/source_mapping -q` green; `pytest --cov=src/bba/source_mapping
   --cov-report=term-missing` ≥80%.
2. Full suite still green: `pytest -q` (no regression in ingest / audit_pipeline /
   existing pilot-covered behavior).
3. **Behavioral parity on real data** (PHI-safe, operator-run under
   `$BBA_PILOT_WORK_DIR`): run the rewired `run_pipeline.py` over the existing
   sampled bundle and diff the resulting `audit_store` against a pre-refactor run.
   Classifications and evidence bundles must match — proves the port preserved the
   pilot's exact extraction semantics. (No new CSVs committed; `.gitignore` already
   blocks them.)
4. Lint/type: `ruff`, `black --check`, `mypy`/`pyright` on the new package.
5. `bandit -r src/bba/source_mapping` clean.

## Out of scope (explicitly)

- No `Protocol`/reader abstraction, no second center, no production de-id changes,
  no flipping `enable_missing_hb_positive_evidence`. Those wait for a real center #2
  to turn the hypothetical seam into a real one.
