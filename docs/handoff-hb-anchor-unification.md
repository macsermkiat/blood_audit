# Plan: Unify Hb-anchor resolution between `run_pipeline.py` and `run_llm_leg.py`

## Context

In `/tmp/bba_mini/review.html`, **case 7 (REQNO 68066907)** displays a deterministic
verdict of `POTENTIALLY_INAPPROPRIATE` (`hb_ge_10`, Hb 10.0) sitting next to "EBL
evidence: EBL 1,200 ml", with the LLM box showing "(deterministic-final; LLM leg not
invoked)". This misleads a reviewer: an obvious peri-operative bleed appears ignored,
and a non-appropriate verdict stands as final with no LLM adjudication.

Investigation showed the headline symptom is real but the deeper cause is a
**divergence bug between two pilot scripts that independently resolve the Hb anchor**:

- The order is a **pre-operative blood reservation** (T/S + G/M LPRC 2U) placed
  2025-10-27 14:43:37 for a C-section the next morning. The EBL 1,200 mL is
  **post-operative** (notes stamped 2025-10-28 12:15 / 16:22), i.e. ~22 h *after* the
  order. The EBL line in the report is display-only (regex `_EBL_RE` in
  `build_review.py`); it never feeds the classifier.
- Hb history has two draws: **10.0 g/dL at 14:46:28** (~3 min after the order REQTIME)
  and 9.4 next day. There is no Hb at/before the REQTIME.

The two scripts anchor Hb differently, so they classify case 7 differently:

| Path | Anchor | Hb found | Verdict | Routed to LLM? |
|------|--------|----------|---------|----------------|
| `run_pipeline.py` -> `build_review.py` (the report) | `blood_bank_visit_fallback` (BDVST 14:47:27) | 10.0 | `POTENTIALLY_INAPPROPRIATE` | n/a |
| `run_llm_leg.py` (the LLM gate) | `order.order_datetime` (14:43:37) only | **missing** | `INSUFFICIENT_EVIDENCE` (`hb_missing`) | **No -- gated out** |

`lookup_hb` keeps only draws `<= anchor` (`src/bba/hb_lookup/lookup.py:64`). With the
REQTIME anchor, both draws are after it -> Hb missing -> `INSUFFICIENT_EVIDENCE`, which is
in the leg's `DETERMINISTIC_FINAL` set, so case 7 is never sent to the LLM. That is why
resuming/re-running the leg "changed nothing": case 7 was never in the submission set,
and never will be while the leg lacks the fallback ladder.

**Intended outcome:** the deterministic report and the LLM gate must resolve the Hb
anchor identically. After the fix, case 7 resolves to Hb 10.0 ->
`POTENTIALLY_INAPPROPRIATE` -> routed to the LLM (which, as for the analogous case 10,
will most likely adjudicate APPROPRIATE on peri-operative context), eliminating the
misleading "deterministic-final" display. The chosen approach is a **single shared
anchor-resolution helper** (not copy-paste) so the two paths can never drift again --
selected explicitly for long-term generalization and accuracy over a quick local patch.

## Root-cause references (read these first)

- `scripts/pilot/run_pipeline.py:520-573` -- builds the fallback anchor-candidate dicts
  (`use_dt_by_reqno`, `exact_issue_anchor_by_reqno`, `blood_bank_anchor_by_reqno`,
  `blood_bank_anchor_display_by_reqno`) from `bdvstdt` rows + `bdvst_by_reqno` parents.
- `scripts/pilot/run_pipeline.py:628-654` -- the resolution policy: primary
  `lookup_hb(anchor=order.order_datetime)`; on miss, try candidates in order
  `(issue_datetime, blood_bank_visit_fallback)`, each only if
  `fallback_anchor >= order.order_datetime`; first non-missing wins; records
  `hb_anchor_display` + `hb_anchor_reason`.
- `scripts/pilot/run_llm_leg.py:872-874` -- the leg's loop: `hb = lookup_hb(observations=
  _hb_observations(lab, order.an), anchor_utc=order.order_datetime)` with **no fallback**.
- `scripts/pilot/run_llm_leg.py:1220` -- `DETERMINISTIC_FINAL = {"APPROPRIATE",
  "INSUFFICIENT_EVIDENCE", "INAPPROPRIATE"}`; line 1237 -- `if cres.classification not in
  DETERMINISTIC_FINAL: llm_contexts.append(ctx)` (the gate that drops case 7).
- `src/bba/hb_lookup/lookup.py:49-65` -- `lookup_hb` signature + the `<= anchor_utc` filter.
- `scripts/pilot/build_review.py:612-614, 1010-1011, 1228-1229` -- `build_review.py`
  only *reads* `det["hb_anchor_datetime_local"]` / `det["hb_anchor_reason"]` from
  `run_pipeline.py` output; it does not compute the anchor. So once `run_pipeline` and
  the leg agree, `build_review` is automatically consistent. **No change needed here.**

## Approach (recommended: shared helper)

Create one canonical resolver and have both scripts call it.

### 1. New shared resolution policy -- `src/bba/hb_lookup/anchor.py`
Pure, unit-testable, no HOSxP/row knowledge:

- `@dataclass(frozen=True) AnchorCandidate: anchor_utc: datetime; display: str; reason: str`
- `resolve_hb_with_fallback(*, observations, order_datetime, candidates) -> tuple[HbLookupResult, str, str]`
  - Run `lookup_hb(observations=observations, anchor_utc=order_datetime)`.
  - If `hb.value_g_dl is not None`: return `(hb, "", "order_datetime")`.
  - Else iterate `candidates` in the given order; **skip any where
    `candidate.anchor_utc < order_datetime`** (preserve the run_pipeline guard); run
    `lookup_hb` at that anchor; first non-missing wins -> return
    `(hb, candidate.display, candidate.reason)`.
  - If none hit: return `(original_missing_hb, "", "order_datetime")`.
- Export from `src/bba/hb_lookup/__init__.py`.
- The candidate **order matters** and must match run_pipeline: `issue_datetime` first,
  then `blood_bank_visit_fallback`.

### 2. Shared candidate builder for the pilot -- `scripts/pilot/_anchor_candidates.py`
Bundle/HOSxP-row logic extracted verbatim from `run_pipeline.py:520-573`:

- `build_anchor_candidates(*, bdvstdt_rows, bdvst_by_reqno) -> dict[str, list[AnchorCandidate]]`
  keyed by REQNO, each list ordered `[issue_datetime?, blood_bank_visit_fallback?]`
  (omit a candidate when its source datetime is absent).
- Reuse existing date/time parsers. To avoid a third copy, prefer importing
  `_combine` / `_parse_hosxp_date` / `_parse_time` / `_fmt_local_datetime` from a single
  owner. Pragmatic minimal-risk choice: keep `run_pipeline.py`'s copies as the source and
  import them into the new module (or move that small parser set into
  `scripts/pilot/_hosxp_dt.py` and have both scripts import from there). Do **not**
  silently leave two diverging parser copies.

### 3. Rewire `run_pipeline.py`
Replace the inline candidate dicts (520-573, the anchor parts) and the resolution block
(628-654) with: build candidates via `build_anchor_candidates(...)`, then call
`resolve_hb_with_fallback(...)`. Keep emitting `hb_anchor_datetime_local` (=display) and
`hb_anchor_reason` exactly as today (`run_pipeline.py:744-745`) so `build_review.py` and
the report schema are unchanged. `use_dt_by_reqno` is also used for
`transfusion_datetime_local` (line 759) -- keep that mapping; only the *anchor* path moves
to the helper.

### 4. Rewire `run_llm_leg.py`
- Before the order loop, build `candidates_by_reqno` with the same
  `build_anchor_candidates(...)` (needs `bdvstdt` + `bdvst_by_reqno`; confirm both are
  available in `_build_inputs()` / the leg's scope -- `bdvstdt` is read for products at
  `run_llm_leg.py:790`).
- Replace `run_llm_leg.py:874` with `hb, _disp, _reason = resolve_hb_with_fallback(
  observations=hb_obs, order_datetime=order.order_datetime,
  candidates=candidates_by_reqno.get(order.reqno, []))`.
- Nothing else in the leg changes: `hb_result=hb` already flows into the context
  (`run_llm_leg.py:1200`) and the `ClassifierInputs` (1227), and the 7-day Hb evidence
  chunks are built from `hb_obs` independently of the anchor, so the LLM already sees all
  Hb values.

### 5. Optional cleanup (only if low-risk)
`_build_hb_observations` (run_pipeline) and `_hb_observations` (leg) are duplicate
implementations -- unifying them would further reduce drift but is **not required** for
this fix. Leave for a follow-up unless trivially identical.

## Already done (uncommitted -- do not redo)
`run_llm_leg.py` (~line 1278) gained a `BBA_PILOT_BATCH_ID` resume path: when set, it
re-attaches to an existing batch instead of re-submitting, and a fresh submit now prints
the resume command. This is **uncommitted** -- keep it; it's orthogonal to this fix.

## Verification

1. **Deterministic, no API (primary gate).** Write a throwaway script (model it on the
   session's `/tmp/_chk_hb.py`) run via `PYTHONPATH=src uv run --no-project python`:
   - Assert the leg now resolves REQNO **68066907** -> Hb **10.0**, reason
     `blood_bank_visit_fallback`, classify -> `POTENTIALLY_INAPPROPRIATE`, and that it is
     appended to `llm_contexts` (not in `DETERMINISTIC_FINAL`).
   - **Regression guard:** for all 20 included orders, assert `run_pipeline` and
     `run_llm_leg` now produce **identical** `(classification, hb.value_g_dl,
     hb_anchor_reason)`. No case other than 68066907 should change.
2. **Unit test** `tests/unit/` for `resolve_hb_with_fallback`: order-time hit wins;
   fallback ladder order (issue before blood-bank); the `>= order_datetime` skip guard
   (a pre-order candidate must NOT be used); all-miss returns missing.
3. **End-to-end (real API, costs money, async/slow).** Per the known store gotcha
   (audit_store idempotent on `(run_id, audit_id)`), use a **fresh run id**:
   `BBA_PILOT_RUN_ID=pilot-mini-anchorfix uv run python scripts/pilot/run_llm_leg.py`
   (the batch poll can take many minutes; `max_wait=3600s`). Confirm `68066907` now
   appears in `/tmp/bba_mini/llm_report.json` with an LLM verdict.
4. **Rebuild + eyeball:** run `scripts/pilot/build_review.py`, open
   `/tmp/bba_mini/review.html` case 7, confirm the LLM box now shows an adjudicated
   verdict (expected APPROPRIATE on peri-op context) instead of "(deterministic-final;
   LLM leg not invoked)".

## Gotchas
- Bump `BBA_PILOT_RUN_ID` (or wipe the store) or the idempotent cache returns stale rows.
- Preserve candidate **order** and the `>= order_datetime` guard exactly -- they encode the
  clinical rule that a fallback anchor may be slightly *after* the order (labs drawn
  minutes post-REQTIME) but never *before* it.
- This is a pre-op reservation whose EBL is post-order; do not "fix" the deterministic
  engine to consume post-order EBL. The correct routing is via the Hb anchor -> LLM.
- The deeper structural gap (the C-section exists only in free-text notes, not the
  structured procedure tables, so the deterministic pre-op-crossmatch bypass at
  `classifier.py:184` can't fire) is **out of scope** here and noted for a separate
  decision.
