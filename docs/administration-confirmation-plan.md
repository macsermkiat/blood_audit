# Administration confirmation for reserve-ahead blood orders

**Status:** IMPLEMENTED — specification #105 Phase 5, tickets #106–#110.
**Scope:** Pilot enablement and operator handoff. The production/live feature
flag remains default-off.

---

## 1. The structured-data gap

The hospital blood bank never advances an order beyond `BDVSTST=4`
(จ่ายแล้ว, dispensed). Status 5 (ให้เลือด, transfused) is never recorded: all
300 orders in the pilot are at status 4. The export also omits the
`BDVSTTRANS` returns table, so it cannot distinguish a dispensed unit that was
returned from one that was administered.

The available timestamps do not close that gap. A bedside-use time (`USETIME`)
is present for only 128/300 orders, while the blood-bank pickup datetime
(`PICK`) is present for 273/300. Neither incomplete field establishes the
disposition of every dispensed unit. Whether blood was actually administered
is therefore **UNCONFIRMABLE from structured data for every pilot order**.
The crossmatch-to-transfusion (C:T) ratio is likewise uncomputable because the
export supplies no reliable transfused-unit denominator.

## 2. The category error this caused

Case `68026306` was a pre-TAVI crossmatch reservation for four LPRC units at
Hb 12.9 g/dL. The units were dispensed the evening before the procedure and
sent to the catheterization laboratory as standby; the available record did
not document administration. The previous path nevertheless sent the case
through `HB_GT_10_OVERRIDE` and returned `INAPPROPRIATE`.

That result conflated a reservation decision with a completed transfusion.
Because final verdicts feed named-physician attribution scorecards, the system
was recording an untransfused standby reservation as an inappropriate
transfusion against a physician. Reservation practice and transfusion practice
are different behaviors and must be assessed separately.

## 3. Solution shipped under specification #105

Tickets #106–#110 introduce a reserve-ahead path without skipping the LLM.
The deterministic classifier rationale `preop_defer_llm` identifies an
upcoming blood-requiring procedure within 72 hours when Hb is at or above the
patient's cohort floor. When the default-off
`RESERVE_AHEAD_ROUTER_ENABLED` flag is enabled, those cases route to the new
`RESERVE_AHEAD_REVIEW` task mode.

The reserve-ahead prompt asks two separate questions:

1. Was reserving blood appropriate for the planned procedure?
2. Is there affirmative documented evidence that blood was administered?

For the second question, the model may cite only the enumerated evidence
types: a ให้เลือด entry, a named blood component given, transfused unit
numbers, intra-operative transfusion, or post-transfusion vitals or a
transfusion-reaction check. The prompt states explicitly that absence of a
ให้เลือด note is not evidence of non-transfusion. No reserve-ahead case is
deterministically finalized before the model sees its evidence bundle.

## 4. Asymmetric administration-confirmation gate

The result-application gate confirms administration when **any** of the
following is present:

- a grounded LLM administration citation whose quote matches a contiguous
  span in that case's own evidence bundle;
- an affirmative fact-only `scan_administration` extractor marker;
- a structured intra-operative transfusion marker; or
- estimated blood loss of at least `PERIOP_MIN_EBL_ML` (500 mL).

An LLM quote that does not ground to the case's evidence is discounted rather
than treated as confirmation. When administration is confirmed, the case
continues through the normal appropriateness-verdict path, while retaining the
reserve-ahead exemption from the `HB_GT_10` assertion.

When none of the affirmative signals is present, the result becomes the
store-only terminal classification `PREOP_RESERVATION_UNCONFIRMED`. This
classification has `needs_human_review=False`, is never queued for human
review, and is projected into the Unresolved bucket on physician and
department scorecards. It is never counted as an inappropriate transfusion or
included in inappropriate-transfusion attribution.

There is one deliberately conservative contradiction path. If structured
intra-operative transfusion or EBL evidence fires but the model does not claim
administration, the result is floored to `NEEDS_REVIEW` with review reason
`administration_signal_contradiction`. This rare disagreement reaches a human
instead of being silently resolved. In every branch, absence of a ให้เลือด
note is never proof that transfusion did not occur.

## 5. Over-reservation reporting seam (future work)

The model's `reservation_assessment` — `APPROPRIATE`, `INAPPROPRIATE`, or
`INSUFFICIENT_EVIDENCE` — is persisted on the audit row even when the final
classification is `PREOP_RESERVATION_UNCONFIRMED`. This preserves the
reservation judgment independently of the unconfirmed administration status.

A future over-reservation report can therefore identify patterns such as four
units reserved for elective TAVI at Hb 12.9 g/dL without re-running the pilot
LLM leg. Building that report, selecting its thresholds, and assigning its
governance are out of scope for this work; only the persistence seam is in
place.

## 6. Formal data request for hospital IT

To close the administration gap at its source, please add the following two
data sources to the hospital export:

1. **`BDVSTTRANS` returns table.** Include the records needed to associate
   returned units with the corresponding admission, order, component, unit
   number, and return datetime. This distinguishes blood that was dispensed
   and later returned from blood that may have been administered.
2. **Nursing transfusion-administration records.** Export the bedside ให้เลือด
   documentation, including administration date/time, blood component, unit
   numbers, and post-transfusion vital signs or reaction documentation.

Together, these sources would let the audit confirm administration from the
source record, reconcile dispensed, returned, and administered units, and
eventually calculate a real C:T ratio with a defensible transfused-unit
denominator. Until both sources arrive and are validated, this audit reports
**order appropriateness**, not **transfusion appropriateness**.

## 7. Pilot enablement and operator runbook

The pilot LLM leg, `scripts/pilot/run_llm_leg.py`, enables
`RESERVE_AHEAD_ROUTER_ENABLED` by default. Set
`BBA_PILOT_RESERVE_AHEAD_ROUTER=0` to override it off. The live pipeline and
resume paths continue to use the default in `bba.feature_flags`, where
`RESERVE_AHEAD_ROUTER_ENABLED` remains `False`.

From the repository root, regenerate the pilot artifacts in this order:

```bash
export ANTHROPIC_API_KEY=<approved-api-key>
uv run python scripts/pilot/run_pipeline.py
uv run python scripts/pilot/run_llm_leg.py
uv run python scripts/pilot/build_review.py
```

`run_pipeline.py` produces the deterministic leg. `run_llm_leg.py` submits the
Anthropic batch and applies returned results through the administration gate.
`build_review.py` then rebuilds the review page from those results. Do not
build the review page before the billable LLM leg has completed and applied
its batch results, or it will display stale verdicts.

Verify the regenerated pilot as follows:

- case `68026306` renders **Administration unconfirmed (pre-op reservation)**
  instead of **Inappropriate**;
- the next-morning-USE reserve-ahead cohort routes through
  `RESERVE_AHEAD_REVIEW`;
- the review page displays the dispense (`BDVST PICK`) and use (`BDVSTDT USE`)
  datetimes as separate fields;
- physician attribution excludes unconfirmed reservations from inappropriate
  counts and displays them in Unresolved; and
- totals from `scripts/pilot/reconcile_verdict_sources.py` remain unchanged
  for non-reserve-ahead cases.
