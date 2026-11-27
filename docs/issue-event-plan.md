# Issue-event audit unit

> Design record for blood_audit #276. It is the owner-approved plan with the gold-set values removed, aligned to the code merged in PRs #279 to #287 (main at 6c3cfda). Where the plan text and the merged code differ, this record states the code. Use [current module contracts](CONTEXT.md), the [pilot README](../scripts/pilot/README.md), and the [report schema](report-schema.md) for operating detail.

Counts in this record are aggregates. They come from read-only measurement scripts run over the full 2025 encrypted HOSxP export, and from the per-PR revision logs of the plan. The scripts and the gold-set CSV are kept outside the repository. No patient, admission, order, or clinician identifier appears here.

## Why the audit unit changed

The REQNO-level audit (v1 and v2) judged one REQNO once, at one anchor (BDVST `PICKTIME`, else BDVSTDT `USETIME`), against one Hb, and credited every unit to `BDVST.DCTREQ`. The blood bank issues later units against the first open REQNO. One REQNO can therefore hold several transfusion decisions, made on different days by different doctors and on different Hb values.

The audit unit is now the issue event. An issue event is a cluster of physical units (`BDVSTTRANS`) that one REQNO issued close together. It is anchored on the first unit's `PAYDATE` and `PAYTIME`, judged against the order that most plausibly caused it, and attributed to that order's `DCTREQ`.

## Owner decisions

These decisions override the defaults that earlier plan text proposed.

- **8.15 declared use on later events, alternative (c), amended 2026-11-22.** Event 1 keeps the existing rule and the existing MSBOS precedence. A standing-order event is never exempt. A fresh-order event is exempt only when its own fresh order declares USETYPE 2 or 3 and an operation follows that fresh order within 6 h before to 72 h after the issue, both ends inclusive. An operation is any `IPTSUMOPRT` record with a readable date, an INCPT surgery-fee charge (`INCGRP` 111), or an INCPT general-anaesthesia charge (`AS056` to `AS059`, read from `O__OPRTACT`). INCPT bedside-procedure and add-on charges alone do not count, and a cancelled INCPT charge (`CANCELDATE` set) is dropped. A blank or unreadable operation time is placed at midnight. Of the 258 later events in 2025 whose fresh order declares surgical use, 3 qualify on `IPTSUMOPRT` alone and 228 under the amended rule. On audit records the counts are 251 and 223.
- **8.4 REQNOs with no issued unit.** Default accepted. Each gets one REQNO-level `excluded` row with rationale `no_issue_record`. It is listed, not scored, and not evaluated for over-reservation.
- **8.11 production `bba` CLI.** Default accepted. The application path stays per REQNO until after the 2025 year. The divergence is documented in [module contracts](CONTEXT.md#application-path-stays-per-reqno).
- **8.16 privacy Day 0.** Acknowledged. The de-identification date shift uses the issue date as Day 0.
- **8.2 first-event attribution.** Default accepted. Event 1 links to its own REQNO. Both linkage columns appear on January v3 for review.
- **1.3 under-covered all-returned events (2026-11-23).** An event whose issued units are all status 3 or 7 is `RETURNED_NOT_TRANSFUSED`, even when its REQNO issued fewer units than ordered. The REQNO-level coverage rule (`inconclusive`) does not apply to events.
- **Notes guard window on all-returned events (2026-11-24, amended 2026-11-26).** The hard-transfusion notes guard counts only notes charted from this event's issue time up to the next red-cell or platelet issue event on the same admission, under any REQNO that `audit_orders` includes. A plasma or cryoprecipitate issue does not end the window. A next issue at the same instant does not end it. With no next issue, the window ends 24 h after this event's issue. The start is inclusive and the end is exclusive.
- **All-returned event 1 of a declaring REQNO (2026-11-24).** `RETURNED_NOT_TRANSFUSED` takes precedence over the MSBOS reservation review and over the declared-use exemption.
- All other section 8 defaults were accepted.

## Define an issue event

### Physical units

`bba.issue_events.units_from_ledger(trans_rows, an)` reads one REQNO's `BDVSTTRANS` rows. It keeps only rows on the order's admission (`rows_for_admission`) and collapses them on (`DNRNO`, `SEQNO`, `BDTYPE`). The terminal `UNITSTAT` follows the returns-ledger precedence 5 > 3 > 7 > 2. Each unit carries `issue_at` (earliest `PAYDATE` + `PAYTIME`), `returned_at` (earliest `RTNDATE` + `RTNTIME`), and `given_at` (earliest `GIVEDATE` + `GIVETIME`). `returned_at` and `given_at` are display fields.

Units sort by (`issue_at`, `DNRNO`, `SEQNO`, `BDTYPE`). The event's first unit fixes its `audit_id`, so this tie-break is part of the contract. In 2025, 2,567 events have two or more units with an identical first issue time.

A unit with no readable `PAYDATE` and `PAYTIME` was never issued. It forms no event and counts in `units_reserved_not_issued` (681 units in 2025). A unit with terminal status 2 or 5 and no readable issue time is a ledger defect, and `units_from_ledger` raises instead of dropping it.

The product family comes from the REQNO's BDVSTDT lines, not from the unit `BDTYPE`. `BDVSTTRANS` uses variant codes outside both order allow-lists. `unit_family_mismatch` flags an event with a unit whose code marker names another family or no recognised family. The unit stays in the event.

### Clustering

`cluster_events(units, order_times, gap=ISSUE_EVENT_GAP)` walks a REQNO's issued units in issue order. A unit joins the current event when both conditions hold. Otherwise it starts a new event.

1. The gap to the previous unit's `issue_at` is at most 4 h (`ISSUE_EVENT_GAP`).
2. No candidate order for the same admission and family was placed in the interval (previous unit issue, this unit issue].

The 4 h window comes from the 12,038 consecutive red-cell gaps measured over the full export. The share of gaps with a new Hb draw in between is 8% at 1 h or less, 14% at 1 to 2 h, 23% at 2 to 4 h, 41% at 4 to 6 h, and 74% at 6 to 8 h. Four hours is the last bucket where most gaps carry no new Hb. The order split covers the case the blood bank misses: a new order placed between two issues is a new decision regardless of the gap.

Measured with the default rule, 2025 has 20,830 events (15,011 red cell and 5,819 platelet) under 13,740 eligible REQNOs. January has 1,830 events (1,321 and 509) under 1,198 REQNOs.

### Returned and transfused units

A unit is returned only by terminal `UNITSTAT` 3, never by `RTNDATE` alone. 160 status-2 units carry a stray `RTNDATE`.

- An event whose units are all status 3 or 7 is `RETURNED_NOT_TRANSFUSED`. It is excluded from scoring and still attributed. On 2025 audit records this covers 1,902 red-cell and 78 platelet events.
- A mixed event is judged as a transfusion. `issue_units_returned` appears on the row and in the prompt. 2025 audit records hold 1,091 mixed red-cell events.
- A returned unit does not affect clustering.
- `issue_units_returned` counts status 3 only. An event of status-7 units is all-returned and reports 0 returned units.
- Status 2 (dispensed) counts as presumed transfused, as before. An issue proves issue, not administration.

## Link each event to an order

### Candidate orders

`candidate_order(record)` turns a BDVST row into a `CandidateOrder` or rejects it. A candidate has an admission, a red-cell or platelet product family, and a placement time (`REQDATE` + `REQTIME`, else `BDVSTDATE` + `BDVSTTIME`). Status 6 (refused), 7 (cancelled), 11, and 12 (cancelled doctor orders) are never candidates. Every other status is a candidate, including pending status 1. The same candidates split events and link them.

### Linkage rule

`link_events(own, events, candidates)` links a REQNO's events in issue order.

1. Event 1 links to its own REQNO, tag `own`.
2. Event k > 1 links to the latest candidate placed in (event k-1 issue, event k issue], tag `fresh_order`.
3. With no such candidate, event k links to its own REQNO, tag `standing_order`.
4. The own REQNO is itself a candidate. Placed after event 1, it would be event 2's fresh order. `order_after_issue` flags a linked order placed after its issue (0 on the 2025 export).

Ties on placement time go to the higher REQNO, so the link is deterministic. `latest_before` records the #276 proposal beside the default. It is the latest candidate placed at or before the issue, with no own-REQNO preference. Both appear on every row (`linked_reqno` and `linked_reqno_latest_before`).

Measured with the default rule, the 2025 events split 13,700 own, 3,462 fresh, and 3,668 standing. January splits 1,193, 290, and 347. In a measurement taken before PR 2, 215 fresh-order links in 2025 pointed at an order that is itself an eligible REQNO with events of its own. The event issued under the older REQNO and the fresh REQNO's own events are all scored and credited, because they are separate issues.

### Attribution

The attributed doctor is the linked order's `DCTREQ`, resolved through `linked_reqno` with `load_reqno_to_doctor`. `CandidateOrder` does not carry `DCTREQ`.

## Anchors and keys

`PICKTIME` is no longer an anchor anywhere in the pilot. It stays as a display column labelled "not an anchor".

| Place | Before | After |
| --- | --- | --- |
| Lab and evidence anchor | PICK, else USE, earliest per REQNO; re-anchor at 24 h or more | `resolve_event_anchors(event)`: one anchor at the issue time; reason `issue_reanchor` when the linked order was placed 24 h or more before the issue, else `issue_datetime` |
| #249 prior-transfusion cutoff | last red-cell dispense within 24 h | `prior_transfusion_cutoff(events_on_an, event)`: the latest red-cell event on the same admission, under any REQNO that `audit_orders` includes, issued strictly earlier and at most 24 h before, that transfused at least one unit |
| Classifier `declared_use` | own REQNO USETYPE | the linked order's USETYPE, passed only when `declared_use_applies` is true |
| Cohort, procedure proximity, ANC, evidence windows | REQNO order time | issue time |
| Crystalloid 4 h total | Hb draw | Hb draw; issue time only when no Hb exists |
| Note-date gate, replay guardrails | order date | issue date |
| Privacy Day 0 (`anchor_date`) | order calendar date | issue calendar date |
| Returns ledger | per REQNO | per event, plus REQNO totals |
| `PREOP_OVER_RESERVATION` | per REQNO | per REQNO, written on event 1 only |
| LLM report merge, re-ask list | by `reqno` | by `audit_id` |
| Rankings, lab statistics | keyed by REQNO | per-event records keyed by `audit_id`, credited through `linked_reqno` |
| Review page | one case per REQNO | one case per event, with the REQNO unit timeline |

The cutoff boundaries match the earlier rule. An event at the same instant is not a prior. A prior exactly 24 h earlier counts. An all-returned event is skipped (#264 option A). A platelet event reads no Hb and has no cutoff. The cutoff pool spans every month in the bundle: in 2025, 24 of 3,559 cutoffs come from a REQNO in a different `REQDATE` month. December 2024 REQNOs are largely absent from the export, so January cutoffs into December cannot be found.

Hb is still read by draw time (`LVSTDATE`, `LVSTTIME`), not result time.

## Identity

`build_event_audit_id(hn, reqno, unit_key)` is the first 32 hex characters of SHA-256 over `"<hn>:<reqno>:<DNRNO>:<SEQNO>:<BDTYPE>"` of the event's first unit. Events of one REQNO start with different units, so their ids are disjoint. `build_audit_id(hn, reqno)` stays as `parent_audit_id` and joins v3 events to v1 and v2 rows.

A REQNO with no event rows keeps one REQNO-level row with `audit_id` equal to `parent_audit_id`. That covers a REQNO that `audit_orders` excludes and a REQNO with no issued unit.

The audit store is idempotent on (`run_id`, `audit_id`, `code_version`). `EVENT_CODE_VERSION` in `scripts/pilot/_pilot_events.py` is the base token `pilot-event-1+gap4h+link-own`. `run_llm_leg.py` appends its seam suffixes (`+returns`, `+declared`, and the others) to it. `run_pipeline.py` uses the base token alone because it writes no store rows. Any change to `ISSUE_EVENT_GAP`, the linkage rule, or the event facts in the prompt needs a new base token and a fresh `BBA_PILOT_RUN_ID`. Comparisons across windows join through `parent_audit_id`, never through `audit_id`.

## Outputs

[The report schema](report-schema.md#pilot-issue-event-outputs) lists the `report.csv`, `events_manifest.csv`, and `llm_report.json` fields. Both pilot legs build events through the shared `scripts/pilot/_pilot_events.py`, so they judge the same events against the same orders and the same cutoff pool.

## Size and call volume

January has 1,830 events against 1,198 REQNOs. 2025 has 20,830 events against 13,740 REQNOs. The January v2 run routed 467 of 1,171 orders to the model.

The January v3 deterministic leg (2026-11-26) produced these rows: `APPROPRIATE` 555, `INSUFFICIENT_EVIDENCE` 26, `NEEDS_REVIEW` 868, `PERIOP_TRANSFUSION_EXEMPT` 150, `POTENTIALLY_INAPPROPRIATE` 45, `PREOP_OVER_RESERVATION` 18, `RETURNED_NOT_TRANSFUSED` 150, and `excluded` 20. The v2 counts were 308, 27, 573, 129, 17, 18, 111, and 15. The routed share per event is measured on the January v3 LLM leg.

## Validate the change

The gold set holds 29 January REQNOs: multi-event red-cell orders (including declaring orders where the MSBOS gate precedes the exemption), #249 cutoff shapes, multi-event platelet orders, single-event controls, no-issue controls, and one `audit_orders`-excluded control. It expects 66 event rows and 3 REQNO-level rows. Expected values were computed by a builder outside the repository with production Hb semantics, independent of the new package. Unit tests use synthetic fixtures only.

Acceptance for the gold run: event count, anchor, unit counts, linkage tag, linked REQNO, `declared_use_applies`, and `prior_event_cutoff_local` match for every event, and Hb matches within 0.05 g/dL. The deterministic dry run matched 69 of 69 gold rows. The live gold run (2026-11-27, main 6c3cfda) issued 28 of 29 privacy requests, made 28 model calls plus one re-ask, and the owner reviewed the page.

Full January v3 invariants: 1,812 event rows from 1,178 REQNOs, 5 `no_issue_record` rows, and 15 `excluded` rows whose REQNOs hold 18 measured events (1,830 in all). Each event has exactly one linked order and one attributed `DCTREQ` or the unattributed sentinel. Every v2 row maps to at least one v3 event or a REQNO-level row. The PR 8 dry run met these counts and mapped 1,198 of 1,198 v2 rows.

## Delivery

PRs #279 to #287 delivered the change in this order: units and clustering, linkage, event identity, event anchors and the #249 cutoff, per-event returns, the deterministic leg, the LLM leg, rankings and attribution, then the review page and `compare_runs.py`. This record and the documentation updates are PR 9.

After PR 9 merges:

1. Re-pin. The owner approves the merge commit as the 2025 code pin. The privacy `policy.json` does not change, because it binds the signing key (`key_id`, `public_key_file`) and `approved_redactor`, none of which change. The owner acknowledges the two changes to what is signed: Day 0 moves from the order date to the issue date, and `audit_id` changes formula. Do not run the review stage on the old pin for an event run.
2. Gold run on the new pin, owner review.
3. January v3: all five stages, then `compare_runs.py` against v2. The owner decides 8.2, 8.3, and 8.15 from the counts. If a default changes, bump the base token, use a fresh run id, re-run the deterministic leg in full, and re-ask every event on the re-ask list through prepare, issue, and llm.
4. February to December, one month at a time, with no code change unless the owner approves a re-pin.

## Open items and limits

- **8.1 window.** 4 h with the order split. A 6 h window gives 1,760 January events and merges 41% of 4 to 6 h red-cell gaps that carry a new Hb. An Hb-based split is not recommended, because a routine CBC drawn mid-transfusion would judge the second unit on the post-first-unit Hb.
- **8.2 first-event attribution.** Own REQNO. Under the #276 rule about 15% of first events would link to another order.
- **8.3 later events with no fresh order.** Own REQNO, tag `standing_order`, no declared-use carry-over. Blood left the bank with no new order recorded.
- **8.6 Hb time.** Draw time, not result time.
- **8.7 platelets.** Anchored on the issue time like red cells (clinician ruling 2026-11-09: dispense time).
- **8.8 month.** A REQNO belongs to its `REQDATE` month, even for units issued the next month.
- **8.9 emergency fields.** `linked_order_emrtype` and `linked_order_bdlevelemr` are display columns. They change no routing.
- **8.11 application path.** `bba audit`, `bba report`, and `bba serve-dashboard` stay per REQNO. See [module contracts](CONTEXT.md#application-path-stays-per-reqno).
- **8.13 prompt identity.** Event facts change `prompt_hash` for every row, so v3 differs from v2 for two reasons at once. `compare_runs.py` counts first events whose Hb equals the v2 Hb.
- **Focus-note window.** The model sees `IPDNRFOCUSDT` notes in [issue - 24 h, issue + 24 h], capped at 5 before and 5 at or after the issue. The review page shows a wider span. See the [pilot README](../scripts/pilot/README.md#what-the-review-page-shows).
- **Out of scope.** Thai dates reaching the model (#269), result-time Hb, administration proof, identifying standing-order events from nursing notes, and human labels keyed by REQNO.
