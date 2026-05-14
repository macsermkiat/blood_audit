# blood_audit — KCMH RBC Transfusion Audit Pipeline (Phase 1)

Post-hoc audit of inpatient adult RBC transfusion appropriateness at Chulalongkorn Hospital, against KCMH PR 17.2 + AABB 2023 guidelines.

**See [issue #1](https://github.com/macsermkiat/blood_audit/issues/1)** for the full PRD. Implementation tickets are #2–#8 and #16–#29.

## Workflow per ticket

```
./scripts/ralph_ticket.sh <N> <module_path>
```

Generates the three-step Claude Code workflow:
1. `/tdd` — scaffold failing tests + interface (RED)
2. `/ralph-loop` — iterate red→green→refactor until `<promise>ISSUE-N-COMPLETE</promise>` fires
3. `/codex:review` — independent review on the diff

Then `gh issue close N` if review passes.

## Build order

| Ticket | Module | Depends on |
|--------|--------|------------|
| #2 | M0 — KCMH IT procedure-table export request (manual) | (none) |
| #3 | `bba.ingest` | (none — foundation) |
| #4 | `bba.audit_orders` | #3 |
| #5 | `bba.hb_lookup` | #3 |
| #6 | `bba.vitals_extractor` | #3 |
| #11 (was #18) | `bba.quote_grounder` | (none — pure function) |
| #16 | `bba.evidence_bundle_builder` | #3, #5, #6 |
| #19 | `bba.audit_store` | #3 |
| #17 | `bba.deid_redactor` | #16 |
| #7 | `bba.cohort_detector` | #3, **BLOCKED on #2** |
| #8 | `bba.deterministic_classifier` | #4, #5, #6, #7 |
| #21 | `bba.prompt_builder` | #17 |
| #22 | `bba.llm_client` | #21 |
| #20 | `bba.eval_harness` | #19 |
| #23 | `bba.confidence_calibrator` | #22 |
| #18 | `bba.quote_grounder` | (none — pure function) |
| #25 | `bba.review_actions` | #19 |
| #24 | `bba.audit_pipeline` | #8, #16, #17, #22, #18, #19 |
| #26 | `bba.dashboard` | #19, #25 |
| #27 | `bba.monitoring` | #20, #24 |
| #28 | `bba.report_generator` | #19 |
| #29 | `bba.cli` | #24, #20, #28, #26, #27 |

Modules without deps can be built in parallel.
