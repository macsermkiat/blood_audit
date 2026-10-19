# Fix the application review findings

- [x] Read the Principles section of the poteto-mode skill.
- [x] Phase A: Frame
- [x] Phase B: Design the workflow
- [x] Phase C: Run the loop
- [x] Phase D: Keep the audit trail
- [x] Phase E: Verify and hand back

## Architect

- [x] Ground
- [x] Sketch
- [x] Agree
- [x] Implement
- [x] Scrap
  skip: the accepted existing-module architecture met its verification gates; no restart was needed.

## Arena

- [x] Frame
- [x] Fan out
- [x] Cross-judge
- [x] Pick
- [x] Graft
- [x] Verify

The design rubric checks privacy enforcement, exact recovery equivalence, fit with existing modules, full reviewer and CLI completion, and a small public interface. Two independent design candidates precede synthesis.

This is high-rigor work because it handles clinical evidence, privacy, and persisted decisions. Scope is the 11 review findings plus its named hardening items, roughly five implementation units across pipeline, transport/privacy, reporting, dashboard, and CLI/delivery. Deployment and live patient-data validation are not requested. The previous review supplied the pre-change synthetic baseline and one failing golden test.

## Concrete requirements and gates

- [x] R1. Enforce privacy before any model submission and verify provenance.
- [x] R2. Use the same fail-closed citation verifier in normal and recovery paths.
- [x] R3. Preserve attempt sequence, winning result, call IDs, and stored bytes on recovery.
- [x] R4. Scope recovery and completion to code version and batch identity.
- [x] R5. Bind cached results to exact requests and original model provenance.
- [x] R6. Generate truthful reports for mixed deterministic and multi-model runs.
- [x] R7. Supply a supported end-to-end production composition and working CLI integration.
- [x] R8. Expose all cases requiring human review with explicit component scope.
- [x] R9. Show evidence and Thai reasoning, persist adjudication and history, reject stale decisions, and display Bangkok times.
- [x] R10. Make completed and interrupted orchestration reruns idempotent.
- [x] R11. Explain and repair the golden failure, enforce coverage, and make required database tests fail rather than silently skip.
- [x] Address additional report findings on snapshot version selection, external UI assets, deployment headers, and comment/type debt.
- [x] Independently review each implementation unit and fix accepted findings.
- [x] Run source checks, complete synthetic tests, real UI journeys, and database integration where available.
- [x] Audit every requirement against actual artifacts and record remaining limitations.

No production data, paid model calls, remote writes, deployment, or user-work changes are required. The full goal remains active until all findings and required verification are resolved.

## Independent review closure

- [x] Recovery and private storage approved after direct reproduction of both prior HIGH findings. See review-recovery-final.md.
- [x] Privacy and reporting approved after real SDK response-loss and malformed CLI output probes. See review-privacy-closure.md.
- [x] Dashboard adjudication security approved; migrated PostgreSQL tests and actual Chromium form journey passed.
- [x] CI, fixture failure handling, dependencies, category types, and constant extraction independently approved. See review-ancillary-final.md.
- [x] Correct canonical ISO progress timestamps and verify preserved vitals.
- [x] Withhold exception values and messages so clinical data and signing credentials cannot leak through traceback rendering.
- [x] Preserve evaluation and monitoring comparison identities with immutable output files.
- [x] Bind production classification to a private immutable source snapshot and independently verify the source-race repair.
- [x] Re-review production CLI findings against the final integrated source.
- [x] Final full synthetic suite with required real PostgreSQL, source lint/format/types, and packaging check.
- [x] Cross-model decision-trail audit and final requirement evidence map.

GPT-5.6 reconciled final evidence and flagged two trail gaps, corrected by appended entries, plus a small independently verified history optimization beyond the original scope. See review-trail-final.md and the final audit rows.
