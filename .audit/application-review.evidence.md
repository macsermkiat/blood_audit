# Evidence for the application review

Review date 2026-10-19, Asia/Bangkok. Repository HEAD c49936590e9a77e56f1b613286f0e43de3050cd3.
Scope is software review and improvement recommendations. No application changes. Existing user changes were left untouched. No patient data, live model calls, paid APIs, deployment, or clinical guideline validation.

Three read-only reviewers covered decision flow, privacy boundaries, and comment debt. Root independently reran all five core pipeline reproductions, inspected the relevant source, and exercised the dashboard with synthetic data through TestClient and Chrome.

## Verification

- `.venv/bin/ruff check .` passed.
- `.venv/bin/ruff format --check .` passed for 348 files.
- `.venv/bin/mypy src --cache-dir /tmp/bba-review-mypy` passed for 201 source files.
- `.venv/bin/pytest -q --tb=short --ignore=tests/unit/test_audit_pipeline_postgres.py --ignore=tests/unit/test_review_actions.py -k 'not TestRealDataReconciliation' --cov=src/bba --cov-report=term --cov-report=json:/tmp/bba-review-coverage.json -o cache_dir=/tmp/bba-review-pytest-cache` returned 1 failed, 4202 passed, 3 deselected.
- Coverage was 9740 of 10218 source statements, 95.32198081816402%. Branch coverage was not measured. This excludes execution of Docker-backed modules and the real-data reconciliation class.
- Failure `tests/unit/test_audit_pipeline.py:2107`, `TestReserveAheadGate.test_flag_off_case_68026306_replays_with_existing_b1_bytes`, reproduced alone. Expected SHA `77ca731682d441a07c5fbf2752d59913c3e6d84a32d200096be501e212869209`, actual `1444c8f31d8ceb6503e8b30f62ca511505aa00dd3938a511a023326d72026926`. The preceding semantic assertions passed. The cause of serialization drift is unverified. Do not recommend blindly updating the golden.
- Docker modules were deliberately excluded to avoid starting containers or fetching images. No Postgres-backed integration verdict is available.
- CI `.github/workflows/ci.yml:86` runs pytest without coverage enforcement. Postgres fixtures can turn fixture setup or teardown exceptions into skips at `tests/unit/test_audit_pipeline_postgres.py:45` and `tests/unit/test_review_actions.py:104`. This does not itself turn ordinary test-body assertion failures into skips.

## Findings

R1, P1, unsafe pilot privacy default. `scripts/pilot/sample_bundle.py:66` defaults to the encrypted export, and lines 428 onward preserve narrative. `scripts/pilot/run_llm_leg.py:2012` copies narrative into evidence. Lines 2388-2389 assert external deid with an all-zero model SHA. `scripts/pilot/README.md:171` explicitly says the default raw encrypted source contains narrative PHI and scripts do not verify deidentification. Main README quickstart omits that preparation step. Privacy reviewer reproduced `_render_payload` to `build_prompt` to `build_anthropic_request` with synthetic name, HN, and date. All survived and route_to_needs_review was false. This proves a possible unsafe documented flow, not a historical disclosure. Require a verified deid artifact bound to exact evidence and stop before transport when missing or failed.

R2, P1, recovery drops verification. `src/bba/audit_pipeline/replay.py:101` default_verifier always returns True. `resume_on_startup` at `resume.py:67` has no verifier input. Calls at resume.py:219 and :449 omit it. Root synthetic comparison using test_audit_pipeline `_row_context`, `_orphan_llm_call`, `_build_llm_call` and `_result_from_cached_call` produced normal NEEDS_REVIEW, verifier_pass=False, hallucination_suspect under a rejecting verifier. Recovering the identical cached response produced INAPPROPRIATE, verifier_pass=True, no review reason. Require the same real verifier in normal and both recovery paths, with missing verifier failing closed.

R3, P1, recovery changes winning attempt. `src/bba/audit_store/store.py:207` sorts hashed filenames. `resume.py:409` groups in that order and :439 replays it. `replay.py:1454` assigns new attempt indices. Root synthetic audit-4 in run-order, attempts INSUFFICIENT_EVIDENCE then INAPPROPRIATE, produced normal INAPPROPRIATE with reasoning attempt-1, recovered INSUFFICIENT_EVIDENCE with reasoning attempt-0. Reviewer also observed original call payloads can be overwritten when replay regenerates IDs from the reversed order. Persist original ordered attempts and preserve call IDs and bytes through recovery.

R4, P1, recovery ignores code version. `resume.py:169-172` and :409 read by run_id only. Root committed v1, created a v2 SUBMITTED batch for same run_id and audit_id, and resumed. State became COMPLETE with zero v2 rows. Version reuse is a precondition, not every run. Scope all recovery inputs and completion checks to the batch identity and reject incompatible versions.

R5, P1, stale CLI batch reuse. `scripts/pilot/_claude_cli_transport.py:195-202` manifest records model and case IDs only, :219 reuses any existing result file, :324-328 labels it with the caller's current model. Privacy reviewer used a fake runner. First submission Sonnet with evidence one, then same batch and audit ID with changed evidence and Opus. Runner called once. Result model said Opus, stored request said Sonnet and evidence one. `replay.py:1784` combines this with current prompt metadata. Bind cached results to canonical request hash, model, schema, and exact membership. Preserve original request provenance.

R6, P1, report rejects normal mixed run. `report_generator/builder.py:271-276` requires uniform model_id; :338 onward rejects mismatches. `audit_pipeline/pipeline.py:660` writes deterministic and replay.py:1780 writes the LLM model. Root ran two synthetic contexts, persisted both, then build_report_inputs raised MixedRunMetadataError for deterministic versus claude-sonnet-5-5 with all other run metadata equal. Support truthful model provenance for deterministic rows and multiple LLM models.

R7, P1 product readiness gap, production route incomplete. `audit_pipeline/pipeline.py:194` unconditionally calls cost guard, and cost_guard.py:31 rejects AnthropicBatchTransport. Root used object.__new__ without initializing any network client and received LiveAnthropicApiError. `cli/main.py:450` onward runs ingest only, :220 raises for evaluate, :283 raises for serve-dashboard, :321 raises for sentinel. These are documented unfinished seams, not stealth defects. Establish one supported end-to-end production composition while keeping tests prohibited from live calls. Do not solve by casually removing the cost boundary.

R8, P1 workflow gap, cases needing human review are absent from queue. `dashboard/app.py:301` filters only NEEDS_REVIEW rather than needs_human_review. Root wrote three valid RBC AuditRows with classifications NEEDS_REVIEW, INSUFFICIENT_EVIDENCE, and POTENTIALLY_INAPPROPRIATE, all needs_human_review=True, to a real temporary AuditStore. list_queue and Chrome showed only NEEDS_REVIEW. Exact queue behavior is documented, but it leaves a product gap in escalation coverage. Require every unresolved case to have a visible route to adjudication, including explicit component scope.

R9, P2 workflow gap, dashboard cannot complete review. Chrome case view rendered English summary and UTC ISO time, omitted known stored synthetic indication quote and Thai reasoning. `templates/case_detail.html:46-49` renders only English reasoning. App route inventory contains no reviewer-decision POST, only break-glass. `review_actions` has append-only decision storage but dashboard never calls record_action. Add evidence and citations, Thai reasoning, agree/override actions with reasons, history, concurrency and version controls, and Bangkok display times with explicit zone. Reviewer action must persist through page reload. Keep platelet and RBC clinical context explicit.

R10, P2 reliability gap, reruns are not idempotent at orchestration level. Root repeated the mixed run and got ValueError batch_id already exists. `pipeline.py:448` always creates its stable ID. Deterministic-only repeat returned already committed audit-a in orphan_audit_ids. `_detect_orphans` at :1527 compares newly written IDs with expected IDs. Verify reruns do not resubmit, create duplicate rows, or report committed rows as orphaned.

R11, P2 delivery check gap. Existing golden test fails. CI lacks coverage threshold despite 80% project requirement. Docker fixtures can convert fixture setup or teardown faults into skips. Restore a meaningful golden gate, enforce source coverage, and require a non-skipped database job. Passing source checks and statement coverage did not detect the integration defects.

## Additional measured and static observations

- dashboard `_read_snapshot_rows` calls snapshot.read_audit_results without a code_version selector at app.py:196. Snapshot explicitly supports multiple versions, so double counting after same-ID re-commit is a static risk. Do not claim this was runtime reproduced.
- Dashboard uses external Tailwind and HTMX scripts at templates/base.html:8-10. Browser confirmed they load. Prefer compiled, locally served versioned assets for deployment reliability. CSP and cache policy require deployment review. No disclosure through CDN was demonstrated.
- Auth fails closed when no reviewer configured. Own physician scorecards check exact identity. Care-team and break-glass record access before resolving raw identifiers. Do not claim an auth bypass.
- Source models are mostly frozen and clear modules exist for redaction, grounding, policy, and persistence. Preserve these boundaries.
- No-comments review made zero edits. Four maintenance flags concern internal object types with type suppressions, untyped category helpers, and duplicate returns-terminal constants. These are lower priority than the runtime faults. Do not spend the first improvement phase deleting comments.

## Principles root read in full

Prove It Works changed verification to compare actual uninterrupted and recovered rows, inspect emitted requests, and drive the actual UI with synthetic data.
Minimize Reader Load changed the recommendation to one supported production composition with explicit run identity, instead of adding more wrappers around divergent pilot and library paths.

## Expected output

A concise, candid application review with prioritized findings, source links, clear verification limits, and an ordered improvement plan. Judgment is that there is substantial module-level rigor, but observed integration and privacy failures prevent treating it as production-ready. Focus on fixing boundaries before visual redesign. Keep a detailed report locally and deliver the most important findings in chat. No merge, PR, or application edits required for this Investigation.
