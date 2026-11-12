# Repair verification

The eleven application-review findings and named hardening items are repaired in the local checkout based on HEAD `c49936590e9a77e56f1b613286f0e43de3050cd3`. This is a software verification record, not clinical validation or deployment certification. No commit, push, deployment, paid provider call, or real-patient-data validation was performed. Pre-existing user changes were preserved.

## Final automated evidence

`BBA_REQUIRE_POSTGRES_TESTS=1 .venv/bin/pytest -q --tb=short -k 'not TestRealDataReconciliation' --cov=src/bba --cov-fail-under=80 --cov-report=term --cov-report=json:.audit/fix-coverage.json --junitxml=.audit/fix-pytest-results.xml` returned **4,369 passed, 3 deselected, zero skipped**, with 148 warnings. Statement coverage is **96.47%**, or 11,232 of 11,643 source statements. Branch coverage was not measured. The three deselected tests require real reviewed workbooks and hospital exports; they were deliberately excluded. Existing fixture serializer warnings remain visible. The log is `.audit/fix-full-tests.log`; the machine-readable results are `.audit/fix-pytest-results.xml` and `.audit/fix-coverage.json`.

Ruff lint passed. Ruff formatting passed for 376 files. Strict mypy passed for 218 source files. `git diff --check` passed. Logs are `.audit/fix-final-ruff.log`, `.audit/fix-final-format.log`, and `.audit/fix-final-mypy.log`. These checks were rerun after the private run-marker correction. An earlier integrated run failed on that correction's partially migrated record_row path; the final stable run above supersedes it.

Docker-backed tests ran against disposable PostgreSQL 16, with infrastructure required rather than silently skipped. The full raw-CSV cassette journey passed with both the in-memory seam and the production deferred Postgres adapter. Its Postgres case uses the unprivileged audit_pipeline_app role. It commits deterministic and admitted model results, renders all three report formats, evaluates reviewed labels, and persists monitoring comparisons. The reviewer browser journey independently used the unprivileged review_actions_app role.

`uv build --out-dir /tmp/bba-reviewed-dist` produced the wheel and source archive. Wheel inspection confirmed the application package and local dashboard CSS/JavaScript. Root Alembic scripts are not bundled into the standalone wheel; documented operations require the source checkout and matching migration project. No clean isolated install or wheel-only dashboard deployment was claimed.

## Requirement evidence

| Requirement | Implemented contract and proof |
|---|---|
| R1 | Ed25519 public-key-only admission authenticates exact prompt/run/audit/redactor proof before API or CLI submission. Missing or rejected evidence commits local review. `test_privacy_admission.py`, `test_application*.py`, and `review-privacy-closure.md`. |
| R2 | Normal and recovered results use the same fail-closed quote grounder. Fabricated citations remain rejected. `test_recovery_integrity.py` and `review-recovery-final.md`. |
| R3 | Original attempt indices, context identity, calls, and complete attempt checkpoints survive crashes and choose the same winner. `test_recovery_integrity.py`. |
| R4 | Run, batch, code version, effective component policy, and context identity bind recovery and completion. Prior-version results cannot complete a new batch. `review-recovery-final.md`. |
| R5 | Canonical exact-request manifests reject changed model/evidence/membership. Original request provenance is retained. SDK retries are disabled; uncertain API or CLI outcomes cannot automatically execute again. Real SDK response-loss probes and `test_request_durability.py`. |
| R6 | Mixed deterministic/model/prompt runs produce truthful summaries and per-row provenance. Conflicting applied policies/redactors still fail. Unused and rejected redactor claims use explicit not-applied/unavailable sentinels. `test_report_builder.py` and privacy closure review. |
| R7 | Supported `bba audit` seals and validates the same canonical source bytes it classifies, composes existing modules, requires privacy admission, and persists real AuditRows. Report/evaluate/sentinel/authenticated dashboard commands work. Seven source-drift, corruption, and resume regressions plus actual Postgres cassette journey. `docs/production-cli.md` and production closure review. |
| R8 | Queue includes every unresolved review flag and both components. Aggregate scorecards explicitly retain red-cell scope. `test_dashboard_adjudication.py`. |
| R9 | English/Thai reasoning, citations, Bangkok times, agree/override/escalate, evidence-bound history, authenticated identity, idempotent submissions, and stale conflicts are implemented. Real Chromium forms and Postgres history verified in `browser-verification.md`. |
| R10 | Repeated completed and interrupted runs preserve immutable rows, avoid duplicate batches, and report no false orphans. Run/source binding is private and durable. `test_recovery_integrity.py`, `test_application_snapshot.py`, and `test_cli.py`. |
| R11 | The golden mismatch was isolated to the prior model rename from claude-sonnet-5 to claude-sonnet-5-5; substituting only that serialized value reproduced the old digest. The corrected golden retains semantic assertions. CI enforces 80% coverage and required Postgres infrastructure; fixture setup/teardown faults propagate. `review-ancillary-final.md`. |
| Named hardening | Version-selected snapshots, local UI assets, CSP/private cache/origin protections, shared returns constants, and literal category types are covered by source checks and focused review. Actual browser testing corrected no-referrer to same-origin because Chromium otherwise emitted Origin null on valid form submissions. |

## Independent review

Privacy/reporting, recovery/storage, dashboard/adjudication security, ancillary source/delivery changes, and production composition security have closure approvals. The reports are `review-privacy-closure.md`, `review-recovery-final.md`, `review-dashboard-security.md`, `review-ancillary-final.md`, and `review-production-closure.md`. The final code-reviewer pass over composition and CLI is approved in `review-composition-code-final.md`; its history-construction warning was repaired and independently closed. The comment-only review found one wording correction, which is applied. Cross-model decision-trail audit completed on GPT-5.6 in `review-trail-final.md`. It found an incorrect closure pointer, an omitted failed verification checkpoint, and a small late history-optimization scope expansion. Append-only audit rows correct the pointer and checkpoint and acknowledge the scope expansion. Final measured evidence reconciles.

## Operational limits

The approved redactor model, real cohort anonymity gate, issuer private-key isolation, provider account, migrated databases, authentication adapter, HTTPS reverse proxy, and hospital adult-cohort export controls remain operator responsibilities. Synthetic tests do not certify clinical decisions or deidentification quality. Live provider execution, external identity-provider integration, deployment, load/capacity testing, power-loss tests, and actual patient-data validation were not performed.

Committed audit results remain immutable. To revisit a locally privacy-rejected case after approval, use a new approved application policy version and prepare proof for the new run. Force does not rewrite a committed result. Cassette contents remain fixed per configured run. Unsafe legacy file permissions and symlink storage are rejected, and sealed clinical snapshots add retained source copies that require an operator retention policy. Unredacted identifier resolution remains unavailable through the supported CLI unless an authorized resolver is supplied to the app factory.

The final rebuilt wheel was inspected after the last code change. Its contexts.py, source_snapshot.py, and phi_scrubber.py bytes match the final workspace source. The wheel SHA-256 is `293f02daa201d75a5ad060a9a5a4e0075960ed9d09ae2fbf4f16f410ebc15068`; the source archive SHA-256 is `142d743105af0b14057e2860d6ba3eeb2eaec6857785df909e6ffab1c8d969c0`. They remain local under `/tmp/bba-reviewed-dist`.

## Principles that changed the work

Model the Domain and Foundational Thinking led to explicit immutable source, request, attempt, and decision identities before orchestration. Minimize Reader Load and Laziness Protocol kept the existing modules as the base instead of adding another workflow engine. Separate Before Serializing Shared State led to unique atomic temporary files and database decision locks. Sequence Work into Verifiable Units separated privacy, recovery, reporting, reviewer, and CLI acceptance gates. Never Block on the Human kept reversible repairs and synthetic verification moving without repeated approval questions. Encode Lessons in Structure put coverage, required database execution, signed admission, and uncertain-submit refusal into executable gates. Prove It Works required the actual SDK response-loss probes, Postgres integration, and Chromium forms that exposed defects missed by green unit tests.
