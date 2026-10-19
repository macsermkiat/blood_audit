# Accepted repair architecture

Candidate A is the base. Candidate B contributes immutable exact-request manifests as the authority for reuse and recovery. The independent cross-judge selected the same base. Keep the current classifiers, stores, quote grounder, redactor, and FastAPI app. No new workflow engine.

## Contracts

- Production evidence admission requires verified deidentification provenance bound to the exact bytes, successful privacy checks, real model identity, and a configured trusted verification mechanism. Missing or failed proof stops submission. Offline synthetic transports remain available for tests.
- A batch manifest binds exact canonical request payloads, model, schema, run identity, and membership. Resume rejects drift and preserves original request provenance.
- LLM calls persist their original attempt sequence. Recovery selects by the same ordering, code version, verifier, and evidence as uninterrupted execution. Repeated runs return committed work without re-submission or false orphans. Uncertain remote submission is explicit, never silently retried.
- Reports summarize every actual model used while retaining authoritative row provenance. Truly run-wide policy identity remains enforced.
- Review actions refer to a complete immutable audit-row identity and the previous decision revision. The existing Postgres store atomically compares and appends. An authenticated reviewer supplies agree, override, or escalation; a form cannot supply reviewer identity. Overrides carry classification and rationale. Stale submissions return conflict and retries are idempotent.
- The queue exposes all unresolved cases requiring review and keeps component-specific interpretation explicit. Evidence, Thai reasoning, decision history, and Bangkok display times support the completed reviewer journey.
- One supported composition assembles existing modules for CLI execution. Pilot callers share privacy and request boundaries. Production auth and live submission require explicit configuration; tests prohibit live providers.

## Implementation units

1. Recovery owner repairs audit_pipeline pipeline/replay/resume, ordered-call persistence, verifier wiring, full identity, cost boundary, and idempotency. Own audit_pipeline existing files and audit_store call schema. Coordinate any interface changes before integration.
2. Privacy owner repairs deidentification admission, request identity and transport caches. Own pilot live runner/CLI transport, deid modules, llm_client, and focused tests. Avoid audit_pipeline source files owned by recovery.
3. Dashboard owner repairs queue, adjudication persistence and concurrency, view evidence, auth boundary, version selection, assets, and security headers. Own dashboard/review_actions and their migrations/tests.
4. Root repairs report provenance, production composition/CLI, golden/delivery gates, integration, documentation, and acceptance evidence. Root reviews every imported diff and obtains independent code/security review.

Workers use separate worktrees. Changes integrate only after focused regression checks and review of the diff. All application policy values remain unchanged.
