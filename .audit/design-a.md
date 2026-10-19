# Candidate A: repair the existing composition

## Usage first

```python
# Supported CLI composition; synthetic transport is the default.
result = execute_audit(source, settings=settings, dependencies=dependencies)
# Restart uses persisted exact requests and the same verifier configuration.
report = resume_on_startup(..., verifier=dependencies.verifier)
# Dashboard handler supplies authenticated identity, never a form reviewer_id.
action = reviews.record_action(command, expected_previous_action_id=revision)
```

`bba audit` performs ingest through persisted classifications; `bba report`, `bba evaluate`, `bba sentinel`, and `bba serve-dashboard` compose existing modules against that run. Pilot commands delegate to the same evidence admission and request preparation functions. Production startup requires configured privacy verification, authentication, stores, attribution, and explicit live-transport authorization. Tests always reject live transports, including subprocess CLI transports.

## Problem and grounding

Keep the frozen models, classifiers, redactor, quote grounder, Parquet audit store, PostgreSQL batch/review stores, and FastAPI app. `cli/main.py::_run_audit_pipeline` currently stops after ingestion; `run_pipeline` receives preassembled contexts. `replay.py::apply_batch_results` is already the shared decision path but manufactures attempt positions and defaults to success verification. `resume.py` reconstructs requests and selects calls by run alone. `ReviewActionsStore.record_action` already owns append-only decisions but lacks row/version identity and concurrency checks. Dashboard authentication currently exposes an unwired dependency seam, not an implemented production login.

## Shape and invariants

Extend existing frozen models rather than introduce a new framework:

```python
class RunIdentity:
    run_id: SafeId
    code_version: str
    manifest_sha256: str  # inputs, policy, features, verifier version

class VerifiedEvidence:
    chunks: tuple[EvidenceChunk, ...]
    redaction: RedactionResult
    evidence_sha256: str
    verification_artifact_sha256: str

class RequestIdentity:
    run: RunIdentity
    audit_id: SafeId
    request_sha256: str  # exact canonical outbound payload

class AttemptIdentity:
    request: RequestIdentity
    attempt_index: int
    call_id: SafeId

class ReviewActionInput:  # extend existing model
    audit_id: SafeId
    run_id: SafeId
    code_version: str
    row_sha256: str
    reviewer_id: SafeId
    action: ActionKind
    override_classification: Classification | None
    override_reason: str | None
    submission_id: SafeId

def execute_audit(source, *, settings, dependencies) -> PipelineRunResult:
    raise NotImplementedError

def admit_evidence(result: RedactionResult, artifact) -> VerifiedEvidence:
    raise NotImplementedError

def record_action(command, *, expected_previous_action_id) -> ReviewAction:
    raise NotImplementedError
```

Per **boundary-discipline**, evidence admission validates artifact provenance, configured verifier identity, successful redaction/privacy checks, and the hash of the exact submitted chunks. A caller-written boolean or self-consistent hash is insufficient. Missing, failed, altered, or placeholder artifacts cannot reach transport; produce an explicit privacy-review outcome. Reuse `RedactionResult` canonical validation and preserve semantic-degradation/k-anonymity routing.

One context builder inside `audit_pipeline` composes existing ingest/order/lookups/evidence/redaction modules. It owns data mapping, not clinical rules. One production entry point hides that wiring, resource lifetime, persisted run identity, and authorization policy; callers expose only source and settings. No second engine in the pilot scripts.

Persist an immutable exact-request manifest before submission. Batch identity derives from run identity and ordered request hashes. Both transport caches and recovery validate exact membership, model, schema, and request bytes. Persist attempt identity when the attempt is created, never infer ordering from filenames. Extend `LlmCall` and batch metadata with these fields. Recovery reads by full identity, sorts by original index, uses the same grounding adapter/configuration, and preserves original call bytes. Unprovable legacy provenance is quarantined rather than guessed. Missing verifier fails closed.

Batch creation becomes compare-or-return for identical manifests and rejects conflicting reuse. Orphans equal expected active cases minus committed cases for this identity. A crash after remote acceptance but before recording its ID remains uncertain: reconcile provider identity or require operator resolution; never claim exactly-once remote submission without provider support.

Report footers summarize sorted model provenance/counts, preserving deterministic and per-row model values; retain equality checks for truly run-wide fields. Dashboard reads an explicit code version and uses full row identity in URLs/actions. Queue all unresolved `needs_human_review` cases, with separate RBC/platelet presentation and metrics.

Extend existing review storage with a migration: atomic compare-and-append under a PostgreSQL transaction keyed by immutable row identity; unique submission IDs make retries idempotent. A stale prior decision returns 409. Decisions for other row hashes/versions never close current cases. Agree/override resolve; escalation stays open; exemplar marking does not resolve. Render history, cited evidence, both languages, component-specific inputs, and Bangkok timestamps. Install a real verified credential-to-reviewer dependency in production, with authorization and CSRF protection; reject default-reviewer production configuration.

## Phases and ownership

1. Privacy/composition owner: `audit_pipeline` context builder, `cli`, pilot adapters, redaction admission. Synthetic outbound-payload and command-flow tests prove R1/R7.
2. Identity/recovery owner: pipeline/replay/resume, batch and audit store models/migrations, CLI transport. Crash-window equivalence and rerun tests prove R2–R5/R10. Agree shared contracts before parallel edits.
3. Review owner: `dashboard`, `review_actions`, migrations. Real PostgreSQL concurrent/stale submissions plus browser reload/auth/component tests prove R8/R9.
4. Reporting/delivery owner: `report_generator`, CI/tests/docs. Mixed models prove R6; diagnose golden serialization before updating expectations, enforce 80% coverage and mandatory database tests for R11. Vendor assets, apply CSP/private cache headers, test version selection, and fix listed typing/duplicate-constant debt surgically.

## Tradeoffs, alternatives, risks

We accept additive schema migrations and explicit legacy rejection for reproducibility. A new event-sourced workflow engine hides transitions but duplicates existing stores and migration burden; rejected. Independent pilot patches expose privacy/provenance rules to every caller; rejected. Candidate A remains a synthesis input; no implementation deviations exist yet. Unresolved deployment choices are trusted privacy-artifact issuer and authentication provider; implementation must configure real adapters, not placeholders. Existing user edits remain authoritative.
