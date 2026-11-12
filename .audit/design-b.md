# Candidate B: immutable manifests own run identity

## Problem

`run_pipeline` receives already assembled `PipelineRowContext` values, but pilot code can assert deidentification without verifying it; recovery rebuilds requests, loses the verifier and attempt order, and the CLI cache accepts changed inputs. The audit store already keys commits by code version; `BatchRun` records run/version/audit IDs, but no durable exact request identity. This design makes an immutable run manifest and per-batch request manifests the authority for execution and recovery. Existing classifier, redactor, prompt, store, and reviewer modules retain their clinical and storage responsibilities.

## Usage (caller's view)

```python
# bba audit --input bundle_dir --transport batch --run-id optional
plan = prepare_run(ingested_bundle, policy, deidentifier, verifier)
receipt = execute_run(plan, transport=approved_transport, stores=stores)
receipt = resume_run(plan.run_id, stores=stores, transport=approved_transport)

# bba report --run-id <id>; dashboard GET /queue, POST /case/<id>/decision
report = build_report_inputs(run_id=plan.run_id, audit_store=stores.audit,
                             manifest_store=stores.manifests, ...)
```

`prepare_run` is pure with respect to transport. It either returns a sealed plan or a typed privacy/identity error; no request is sent. `execute_run` and `resume_run` use the same sealed plan and the same required citation verifier. The CLI completes only when every expected audit row is committed at the plan's code version. The pilot runner uses this preparation path; its sampler defaults to a verified deidentified bundle and refuses raw narrative input without a successful local verification artifact.

## Shape

```python
@dataclass(frozen=True)
class RunManifest:
    run_id: str
    code_version: str
    input_digest: str
    policy_digest: str
    privacy_verification: PrivacyProof
    expected_audit_ids: tuple[str, ...]
    batches: tuple[BatchManifest, ...]

@dataclass(frozen=True)
class BatchManifest:
    batch_id: str
    run_manifest_digest: str
    requests: tuple[RequestRecord, ...]  # canonical order, exact membership

@dataclass(frozen=True)
class RequestRecord:
    audit_id: str
    attempt_index: int
    model_id: str
    schema_digest: str
    request_digest: str
    request_bytes: bytes  # exact outbound bytes; protected local storage
    context_digest: str

def prepare_run(bundle, policy, deidentifier, verifier) -> RunManifest: ...
def execute_run(manifest: RunManifest, *, transport, stores) -> PipelineRunResult: ...
def resume_run(run_id: str, *, transport, stores) -> ResumeReport: ...
```

`audit_pipeline/manifest.py` owns canonical serialization, hashes, immutable writes, and equality checks. Reuse `BatchRunStore` for mutable checkpoint state, extended with manifest digest; one batch ID derives from the immutable run digest and ordered request digests. In `pipeline.py`, prepare prompts through existing `build_prompt` and `build_anthropic_request`, seal their bytes, then submit those bytes through the existing transport adapter. In `resume.py`, load the manifest and stored redacted context snapshot; reject differing version, membership, or request digest. Replay stored attempts ordered by persisted `attempt_index`, preserving original `call_id`, request bytes, response bytes, and model ID. Keep `replay.py` decision logic and make the same non-optional verifier available to normal and resumed application; a missing verifier blocks completion. `AuditStore` remains the row/call owner; it adds exact batch/version reads, never infers order from filenames. Repeated completed runs return their committed IDs; partial runs reconcile only missing work. If remote submission succeeded but its ID was not checkpointed, stop for reconciliation rather than risk a duplicate submit.

`PrivacyProof` records the exact redacted evidence digest, source digest, deidentifier model/version digest, local PHI verification method/version and pass result. Preparation runs deidentification and a real local PHI check on the exact prompt-bound evidence; known identifiers, dates, failed/indeterminate checks, placeholder hashes, or changed bytes block transport and route the case for local review. No caller can supply an unchecked `PipelineRowContext` to the supported production entrypoint. The proof binds to each request, while raw source stays in the protected ingest store.

Reporting uses the manifest for run-level policy/version and a set of actual per-row model IDs; it drops only `model_id` from the uniform-footer assertion and changes footer/CSV/PDF schema to represent mixed provenance truthfully. The dashboard selects the manifest's version in `SnapshotView`, queues every unresolved `needs_human_review` row, and states whether platelet cases live in this queue or a separate one. Case view renders stored citations, indication quote, Thai reasoning, and `Asia/Bangkok` times. `ReviewActionsStore` remains the sole decision writer: extend its append-only action with run/version, current row digest, expected latest action ID, and transactional compare-and-insert. The POST uses authenticated reviewer identity, validates case access and CSRF protection, returns conflict on stale state, and reloads chronological history.

## Tradeoffs and alternatives

- We accept durable canonical request bytes and a schema migration in exchange for exact recovery and auditability; apply existing protected storage permissions and retention rules.
- We accept a stopped batch after ambiguous remote submission in exchange for preventing unverified duplicate calls.
- Patching `run_pipeline`/`resume` parameters separately is smaller today but leaves request identity and provenance split across callers. A new event-sourced framework would centralize history but duplicates `AuditStore` and `BatchRunStore`; reject it.

## Atomic implementation and verification

1. **Manifest and privacy owner:** add immutable manifest storage and production preparation; bind pilot sampler/runner and CLI ingest-to-context assembly to it. Test synthetic PHI and changed bytes before transport.
2. **Pipeline/recovery owner:** bind batch state and CLI cache to manifest digests; persist ordered attempts; require verifier; make reruns idempotent. Compare uninterrupted and crash-window rows, winners, calls, and bytes, including version reuse.
3. **Report owner:** migrate provenance schema and writers; test deterministic-only, mixed, multi-model, version collision, and retained row provenance.
4. **Dashboard/review owner:** add queue coverage, case evidence, authenticated decision POST/history, atomic stale-write rejection, versioned snapshots, and local static assets/CSP/cache headers. Exercise complete synthetic browser journey.
5. **Delivery owner:** diagnose the existing golden byte drift before updating it; enforce 80% source coverage; make a required Postgres job fail on fixture faults or zero collected integration tests. Run lint, format, mypy, full synthetic suite, UI, and database tests; report any unavailable gate explicitly. Handle comment/type debt only where touched.

## Open risk

Confirm the available local PHI detector's failure semantics and the remote batch API's idempotency support before enabling live submission. No live request or patient-data validation is part of this design.
