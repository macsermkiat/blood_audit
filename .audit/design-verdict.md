# Design verdict

Use **Candidate A as the implementation base**, with Candidate B's immutable run/request manifest as an internal persistence contract. The review favors one supported composition that reuses the existing audit, replay, store, report, and review modules. Candidate B gives the clearest exact-request and recovery identity, but elevating `prepare_run`/`execute_run`/`resume_run` and manifest handling into a new public workflow surface adds coordination and risks duplicating existing responsibilities. Keep A's single `execute_audit(source, settings, dependencies)` entry point and make manifests an implementation detail shared by execution and resume.

| Candidate | Privacy enforcement | Exact recovery equivalence | Fit with existing modules | Full reviewer/CLI completion | Small public interface |
|---|---:|---:|---:|---:|---:|
| A | 4 | 4 | 5 | 4 | 4 |
| B | 5 | 5 | 4 | 5 | 3 |

A's advantage is its close fit to the modules and the review's request for one supported composition. B's advantage is stronger articulation of immutable request bytes, privacy proof binding, and versioned recovery. Neither design alone fully specifies the trust root for privacy verification or resolves the remote-acceptance crash window; both correctly require operator/provider reconciliation rather than promising exactly-once delivery. B's `resume_run(run_id, ...)` should also load authoritative identity and policy from persisted state rather than trust a caller-held plan, while A's resume contract should specify the same.

## Grafts and contract refinements

1. Graft B's immutable manifest fields into A's existing run/batch persistence: run and code identity, manifest digest, ordered expected audit IDs, and per-request exact canonical bytes/digest, model, schema, context digest, attempt index, and original call ID. Persist attempt identity at creation. Require full run/version/batch/request identity for cache reads, completion checks, and orphan calculation. Identical manifests return existing state; conflicting reuse fails. Retain original request and response bytes and quarantine legacy records whose provenance cannot be proved.
2. Tighten the privacy boundary before implementation: only a configured verifier may issue admission for the exact redacted evidence and exact prompt-bound chunks. Define the artifact schema, issuer trust/configuration, verification version, and behavior for missing, failed, indeterminate, altered, or placeholder evidence. The production composition must derive contexts through this boundary; caller booleans and unchecked `PipelineRowContext` values cannot authorize transport. Synthetic identifiers must be tested against emitted outbound bytes.

The public contract should make the citation verifier mandatory and identical for normal execution and every recovery path. Recovery must retain attempt order and call IDs and reproduce classification, review reasons, and verification fields. Batch/CLI cache matching must include exact ordered membership, request bytes, model, and schema. A remote-acceptance/no-checkpoint state remains `uncertain` until provider reconciliation or operator resolution.

For reviewer completion, extend the existing append-only recorder with an atomic compare-and-append keyed by row hash plus run/code version, expected prior action, authenticated reviewer, and idempotent submission ID. Define which actions resolve a case (agree/override) versus leave it open (escalate/exemplar). The dashboard must use that identity for stale-write rejection and version-selected reads; queue every unresolved review flag and state platelet routing.

Include the review's named hardening in acceptance: truthful mixed-model report footers with row provenance retained; dashboard version selector; local versioned static assets and CSP/private cache headers; diagnose the golden byte drift before changing expectations; enforce 80% coverage and a Postgres job that cannot pass through fixture skips; handle typing/comment debt only where touched. Keep existing live-transport test/cost guards and require explicit production authorization.
