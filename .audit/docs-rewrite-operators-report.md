# Operator documentation rewrite

Updated 2026-10-19 from the current source checkout. All changes are uncommitted in `/tmp/bba-docs-operators`. Apply `/tmp/bba-docs-operators.patch` to the integration branch.

## Changes

Rewrote all 22 English and Thai operator pages. They put task instructions first, separate the supported application from research scripts, and remove repeated AtAGlance cards and obsolete Phase 1.5 status. Existing page slugs remain stable. Thai instructions retain exact technical identifiers.

The local tutorial now describes committed deterministic AuditRows and local privacy review results, rather than ingest-only markers. It separates optional ingest validation, audit completion, and RBC report generation. It warns before audit execution that the disabled platelet review route can leave uncommitted work and prevent completion.

Model guidance covers the validated application JSON, plain `postgresql://` DSN, migrated checkpoints, privacy trust policy, raw Ed25519 public key, exact signed admission artifacts, and the external approved redaction authority. It states that the application does not receive the private signing key or install a hospital redactor. Live mode is separate and explicitly chargeable.

Dashboard guidance covers loopback binding behind HTTPS, the externally visible origin, 32-byte minimum shared secret, verified Reviewer identity, signed sessions with a maximum one-hour expiry, cookie flags, CSRF, and revision conflicts. It describes unredacted lookup as an app-factory deployment dependency. The review-store integrity checks happen before operating, rather than claiming that opening the listening server proves database readiness.

Report guidance distinguishes HTML, JSON, and PDF artifacts, explains ordering-doctor department attribution, per-row provenance, RBC-only report scope, mixed-month and metadata rejection, and recipient review. Evaluation and monitoring pages give the actual reviewed-label and explicit comparison inputs. Immutable completed runs, retained source snapshots, and uncertain submission reconciliation have explicit recovery instructions.

Research pilot ranking retains the merged-report denominator warning, script-specific defaults, explicit input variables, per-component outputs, red-cell attribution behavior, and source/version scope. It no longer implies that the pilot's partial model audit store is the supported application's full audit store.

The environment presence-check examples print names and set/unset status, never values. EnvVarChecklist localizes its labels and descriptions from the page path. Its default values use current configuration rather than phase status.

Deleted `CliCommandCard.astro` and `IntegrationSeamCard.astro`. A search of the entire worktree `docs/src` found no remaining imports or usages after the operator rewrite. Both were used only by these operator pages. Keep the current EnvVarChecklist, which both environment pages use.

## Source comparisons

- `src/bba/cli/main.py` supplies the six commands, options, run completion/no-op logs, forced-run compliance record, report formats, attribution inputs, evaluation, monitoring, and dashboard composition.
- `src/bba/application/config.py`, `audit.py`, `operations.py`, and `source_snapshot.py` supply transport validation, admission input loading, completion checks, immutable analysis persistence, source identity, sealed snapshot modes, and recovery bindings.
- `src/bba/deid_redactor/admission.py` supplies the Ed25519 admission model and approved redaction authority described in the production guide.
- `src/bba/audit_pipeline/pipeline.py` and `src/bba/feature_flags.py` confirm the default-disabled platelet review route and orphan handling.
- `src/bba/ingest/pipeline.py`, `normalize.py`, and `schemas.py` distinguish header projection and standalone normalization markers from audit persistence.
- `src/bba/report_generator/builder.py` confirms RBC-only monthly reports and their metadata and local-month checks.
- `src/bba/dashboard/auth.py`, `app.py`, and review-action storage supply signed sessions, CSRF, revision checks, and database integrity guards.
- `scripts/pilot/rank_doctors.py`, `rank_merged.py`, and the pilot README supply the separate ranking and model-review workflows.
- `docs/production-cli.md` is the current operational baseline. No clinical policy thresholds or runtime behavior changed.

## Verification

- Current CLI help passed for audit, evaluate, ingest, report, sentinel, and serve-dashboard using the existing Python environment and this worktree's source.
- Both cassette JSON examples validate as ApplicationConfig.
- The shared plain Postgres DSN parses with psycopg without opening a connection.
- Both environment-check snippets ran against sentinel credentials. Neither printed the sentinel values.
- All 22 operator pages reference existing local documentation page targets. This check does not establish final generated anchor validity.
- No rewritten operator page contains AtAGlance, stale Phase 1.5 text, em dashes, or en dashes.
- `git diff --check` passes.
- English sloplint scores range from 0 to 20, below the target of 25. Highest is ingest-csv at 20 for legitimate three-item descriptions. Thai scanner output is not a reliable measure of natural Thai.

No live provider calls, patient-data reads, database connections, dependency installations, commits, or remote writes occurred. The application test suite was not rerun for prose-only changes. Root owns the integrated Astro build, final link/anchor check, browser review, and independent review gate.

## Changed files

All eleven slugs in each of `docs/src/content/docs/en/operators/` and `docs/src/content/docs/th/operators/` changed: audit-command, doctor-ranking, env-vars, first-audit-run, generate-report, ingest-csv, install, integration-seams, overview, run-pipeline, and troubleshooting.

`docs/src/components/EnvVarChecklist.astro` changed. `docs/src/components/CliCommandCard.astro` and `docs/src/components/IntegrationSeamCard.astro` were deleted. No files outside the assigned ownership changed.
