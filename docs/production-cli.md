# Run a supported HOSxP audit

Set `BBA_DATA_DIR` to a persistent, access-controlled directory. Stage all 13 canonical CSV files beside `BDVST.csv`. The adult cohort prefilter remains the exporter's responsibility because the export has no age column.

```sh
export BBA_DATA_DIR=/srv/bba/data
uv run bba audit --input /srv/bba/export/BDVST.csv
```

The command validates and normalizes the export, assembles canonical orders, and calls `bba.audit_pipeline.run_pipeline`. The adapters use the existing Hb and platelet lookups, cohort detector, evidence builder, returns ledger, declared-use interpretation, MSBOS references, and feature flags. They preserve the issue-time evidence-anchor policy. They do not infer administration from a blood-bank issue timestamp.

Without model configuration, deterministic results commit to `BBA_DATA_DIR/audit_store`. Orders that require model review commit as `NEEDS_REVIEW` with `privacy_admission_failed`. No model credentials or Postgres connection are needed for this local mode. Raw notes stay local and do not enter the transport or persisted calls. The application creates an owner-readable `identity.key` for stable HMAC patient and admission identifiers. Back up this file with the data directory.

A successful audit completion counts committed AuditRows. Ingested tables never count as audited rows. An empty eligible cohort or uncommitted result fails the command without a completion marker. A platelet order suppressed within an MTP window intentionally emits no AuditRow.

Re-running the same source and configuration returns committed work. `--force` bypasses the CLI completion shortcut and records an override, while the pipeline still preserves immutable committed results and request manifests. Change the policy version or application configuration to create a new audit identity for changed analysis. `bba audit --run-id ID` can resume an interrupted run while its recorded source bundle and application configuration remain unchanged. The input record contains the local source path.

## Admit model evidence

Place a JSON application configuration in an operator-controlled file and set `BBA_APPLICATION_CONFIG` to that path. Unknown fields fail validation. For an offline cassette run, use this shape.

```json
{
  "policy_version": "your-approved-policy-version",
  "transport": "cassette",
  "cassette": "/srv/bba/cassettes/approved-synthetic.json"
}
```

For a live run, install the provider dependency with `uv sync --extra live`, set `transport` to `live`, and set `allow_live_api` to `true`. Run the audit with `uv run --extra live bba audit --input /srv/bba/export/BDVST.csv` and set `ANTHROPIC_API_KEY` for the live transport. Unit and integration tests still reject the live provider even when this option is true. The optional `enable_missing_hb_positive_evidence` and `enable_missing_platelet_defer` settings default to false. Enable them only for the corresponding approved policy.

Set `BBA_PRIVACY_POLICY` to the trust-policy JSON file. It contains `key_id`, `public_key_file`, and `approved_redactor`. `approved_redactor` contains the approved `version`, nonzero SHA-256 `model_sha`, and `gazetteer_version`. The key file contains the 32-byte Ed25519 public key. The application never receives the issuer's private signing key.

Set `BBA_PRIVACY_ARTIFACTS` to a JSON object keyed by audit ID. Each value is a `bba.deid_redactor.admission.PrivacyArtifact` with exact redacted `evidence_chunks` and an `attestation`. An approved local redaction service produces this artifact through `RedactionAuthority.prepare`. That service owns the approved backend, model artifact, k-anonymity gate, and private signing key. Do not replace it with a pass-through backend for hospital data.

The attestation binds the exact prompt, audit ID, run ID, approved redactor version, and passed privacy checks. Compute the run ID with the same bundle, package version, and application configuration used for execution. An explicit application configuration contributes its canonical JSON to the CLI run identity. Missing policy or rejected evidence results in local review. A digest or caller assertion does not authorize submission.

Set `BBA_DB_URL` to a migrated Postgres database for admitted cassette or live model runs. The batch checkpoint store uses this DSN. Configure a role with the permissions required by the batch migrations. Request manifests and ordered calls preserve recovery identity. An uncertain remote submission requires operator reconciliation and is never silently retried.

## Render a report

Set `BBA_BDVST_CSV` to the corresponding BDVST export and `BBA_DCT_CSV` to its doctor registry. The report uses `DCTREQ` to attribute a physician and department. The department fills the report's existing ward grouping.

```sh
export BBA_BDVST_CSV=/srv/bba/export/BDVST.csv
export BBA_DCT_CSV=/srv/bba/export/DCT.csv
uv run bba report --run-id RUN_ID --format html
uv run bba report --run-id RUN_ID --format json
uv run bba report --run-id RUN_ID --format pdf
```

Each command prints the matching artifact path. HTML contains escaped row data and provenance. JSON contains report inputs and row provenance. PDF uses the monthly report generator and its committee and physician CSV artifacts. Mixed deterministic and model results retain per-row provenance. A report still rejects conflicting applied policy or redactor versions and multiple local calendar months.

## Evaluate reviewed labels

Create a reviewed CSV with exactly `audit_id,classification` as its header. Each reviewed audit ID must exist in the selected run and code version. Duplicate IDs, unknown labels, and unmatched reviewed cases fail validation.

```sh
uv run bba evaluate --run-id RUN_ID --labels /srv/bba/reviewed.csv
```

The JSON result reports Cohen's kappa, Gwet's AC1, PABAK, observed agreement, and reviewed and unreviewed counts. `--code-version VERSION` selects an older committed version. Unreviewed rows do not receive fabricated labels.

## Compare monitoring runs

Create a monitoring JSON file containing `baseline_run`, `current_run`, `baseline_code_version`, `current_code_version`, and `sentinel`. The `sentinel` value uses `bba.monitoring.SentinelManifest` with `size`, `seed`, `audit_ids`, and a timezone-aware `built_at`. Every unique manifest audit ID must exist in both runs.

```sh
uv run bba sentinel --weekly --manifest /srv/bba/monitoring-runs.json
uv run bba sentinel --quarterly --manifest /srv/bba/monitoring-runs.json
```

Weekly mode calls `evaluate_sentinel_run` for paired classification agreement. Quarterly mode calls `evaluate_golden_set_drift` for classification and cited-indication changes. These commands compare already committed re-audits. They do not manufacture a baseline or submit new model calls. Each command prints its persisted JSON report path.

## Serve authenticated review

Configure the review database with an unprivileged reviewer role that passes `ReviewActionsStore` migration and privilege checks. Set `BBA_DB_URL`, `BBA_DASHBOARD_AUTH_SECRET`, and `BBA_DASHBOARD_PUBLIC_ORIGIN`. The secret must contain at least 32 UTF-8 bytes. The origin must be the externally visible HTTPS origin. Set both attribution inputs used by reports.

```sh
uv run bba serve-dashboard --port 8000 --run-id RUN_ID --code-version VERSION
```

The CLI binds to `127.0.0.1` behind your HTTPS reverse proxy. Your trusted identity adapter authenticates a reviewer and calls `bba.dashboard.auth.sign_session` with the verified `Reviewer`, shared authentication secret, and an expiry no more than one hour away. It sets the `bba_session` cookie as Secure, HttpOnly, SameSite=Strict, and Path=/. The CLI does not issue identities or accept reviewer identity from a form. Requests without a valid session receive HTTP 401. The dashboard protects adjudication requests with session-bound CSRF tokens and expected decision revisions.

Unredacted PHI lookup remains unavailable from this CLI composition. Deployments that require it must supply an authorized resolver through the app factory. The application does not fabricate identifiers or fall back to an anonymous reviewer.

## Verify without live calls

Run `tests/unit/test_application.py` and `tests/unit/test_application_cassette.py`. The tests construct synthetic canonical CSV exports. They prove local privacy review, authenticated artifact admission, cassette classification, committed row persistence, report format selection, evaluation, and monitoring. The cassette test injects an in-memory batch store. Production model runs always use Postgres checkpoints. These tests do not establish clinical validity, production authentication integration, or Postgres deployment readiness.

## Preserve local evidence

Clinical audit rows, calls, manifests, snapshots, and analysis JSON use owner-only directories (0700) and files (0600). Existing storage with broader permissions or final symlinks is rejected. Review ownership and access offline before migrating older storage. The application does not silently change permissions on existing clinical records.

Exception logs retain error types and code locations. They withhold exception messages and local values because arbitrary clinical text and credentials cannot be recognized safely by a blacklist. Evaluation JSON retains the selected run, code version, reviewed-label digest, and audited-row digest. Monitoring JSON retains the comparison manifest and cadence. Both use immutable content-addressed paths so a later comparison preserves earlier results.

Admission inputs do not change an already completed audit identity. To reevaluate a locally rejected case after obtaining approval, change the approved application policy version and prepare attestations for that new run. `--force` does not replace committed classifications. Cassette content must remain fixed for a configured run; changed evidence or requests require a new policy version and run.

Run these commands from the source checkout after `uv sync`. The Alembic migration directory is an operational dependency in this checkout. Standalone wheels currently include application code and dashboard assets but do not bundle the root migration project. Keep the migrated database and matching migration scripts together when deploying the reviewer service.

The audit command seals the canonical CSV bytes in an owner-only source snapshot before validation or context assembly. Snapshot files and their manifest are read-only (0400) inside a read-only owner directory (0500). Their full input identity, schema, application configuration, and code version are checked before classification. Exporter edits cannot change an in-progress snapshot, and an interrupted run can resume from its recorded snapshot after the exporter removes the originals. Treat sealed snapshots as clinical records with the same retention and access policy as the source export.
