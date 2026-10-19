# blood_audit

Retrospective quality-improvement audits of inpatient adult red-cell and platelet orders at King Chulalongkorn Memorial Hospital (KCMH). The application reads completed HOSxP exports, applies the repository's PR 17.2 and AABB 2023 policy implementation, and stores classifications with evidence and provenance.

This software supports review after care. It does not authorize transfusion orders or provide bedside alerts. Software tests do not establish clinical validity or de-identification quality.

## Start here

- [Operators](docs/src/content/docs/en/operators/overview.mdx) can install the application, prepare an export, and run an audit.
- [Developers](docs/src/content/docs/en/developers/architecture.mdx) can trace the application and its module contracts.
- [Clinical reviewers](docs/src/content/docs/en/clinical/what-this-is.mdx) can check scope, implemented rules, and report interpretation.
- The [documentation site](https://macsermkiat.github.io/blood_audit/) has English and Thai guides.

## Run a local audit

Use Python 3.13 or later and `uv`. Run from the source checkout so the matching database migration project is available when needed.

```sh
git clone https://github.com/macsermkiat/blood_audit.git
cd blood_audit
uv sync
export BBA_DATA_DIR=/srv/bba/data
uv run bba audit --input /srv/bba/export/BDVST.csv
```

Stage the complete canonical bundle beside `BDVST.csv`. The exporter must apply the approved adult-cohort filter because the bundle has no age column. Use an access-controlled directory. Raw and encrypted clinical exports, audit outputs, and partly de-identified data must stay outside version control.

The default local mode needs no provider credentials or PostgreSQL connection. It persists deterministic results and routes eligible unresolved red-cell orders to local privacy review. It makes no model calls. A successful completion counts committed AuditRows, not ingested tables.

The platelet model route is disabled by default through `PLATELET_LLM_ENABLED`. Unresolved platelet orders need that route enabled; the CLI fails completion if expected results remain uncommitted. Deterministic platelet outcomes and MTP suppression follow their own rules. See the [platelet guide](docs/src/content/docs/en/developers/platelet-auditor.mdx) before auditing a mixed bundle.

The [operator contract](docs/production-cli.md) covers signed privacy admission, cassette and live transports, PostgreSQL checkpoints, reports, evaluation, monitoring, and authenticated review. The repository does not supply a production-approved redactor or an identity provider. A provider key alone is insufficient to authorize clinical evidence submission.

## Keep audit identity stable

Input bytes, schema, code version, and explicit application configuration determine the run identity. The audit seals its input bundle before classification. Repeat execution reuses immutable committed results. `--force` bypasses the CLI completion shortcut and records an override; it does not replace committed classifications or resubmit an uncertain provider request.

For changed analysis, use a new approved policy version and prepare matching admission evidence for the new run. Keep `BBA_DATA_DIR/identity.key` with its protected data directory so patient and admission identifiers stay stable.

## Prepare the input bundle

File names are case-sensitive. The canonical schema registry currently contains:

```text
BDVST.csv
BDVSTDT.csv
BDVSTST.csv
BDVSTTRANS.csv
BDTYPE.csv
Diagnosis.csv
Lab.csv
Med.csv
IPDADMPROGRESS.csv
IPDNRFOCUSDT.csv
IPTSUMOPRT.csv
INCPT_OPRTACT.csv
ICD9CM.csv
```

The [ingest schemas](src/bba/ingest/schemas.py) define required columns. Normalization projects recognized columns and handles supported export variants before strict schema validation. Missing required columns still fail. See [export joins](docs/hosxp-export-reference.md) and the [ingest guide](docs/src/content/docs/en/operators/ingest-csv.mdx) for source-system details.

## Review and report

`bba report` renders HTML, JSON, or PDF for a committed run. Attribution requires the matching `BBA_BDVST_CSV` and `BBA_DCT_CSV` inputs. Reports retain per-row policy, redactor, prompt, and model provenance. Red-cell scorecards and appropriateness denominators do not silently include platelet results or routing exemptions.

`bba serve-dashboard` serves authenticated review behind an HTTPS reverse proxy. The queue includes committed unresolved cases from both components. Reviewers can agree, override, or escalate, with append-only history and stale-write checks. Deployment must supply trusted reviewer authentication and migrated database roles.

Historical pilot scripts and the human-review ranking workflow are documented separately in [scripts/pilot/README.md](scripts/pilot/README.md). They are dataset-specific tools with their own inputs and flags. Use the supported CLI guide for operational setup.

## Check a change

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy src
BBA_REQUIRE_POSTGRES_TESTS=1 uv run --extra live pytest -q --tb=short --cov=src/bba --cov-fail-under=80
cd docs
ASTRO_TELEMETRY_DISABLED=1 npm run build
```

PostgreSQL tests require Docker and test dependencies. CI requires those tests and at least 80% source coverage. Real-data reconciliation tests skip when their protected workbooks and exports are absent. Do not copy those materials into the repository to satisfy a test.

## References

- [Current module contracts](docs/CONTEXT.md) describe boundaries and persistent identities.
- [Feature flags](src/bba/feature_flags.py) define current defaults. Pilot environment overrides do not configure every supported CLI path.
- [Clinical policy implementation](docs/src/content/docs/en/clinical/policy-summary.mdx) describes precedence and boundary cases.
- [Requirements](https://github.com/macsermkiat/blood_audit/issues/1) retain the original scope.
- [Historical documentation](docs/history/README.md) preserves earlier build notes and plans. Historical instructions are not operational guidance.
