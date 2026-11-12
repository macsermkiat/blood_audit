# Module contracts

This reference describes current boundaries in `src/bba/`. Use the [operator guide](production-cli.md) for commands and deployment requirements. The [historical glossary](history/CONTEXT-2026-10-19.md) preserves earlier specifications, ticket discussions, and superseded behavior. It is not the current contract.

## Application

`application.audit.audit_bundle` composes the supported raw-export audit. `hosxp` normalizes canonical tables and assembles orders. `contexts` builds inputs for the existing clinical modules. `config.ApplicationConfig` validates operator configuration and rejects unknown fields.

`source_snapshot` seals source bytes before validation and classification. Its identity binds the input bundle, schema, explicit application configuration, and code version. An interrupted run can use its recorded snapshot after the exporter removes the originals. A completion marker requires committed expected results; an ingest marker is insufficient.

Local mode makes no provider calls. Eligible unresolved red-cell work becomes committed privacy review. The platelet model route has a separate default-off flag. Expected platelet results left uncommitted prevent CLI completion. [feature_flags.py](../src/bba/feature_flags.py) defines current defaults.

## Ingest

The canonical registry contains 13 tables: `BDVST`, `BDVSTDT`, `BDVSTST`, `BDVSTTRANS`, `BDTYPE`, `Diagnosis`, `Lab`, `Med`, `IPDADMPROGRESS`, `IPDNRFOCUSDT`, `IPTSUMOPRT`, `INCPT_OPRTACT`, and `ICD9CM`. File names preserve this case.

`ingest.schemas.all_tables()` supplies declaration order. `get_schema()` supplies required columns. `normalize_header()` applies supported naming adjustments, first-occurrence deduplication, and projection before strict validation. Extra source columns can be recorded as dropped; missing required columns remain errors. The [normalization implementation](../src/bba/ingest/normalize.py) defines export-specific transformations, including note-year filtering.

Ingest identity binds source hashes, schema fingerprint, and code version. Explicit application configuration also contributes to the supported audit run identity. Source timestamps are interpreted in Bangkok time. Persisted timestamps use UTC where their models require it; reviewer displays use Bangkok time.

## Audit orders and components

`audit_orders` filters orders, applies explicit exclusions, resolves anchors, and builds stable audit IDs. An order is not a unit of administered blood. A blood-bank issue timestamp does not prove administration.

`component_map` identifies red cells and platelets and excludes unsupported component families with reasons. Component identity must survive downstream joins and aggregation.

## Hb lookup

`hb_lookup` selects valid pre-anchor observations, resolves supported fallback anchors, and records freshness and source provenance. Delta-Hb windows and isolated-low-Hb flags are separate outputs. An absent Hb differs from an expired or rejected observation.

## Vitals and perioperative evidence

`vitals_extractor` extracts windowed observations with provenance and sanity checks. Hemodynamic and perioperative summaries retain supporting evidence such as blood loss and intraoperative transfusion. They do not independently authorize a transfusion or replace classifier precedence. The evidence anchor can differ from the original blood-bank timestamp.

## Cohort detection

`cohort_detector` returns a label and threshold together. Current thresholds are 7.0 g/dL for `default` and `esrd_epo`, 7.5 for `cardiac_surgery`, and 8.0 for `ortho_surgery` and `cardiopulmonary_comorbidity`. The latter label remains compatible with older rows, but its trigger set is heart-disease only. The detector no longer emits `ortho_cardiac`.

MTP, active hematologic malignancy, and unknown context have no numeric threshold. Missing procedure data differs from a known empty procedure history. [rules.py](../src/bba/cohort_detector/rules.py) and [detector.py](../src/bba/cohort_detector/detector.py) define matching and precedence.

## Deterministic classification

`deterministic_classifier.classify` is a pure function of validated inputs. Returns-ledger and declared-use routes precede Hb judgments. After missing-Hb handling, the universal rule accepts Hb at or below 7.0 before MTP and unknown-context routing. Remaining cohort thresholds are inclusive. Hb at or above 10.0 uses the high-Hb route after preceding rules.

Safety flags, hard perioperative evidence, delta-Hb, and legacy flag-off procedure routes have ordered conditions. A procedure planned within 72 hours is not an unconditional appropriate result. The [Hb guide](src/content/docs/en/developers/three-tier-hb.mdx) describes the sequence. This is an implementation contract, not independent clinical validation.

## Returns, declared use, and reservation

`returns_ledger` aggregates unit dispositions. A supported complete all-returned result can yield `RETURNED_NOT_TRANSFUSED`. `declared_use` interprets order-time `USETYPE`; a qualifying declaration can yield `PERIOP_TRANSFUSION_EXEMPT`. Both routing exemptions are excluded from appropriateness scoring.

`preop_reservation` compares reserved units with the selected MSBOS recommendation and records picker provenance. `PREOP_OVER_RESERVATION` is a reservation finding, not an Hb verdict. Ambiguous operation selection, bridge disagreements, and missing evidence require review. The default-off administration-confirmation route can yield `PREOP_RESERVATION_UNCONFIRMED`. Flags and their prerequisites remain explicit.

## Platelet audit

`platelet_lookup` selects a count and records observation provenance. `platelet_classifier` applies deterministic gates. `platelet_guardrail` checks supported model clearances against cited evidence and configured indication rules. A count threshold alone does not represent the entire platelet policy.

The model indication boundaries at 10, 20, 50, and 80 thousand per microliter are inclusive. The 100-thousand model indication exception remains strict. MSBOS reservation rules are separate. The [platelet guide](src/content/docs/en/developers/platelet-auditor.mdx) describes routing and feature-flag limits.

## Evidence bundles

`evidence_bundle_builder` produces stable evidence IDs and canonical content with per-source windows and truncation rules. Pinned evidence survives ordinary dropping. Bundle hashes identify exact content; they do not establish de-identification or clinical sufficiency.

## De-identification and admission

`deid_redactor` runs the configured local backend and privacy checks. Deployment must supply an approved model artifact and cohort anonymity gate. Synthetic test backends are not production redactors.

`deid_redactor.admission.RedactionAuthority` prepares exact evidence and an Ed25519 attestation. The application receives a public verification key, approved redactor metadata, and signed artifacts. It does not receive the issuer's private key. Admission binds the prompt, audit ID, run ID, approved redactor version, and privacy checks. Missing or rejected admission stays local.

Raw exports, encrypted exports, replaced identifiers, and model outputs are not safe to commit because of a development filename. The [privacy guide](src/content/docs/en/developers/deid.mdx) separates synthetic fixtures from clinical material.

## Prompt building

`prompt_builder` builds task-specific evidence envelopes, checks injection patterns, and computes prompt identity. Red-cell and platelet tasks remain distinct. Clinical evidence is data, including when it contains instructions. Changed prompt content requires matching request identity and admission.

## Model requests and recovery

`llm_client` supplies cassette and live transports, parsing, ordered attempts, and configured escalation. [models.py](../src/bba/llm_client/models.py) pins model IDs. A cassette proves replay behavior, not live provider availability.

`request_identity` binds the canonical request, including model, evidence, and membership. Immutable manifests and call checkpoints preserve recovery identity. An uncertain submission cannot automatically execute again. Provider-level retries are disabled. Recovery preserves original attempt indices and provenance.

## Quote grounding

`quote_grounder` validates claims against actual cited evidence. Ordered checks stop at the first rejection. Normal and recovered results use the verifier. Rejected claims become review work. A grounding pass does not establish clinical truth. The [grounding reference](src/content/docs/en/developers/quote-grounding.mdx) describes checks and limits.

## Audit pipeline

`audit_pipeline.run_pipeline` classifies deterministic work, submits eligible admitted work, verifies results, and persists rows and calls. Context manifests bind run, batch, component policy, code version, and evidence. Production model runs use PostgreSQL batch checkpoints.

Repeated execution preserves committed results. Expected uncommitted audit IDs are orphans and prevent supported CLI completion. Platelet orders suppressed within an MTP window intentionally emit no AuditRow.

## Audit storage

`audit_store` retains `AuditRow` and `LlmCall` data with versioned identity and provenance. Selected code versions control snapshots and reads. Stores reject incompatible rewrites, unsafe permissions, and final symlinks.

Clinical rows, calls, manifests, and analysis artifacts use owner-only storage. Sealed source files are read-only. Snapshots and cold storage remain clinical records subject to retention and access policy.

## Review actions and dashboard

`review_actions` stores append-only actions and PHI-access events in migrated PostgreSQL. Evidence-bound decisions retain the reviewed evidence identity and expected revision. Replaying a submission is idempotent; stale competing decisions conflict.

`dashboard` serves committed unresolved red-cell and platelet cases with reasoning, citations, Bangkok times, and history. Aggregate scorecards retain red-cell scope. Classification alone does not establish whether review is resolved.

The supported CLI requires a trusted identity adapter, signed expiring session, authentication secret, public HTTPS origin, and reverse proxy. Reviewer identity does not come from form fields. Session-bound CSRF and origin checks guard writes. Raw-identifier lookup requires a separately authorized resolver through the app factory.

## Reporting and attribution

`report_generator` builds monthly inputs and renders committee and physician artifacts. Mixed deterministic and model results retain row-level provenance. Conflicting applied policy or redactor versions are rejected. Routing exemptions and component boundaries control denominators.

`attribution` joins the ordering physician and department from selected inputs. Human-reviewed labels and pipeline verdicts are separate sources. Unreviewed orders do not receive fabricated labels. The [report schema](src/content/docs/en/reference/report-schema.mdx) describes rendered fields.

## Evaluation and calibration

`eval_harness` supplies sampling, agreement, weighting, and diagnostic metrics. `bba evaluate` requires reviewed audit IDs to exist in the selected run and code version. Duplicate IDs, unknown labels, and unmatched cases fail validation. Artifacts retain input digests and selection identity.

`confidence_calibrator` fits and evaluates calibration from supplied observations. Metrics describe the evaluated sample and do not replace clinical review. Historical sample sizes and pass totals belong in dated evidence records.

## Monitoring and verification

`monitoring` compares committed baseline and current runs through a manifest. Weekly and quarterly CLI modes do not launch model re-audits or invent a baseline. Artifacts retain comparison identity and cadence.

`verification` supports explicit before-and-after checks for selected changes. The [testing guide](src/content/docs/en/developers/testing.mdx) describes infrastructure and coverage. CI requires PostgreSQL test infrastructure and at least 80% source coverage. Protected real-data tests skip when their external fixtures are absent.

## CLI and errors

`cli.main` exposes ingest, audit, report, evaluate, sentinel, and authenticated dashboard commands. See each command's `--help` and the [operator contract](production-cli.md). Historical pilot scripts use separate environment overrides and dataset assumptions.

The CLI installs its exception renderer explicitly. Logs retain error types and code locations while withholding messages and local values. Verify configuration without printing environment variable values, which can expose credentials or arbitrary clinical text.
