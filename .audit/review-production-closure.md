# Production CLI closure review

Approve the reviewed production composition and CLI corrections. No remaining actionable finding was identified in this scoped re-review. This verdict closes the three P1 findings and the P2 comparison-provenance finding in `.audit/review-production-cli.md` for the current root worktree.

Reviewed on 2026-10-19, Bangkok time, against HEAD `c49936590e9a77e56f1b613286f0e43de3050cd3` and its integrated dirty changes. The scope includes application assembly, source snapshots and bindings, CLI run-store storage, exception handling, evaluation, and monitoring persistence. No application source changed during this review.

## Source identity and recovery

`source_snapshot.py` copies each canonical export file into a private staging directory and hashes the bytes written. It rejects a snapshot whose source identity and configuration do not reproduce the requested run ID. Publication follows file and directory fsyncs. Published source files and the manifest are owner-readable and read-only. The directory is private and read-only. Verification rejects unexpected membership, symlinks, ownership or permission changes, version or configuration mismatch, and changed content hashes.

Ingestion and context assembly consume that same sealed directory. Verification runs again before the pipeline can submit or commit results. The durable source binding records the snapshot path, so recovery does not depend on the exporter retaining its original directory. FileBackedAuditRunStore now creates private directories and writes its completion and row records through durable private writes. Snapshot bindings require private existing storage instead of silently changing legacy directory permissions.

The tests exercised real CLI invocation for exporter changes after ingestion, source changes between run identity and snapshot copy, configuration changes after identity calculation, snapshot tampering after context reads, and retry after the original export path disappeared. The post-ingest exporter edit retained Hb 6.0 from the snapshot. Earlier source or configuration drift and private snapshot tampering stopped before commits. Interrupted recovery succeeded from the retained snapshot. These checks passed in the final focused run.

## Clinical timestamp preservation

The boundary parser now handles canonical ISO progress-note timestamps explicitly. The synthetic case retained its progress note and produced SBP 120 and HR 80 in the earlier direct follow-up probe. The focused application test continues to assert those clinical values. Missing or invalid times remain missing rather than receiving an invented midnight timestamp.

## Credential and narrative protection

The exception renderer now withholds exception messages and every frame-local value. It records error type, frame location, and opaque type and length markers. This removes the previous dependence on PHI-name and text-pattern coverage for arbitrary nested values.

A fresh actual `serve-dashboard` invocation used synthetic database credentials and a signing secret with an invalid HTTP public origin. The resulting ValidationError was passed through the installed `sys.excepthook` into a recording logger. Neither credential value appeared. The error type and `main.py` location remained available. The focused CLI regression also covers nested clinical text and credentials in the exception message and frame locals.

## Comparison provenance and immutable artifacts

Evaluation reads label bytes once, derives both labels and their SHA-256 from that read, records the selected run and code version, and retains a digest of the audited rows. Monitoring records the complete comparison manifest and cadence. Both commands persist private, content-addressed results. An existing artifact is reused only after its bytes match the expected payload.

A separate direct CLI probe evaluated one run at two committed code versions. The outputs retained their selected version and matching label-byte SHA-256. Their audited-row digests differed. Their artifact paths differed, and the earlier bytes stayed unchanged. File modes were 0600 and output directory modes were 0700. Each filename matched the SHA-256 of its content. A second probe changed the sentinel manifest and confirmed distinct retained outputs with the complete comparison selection.

## Verification and limits

The final command `.venv/bin/python -m pytest tests/unit/test_application_snapshot.py tests/unit/test_application.py tests/unit/test_application_cassette.py tests/unit/test_cli.py -q` returned 129 passed and one skipped in 1.29 seconds. The skipped case is the real PostgreSQL cassette variant. This reviewer did not observe that database-backed run. The root owns the separate real PostgreSQL and browser evidence. An earlier focused attempt overlapped the root's creation-mode edit and failed the legacy-marker test. The final coherent run above passed that test.

No live provider, patient data, external identity provider, TLS deployment, browser driver, or external network was used by this reviewer. PyArrow emitted sandbox CPU-introspection warnings during direct synthetic probes. The assertions completed with exit code zero. Coverage was not independently measured in this re-review.

Completed privacy-filter rows remain immutable when approved evidence arrives later. The documented operator workflow changes policy version or application configuration to create a new analysis identity. Privacy trust-policy and artifact contents do not independently redefine an already completed CLI run. Unredacted identifier lookup still requires explicit app-factory wiring. Adult-cohort filtering remains the exporter's responsibility. Production identity issuance and the HTTPS proxy remain deployment responsibilities. Migration files still require the documented source-checkout operational setup. Those explicit limits are outside this scoped approval.

Prove It Works led to checking actual error logs, retained clinical values, source-race outcomes, and content-addressed artifacts. Boundary Discipline led to checking timestamp parsing and source sealing before classification. Make Operations Idempotent led to checking interrupted resume against retained source bytes and ensuring repeated comparison commands preserve prior results. The no-comments skill was read in this review session. Nested review dispatch and source edits remained prohibited by the caller.
