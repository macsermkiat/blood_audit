# Production composition and CLI review

The imported production composition had three blocking findings. The focused suite passed 120 tests, but adversarial synthetic checks reproduced dropped clinical notes, leaked credentials, and incorrect source provenance. A concurrent correction to ISO timestamp parsing passed a follow-up check before this review returned. The other findings require their own correction and revalidation.

Reviewed the root worktree on 2026-10-19, Bangkok time, against HEAD `c49936590e9a77e56f1b613286f0e43de3050cd3` and the imported dirty changes listed in `/tmp/bba-fix-cli-owned.txt`. No source files changed during this review. No patient data, provider calls, or external network calls were used.

## P1. Progress notes with canonical ISO timestamps silently disappear

`src/bba/application/hosxp.py:88` passes a full ISO datetime to `parse_hosxp_time` when no separate time field exists. That parser accepts time-only and English-locale forms, not ISO datetimes. The progress-note reader uses this path at `src/bba/application/hosxp.py:226`.

The existing synthetic export contains one progress row with `PROGDATE=2025-06-01 13:30:00` and documented BP 120/80 and HR 80. Direct execution returned `moment(...) == None`, one source row, zero parsed progress notes, and `VitalSigns(sbp=None, dbp=None, hr=None, rr=None, bt=None)`. Perioperative and administration extraction also receive no such notes. This can change classification and suppress evidence that reviewers expect to see.

Parse the canonical full timestamp explicitly at the input boundary. Preserve its time component without inventing a midnight fallback. Add acceptance coverage that asserts the documented vitals and perioperative evidence survive context construction.

A concurrent root correction added explicit ISO datetime handling during this review. A fresh synthetic context build retained the progress note and returned SBP 120 and HR 80. The original loss is corrected for that reproduced case. The original locations above refer to the imported code before that correction.

## P1. Dashboard startup failures expose signing secrets and database credentials

`src/bba/cli/main.py:287` and `src/bba/cli/main.py:288` retain the database DSN and dashboard signing secret as plain local strings. The installed traceback renderer at `src/bba/cli/phi_scrubber.py:164` serializes frame locals. Its PHI regexes do not redact these credentials.

A real `serve-dashboard` invocation with synthetic credentials and an invalid HTTP public origin raised `ValidationError`. Rendering that exception through the installed `scrub_traceback` function included both complete values below.

```text
dsn = 'postgresql://synthetic:synthetic-password@localhost/db'
secret = 'synthetic-auth-secret-value-32-bytes-minimum'
```

Anyone who can read the resulting logs can obtain a production signing key and forge reviewer sessions. The same renderer also serializes arbitrary nested raw-export objects whose local names do not match its PHI-name prefix. A safe fix must protect exception messages and nested data as well as the two variable names. Omit frame-value dumps or allowlist safe diagnostics. Add a failure-path test through the real CLI exception handler.

## P1. Context construction reopens mutable source files after validation and identity calculation

The CLI computes run identity at `src/bba/cli/main.py:487`. `src/bba/application/audit.py:76` ingests the export, then `src/bba/application/audit.py:86` independently reopens the original source directory for context construction. The normalized, validated ingest output is not the input consumed by classification.

A deterministic synthetic check wrapped the real ingest function and changed the first Lab result from Hb 6.0 to 12.0 immediately after ingestion returned. The CLI succeeded. Both committed rows retained the pre-edit run ID, while the classified Hb values were `[12.0, 11.0]`. Recomputing identity from those source files produced a different run ID. The records therefore claim source bytes that did not produce them.

Classify an immutable source snapshot whose bytes determine the run identity and whose normalized contents pass ingest validation. A per-run file lock does not prevent an external exporter from rewriting source files. Add a regression that changes a source file between identity, ingest, and assembly and proves that the run refuses drift before submission or commit.

## P2. Evaluation and monitoring outputs lose comparison identity

`src/bba/application/operations.py:62` returns evaluation counts and agreement without the selected run, code version, reviewed-label digest, or membership identity. `src/bba/cli/main.py:238` writes that result to a path based only on run ID, so an evaluation with another `--code-version` overwrites the earlier artifact. The monitoring command similarly serializes metric results without its baseline/current manifest and writes a path based only on current run and cadence. Comparisons against another baseline or current code version overwrite that artifact.

This is a source-level finding. Include the complete selection and input digests in the persisted result and artifact identity. Preserve prior comparison records so a maintainer can reproduce a reported metric.

## Material workflow limits

Run identity includes CSV contents and explicit `ApplicationConfig` JSON. It omits privacy-artifact contents, privacy trust-policy contents, and cassette contents. A synthetic check confirmed that changing the privacy input environment paths leaves run identity unchanged. If a configured model run first commits `privacy_admission_failed` rows, supplying approved proof later does not revisit them. The CLI completion shortcut and immutable pipeline rows preserve the first result, even with `--force`. The documented workaround is to change the policy version or application configuration. This behavior needs an explicit operator workflow and provenance if it remains intentional.

Unredacted identifier lookup is unavailable through the CLI composition. The adult cohort filter remains an exporter responsibility. Production authentication still requires an external identity adapter and HTTPS reverse proxy. Model execution requires a migrated PostgreSQL checkpoint store and trusted signed privacy artifacts. These limits are documented in `docs/production-cli.md`; they were not verified against a deployment.

The reviewed code validates explicit label membership and rejects unmatched labels, duplicate IDs, incomplete sentinel membership, and baseline/current self-comparisons. HTML report cells and provenance are escaped. The CLI binds the dashboard to loopback and supplies signed-session configuration rather than a default reviewer. No additional confirmed finding arose from those paths.

## Verification

`.venv/bin/python -m pytest tests/unit/test_application.py tests/unit/test_application_cassette.py tests/unit/test_cli.py -q` returned 120 passed in 1.36 seconds. The source-provenance, progress-note, and credential-leak reproductions each ran against the actual imported source using temporary synthetic exports. These checks used TestClient or CliRunner and no live provider. Coverage was not independently measured. Real PostgreSQL and browser checks remain with the root agent.

Prove It Works changed this review from accepting the test summary to inspecting actual parsed notes, persisted Hb values, run identities, and rendered exception output. Boundary Discipline led to requesting timestamp and source validation at the export boundary. Make Operations Idempotent led to checking what happens when approved evidence arrives after a privacy-failed run has committed. The no-comments skill was read. Its nested reviewer dispatch was not run because this independent review explicitly prohibits nested agents and source edits.
