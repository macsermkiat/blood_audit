# CONTEXT.md — Ingest-Domain Glossary

This file names the load-bearing concepts used by `src/bba/ingest/`. Future
architecture reviews and AI agents joining the project should speak these terms
verbatim; update entries here (instead of forking new vocabulary in code) when a
concept changes shape.

Scope: Phase 1 of the KCMH RBC transfusion audit pipeline. PRD: [issue #1].
Ingest is the foundation for tickets [#4–#7] (audit_orders, hb_lookup,
vitals_extractor, cohort_detector). The PRD's "Implementation Decisions §1"
defines the contract.

## Concepts

### HOSxP table

One of the 10 canonical CSVs exported monthly from KCMH's HOSxP EMR:
`BDVST`, `BDVSTDT`, `BDTYPE`, `BDVSTST`, `Diagnosis`, `Lab`, `MED`,
`IPDADMPROGRESS`, `IPDNRFOCUSDT`, `UnUSE_Patient_Background`. The literal set
is the `bba.ingest.models.CSVTable` type; `bba.ingest.schemas.all_tables()`
returns the canonical tuple in declaration order. The set does not vary
within Phase 1 — adding a table is a Phase-2 change that requires a schema
version bump.

### Schema (v1)

A pandera `DataFrameSchema` declaring the columns and dtypes one HOSxP table
must contain. Lives in `bba.ingest.schemas._REGISTRY_V1`. The v1 suffix is a
version anchor: changing any schema means bumping to v2 (which produces a new
`schema_fingerprint` → new `run_id` → fresh ingest).

### Schema fingerprint

A stable sha256 hex digest over all v1 schemas, computed by
`bba.ingest.schemas.schema_fingerprint()`. Visits schemas in canonical table
order, and within each schema visits columns in their declared order. A column
rename or reorder produces a new fingerprint. The fingerprint participates in
`run_id`, so a schema bump cannot silently mutate prior outputs.

### Drift

A mismatch between an input CSV's header and its v1 schema. Two forms, both
fatal:

* **Unknown column** — present in the CSV, not declared in the schema.
* **Missing required column** — declared in the schema, absent from the CSV.

`bba.ingest.schemas.validate_header(table, header)` is the one place the rule
is enforced; it raises `SchemaDriftError` and the pipeline never writes a
completion marker on a raise.

### Incomplete input

The input directory is missing or is missing one or more of the 10 canonical
HOSxP tables. Distinct from drift (which is per-table). Raised as
`IncompleteInputError` before any side-effect — a typo or partial export never
masquerades as a successful ingest, and never no-ops as idempotent on retry.

### Parse warning

The strict HOSxP time parser refuses to silently shift an unrecognized format
into a wrong-but-plausible value. Allow-list: `HHMMSS` (6 zero-padded digits)
and `HH:MM`. Everything else — decimal hour, Excel serial fraction,
Buddhist-year prefix, sentinels `0`/`9999`/`null`, empty, garbage, `None` —
yields a `ParseResult(value=None, parse_warning=<reason>, raw=<input>)`. The
warning is a short string naming the rejection reason; it lands in a
`parse_warning` column at row-write time (#4–#7).

### Sentinel time

A historical anti-pattern, deliberately avoided. The strict parser used to
return `datetime(1900, 1, 1, h, m, s)` for time-only parses — a sentinel date
that callers had to remember to ignore. Replaced by `ParsedTimeOfDay`, which
has no date component. Callers MUST combine it with the row's date column via
`RowTimestamp.from_parts(date, time, tz)` to get a persistable moment.

### Run identity

The stable fingerprint of a single ingest invocation:
`sha256(input_csv_hash + schema_fingerprint + code_version)`. Lives behind
`bba.ingest.run_identity.RunIdentity`. Same byte content + same schemas + same
code yields the same `run_id`, which is the entire basis of idempotent
re-runs.

### Completion marker

`_run_<run_id>.complete` — an empty file in the output directory written by
`RunIdentity.mark_complete()` after all per-table writes succeed. Written
atomically via write-then-rename so a crash mid-write cannot leave a
half-formed marker. `RunIdentity.is_complete(output_dir)` checks for it.

### Idempotent re-run

Calling `ingest(config)` a second time with the same config + same input bytes
+ same code version no-ops: the run_id matches, the completion marker is
already on disk, the function returns `IngestResult(skipped_idempotent=True)`
without re-reading or re-writing anything. The opposite of an idempotent
re-run is a forced re-run (Phase 2 — `--force` flag, not yet implemented).

### tz-aware UTC

Every persisted timestamp carries `tzinfo == UTC`. `Asia/Bangkok` is the
*source* zone (where HOSxP records originate) and the *display* zone (the
reviewer dashboard). The only path from a (date, time, tz) triple to a stored
timestamp is `RowTimestamp.from_parts`; there is no naive datetime in the
ingest module's persisted output, and a structural test
(`TestNaiveDatetimeBan`) plus a planned ruff rule keep `datetime.now()` /
`datetime.utcnow()` out of the source files.

## Vocabulary not in this file

Domain terms used in the broader project but defined elsewhere (PRD issue
body, or future Phase-1 modules):

* `audit_id`, `INAPPROPRIATE`, `NEEDS_REVIEW`, `INSUFFICIENT_EVIDENCE`,
  `cohort_threshold`, `delta-Hb bypass`, `MTP`, `T1.MTP`, `quote_grounder`,
  `hallucination_suspect`, etc. — see the PRD (issue #1) for definitions.

Add a concept here when a module exposes it as part of its interface. Update
when a concept's shape changes; remove when it leaves the codebase.
