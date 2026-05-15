# CONTEXT.md — Phase 1 Module Glossary

This file names the load-bearing concepts exposed by the `src/bba/` modules.
Future architecture reviews and AI agents joining the project should speak
these terms verbatim; update entries here (instead of forking new vocabulary in
code) when a concept changes shape.

Scope: Phase 1 of the KCMH RBC transfusion audit pipeline. PRD: [issue #1].
Sections are added per ticket as each module merges to `main`. Currently
covered: `#3 ingest`, `#5 hb_lookup`, `#6 vitals_extractor`, `#18 quote_grounder`,
`#19 audit_store`.

## Ingest concepts (#3)

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

## Hb-lookup concepts (#5)

### Hb observation

A validated single Hb result from the `Lab` table: numeric in `[2, 25] g/dL`,
test code in `{LABEXM 290095 HEMATOLOGY, LABEXM 500001 POCT}`, with a tz-aware
collection timestamp. `bba.hb_lookup.models.HbObservation`.

### Hb source

Enum naming the lab assay: `HEMATOLOGY` (preferred when both available) vs
`POCT` (fallback). Source preference is a deterministic invariant, not a
clinical judgment. `bba.hb_lookup.models.HbSource`.

### Freshness tier

The age of the chosen Hb at order anchor: `fresh` (<24 h), `stale_24_72h`,
`stale_3_7d`, `missing` (>7 d or none). `bba.hb_lookup.models.HbFreshness`.
The tier annotates every audit row so the reviewer dashboard can reason about
"the doctor decided with day-old data."

### Delta-Hb window

A `(window_hours, threshold_g_per_dL)` pair: `(6, 1.5)`, `(12, 2.0)`,
`(24, 2.5)`. Each is checked independently; any window tripping fires the
**delta-Hb bypass** flag. `bba.hb_lookup.models.DeltaHbWindow`.

### Delta-Hb bypass

A Boolean emitted in `HbLookupResult` when at least one delta-Hb window's drop
threshold is met. Downstream `bba.deterministic_classifier` (#8) treats this
as a hard short-circuit to `APPROPRIATE` (acute-blood-loss catch — Hb may not
yet reflect a fast bleed).

### Most-recent-before tie-break

When two Hb observations share `(lvstdate, lvsttime)`, the row with the
highest `ITEMNO` wins (later database entry, presumed corrected). Implemented
in `bba.hb_lookup.lookup.lookup_hb`; the function never returns the
lowest-in-window — hindsight bias is a deliberate non-feature.

### Hb lookup result

The unit of output: chosen `HbObservation`, its `freshness`, `source`, and the
list of `DeltaHbWindow` evaluations. `bba.hb_lookup.models.HbLookupResult`.

## Vitals-extractor concepts (#6)

### Vitals note

A single source row carrying `(table, tz-aware timestamp, free text)`.
`bba.vitals_extractor.models.VitalsNote`. Built from `IPDADMPROGRESS.OBJECTIVE`
or `IPDNRFOCUSDT` `FOCUS/ACTION/RESPONSE` columns by the ingest layer; never
constructed from raw CSV by the extractor.

### Source provenance

Enum naming where the chosen vitals came from: `IPDADMPROGRESS` (preferred,
SOAP-clean), `IPDNRFOCUSDT` (fresher, noisier), `LLM_extracted` (regex
failed), or `none_in_window`. `bba.vitals_extractor.models.SourceProvenance`.
Surfaced on every result so reviewers can weigh the trustworthiness of the
reading.

### Vital signs

The structured tuple: `(sbp, dbp, hr, rr, bt)` in mmHg / bpm / breaths-min /
°C. `bba.vitals_extractor.models.VitalSigns`. Any field may be `None` when
absent from the source note.

### Sanity bounds

Fixed numeric ranges enforced before a value enters a `VitalSigns`:
SBP 60–220, DBP 30–150, HR 30–200, RR 5–50, BT 30–43. Values outside the
range are dropped and the `VitalsResult` is flagged `vitals_data_error`.
`bba.vitals_extractor.bounds`.

### ±6 h window

Selection window from order anchor. Most-recent-before-anchor wins; if none
exists in the lookback half, falls forward to the most-recent-within-+6 h and
flags `vitals_post_order`. The `VitalsResult` carries the lag in minutes.

### LLM fallback

Invoked only when regex finds neither SBP nor HR in any candidate note within
the ±6 h window. Boundary is sharp on purpose — the LLM never refines a
regex-extracted value. `bba.vitals_extractor.models.LLMFallback`.

### Vitals flag

Quality annotations on a `VitalsResult`: `vitals_post_order`,
`vitals_data_error`, and others. `bba.vitals_extractor.models.VitalsFlag`.
Downstream consumers (#8 deterministic_classifier; #16
evidence_bundle_builder) read these to gate rule branches and to prioritize
human review.

## Quote-grounder concepts (#18)

### Citation

The LLM-output unit the grounder verifies: a verbatim `quote` attributed
to a source by `cited_id`, optionally carrying a structured `lab_tuple`.
Lives in `bba.quote_grounder.models.Citation` (Pydantic frozen). The
verifier consumes one citation at a time; the audit pipeline (#24) passes
the LLM's `indications_json` list through `verify_citations` to get a
parallel tuple of verdicts.

### Evidence source

One redacted source in the bundle: `(source_id, text)`. `source_id` is the
opaque key the LLM is expected to echo back as a citation's `cited_id`;
`text` is post-redaction (the `bba.deid_redactor`, #17, runs upstream).
Bundle-wide `source_id` uniqueness is the upstream bundle builder's
contract (#16); the grounder treats a duplicate `source_id` as the same
failure mode as missing-cited_id.

### Six grounding layers

Applied in canonical order; the verifier short-circuits on the first
failure and the failure's `VerdictReason` names the layer:

1. **NFC normalization** — `nfc_normalize(text)` applies Unicode NFC on
   both sides before any comparison. Defeats Thai NFC-vs-NFD adversarial
   fixtures.
2. **Word-boundary-anchored contiguous match** — the quote must occur as
   a substring in the cited source AND start/end at a non-alphanumeric
   character (or string boundary). Defeats concatenated-quote attacks and
   1-character-shift attacks (deleting a boundary char yields a still-
   contained substring pinned mid-word).
3. **Strict cited_id match** — `find_cited_source(cited_id, sources)`
   returns the unique source whose `source_id` equals `cited_id`, or
   `None` if zero or ≥2 match. Defeats cross-source attribution.
4. **Within-document uniqueness** — the quote occurs at exactly one
   word-boundary-aligned position in the cited source. Defeats short-
   common-phrase attacks ("no bleeding" appearing in unrelated context).
5. **Minimum length** — NFC-length ≥ `MIN_QUOTE_LENGTH` (25). PRD §9.
6. **Numeric-tuple grounding** — for lab citations, the verifier extracts
   `(analyte, value, unit)` triples from BOTH the quote and the source
   (`extract_lab_triples`); every quote triple must be present in the
   source, and any supplied `citation.lab_tuple` must match a triple
   parsed from the QUOTE (after analyte aliasing — `Hgb`/`Hb`/`hemoglobin`
   collapse to canonical `hb` — and unit canonicalization).
7. **Medical-NLI entailment gate** (optional) — `NLIEntailmentGate`
   protocol callable supplied by the caller. PRD §9 explicitly allows
   omitting Layer 7 when a medical-domain NLI model is unavailable.

### Verdict + verdict reason

`Verdict(passed, reason, citation)` is the per-citation outcome (Pydantic
frozen). `reason` is `VerdictReason.PASS` on accept; otherwise one of seven
mutually-exclusive failure tags matching the canonical short-circuit order.
Persisted as part of the audit row's `indications_json` so reviewers see
"why" alongside "rejected".

### Verifier-as-classifier

The grounder's accept/reject decision treated as a binary classification:
predicted-positive == `Verdict.passed`, gold-positive == "the citation is
genuinely grounded" (human label). `confusion_matrix(verdicts, labels)`
produces the 2x2 contingency the eval harness (#20) consumes for the 200-row
hand-labeled set. The grounder ships a 24-row mini-set in tests to
regression-guard the function's output shape.

### Pure function

The module performs no I/O, mutates no global state, imports no Anthropic
SDK / HuggingFace transformers. Determinism is asserted by
`test_determinism_same_input_same_verdict`; the no-I/O contract by
`test_no_stdout_or_stderr_writes`. The optional NLI gate is the only
boundary where a model can be plugged in — and the gate is a Protocol
callable supplied by the caller, never imported by the grounder.

## Audit-store concepts (#19)

### Audit row

One immutable record per audited RBC order. Persisted append-only to
`audit_results.parquet`. `bba.audit_store.models.AuditRow`. Fields enumerated
in PRD §"Output schema"; once committed, mutations are forbidden — corrections
are new rows tied to the same `audit_id`.

### LLM call

One Anthropic-API invocation against an `audit_id`. Multiple calls per
`audit_id` are normal (retry, Sonnet→Opus escalation, sentinel re-run).
Always written to `llm_calls.parquet` **before** the matching `AuditRow`
lands. `bba.audit_store.models.LlmCall`.

### Transactional ordering

The invariant: every committed `AuditRow` has at least one matching `LlmCall`
already on disk; the `AuditRow` write is the **commit marker** for the whole
operation. Violations raise `TransactionalOrderingError`; reconciliation finds
orphaned `LlmCall`s with no matching `AuditRow`.

### Reconciliation

Output of `AuditStore.reconcile(run_id)` — a `ReconciliationReport` listing
orphan `LlmCall`s plus `orphan_audit_ids`. Run on startup so a process killed
mid-batch doesn't silently lose work.
`bba.audit_store.models.ReconciliationReport`.

### Snapshot view

A read-only, daily-rotated DuckDB view over `audit_results.parquet` consumed
by the reviewer dashboard (#26). Prevents an in-flight batch write from
producing inconsistent reads mid-query. `bba.audit_store.snapshot.SnapshotView`.

### Cold storage

Migration of Opus extended-thinking blocks from hot Parquet to cheaper object
storage after 90 days. `bba.audit_store.cold_storage.migrate_cold_storage`
returns a `ColdStorageReport`. The hot path keeps the final output and usage;
auditability is preserved end-to-end via the cold blob's content hash.

### Run-level idempotency

Re-writing the same `(audit_id, run_id)` is a no-op (deterministic
winning-attempt rule = last verifier-passed wins). `WriteResult` reports
`wrote`, `noop`, or `escalated` so the orchestration layer (#24) can drive
batch progress correctly.

### Classification

Enum of canonical audit labels: `APPROPRIATE`, `INAPPROPRIATE`,
`NEEDS_REVIEW`, `INSUFFICIENT_EVIDENCE`, `POTENTIALLY_INAPPROPRIATE`.
`bba.audit_store.models.Classification`. Single source of truth across the
pipeline; downstream modules import from here, never re-define.

## Vocabulary not in this file

Domain terms used in the broader project but defined elsewhere (PRD issue
body, or future Phase-1 modules):

* `audit_id`, `cohort_threshold`, `MTP`, `T1.MTP`,
  `hallucination_suspect`, etc. — see the PRD (issue #1) for definitions.

Add a concept here when a module exposes it as part of its interface. Update
when a concept's shape changes; remove when it leaves the codebase.
