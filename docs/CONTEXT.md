# CONTEXT.md — Module Glossary

This file names the load-bearing concepts used across the `src/bba/` modules.
Future architecture reviews and AI agents joining the project should speak
these terms verbatim; update entries here (instead of forking new vocabulary
in code) when a concept changes shape.

Scope: Phase 1 of the KCMH RBC transfusion audit pipeline. PRD: [issue #1].
Modules in this document:

* `bba.ingest` (issue #3) — foundation for #4–#7
* `bba.evidence_bundle_builder` (issue #16) — note ranking + canonical bundle

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

## Concepts — `bba.evidence_bundle_builder` (issue #16)

### Per-source window

Each source family has its own time window relative to the order anchor —
all enforced by `bba.evidence_bundle_builder.builder`:

* `Diagnosis` — **AN-scoped** (no time window; full ICD-10 list for the encounter)
* `IPDADMPROGRESS` — `±24h`, cap 8 closest entries (closest-by-abs-offset)
* `IPDNRFOCUSDT` — `±24h`, cap 10 entries via 5-before / 5-after closest-first
* `MED` — `[-72h, +24h]` (asymmetric: drug history + post-order administration)
* `Lab` (Hb history) — `[-7d, anchor]` strict at lower bound (matches
  `bba.hb_lookup`'s `< _LOOKBACK` so a 7-d-old Hb is invisible to the bundle
  iff it is invisible to the deterministic classifier)
* `Vitals` — `±6h`

### Stable evidence ID

Every item in the bundle gets a sequential ID `E1, E2, ..., EN` assigned in
canonical emission order. The IDs are byte-stable across re-runs of the
same input — what `bba.quote_grounder` (#18) cites in LLM output, and what
`bba.audit_store.AuditRow.evidence_bundle_hash` is computed against.

### Canonical emission order

Outer order = the literal source order in
`bba.evidence_bundle_builder.models.EvidenceSource`:
`Diagnosis, IPDADMPROGRESS, IPDNRFOCUSDT, MED, Lab, Vitals`. Inner order
within each source is tuned for **truncation safety** so cap-pressure tail-drop
discards the least-relevant item first:

* `IPDADMPROGRESS` / `Vitals` / `IPDNRFOCUSDT` — closest-to-anchor first
* `Lab` (Hb) — HEMATOLOGY before POCT (PRD §3 source preference, regardless
  of recency); within source: newest-first; corrected (max `item_no`) before
  stale for same-(source, timestamp) ties
* `MED` — pre-anchor (decision context) before post-anchor (treatment after);
  within pre: newest-first; within post: closest-to-anchor first
* `Vitals` — pre-anchor before post-anchor (matches `bba.vitals_extractor`
  contract that pre wins regardless of distance)
* `Diagnosis` — by `(icd10, description is not None, description)`

### Canonical JSON

Sorted keys + 2-space indent + UTF-8 NFC-normalized strings (both keys AND
values) + no trailing newline. The contract is **byte-stable**: same input →
byte-identical output → same `bundle_hash`. NFC normalization is applied
recursively at every nesting level; non-finite floats (NaN, ±Inf) are rejected
because they are not valid JSON per RFC 7159.

### Bundle hash

`sha256(canonical_json.encode("utf-8")).hexdigest()`. The `EvidenceBundle`
model validator recomputes it at construction and rejects mismatches —
prevents a downstream rebuilder from pairing real canonical bytes with a
forged or stale hash. The model also locks the envelope shape (exactly
`{anchor, items}`, anchor must have `{order_datetime, hn_hash, an_hash,
products}` with tz-aware UTC `order_datetime`).

### Char-cap drop priority

Whole-item drop order under cap pressure (lowest clinical relevance first):
`IPDADMPROGRESS → IPDNRFOCUSDT → MED → Vitals → Lab → Diagnosis`. Within Lab,
POCT drops before HEMATOLOGY (PRD §3 source preference). When even an
anchor-only envelope exceeds the cap, the builder raises
`EvidenceBundleTooLargeError` rather than emitting an over-budget bundle —
the AC explicitly forbids silent over-cap.

### SOAP section priority

`IPDADMPROGRESS` notes are parsed into Subjective / Objective / Assessment /
Plan sections via inline-header regex (matches headers anywhere in the text,
not just line starts). Section emission order is `(ASSESSMENT, PLAN,
OBJECTIVE, SUBJECTIVE)` — most-important-first so the LLM reads the
clinician's diagnosis-time interpretation before supporting data. Truncation
walks the priority tuple in **reverse** (drop SUBJECTIVE first, ASSESSMENT
last). Empty progress items (after truncation OR construction) are pruned so
no `E_N` citation points at zero quoteable content.

### Deferred review (post-merge)

Codex adversarial review found 22 rounds of substantive issues, all closed.
Two operationally-relevant items intentionally not pursued in #16; defer to
a follow-up ticket if production usage surfaces them:

* **Section-truncation collapse for tied-section items** — when two
  `IPDADMPROGRESS` notes share the same single SOAP section and the
  section-drop pass empties both at once, both items get pruned together
  by `_drop_empty_progress_items` BEFORE the priority-aware whole-item
  drop runs. In practice this only matters when char_cap is small enough
  to force section-level truncation AND every progress note has the same
  one-section structure. The closest-first emission key + the broader cap
  budget make this rare; covered indirectly by Vitals truncation tests.
* **Hypothesis property test on the canonical-emission contract** — the
  current property tests cover hash invariance under input shuffles. A
  future hypothesis test could generate adversarial mixed-source bundles
  and assert the DROP_PRIORITY contract holds for any cap value above the
  anchor envelope size.

## Vocabulary not in this file

Domain terms used in the broader project but defined elsewhere (PRD issue
body, or future Phase-1 modules):

* `audit_id`, `INAPPROPRIATE`, `NEEDS_REVIEW`, `INSUFFICIENT_EVIDENCE`,
  `cohort_threshold`, `delta-Hb bypass`, `MTP`, `T1.MTP`,
  `hallucination_suspect`, etc. — see the PRD (issue #1) for definitions.
* `quote_grounder` — issue #18; consumes `bba.evidence_bundle_builder`'s
  stable `E_N` IDs to ground LLM citations.

Add a concept here when a module exposes it as part of its interface. Update
when a concept's shape changes; remove when it leaves the codebase.
