# CONTEXT.md — Phase 1 Module Glossary

This file names the load-bearing concepts exposed by the `src/bba/` modules.
Future architecture reviews and AI agents joining the project should speak
these terms verbatim; update entries here (instead of forking new vocabulary in
code) when a concept changes shape.

Scope: Phase 1 of the KCMH RBC transfusion audit pipeline. PRD: [issue #1].
Sections are added per ticket as each module merges to `main`. Currently
covered: `#3 ingest`, `#4 audit_orders`, `#5 hb_lookup`, `#6 vitals_extractor`,
`#16 evidence_bundle_builder`, `#18 quote_grounder`, `#19 audit_store`.

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

## Evidence-bundle-builder concepts (#16)

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

## Audit-orders concepts (#4)

### Blood-order input

One pre-joined `(HN, REQNO)` row carrying everything the filter needs:
BDVST identity + status fields, both anchor candidates (`REQ` and `BDVST`
date/time), the joined BDTYPE products, AN-scoped Diagnosis ICD-10 codes,
and the joined patient birthdate/sex from `UnUSE_Patient_Background`.
`bba.audit_orders.models.BloodOrderInput`. Joining lives upstream so the
filter is pure-Python and unit-testable without DuckDB.

### Audit order

One canonical row in the `audit_orders` table — the output of the filter
for an included input. `bba.audit_orders.models.AuditOrder`. Carries the
PRD §"Output schema" identity + anchor fields: `audit_id`, `hn`, `an`,
`reqno`, `order_datetime` (tz-aware UTC), `anchor_imputed`,
`products_ordered`, `age_years`, `sex`, `diagnosis_codes`. Consumed by
#5 (hb_lookup), #6 (vitals_extractor), #7 (cohort_detector), #8
(deterministic_classifier).

### Exclusion reason

Typed `Literal` enum of the gate that rejected a record:
`not_rbc_product`, `status_not_eligible`, `cancelled`, `no_an`,
`inter_hospital`, `hemoglobinopathy`, `aiha`, `tma`, `obstetric`,
`pediatric`. `ExcludedRecord.detail` carries the specific evidence (e.g.,
the ICD-10 code that triggered hemoglobinopathy) so reviewers can audit
the rule firing without re-joining source CSVs.

### Hard-exclusion ICD-10 set

Codes that block an order from the audit set regardless of other factors:
hemoglobinopathy `D55` / `D56` / `D57` / `D58`, AIHA `D59.x`, TMA `M31.1`,
obstetric `O`-chapter. Round 1 B1 (hemoglobinopathy hard-exclude); the
issue #4 AC is the authoritative list. `bba.audit_orders.rules` constants
(`HEMOGLOBINOPATHY_PREFIXES`, `AIHA_PREFIX`, `TMA_PREFIXES`,
`OBSTETRIC_PREFIX`).

### Boundary-aware ICD prefix match

`bba.audit_orders.rules._code_matches_prefix`. A raw `startswith` would
collapse the ICD-10 chapter boundary (e.g., `D550` would match `D55`).
The matcher requires either exact-length match or a `"."` continuation —
except for single-letter chapter prefixes (`"O"`), which accept digit
continuation to form the 3-char category (`O80`, `O09.9`). Case-sensitive;
ICD-10 is uppercase by convention and tolerating lowercase would also
tolerate other formatting drift.

### Anchor resolution

`bba.audit_orders.anchor.resolve_anchor` picks the best
`order_datetime` for a record. Primary: `REQDATE + REQTIME`. Fallback:
`BDVSTDATE + BDVSTTIME`, with `AnchorResolution.imputed = True` so the
emitted row carries `anchor_imputed = True`. A partial pair (date xor
time) is **not** a fallback — per PRD §1 the strict parser already
refused to invent a time, and the audit_orders filter must not re-introduce
that drift.

### Unrecoverable anchor

Neither REQ nor BDVST pair is usable. `bba.audit_orders.exceptions.UnrecoverableAnchorError`
is raised rather than silently dropping the record or emitting a row with
a null anchor — downstream stages (`hb_lookup`'s −7 d window,
`vitals_extractor`'s ±6 h window) cannot tolerate a missing anchor.

### Audit ID

Stable identity of a single audited order. `bba.audit_orders.identity.build_audit_id`
returns `sha256("<hn>:<reqno>")[:32]`. Same `(hn, reqno)` → same id
forever; filesystem-safe (lowercase hex only). Generated here, persisted
downstream as `AuditRow.audit_id`. The truncation is safe because
`(HN, REQNO)` is already unique by construction (PRD §2); the hash is
for stable filesystem-safe naming, not adversarial collision resistance.

### Filter result

Outcome of `bba.audit_orders.build_audit_orders` — a frozen pydantic
model partitioning every input into either `included: tuple[AuditOrder,
...]` or `excluded: tuple[ExcludedRecord, ...]`. The partition is total
(no silent drops) and disjoint (no double-counting), input ordering is
preserved within each bucket, and identity is deterministic across runs.
The two pipeline-level Hypothesis property tests assert these invariants
on generated input matrices.

## Vocabulary not in this file

Domain terms used in the broader project but defined elsewhere (PRD issue
body, or future Phase-1 modules):

* `cohort_threshold`, `MTP`, `T1.MTP`,
  `hallucination_suspect`, etc. — see the PRD (issue #1) for definitions.

Add a concept here when a module exposes it as part of its interface. Update
when a concept's shape changes; remove when it leaves the codebase.
