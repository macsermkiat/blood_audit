# CONTEXT.md — Phase 1 Module Glossary

This file names the load-bearing concepts exposed by the `src/bba/` modules.
Future architecture reviews and AI agents joining the project should speak
these terms verbatim; update entries here (instead of forking new vocabulary in
code) when a concept changes shape.

Scope: Phase 1 of the KCMH RBC transfusion audit pipeline. PRD: [issue #1].
Sections are added per ticket as each module merges to `main`. Currently
covered: `#3 ingest`, `#4 audit_orders`, `#5 hb_lookup`, `#6 vitals_extractor`,
`#16 evidence_bundle_builder`, `#17 deid_redactor`, `#18 quote_grounder`,
`#19 audit_store`, `#21 prompt_builder`, `#25 review_actions`,
`#28 report_generator`.

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

## Deid-redactor concepts (#17)

### Role token

The vocabulary of post-processed PHI tokens emitted by the wrapper:
`[ATTENDING]`, `[NURSE]`, `[PATIENT]`, `[FAMILY]` (the four role-bearing
upgrades), plus `[PERSON]` (generic, when no role cue is detected) and
the pass-through type-matching tokens `[DATE]`, `[LOCATION]`,
`[HOSPITAL]`, `[ID]`, `[PHONE]`. `bba.deid_redactor.models.RoleToken`.
Role-bearing tokens preserve the speaker/subject signal for downstream
LLM reasoning without leaking the underlying name (PRD §8: "post-
processing wrapper, not a fork of the redactor").

### PERSON-class tokens

The five role-bearing tokens (`[PERSON]`, `[ATTENDING]`, `[NURSE]`,
`[PATIENT]`, `[FAMILY]`) counted by the semantic-degradation detector.
`bba.deid_redactor.models.PERSON_CLASS_TOKENS`. Other token families
(`[DATE]`, `[LOCATION]`, ...) are excluded — their redaction does not
erode semantic content, so date-dense notes do not falsely fire the
flag.

### Redactor backend

`bba.deid_redactor.models.RedactorBackend` — the Protocol the wrapper
consumes for the underlying redactor. Production wires it to the
vendored `thai-medical-deid` package (PRD §"Stack": `TRANSFORMERS_OFFLINE=1`,
`HF_HUB_OFFLINE=1`, `HF_HOME=/opt/models`); tests use deterministic
stubs. The wrapper NEVER imports the HF model directly — keeps the
post-processing logic pure-function and zero-deps on the Anthropic SDK
or transformers, mirroring the `bba.quote_grounder` Protocol pattern.

### Backend redaction result

`BackendRedactionResult(text, spans)` — what the backend hands back per
input note. `text` is the redacted text already carrying placeholder
tokens; `spans` is a tuple of `RedactionSpan(start, end, entity_type,
original_text)` in document order, one per placeholder. The wrapper
walks `spans` in order to upgrade `[PERSON]` → role and `[DATE]` →
`Day N`; a count mismatch (more placeholders than spans, or vice
versa) raises `BackendRedactionError`.

### Quasi-identifiers

The five fields defining a k-anonymity equivalence class: `ward`,
`icd_3char`, `age_band`, `sex`, `admission_month`.
`bba.deid_redactor.models.QuasiIdentifiers` (Pydantic frozen,
hashable). Each field is format-validated at construction so
equivalent-but-formatted-differently inputs cannot silently split a
group: `admission_month` enforces `YYYY-MM`, `age_band` enforces
`LO-HI` with `lo ≤ hi`, `icd_3char` enforces exactly three characters,
and `sex` is the `SexCode` literal set `{M, F, U}` (`U` covers HOSxP
exports where sex is missing). Free-form strings here would silently
halve k by fragmenting the population — the validators are a
correctness contract, not a UX nicety.

### k-anonymity gate

PRD §8: bundles whose QI combo has fewer than 5 records in the
population (`k < K_ANONYMITY_MIN`) are routed to `NEEDS_REVIEW`, not
silently dropped. Two-step API: `compute_k_groups(records)` returns a
read-only mapping `QuasiIdentifiers → group size` over the full
population (one month of orders); the caller adapts it to the
`KAnonymityGate` Protocol (`lambda qi: groups.get(qi, 0)`) and passes
it to `redact_bundle`. The wrapper calls the gate once per request and
records `(k_anonymity_size, k_anonymity_passed)` on the result.
`bba.deid_redactor.k_anonymity`.

### Age cap

PRD §8: ages above 89 are collapsed to exactly 89 (HIPAA-derived
re-identification guard on the elderly tail). `apply_age_cap(age)`
returns `(capped, was_capped)`; the boundary is inclusive (age 89 is
NOT capped). The flag lands on `RedactionResult.age_capped` so the
audit row records whether the cap fired.
`bba.deid_redactor.age.apply_age_cap`.

### Date shift

PRD §8: dates inside redacted notes are remapped to Δ-days-from-
admission. Two paths run in sequence:

* **Backend-tagged spans first** — `shift_date_spans_in_text` walks
  `RedactionSpan(entity_type="DATE")` entries in document order and
  rewrites each `[DATE]` placeholder using the span's `original_text`
  parsed against the four canonical formats (`YYYY-MM-DD`,
  `YYYY/MM/DD`, `DD/MM/YYYY`, `DD-MM-YYYY` — all four-digit-year and
  zero-padded). Unparseable `original_text` keeps the placeholder
  (fail-open — the PHI itself is still redacted, only the Δ-day
  annotation is omitted).
* **Literal-date regex pass** — `shift_dates_in_text` then catches any
  backend-missed prose dates (e.g., when a date wasn't tagged as PHI).
  ISO formats win when overlap-resolved against day-first variants.

Output form is `Day N`: `Day 0` (admission), `Day +3` (positive
offset), `Day -2` (negative). The explicit `+` on positive offsets
distinguishes Δ-days from absolute hospital-day-of-admission counts.
Ambiguous source formats (two-digit year, Buddhist-year prefix,
decimal hour fragments) are NOT matched — the upstream ingest parser
(#3) has already routed those records to `parse_warning`.
`bba.deid_redactor.date_shift`.

### Semantic-degradation flag

PRD §8: a redacted note with strictly more than four PERSON-class
token starts within any 50-character sliding window flags as
semantically degraded → routes to `NEEDS_REVIEW`. The threshold uses
strict `>` (exactly four does NOT fire) and the unit is NFC characters
in the redacted text (post-role-mapping). Two-pointer sliding-window
implementation counts token-START positions; non-PERSON tokens
(`[DATE]`, `[LOCATION]`, ...) do not contribute. Fires when the LLM
would lose too much referential content to reason about the clinical
event safely. `bba.deid_redactor.semantic.detect_semantic_degradation`.

### Honorifics-only lexicon

`bba.deid_redactor.roles._HONORIFICS` — a narrower vocabulary used
ONLY for the span-internal cue pass in `classify_honorific_in_span`:
unambiguous titles and abbreviations (`Dr.`, `MD`, `RN`, `Pt.`,
`physician`, `attending`, `นพ.`, `พญ.`, `อาจารย์หมอ`,
`อาจารย์แพทย์`). Family terms (`son`, `mother`, `wife`) and the bare
`patient` / `nurse` words are excluded because they double as common
KCMH given/surnames; matching them in-span would mislabel a patient
named "Son" as `[FAMILY]` (codex GitHub review on PR #40 round 2).
The full role lexicon (`_ROLE_PATTERNS`) is still in use for the
proximity scan over the surrounding context.

### Proximity-aware role classification

`_classify_by_proximity(before, after)` — picks the role whose cue
match is closest to the span boundary. Each role's regex is scanned
over the before-window (distance = `len(before) - match.end()`) and
the after-window (distance = `match.start()`); smallest distance wins.
Equal-distance ties resolve by the global priority order (ATTENDING >
NURSE > PATIENT > FAMILY) so the classifier remains deterministic for
bundle-hash stability. Required because multi-actor sentences like
`"Dr. Smith saw patient John Doe"` would otherwise misclassify the
John-Doe span as `[ATTENDING]` on global priority alone (codex GitHub
review on PR #40 round 1).

### Default role classifier

`default_role_classifier(*, original_text, context, span)` — the
wrapper's built-in `RoleClassifier`. Three-step resolution: (1)
`classify_honorific_in_span` on the span's own `original_text` —
in-span titles ("Dr. Smith" as one span) are unambiguous; (2)
`_classify_by_proximity` on the ±40-char (`ROLE_CONTEXT_WINDOW`)
original-text window around the span; (3) priority-only
`classify_role_by_cues` on the caller-supplied `context` as a final
fallback. Returns `None` when no signal is present — the wrapper then
keeps the generic `[PERSON]` token rather than fabricating a role.
`bba.deid_redactor.roles`.

### Redactor version

`RedactorVersion(version, model_sha, gazetteer_version)` — the three-
field metadata stamped on every `RedactionResult` and persisted on
`AuditRow.redactor_version` + `redactor_model_sha`. PRD §"Output
schema" requires both fields so a future redactor upgrade does not
silently change the bundle hash on an old audit row's replay. The
gazetteer version covers the pinned PyThaiNLP vocabulary. Different
versions of these metadata fields produce different bundle hashes —
participate in the canonical envelope.
`bba.deid_redactor.models.RedactorVersion`.

### Redaction hash

`sha256(canonical_serialize(envelope).encode("utf-8")).hexdigest()` —
the byte-stable bundle hash for the redacted output, mirroring
`bba.evidence_bundle_builder.EvidenceBundle.bundle_hash`. The envelope
locks 8 fields (`notes`, `redactor_version`, `redacted_age`,
`age_capped`, `k_anonymity_size`, `k_anonymity_passed`,
`route_to_needs_review`, `needs_review_reasons`); any change in any
field changes the hash. The `RedactionResult` model validator
recomputes the hash at construction and rejects mismatches —
guarantees a downstream caller cannot forge a result whose hash
disagrees with its content.
`bba.deid_redactor.canonical.compute_redaction_hash`.

### NEEDS_REVIEW reason

`NeedsReviewReason` enum: `K_ANONYMITY_FAIL`,
`SEMANTIC_DEGRADATION`. Routing decision is OR-of-reasons —
multiple reasons may co-fire and all are persisted on
`RedactionResult.needs_review_reasons`. The downstream audit pipeline
(#24) reads this list into the row-level `review_reason` field so
reviewers see WHY a redacted bundle landed in their queue.
`bba.deid_redactor.models.NeedsReviewReason`.

### Deferred (post-merge)

Two items intentionally not landed in #17; defer to follow-up tickets:

* **`thai-medical-deid==X.Y.Z` pin in `pyproject.toml`** — the
  Protocol-based wrapper does not need the dep installed for unit
  tests; runtime metadata (`RedactorVersion`) carries audit-chain
  replay information today. The actual pin (alongside the vendored HF
  model SHA + PyThaiNLP gazetteer version) belongs with the audit-
  pipeline integration ticket (#24) where the backend is wired in for
  production. Codex acknowledged the deferral in both review rounds.
* **Honorifics-lexicon extension** — the current `_HONORIFICS` set
  intentionally errs on the side of caution (FAMILY has no honorifics
  because every family term doubles as a real name in KCMH
  demographics). If a future hand-labeled corpus surfaces clinically
  meaningful titles for FAMILY (e.g., religious / hierarchical
  Thai-context honorifics), they would land here with adversarial
  fixtures that demonstrate no name/cue collision.

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

## Eval-harness concepts (#20)

### Stratum

Mutually exclusive partition of the audited-orders population:
`HB_LT_7`, `HB_7_10`, `HB_GT_10`, `INSUFFICIENT`, `ADVERSARIAL`,
`COHORT_EXCEPTION`. `bba.eval_harness.models.Stratum`. The stratification
key is a function of the deterministic-classifier inputs (Hb tier,
evidence sufficiency, adversarial flag, cohort tag) — never of the LLM
prediction. Per-stratum target sizes from PRD User Story #25 live in
`DEFAULT_STRATUM_TARGETS`.

### Stratified-with-enrichment sample

Output of `stratified_with_enrichment(population, targets, rng_seed)`.
Per stratum: `drawn_positives = clamp(max(enrichment_cap,
target - pop_negatives), upper=pop_positives)`, remainder filled from
negatives. PRD §11's "~138 INAPPROPRIATE-positives per stratum" is the
ceiling; capped by target (adversarial = 80) and the available positive
pool. Per-stratum RNG fork (`rng.randrange` → sub-seed) means adding a
stratum to `DEFAULT_STRATUM_TARGETS` does not perturb existing draws.

### Inclusion probability

Per-case probability of being in the sample, recorded per stratum as a
pair: `positive_inclusion_probability` (drawn_positives / pop_positives)
and `base_inclusion_probability` (drawn_negatives / pop_negatives).
Consumed by the HT reweighter to undo the enrichment when the metric
the report cites is the *population* prevalence rather than the
*sample* prevalence. `bba.eval_harness.models.StratumDraw`.

### Horvitz-Thompson estimate

Population-prevalence point + variance + SE reweighted by inverse
inclusion probability. `bba.eval_harness.models.HorvitzThompsonEstimate`.
Variance uses the canonical Sarndal et al. 1992 eq 3.4.5 SRS-with-
replacement approximation, computed via the algebraic identity
`sum((w - mean)²) = sum(w²) - n*mean²`. A drawn case with non-positive
inclusion probability is a sampling-design bug and raises loud — never
silently skipped (would bias HT downward).

### Wilson score interval

Binomial-proportion confidence interval bounded in `[0, 1]`.
`bba.eval_harness.models.WilsonInterval`. PRD §11 mandates the Wilson
form over the normal approximation — Wilson does not leak out of `[0,1]`
at boundary prevalences (Hb<7, Hb>10), which would be a regulator-visible
defect. The inverse-normal quantile is the Acklam (2003) approximation
(absolute error ≤ 1.15e-9) so scipy is not a runtime dependency.

### Inter-rater agreement bundle

`AgreementResult` carries Cohen's κ, Gwet's AC1, PABAK, and the observed
agreement together — by design. κ deflates in the high-prevalence Hb>10
stratum because chance agreement inflates; AC1 (prevalence-resistant
chance baseline) and PABAK (`2*p_o - 1` for binary,
`(k*p_o - 1)/(k - 1)` for k-category) stay informative. When κ and AC1
diverge, the divergence is the prevalence signal the report cites, not
a coding bug. `bba.eval_harness.agreement.agreement_with_metrics`.

### Cluster-robust SE

CR0 sandwich variance estimator (Liang & Zeger 1986) for a binomial
proportion: `Var(p̂) = (1/n²) * sum_g (sum_{i ∈ g} (y_i - p̂))²`.
`bba.eval_harness.models.ClusterRobustEstimate`. Audited orders cluster
on physician + ward (PRD User Story #31); the naive binomial SE
understates uncertainty 1.5-2× under this design. Singletons collapse
to the naive SE by construction. **Refuses on `len(clusters) < 2`** — one
cluster yields zero residuals → zero-width CI → silent overconfidence.

### Temporal split strategy

`SplitStrategy = Literal["lomo", "blocked"]`, auto-selected by
`select_split_strategy` from the dataset's calendar-month span (NOT the
count of distinct months with data). `dataset_month_span` returns the
inclusive range `(last - first) * 12 + delta + 1`. Span `< 12` → LOMO,
`≥ 12` → blocked. `bba.eval_harness.splits`. The threshold matches the
seasonal-confounder horizon (one full year). Below-2-month single-month
data routes to blocked fallback via `temporal_cv_splits` so the report
never sees an empty-train fold.

### Temporal split

One train/holdout pair from LOMO or blocked CV. `train_audit_ids` and
`holdout_audit_ids` are tuples of `audit_id` strings; `holdout_label` is
`"YYYY-MM"` for LOMO and `"block-N"` for blocked. Splits are 1-1 with
the original cases — disjoint holdouts, complete coverage.
`bba.eval_harness.models.TemporalSplit`. Single-case input raises in
`blocked_temporal_split`; CV is undefined for `n=1`.

### Hierarchical multiple correction

Two-family multiplicity correction returned as
`HierarchicalCorrectionResult(primary, exploratory, alpha)`. **Primary**
hypotheses (confirmatory) get family-wise error control via Bonferroni
(`adjusted_p = min(1, raw_p * k)`); **exploratory** hypotheses
(hypothesis-generating) get FDR control via Benjamini-Hochberg step-up
(`q_(i) = min_{j ≥ i} m * p_(j) / j`, capped at 1). The two families
are corrected *independently* — pooling would inflate primary's any-
false-positive rate while deflating exploratory's discovery rate.
`bba.eval_harness.correction`.

### Verifier-as-classifier metrics

`ClassifierMetrics` consumes a `bba.quote_grounder.ConfusionMatrix`
(produced by #18 over the 200-row hand-labeled verifier-evaluation set)
and emits accuracy / sensitivity / specificity / PPV / NPV each with a
Wilson CI, plus a point-only F1. Empty-denominator edge cases (zero
positives, all-zero matrix) collapse to degenerate `[0, 0]` intervals
rather than raising — a fresh-pipeline run against an empty labeled set
still produces a valid (if uninformative) report row.
`bba.eval_harness.classifier.evaluate_confusion_matrix`.

### Outcome-anchored falsification

`FalsificationResult` grades INAPPROPRIATE predictions against hand-
coded chart-review outcomes on a labeled subset (PRD §11). Support
means the patient did not require further transfusion; contradiction
means `FURTHER_TRANSFUSION_24H` or `DEATH_FROM_ANEMIA_30D`. APPROPRIATE
predictions are skipped — the falsification target is the pipeline's
*positive* call, not its negative call. `FalsificationOutcome` is the v1
3-enum set; additional endpoints (24h Hb response, transfusion
reactions, doc amendment) are out-of-scope for #20 and tracked under
follow-up tickets.

### Deferred review (post-merge)

In-session Codex review went through 4 rounds; 7 findings (4 P0 + 3 P2)
were addressed before merge. Items intentionally **not** in scope and
documented here so a future reader does not re-litigate:

* **scipy reference in tests** — Wilson CI test fixtures use the
  precise mathematical formula (≤ 1e-6 agreement with scipy's
  `binomtest(...).proportion_ci(method='wilson')`), but scipy itself is
  not a runtime dependency. The publication script may compare against
  scipy externally; the harness never imports it.
* **HT variance under without-replacement sampling** — the variance
  estimator uses the SRS-with-replacement approximation (Sarndal et al.
  eq 3.4.5), which slightly inflates SE vs the without-replacement
  truth. Conservative for the regulator submission; tightening it would
  require finite-population-correction terms that complicate the report
  writer without changing the headline conclusion.
* **F1 confidence interval** — `ClassifierMetrics.f1` is a point
  estimate only. F1 has no published closed-form CI; bootstrapping would
  couple the harness to a heavy dependency without materially changing
  the reviewer-grade conclusion. Each component rate (sensitivity, PPV)
  carries its Wilson CI separately.

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

## Review-actions concepts (#25)

### Review action

A reviewer's verdict on a classified audit row. One of four kinds
(`ActionKind`): `agree` (confirm the pipeline), `override` (dissent;
requires `override_reason`), `escalate` (flag for senior review), and
`use_as_few_shot_candidate` (mark as exemplar — the audit pipeline #24
reads these back for prompt-cache few-shot blocks).
`bba.review_actions.models.ReviewAction`. Persisted append-only: revisions
file new rows rather than mutating prior ones, so the reviewer-decision
timeline for an audit row never loses a beat.

### Append-only invariant

Defense in depth on the mutable-state contract. Three independent layers,
each able to stop a mutation alone:

1. **REVOKE** — `UPDATE`, `DELETE`, `TRUNCATE` are revoked from
   `review_actions_app` and from `PUBLIC` on both tables.
2. **Trigger guard** — six `BEFORE` triggers (UPDATE / DELETE row + STMT
   TRUNCATE per table) fire
   `review_actions_block_mutation`, which raises `SQLSTATE P0001` with
   message `append_only_violation`.
3. **Startup integrity check** — see *Schema integrity check* below.

Layer (2) is load-bearing because (1) is bypassable by a superuser
misconfiguration; layer (3) catches operator drift between migrations
(a manually-dropped trigger, a re-granted privilege, a swapped guard
function).

### review_actions_app role

Least-privilege Postgres role the store connects as. Granted only
`INSERT, SELECT` on the two append-only tables (`review_actions`,
`phi_access_log`) and `USAGE, SELECT` on their sequences, plus `SELECT`
on `alembic_version` for the schema-head check. Migrations run as a
privileged operator role; the app role only ever inserts. The role lives
in the migration's idempotent `DO`-block so re-running the migration
against a partially-rolled-back DB does not trip on "role exists".

### PHI access log

Append-only record of every dashboard read of un-redacted text. One row
per `(reviewer_id, audit_id, hn_hash, an_hash, accessed_at)`, optionally
carrying `break_glass_justification` for clinical-emergency overrides of
the standard redaction policy. `bba.review_actions.models.PhiAccessLog`.
The `hn_hash` / `an_hash` columns are the SHA-256 hashes already computed
upstream (`bba.deid_redactor` #17); the un-hashed identifiers never enter
this module.

### Break-glass justification

Reviewer's free-text rationale for accessing PHI outside the standard
redaction policy (clinical emergency, attending request during code).
Required by the dashboard layer's break-glass workflow; persisted on
the `phi_access_log` row so post-hoc audit can trace the override.
Empty / whitespace-only values are rejected at the model boundary
(`PhiAccessInput`) — an empty justification defeats the audit trail
the field exists to defend.

### access_phi context manager

`ReviewActionsStore.access_phi(PhiAccessInput) -> Iterator[PhiAccessLog]`.
Writes the log row at `__enter__` and yields the persisted row; the
dashboard wraps its un-redacted-text retrieval in this context so the
log row is always committed before the text is surfaced. The row stays
on disk even if the body raises (an attempted access that crashed
mid-render is still an access for audit purposes). The structural
pattern is what makes the "log before reading" discipline the obvious
path in the dashboard code; programmatic enforcement is out of scope
for this module since the un-redacted text source-of-truth lives upstream.

### Trigger guard

A `BEFORE UPDATE | DELETE | TRUNCATE` trigger raising
`SQLSTATE P0001 'append_only_violation'`. The trigger function is
`public.review_actions_block_mutation`; six trigger rows install it on
the two protected tables. The user-facing message is the bare token
`append_only_violation` (Codex round 2 — earlier forms included
`TG_OP` / `TG_TABLE_NAME` and leaked schema details on the client side;
Postgres' own server-log machinery still records the full operation
context for operators).

### Schema integrity check

`ReviewActionsStore._ensure_migrated()` runs three independent checks
exactly once per process lifecycle, before any data mutation:

1. **Privilege check** — the connecting role is not a superuser, does
   not own a protected table, and is not a `pg_has_role(..., 'MEMBER')`
   member of the owner role. `'MEMBER'` (vs `'USAGE'`) catches
   NOINHERIT members who can still `SET ROLE` to the owner and then
   `DISABLE TRIGGER`.
2. **Alembic head match** — `current_revision()` and `head_revision()`
   are equal scalars; both functions raise `RuntimeError` on multi-head
   DBs or multi-head script directories, which the store wraps into
   `MigrationStateError`.
3. **Trigger shape** — every required trigger exists in the `public`
   schema with the right `(tgname, relname, relnamespace, proname,
   pronamespace, tgenabled, tgtype)`. Name-only verification is
   insufficient: a same-named trigger in another schema, a
   `DISABLE TRIGGER`-ed row, a swapped guard function, or a wrong-event
   trigger (BEFORE INSERT instead of BEFORE UPDATE) all slip past a
   simpler check.

Failures flow through `_raise_security(event, message, **fields)` which
logs a structured `WARNING` to the `bba.review_actions.security` logger
with stable event tags (`superuser_role_rejected`,
`owner_role_membership_rejected`, `alembic_revision_drift`,
`required_triggers_missing`, `trigger_guards_corrupted`,
`no_alembic_revisions`, `current_role_introspection_failed`) before
raising `MigrationStateError`.

### MigrationStateError

The single typed exception the store uses for every refused-to-operate
condition: unmigrated DB, drifted alembic head, missing / disabled /
swapped / wrong-event trigger, superuser or owner-member role.
`bba.review_actions.MigrationStateError`. Callers see one exception type
across all schema-health failure modes; the structured logger
distinguishes the specific event for alerting / dashboards.

### Multi-head alembic state

A revision graph with two unmerged heads, on disk or in the live DB's
`alembic_version` table. `current_revision` / `head_revision` raise
`RuntimeError` rather than silently picking an arbitrary head, and the
store translates the raise into `MigrationStateError`. Phase-1's single-
migration history can only enter this state via operator error
(directly editing `alembic_version`, or branching the script directory);
catching it loud prevents a deploy from running against a graph the
migrator cannot apply deterministically.

### AppendOnlyViolationError

The typed exception surfaced to the store's callers when a mutation
attempted through the store hits the trigger guard (currently only
reachable via a future defense-in-depth BEFORE INSERT trigger or an
operator-attached external trigger reusing the SQLSTATE+message
convention; the canonical UPDATE/DELETE path is blocked by REVOKE
before the trigger fires). `_translate_raise` matches by `SQLSTATE
P0001` AND the substring `append_only_violation` in the diagnostic
message — both must hold so a future P0001 raise for a different reason
does not get mis-typed.

### Security event logger

`logging.getLogger("bba.review_actions.security")`. Every integrity-check
rejection emits a `WARNING`-level record with an `extra` dict carrying
`review_actions_security_event`, `app_name`, and per-event fields
(`current_user`, `owner`, `missing`, `broken`, `current`, `head`).
Operators subscribe to alert on repeated bypass attempts; the event tag
is the stable identifier to grep / route on, never the free-form message.

## Report-generator concepts (#28)

### Monthly report row

The input contract for the report generator, named `MonthlyReportRow`. It
is **decoupled** from `bba.audit_store.AuditRow`: the audit store persists
the technical row (hashes, model_id, evidence-bundle hash, ...) while the
report generator consumes a report-shaped row that adds the ingest-side
metadata the audit-store schema does not carry (`ward_id`,
`physician_id`). The upstream layer — the monthly CLI or a dashboard
query — joins the two and produces a tuple of `MonthlyReportRow`. Keeping
the input shape report-specific isolates the generator from
audit-store schema drift; the join is the documented seam.

### Reproducibility footer

Six identifiers stamped on every CSV data row and every PDF page:
`policy_version`, `model_id`, `redactor_version`, `redactor_model_sha`,
`prompt_hash`, `evidence_bundle_hash`. The first three are named
explicitly in the ticket scope; the other three come from the broader
PRD §"Output schema" so a six-month-later auditor can reconstruct any
report line from the source `AuditRow`s. Stamping **per row** (not only
as a trailing line) keeps the chain intact under `grep` / `awk` / partial
extracts; the PDF mirrors the same six fields across two lines per page
(single-line layout overflows the printable LETTER width for realistic
SHA values).

### Per-physician artifact separation

The "own-data only" property of `physician_own_view` (PRD user story #10
— a physician sees their own ordering vs peer-anonymous benchmarks
without other physicians being publicly identified) is enforced at the
**artifact level**, not by trusting downstream distribution to filter
rows. `ReportArtifacts` carries two distinct mappings: `csv_paths` keyed
by the five committee-wide `SectionName`s
(`hospital_trend`, `ward_scorecard`, `indication_distribution`,
`cohort_exception`, `pipeline_health`) and `physician_own_view_csv_paths`
keyed by `physician_id`. Each per-physician file holds exactly one data
row; distributing the file to that physician therefore cannot leak
another physician's rate. The committee PDF still renders an internal
physician-own-view table for committee review.

### Month bucket

The monthly report bucket is the hospital business month
(**Asia/Bangkok**, no DST), not the UTC month. PRD §"Tz-aware
throughout" stores datetimes UTC and renders Asia/Bangkok; user
story #19 spells out "so that month-boundary orders bucket correctly".
`filter_rows_for_month` converts `[month, next_month)` interpreted in
`zoneinfo("Asia/Bangkok")` to UTC for the comparison, so an order
placed at 23:00 Bangkok on the 31st (which is 16:00 UTC, in the
local-month and the prior UTC-month) lands in the correct report. The
year-rollover branch in `_next_month_first_of` is pinned by a
December→January regression test.

### Section

One of six report views, declared by the `SectionName` Literal:
`hospital_trend`, `ward_scorecard`, `physician_own_view`,
`indication_distribution`, `cohort_exception`, `pipeline_health`. The
PDF iterates all six; the CSV writer emits five of them as
committee-wide files and the sixth as per-physician files (see
"per-physician artifact separation" above). Each section has a fixed
column order locked in
`bba.report_generator.csv_writer._data_columns` and asserted by a
byte-identical golden-snapshot test.

### Empty-section sentinel row

When a section's aggregation yields zero rows for the month, the CSV
writer still emits one synthetic data row instead of leaving only the
header. The data columns are **type-appropriate** placeholders — `"0"`
for int columns, `"0.0"` for float columns, `""` for string / date
columns, declared in `_EMPTY_SECTION_DEFAULTS` — and the footer columns
hold the real reproducibility identifiers. Blank cells in
`total_orders` / `*_rate` would NaN-poison a pandas read; the typed
zero placeholders parse cleanly and semantically mean "zero orders in
this empty section". Without the row at all, a downstream consumer
could not tell which policy / model / redactor versions produced the
empty result vs. a stale file in the same directory.

### Pipeline-health bucket exclusivity

`PipelineHealthRow` carries three counters that **overlap by design**:
`classified_orders` (rows whose `final_classification` is `APPROPRIATE`
or `INAPPROPRIATE`), `needs_review_count`, and
`insufficient_evidence_count`. The overlap arises because a row whose
terminal label is `APPROPRIATE`/`INAPPROPRIATE` and whose
`needs_human_review` flag is set contributes to both
`classified_orders` and `needs_review_count`. The flag is **scoped to
APPROPRIATE/INAPPROPRIATE rows only** — an `INSUFFICIENT_EVIDENCE` row
with the flag set is counted exclusively in
`insufficient_evidence_count`, so a documentation-absence spike (PRD
§"Documentation absence ≠ INAPPROPRIATE") cannot masquerade as an
LLM-review spike. The schema therefore does **not** assert
`classified + needs_review + insufficient_evidence == total`.

### Indication-code dedup

`aggregate_indication_distribution` counts each *distinct* indication
code at most once per order. A row whose `indication_codes` tuple
contains the same code twice (e.g., from an upstream join error)
contributes 1, not 2, so `total_orders` for any code never exceeds
`len(rows)` and `share` never exceeds 1.0. A multi-indication order
with two *different* codes still contributes 1 to each (and `share`
across codes can exceed 1.0; this is documented behavior, not a bug).

### Golden snapshot

A per-section byte-identical CSV test. Each of the six sections has a
class in `tests/unit/test_report_generator.py` that builds a fixed
input, runs the aggregation + writer, and asserts the resulting bytes
match a hand-written expected literal (header + data rows + footer
cells). The expected literal locks every detail that would otherwise
drift silently: column order, row ordering, float formatting
(`0.5`/`0.0` never `0`), line endings (`\n`), encoding (UTF-8 no BOM),
the six-field footer. A refactor that touches any of those produces a
mismatch and forces a deliberate update.

### Filesystem-safe ID

`MonthlyReportRow.audit_id`, `.ward_id`, `.physician_id` and
`PhysicianOwnViewRow.physician_id` are annotated `SafeFsId` —
`Annotated[str, AfterValidator(_validate_safe_fs_id)]`. The validator
requires `[A-Za-z0-9._-]+`, rejects empty, and rejects exactly `.` or
`..`. The defense exists because `physician_id` flows directly into
`physician_own_view_<physician_id>.csv` and a hostile or buggy upstream
value containing `/` or `../` would otherwise let the per-physician
write escape `output_dir`. Mirrors `bba.audit_store.models.SafeId`.
`physician_own_view_filename` re-validates the id as defense in depth
for callers that bypass the model boundary.

### Deferred review (post-merge)

Codex review went through 4 in-session rounds and 2 GitHub PR rounds;
every finding was either fixed before merge or is documented below as a
deliberate non-goal:

* **PDF byte-identical contract** — reportlab embeds a non-deterministic
  generation timestamp in the trailer; the PDF test asserts magic-bytes
  + non-trivial size only. Locking byte-identity would require either
  patching reportlab's timestamp or switching to a hand-rolled minimal
  PDF; both are out of scope for a "thin" module.
* **Empty-section row validation** — the sentinel row in
  `indication_distribution.csv` carries `total_orders="0"`, but
  `IndicationDistributionRow.total_orders` has `ge=1` in the Pydantic
  model. The CSV writer bypasses the model when emitting the sentinel
  (it writes raw text via `csv.writer`), so the constraint is honored
  in-memory but the persisted empty-section file is technically
  inconsistent with the schema. The pragmatic choice over either
  weakening the model constraint or dropping the sentinel — pin if a
  downstream consumer ever round-trips an empty `indication_distribution.csv`
  through the model.
* **Hypothesis property test breadth** — the existing property test
  only varies classification labels with `min_size=1`. A future
  expansion could cover adversarial dates (year-rollover, DST-adjacent
  zones if scope grows beyond Asia/Bangkok), empty inputs, and
  needs_human_review-flag interactions. Acceptable for a "thin"
  module per the original Codex round-1 INFO note.

## Prompt-builder concepts (#21)

### Task mode

The two LLM-eligible audit branches the prompt builder dispatches on:
`HB_7_10_REVIEW` (gray-zone Hb 7-10 g/dL — the LLM searches the ±24-h
note window for Tier-1 indications and Tier-2 supportive context) and
`HB_GT_10_OVERRIDE` (Hb > 10 g/dL pre-classified `POTENTIALLY_INAPPROPRIATE`
by the deterministic engine — the LLM looks for Tier-1 override
conditions that would justify the order). `bba.prompt_builder.TaskMode`
(Literal) and `TASK_MODES` (runtime frozenset). Adding a third branch
is a Phase-2 contract change. Each mode has its own system-prompt
template; the templates differ in their mode-discriminating phrasing
so a downstream test can distinguish the two without parsing.

### Cohort threshold

The hemoglobin threshold (in g/dL) the deterministic engine assigns
to a cohort, passed to the LLM as a **hard numeric input** — never
re-derived (PRD §"Cohort detection is deterministic, not LLM-judged").
Exactly three values: `7.0` (default), `7.5` (cardiac surgery), `8.0`
(ortho+cardiac comorbidity / ESRD on EPO). `ALLOWED_COHORT_THRESHOLDS`
is the frozen set; `CohortThreshold` is the Pydantic-validated type;
the system prompt renders the value as a single-decimal numeric literal
(`"7.0"` / `"7.5"` / `"8.0"`) so the LLM sees the same string for the
same cohort across audit rows.

### Evidence chunk

The prompt builder's input unit: `EvidenceChunk(evidence_id, source,
text)`, where `evidence_id` is the stable bundle ID (`E1`, `E2`, ...)
preserved through `bba.evidence_bundle_builder` and
`bba.deid_redactor`, and `text` is the **post-redaction** content
treated as opaque (the prompt builder never re-redacts). Empty /
whitespace-only `text` is rejected at the model boundary so a blank
chunk cannot smuggle past the `EMPTY_EVIDENCE` routing into the LLM.

### Evidence envelope

Every redacted evidence chunk wraps in `<evidence id="E1"
untrusted="true">...</evidence>` (PRD §38). The `untrusted="true"`
attribute is a fixed signal — every chunk has crossed the deid
boundary and must be treated as adversarial regardless of source.
Wrapping is **byte-identity preserving**: NFC-normalize only, no XML
escape. Clinical text routinely contains `<`, `>`, `&` in comparisons
(`Hb < 8`, `SBP > 90`, `K&Na panel`); escaping them would make the
LLM see a different byte sequence than the redacted source text that
`bba.quote_grounder` verifies citations against, silently failing
all grounding for affected chunks. `bba.prompt_builder.wrap_evidence`
+ `wrap_evidence_chunks`.

### Envelope-escape attack

A chunk whose content embeds literal `</evidence>` or `<evidence ...>`
tokens, intended to break out of the wrapper's boundary and start a
nested envelope the LLM might trust. Defense lives in the injection
scanner (`ENVELOPE_ESCAPE` category, two patterns:
`envelope_close_tag_v1`, `envelope_open_tag_v1`), not the wrapper —
the wrapper preserves byte identity for legitimate citations and
relies on the scanner to route adversarial content to `NEEDS_REVIEW`
before assembly.

### Injection scanner

Pre-LLM regex-based detector for adversarial content in evidence
chunks. NFC-normalized, case-insensitive for ASCII. The shipped
catalog is `INJECTION_PATTERNS`, 24 patterns across 8 categories:
`IMPERATIVE_VERB_EN` (5), `FAKE_GUIDELINE` (5),
`SYSTEM_PROMPT_EXFIL` (3), `ROLE_PRETEND` (3), `IMPERATIVE_VERB_TH`
(3), `JAILBREAK_TH` (2), `JAILBREAK_EN` (1), `ENVELOPE_ESCAPE` (2).
`MIN_REQUIRED_INJECTION_PATTERNS = 20` is the contract floor (issue
#21 AC). `scan_injection` runs on one chunk's text; `scan_chunks`
aggregates over a chunk sequence into an `InjectionVerdict`.

### Injection match

One detected hit: `InjectionMatch(category, pattern_id, evidence_id,
span_text, start, end)`. The full record participates in the
`prompt_hash` envelope — swapping `evidence_id` / `span_text` /
offsets between two otherwise-identical results yields different
hashes, preserving the audit-chain replay invariant for
reviewer-visible injection evidence.

### Injection verdict

The aggregate verdict over all chunks: `InjectionVerdict(flagged,
matches)`. `flagged` is `True` iff `matches` is non-empty — the two
fields exist together so callers can short-circuit on the boolean
without unpacking, and a model validator rejects any desync.
`PromptBuildResult` additionally enforces that
`injection_verdict.flagged` agrees with `INJECTION_DETECTED`
membership in `needs_review_reasons`; reconstructing the result with
the two desynced is rejected even when the hash is recomputed.

### Fabricated-version criteria

The fake-guideline patterns flag only **fabricated** version
references — minor versions with >=2 trailing nines (e.g. `99.99`,
`17.999`), a leading 200+ digit (`200..`), or 3+ minor digits.
Applies to both English (`fake_pr_guideline_v1`) and Thai
(`fake_thai_pr_v1`) patterns. The real KCMH `PR 17.2` and AABB
`PR 17.2`-style legitimate references stay below the threshold and
do not false-flag in either language.

### Few-shot example + few-shot block

A committee-approved exemplar: `FewShotExample(name, user_payload,
assistant_output)`. The block is the LAST cacheable region of the
prompt — Anthropic prompt-cache marker boundary — so its byte
stability across audit rows underwrites cache-hit rate. Input order
is preserved (the committee ranks examples). NFC-normalized, no
trailing newline, `"\n\n"` between-example separator.
`bba.prompt_builder.build_few_shot_block`.

### Prompt block + cache marker

The unit of the assembled prompt: `PromptBlock(role, text,
cache_marker)`. Roles are `"system"` or `"user"`; the builder never
emits assistant blocks (few-shot examples land inside the leading
user block). `cache_marker=True` is the Anthropic cache-breakpoint
signal at the END of that block. The builder emits TWO cache markers
when few-shot is present (system end + few-shot end — partial-prefix
hits for system-alone vs system+few-shot) and ONE when no few-shot
examples are supplied. The trailing per-row user-payload block is
NEVER cacheable (changes per audit row).

### Prompt build request

`PromptBuildRequest(task_mode, cohort_threshold, evidence_chunks,
few_shot_examples)`. Frozen Pydantic; rejects duplicate
`evidence_id` values across `evidence_chunks` at the model boundary —
downstream `bba.quote_grounder` treats non-unique `cited_id` as
`CITED_ID_NOT_FOUND`, so duplicates would silently fail every
legitimate citation against that ID.

### Prompt build result

`PromptBuildResult(blocks, task_mode, cohort_threshold,
injection_verdict, route_to_needs_review, needs_review_reasons,
prompt_hash)`. Frozen; the model validator recomputes
`compute_prompt_hash(envelope)` at construction and rejects any
mismatch, mirroring the `EvidenceBundle.bundle_hash` and
`RedactionResult.redaction_hash` audit-chain invariants. Two
internal-consistency validators also fire: `route_to_needs_review`
must be the OR of `needs_review_reasons`, and
`injection_verdict.flagged` must agree with `INJECTION_DETECTED`
membership in `needs_review_reasons`.

### Needs-review reason

`NeedsReviewReason` enum carrying the routing tag the audit pipeline
(#24) reads into the row-level `review_reason` field:
`INJECTION_DETECTED` (one or more scanner hits across the chunks),
`EMPTY_EVIDENCE` (zero chunks supplied). OR-of-reasons routing —
multiple reasons may co-fire and all are persisted on
`PromptBuildResult.needs_review_reasons`.

### Prompt hash

`sha256(canonical_serialize(envelope).encode("utf-8")).hexdigest()` —
the byte-stable hash for an assembled prompt, mirroring
`EvidenceBundle.bundle_hash` and `RedactionResult.redaction_hash`.
The canonical envelope locks 6 fields (`blocks`, `task_mode`,
`cohort_threshold`, `injection_matches` with FULL match records,
`route_to_needs_review`, `needs_review_reasons`); any change to any
field — including a single match's `evidence_id` / `span_text` /
offsets — changes the hash.
`bba.prompt_builder.canonical.compute_prompt_hash`.

### Deferred review (post-merge)

Codex review went through 4 in-session rounds plus a GitHub Codex
bot review on PR #43; every finding addressed before merge. Items
intentionally **not** in scope and documented here so a future
reader does not re-litigate:

* **Medical-NLI gate on prompt content** — the prompt builder does
  not invoke the optional medical-NLI entailment gate that
  `bba.quote_grounder` Layer 7 supports; the gate is a verifier
  concern, not an assembly concern. The prompt builder hands a
  byte-identical envelope to the LLM and downstream verification
  runs the NLI gate when configured.
* **Adversarial pattern catalog growth** — the shipped 25-pattern
  catalog clears the AC floor of 20 across all required categories;
  expansion against a future hand-labeled adversarial corpus is a
  follow-up ticket. The catalog structure (`InjectionPattern`
  records keyed by stable `pattern_id`) was designed so additions
  do not perturb the existing pattern_id namespace.
* **Anthropic Messages-API translation** — the prompt builder emits
  abstract `PromptBlock(role, text, cache_marker)` segments; the LLM
  client (#22) translates these into the Anthropic Messages API
  shape at its boundary. Keeping the translation in `bba.llm_client`
  isolates the prompt builder from SDK-version drift.

## Vocabulary not in this file

Domain terms used in the broader project but defined elsewhere (PRD issue
body, or future Phase-1 modules):

* `MTP`, `T1.MTP`, `hallucination_suspect`, etc. — see the PRD
  (issue #1) for definitions.

Add a concept here when a module exposes it as part of its interface. Update
when a concept's shape changes; remove when it leaves the codebase.
