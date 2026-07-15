# CONTEXT.md — Phase 1 Module Glossary

This file names the load-bearing concepts exposed by the `src/bba/` modules.
Future architecture reviews and AI agents joining the project should speak
these terms verbatim; update entries here (instead of forking new vocabulary in
code) when a concept changes shape.

Scope: Phase 1 of the KCMH RBC transfusion audit pipeline. PRD: [issue #1].
Sections are added per ticket as each module merges to `main`. Currently
covered: `#3 ingest`, `#4 audit_orders`, `#5 hb_lookup`, `#6 vitals_extractor`,
`#16 evidence_bundle_builder`, `#17 deid_redactor`, `#18 quote_grounder`,
`#19 audit_store`, `#21 prompt_builder`, `#22 llm_client`,
`#23 confidence_calibrator`, `#24 audit_pipeline`, `#25 review_actions`,
`#26 dashboard`, `#27 monitoring`, `#28 report_generator`, `#29 cli`.
Also covers: `#76 hemodynamic/periop evidence`, Hb-anchor unification,
missing-Hb positive-evidence pre-pass, and clinical-salience MED ordering.

Phase 2 (platelet auditor) is now **fully integrated and MERGED** (PR #85): the
deterministic CORE (`component_map`, `platelet_lookup`, `platelet_classifier`
§5.1 gate) plus intake (`audit_orders` admits platelet-only orders as
`component="platelet"`, isolated from RBC report/dashboard stats),
pipeline/replay dispatch, the platelet LLM leg (`PLATELET_REVIEW` prompt +
grounded-hard-signal over-clear guardrail + resume/injection handling),
sampling, and the pilot drivers. The LLM leg is gated behind default-off
`feature_flags.PLATELET_LLM_ENABLED` — only the deterministic leg is live
(validated on real data; auto-clears NOTHING: plt≥100→review, else→LLM). The
RBC path is byte-identical throughout. Concepts are grouped under the
`(Phase 2)` sections at the end of this file.

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

* `Hemodynamic` — **pinned, no time window** (caller pre-filters notes;
  summary is always emitted as E1 and is exempt from whole-item truncation)
* `Periop` — **pinned, no time window** (same pinned/exempt contract as Hemodynamic)
* `Diagnosis` — **AN-scoped** (no time window; full ICD-10 list for the encounter)
* `IPDADMPROGRESS` — `±24h`, cap 8 closest entries (closest-by-abs-offset)
* `IPDNRFOCUSDT` — `±24h`, cap 10 entries via 5-before / 5-after closest-first
* `MED` — `[-72h, +24h]` (asymmetric: drug history + post-order administration)
* `Lab` (Hb history) — `[-7d, anchor]` strict at lower bound (matches
  `bba.hb_lookup`'s `< _LOOKBACK` so a 7-d-old Hb is invisible to the bundle
  iff it is invisible to the deterministic classifier); upper bound is
  `hb_anchor` when the resolver anchored the Hb on a post-order draw
  (see `OrderAnchor.hb_anchor`), otherwise `order_datetime`
* `Vitals` — `±6h`

### Stable evidence ID

Every item in the bundle gets a sequential ID `E1, E2, ..., EN` assigned in
canonical emission order. The IDs are byte-stable across re-runs of the
same input — what `bba.quote_grounder` (#18) cites in LLM output, and what
`bba.audit_store.AuditRow.evidence_bundle_hash` is computed against.

### Canonical emission order

Outer order = the literal source order in
`bba.evidence_bundle_builder.models.EvidenceSource`:
`Hemodynamic, Periop, Diagnosis, IPDADMPROGRESS, IPDNRFOCUSDT, Med, Lab, Vitals`
(the literal is `Med`; the rest of this doc writes `MED` as shorthand for the
MED.csv source). Inner order within each source is tuned for **truncation safety**
so cap-pressure tail-drop discards the least-relevant item first:

* `IPDADMPROGRESS` / `Vitals` / `IPDNRFOCUSDT` — closest-to-anchor first
* `Lab` (Hb) — HEMATOLOGY before POCT (PRD §3 source preference, regardless
  of recency); within source: newest-first; corrected (max `item_no`) before
  stale for same-(source, timestamp) ties
* `MED` — clinical salience is the **PRIMARY** key (CRITICAL → ROUTINE →
  MAINTENANCE; see *Clinical salience* below). The sort is STABLE, so within a
  salience bucket the underlying order is preserved: pre-anchor (decision context)
  before post-anchor (treatment after), newest-first within pre, closest-to-anchor
  first within post
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

### EXEMPT_FROM_DROP

`bba.evidence_bundle_builder.builder.EXEMPT_FROM_DROP` — a frozenset of
`EvidenceSource` literals that the whole-item truncation pass may never shed:
`{"Hemodynamic", "Periop", "Lab"}`. These are the load-bearing decision signals:
`Lab` (Hb) is the decision-time anemia value the deterministic classifier keys on;
`Hemodynamic` is the pinned MAP/vasopressor summary starved in Case 2 / REQNO
68012352; `Periop` is the pinned surgical-context/EBL summary the LLM skimmed
past on Case 107 / REQNO 68074627. All are bounded in count, so exempting them
cannot make the bundle grow without limit. If the exempt set plus the anchor
envelope alone exceeds the cap, `EvidenceBundleTooLargeError` is raised rather
than shedding a load-bearing signal.

### Char-cap drop priority

Whole-item drop order under cap pressure (lowest clinical relevance first):
`IPDADMPROGRESS → IPDNRFOCUSDT → MED → Vitals → Diagnosis`. `Lab`,
`Hemodynamic`, and `Periop` are absent — they are in `EXEMPT_FROM_DROP`.
Within `Lab`, POCT drops before HEMATOLOGY (PRD §3 source preference).
When even an anchor-only envelope exceeds the cap, the builder raises
`EvidenceBundleTooLargeError` rather than emitting an over-budget bundle —
the AC explicitly forbids silent over-cap.

### DEFAULT_CHAR_CAP

`DEFAULT_CHAR_CAP = 40_000`. Raised from the original 8 K (issue #16) to 40 K
(issue #76): the old cap was a root cause of Case 2 / REQNO 68012352 — narrative
suppression plus the 8 K ceiling shed the vasopressor and MAP evidence the LLM
needed. The bundle now ships the full SOAP prose narrative **plus** the pinned
hemodynamic and peri-op summaries, which require the headroom. 40 K is
**provisional**: it will be finalized against the pilot worst-case bundle
measurement before any clinical use.

### Clinical salience (MED ordering)

`bba.evidence_bundle_builder.salience.med_salience(drug) → SalienceBucket` —
maps a MED `drug` string to one of three coarse buckets:

* `CRITICAL (0)` — vasopressors, inotropes, blood products (drop last under
  cap pressure). Matched by `_CRITICAL_RE` with word boundaries. Full generic
  names only; ambiguous short tokens (`NA`, `NAD`) are deliberately absent.
* `ROUTINE (1)` — the default; anything not recognised as critical or maintenance.
* `MAINTENANCE (2)` — crystalloids, saline flushes, irrigation (drop first).

Precedence is CRITICAL → MAINTENANCE → ROUTINE so a pressor diluted in saline
stays CRITICAL. `SalienceBucket` is an `IntEnum`: lower value = emitted earlier =
shed later. This is an ORDERING signal only — it never gates, classifies, or
weights a transfusion decision. `bba.evidence_bundle_builder.salience`.

### hb_anchor and window_anchor (OrderAnchor)

Two optional fields on `bba.evidence_bundle_builder.models.OrderAnchor`:

* `hb_anchor` — the Hb-lookup upper-bound anchor when it differs from
  `order_datetime`. `bba.hb_lookup.resolve_hb_with_fallback` can anchor on a
  post-order draw (a lab drawn minutes after REQTIME). Without this field the
  default `h.timestamp <= order_datetime` filter would drop the very value that
  routed the case to the LLM. Set to the fallback draw's timestamp on a hit;
  `None` on the common order-time path.
* `window_anchor` — the point every per-source window (progress, focus, meds,
  Hb, vitals) is centred on. Defaults to `order_datetime`. For reserve-ahead
  elective orders (blood crossmatched days before the transfusion), the caller
  sets this to the transfusion datetime via `bba.hb_lookup.resolve_evidence_anchor`
  so the bundle captures op-day evidence rather than reservation-day context.
  `order_datetime` stays as the audit identity anchor; `window_anchor` is
  windowing-only and does NOT participate in the hashed bundle envelope.

### periop_summary sidecar (EvidenceBundle)

`EvidenceBundle.periop_summary: PeriopSummary | None` — a convenience return
handle carrying the `PeriopSummary` scanned from the same note set that became
the bundle's items. Used by downstream deterministic guardrails (e.g. the
replay contradiction check in `bba.audit_pipeline.replay`). It is **not**
serialized into `canonical_json` and therefore does **not** participate in
`bundle_hash`. A bundle reconstructed from stored bytes carries `None`; the
guardrail simply has no signal to act on in that case.

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

### AnchorCandidate

A fallback Hb anchor with display text and provenance for the review page.
`bba.hb_lookup.anchor.AnchorCandidate` — frozen dataclass `(anchor_utc,
display, reason)`. `anchor_utc` is the tz-aware UTC datetime to look the Hb
back from; `display` is the local-time string surfaced in the review page;
`reason` records which fallback fired (e.g. `"issue_datetime"`,
`"blood_bank_visit_fallback"`). The pilot builds the ordered list from BDVSTDT
issue/use rows in `scripts/pilot/_anchor_candidates.py`.

### resolve_hb_with_fallback

`bba.hb_lookup.anchor.resolve_hb_with_fallback(*, observations, order_datetime,
candidates) -> (HbLookupResult, anchor_display, anchor_reason)`.

The primary anchor is always `order_datetime`. On a miss, each candidate is
tried in order; a candidate strictly **before** the order is skipped (a fallback
may be slightly after — lab drawn minutes post-REQTIME — but never before).
The first candidate that yields a non-missing Hb wins. If none do, the original
order-time (missing) result is returned. Closed over the ordering rule so the
deterministic report leg and the LLM gate leg resolve the Hb identically —
preventing the divergence bug documented in
`docs/handoff-hb-anchor-unification.md` (case 7 / REQNO 68066907).

### DEFAULT_REANCHOR_THRESHOLD

`DEFAULT_REANCHOR_THRESHOLD = timedelta(hours=24)`. Blood reserved for elective
surgery is crossmatched and held days before it is issued. Anchoring the evidence
windows on the reservation REQTIME then misses the entire transfusion context
(the op-day Hb drop and operative notes). When the issue datetime lands this far
or more after the order, `resolve_evidence_anchor` re-anchors the evidence
windows onto the issue datetime. 24 h cleanly separates same-admission same-day
issue (no re-anchor) from the reserve-ahead case.

### EvidenceAnchor

`bba.hb_lookup.anchor.EvidenceAnchor` — frozen dataclass `(anchor_utc, reason,
gap_hours, display)` specifying where the per-source evidence windows should be
centred. `reason` is `"order_datetime"` for the common case (windows track the
order) or `"issue_reanchor"` when an elective pre-reserved order was issued
materially later than it was reserved. `gap_hours` is the issue-minus-order lag in
hours (0.0 when not re-anchored); `display` is the issue anchor's local-time
string for the review page (`""` when not re-anchored).

### resolve_evidence_anchor

`bba.hb_lookup.anchor.resolve_evidence_anchor(*, order_datetime, candidates,
threshold) -> EvidenceAnchor`. Picks the evidence-window anchor for one order.
The order anchor wins unless an `"issue_datetime"` candidate lands `threshold`
or more after the order (the reserve-ahead elective case). Only
`"issue_datetime"` candidates re-anchor: the blood-bank *visit* timestamp tracks
the reservation, not the transfusion, and must never move the windows. The first
qualifying issue candidate wins (the pilot builder orders the issue datetime
ahead of the blood-bank fallback).

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

### Hemodynamic scan

`bba.vitals_extractor.hemodynamic.scan_hemodynamics(notes) → HemodynamicSummary`.
A **window-wide** pass (distinct from the single-note `extract_vitals` selection)
over every note in the window for two signals starved in Case 2 / REQNO 68012352:

* **MAP nadir** — the lowest *measured* mean arterial pressure across the whole
  window (not the most-recent reading), because shock severity is the worst
  point, not the latest. Two notations recognised: labelled `MAP 56` (goal/target
  phrasing excluded) and parenthesised `ABP 77/44 (56)` (kept only when the
  bracketed value is physiologically between diastolic and systolic).
* **Vasopressor mentions** — the canonical agent name and dose phrase (if
  charted), deduplicated by agent, keeping the earliest mention.

Pure aggregation; empty input yields an empty summary. Caller owns windowing.

### VasopressorMention

`bba.vitals_extractor.models.VasopressorMention` — frozen Pydantic model: the
canonical `agent` name, the raw `dose` phrase if charted (`None` otherwise), and
provenance (`at` = tz-aware UTC timestamp, `source` = origin table). Recognised
agents: norepinephrine (Levophed / NE / nor-adrenaline / นอร์อะดรีนาลีน),
epinephrine, dopamine (โดพามีน), dobutamine, vasopressin. Ambiguous abbreviations
(`NA` = sodium, `NAD` = no acute distress) are deliberately absent.

### HemodynamicSummary

`bba.vitals_extractor.models.HemodynamicSummary` — fact-only, frozen Pydantic
model. Fields: `map_nadir: int | None`, `map_nadir_at: datetime | None`,
`map_nadir_source: Literal[...] | None`, `vasopressors: tuple[VasopressorMention, ...]`.
`is_empty` property returns `True` when neither a MAP nor any vasopressor was
found. **Binding guardrail**: carries facts only — no "refractory" /
"instability" / appropriateness field. The deterministic classifier has no
hemodynamic gate; all weighting stays with the LLM and the auditor.

### Periop scan

`bba.vitals_extractor.periop.scan_periop(notes) → PeriopSummary`.
Recovers three fact-only signals from free-text notes for cases where structured
procedure rows are absent (Case 107 / REQNO 68074627 — ORIF + EBL 1500 mL
documented only in a nursing note, LLM wrote "no operative procedure documented"):

* **Surgical context** — any surgery cue (ผ่าตัด, post-op, ORIF/CRIF, TKA/THA,
  craniotomy, laparotomy, "to OR", "under GA/spinal", …). Boolean.
* **Estimated blood loss** — EBL volume normalised to millilitres (litres ×1000,
  "cc" as-is). MAX across the window kept (worst loss, not latest charted).
* **Intra-operative transfusion** — a specific blood component (LPRC, PRBC, FFP,
  platelets, SDP, cryo) co-located with an intra-op marker within 40 chars.
  Generic "blood" near "intra-op" deliberately NOT matched.

Pure aggregation; empty input yields an empty summary. Caller owns windowing.

### PeriopFinding

`bba.vitals_extractor.models.PeriopFinding` — frozen Pydantic model carrying one
located peri-operative fact: `category` (Literal `"surgery"` | `"blood_loss"` |
`"intraop_transfusion"`), a verbatim `snippet` (±60 chars around the match),
`at` (tz-aware UTC timestamp), `source` (origin table). At most one finding per
category per summary.

### PeriopSummary

`bba.vitals_extractor.models.PeriopSummary` — fact-only, frozen Pydantic model.
Fields: `surgical_context: bool`, `blood_loss_ml: int | None`,
`intraop_transfusion: bool`, `findings: tuple[PeriopFinding, ...]`.
`is_empty` returns `True` when no surgery, no EBL, and no intra-op transfusion
were found. **Binding guardrail**: carries facts only — the deterministic
classifier's procedure bypass keys on structured timing, not on this scan.
Do not add appropriateness fields.

## Cohort-detector concepts (#7)

### Cohort label

Seven-member `StrEnum` naming the deterministic cohort outcome:
`CARDIAC_SURGERY`, `ORTHO_CARDIAC`, `ESRD_EPO`, `MTP`,
`HEME_MALIGNANCY_ACTIVE`, `DEFAULT`, `UNKNOWN`.
`bba.cohort_detector.models.CohortLabel`. Snake-case `value`s
(`"cohort_unknown"`, `"esrd_epo"`, …) are the contract for downstream
classifier dispatch and dashboard joins; the enum names are private.

### Cohort assignment

Pure-function output: `(label, threshold, evidence_code, evidence_name)`.
`bba.cohort_detector.models.CohortAssignment`. Frozen Pydantic v2 model
consumed directly by `bba.deterministic_classifier` (#8); never
re-interpreted between assignment and decision.

### Cohort threshold (numeric)

The Hb decision threshold in g/dL, `float | None`. **Numeric, never
enum** — the classifier compares it against the audit-time Hb without
any string-to-number step in the hot path. `MTP`,
`HEME_MALIGNANCY_ACTIVE`, and `UNKNOWN` carry `threshold=None` because
they are not threshold-driven (auto-bypass, T2-supportive, NEEDS_REVIEW).
The single source of truth is `bba.cohort_detector.rules.COHORT_THRESHOLDS`;
adding a `CohortLabel` without a threshold-map entry is a test failure.

### Cohort precedence

Top wins: `UNKNOWN` → `MTP` → `ORTHO_CARDIAC` → `CARDIAC_SURGERY` →
`ESRD_EPO` → `HEME_MALIGNANCY_ACTIVE` → `DEFAULT`. UNKNOWN runs first
(before MTP) so a missing procedure join does not silently auto-bypass
to APPROPRIATE: a human must verify the apparent MTP context.
`bba.cohort_detector.detector.assign_cohort`.

### Procedure data unavailable

`CohortInputs.procedure_events` is `tuple | None`. **`None` means "the
IPTSUMOPRT join was skipped / data unavailable"** and produces
`UNKNOWN`. **`()` means "the patient had no operative events"** and
falls through to other cohort signals (clean negative). The distinction
is load-bearing — collapsing it would silently apply `DEFAULT 7.0` to
patients whose surgical history is unknown (PRD §5 + Round 2 N1 + user
constraint: NEVER silent 7.0).

### Cardiac-surgery cohort

Threshold 7.5 g/dL. Triggers when an operative event with `or_flag=True`
and an ICD-9-CM Vol 3 code matching `CARDIAC_SURGERY_CODE_PREFIXES`
(`36xx` PTCA / stents / CABG, `38xx` aortic resection, `39xx`
aorta-iliac-femoral bypass) occurred within 30 days of the audit anchor.
`CARDIAC_SURGERY_EXCLUDED_CODES = {"894", "3796"}` carves out cardiac
stress test and pacemaker pulse generator as defense-in-depth — non-OR
items must not trigger surgical recovery thresholds.

### Ortho-cardiac cohort

Threshold 8.0 g/dL. Requires both an orthopedic operative event
(`or_flag=True` + `ORTHO_SURGERY_CODE_PREFIXES` `78xx`/`79xx`/`81xx`
within 30 days) **AND** an ICD-10 cardiac-history diagnosis
(`I20`–`I25`, `I50`). Checked before plain `CARDIAC_SURGERY` because
its higher threshold dominates when both signals are present. Plain
ortho without cardiac history falls through to `DEFAULT` — ortho
alone is not its own cohort.

### ESRD-EPO cohort

Threshold 8.0 g/dL. Round 2 fix N1: requires **both** an ICD-10 ESRD
code (`N18.5` / `N18.6`) **AND** a dialysis-context medication active
within `DIALYSIS_LOOKBACK` (14 days). Either signal alone falls
through to `DEFAULT`. Dialysis keywords (`heparin`, `sevelamer`,
`cinacalcet`) are scoped by the co-required ESRD diagnosis, so a
non-ESRD patient on heparin for DVT prophylaxis is not at risk of
misclassification.

### MTP auto-bypass

`label=MTP`, `threshold=None`. Two arms over a closed-closed
`[anchor − 1 h, anchor]` window:

- **Cluster:** total `rbc_units` across all orders in the window
  ≥ `MTP_RBC_UNIT_THRESHOLD` (4).
- **Co-order:** FFP and platelets co-ordered evidence appears anywhere
  in the window (`any(co_ordered_with_ffp)` AND
  `any(co_ordered_with_platelets)`). Either flag pair on a single
  order or split across separate orders inside the window counts.

`bba.cohort_detector.rules.detect_mtp_pattern`. Auto-APPROPRIATE
bypass routing is the downstream effect; the detector itself just
labels the cohort.

### Heme-malignancy active cohort

`label=HEME_MALIGNANCY_ACTIVE`, `threshold=None` (T2-supportive, not
hard-threshold). Round 2 fix N3: requires an ICD-10 heme-malignancy
diagnosis (`C8x`/`C9x`), a chemo medication active within
`CHEMO_LOOKBACK` (30 days), **and** a measured ANC strictly less than
`ANC_NEUTROPENIA_THRESHOLD` (500 cells/µL). Missing ANC (`None`) is
**not** neutropenic — positive evidence is required, not absence.

### Allow-list seed (clinical sign-off pending)

Every constant in `bba.cohort_detector.rules`
(`CARDIAC_SURGERY_CODE_PREFIXES`, `ORTHO_SURGERY_CODE_PREFIXES`,
`CARDIAC_HISTORY_ICD10_PREFIXES`, `ESRD_ICD10_CODES`,
`HEME_MALIGNANCY_ICD10_PREFIXES`, `DIALYSIS_MED_KEYWORDS`,
`CHEMO_MED_KEYWORDS`) is a Phase-1 SEED awaiting clinical sign-off
before production. Tests in `TestAllowListSeeds` surface the seeds
verbatim so the clinical-review PR can diff them line by line.

### Medication recency window

`DIALYSIS_LOOKBACK = 14d`, `CHEMO_LOOKBACK = 30d`. The window check
lives inside `find_dialysis_med` and `find_chemo_med` so all callers
are scoped consistently — pre-filtering by the orchestrator would
risk silent inconsistency. A med outside the window does not count
as active even if it matches a keyword (a long-completed chemo
regimen does not perpetually flag heme-active).

### ICD-10 strict-case contract

The cohort matchers are **case-sensitive and whitespace-strict**,
mirroring `bba.audit_orders`. Lowercase `"i25.10"`, padded
`" I25.10 "`, and full-width digits do not match. Loosening the
matcher would also loosen drift detection at ingest (codes that
arrive in unexpected shapes are a data-quality problem to fix
upstream, not paper over downstream and silently broaden the
cohort allow-lists). `TestIcd10StrictCaseContract` pins the rule.

## Deterministic-classifier concepts (#8)

### Classifier inputs

Per-audit composition of upstream module outputs: `hb_result`
(`HbLookupResult` from #5), `cohort_assignment` (`CohortAssignment`
from #7), `order_datetime`, `procedure_proximity_hours` (hours from
most-recent operative event to the order anchor, `None` if no event),
and `crystalloid_liters_prior_4h` (4 h totaling from MED.csv).
`bba.deterministic_classifier.models.ClassifierInputs`. Frozen
Pydantic v2 — the classifier is pure, never mutates its input.

### Classifier result

Frozen output of `classify`: `(classification, bypass_reason,
cohort_threshold, rationale)`.
`bba.deterministic_classifier.models.ClassifierResult`.
`classification` is the canonical `bba.audit_store.Classification`
Literal (no module-local enum). `cohort_threshold` echoes the
threshold actually applied so the audit row is reproducible without
re-running the cohort detector. `rationale` is a short slug naming
which rule fired (`"hb_lt_7_universal"`, `"hb_lt_threshold"`,
`"bypass_hemodilution"`, `"single_low_hb_no_trend"`, …); free-form prose belongs to the LLM
stage, not here.

### Bypass reason

Six-member `StrEnum` identifying which deterministic bypass fired:
`DELTA_HB`, `PERI_PROCEDURAL_6H`, `PRE_OP_CROSSMATCH`, `MTP`,
`HEMODILUTION_FLAGGED`, `NONE`.
`bba.deterministic_classifier.models.BypassReason`. Structured
enum, not a free string — each path sets exactly one member; non-bypass
classifications carry `NONE`. The audit row persists this verbatim so
dashboards can group by reason without re-deriving from `rationale`.

### Classification precedence

Eleven-step composition in `bba.deterministic_classifier.classify`
(top wins): (1) Hb missing → MTP / peri-procedural / hard-peri-op-evidence
positive-evidence pre-pass (`APPROPRIATE` on an Hb-independent signal);
with the pre-pass flag on but no hard evidence → `NEEDS_REVIEW` (defer to
LLM), with the flag off → `INSUFFICIENT_EVIDENCE` (see "Missing-Hb
positive-evidence pre-check"); (2) Hb < 7.0 →
`APPROPRIATE`/`NONE`; (3) cohort `MTP` → `APPROPRIATE`/`MTP`;
(4) cohort `UNKNOWN` → `NEEDS_REVIEW`; (5) procedure ≤ 6 h →
`APPROPRIATE`/`PERI_PROCEDURAL_6H`; (6) upcoming procedure ≤ 72 h →
`APPROPRIATE`/`PRE_OP_CROSSMATCH`; (7) delta-Hb fired →
`APPROPRIATE`/`DELTA_HB`; (8) non-threshold cohort →
`NEEDS_REVIEW`; (9) hemodilution (Hb < threshold ∧ ≥ 2 L crystalloid)
→ `NEEDS_REVIEW`/`HEMODILUTION_FLAGGED`; (10)
single-low-Hb-no-trend (Hb < threshold ∧ `needs_review_single_low_hb`)
→ `NEEDS_REVIEW`/`NONE`; (11) plain Hb-tier: `< threshold` →
`APPROPRIATE`, `[threshold, 10)` → `NEEDS_REVIEW`, `≥ 10` →
`POTENTIALLY_INAPPROPRIATE`.

### INAPPROPRIATE never deterministic

The engine **never** emits `INAPPROPRIATE` — that label requires
positive-evidence reasoning only the LLM stage performs (Round 1 B2).
Documentation absence cannot produce `INAPPROPRIATE` at this layer;
the worst Hb-tier output is `POTENTIALLY_INAPPROPRIATE` and the LLM
may downgrade later. Pinned by `TestB2InvariantProperty`, a hypothesis
sweep over the full [2.0, 25.0] g/dL Hb range × every bypass-flag
combination — a future refactor that quietly adds an INAPPROPRIATE
path fails the property check.

### POTENTIALLY_INAPPROPRIATE tier (canonical)

`bba.audit_store.Classification` extended in #8 to include
`POTENTIALLY_INAPPROPRIATE`, so the deterministic engine returns the
Hb ≥ 10 tier directly without a string round-trip. Single canonical
Literal across `audit_store`, `deterministic_classifier`,
`prompt_builder`, and `llm_client`; no module-local variants
(user constraint: one enum, one source of truth).

### Cohort UNKNOWN routing

`label=UNKNOWN`, `threshold=None` → `NEEDS_REVIEW`/`NONE` when Hb is
≥ 7.0. Hb < 7.0 now stops earlier as `APPROPRIATE`. For the remaining
UNKNOWN cases, the classifier refuses to silently default to
`DEFAULT_THRESHOLD=7.0` when procedure data was missing upstream. Hb
missing still wins over UNKNOWN — no current Hb means no anchor for any
interpretation. `TestCohortUnknownRoutesToNeedsReview`.

### Non-threshold cohort fallthrough

After the global Hb < 7.0 rule and bypass chain, a `threshold=None`
cohort surviving the earlier steps (in practice, Hb ≥ 7.0
`HEME_MALIGNANCY_ACTIVE` — `MTP` and `UNKNOWN` exited earlier) routes
to `NEEDS_REVIEW`/`NONE` with
`rationale="cohort_non_threshold"`. The T2-supportive heme cohort
is not Hb-tier-driven; the LLM stage handles its context
interpretation for Hb ≥ 7.0. Hb < 7.0 is globally `APPROPRIATE`
before this cohort branch.

### Hemodilution carve-out

After the global Hb < 7.0 rule, inside the remaining sub-threshold branch
(Hb < `cohort_threshold`), if
`crystalloid_liters_prior_4h ≥ 2.0` (Round 1 B5), classification is
`NEEDS_REVIEW` with `bypass_reason=HEMODILUTION_FLAGGED` instead of
auto-APPROPRIATE. Scoped to the would-be auto-APPROPRIATE branch
only — never promotes a gray-zone or high-Hb result. Threshold lives
on the classifier module as `HEMODILUTION_CRYSTALLOID_LITERS = 2.0`.

### Single-low-Hb-no-trend carve-out

After the global Hb < 7.0 rule and the hemodilution check, if
`hb_result.needs_review_single_low_hb` is set (upstream contract:
isolated Hb < 8 g/dL with no prior 24 h observation), classification is
`NEEDS_REVIEW`/`NONE` with
`rationale="single_low_hb_no_trend"`. PR #52 Codex P1: a lone
unconfirmed low value cannot be treated as confirmed anemia. The
MTP / peri-procedural / delta-Hb bypasses still win above this check
because each is itself a positive-evidence anchor that supersedes
the "no trend" concern.

### Peri-procedural bypass window

`PERI_PROCEDURAL_WINDOW_HOURS = 6.0`. A procedure with
`procedure_proximity_hours ≤ 6.0` auto-bypasses to `APPROPRIATE`
with `bypass_reason=PERI_PROCEDURAL_6H`. Boundary is inclusive per
PRD §6 ("within 6 h"). Proximity is the elapsed hours from the
most-recent operative event to the order anchor — distinct from
"no event" (`None`); future-dated events are filtered upstream by
the orchestrator before reaching the classifier.

### Missing-Hb positive-evidence pre-check

**SEED pending clinical sign-off** (same status as the
`bba.cohort_detector` allow-lists). When `hb_result.value_g_dl is None`,
the classifier optionally runs a two-stage policy instead of dead-ending:
it auto-classifies `APPROPRIATE` on hard, Hb-independent positive evidence
exactly as the Hb-present path would, and **defers everything else to the
LLM** (`NEEDS_REVIEW`/`hb_missing_defer_llm`) rather than terminating. This
closes two gaps: the indefensible asymmetry where the *same* case parked as
`INSUFFICIENT_EVIDENCE` with no Hb but auto-classified `APPROPRIATE` with
one; and the dead-end where a deterministic `INSUFFICIENT_EVIDENCE` is
terminal (`DETERMINISTIC_FINAL`) so well-documented elective/peri-op cases
never reached the LLM (which auto-resolves the majority from the free-text
prose + the peri-op evidence block).

Window analysis (`/tmp/bba_mini`, 39 cases) confirmed the missingness is
**real, not a lookback-window artifact** — most cases only have a
*post-transfusion* Hb (inflated by the transfusion itself), so extending
the Hb window is the wrong lever (it would feed the gate a hindsight value).
The accuracy invariant therefore holds: the deterministic gate never
decides on a (possibly post-transfusion) Hb here — there is none; op-day/
post-op Hb may reach the LLM only as clearly-labeled *context*.

**The policy is gated behind a disabled-by-default flag**
(`ClassifierInputs.enable_missing_hb_positive_evidence`, forwarded from
`PipelineRowContext.enable_missing_hb_positive_evidence`). When the flag
is `False` (the default; production state until the QI committee signs off),
missing Hb always returns `INSUFFICIENT_EVIDENCE`/`NONE`/`hb_missing`,
regardless of cohort, procedure proximity, or peri-op evidence — matching
the original PRD spec line in `scripts/create_issues.sh` ("Hb missing →
INSUFFICIENT_EVIDENCE"). The orchestrator that builds `PipelineRowContext`
is the sign-off binding point.

When the flag is `True`, the pre-pass applies and order is preserved
(MTP → UNKNOWN → peri-procedural → hard peri-op evidence):

- cohort `MTP` → `APPROPRIATE`/`MTP`, `rationale="bypass_mtp_hb_missing"`;
- cohort `UNKNOWN` → no deterministic auto-approve (peri-procedural / peri-op
  evidence must not override UNKNOWN, mirroring the Hb-present order); defers
  to the LLM below;
- `procedure_proximity_hours ≤ 6.0` → `APPROPRIATE`/`PERI_PROCEDURAL_6H`,
  `rationale="bypass_peri_procedural_hb_missing"`;
- hard peri-op note evidence — a charted intra-op transfusion, or estimated
  blood loss ≥ `PERIOP_MIN_EBL_ML` (500 mL) — → `APPROPRIATE`/`PERIOP_EVIDENCE`,
  `rationale="bypass_periop_evidence_hb_missing"`. Soft cues (a surgery is
  merely documented / an upcoming procedure is scheduled) deliberately do
  NOT fire — those defer;
- otherwise → `NEEDS_REVIEW`/`NONE`, `rationale="hb_missing_defer_llm"` (route
  to the LLM, do not dead-end).

The peri-op signals (`periop_blood_loss_ml`, `periop_intraop_transfusion`,
`periop_surgical_context`) are threaded onto `ClassifierInputs` from
`bba.vitals_extractor.periop.scan_periop` (mirroring `PeriopSummary`);
`periop_surgical_context` is carried for traceability only and never gates a
verdict. `PERIOP_MIN_EBL_ML` lives on the classifier module as the single
source of truth and is re-exported as
`bba.audit_pipeline.replay.PERIOP_GUARDRAIL_MIN_EBL_ML` (the contradiction
guardrail's bar) so the two thresholds cannot drift.

The distinct `*_hb_missing` rationale slugs keep "approved with no
documented Hb" cases auditable for the QI committee (`bypass_reason` is
`MTP`/`PERI_PROCEDURAL_6H`/`PERIOP_EVIDENCE` for grouping);
`hb_missing_defer_llm` marks the LLM-deferred set.
`TestMissingHbPositiveEvidence` + `TestMissingHbPeriopEvidenceAndDeferral`
pin all flag-on paths; the `*_flag_default_off_*` /
`test_hard_periop_evidence_flag_off_stays_insufficient` tests pin the
disabled-by-default contract.
`TestMissingHbBypassPersistence` covers the end-to-end pipeline
persistence path for both flag states.

**Pilot enablement:** set
`BBA_PILOT_ENABLE_MISSING_HB_POSITIVE_EVIDENCE=1` for the pilot re-run
(`run_pipeline.py` deterministic leg + `run_llm_leg.py` LLM leg) once
clinical sign-off is on record. The flag stays code-default `False` for
multicenter rollout control.

### Delta-Hb / pre-op crossmatch excluded on missing Hb

The pre-pass above deliberately does **not** auto-approve on delta-Hb or
the pre-op crossmatch on a missing Hb. Delta-Hb needs a current Hb to
compute, so a stale upstream `delta_hb_bypass=True` set together with a
missing Hb is a structural inconsistency — the classifier prefers the "no
Hb" fact over the orphan flag (and, flag-off, the case stays
`INSUFFICIENT_EVIDENCE`). Pre-op crossmatch (`upcoming_procedure_hours`) is
a soft "surgery is scheduled" cue: transfusing pre-op with no documented Hb
is exactly what an audit should look at, so it is never deterministically
auto-approved. With the pre-pass flag on it no longer dead-ends, though —
it defers to the LLM (`NEEDS_REVIEW`/`hb_missing_defer_llm`).
`TestBypassPathways::test_delta_hb_bypass_does_not_fire_when_hb_missing`
and
`TestMissingHbPositiveEvidence::test_pre_op_crossmatch_only_with_missing_hb_defers_to_llm`
pin these exclusions.

### Monotonicity in Hb

For the same cohort + same evidence with no bypass active, increasing
Hb never moves classification toward more-appropriate. Ordering:
`APPROPRIATE` < `NEEDS_REVIEW` < `POTENTIALLY_INAPPROPRIATE`.
Hypothesis property test (`TestMonotonicityProperty`) sweeps thresholds
{7.0, 7.5, 8.0} and Hb pairs across tier boundaries. A regression
that, say, downgraded high-Hb to `NEEDS_REVIEW` based on a normal-but-
suspicious heuristic would fail the property check.

### Crystalloid totaling helper

`bba.deterministic_classifier.crystalloid.total_crystalloid_liters`.
Thin sum-and-window utility: filter events to
`(order_datetime − 4 h, order_datetime]`, parse the dose suffix
(`mL`/`cc`/`L`, case-insensitive), sum in liters. Future-dated events
and unparseable rows contribute 0.0. Drug-name → "is crystalloid?"
classification lives upstream in the MED-table reader; this helper
trusts that `med_events` are already crystalloid-only.

### Infusion-rate exclusion

The dose regex uses `\b(?!\s*/)` after the unit so
`"NSS 500 mL/h"`, `"D5W 200 cc/hour"`, `"RLS 1 L/hr"`, and
`"500 mL / h"` are all rejected. `\b` alone matched the L→/ boundary
(`/` is non-alphanumeric), which counted a rate as a delivered bolus
and could push a 4 h total over the 2 L hemodilution threshold —
flipping a sub-threshold Hb from APPROPRIATE to NEEDS_REVIEW
incorrectly. PR #52 Codex P2.

### Pure function contract

`classify(inputs) -> result` is pure: same input always yields the
same result, no module-global state, no mutation of `inputs`, never
raises on a well-formed `ClassifierInputs` (invalid types fail at
the Pydantic boundary, not inside `classify`). This is load-bearing
for the audit row's reproducibility metadata — re-running the
classifier with persisted inputs must reproduce the persisted
classification byte-for-byte.

### Deferred review (post-merge)

Items intentionally **not** in scope for #8, documented here so a
future reader does not re-litigate:

* **AN-join validation against real `IPTSUMOPRT.csv`** — the
  classifier composes upstream module outputs by reference. Real
  re-encrypted ANs arrive next week; smoke-testing the full join
  against `data/encrypted/IPTSUMOPRT.csv` lives with the orchestrator
  (#24, audit_pipeline), not here.
* **Crystalloid drug-name classification** — `total_crystalloid_liters`
  trusts that `med_events` are already filtered to crystalloid drugs
  upstream. The MED-table reader (an orchestrator concern) owns the
  drug-name allow-list; tightening it here would duplicate logic and
  invite drift.
* **`needs_review_single_low_hb` propagation to the LLM stage** —
  the deterministic engine routes to `NEEDS_REVIEW` and stops. The
  LLM stage receives the rationale slug `"single_low_hb_no_trend"`
  and may decide whether the note narrative provides the missing
  trend evidence. Wiring that branch into the prompt selector is a
  separate ticket (#21 prompt_builder follow-up).

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
Bounds are clamped around the point estimate (`lower = max(0, min(p_hat,
center - margin))`, `upper = min(1, max(p_hat, center + margin))`) to
correct floating-point noise at the boundaries — in exact arithmetic the
Wilson interval always contains `p_hat`, but at boundary prevalences
(successes == 0 or successes == n) rounding can produce a tiny epsilon
on the wrong side (e.g. `5.5e-17 > 0.0`), which would break downstream
`lower <= point <= upper` invariants.

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
`inter_hospital`, `hemoglobinopathy`, `pediatric`.
`ExcludedRecord.detail` carries the specific evidence (e.g.,
the ICD-10 code that triggered hemoglobinopathy) so reviewers can audit
the rule firing without re-joining source CSVs.

### Hard-exclusion ICD-10 set

Codes that block an order from the audit set regardless of other factors:
hemoglobinopathy `D55` / `D56` / `D57` / `D58`. Round 1 B1
(hemoglobinopathy hard-exclude); the issue #4 AC is the authoritative
list. `bba.audit_orders.rules` constant `HEMOGLOBINOPATHY_PREFIXES`.
AIHA (`D59.x`), TMA (`M31.1`), and obstetric (`O`-chapter) were dropped
from the hard-exclusion set on 2026-05-29 — those cohorts are now
in-scope and pass the filter as auditable orders.

### Boundary-aware ICD prefix match

`bba.audit_orders.rules._code_matches_prefix`. A raw `startswith` would
collapse the ICD-10 chapter boundary (e.g., `D550` would match `D55`).
The matcher requires either exact-length match or a `"."` continuation —
except for single-letter chapter prefixes (e.g., `"O"`), which accept
digit continuation to form the 3-char category (`O80`, `O09.9`). The
single-letter branch is retained generically but currently has no caller:
the only prefix set in use is the 3-char `HEMOGLOBINOPATHY_PREFIXES`.
Case-sensitive; ICD-10 is uppercase by convention and tolerating lowercase
would also tolerate other formatting drift.

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

## Dashboard concepts (#26)

### Reviewer dashboard

`bba.dashboard` — FastAPI + HTMX + Tailwind reviewer UI per PRD §17.
Composes `bba.audit_store` (read-only, via the daily DuckDB snapshot view
from #19) and `bba.review_actions` (Postgres-backed `phi_access_log`
writes from #25). Owns no persistence of its own; every piece of state
the dashboard surfaces is read from one of those two upstream stores.
Module structure: `app.py` (handlers + `create_app` FastAPI factory),
`models.py` (frozen Pydantic v2 DTOs + injectable-resolver type
aliases), `exceptions.py` (typed `DashboardError` hierarchy),
`templates/` (Jinja2: `base.html` + per-view templates +
`_queue_table.html` HTMX fragment).

### hb_freshness in queue and case-detail

`AuditRow.hb_freshness` is now surfaced on both `QueueItem` and `CaseDetail`
DTOs so reviewers can see the age-tier of the Hb used at decision time (one of
`fresh`, `stale_24_72h`, `stale_3_7d`, `missing`) alongside the numeric value in
the review queue and the case-detail view. Populated via `_audit_row_to_queue_item`
and `_audit_row_to_case_detail` in `bba.dashboard.app`.

### Five views

PRD §17 surface, exposed as five FastAPI routes:

* `GET /queue` — NEEDS_REVIEW triage queue, sortable by
  `order_datetime` | `hb_value` | `confidence` | `audit_id` via
  `sort_key`/`sort_direction` query params. HTMX form swaps the
  table in-place (`hx-target="#queue-table"`, `hx-swap="outerHTML"`).
* `GET /case/{audit_id}` — case-detail; un-redacted iff the audit is
  in the reviewer's `care_team_memberships`.
* `POST /case/{audit_id}/break-glass` — break-glass un-redacted
  capture; form-encoded `justification` (HTMX form).
* `GET /scorecard/ward/{ward_id}` — per-ward aggregations (counts +
  average confidence) over the snapshot.
* `GET /scorecard/physician/{physician_id}` — per-physician own-view
  scorecard, route-guarded.
* `GET /pipeline-health` — NEEDS_REVIEW rate, verifier pass rate,
  Sonnet→Opus escalation rate, average LLM-call latency.

### DashboardConfig

`bba.dashboard.DashboardConfig` — frozen pydantic v2 model bundling
every runtime dependency the dashboard composes: the
`AuditStore` + `ReviewActionsStore` handles, a `snapshot_dir`, a
`template_dir` override, an optional `default_reviewer` (dev / smoke
identity), and three injectable resolvers. `arbitrary_types_allowed`
is enabled because the two store handles are not pydantic models.

### Injectable resolvers

The dashboard does NOT own ward / physician / un-redacted-PHI lookups.
Three callable type aliases on `DashboardConfig`:

* `UnredactedPhiResolver = Callable[[str, str], tuple[str, str]]` —
  resolves `(hn_hash, an_hash) → (hn, an)`. Production wires the
  HIS / de-id-twin store; tests provide a fake.
* `WardAttributionResolver = Callable[[AuditRow], str]` — resolves
  an audit row to its ward id (the audit_store schema does not carry
  `ward_id` directly).
* `PhysicianAttributionResolver = Callable[[AuditRow], str]` — same
  shape for physician.

A `DashboardConfig` with `unredacted_phi_resolver=None` causes any
un-redacted code path to raise `UnredactedSourceUnavailableError`
(translated to HTTP 503). The dashboard refuses to fabricate
placeholder PHI — a deploy that forgets to wire the resolver fails
LOUD. Ward / physician resolvers fall through to sentinel ids
(`unattributed-ward`, `unattributed-physician`) which are visibly
broken in rendered output rather than silently misattributed.

### Care-team-of-record gating

`get_case_detail` returns un-redacted PHI iff
`audit_id ∈ context.reviewer.care_team_memberships`. The reviewer's
care-team list is set by the auth dependency at request boundary; it
is not derived from any audit-store column. Outsiders see the
redacted projection (`unredacted=False`, `raw_hn=raw_an=None`);
break-glass is the only override path.

### PHI access logging — every surfacing

PRD §17: every un-redacted access writes a `phi_access_log` row. The
dashboard treats CARE-TEAM access and BREAK-GLASS access as both
PHI-access events. Both paths route through
`ReviewActionsStore.access_phi(PhiAccessInput) -> Iterator[...]`
(from #25). The care-team path passes
`break_glass_justification=None`; the break-glass path passes the
reviewer's justification text. The log row is committed at
`__enter__`, before any un-redacted text is yielded — structurally
"log before reading", not a developer discipline. A redacted view
(outsider with no break-glass) writes no log row.

### Break-glass justification flow

`record_break_glass_access(config, context, audit_id, request)`:

1. Re-checks `request.justification` non-empty (defense in depth
   against a caller that bypasses `BreakGlassRequest`'s
   `NonEmptyStr` validator via `model_construct`).
2. Looks up the audit row (404 via `AuditNotFoundError` if missing).
3. Builds `PhiAccessInput` carrying the justification.
4. Enters `access_phi(...)` — log row committed here.
5. Returns the un-redacted `CaseDetail`.

If un-redacted resolution raises (no resolver, HIS unreachable), the
log row stays committed: PRD §17 treats an attempted access that
crashed mid-render as still an access for audit purposes.

### Strict own-view route guard

`get_physician_scorecard` denies whenever
`context.reviewer.physician_id != physician_id`, regardless of role.
A reviewer with `role='reviewer'` and `physician_id=None` cannot
enumerate any physician's scorecard through this route, nor can a
`senior_reviewer`. Cross-physician aggregation (admin / senior
reviewer fleet view) belongs on a SEPARATE explicitly-privileged
route — out of scope for #26. The guard was role-conditional in the
first GitHub PR review pass; the strict role-agnostic form is the
codex round 4 fix.

### Snapshot-frame consistency

The dashboard reads via `SnapshotView` (from #19) keyed to
`datetime.now(UTC).date()` — same-day reads share a materialized
snapshot, mid-batch writes are isolated. `get_pipeline_health`
extends this discipline to the latency aggregation: it filters
`audit_store.read_llm_calls()` to the snapshot's `audit_id` set
before averaging. Without the filter, an `LlmCall` whose audit row
has not yet committed to the snapshot (PRD §10 writes `llm_calls`
BEFORE `audit_results`) would skew `average_latency_ms` while not
contributing to `total_audits` — codex round 4 P2 fix.

### HX-Request fragment vs full-page

The `/queue` route inspects the `HX-Request` header. An HTMX request
gets `_queue_table.html` (a table-shaped fragment); a plain browser
navigation gets `queue.html` (full document with nav chrome,
Tailwind, HTMX script tags). Required because the page's HTMX form
swaps `#queue-table` with `outerHTML` — returning a full document
into a table-shaped slot is visually broken once the user
interacts with sort. Codex round 2 finding.

### get_route_context — overridable auth dependency

`bba.dashboard.app.get_route_context(request)` — a MODULE-LEVEL
FastAPI dependency that resolves the active `RouteContext`. Stored
at module level (not a closure inside `create_app`) so production
replaces it via the canonical FastAPI override mechanism:

```python
app = create_app(config)
app.dependency_overrides[get_route_context] = real_auth_dep
```

The default reads `request.app.state.dashboard_config` (set by
`create_app`) and returns `config.default_reviewer` if set, else
raises HTTP 401. The closure form was the codex round 4 P2 finding
— production could not reach a closure dependency for override.

### DashboardError hierarchy

Five typed exceptions, all subclasses of `DashboardError`, each
translated to an HTTP response at the route layer:

* `AuditNotFoundError` → 404 — `audit_id` not in today's snapshot.
* `PhysicianAccessDeniedError` → 403 — strict own-view guard fired.
* `CareTeamAccessDeniedError` → 403 — reserved for care-team gating
  failures (currently unused; the no-access path returns a redacted
  view rather than raising).
* `MissingJustificationError` → 400 — break-glass with empty / blank
  justification.
* `UnredactedSourceUnavailableError` → 503 — un-redacted PHI
  requested but no resolver is configured.
* `SnapshotInconsistencyError` — reserved for snapshot-read paths
  that observe rows off the materialization point (currently a
  type-existence contract; the snapshot model from #19 makes this
  unreachable in practice).

### Sentinel attribution ids

`unattributed-ward` / `unattributed-physician` — the fallback values
returned when no resolver is configured. Deliberately not
plausible-looking; codex round 1 flagged that earlier fallback names
(`default-ward`, `default-physician`) silently masked the
misconfiguration. The sentinel cannot collide with any real
`SafeId` from production (real ids never carry the literal substring
`unattributed-`).

### Deferred review (post-merge)

Codex review went through 3 in-session rounds and 1 GitHub PR review
round (a P1 + two P2 findings, all fixed before merge). Items left
deferred to follow-up work, not merge blockers:

* **Cross-physician scorecard route** — the strict own-view guard
  blocks admin / senior-reviewer visibility into other physicians'
  metrics through `/scorecard/physician/{physician_id}`. The
  intended path for that visibility is a separate, explicitly
  privileged endpoint (e.g., `/scorecard/admin/physicians`),
  out of scope for #26.
* **Real un-redacted PHI resolver wiring** — production wiring of
  `UnredactedPhiResolver` to the HIS / de-id-twin store depends on
  a follow-up ticket that exposes the lookup. The dashboard's
  failure mode (HTTP 503 + `UnredactedSourceUnavailableError`) is
  loud enough that the gap is unmistakable in operator testing.
* **Real ward / physician attribution wiring** — similar story for
  `WardAttributionResolver` / `PhysicianAttributionResolver`. The
  `unattributed-*` sentinels render in the dashboard until the
  registries are wired.
* **Tailwind / HTMX via CDN** — the templates load both via CDN
  (`cdn.tailwindcss.com`, `unpkg.com/htmx.org@1.9.10`). Production
  builds compile Tailwind to a static stylesheet and pin HTMX to a
  vendored bundle; the CDN form is acceptable for Phase 1 dev /
  smoke per the "thin" ticket label.
* **`/queue` no per-reviewer scope** — the queue is currently a
  global triage surface. `RouteContext` is passed through the
  handler signature for forward-compatibility (per-reviewer
  filtering, ward-scoped views) but unused today.
* **`SnapshotInconsistencyError`** — declared but not raised
  anywhere in #26. Reserved for a future snapshot-read invariant
  check; the existing `SnapshotView` design from #19 makes the
  failure mode structurally unreachable, so the exception lives on
  in the contract surface as a placeholder rather than a live
  signal.

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

## LLM-client concepts (#22)

### Allow-set-pinned model ID

The two model IDs the client may invoke are bare aliases
(`claude-sonnet-5`, `claude-opus-4-8`) — Claude Sonnet 5 and Opus 4.8
ship without dated snapshots, so the alias is the canonical ID. They
live behind `SONNET_MODEL_ID` and `OPUS_MODEL_ID` in
`bba.llm_client.models`; the runtime allowlist is
`ALLOWED_MODELS = frozenset({SONNET_MODEL_ID, OPUS_MODEL_ID})`. Any
model not in that set (e.g. `claude-sonnet-4-6`) is rejected at
`LlmClientConfig` construction. PRD §13: the allowlist forces a code
change (edit the constant) for every model *swap*, so the golden-set
drift probe has a stable anchor — but because the IDs are bare aliases,
an Anthropic point release under the same alias is no longer caught
here.

### Custom-ID assertion

The Anthropic Batch API returns one result per submitted `custom_id`,
not in submission order. `bba.llm_client.custom_id.assert_custom_ids_match`
is the only place the contract `custom_id == audit_id` is checked.
Mismatches (missing, extra, duplicate) raise `CustomIdMismatchError`
naming the offending IDs and abort the batch before any per-row work.
The function returns a `{audit_id: BatchSubmissionResult}` mapping so
downstream code never positional-zips — a positional zip would silently
swap audit rows under partial failure.

### Fail-closed parser

`bba.llm_client.parser.parse_structured_response` accepts any
`BatchSubmissionResult` and returns a `ParseOutcome`. It NEVER raises.
Every failure mode lands as `parse_failure=True` with a structured
`ParseFailureReason`:

* `MALFORMED_JSON` — tool-use `input` arrived as a string that fails
  `json.loads`.
* `SCHEMA_MISMATCH` — JSON parses but is missing required keys or has
  wrong types per `LlmClassificationResponse`.
* `CLASSIFICATION_OUT_OF_SET` — the four-label set is the only
  permitted classification; anything else is a fail-closed exit.
* `EMPTY_RESPONSE` — no `content` array, or empty array. Includes the
  Anthropic Batch API's `errored`/`canceled`/`expired` row types, which
  the transport synthesizes into a content-empty envelope.
* `TOOL_USE_MISSING` — content exists but contains no `tool_use` block.

A raise inside the parser would lose the audit chain — the caller's
persistence layer would never see the failure, the row would be
quarantined without a record, and the PRD §"reproducibility = we
have the original answer" invariant would break.

### Parse outcome

Frozen result of one parser invocation. The fail-closed contract is
enforced by a model validator: `parsed is None` iff `parse_failure is
True` iff `parse_failure_reason is not None`. The complementary state
(parsed set + no failure reason) is also enforced. Callers cannot
forge an inconsistent `ParseOutcome`.

### Retry → escalation policy

`bba.llm_client.escalation.run_with_escalation` runs one audit row
through up to `MAX_SONNET_ATTEMPTS = 2` Sonnet calls, then escalates
to Opus 4.7 iff every Sonnet attempt produced `parse_failure=True`.
PRD §13 cap of two Sonnet retries is structural: `LlmClientConfig`'s
`max_sonnet_attempts` field is bounded `ge=2, le=MAX_SONNET_ATTEMPTS`.
The policy is pure-control-flow over `ParseOutcome`; the HTTP
boundary is the injected `AnthropicTransport`. `should_escalate(outcomes,
config)` is the introspectable predicate — used by tests and the
orchestrator alike.

### Escalation log

`EscalationLog` is the per-row record persisted on
`LlmClientResult.escalation`. Fields: `sonnet_attempts` (0–2),
`sonnet_parse_failures` (tuple of reasons, one per attempt),
`escalated_to_opus` (bool), `opus_parse_failure` (reason or None).
A model validator rejects inconsistent state: `escalated_to_opus=True`
requires the full Sonnet budget exhausted; `opus_parse_failure` is
None unless Opus actually ran.

### Disagreement verdict

`DisagreementVerdict` records the outcome of comparing a Sonnet and
an Opus classification. Surfaces only when both sides produced a
parseable response. Fields: `sonnet_classification`,
`opus_classification`, `agreed`, `routed_to_needs_review`. A model
validator rejects routing/agreement state desync — `agreed=True
xor routed_to_needs_review=True` whenever both sides exist. When one
side is None (the typical escalation case: Sonnet failed parse twice,
Opus succeeded) the verdict carries `agreed=False, routed_to_needs_review=False`
and the orchestrator records the Opus answer without a NEEDS_REVIEW
gate.

### Cross-check Opus

Opt-in flag `LlmClientConfig.cross_check_with_opus = False` (default
off). When True, every successful Sonnet attempt is shadow-followed
by an Opus call; the two classifications are compared via
`detect_disagreement`. Off by default because routine Opus shadow is
a ~5x cost regression; PRD §13 reserves Opus for failure-driven
escalation. Tests `TestDisagreementDetection::test_cross_check_*`
pin both the agreement (record Sonnet, persist both calls) and
disagreement (NEEDS_REVIEW with `review_reason="disagreement"`) paths.

### Opus cross-check parse failure

When `cross_check_with_opus=True` is enabled and Opus's shadow call
returns malformed output, the quality gate has degraded — the row
cannot be cross-checked. Distinct review reason:
`review_reason="opus_cross_check_parse_failure"`. Both calls (Sonnet
+ Opus) are still persisted; the row routes to NEEDS_REVIEW so a
human reviewer knows this is NOT a model disagreement but a degraded
gate.

### Anthropic transport (Protocol)

`AnthropicTransport` is a `typing.Protocol` exposing one method:
`submit_batch(model, requests, prompt_cache_enabled) -> RawBatchResponse`.
The whole pipeline speaks to the Protocol, never to a concrete client.
Two implementations: `AnthropicBatchTransport` (production, lazy-
imports the `anthropic` SDK) and `CassetteTransport` (test-time,
loads a recorded JSON cassette). The Protocol is the seam that lets
unit tests run offline.

### Cassette transport

`bba.llm_client.cassette.CassetteTransport` is a Betamax/VCR-style
recorder/replayer over a JSON file. The cassette format is a single
`{"interactions": [...]}` array; each interaction is keyed on
`(model, sorted_tuple(custom_ids))`. A cassette miss raises `KeyError`
loudly — silently substituting a default response would let a
regression in submission shape slip past CI. `load_cassette(path)`
returns `tuple[CassetteInteraction, ...]`.

### Production Anthropic batch transport

`bba.llm_client.transport.AnthropicBatchTransport` is the SDK-backed
production implementation. Lazy-imports `anthropic` from inside
`submit_batch` so importing `bba.llm_client` in offline / CI contexts
does not require the SDK extra. Constructor validates
`ANTHROPIC_API_KEY` at instantiation (empty string treated as
missing) so a misconfigured deployment fails loud at startup, not
deep inside the SDK on first call. `MAX_OUTPUT_TOKENS = 4096`
constant sizes the structured-output budget.

### Anthropic request builder

`bba.llm_client.transport.build_anthropic_request(request, *, model,
prompt_cache_enabled)` translates one abstract `BatchSubmissionRequest`
into the Anthropic Messages-API payload. The translation:

* Splits `PromptBlock`s by `role` into the `system` array and the
  user `messages[0].content` array.
* On every `PromptBlock` with `cache_marker=True`, sets
  `cache_control={"type": "ephemeral"}` on the corresponding payload
  element — the Anthropic prompt-cache breakpoint marker.
* Forces `tool_choice` on the `classify_transfusion_order` tool with
  an `input_schema` that restricts `classification` to the four-label
  enum (defense in depth alongside the post-hoc parser check).

### Batch result type discriminator

Anthropic Batch result entries carry a `type` discriminator —
`succeeded`, `errored`, `canceled`, or `expired`. Only `succeeded`
entries expose `result.message`; the other three expose `result.error`
(or nothing). `_result_from_batch_entry` branches on the type and
synthesizes a content-empty envelope (`raw_response_json["content"]
= []`, `stop_reason = "batch_error"`) for non-succeeded entries with
the error detail preserved under `_batch_error`. The parser then
routes the row to `EMPTY_RESPONSE` → NEEDS_REVIEW. PRD §"Anthropic
API outage mid-batch": chain of custody must hold; an `AttributeError`
on `entry.result.message` would silently lose the row.

### Deterministic call_id

`LlmCall.call_id` is derived from STABLE inputs only:
`call-<audit_id>-<attempt_index>-<sha256(run_id|audit_id|model_id|attempt_index|sha256(canonical_request_json))[:16]>`.
Notably excludes `request_timestamp`. A re-run with identical inputs
produces an identical `call_id` so the `audit_store` append-only
contract holds — re-running with the same `(audit_id, run_id,
code_version)` is a no-op, not a duplicate append.

### Response-headers persistence

`bba.audit_store.LlmCall` has no dedicated `response_headers` column,
but PRD §"reproducibility" requires the `anthropic-version` header to
survive persistence. `client._call_from_result` folds the headers
into `LlmCall.response_json` under the key `__bba_response_headers__`
(double-underscore namespaced). The writer raises
`BatchSubmissionError` if Anthropic ever ships that exact top-level
key, so a future SDK change cannot silently overwrite headers with
vendor data.

### Frozen JSON payloads (cross-module)

`BatchSubmissionResult.raw_response_json`, `request_json`,
`response_headers`, and `extended_thinking_blocks` use the
`FrozenJsonDict` / `FrozenJsonList` types re-exported from
`bba.audit_store.models`. Every nested dict/list is wrapped in
`MappingProxyType` / `tuple` so a downstream caller cannot patch the
response between parse and persist. Mutation raises `TypeError` at
the boundary.

### LLM client result

`LlmClientResult` is the top-level per-row output of `process_batch`.
Model-validator invariants:

* `parse_failure=True` forces `response is None`,
  `final_classification == "NEEDS_REVIEW"`, and `needs_review=True`.
* `needs_review=True` requires a non-None `review_reason`.
* Every `LlmCall` in `persisted_calls` must share the result's
  `(audit_id, run_id)` — same constraint the `audit_store.write`
  contract enforces, surfaced earlier so the bug is loud.

### Deferred review (post-merge)

Codex review ran for 4 in-session rounds before PASS. Items
intentionally **not** in scope so a future reader does not
re-litigate:

* **Transport-level retries on rate-limit / 5xx** — the v1 cut covers
  parse-failure retries via the Sonnet→Opus escalation policy, not
  network transients. `AnthropicBatchTransport.submit_batch` raises
  `AnthropicAPIError` on any non-recoverable failure; rate-limit
  backoff is a follow-up ticket.
* **Full SDK-backed integration test** — the unit suite uses
  `CassetteTransport` and a monkeypatched fake `anthropic` module
  (`_FakeBatchEntry`, `_build_fake_anthropic_sdk`). The live SDK
  round-trip lands in `bba.audit_pipeline` (#24) per PRD §"Testing
  Decisions" ("`bba.llm_client` — smoke tests + golden-path E2E").
* **Prompt-cache marker recording in cassettes** — cassettes are
  keyed on `(model, sorted(custom_ids))` only, not on
  `prompt_cache_enabled`. The cache flag is validated at the
  `build_anthropic_request` boundary
  (`TestAnthropicRequestCacheControlTranslation`), not at the
  cassette replay layer.

## Confidence-calibrator concepts (#23)

### Calibrated confidence

A probability in `[0.0, 1.0]` produced by feeding the raw LLM-reported
confidence through the fitted isotonic curve. This is the value the
audit pipeline compares against `REVIEW_CONFIDENCE_THRESHOLD = 0.7`
to gate `NEEDS_REVIEW`. PRD §14 / user-story #40: the deployed
threshold must reflect empirical P(correct), not the model's
self-reported number — every calibration step exists to make that
substitution defensible.

### Raw LLM confidence

The number Sonnet/Opus emits inside its structured-output tool call
(`LlmClassificationResponse.confidence`) before any calibration. The
calibrator's training set is `(raw_confidence, gold_label)` pairs
drawn from the held-out evaluation split; predicting on a new raw
confidence yields a calibrated probability.

### Isotonic calibrator

`bba.confidence_calibrator.IsotonicCalibrator` — stateful wrapper
around an `IsotonicFit`. `fit(scores, labels)` runs PAV and stores
the curve; `predict(scores)` interpolates and clips. The instance
starts unfitted; calling `predict` or accessing `fit_result` before
`fit` raises `CalibratorNotFittedError` so a missing calibration
step surfaces immediately instead of silently returning the identity
mapping or zero.

### Refit overwrites

Successive `fit` calls overwrite the prior curve — monthly
recalibration semantics. PRD §14 "Monthly recalibration check via
ECE on held-out 200": when ECE on the holdout drifts past the
operator's tolerance, the recalibration job calls `fit` again on
the freshest training-split outputs and the new curve replaces
the old one.

### Pool-adjacent violators (PAV)

The standard algorithm behind `pav_fit`. Sort by score, treat each
point as a unit-weight block with `mean_y = label`, scan left-to-
right; whenever a block's `mean_y` exceeds its right neighbor's,
merge the two by weighted average and re-check leftward. The
single-pass stack implementation in `isotonic.py` is O(n log n)
including the sort; duplicate `scores` are merged by weighted
average before PAV runs, matching `sklearn.isotonic.IsotonicRegression`.

### Isotonic fit

Frozen dataclass `IsotonicFit` storing the materialized curve as
`x_thresholds` (strictly increasing) and `y_values` (non-decreasing,
each in `[0.0, 1.0]`), plus `n_training` for monthly recalibration
audit. Each PAV block contributes two threshold points
`(left_x, mean_y)`, `(right_x, mean_y)` so the piecewise-constant
shape is preserved; linear interpolation between blocks happens at
`predict` time, not at fit time.

### Boundary clipping

A `predict` input below `min(x_thresholds)` clips to `y_values[0]`,
above `max(x_thresholds)` clips to `y_values[-1]`. Isotonic regression
is undefined outside the fitted range; extrapolating could produce
calibrated probabilities outside `[0.0, 1.0]` and silently break the
0.7 gate. Clip is the explicit choice over extrapolation or raising.

### Sklearn reference pinning

The audit container has no scikit-learn runtime dependency (mirrors
`bba.eval_harness.intervals` which references scipy without
importing it). `tests/unit/test_confidence_calibrator.py` pins the
`SKLEARN_REF_SCORES` / `SKLEARN_REF_LABELS` / `SKLEARN_REF_PRED`
vectors computed offline against
`sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')`. A drift
in `pav_fit` semantics breaks the reference test, not a runtime
import.

### Expected Calibration Error (ECE)

Guo, Pleiss, Sun, Weinberger (2017) eq. (3): the count-weighted sum
of `|mean_confidence_in_bin - accuracy_in_bin|` over equal-width
bins. `bba.confidence_calibrator.compute_ece` returns an `EceResult`
with the scalar plus a per-bin `BinStats` tuple so the reliability
diagram can render every interval. The ECE_REF in the test suite
is a hand-derivable 4-sample / 2-bin example yielding `ECE = 0.275`.

### Closed-right last bin

Bin edges are `[i/n_bins, (i+1)/n_bins)` half-open except the final
bin, which closes on the right so a prediction at exactly `1.0`
lands in the last bin rather than overflowing. Implemented as
`min(int(p * n_bins), n_bins - 1)` — concise enough to be the only
place the convention lives, audited by
`test_probability_at_one_lands_in_last_bin`.

### Empty bin contributes zero weight (ECE)

A bin with `count == 0` carries `mean_confidence = 0.0` and
`accuracy = 0.0` in the returned `BinStats` and contributes zero
weight to the weighted ECE sum. Matches the Guo et al. convention
and is asserted by `test_empty_bins_contribute_zero_weight` — the
alternative (treating empty bins as a 0-vs-0 gap of zero, or
skipping them entirely) would produce the same scalar ECE but a
different `BinStats` shape, which the reliability renderer relies on.

### Reliability diagram (SVG)

`generate_reliability_diagram(probs, labels, out_path, n_bins,
title)` writes an SVG calibration plot to `out_path` (default
target: `docs/eval/`). SVG rather than PNG so the audit container
has no `matplotlib` runtime dependency (on-prem KCMH deployment
runs with `TRANSFORMERS_OFFLINE=1`); SVG renders natively in
browsers, GitHub markdown, and the reviewer dashboard.

### Empty bin skipped (renderer)

The reliability renderer emits one `<circle class="reliability-bin"/>`
per **non-empty** bin only. An empty bin's `accuracy` is by
construction `0.0` (see *empty bin contributes zero weight*); emitting
a marker for it would falsely paint a zero-accuracy point on the
diagram and mislead the transfusion committee's drift review. Note
the subtle asymmetry with ECE: the empty bin still appears in
`EceResult.bins` so the operator can introspect the empty interval,
but it is invisible on the rendered plot.

### XML-escaped title

The caller-supplied `title` parameter is passed through
`xml.sax.saxutils.escape` before interpolation into the SVG
`<title>` element. A title containing `&` / `<` / `>` would
otherwise either malform the XML or inject markup. Numeric fields
(ECE value, coordinates) are formatted internally and cannot
contain unsafe characters; escaping there would be noise.

### Validate-before-IO

`generate_reliability_diagram` delegates input validation to
`compute_ece`, which raises `InvalidCalibrationDataError` before any
`Path.mkdir` or `write_text` runs. PRD §"reproducibility" anti-
pattern: a half-written or stale diagram on disk after a failed
recalibration job would mislead the operator who sees a fresh
mtime. Asserted by `test_invalid_inputs_raise_before_writing`.

### Agreement-based confidence

PRD §14 / user-story #40 alternative to isotonic regression: run
Sonnet three times with reshuffled few-shot ordering; confidence =
fraction of runs that agree on the majority classification.
`bba.confidence_calibrator.agreement_confidence` is the pure-Python
piece that consumes the three classification strings and returns an
`AgreementResult`. The actual prompt reshuffle and LLM dispatch are
out of scope (they live in `bba.prompt_builder` / `bba.llm_client`);
this module handles seed generation and vote tabulation only.

### Shuffle seeds (deterministic)

`shuffle_seeds(base_seed, n_runs=3)` returns a tuple of `n_runs`
integer seeds derived from `int.from_bytes(sha256(f"{base_seed}:
{i}").digest()[:4], "big")`. Same `base_seed` + same `n_runs`
yields the same tuple, every time, forever — the cornerstone of
audit-row reproducibility for the agreement path. SHA-256 mixing
(not `base_seed + i`) so a small change in `base_seed` does not
produce three near-correlated shufflings. Negative `base_seed` or
`n_runs < 1` raises `InvalidCalibrationDataError`.

### First-seen tie-breaking

When `agreement_confidence` receives a 1-1-1 three-way split, the
classification appearing first in the input wins. Audited by
`test_three_way_tie_resolves_to_first_seen`. The choice is
deterministic so a tie does not race the dict-ordering or sort-
stability internals; PRD §14 "deterministic seed control" is a
contract that extends to tie-breaking, not just to the seed
generator itself.

### REVIEW_CONFIDENCE_THRESHOLD

`REVIEW_CONFIDENCE_THRESHOLD = 0.7` — the deployed gate from PRD §14.
A calibrated confidence below this routes the audit row to
`NEEDS_REVIEW`. The constant lives in
`bba.confidence_calibrator.models` so the downstream pipeline reads
the value from one place; the calibrator itself does not apply the
threshold (it produces the probability, the audit-pipeline gate
applies the comparison).

### ECE_RECAL_HOLDOUT_SIZE

`ECE_RECAL_HOLDOUT_SIZE = 200` — the row count of the held-out
sample the monthly recalibration job evaluates ECE on. The
calibrator does not run the recalibration job itself; the constant
is exposed so the scheduling module (`bba.monitoring`, #29) and
this module agree on the sample size.

### Calibrator-not-fitted fail-loud

Calling `predict` or accessing `fit_result` on an unfitted
`IsotonicCalibrator` raises `CalibratorNotFittedError` (descended
from `ConfidenceCalibratorError`). The silent alternatives
(returning the identity, returning all-zero, raising `AttributeError`
deep inside numpy) would all mask a deployment that skipped
calibration entirely. PRD §"reproducibility = we have the original
answer": the audit row's calibrated confidence must trace to a
specific fit invocation.

### Deferred review (post-merge)

Codex review ran for 2 in-session rounds (NEEDS-CHANGES → fixes →
PASS). Items intentionally **not** in scope so a future reader does
not re-litigate:

* **`docs/eval/` path enforcement on the renderer** — codex flagged
  the renderer for not enforcing the output directory. Not adopted:
  the renderer is a generic SVG writer; path choice is the caller's,
  matching every other "write artifact" function in the repo
  (`bba.report_generator.generator`, `bba.report_generator.csv_writer`).
  Adding a path guard would couple the renderer to a directory
  convention beyond the ticket's plain reading.
* **Live SDK integration for the agreement-based path** — this
  module handles seed generation + vote tabulation only. Wiring
  the three reshuffled Sonnet calls into the audit pipeline lives
  in `bba.audit_pipeline` (#24), not here.
* **Monthly recalibration scheduler** — `ECE_RECAL_HOLDOUT_SIZE`
  is the contract anchor, not the scheduler. The "compute ECE on
  200 → refit if drifted" job lives in `bba.monitoring` (#29).
* **Confidence-interval / Platt-scaling alternative** — PRD §14
  mentions "Platt scaling" as an alternative to isotonic; the v1
  cut implements isotonic only because the held-out 200 is large
  enough to fit a non-parametric curve without overfitting. Platt
  is a follow-up ticket if the empirical reliability diagram
  shows over-fitting.

## Audit-pipeline concepts (#24)

### Pipeline row context

`bba.audit_pipeline.PipelineRowContext` — frozen pydantic model bundling
the upstream-derived inputs the orchestrator needs to audit one RBC order:
:class:`AuditOrder`, :class:`HbLookupResult`, :class:`VitalsResult`,
:class:`CohortAssignment`, procedure proximity, crystalloid totals,
de-identified `hn_hash` / `an_hash`, prior-RBC counts, redacted evidence
chunks, and reproducibility metadata (`redactor_version`,
`redactor_model_sha`, `policy_version`, `prompt_hash`,
`evidence_bundle_hash`). The pipeline NEVER fabricates any of these —
the caller assembles a context from upstream module outputs and the
orchestrator composes deterministic_classifier / prompt_builder /
llm_client / audit_store on top.

### Batch run

`bba.audit_pipeline.BatchRun` — one row in the Postgres `batch_runs`
table. Identifies one Anthropic Batch API submission (multiple
`audit_id`s per row) and the five-state machine the pipeline transitions
it through. `batch_id` is locally generated and stable across re-runs;
`anthropic_batch_id` is set only at the `SUBMITTED` transition so a
crash between local-create and Anthropic-submit is recoverable.

### Batch run state

`bba.audit_pipeline.BatchRunState` — the five-state machine: `PENDING →
SUBMITTED → PARTIAL → COMPLETE` with `FAILED` as a sink. `COMPLETE` and
`FAILED` are terminal. Transitions go through
`bba.audit_pipeline.state_machine.transition`, which validates against
`VALID_TRANSITIONS` and rejects illegal moves with
`BatchStateTransitionError`. `PENDING → SUBMITTED` requires the caller
to supply `anthropic_batch_id`; any transition to `FAILED` requires
`error_message`.

### Split-phase submission

The Anthropic transport boundary exposes `submit_batch_only(...)` (creates
the remote batch, returns the `batch_id` immediately) and
`fetch_batch_results(batch_id, ...)` (polls until completion) as separate
methods so the orchestrator can persist `SUBMITTED + anthropic_batch_id`
to `batch_runs` BEFORE waiting for results. A SIGTERM during the polling
window now leaves a recoverable `SUBMITTED` row whose `batch_id` the
resume reconciler can pick up — without the split, an atomic
submit-then-poll call could orphan a created-but-never-recorded
Anthropic batch. `submit_batch(...)` remains as a backward-compatible
convenience wrapper.

### Row-level checkpoint

The contract that every audit pipeline crash leaves the system in a
state the resume reconciler can interpret. `bba.audit_pipeline.run_pipeline`
writes the `batch_runs` row at four distinct moments: `PENDING` (before
Anthropic submit), `SUBMITTED` (immediately after `submit_batch_only`
returns), then `COMPLETE` or `FAILED` once the response is persisted /
the row exhausts retries. PRD §15 calls this out explicitly so the
operator never has to manually re-derive batch state from logs.

### Resume reconciler

`bba.audit_pipeline.resume_on_startup` — the load-bearing primitive for
SIGTERM safety. On boot it walks every non-terminal `batch_runs` row
and handles three failure-window classes:

1. **PENDING with no `anthropic_batch_id`** — local-create succeeded,
   Anthropic was never asked. Transitioned to `FAILED` with an
   operator-visible `error_message`.
2. **SUBMITTED / PARTIAL with cached llm_calls** — orphan-call case
   (phase 1 of `audit_store.write` landed, phase 2 didn't). Each orphan
   is re-emitted through `apply_batch_results` using the cached
   `LlmCall.response_json` payload (PRD §10 makes that byte-exact).
3. **SUBMITTED / PARTIAL with NO cached llm_calls** — Anthropic-polling
   crash. When `transport` + `llm_config` are supplied, the reconciler
   calls `fetch_batch_results(anthropic_batch_id, ...)` to retrieve the
   in-flight response, applies it via `apply_batch_results`, and
   transitions the row to `COMPLETE`.

The reconciler is itself idempotent — a second pass produces zero new
writes (audit_store's `WriteResult.skipped_idempotent` rejects the
duplicate).

### Winning attempt rule

`bba.audit_pipeline.select_winning_attempt` — given a sequence of
verified attempts for one `audit_id`, returns the one with the latest
`attempt_id` whose `verifier_pass=True`. Returns `None` when no attempt
passes verifier; the caller then routes the row to `NEEDS_REVIEW` with
`review_reason="hallucination_suspect"`. Wired through
`apply_batch_results` so the rule applies to retry / Sonnet→Opus
escalation chains without the orchestrator re-implementing it.

### Verifier callable

`bba.audit_pipeline.replay.Verifier` — the `(BatchSubmissionResult,
PipelineRowContext) -> bool` callable the orchestrator passes to
`apply_batch_results` to decide whether each attempt grounds. Phase-1
default (`default_verifier`) returns `True`; production wires
`bba.quote_grounder.verify_citations` once that integration lands. Tests
inject stubs that always reject to exercise the hallucination-suspect
branch.

### Apply batch results

`bba.audit_pipeline.apply_batch_results` — translates a
`RawBatchResponse` into committed `AuditRow` + `LlmCall` pairs via
`audit_store.write`. Idempotent on `(audit_id, run_id, code_version)`;
applying the same response twice writes zero new rows. Required kwargs
include a `Mapping[audit_id, PipelineRowContext]` so every persisted
row's clinical and reproducibility fields come from caller-supplied
data — no fabrication. Missing context for any `custom_id` raises
`KeyError` rather than silently filling defaults.

### Parse failure reason

A typed slug persisted on `AuditRow.review_reason` when the LLM's
structured-output payload drifts from the schema:
`empty_response` (no content), `tool_use_missing` (first block isn't
`tool_use`), `schema_mismatch` (input isn't a mapping or
classification isn't a string), `classification_out_of_set` (value
outside the Classification Literal). Operators can distinguish
clinical `NEEDS_REVIEW` from API drift without re-reading the raw
Anthropic payload.

### Cost guard

`bba.audit_pipeline.assert_test_safe_transport` — refuses the live
`AnthropicBatchTransport` when called from a test context, raising
`LiveAnthropicApiError`. The class-identity check is on
`isinstance(transport, AnthropicBatchTransport)`, so a wrapper that
inherits from the live class is still considered live. Tests inject
`CassetteTransport`; production callers skip the guard.

### In-memory batch-run store

`bba.audit_pipeline.InMemoryBatchRunStore` — dict-backed test
implementation of the `BatchRunStore` Protocol. Single-process, not
thread-safe. Used by every unit test and the property test.

### Postgres batch-run store

`bba.audit_pipeline.PostgresBatchRunStore` — production
`BatchRunStore` backed by psycopg v3 + a connection pool. Lazy-opens
so tests can construct against an unmigrated DB and verify the
migration / connection failure modes. Schema installed by alembic
migration `a1c2e3f4b5d6_audit_pipeline_batch_runs.py`; DB-level CHECK
constraints mirror the pydantic invariants so a raw-SQL caller that
bypasses the model layer still cannot persist a contradictory row.
A partial UNIQUE index on `(run_id, anthropic_batch_id) WHERE NOT NULL`
prevents two batches from sharing an Anthropic batch_id within one run.

### Deterministic-final routing

Contexts whose `bba.deterministic_classifier.classify` result is
`APPROPRIATE`, `INSUFFICIENT_EVIDENCE`, or `INAPPROPRIATE` skip the LLM
entirely — the orchestrator writes the audit row directly with a
`model_id="deterministic"` marker LLM call (the audit_store's
transactional-ordering invariant rejects rows without a paired
`LlmCall`). `POTENTIALLY_INAPPROPRIATE` / `NEEDS_REVIEW` route through
the LLM stage where the cassette / production transport supplies the
response.

### Qualified-bleeding exemption & stale-volume gate

`bba.audit_pipeline.bleeding` is the ONLY place the RBC over-clear
guardrail trusts the LLM's free-text prose: when the model clears a
withheld gray-zone verdict with no structured hard signal, the assert
to `INAPPROPRIATE` is skipped only for a grounded `ACTIVE_BLEEDING`
indication with its own confidence ≥ 0.8 AND either a quoted volume
strictly > 300 mL or an explicit life-threatening / uncontrolled
marker (spec #89 locked decision 2).

**Stale-volume temporal gate** (owner ruling, case 68080335): a
quantified bleed charted for a dated earlier event must not clear the
current order — the motivating case cited a 400 mL index bleed dated
`Hx.1/12/68` for an order placed 22 days later while the ongoing
bleeding was unquantified. When the caller passes the order's
Asia/Bangkok calendar date, quote spans governed by a `d/m/y` anchor
(two-digit Thai BE year, four-digit BE 2400–2700, or CE 1900–2100)
more than `LLM_OVERCLEAR_MAX_BLEED_AGE_DAYS` (7, strictly older)
before the order are blanked ahead of the volume / marker scans. An
anchor governs text up to the next anchor or newline (the
line-oriented `date: content` style of Thai focus notes); undated
text, unparseable tokens, and future dates govern nothing — the gate
can only WITHHOLD the exemption, never widen the auto-clear surface.
The RBC system prompt states the same CURRENT-episode requirement
(both golden prompt hashes re-pinned), so model verdict and guardrail
agree instead of the guardrail silently overriding an
appropriate-sounding rationale.

Open policy question (pending committee ruling): ongoing but
UNQUANTIFIED bleeding with a falling gray-zone Hb currently lands as a
guardrail-asserted `INAPPROPRIATE` with no human-review flag; flooring
that pattern to `NEEDS_REVIEW` instead is a one-line change in the
assert path once ruled.

### Deferred review (post-merge)

Two-round Codex review on PR #54 (NEEDS-CHANGES → fixes → PASS), with
items intentionally **not** in scope so a future reader doesn't
re-litigate:

* **Live Anthropic polling in the reconciler beyond cached responses**
  — Phase-1 reconciler polls `fetch_batch_results` once per
  SUBMITTED/PARTIAL row that has no cached llm_calls. Cross-batch
  catch-up (multiple in-flight batches, exponential-backoff polling,
  webhook delivery) is a follow-up ticket (`bba.monitoring` #29 or a
  dedicated `bba.batch_poller`).
* **Concurrent pipeline workers** — `PostgresBatchRunStore` is
  thread-safe via the connection pool, but the orchestrator runs
  single-threaded in v1. Multi-worker batching is out of scope until
  the DuckDB single-writer constraint (PRD §10) is lifted.
* **Resume polling failure backoff** — transient Anthropic errors
  leave the `batch_runs` row in its current state for the next resume
  to retry. Explicit retry-budget tracking + dead-letter routing is a
  follow-up.
* **Full integration with `bba.quote_grounder`** — the verifier
  callable is a Protocol; the Phase-1 default returns `True`. Wiring
  `verify_citations` into the verifier hook is the next ticket's job.

## Monitoring concepts (#27)

### Weekly reviewer sample

50–75 audit rows drawn deterministically each ISO week for human
clinical review. `bba.monitoring.draw_weekly_reviewer_sample(audit_rows,
*, week_iso, sample_size, seed)` returns a `WeeklyReviewerSample`
whose `audit_ids` are stable across processes, machines, and Python
restarts: the RNG seed is derived from `sha256("week_iso|sample_size|
seed")[:8]`, not from `hash()` (which is `PYTHONHASHSEED`-dependent).
"Same week → same audit_ids" is the historical-audit invariant the
operator relies on to re-derive a past sample from the manifest alone.

### Sentinel set / sentinel manifest

A fixed 200-case cohort (default seed=42, exported as
`SENTINEL_SET_SEED`) constructed once per deployment via
`build_sentinel_manifest(audit_rows, size, seed)` and re-run weekly
through `bba.audit_pipeline` to verify model output is stable
week-over-week. The persisted `SentinelManifest` (size + seed +
`audit_ids` tuple + `built_at`) is the contract the weekly κ comparison
joins against.

### Intra-model κ alarm

`evaluate_sentinel_run(*, manifest, previous, current, kappa_threshold=0.90)`
pairs the last week's `final_classification` map against this week's
and computes Cohen's κ + Gwet's AC1 — imported from
`bba.eval_harness.agreement`, NOT re-implemented. `alarm_fired=True`
when `cohen_kappa < kappa_threshold`. Empty `previous` (no prior week)
raises `InsufficientHistoryError`; a fully-disjoint previous/current
also raises (nothing to pair).

### Wald SPRT

Wald's Sequential Probability Ratio Test for binomial drift, in
`bba.monitoring.drift_sprt`. The two hypotheses are H0 = `rate ==
p_null` (long-run baseline) and H1 = `rate == p_alt` (operator-chosen
minimum detectable shift). The cumulative log-likelihood ratio
increments by `log(p_alt/p_null)` on a success and
`log((1-p_alt)/(1-p_null))` on a failure; the verdict is computed
against the Wald bounds.

### Wald bounds (A, B)

`wald_bounds(*, alpha, beta) → (lower, upper)` where
`upper = log((1-beta)/alpha)` and `lower = log(beta/(1-alpha))`.
Symmetric in α/β: when `alpha == beta`, `upper == -lower`. Both rates
MUST be in `(0, 1)` open interval — boundary values would push the
bounds to ±∞ and are rejected by the `SprtConfig` validator at
construction time.

### Drift signal

The named binomial rate the SPRT watches. Two literals in Phase 1:
`quote_grounding_failure_rate` (`verifier_pass=False`) and
`needs_review_rate` (`needs_human_review=True`). The signal is carried
on `SprtConfig.signal` and surfaces as `SprtState.signal` so an alarm
record can identify which rate tripped.

### SPRT verdict / cycle reset

`SprtState.verdict ∈ {"continue", "reject_null", "accept_null"}`.
`run_sprt_on_window(observations, config)` walks the iterable through a
monitor: on `reject_null` it returns immediately (alarm fires); on
`accept_null` it resets the monitor and continues (a single
accept-null crossing under H0 is expected, not an alarm); at end of
window it returns the terminal state. `n_observations` on the returned
state is the TOTAL observations consumed across cycles, not the count
in the final cycle.

### Min-N gate

`SprtConfig.min_n` (default `SPRT_DEFAULT_MIN_N = 30`). The monitor
returns `verdict="continue"` until at least `min_n` observations have
been processed, even if log_lr has crossed a bound. Prevents
single-observation alarms when the random walk happens to start with a
long success run.

### ARL₀

Average run length under H0 — the expected number of observations
between false alarms. PRD §18 sets the target at ≥ 500 (≈1 false
alarm/year at expected throughput). The empirical regression check
computes `total_observations / max(n_false_alarms, 1)` over 50 null
streams × 2000 obs; the zero-alarm case is treated as right-censored
(`ARL₀ > total_observations`) rather than reported as a concrete
estimate.

### Golden-set drift probe

A fixed 100-row cohort re-run quarterly through `bba.audit_pipeline`
against the same Anthropic snapshot ID.
`evaluate_golden_set_drift(*, baseline, current, ...)` pairs entries
by `audit_id` and emits independent classification drift
(`classification_changed_pct > 0.05`) and indication drift
(`indications_changed_pct > 0.10`) alarms. Indication comparison is
set-based — reordering is NOT a change. Any `audit_id` missing from
either side raises `GoldenSetMismatchError` (the golden set itself was
edited; the comparison contract is broken).

### Monitoring alarm

Append-only record raised by one of the three alarm-emitting monitors
(SPRT, sentinel-κ, golden-set drift). The `AlarmKind` literal tags the
source; `signal` (nullable) names the SPRT signal for `drift_sprt`
alarms; `detail` is a free-form `Mapping[str, str|int|float|bool]` of
monitor-specific fields. Weekly reviewer-sample draws are NOT alarms
(no deviation signal) — they persist via `persist_sample_manifest`
instead.

### Alerting stub (Phase 1)

Two passive channels: a stdlib `logging.WARNING` event on the
`bba.monitoring.alarms` logger carrying structured `extra` fields
(`monitoring_alarm_kind`, `monitoring_alarm_signal`,
`monitoring_alarm_detail`) for log-aggregation tools, and an
append-only row in the in-memory `MonitoringStore`. Slack / email /
paging integration and a Postgres-backed store are explicit Phase 1.5
follow-ups. The boundary is structurally enforced by
`TestNoSlackEmailPagingImport` (no `slack_sdk` / `smtplib` /
`pagerduty` / `opsgenie` / `twilio` imports anywhere in the package).

### MonitoringStore

In-memory append-only store for `monitoring_alarms` rows and weekly
sample manifests. Construct with `MonitoringConfig(dsn, app_name)` —
both fields are carried for Phase 1.5 substitutability (Postgres swap)
but ignored by the in-memory implementation. `persist_sample_manifest`
is idempotent on `(week_iso, sample_size, seed)`: first write wins,
later writes are silently no-op so cron retries don't multiply rows.
Lock-guarded for multi-threaded write safety; the
`TestMonitoringStoreConcurrentWrites` regression uses a
`threading.Barrier` to force contention.

### Operational, not clinical

`bba.monitoring` reads pipeline artifacts (`audit_results`, `llm_calls`
via `bba.audit_store`) and statistical helpers (κ / AC1 via
`bba.eval_harness.agreement`); it does NOT import any clinical-logic
module (`hb_lookup`, `vitals_extractor`, `cohort_detector`,
`deterministic_classifier`, `quote_grounder`, `prompt_builder`,
`evidence_bundle_builder`, `deid_redactor`). The structural import
gates (`TestNoClinicalImports`, `TestNoSchedulerImports`,
`TestNoLiveAnthropicInDriftProbe`) prevent accidental coupling. Cron
scheduling itself is `#29 cli`'s job; the monitoring module exposes
callables only.

## CLI concepts (#29)

### bba root group

The `click.Group` named `bba`, declared at `bba.cli.main.cli`. Six
subcommands attach to it: `ingest`, `audit`, `evaluate`, `report`,
`serve-dashboard`, `sentinel`. The group callback installs the
PHI-scrubbing `sys.excepthook` before any subcommand body runs, so a
crash anywhere downstream produces a redacted traceback instead of
leaking PHI to the operator log.

### Thin-glue rule

Every subcommand body is a ≤ 20-statement wrapper over an
already-tested module entrypoint. If a subcommand grows business
logic, the right move is to push the logic down to the module and
re-export. The body-statement budget is enforced by
`TestMainModuleIsThin` against the AST of each callback (decorators
and docstrings excluded).

### CLI run_id

The 16-char hex prefix of `bba.ingest.RunIdentity`'s 64-char digest,
computed over **every HOSxP-named CSV in `input_csv.parent`** plus the
schema fingerprint plus `code_version()` (the package version from
`importlib.metadata`). Living at `bba.cli.identity.compute_run_id`.

Bundle-aware by construction: hashing only the operator's `--input`
file would make `bba audit` blind to sibling-table edits — a freshly-
exported `Diagnosis.csv` next to an unchanged `BDVST.csv` would
otherwise yield the same run_id as the prior audit and the idempotency
guard would no-op against stale results. The CLI's run_id and the
ingest module's `RunIdentity.run_id` therefore share the same
construction (the CLI just truncates to 16 chars for grep-friendly
logs); they cannot drift.

### Audit run store

A narrow Protocol at `bba.cli.store_protocol.AuditRunStore` (six
operations: `run_complete`, `run_count`, `record_row`,
`mark_run_complete`, `record_idempotency_override`,
`audit_log_entries`, `acquire_run_lock`). Phase 1 ships the
file-backed implementation `bba.cli.audit_run_store.FileBackedAuditRunStore`
rooted at `$BBA_DATA_DIR/audit_runs/`. A future Postgres-backed
adapter (covered by `bba.audit_store`'s extension ticket) can
implement the same Protocol and swap in via
`bba.cli.main._get_audit_run_store` without touching the CLI surface.

### Run lock

An exclusive `fcntl.flock(LOCK_EX)` on
`$BBA_DATA_DIR/audit_runs/run_<run_id>.lock`, exposed by
`AuditRunStore.acquire_run_lock(run_id)` as a context manager.
`bba_audit` wraps the check-then-act sequence
(`run_complete` → run pipeline → `mark_run_complete`) in this lock so
two concurrent CLI invocations on the same input cannot both pass the
guard and double-execute. The lock is per-`run_id`; concurrent audits
of *different* inputs run in parallel. Crash safety is OS-provided:
a crashed process releases its kernel-held flock automatically, so a
retry after a crash does not stall.

### Idempotency override

A `--force` flag on `bba audit` that causes the subcommand to skip the
`run_complete` no-op branch and re-execute the pipeline. Each
invocation under `--force` writes one row to the file-backed audit log
naming the run_id, the override reason, and the wall-clock timestamp.
The override is a compliance surface — durability is enforced by
writing through `os.write(fd, line)` on an `O_APPEND`-opened file
descriptor (POSIX-atomic up to `PIPE_BUF`) followed by `os.fsync(fd)`.

### audit_log.jsonl

The on-disk JSONL stream of idempotency-override events under
`$BBA_DATA_DIR/audit_runs/audit_log.jsonl`. One JSON object per
override invocation with the schema `{run_id, idempotency_override:
true, reason, ts}`. Append-only by construction; blank lines are
skipped on read so an editor's trailing newline never trips JSON
parsing.

### PHI scrubber

The `bba.cli.phi_scrubber.scrub_traceback` function plus the
`install_excepthook` installer. Walks the traceback frame chain and,
for each frame: replaces PHI-named locals (whose name matches the
prefix regex `bundle|patient|note|hn|an|encounter`, case-insensitive)
with `<REDACTED:type=<typename> len=<N>>`, then regex-sweeps every
remaining string value against `PHI_REGEXES` and replaces each match
with `<REDACTED:phi>`. The scrubbed traceback is emitted through the
structlog-to-stdlib bridge in `bba.cli._logging` so `caplog`, file
handlers, and any SIEM forwarder all see the same redacted JSON
event.

### PHI_REGEXES

The single-source-of-truth tuple of PHI patterns at
`bba.deid_redactor.PHI_REGEXES`. Three patterns: HN/AN-shaped digit
runs (7–10 digits), Western honorific + capitalised name tokens
(`Mr Smith`), and Thai honorific + the trailing Thai- or Latin-script
name span. The third matches the **full** name span — not just the
honorific — because `pattern.sub("<REDACTED:phi>", text)` would
otherwise leave the patient's given+family name in the operator log.

`bba.cli.phi_scrubber.PHI_REGEXES` re-exports this tuple unchanged so
the redactor and the traceback scrubber cannot drift.

### Phase 1 ingest leg

The Phase 1 deliverable for `bba audit`: call
`bba.ingest.ingest(IngestConfig(...))` against `input_csv.parent` and
record one `phase1_ingest_<table>` row marker per validated HOSxP
table from `IngestResult.tables_written`. The LLM-driven analysis leg
(deterministic classifier → evidence bundle → de-id → prompt → batch
submit → quote ground → calibrate → audit_store write) requires a
context-builder composition over `audit_orders` /
`deterministic_classifier` / `evidence_bundle_builder` /
`deid_redactor` / `prompt_builder` that `bba.audit_pipeline`'s contract
explicitly delegates to its caller. When that orchestration facade
lands, the `_run_audit_pipeline` seam delegates to it.

### Zero-table guard

The defensive check in `_run_audit_pipeline` that raises `CliError` if
`IngestResult.tables_written` is empty. Real ingest raises
`IncompleteInputError` before returning an empty tuple, but the guard
remains so a future ingest revision cannot silently mark a run
complete with zero row markers — keeping `run_complete=True` and
`run_count==0` an impossible state.

### Integration-contract CliError

The pattern used by the four subcommands (`evaluate`, `report`,
`serve-dashboard`, `sentinel`) whose underlying module does not yet
expose a single-call facade matching the CLI's flag surface. Each
body validates inputs via Pydantic, emits a structured `<sub>.start`
log, and raises `CliError` naming the integration seam (e.g.
`bba.cli.main.bba_evaluate`) and the underlying module's currently
exposed primitives. The error is not a TODO — it's the CLI's
"fail loud at the integration boundary" contract from PRD §20, and is
the regression guard that keeps the subcommands' bodies thin until
the underlying facades land.

### Excepthook + faulthandler sidecar

`install_excepthook(faulthandler_sidecar=...)` installs both
`sys.excepthook` (which scrubs Python-level exceptions via
`scrub_traceback` and emits the result through structlog) and an
optional `faulthandler` redirect to `$BBA_DATA_DIR/logs/faulthandler.sidecar`.
The sidecar covers hard interpreter crashes (e.g. SIGSEGV in a C
extension) that bypass `sys.excepthook`; the sidecar is scrubbed on
read, not on write — the write path is too hot to traverse frames.

### Env-var allow-list

The CLI reads exactly three environment variables: `BBA_DATA_DIR` (the
audit data root), `BBA_DB_URL` (Postgres DSN for the future
audit_store adapter), and `ANTHROPIC_API_KEY` (the live transport's
credential). `TestEnvVarSurfaceIsTight` is a static regression guard
that scans `src/bba/cli/` for any other `os.environ[...]` /
`os.getenv(...)` literal and fails CI if it grows.

### Sentinel cadence flag

A mutually-exclusive `--weekly | --quarterly` pair on `bba sentinel`
implemented as two `is_flag=True` booleans plus an explicit XOR check.
The click `flag_value` shortcut silently last-wins on conflict, so
two booleans + a body-level guard are the only way to reject "both
flags supplied" as a usage error.

### Structlog stdlib bridge

`bba.cli._logging.get_logger("bba.cli")` returns a
`structlog.stdlib.BoundLogger` configured (once per process) with the
JSON renderer chained through `structlog.stdlib.LoggerFactory`. Every
CLI log event renders to a JSON line and is handed to
`logging.getLogger("bba.cli")`, so `caplog` /
`logging.handlers.RotatingFileHandler` / SIEM forwarders all consume
the same stream without extra adapters.

## Component-map concepts (Phase 2)

Deterministic-core module `bba.component_map` — the intake gating prerequisite
that lets the auditor tell a platelet order from a red-cell one without
blending their statistics. Docs plan §5.3 gating prerequisite (AR-M8).

### Component family

Six-member `StrEnum` naming the blood-component family a BDTYPE product
belongs to: `RED_CELL`, `PLATELET`, `FFP`, `CRYO`, `WHOLE_BLOOD`, `UNKNOWN`.
`bba.component_map.ComponentFamily`. Only `RED_CELL` (Phase 1) and `PLATELET`
(Phase 2) have auditors; `FFP` / `CRYO` / `WHOLE_BLOOD` are recognised so the
intake gate can EXCLUDE them with a precise reason (docs plan §6). `UNKNOWN`
is the honest fallback for an unrecognised name — never guessed into a real
family.

### classify_component (NAME classifier)

`bba.component_map.classify_component(product_name) → ComponentFamily`. A
keyword classifier over the BDTYPE dictionary's `NAME` column — the source of
truth. Robust to new product codes: irradiated / filtered / pooled variants
all resolve to the same family from their descriptive name.

### BDTYPE_FAMILY and PLATELET_PRODUCTS

`bba.component_map.BDTYPE_FAMILY` is the frozen, verified code→family map for
the fast intake gate (mirrors `bba.audit_orders.rules.RBC_PRODUCTS`);
`component_of_code` reads it. `PLATELET_PRODUCTS` is the 8-code platelet
allow-list (`LDPPC, LDPPCI, SDPFI, SDPF, SDPPI, LPPC, PC, LDPC`);
`is_platelet_product` tests membership. Every entry was cross-checked against
the KCMH dictionary's `GRPCAUSELABCBC` grouping AND `classify_component`; the
`test_component_map` suite asserts the two never disagree on a known code.

### Family volume (raw BDVSTDT, verified 2026-07-08)

Issued-unit counts recount from the raw `BDVSTDT` feed (join by column name —
the raw column order differs from the bundle projection, `BDTYPE` at index 3):
`red_cell 23867, platelet 8193, ffp 5350, cryo 2335, whole_blood 4`. The 8,193
platelet units break down as `LDPPC 6392, LDPPCI 792, SDPFI 600, SDPF 398,
LPPC 9, PC 1, LDPC 1` (`SDPPI` is in the dictionary but 0 issued).

## Platelet-lookup concepts (Phase 2)

Deterministic-core module `bba.platelet_lookup` — the platelet counterpart to
`bba.hb_lookup`'s value layer. **Config + parser + observation model ONLY**;
the recent-value selection engine is deferred (see below).

### Platelet lab config

`bba.platelet_lookup.PLATELET_LABEXM = "290078"` ("Platelets Counts", LABGRP
29 HEMATOLOGY, unit ×10³/µL, reference range 150-450 — confirmed against the
Lab dictionary 2026-07-08). Unlike Hb (which has a POCT fallback, LABEXM
500001), the platelet count is HEMATOLOGY-only in this dataset, so
`PlateletSource` is the single-valued `Literal["HEMATOLOGY"]`.

### parse_platelet_count

`bba.platelet_lookup.parse_platelet_count(raw) → float | None`. NOT a copy of
the Hb parser — the count column is messier. Handles comma-grouped thousands
(`1,117`), left/right-censored `<N` / `>N` values (mapped to the bound `N` —
`<2` is a REAL critically-low measurement, not missing), and the `--` / empty
/ non-numeric sentinels (→ `None`). Out-of-range values (outside
`[MIN_PLATELET=1.0, MAX_PLATELET=3000.0]` ×10³/µL) and `nan` / `inf` → `None`,
failed loud as transcription/unit errors rather than coerced.

### PlateletObservation

`bba.platelet_lookup.PlateletObservation` — a frozen, validated single count
from the Lab table: `value_k_ul` in `[MIN_PLATELET, MAX_PLATELET]`, tz-aware
UTC `datetime_utc`, `source`, and `item_no` (the tie-breaker when two
observations share an exact datetime, mirroring `HbObservation.item_no`).

### Deferred: recent-value selection engine

The recent-value SELECTION engine (source preference, freshness tiers, trend /
staleness) is shared with `bba.hb_lookup` and is extracted into a
component-parameterised core in a follow-up (docs plan §5.3 stage 2). This
module does NOT duplicate that logic — `hb_lookup`'s `[2, 25]` g/dL parser
would reject every platelet count, which is why it cannot simply be reused.

## Platelet-classifier concepts (Phase 2)

Deterministic-core module `bba.platelet_classifier` — the v1 platelet §5.1
gate, mirroring `bba.deterministic_classifier` for the RBC path but
**auto-clearing NOTHING**.

### classify_platelet (§5.1 gate)

`bba.platelet_classifier.classify_platelet(inputs) → PlateletClassifierResult`.
Precedence (top wins): count missing → `NEEDS_REVIEW` /
`plt_missing_defer_llm` (defer flag on) or `INSUFFICIENT_EVIDENCE` /
`plt_missing` (flag off, the default); count ≥ `PLATELET_REVIEW_CEILING` →
`POTENTIALLY_INAPPROPRIATE` / `plt_ge_100` (routes to review, NOT terminal);
count < ceiling (including < 10) → `NEEDS_REVIEW` / `plt_defer_llm`. The
missing-count defer flag mirrors the RBC `enable_missing_hb_positive_evidence`
opt-in and is a SEED pending clinical sign-off.

### PLATELET_REVIEW_CEILING

`bba.platelet_classifier.PLATELET_REVIEW_CEILING = 100.0` (×10³/µL). At or
above this count a platelet order is `POTENTIALLY_INAPPROPRIATE`. A SEED
pending clinician sign-off (docs plan §7); OPEN (§5.4) whether 100 is the
right ceiling or an 80-100k high-bleeding-risk surgical band should also
defer. One named constant so a ruling change is one line.

### Nothing auto-cleared (CR-C1 invariant)

The original `plt < 10 → APPROPRIATE` auto-clear was REMOVED as a
patient-safety defect (§8/CR-C1): `APPROPRIATE` is deterministic-final (no
LLM, no human), but the policies WITHHOLD platelets at very low counts for 6+
exclusion populations (dengue-no-bleed, TTP, HIT, ITP, aplastic-no-bleed,
snakebite-no-bleed — TTP/HIT transfusion is actively harmful). The classifier
therefore NEVER emits `APPROPRIATE` or `INAPPROPRIATE` in v1; every present
count routes onward, so no present-count verdict is ever deterministic-final.
Pinned by a hypothesis property test (CR-M2).

### Deferred: MTP-window suppression

MTP-window suppression is intentionally NOT in this pure single-order gate.
Platelet units co-ordered inside an active massive-transfusion-protocol window
are suppressed as independent audit units at dispatch (docs plan §5.3
stage 4) — the MTP co-order stays a *signal*, never a second audit row. That
needs cross-order context this gate does not have.

## Vocabulary not in this file

Domain terms used in the broader project but defined elsewhere (PRD issue
body, or future Phase-1 modules):

* `MTP`, `T1.MTP`, `hallucination_suspect`, etc. — see the PRD
  (issue #1) for definitions.

Add a concept here when a module exposes it as part of its interface. Update
when a concept's shape changes; remove when it leaves the codebase.

## Administration confirmation / reserve-ahead router (#105 Phase 5)

The pilot export records every blood order at `BDVSTST=4` (dispensed) and
never records status 5 (transfused); `USETIME` is populated for only 128/300
orders. Administration therefore cannot be confirmed from structured status
data, and reserve-ahead orders must not be treated as completed transfusions.

Default-off `feature_flags.RESERVE_AHEAD_ROUTER_ENABLED` routes
`preop_defer_llm` cases to `RESERVE_AHEAD_REVIEW`. Its asymmetric result gate
accepts grounded affirmative administration evidence, including the fact-only
`scan_administration` extractor; without confirmation it stores
`PREOP_RESERVATION_UNCONFIRMED` (`needs_human_review=False`), projects it to
Unresolved, and excludes it from inappropriate-transfusion attribution.

See `docs/administration-confirmation-plan.md` for the data request, gate
contract, pilot enablement, and regeneration runbook.

## Verification concepts

Deterministic-core module `bba.verification` — honest before/after scoring for
the peri-op fix. Pure and data-source-agnostic: callers assemble
`reqno -> CaseVerdict` maps from whatever run artifacts they have (`report.csv`
/ `llm_report.json`) and pass them in; the end-to-end scoring run over the pilot
bundle lives in the pilot scripts, gated on the bundle being present.

### CaseVerdict and the 3-bucket space

`bba.verification.CaseVerdict` binds a REQNO to its verdict; `bucket_of` /
`BUCKETS` collapse every classification into the same 3-bucket space the
300-case human labels live in (`bba.attribution.human_label_verdict_source`), so
a run is scored on the axis a reviewer actually used. `Bucket` and `Mechanism`
(deterministic vs LLM) are the `StrEnum`s that name the axes.

### compare_runs (before/after)

`bba.verification.compare_runs(labels, before, after) → RunComparison` judges the
fix on its real costs, not one headline. `labels` is the human-label map
(`reqno -> label`); `before` / `after` are the two runs' `reqno -> CaseVerdict`
maps. `find_regressions(labels, before, after)` surfaces the cases the after-run
made worse, and the comparison carries the LLM-volume delta alongside the
accuracy move. `build_matrix(labels, verdicts, *, scope=...)` and
`confusion_by_mechanism(labels, verdicts)` score a SINGLE run into the
`ConfusionMatrix` / `ConfusionCell` grids, split by `MatrixScope` (`all`,
`deterministic`, `llm`). Full surface: `bba.verification.__all__`.

## Returns-ledger concepts (#120)

Read-only plumbing module `bba.returns_ledger` — BDVSTTRANS returns-ledger
disposition aggregation (spec #119). Public surface is the pure
`summarize_returns` aggregator and the frozen `ReturnsSummary` it produces; the
disposition ROUTER that consumes the summary (`RETURNED_NOT_TRANSFUSED` /
`PERIOP_TRANSFUSION_EXEMPT`) lives in the pipeline, gated behind default-on
`feature_flags.RETURNS_LEDGER_ENABLED`.

### summarize_returns and ReturnsSummary

`bba.returns_ledger.summarize_returns(trans_rows, unitamt_lines) → ReturnsSummary`
aggregates one REQNO's BDVSTTRANS ledger rows (`trans_rows`) plus the order's
BDVSTDT `UNITAMT` detail lines (`unitamt_lines`, which give the ordered amount)
into a frozen summary. `units_total` counts DISTINCT physical units, not raw
rows: `physical_units(trans_rows)` collapses rows sharing a fully-present
`(DNRNO, SEQNO, BDTYPE)` key to one unit at its terminal status before counting.
`BDTYPE` is part of the key so distinct products of one donation (split
`SDRF`/`SDRF2`, a pooled `LDPC`, or an irradiation relabel) are never merged and
a presumed-transfused unit can never hide behind another product's return; a row
missing ANY key component becomes its own unit (fail-closed). `rows_for_admission`
is the row-scoping helper; `terminal_status` reduces a unit's rows to its
terminal `Unitstat`.

### Disposition (derived, fail-closed)

`bba.returns_ledger.Disposition = Literal["not_transfused", "transfused",
"inconclusive"]`. `ReturnsSummary.disposition` is DERIVED, never stored, so a
summary cannot exist in an inconsistent state. It reads `not_transfused` only
when the ledger COVERS the order (`ledger_complete`) AND every unit reached a
non-transfusion terminal — returned (`Unitstat==3`) OR crossmatch-incompatible
(`Unitstat==7`, never given); `transfused` when complete and at least one unit
is neither; `inconclusive` otherwise (fail-closed on a partial export). Go-live
evidence is produced read-only by
`scripts/pilot/preflight_returns_validation.py` (#125).

## Declared-use / USETYPE concepts (#147)

Leaf module `bba.declared_use` — the single source of truth for interpreting the
order-time declared use from `BDVSTDT.USETYPE`, so the ingest, classifier,
bundle, and pilot layers cannot drift. Threaded through both pilot legs behind
`feature_flags.DECLARED_USETYPE_ENABLED` and the `BBA_PILOT_DECLARED_USETYPE`
seam.

### USETYPE_LABELS and the declared surgical group

`bba.declared_use.USETYPE_LABELS` maps raw codes to a `DeclaredUseLabel`
(`1→ward`, `2→surgery`, `3→type_screen`, `4→day_care`); `label_for(code)`
applies it, defaulting unknown codes — including `5` — to `unknown`, which drives
no routing. `DECLARED_SURGICAL_LABELS = {surgery, type_screen}` is the declared
surgical group that widens the step-6 preop deferral.

### collapse_usetype and DeclaredUse

`bba.declared_use.collapse_usetype(values)` reduces an order's detail lines to
one code: one distinct non-blank value wins; blank-only inputs and mixed codes
both return `None` (mixed logs a warning, never raises). `DeclaredUse` is the
frozen `(code, label)` pair whose validator FAILS LOUD if the label does not
equal `label_for(code)` — a surgical code can never be represented as
non-surgical for the `.label`-keyed routing. Prefer `DeclaredUse.from_code`,
which derives the label.

## Platelet-guardrail concepts (Phase 2)

Deterministic-core module `bba.platelet_guardrail` — the platelet LLM
over-clear guardrail (docs plan §5.3 stage 5; user ruling 2026-07-08). The
mechanism that makes the "ADD platelet hard signals" ruling safe: the LLM CAN
clear any platelet count, but only with a grounded positive indication.

### platelet_overclear_suspect

`bba.platelet_guardrail.platelet_overclear_suspect(final, rule, hard_signals) →
bool`. True iff the LLM returned `APPROPRIATE`, the deterministic gate had
WITHHELD the clear (`NEEDS_REVIEW`, `INSUFFICIENT_EVIDENCE`, or
`POTENTIALLY_INAPPROPRIATE` — the last included because transfusing at/above the
100k ceiling is almost never appropriate and has no separate prompt backstop),
and `hard_signals.any_signal()` is false. On a hit the pipeline FLOORS the row
to human review (never to `INAPPROPRIATE`; an unauditable clear is reviewed, not
condemned) and stamps `PLATELET_OVERCLEAR_REVIEW_REASON =
"platelet_llm_overclear_suspect"` so the dashboard can triage suspects
separately. Pure and side-effect-free — a replay-stable mirror of
`bba.audit_pipeline.replay.llm_overclear_suspect`. `PlateletHardSignals` carries
the structured grounded signals and their `any_signal()` test.
