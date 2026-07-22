# MSBOS reservation annotation for returns-terminal orders

**Status:** SHIPPED (2026-07-17). Landed as spec #176 → tickets #177–#180 →
PRs #181/#182/#183/#184, all merged to main. This file is the design plan of
record: consortium-reviewed (Codex advisor + 3 Claude lenses; all blockers
folded) and grilled (4 user rulings recorded in section 6).
**Scope:** Surface the already-computed MSBOS reservation-appropriateness
judgment on orders whose final classification is a returns terminal
(`RETURNED_NOT_TRANSFUSED` / `PERIOP_TRANSFUSION_EXEMPT`) — as an
**annotation**, never a verdict change.

---

## 1. Problem

Returns-terminal orders skip the LLM (correct — no scorable transfusion) and
also skip the MSBOS over-reservation verdict: the overlay predicates
(`is_over_reservation` and twins, `src/bba/audit_pipeline/replay.py:161-210`)
return `False` whenever the classifier verdict is in `_RETURNS_TERMINALS`
(`replay.py:158`), and the pilot report treats an over-marker on a
returns-terminal row as a double-fire FAIL
(`src/bba/preop_reservation/pilot_report.py:255-269`).

That precedence is correct bookkeeping (one `final_classification` per row)
but it hides the clinical question: these orders still carried a pre-op M/G or
T/S reservation. `RETURNED_NOT_TRANSFUSED` is the classic C:T-ratio waste case
— blood crossmatched, never used — exactly where an over-reservation finding
pays off most. In the current pilot artifact (`/tmp/bba_mini`, 300 audited
orders): 43 returned + 177 peri-op-exempt ROWS, zero MSBOS judgment on any
of them (earlier drafts cited 87/355, which counted string occurrences in
review.html, not rows — never hardcode these counts).

Key architecture fact: `review.html` is built by `scripts/pilot/build_review.py`
from the **deterministic leg's** `report.csv` (`scripts/pilot/run_pipeline.py`),
which today loads no MSBOS reference and computes no `ReservationDecision`.
The LLM leg (`scripts/pilot/run_llm_leg.py:1969-1992`) computes the decision
for every RBC order but never persists returns-terminal rows at all
(`run_llm_leg.py:2146` drops deterministic-final rows from submission).

## 2. Locked rulings (do not relitigate in implementation)

1. `final_classification` stays the returns terminal. No precedence flip, no
   new classification, no change to scoring buckets or excluded-from-scoring
   semantics. Returned/exempt describes the transfusion outcome; over-reserved
   describes the ordering decision — orthogonal axes, annotation only.
2. No new audit-store marker keys. `reconcile_returns` double-fire FAIL and
   `_validate_join` invariants stay byte-identical. The store-side annotation
   (T3) rides inside the existing deterministic marker-call payload
   (`model_id="deterministic"`), which `pilot_report._marker_calls` does not
   match.
3. Flag seam: reuse the existing MSBOS seam, no new flag. Pilot legs:
   `BBA_PILOT_MSBOS_RESERVATION` overrides, else
   `feature_flags.MSBOS_RESERVATION_ENABLED` (default-ON since PR #175).
   Mirror `run_llm_leg.py:191-196` exactly. Forced-off runs must be
   byte-identical to today (report.csv columns included).
4. Deterministic leg computes the annotation for returns-terminal rows ONLY.
   Non-returns rows keep blank annotation columns — the billed
   `PREOP_OVER_RESERVATION` path (LLM leg, reserve-ahead router) remains the
   single source of over-reservation verdicts and must not be duplicated or
   contradicted by this report.
5. MSBOS cutoffs, reference data, note disambiguation, and reserved-unit
   keying are reused read-only, as-is. No threshold or reference edits.

## 3. Tickets

Stacked, smallest-first; each independently green (tests + ruff). T1-T3 are
flag-off byte-identical; T0 is an unconditional bugfix to a regenerated
artifact (review.html is rebuilt every run, not a persisted store output),
so T0 deliberately changes review.html for all runs.

### T0 — review.html summary-table and pill repair (prefactor, no MSBOS logic)

Fixes the style defects found by the 2026-07-17 /impeccable audit (full
findings in section 6) and clears the structural blocker for adding a
summary-table column. All in `scripts/pilot/build_review.py`:

- **Define the three missing pill classes** in the `css` string (~line 1709
  block). `.cls-returned_not_transfused` and `.cls-periop_transfusion_exempt`
  (43 + 177 uses in the current artifact render as colorless pills — the two
  biggest buckets) use the neutral pill role (`--neu-bg`/`--neu-fg`; they are
  excluded-informational, not warnings). `.cls-preop_over_reservation` uses
  the err role (`--err-bg`/`--err-fg`; it projects INAPPROPRIATE).
- **Move summary-table pill wrapping server-side.** The current JS
  (`build_review.py:1937-1957`) derives CSS classes by normalizing the
  DISPLAY label, producing garbage like
  `cls-returned_—_not_transfused_(excluded)` and
  `cls-administration_unconfirmed_(pre-op_reservation)` (em-dash and
  parentheses in class names, none defined → every non-core verdict in the
  summary table renders colorless). The raw classification is available at
  build time: wrap the Deterministic/LLM cells in
  `<span class='cls cls-<raw-lowercase>'>` when building `summary_rows`
  (~line 1189), and DELETE the fragile JS block — its hardcoded
  `cells[8]`/`cells[9]` indexes would silently break the moment T1 adds a
  column.
- **Unify the two mismatch definitions on the Python one.** Today the
  report computes verdict-mismatch TWICE with different semantics: the JS
  row highlight compares normalized display strings with a sentinel skip,
  while `case_mismatch_tags` (`build_review.py:1158-1187`) compares raw
  classifications, no-flags when the LLM leg did not run, and carves out
  `PREOP_RESERVATION_UNCONFIRMED` — so a row can be shaded as a mismatch
  while the header count (`n_mismatches`), the nav [!] flags, and the
  "Mismatches only" filter all disagree. When porting server-side, drive
  `tr.verdict-mismatch` from the truthiness of `case_mismatch_tags[i]`
  (available per-row when `summary_rows` is built) instead of re-deriving a
  second definition. This is a deliberate unification: rows where the LLM
  verdict is `PREOP_RESERVATION_UNCONFIRMED` lose today's (inconsistent)
  shading, matching the count/filter/nav behavior that already suppresses
  them.
- **Escaping (Codex).** `render_table` HTML-escapes every cell
  (`build_review.py:541`), so span markup placed in `summary_rows` values
  will render as literal text. Do NOT relax escaping globally — the same
  helper renders clinical-note-derived tables and that would open HTML
  injection. Use a dedicated summary-table renderer (or an explicitly typed
  trusted-cell mechanism) where ONLY the generated pill markup and row
  class bypass escaping; every data value stays escaped.
- **Pill spacing.** The deleted JS added inline `margin-bottom:0` to table
  pills (`build_review.py:1951`); the server-side port needs a summary-
  scoped CSS rule (e.g. `table .cls { margin-bottom: 0; }`) or pills regain
  the base 8px bottom margin and misalign rows.
- **Intended behavior change to state in the ticket:** today the JS
  sentinel-skip (`build_review.py:1947`) leaves any row whose LLM cell is
  "—" or "(...)" — i.e. every returns-terminal row — as plain unwrapped
  text in the summary table. After the server-side port those rows gain
  neutral pills in the Deterministic column. That is the intent, not a
  regression; say so explicitly so a reviewer diffing the artifact does not
  flag it.
- Note: `tests/unit/test_pilot_build_review.py` already asserts the exact
  strings `cls-returned_not_transfused` / `cls-periop_transfusion_exempt`
  — keep those class names verbatim; define them, do not rename them.
- **Wrap the summary table in an `overflow-x: auto` container.** 10 columns,
  and clinical reviewers read on tablets (PRODUCT.md audience 3); today the
  whole page scrolls horizontally.
- **Kill the off-system inline-styled button.** The kbd-hint Dismiss button
  (`build_review.py:1858`) carries inline `font-size:0.7rem;padding:1px 6px`;
  move it to a class using the system small size (0.75rem). Same pass: drop
  the `h2#summary` inline `style='margin-top:0;'` into the css string.
- **Add `:focus-visible` styles** for buttons, links, and the jump input
  (2px `--accent` outline, offset 2px). The report advertises keyboard
  shortcuts (j/k/e/f/E/x) but designs no focus indicator.
- No behavior change to case rendering, filtering, mark-reviewed, or any
  CSV/store output. Verify by regenerating review.html from the same
  report.csv and diffing: only the summary table markup, css block, and
  kbd-hint line may differ.

### T1 — Deterministic leg: RBC reservation annotation in report.csv + review.html

`scripts/pilot/run_pipeline.py`:

- Add the pilot flag seam (mirror `run_llm_leg.py:191-196`).
- When the seam is on, load `load_msbos_reference()` and
  `reserved_units_by_component(bdvstdt)` (recipe:
  `run_llm_leg.py:1107-1112`; imports from `bba.preop_reservation`).
- Mirror `_planned_op_icd9` from `run_llm_leg.py` as a local helper — the
  legs mirror helpers by convention rather than cross-importing (see the
  `_RETURNS_TERMINAL_CLASSIFICATIONS` local-mirror comment,
  `scripts/pilot/run_pipeline.py:259-263`). The leg already builds
  `op_events` (`run_pipeline.py:939`).
- For an RBC order whose classifier result classification is a returns
  terminal: compute the `ReservationDecision` as `run_llm_leg.py:1969-1992`
  does — including the `"\x00AMBIG"` → `reason="ambiguous_planned_op"`
  sentinel branch — with ONE deliberate divergence (Q1 RULING, 2026-07-17):
  **no note disambiguation in this leg.** Call the plain
  `evaluate_reservation(...)`, NOT `evaluate_reservation_with_notes(...)`;
  conflicting-code orders keep `reason="ambiguous_code"` and annotate as
  unresolved. Rationale: the billed path's `note_texts` are the emitted
  evidence-bundle chunks (windowed + capped + ranked, `builder.py:1062`);
  this leg has no bundle builder, `vitals_notes_for` is un-windowed, and a
  wrong resolution is worse than an unresolved one (house never-guess
  rule). Component key: `(hn, reqno, ComponentFamily.RED_CELL)`.
  Consequence: `note_resolved` is constant False here, so the report.csv
  schema DROPS the `msbos_note_resolved` column (reintroduce only if the
  windowed-notes follow-up in section 7 ever ships).
- **Join-miss guard (never-guess).** `reserved_units_map.get(key, 0)` makes
  "genuinely zero units reserved" and "BDVSTDT join silently failed" (blank
  HN, REQNO mismatch, whitespace drift) indistinguishable — a broken join
  would render "within recommendation", a false clean pass. T1 must check
  key PRESENCE: when `(hn, reqno, component)` is absent from the map and the
  order's HN/REQNO are non-blank, emit a distinct leg-local annotation
  reason (`reservation_lookup_miss`, warn-style rendering), never a
  within/clean reason. This reason lives in the pilot leg's report
  vocabulary only — do NOT add it to the library `ReservationReason` enum
  (the billed path never emits it). Unit tests: blank-HN order and
  unmatched (HN, REQNO) both produce `reservation_lookup_miss`, distinct
  from a real zero-reservation order.
- **Fieldnames presence test.** `DictWriter` uses `extrasaction="ignore"`
  (`run_pipeline.py:1136-1144`), so forgetting to append `msbos_*` to the
  gated `fieldnames` list silently drops the columns and flag-on output
  masquerades as flag-off. Add a test asserting the `msbos_*` keys are in
  `fieldnames` when the seam is on.
- **CODE_VERSION is out of scope here, deliberately.** Unlike
  `run_llm_leg.py:210-221`, this leg keeps its static `"pilot-mini"`
  (`run_pipeline.py:103`): it has NO store writer — `AuditOrdersConfig
  (code_version=...)` is documented inert, and report.csv is fully
  regenerated each run, so the store-idempotency suffix pattern does not
  apply. Add a one-line comment at `run_pipeline.py:103` saying so, to stop
  a future reader from "fixing" the asymmetry.
- New report.csv columns, appended via the same conditional-fieldnames
  pattern as `RETURNS_LEDGER_FIELDNAMES` (`run_pipeline.py:1135-1140`),
  populated only on returns-terminal rows: `msbos_reserved_units`,
  `msbos_token`, `msbos_recommended_units`, `msbos_reason`, `msbos_is_over`,
  `msbos_resolved_icd9` (no `msbos_note_resolved` — see the Q1 ruling;
  `msbos_reference_hash` added per the Codex finding below).

`scripts/pilot/build_review.py`:

- In the two excluded buckets (display names at `build_review.py:260-266`),
  render the annotation per case, driven by `msbos_reason` (exact strings:
  section 6 Q4; T/S wording says "reserved", never "crossmatched" — proxy
  ruling). **Q2 RULING (user, 2026-07-17): the reservation-QUANTITY
  judgment (over / within / unresolved) renders identically in BOTH
  buckets** — actual peri-op use does not retroactively justify the
  reservation size, and the transfusion indication itself is deliberately
  NOT judged on exempt rows (the anesthesiologist judges when to transfuse;
  that is exactly why the bucket is exempt from transfusion scoring).
  PERIOP_TRANSFUSION_EXEMPT rows ADDITIONALLY report the transfused and
  returned unit counts from the existing returns columns
  (`returns_units_transfused` / `returns_units_returned`, already in
  report.csv), e.g. "reserved 4; tariff G/M 2; 2 transfused, 2 returned".
  The glossary states this split explicitly: on exempt rows the annotation
  judges ordering quantity only, never the transfusion decision. Blank
  columns render nothing (flag-off and non-returns rows unchanged).
- Bucket count lines are PER-BUCKET, never merged: "N of 43 returned orders
  over tariff" and "M of 177 exempt orders over tariff" as separate lines
  (counts computed at build time; wording per Q4).
- **Add an "MSBOS" column to the summary table — emitted in flag-ON runs
  ONLY.** (Codex blocker: an always-present column with em-dashes violates
  the flag-off byte-parity AC. When the seam is off, the column, its
  header, the bucket counts, the glossary lines, and any MSBOS-specific CSS
  are omitted ENTIRELY — blank cells are still byte changes.) Built at
  `build_review.py:1189-1204`; T0's server-side pill port is a
  prerequisite. Flag-on content: a terse pill driven by `msbos_reason` for
  returns-terminal rows, an em-dash for all other rows. Pill color and
  exact strings per the section 6 Q3/Q4 rulings (warn/amber for over-tariff
  and unresolved, fact-first strings); reserved-unit counts are a
  crossmatch PROXY (`evaluate.py:40`; the T2 ruling made BDVSTST 3/10
  unusable), so display text says "reserved", never "crossmatched".
- **Include `msbos_reference_hash` in the report.csv columns** — it is part
  of the frozen decision snapshot (`models.py:54`) and lets a reviewer tie
  an annotation to the exact vendored reference version.
- **Join integrity (Codex).** `build_review.py:736` collapses deterministic
  rows into a bare-REQNO dict, while the pipeline documents that REQNO can
  recur across admissions (`run_pipeline.py:955`). Before rendering MSBOS
  annotations, fail loud on duplicate REQNOs in the report scope (or key
  the join on `(HN, REQNO)`). All map keys use `.strip()` normalization,
  mirroring `run_llm_leg.py:1976` and `reserved_units.py:31`.
- Add bucket-level counts to the summary, COMPUTED from the rows at build
  time — never hardcoded (the current artifact has 43 returned + 177
  exempt of 300; earlier drafts citing 87/355 were counting string
  occurrences in review.html, not rows). Wording per section 6 Q4. Add a
  glossary line explaining the annotation is informational, not a billed
  verdict (extend the flag-gated `msbos_glossary_html` block at
  `build_review.py:1842-1848`).

Tests: extract the annotation computation into small pure helpers and
unit-test them (returns-terminal RBC over / within / T/S-crossmatched /
ambiguous-op / no-planned-op; non-returns row → blank; seam-off → columns
absent). Follow `tests/unit/test_pilot_msbos_reservation.py` conventions.

### T2 — Deterministic leg: platelet mirror (component-aware schema)

Same computation as T1 for platelet returns-terminal rows: key
`(hn, reqno, ComponentFamily.PLATELET)`, judge via
`evaluate_platelet_reservation` with
`procedure_groups=msbos_reference.groups_for(planned)` and the leg's existing
`plt_result.value_k_ul` (recipe: `run_llm_leg.py:1461-1492`).

Codex blocker folded: the seven RBC-shaped `msbos_*` columns CANNOT express
a `PlateletReservationDecision` — it carries category, pre-op count,
`over_above_per_ul`, and `clinician_signed`, with a different reason
vocabulary (`platelet_evaluate.py:24,61`: per-category `within_*`/`over_*`,
`no_reserved_units`, `missing_pre_op_count`, `uncategorised_procedure`,
`ambiguous_category`, ...). The schema is component-aware: shared columns
(`msbos_reserved_units`, `msbos_is_over`, `msbos_reason`,
`msbos_reference_hash`) plus platelet-specific columns (e.g.
`msbos_plt_category`, `msbos_plt_count_k_ul`, `msbos_plt_over_above_per_ul`,
`msbos_plt_clinician_signed`; RBC-only columns stay blank on platelet rows
and vice versa). The review.html display mapping must enumerate the platelet
reason vocabulary explicitly — no fall-through to the RBC mapping. Exact
column set is frozen at /to-spec. Never-guess rules carry over unchanged
(missing count with reserved units → review reason; reserved <= 0 never
over), and the T1 join-miss guard applies with the PLATELET key.

### T3 — Library store annotation (forward-looking, small)

Codex blocker folded: this ticket touches BOTH marker builders, with
component-specific payload schemas — platelet deterministic rows persist via
`_platelet_marker_call` (`pipeline.py:727`, invoked at `:778`), NOT
`_deterministic_marker_call`.

- `_deterministic_marker_call` (`pipeline.py:623`): when
  `feature_flags.MSBOS_RESERVATION_ENABLED`, the classification is a returns
  terminal, and `context.reservation_decision` is not None, embed
  `"reservation_annotation": {reserved_units, msbos, recommended_units,
  is_over, reason, resolved_icd9, note_resolved, reference_hash}` in the
  marker's `response_json`.
- `_platelet_marker_call` (`pipeline.py:727`): same gating against
  `context.platelet_reservation_decision`, embedding the platelet shape
  `{reserved_units, category, pre_op_count_k_ul, over_above_per_ul,
  is_over, reason, clinician_signed, reference_hash}`.

No new marker key, no AuditRow change, no `final_classification` change in
either builder.

Note: this is inert in today's pilot flow (neither pilot leg persists
returns-terminal rows to the store; the library batch producer for
`reservation_decision` is the known deferred wiring item). It exists so any
future store-persisted returns row carries the annotation for free. Unit
tests in the style of `tests/unit/test_over_reservation_persist.py`:
annotation present flag-on with decision, absent flag-off, absent when
decision is None, absent on non-returns classifications;
`reconcile_returns` unaffected. Use STRICT `response_json == {...}`
equality (the pattern at `test_over_reservation_persist.py:319-322`), not
loose `.get()` checks, so future payload-shape drift fails loudly. Forward
obligation to record in the ticket: the store is idempotent on
`(run_id, audit_id, code_version)` via `pipeline_config.code_version`
(caller-supplied, `pipeline.py:402`) — when the deferred library
`run_pipeline` wiring eventually ships a producer for these contexts, that
caller must bump its code_version or pre-T3 committed marker rows will
never pick up the new field on re-runs.

## 4. Acceptance criteria

- T0: regenerating review.html from an unchanged report.csv differs ONLY in
  the summary-table markup (server-side pills + mismatch class + overflow
  wrapper), the css block, and the kbd-hint button; every verdict pill in
  the legend, summary table, and verdict boxes has a defined background and
  text color; the pill-wrapping JS block is gone; `report.csv` and all store
  writes untouched.
- T1-T3 with `BBA_PILOT_MSBOS_RESERVATION=0` (or library flag off):
  `report.csv`, `review.html`, and all store writes byte-identical to the
  post-T0 output.
- Flag-on: every returns-terminal row in `report.csv` carries populated
  `msbos_*` columns (or an explicit unresolved reason); all other rows carry
  blanks; no row changes classification, bucket, or scoring status.
- `review.html` returned/exempt buckets show the per-case annotation, the
  summary table shows the MSBOS column (flag-on runs only: pill for
  returns-terminal rows, em-dash otherwise; flag-off runs omit the column
  and all MSBOS markup entirely), and the summary shows the bucket counts
  computed from the rows.
- `msbos_reservation_pilot_report.py` / `reconcile_returns` output unchanged
  (no new markers, no double-fire).
- Full test suite green, ruff clean (`ruff check` AND `ruff format --check`),
  no new dependencies.

## 5. review.html style-audit findings (2026-07-17, /impeccable audit)

Reference for T0. Audited artifact: `/tmp/bba_mini/review.html` (generated by
`scripts/pilot/build_review.py`). Health score 15/20 (Good): A11y 3,
Performance 4, Theming 2, Responsive 2, Anti-patterns 4. The OKLCH palette
itself is clean and matches `docs` DESIGN.md tokens — all measured text/pill
contrast pairs are 5.4:1 to 16.9:1, comfortably past WCAG AA; no AI-slop
tells (no gradients, side-stripes, or glassmorphism). The defects are
component-state gaps, not palette problems:

- **[P1] 221 colorless verdict pill occurrences.**
  `cls-returned_not_transfused` (43 class-attr occurrences),
  `cls-periop_transfusion_exempt` (177), `cls-preop_over_reservation` (1)
  are emitted by the verdict boxes (`build_review.py:1272`) — and the two
  returns classes also by the legend (`:1826-1829`); the over-reservation
  class appears in verdict boxes only — but none is defined in the css
  string. The base `.cls` rule gives bold text with padding and rounded
  corners and NO background/color. The two largest buckets in the report
  are the broken ones.
- **[P1] Summary-table pills derive CSS classes from display text.** The JS
  at `build_review.py:1937-1957` normalizes the rendered label, producing
  undefined garbage classes containing em-dashes and parentheses for rows
  where the LLM ran with an off-vocabulary display label (e.g.
  `PREOP_RESERVATION_UNCONFIRMED`). Rows whose LLM cell is a sentinel
  ("—" / "(...)") — which includes every returns-terminal row — are skipped
  entirely and render as plain unwrapped text, no pill at all. The
  `cells[8]`/`cells[9]` indexes are fragile to column insertion before
  index 8 (appending after would not break them, but the class derivation
  is broken regardless). Root fix is server-side rendering (raw
  classification and mismatch are both known at build time).
- **[P2] Summary table can force page-level horizontal scroll** on narrow
  viewports (10 columns, no `overflow-x: auto` wrapper); clinical reviewers
  are a tablet audience.
- **[P2] Off-system one-off button.** kbd-hint Dismiss button uses inline
  `font-size:0.7rem` (system smalls are 0.75rem) at `build_review.py:1858`.
- **[P2] No `:focus-visible` styles** despite advertised keyboard shortcuts.
- **[P3] Phantom webfonts.** The css names Inter / JetBrains Mono / Noto
  Sans Thai but the standalone file loads no fonts, so rendering silently
  falls back per machine (macOS: SF Pro / Menlo / Thonburi). Acceptable for
  an internal artifact; do not add network font loads (the report must stay
  offline-safe). Optional cleanup only: drop the phantom first names and
  design for the system stack honestly.
- **[P3] `.case.is-reviewed { opacity: 0.6 }`** drops effective contrast of
  muted text below AA while dimmed; deliberate de-emphasis of finished
  work — leave unless reviewers complain.
- Dark mode: absent by design in this artifact; DESIGN.md dark tokens apply
  to the docs site, not the pilot report. Not a T0 item.

## 6. Grill rulings (ALL RESOLVED 2026-07-17 — plan is spec-ready)

- **Q1 — RESOLVED (a), user ruling 2026-07-17: skip note disambiguation in
  the deterministic leg.** Conflicting codes annotate "unresolved"
  (`ambiguous_code`); plain `evaluate_reservation`, never
  `evaluate_reservation_with_notes`; `msbos_note_resolved` column dropped.
  Windowed-notes upgrade via shared bundle code recorded as an optional
  follow-up in section 7.
- **Q2 — RESOLVED (user ruling, 2026-07-17): judge reservation quantity on
  BOTH buckets; report transfused units on exempt rows.** The MSBOS
  quantity judgment is an ordering-time question independent of the
  transfusion outcome; peri-op use does not justify the reservation size.
  The transfusion INDICATION stays unjudged on exempt rows (anesthesiologist
  territory — the reason the bucket is exempt). Exempt rows display
  transfused/returned unit counts from the existing returns columns. Count
  lines per-bucket, never merged. (This supersedes the clinical reviewer's
  suppress/neutralize recommendation — the misleading-committee risk is
  mitigated by the glossary split and the Q4 fact-first wording instead.)
- **Q3 — RESOLVED (user ruling, 2026-07-17): warn/amber for over-tariff
  annotation pills.** Three-tier grammar locked: err/red is reserved for
  scored verdicts (including the billed PREOP_OVER_RESERVATION pill), warn/
  amber for the informational over-tariff annotation (both buckets),
  ok/green for within-tariff, neutral for the returns-terminal verdict
  pills themselves. Unresolved/lookup-miss annotation reasons also render
  warn/amber.
- **Q4 — RESOLVED (user ruling, 2026-07-17): fact-first wording set (a).**
  Summary pill strings: "3 vs G/M 2" (over, amber), "within" (green),
  "T/S; 3u reserved" (amber), "code unresolved" / "op unresolved" /
  "unlinked" (amber, for `ambiguous_code`+`unresolved_code` /
  `operation_unresolved`-family / `reservation_lookup_miss` respectively),
  "—" for non-returns rows. Bucket case line: "Reserved 3; MSBOS tariff
  G/M 2", exempt rows append "; 2 transfused, 2 returned". Count lines,
  per bucket with visible unresolvable remainder: "Returned (43): N above
  tariff / M within / K unresolved" and "Peri-op exempt (177): N above
  tariff / M within / K unresolved" (all counts computed at build time).
  Glossary defines "above tariff" as "reserved units exceeded the elective
  MSBOS tariff for the resolved planned procedure code; does not account
  for anticipated hemorrhage, case cancellation, or emergency status — the
  schedule data cannot see these" (cancelled-op confounder is structural:
  `_planned_op_icd9` picks the nearest UPCOMING op and never verifies it
  happened, `run_llm_leg.py:295-307`), plus the Q2 quantity-vs-indication
  split for exempt rows. The clinical reviewer's note-provenance surfacing
  is MOOT under the Q1 ruling (no note-inferred annotations exist in this
  leg).

## 7. Out of scope (record as follow-ups, do not implement)

- Committee pilot-report section for these annotations — blocked on a store
  producer that persists returns-terminal rows with reservation decisions
  (the deferred library `run_pipeline` wiring).
- Precedence flip (`PREOP_OVER_RESERVATION` outranking returns) — rejected.
- MSBOS annotation for non-returns rows in the deterministic leg — the LLM
  leg's billed path owns those.
- Backfill of existing runs; the annotation appears on the next pilot run.
- Note disambiguation for the deterministic-leg annotation (Q1 ruling:
  skipped). If the unresolved rate proves too high in practice, a follow-up
  may reuse the evidence-bundle building code to reproduce the billed
  path's windowed/capped/ranked note selection — do not shortcut with a
  bare date window or `vitals_notes_for`.
