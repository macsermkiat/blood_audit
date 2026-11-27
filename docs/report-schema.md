# Monthly RBC report schema

`bba.report_generator` writes monthly RBC summaries from a committed audit run.
The CLI builder selects `component = "red_cell"` before aggregation. Platelet
rows do not contribute to RBC totals or rates.

The output contains five committee CSVs, one own-view CSV per requested
physician, a committee PDF, and `provenance.json`. The number of files therefore
depends on the number of requested physicians.

`bba report` counts one row per REQNO and attributes it through
`AuditRow.reqno`. The research pilot writes one row per issue event; its files
are described in [Pilot issue-event outputs](#pilot-issue-event-outputs). Do
not render `bba report` from a pilot audit store. It would credit every
`fresh_order` event to the doctor of the REQNO it was issued under, with no
error.

## File layout

```
<output_dir>/
  hospital_trend.csv                              # committee
  ward_scorecard.csv                              # committee
  indication_distribution.csv                     # committee
  cohort_exception.csv                            # committee
  pipeline_health.csv                             # committee
  physician_own_view_<physician_id>.csv           # one per requested physician
  report_<YYYY-MM-DD>.pdf                         # committee PDF
  provenance.json                                # per-RBC-row provenance
```

The PDF filename embeds the first-of-month date (`report_2026-05-01.pdf`).

Each `physician_own_view_<physician_id>.csv` contains one physician's data row
with anonymous peer quartiles. The requested physicians come from
`ReportInputs.physician_ids_for_own_view`. The committee PDF also includes a
physician table. Output-directory permissions and distribution controls must
match these audiences.

## Run-level provenance summary

Each CSV data row carries six summary fields. Empty sections receive a synthetic
row with the same fields. The PDF repeats the summary on each page.

| Column | Meaning |
| --- | --- |
| `policy_version` | Policy identifier shared by the run's RBC rows. |
| `model_id` | A shared model identifier, or a sorted JSON list of distinct identifiers. Rule-only rows use `deterministic`. Other non-provider paths use their own marker identifiers. |
| `redactor_version` | Applied redactor version. A mixture of applied and unapplied rows is represented as a JSON list including `not-applied`. |
| `redactor_model_sha` | Applied redactor checkpoint identifier. Unapplied rows use `unavailable`. |
| `prompt_hash` | A shared prompt hash, or a digest of sorted `(audit_id, prompt_hash)` pairs when prompts differ. |
| `evidence_bundle_hash` | Digest of sorted `(audit_id, evidence_bundle_hash)` pairs from RBC source rows. This is not an evidence-schema hash. |

The builder validates code version, policy, local month, and applied-redactor
identity across the full RBC set before excluding non-scorable rows. Distinct
model and prompt identifiers are preserved rather than attributed to one model.
Deterministic and rejected rows do not claim that a provider or redactor ran.

## Per-row provenance

`provenance.json` contains one object for each RBC source row, including rows
excluded from scoring. Each object has these fields:

- `audit_id`
- `run_id`
- `code_version_slug`
- `model_id`
- `policy_version`
- `prompt_hash`
- `evidence_bundle_hash`
- `redactor_version`
- `redactor_model_sha`

Aggregate CSV rows contain the run summary, not the complete provenance of each
underlying order. The sidecar supplies that mapping. Retained source snapshots,
configuration, code, and response artifacts are also needed to trace a stored
answer. A new provider call is not guaranteed to reproduce that answer.

## Month bucketing

`ReportInputs.month` is a `date` at day-1 of the month. The filter bucket
is half-open `[month_local_start, next_month_local_start)` interpreted in
Asia/Bangkok (PRD §"Tz-aware throughout": datetimes stored UTC, rendered
Asia/Bangkok). An order at 23:00 Bangkok on the 31st belongs to the
month it was placed locally, not to the UTC-month it happens to fall in.

## Classification projection and exclusions

The audit store persists nine `final_classification` values, but the report
counts are computed over the four reportable labels
(`APPROPRIATE` / `INAPPROPRIATE` / `NEEDS_REVIEW` / `INSUFFICIENT_EVIDENCE`).
`bba.report_generator.builder` reconciles the difference before the counts
below are tallied:

- **Excluded before projection (non-scorable).** `RETURNED_NOT_TRANSFUSED`
  and `PERIOP_TRANSFUSION_EXEMPT` rows are dropped from the report input
  before aggregation. They are absent from `total_orders` and every count
  and rate. The default projector rejects them if they reach projection.
- **`PREOP_OVER_RESERVATION` → `INAPPROPRIATE`** (ticket #162). It counts
  as an inappropriate-side finding in the tallies below; the aggregate
  models additionally track an `over_reservation_count` field that the CSV
  writer does not yet serialize as a column.
- **Requires an injected projector.** `POTENTIALLY_INAPPROPRIATE` and
  `PREOP_RESERVATION_UNCONFIRMED` have no default mapping. The caller must
  supply a `classification_projector`, or the builder rejects the run.

## CSV section schemas

### `hospital_trend.csv`

One row for the report month. The builder rejects a run spanning local months.

| Column                   | Type    | Notes                                |
|--------------------------|---------|--------------------------------------|
| `month`                  | date    | First-of-month, ISO 8601.            |
| `total_orders`           | int     | Scorable RBC orders after component filtering, exclusions, and projection.     |
| `appropriate`            | int     | Count where `final_classification = APPROPRIATE`. |
| `inappropriate`          | int     | Count where `final_classification = INAPPROPRIATE`. |
| `needs_review`           | int     | Count where `final_classification = NEEDS_REVIEW`. |
| `insufficient_evidence`  | int     | Count where `final_classification = INSUFFICIENT_EVIDENCE`. |
| `inappropriate_rate`     | float   | `inappropriate / total_orders`, in `[0, 1]`. |

### `ward_scorecard.csv`

One row per distinct `ward_id`, sorted ascending.

| Column                   | Type   | Notes                                  |
|--------------------------|--------|----------------------------------------|
| `ward_id`                | string | Resolved grouping identifier. The supported CLI uses the ordering doctor's department text, or a sentinel. |
| `total_orders`           | int    |                                        |
| `appropriate`            | int    |                                        |
| `inappropriate`          | int    |                                        |
| `needs_review`           | int    |                                        |
| `insufficient_evidence`  | int    |                                        |
| `inappropriate_rate`     | float  |                                        |

### `physician_own_view_<physician_id>.csv`

One file per `physician_id` in `ReportInputs.physician_ids_for_own_view`;
each file contains exactly one data row.
Peer benchmarks use all physicians with at least one scorable RBC order in
the report month, including physicians whose own-view files were not requested.

Each own-view row contains the physician's absolute numbers and aggregate
peer quartiles. It has no field for another physician's individual rate.

| Column                                | Type   | Notes                                                  |
|---------------------------------------|--------|--------------------------------------------------------|
| `physician_id`                        | string | Resolved physician identifier. The supported CLI uses the ordering-doctor code from `DCTREQ`, or a sentinel. |
| `own_total`                           | int    |                                                        |
| `own_inappropriate_rate`              | float  | In `[0, 1]`.                                           |
| `peer_median_inappropriate_rate`      | float  | Median across all physicians in the month.             |
| `peer_p25_inappropriate_rate`         | float  | 25th-percentile rate (linear interpolation).           |
| `peer_p75_inappropriate_rate`         | float  | 75th-percentile rate (linear interpolation).           |

### `indication_distribution.csv`

One row per distinct `indication_code` that appears at least once. An order
with N indication codes contributes 1 to each code's count (multi-indication
orders are not split). Sorted by `total_orders` descending, then
`indication_code` ascending.

| Column            | Type   | Notes                                                       |
|-------------------|--------|-------------------------------------------------------------|
| `indication_code` | string | Indication identifier (e.g., `anemia_symptomatic`).         |
| `total_orders`    | int    | `>= 1`.                                                     |
| `share`           | float  | `total_orders / total_rows_in_month`; sums to >1.0 if any order had multiple indications. |

### `cohort_exception.csv`

One row per distinct `cohort_applied`. The `"default"` cohort is included so
the breakdown sums to the total.

| Column               | Type   | Notes                                              |
|----------------------|--------|----------------------------------------------------|
| `cohort_applied`     | string | Cohort identifier (`default`, `cardiac_surgery`, ...). |
| `total_orders`       | int    |                                                    |
| `inappropriate`      | int    |                                                    |
| `inappropriate_rate` | float  | In `[0, 1]`.                                       |

### `pipeline_health.csv`

Single-row summary of the month's operational health.

| Column                          | Type  | Notes                                                                                       |
|---------------------------------|-------|---------------------------------------------------------------------------------------------|
| `total_orders`                  | int   |                                                                                             |
| `classified_orders`             | int   | Rows where `final_classification ∈ {APPROPRIATE, INAPPROPRIATE}`.                            |
| `needs_review_count`            | int   | Rows where `final_classification = NEEDS_REVIEW`, **or** `final_classification ∈ {APPROPRIATE, INAPPROPRIATE}` **and** `needs_human_review` flag is set. INSUFFICIENT_EVIDENCE rows are excluded even when flagged. They count only in `insufficient_evidence_count`. |
| `needs_review_rate`             | float | `needs_review_count / total_orders`, in `[0, 1]`.                                            |
| `insufficient_evidence_count`   | int   | Rows where `final_classification = INSUFFICIENT_EVIDENCE`. PRD §"Documentation absence ≠ INAPPROPRIATE": its own bucket. |
| `insufficient_evidence_rate`    | float | `insufficient_evidence_count / total_orders`, in `[0, 1]`.                                   |

## Encoding conventions

* Line terminator: `\n` (locked for byte-stable golden snapshots).
* Encoding: UTF-8, no BOM.
* Floats render with trailing zeros stripped but at least one digit after the
  decimal point (`0.5`, `0.0`, never `0`).
* Booleans render as lowercase `true` / `false` (no section currently emits
  bool cells, but the writer's contract supports them for forward
  compatibility with sentinel / dashboard sections).

## Pilot issue-event outputs

The research pilot (`scripts/pilot/`) audits one issue event per row (#276).
The [issue-event design record](issue-event-plan.md) defines events, linkage,
and identity. Both legs build events through `scripts/pilot/_pilot_events.py`.

### `report.csv`

`run_pipeline.py` writes one row per issue event, plus one REQNO-level row for
each REQNO with no event rows. Existing columns keep their names and now
describe the event. `dispense_datetime_local` (the PICK time) is gone;
`use_datetime_local` stays as a display column. Seam columns (`returns_*`,
`declared_use_*`, `msbos_*`) are appended when their flags are on, as before.

| Column | Meaning |
| --- | --- |
| `audit_id` | Event id: `build_event_audit_id(hn, reqno, unit_key)` of the event's first unit. On a REQNO-level row it equals `parent_audit_id`. |
| `parent_audit_id` | `build_audit_id(hn, reqno)`, the REQNO's id. It joins v3 rows to v1 and v2 rows. |
| `reqno` | The REQNO the units were issued under. |
| `event_seq` | 1-based event number within the REQNO. Empty on a REQNO-level row. |
| `events_in_reqno` | Number of events the REQNO issued. On an excluded REQNO it is the measured count, 0 when the REQNO is not a linkage candidate. |
| `order_datetime_utc` | Placement time of the REQNO's own order. |
| `issue_datetime_local` | Event anchor: the first unit's `PAYDATE` and `PAYTIME`, Bangkok time. |
| `issue_units` | Physical units in the event. |
| `issue_units_returned` | Units with terminal status 3. Status-7 units are not counted here. |
| `issue_unit_statuses` | Terminal `UNITSTAT` per unit in issue order, joined by `\|`. |
| `units_reserved_not_issued` | REQNO units with no issue time (crossmatch holds). |
| `unit_family_mismatch` | `True` when a unit code names another product family than the order, or no recognised family. |
| `linked_reqno` | The order the event is judged against and credited to. |
| `linked_order_datetime_local` | Placement time of the linked order. |
| `linked_order_status` | `BDVSTST` of the linked order. |
| `linkage_rule` | `own`, `fresh_order`, or `standing_order`. |
| `linked_reqno_latest_before` | The order the #276 rule would link: the latest candidate placed at or before the issue. |
| `order_after_issue` | `True` when the linked order was placed after the issue. |
| `attributed_doctor_differs` | `True` when the linked order's `DCTREQ` differs from the REQNO's. Empty when either is unknown. Read from `BBA_BDVST_CSV`. |
| `order_to_issue_hours` | Issue time minus linked order time, in hours. |
| `declared_use_applies` | Whether the linked order's declared use counts for this event (owner decision 8.15 (c)). |
| `prior_event_cutoff_local` | Issue time of the #249 prior transfusion that starts the 24 h Hb minimum. Empty when none. |
| `linked_order_emrtype`, `linked_order_bdlevelemr` | Distinct `EMRTYPE` and `BDLEVELEMR` codes of the linked order. Display only. |
| `evidence_anchor_reason` | Red-cell rows only. `issue_reanchor` when the linked order was placed 24 h or more before the issue, else `issue_datetime`. |
| `evidence_anchor_datetime_local` | Red-cell rows only. Equals `issue_datetime_local`. |
| `hb_anchor_reason` | `issue_datetime`, or a fallback-ladder reason when no Hb exists at the issue. |

A REQNO-level row has `classification` = `excluded` and `rationale` set to
the `audit_orders` exclusion reason or `no_issue_record`. It fills only
`audit_id`, `parent_audit_id`, `reqno`, `event_seq` (empty),
`events_in_reqno`, `units_reserved_not_issued`, `classification`, and
`rationale`. It carries no `component`.

With the returns seam on, `returns_disposition` and the `returns_units_*`
counts describe the event. `returns_ordered_unit_amount` and
`returns_ledger_complete` describe the REQNO. `returned_blood_datetime_local`
reads only the event's status-3 units.

`PREOP_OVER_RESERVATION` and the MSBOS columns are evaluated once per REQNO and
written on event 1 only. An all-returned event 1 is `RETURNED_NOT_TRANSFUSED`
ahead of the MSBOS review and the declared-use exemption.

### `events_manifest.csv`

`run_pipeline.py` writes one row per event with the columns `HN`, `REQNO`,
`AN`, `component`, `audit_id`, and `event_seq`. `component` is `red_cell` or
`platelet`; `sample_manifest.csv` uses `rbc` for the same family.
`build_review.py` requires this file and exits without it.

### `llm_report.json`

`run_llm_leg.py` writes one entry per event that reached the LLM leg. Entries
merge by `audit_id`. Each entry carries `audit_id` and `reqno`, the existing
result fields, and these event fields: `parent_audit_id`, `event_seq`,
`events_in_reqno`, `linked_reqno`, `linkage_rule`,
`linked_order_datetime_local`, `issue_datetime_local`, `issue_units`,
`issue_units_returned`, `order_to_issue_hours`, `declared_use_applies`, and
`prior_event_cutoff_local`.

The store's `AuditRow.order_datetime` holds the event's issue time, not the
order time. Read the linked order time from `linked_order_datetime_local`.

### `compare_runs.py` outputs

`compare_runs.py <v2_dir> <v3_dir>` writes two files into the v3 folder.

- `compare_v2_v3.csv` has one row per v2 REQNO found in the v2 `sample_manifest.csv`, with the columns `reqno`,
  `v2_final`, `v2_hb`, `v3_events`, `v3_finals`, `v3_hbs`, and `v3_linkage`.
  The `v3_*` lists join event values with `|` in `event_seq` order.
- `compare_linkage_audit_ids.txt` is a comma-separated list of event
  `audit_id`s to re-ask if the owner changes the linkage default. See the
  [pilot README](../scripts/pilot/README.md#compare-a-v3-run-with-v2).

## Implementation references

- [`builder.py`](../src/bba/report_generator/builder.py) selects rows, validates run metadata, projects classifications, and constructs provenance.
- [`aggregate.py`](../src/bba/report_generator/aggregate.py) computes section counts and rates.
- [`csv_writer.py`](../src/bba/report_generator/csv_writer.py) defines emitted headers and formatting.
- [`generator.py`](../src/bba/report_generator/generator.py) writes CSVs, the PDF, and `provenance.json`.
