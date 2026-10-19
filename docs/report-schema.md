# Monthly RBC report schema

`bba.report_generator` writes monthly RBC summaries from a committed audit run.
The CLI builder selects `component = "red_cell"` before aggregation. Platelet
rows do not contribute to RBC totals or rates.

The output contains five committee CSVs, one own-view CSV per requested
physician, a committee PDF, and `provenance.json`. The number of files therefore
depends on the number of requested physicians.

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

## Implementation references

- [`builder.py`](../src/bba/report_generator/builder.py) selects rows, validates run metadata, projects classifications, and constructs provenance.
- [`aggregate.py`](../src/bba/report_generator/aggregate.py) computes section counts and rates.
- [`csv_writer.py`](../src/bba/report_generator/csv_writer.py) defines emitted headers and formatting.
- [`generator.py`](../src/bba/report_generator/generator.py) writes CSVs, the PDF, and `provenance.json`.
