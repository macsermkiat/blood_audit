# Monthly report schema (issue #28)

`bba.report_generator` emits six CSV files plus one PDF for every monthly run.
This doc is the contract between the report writer, the committee dashboard
preview, and the downstream consumers that grep/join the CSVs.

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
```

The PDF filename embeds the first-of-month date (`report_2026-05-01.pdf`).

**Per-physician separation.** The physician own-view does **not** ship as a
single committee CSV because that file would expose every physician's own
rate to every recipient. Instead the generator emits one
`physician_own_view_<physician_id>.csv` per physician requested in
`ReportInputs.physician_ids_for_own_view`. Each file contains exactly one
data row plus the reproducibility footer; distributing the file to that
physician therefore never leaks another physician's rate. PRD user story
#10: "comparing my RBC ordering against peer-anonymous benchmarks ...
without being publicly identified."

The committee PDF still renders a physician-own-view table for the
committee's internal review.

## Reproducibility footer

Every CSV data row (and the synthetic row emitted for an empty section)
carries six additional columns at the end:

| Column                 | Meaning                                                                            |
|------------------------|------------------------------------------------------------------------------------|
| `policy_version`       | KCMH PR-17.x policy revision that drove the deterministic classifier.              |
| `model_id`             | Snapshot-pinned Anthropic model ID that produced the LLM classification.           |
| `redactor_version`     | Version of the `macsermkiat/thai-medical-deid` redactor that processed the bundle. |
| `redactor_model_sha`   | SHA of the redactor's HuggingFace model checkpoint.                                |
| `prompt_hash`          | SHA of the system prompt + few-shot bundle the LLM call used.                      |
| `evidence_bundle_hash` | SHA of the canonical evidence-bundle schema the run used.                          |

The ticket scope (issue #28) names only the first three; the rest follow
PRD §"Output schema" so the reproducibility chain remains intact for a
six-month-later audit (PRD §"Reproducibility = we have the original answer").

Stamping the footer **per row** (rather than only in a trailing line) keeps the
reproducibility chain intact under `grep`, `awk`, or partial-section extracts.
The PDF carries the same six identifiers on every page.

## Month bucketing

`ReportInputs.month` is a `date` at day-1 of the month. The filter bucket
is half-open `[month_local_start, next_month_local_start)` interpreted in
Asia/Bangkok (PRD §"Tz-aware throughout": datetimes stored UTC, rendered
Asia/Bangkok). An order at 23:00 Bangkok on the 31st belongs to the
month it was placed locally, not to the UTC-month it happens to fall in.

## CSV section schemas

### `hospital_trend.csv`

One row per month covered by the run (Phase 1 emits a single row).

| Column                   | Type    | Notes                                |
|--------------------------|---------|--------------------------------------|
| `month`                  | date    | First-of-month, ISO 8601.            |
| `total_orders`           | int     | All audited orders in the month.     |
| `appropriate`            | int     | Count where `final_classification = APPROPRIATE`. |
| `inappropriate`          | int     | Count where `final_classification = INAPPROPRIATE`. |
| `needs_review`           | int     | Count where `final_classification = NEEDS_REVIEW`. |
| `insufficient_evidence`  | int     | Count where `final_classification = INSUFFICIENT_EVIDENCE`. |
| `inappropriate_rate`     | float   | `inappropriate / total_orders`, in `[0, 1]`. |

### `ward_scorecard.csv`

One row per distinct `ward_id`, sorted ascending.

| Column                   | Type   | Notes                                  |
|--------------------------|--------|----------------------------------------|
| `ward_id`                | string | Hashed ward identifier.                |
| `total_orders`           | int    |                                        |
| `appropriate`            | int    |                                        |
| `inappropriate`          | int    |                                        |
| `needs_review`           | int    |                                        |
| `insufficient_evidence`  | int    |                                        |
| `inappropriate_rate`     | float  |                                        |

### `physician_own_view_<physician_id>.csv`

One file per `physician_id` in `ReportInputs.physician_ids_for_own_view`;
each file contains exactly one data row.
Peer benchmarks are quartile statistics computed over **all** physicians with
at least one order in the month — not only the requested subset — so a small
subset cannot mis-report the population distribution.

The "own-data only" property is structural: each row carries the physician's
own absolute numbers plus aggregate peer quartiles. There is no field that
exposes another physician's individual orders or rate.

| Column                                | Type   | Notes                                                  |
|---------------------------------------|--------|--------------------------------------------------------|
| `physician_id`                        | string | Hashed physician identifier.                           |
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
| `needs_review_count`            | int   | Rows where `final_classification = NEEDS_REVIEW`, **or** `final_classification ∈ {APPROPRIATE, INAPPROPRIATE}` **and** `needs_human_review` flag is set. INSUFFICIENT_EVIDENCE rows are excluded even when the flag is set — they live in `insufficient_evidence_count` only, so a documentation-absence spike is not double-counted as an LLM-review spike. |
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
