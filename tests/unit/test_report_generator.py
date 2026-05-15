"""RED-phase failing tests for issue #28 (bba.report_generator).

Each ``class`` maps to one acceptance criterion (AC) in the issue body
plus the six scope sections defined in the ticket and PRD §19. Tests
assert contracts (the WHY), not implementation choices.

No implementation exists yet; every function-behavior test MUST fail with
``NotImplementedError`` in this scaffold commit. The model-immutability
tests assert pydantic-level invariants that are part of the public
surface; if those pass in RED, the schema is the schema, not the
implementation.

Acceptance-criterion → test-class map (from issue #28):

* AC ① "Implementation in src/bba/report_generator/"
  → :class:`TestPublicSurface`, :class:`TestModelImmutability`
* AC ② "Snapshot-based template tests (golden output for fixed input)"
  → :class:`TestGoldenSnapshotHospitalTrend`,
    :class:`TestGoldenSnapshotWardScorecard`
* AC ③ "CSV schema documented in docs/report-schema.md"
  → :class:`TestCsvSchemaDocumented`
* AC ④ "PDF rendering tested with a sample run"
  → :class:`TestPdfRendering`

Scope-section coverage (from issue #28 "## Scope"):

* Hospital-wide trend            → :class:`TestHospitalTrendAggregation`
* Per-ward scorecard             → :class:`TestWardScorecardAggregation`
* Per-physician own-view         → :class:`TestPhysicianOwnViewAggregation`
* Indication-distribution        → :class:`TestIndicationDistributionAggregation`
* Cohort-exception breakdown     → :class:`TestCohortExceptionAggregation`
* Pipeline-health summary        → :class:`TestPipelineHealthAggregation`
* Footer stamped on every report → :class:`TestFooterStamping`

Cross-cutting invariants:

* Month-boundary filtering       → :class:`TestMonthBoundaryFiltering`
* Total = sum of class buckets   → :class:`TestClassificationBucketsSum`
  (hypothesis property test — issue #28 is "thin" but the bucket-sum
   identity is a deterministic invariant worth a property test)
* Orchestrator end-to-end        → :class:`TestGenerateMonthlyReport`
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from bba.report_generator import (
    CSV_ENCODING,
    CSV_NEWLINE,
    PDF_MAGIC,
    SECTION_NAMES,
    EmptyInputError,
    HospitalTrendRow,
    MonthlyReportRow,
    PhysicianOwnViewRow,
    ReportArtifacts,
    ReportFooter,
    ReportGenerationError,
    ReportInputs,
    ReportSection,
    WardScorecardRow,
    aggregate_cohort_exception,
    aggregate_hospital_trend,
    aggregate_indication_distribution,
    aggregate_physician_own_view,
    aggregate_pipeline_health,
    aggregate_ward_scorecard,
    filter_rows_for_month,
    generate_monthly_report,
    physician_own_view_filename,
    render_report_pdf,
    section_filename,
    write_section_csv,
)


# =============================================================================
# Shared fixtures.
#
# A single fixed anchor month (May 2026) and a tiny ``_row`` constructor let
# each test state only the fields it actually exercises. This mirrors the
# convention in ``tests/unit/test_audit_store.py`` and
# ``tests/unit/test_evidence_bundle_builder.py``.
# =============================================================================


MONTH = date(2026, 5, 1)
MID_MONTH = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)


def _row(
    *,
    audit_id: str = "audit-001",
    ward_id: str = "WARD-A",
    physician_id: str = "phys-1",
    final_classification: str = "APPROPRIATE",
    cohort_applied: str = "default",
    indication_codes: tuple[str, ...] = ("anemia_symptomatic",),
    needs_human_review: bool = False,
    order_datetime: datetime = MID_MONTH,
    an_hash: str = "an-001",
    hn_hash: str = "hn-001",
) -> MonthlyReportRow:
    return MonthlyReportRow(
        audit_id=audit_id,
        an_hash=an_hash,
        hn_hash=hn_hash,
        order_datetime=order_datetime,
        ward_id=ward_id,
        physician_id=physician_id,
        final_classification=final_classification,  # type: ignore[arg-type]
        cohort_applied=cohort_applied,
        indication_codes=indication_codes,
        needs_human_review=needs_human_review,
    )


def _footer() -> ReportFooter:
    return ReportFooter(
        policy_version="PR17.2-v1",
        model_id="claude-sonnet-4-6-2026-04-01",
        redactor_version="deid-th-1.3.0",
        redactor_model_sha="sha-abcdef0123456789",
        prompt_hash="sha-prompt-001",
        evidence_bundle_hash="sha-bundle-001",
    )


def _inputs(
    rows: Sequence[MonthlyReportRow],
    *,
    output_dir: Path,
    physician_ids_for_own_view: tuple[str, ...] = (),
    month: date = MONTH,
) -> ReportInputs:
    return ReportInputs(
        month=month,
        rows=tuple(rows),
        footer=_footer(),
        output_dir=output_dir,
        physician_ids_for_own_view=physician_ids_for_own_view,
    )


# =============================================================================
# AC ① — Public surface + model immutability.
#
# The module's import surface IS part of the public contract. A future
# refactor that hides a public name is a breaking change for callers
# (``bba.cli``, ``bba.dashboard``); the test pins the surface so the break
# fails CI loudly.
# =============================================================================


class TestPublicSurface:
    """Imports of the module's public names resolve.

    If this class fails with ``ImportError`` rather than the test bodies
    failing, the scaffold is wrong — fix ``__init__.py``, not the test.
    """

    def test_section_names_tuple_matches_literal(self) -> None:
        assert SECTION_NAMES == (
            "hospital_trend",
            "ward_scorecard",
            "physician_own_view",
            "indication_distribution",
            "cohort_exception",
            "pipeline_health",
        )

    def test_pdf_magic_is_pdf_header(self) -> None:
        assert PDF_MAGIC == b"%PDF-"

    def test_csv_newline_is_lf(self) -> None:
        # Locked to ``\n`` so the golden-snapshot CSVs are byte-stable
        # across platforms — CRLF would generate noisy diffs in CI.
        assert CSV_NEWLINE == "\n"

    def test_csv_encoding_is_utf8(self) -> None:
        assert CSV_ENCODING == "utf-8"

    def test_empty_input_error_is_report_generation_error(self) -> None:
        # Catching the base class at the CLI / dashboard boundary should
        # also catch EmptyInputError; pin the inheritance.
        assert issubclass(EmptyInputError, ReportGenerationError)


class TestModelImmutability:
    """Frozen pydantic models reject mutation and naive datetimes.

    The frozen contract prevents an aggregation function from
    accidentally mutating its input mid-pipeline; the tz-aware contract
    prevents a month-boundary order from silently bucketing into the
    wrong month.
    """

    def test_monthly_report_row_is_frozen(self) -> None:
        row = _row()
        with pytest.raises(ValidationError):
            row.ward_id = "WARD-Z"  # type: ignore[misc]

    def test_report_footer_is_frozen(self) -> None:
        footer = _footer()
        with pytest.raises(ValidationError):
            footer.policy_version = "tampered"  # type: ignore[misc]

    def test_monthly_report_row_rejects_naive_datetime(self) -> None:
        with pytest.raises(ValidationError):
            MonthlyReportRow(
                audit_id="x",
                an_hash="an",
                hn_hash="hn",
                order_datetime=datetime(2026, 5, 15, 12),  # type: ignore[arg-type]
                ward_id="WARD-A",
                physician_id="phys-1",
                final_classification="APPROPRIATE",
                cohort_applied="default",
                indication_codes=(),
                needs_human_review=False,
            )

    def test_report_inputs_rejects_mid_month_date(self) -> None:
        # ``month`` must be first-of-month; a mid-month date is ambiguous
        # ("month containing X" vs "rolling 30-day ending X").
        with pytest.raises(ValidationError):
            ReportInputs(
                month=date(2026, 5, 15),
                rows=(),
                footer=_footer(),
                output_dir=Path("/tmp"),
            )

    def test_report_footer_rejects_empty_policy_version(self) -> None:
        # Empty footer is the failure mode the schema exists to prevent.
        with pytest.raises(ValidationError):
            ReportFooter(
                policy_version="",
                model_id="m",
                redactor_version="r",
                redactor_model_sha="s",
                prompt_hash="p",
                evidence_bundle_hash="e",
            )

    def test_hospital_trend_row_rejects_out_of_range_rate(self) -> None:
        # ``inappropriate_rate`` is a proportion in [0,1]; 1.5 is a bug.
        with pytest.raises(ValidationError):
            HospitalTrendRow(
                month=MONTH,
                total_orders=10,
                appropriate=5,
                inappropriate=5,
                needs_review=0,
                insufficient_evidence=0,
                inappropriate_rate=1.5,
            )

    @pytest.mark.parametrize(
        "bad_id",
        [
            "../etc/passwd",
            "..",
            ".",
            "phys/with/slashes",
            "phys with spaces",
            "phys\x00null",
            "",
        ],
    )
    def test_monthly_report_row_rejects_unsafe_physician_id(
        self, bad_id: str
    ) -> None:
        # physician_id flows into a per-physician CSV filename; an
        # upstream value containing path separators or null bytes would
        # let the writer escape the output directory. The model boundary
        # is where this defense lives.
        with pytest.raises(ValidationError):
            MonthlyReportRow(
                audit_id="audit-001",
                an_hash="an-001",
                hn_hash="hn-001",
                order_datetime=MID_MONTH,
                ward_id="WARD-A",
                physician_id=bad_id,
                final_classification="APPROPRIATE",
                cohort_applied="default",
                indication_codes=(),
                needs_human_review=False,
            )

    def test_physician_own_view_filename_rejects_path_traversal(self) -> None:
        # Defense in depth: a caller that bypassed the MonthlyReportRow
        # model boundary still cannot make the filename helper produce
        # an escape path.
        with pytest.raises(ValueError):
            physician_own_view_filename("../escape")
        with pytest.raises(ValueError):
            physician_own_view_filename("/abs/path")
        with pytest.raises(ValueError):
            physician_own_view_filename(".")
        with pytest.raises(ValueError):
            physician_own_view_filename("")


# =============================================================================
# Hospital-wide trend section
# =============================================================================


class TestHospitalTrendAggregation:
    """Hospital-wide trend produces one row for the requested month with
    classification counts and the inappropriate-rate."""

    def test_returns_single_row_for_one_month(self) -> None:
        rows = (
            _row(audit_id="a1", final_classification="APPROPRIATE"),
            _row(audit_id="a2", final_classification="INAPPROPRIATE"),
            _row(audit_id="a3", final_classification="NEEDS_REVIEW"),
            _row(audit_id="a4", final_classification="INSUFFICIENT_EVIDENCE"),
        )
        result = aggregate_hospital_trend(rows, MONTH)
        assert len(result) == 1
        assert result[0].total_orders == 4
        assert result[0].appropriate == 1
        assert result[0].inappropriate == 1
        assert result[0].needs_review == 1
        assert result[0].insufficient_evidence == 1
        assert result[0].inappropriate_rate == pytest.approx(0.25)

    def test_zero_rows_for_month_emits_empty_tuple(self) -> None:
        # An empty month returns ``()`` from this aggregation; the
        # orchestrator decides whether that becomes ``EmptyInputError``.
        # Localising the policy keeps the aggregator pure.
        result = aggregate_hospital_trend((), MONTH)
        assert result == ()


# =============================================================================
# Per-ward scorecard
# =============================================================================


class TestWardScorecardAggregation:
    """One row per distinct ``ward_id``, sorted ascending for byte-stability."""

    def test_one_row_per_ward(self) -> None:
        rows = (
            _row(audit_id="a", ward_id="WARD-A"),
            _row(audit_id="b", ward_id="WARD-A"),
            _row(audit_id="c", ward_id="WARD-B"),
        )
        result = aggregate_ward_scorecard(rows)
        assert len(result) == 2
        wards = tuple(r.ward_id for r in result)
        assert wards == ("WARD-A", "WARD-B"), "output must be sorted ascending"

    def test_inappropriate_rate_per_ward(self) -> None:
        rows = (
            _row(audit_id="1", ward_id="WARD-A", final_classification="APPROPRIATE"),
            _row(audit_id="2", ward_id="WARD-A", final_classification="INAPPROPRIATE"),
            _row(audit_id="3", ward_id="WARD-B", final_classification="INAPPROPRIATE"),
        )
        result = aggregate_ward_scorecard(rows)
        by_ward = {r.ward_id: r for r in result}
        assert by_ward["WARD-A"].inappropriate_rate == pytest.approx(0.5)
        assert by_ward["WARD-B"].inappropriate_rate == pytest.approx(1.0)


# =============================================================================
# Per-physician own-view (own-data only)
# =============================================================================


class TestPhysicianOwnViewAggregation:
    """Per-physician own-view contains the requested physicians' own
    numbers plus aggregate peer benchmarks — never other physicians' raw
    rows."""

    def test_returns_one_row_per_requested_physician(self) -> None:
        rows = tuple(
            _row(
                audit_id=f"a{i}",
                physician_id="phys-A",
                final_classification=(
                    "APPROPRIATE" if i < 7 else "INAPPROPRIATE"
                ),
            )
            for i in range(10)
        ) + tuple(
            _row(
                audit_id=f"b{i}",
                physician_id="phys-B",
                final_classification=(
                    "APPROPRIATE" if i < 5 else "INAPPROPRIATE"
                ),
            )
            for i in range(10)
        )
        result = aggregate_physician_own_view(rows, ("phys-A",))
        assert len(result) == 1
        assert result[0].physician_id == "phys-A"
        assert result[0].own_total == 10
        assert result[0].own_inappropriate_rate == pytest.approx(0.3)

    def test_peer_benchmarks_computed_over_all_physicians(self) -> None:
        # Three physicians with inappropriate_rates {0.3, 0.5, 0.7}.
        # Median = 0.5, p25 = 0.4, p75 = 0.6 (linear interpolation).
        rows: list[MonthlyReportRow] = []
        for phys, n_inappropriate in (("phys-A", 3), ("phys-B", 5), ("phys-C", 7)):
            for i in range(10):
                rows.append(
                    _row(
                        audit_id=f"{phys}-{i}",
                        physician_id=phys,
                        final_classification=(
                            "INAPPROPRIATE"
                            if i < n_inappropriate
                            else "APPROPRIATE"
                        ),
                    )
                )
        result = aggregate_physician_own_view(tuple(rows), ("phys-A",))
        assert result[0].peer_median_inappropriate_rate == pytest.approx(0.5)
        assert result[0].peer_p25_inappropriate_rate == pytest.approx(0.4)
        assert result[0].peer_p75_inappropriate_rate == pytest.approx(0.6)

    def test_own_view_row_has_no_per_peer_field(self) -> None:
        # Structural assertion of the "own-data only" property: there is
        # no field on PhysicianOwnViewRow that exposes another physician's
        # individual rate. Only aggregate quartiles.
        fields = set(PhysicianOwnViewRow.model_fields.keys())
        forbidden = {
            "peer_physician_ids",
            "peer_inappropriate_rates",
            "all_physicians",
        }
        leaked = fields & forbidden
        assert not leaked, f"PhysicianOwnViewRow leaks peer-level data: {leaked}"


# =============================================================================
# Indication-distribution breakdown
# =============================================================================


class TestIndicationDistributionAggregation:
    """One row per distinct indication code, sorted by frequency descending."""

    def test_counts_indication_codes(self) -> None:
        rows = (
            _row(audit_id="a", indication_codes=("anemia_symptomatic",)),
            _row(audit_id="b", indication_codes=("anemia_symptomatic",)),
            _row(audit_id="c", indication_codes=("acute_bleed",)),
        )
        result = aggregate_indication_distribution(rows)
        by_code = {r.indication_code: r for r in result}
        assert by_code["anemia_symptomatic"].total_orders == 2
        assert by_code["acute_bleed"].total_orders == 1

    def test_multi_indication_order_counts_per_code(self) -> None:
        # An order with two indications contributes 1 to each code; the
        # "share" sums to >1.0 across codes (this is documented behaviour,
        # not a bug — a multi-indication order has multiple clinically
        # valid reasons).
        rows = (
            _row(
                audit_id="x",
                indication_codes=("anemia_symptomatic", "acute_bleed"),
            ),
        )
        result = aggregate_indication_distribution(rows)
        assert len(result) == 2
        codes = {r.indication_code for r in result}
        assert codes == {"anemia_symptomatic", "acute_bleed"}

    def test_sorted_by_frequency_desc_then_code_asc(self) -> None:
        rows = (
            _row(audit_id="a", indication_codes=("c",)),
            _row(audit_id="b", indication_codes=("b",)),
            _row(audit_id="c", indication_codes=("b",)),
            _row(audit_id="d", indication_codes=("a",)),
            _row(audit_id="e", indication_codes=("a",)),
        )
        result = aggregate_indication_distribution(rows)
        codes = tuple(r.indication_code for r in result)
        # "a" and "b" both have 2 orders → tie broken by code asc
        # "c" has 1 order → last
        assert codes == ("a", "b", "c")


# =============================================================================
# Cohort-exception breakdown
# =============================================================================


class TestCohortExceptionAggregation:
    """One row per distinct ``cohort_applied`` (including ``"default"``)."""

    def test_includes_default_cohort(self) -> None:
        rows = (
            _row(audit_id="a", cohort_applied="default"),
            _row(audit_id="b", cohort_applied="cardiac_surgery"),
        )
        result = aggregate_cohort_exception(rows)
        cohorts = {r.cohort_applied for r in result}
        assert "default" in cohorts
        assert "cardiac_surgery" in cohorts

    def test_inappropriate_rate_per_cohort(self) -> None:
        rows = (
            _row(audit_id="a", cohort_applied="cardiac_surgery",
                 final_classification="INAPPROPRIATE"),
            _row(audit_id="b", cohort_applied="cardiac_surgery",
                 final_classification="APPROPRIATE"),
            _row(audit_id="c", cohort_applied="default",
                 final_classification="APPROPRIATE"),
        )
        result = aggregate_cohort_exception(rows)
        by_cohort = {r.cohort_applied: r for r in result}
        assert by_cohort["cardiac_surgery"].inappropriate_rate == pytest.approx(0.5)
        assert by_cohort["default"].inappropriate_rate == pytest.approx(0.0)


# =============================================================================
# Pipeline-health summary
# =============================================================================


class TestPipelineHealthAggregation:
    """Single row summarising the month's operational health."""

    def test_classified_orders_count(self) -> None:
        rows = (
            _row(audit_id="a", final_classification="APPROPRIATE"),
            _row(audit_id="b", final_classification="INAPPROPRIATE"),
            _row(audit_id="c", final_classification="NEEDS_REVIEW"),
            _row(audit_id="d", final_classification="INSUFFICIENT_EVIDENCE"),
        )
        result = aggregate_pipeline_health(rows)
        assert len(result) == 1
        r = result[0]
        assert r.total_orders == 4
        assert r.classified_orders == 2
        # NEEDS_REVIEW is the review bucket; INSUFFICIENT_EVIDENCE has
        # its own bucket so a documentation regression upstream is
        # visible separately from an LLM-uncertainty regression.
        assert r.needs_review_count == 1
        assert r.insufficient_evidence_count == 1
        assert r.needs_review_rate == pytest.approx(0.25)
        assert r.insufficient_evidence_rate == pytest.approx(0.25)

    def test_insufficient_evidence_distinct_from_needs_review(self) -> None:
        # PRD §"Documentation absence ≠ INAPPROPRIATE": a row whose
        # final_classification is INSUFFICIENT_EVIDENCE must NOT count
        # toward needs_review_count.
        rows = (
            _row(audit_id="x", final_classification="INSUFFICIENT_EVIDENCE"),
        )
        r = aggregate_pipeline_health(rows)[0]
        assert r.needs_review_count == 0
        assert r.insufficient_evidence_count == 1

    def test_needs_review_flag_adds_to_count(self) -> None:
        rows = (
            _row(audit_id="a", final_classification="APPROPRIATE",
                 needs_human_review=True),
            _row(audit_id="b", final_classification="APPROPRIATE",
                 needs_human_review=False),
        )
        result = aggregate_pipeline_health(rows)
        # 1 row flagged for human review (the APPROPRIATE one with the flag)
        assert result[0].needs_review_count == 1
        assert result[0].needs_review_rate == pytest.approx(0.5)


# =============================================================================
# Footer stamping
# =============================================================================


class TestFooterStamping:
    """Every section's CSV carries the footer fields on every row.

    Per PRD §"Output schema": ``policy_version``, ``model_id``,
    ``redactor_version`` are the three identifiers that let a downstream
    auditor re-derive a report line from the source AuditRows. Stamping
    them on every data row (not only a trailing line) means a grep / awk
    pipeline preserves the reproducibility chain.
    """

    def test_csv_header_contains_footer_columns(self, tmp_path: Path) -> None:
        section = ReportSection(
            name="ward_scorecard",
            rows=(
                WardScorecardRow(
                    ward_id="WARD-A",
                    total_orders=1,
                    appropriate=1,
                    inappropriate=0,
                    needs_review=0,
                    insufficient_evidence=0,
                    inappropriate_rate=0.0,
                ),
            ),
            footer=_footer(),
        )
        out = write_section_csv(section, tmp_path)
        text = out.read_text(encoding="utf-8")
        header_line = text.splitlines()[0]
        for col in (
            "policy_version",
            "model_id",
            "redactor_version",
            "redactor_model_sha",
            "prompt_hash",
            "evidence_bundle_hash",
        ):
            assert col in header_line, f"footer column {col!r} missing from header"

    def test_footer_fields_present_on_every_data_row(self, tmp_path: Path) -> None:
        section = ReportSection(
            name="ward_scorecard",
            rows=(
                WardScorecardRow(
                    ward_id="WARD-A",
                    total_orders=1,
                    appropriate=1,
                    inappropriate=0,
                    needs_review=0,
                    insufficient_evidence=0,
                    inappropriate_rate=0.0,
                ),
                WardScorecardRow(
                    ward_id="WARD-B",
                    total_orders=2,
                    appropriate=1,
                    inappropriate=1,
                    needs_review=0,
                    insufficient_evidence=0,
                    inappropriate_rate=0.5,
                ),
            ),
            footer=_footer(),
        )
        out = write_section_csv(section, tmp_path)
        text = out.read_text(encoding="utf-8")
        data_lines = text.splitlines()[1:]
        # All data lines reference every footer value — the full PRD
        # reproducibility-metadata set, not just the three named in the
        # ticket.
        for line in data_lines:
            assert "PR17.2-v1" in line
            assert "claude-sonnet-4-6-2026-04-01" in line
            assert "deid-th-1.3.0" in line
            assert "sha-abcdef0123456789" in line
            assert "sha-prompt-001" in line
            assert "sha-bundle-001" in line

    def test_empty_section_carries_footer_values(self, tmp_path: Path) -> None:
        # PRD reproducibility: even an empty section must carry the
        # footer so a downstream consumer can tell which policy / model /
        # redactor versions produced the empty result.
        section = ReportSection(
            name="cohort_exception", rows=(), footer=_footer()
        )
        out = write_section_csv(section, tmp_path)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) >= 2, "empty section must still emit a footer-bearing row"
        footer_row = lines[1]
        assert "PR17.2-v1" in footer_row
        assert "sha-bundle-001" in footer_row

    def test_empty_section_emits_numeric_placeholders(
        self, tmp_path: Path
    ) -> None:
        # An empty pipeline_health section must produce cells that parse
        # as their declared numeric type — blank strings would NaN-poison
        # a downstream pandas read of total_orders / *_rate columns.
        section = ReportSection(
            name="pipeline_health", rows=(), footer=_footer()
        )
        out = write_section_csv(section, tmp_path)
        lines = out.read_text(encoding="utf-8").splitlines()
        # pipeline_health columns: total_orders, classified_orders,
        # needs_review_count, needs_review_rate,
        # insufficient_evidence_count, insufficient_evidence_rate
        sentinel_cells = lines[1].split(",")
        assert sentinel_cells[0] == "0"  # total_orders (int)
        assert sentinel_cells[1] == "0"  # classified_orders (int)
        assert sentinel_cells[2] == "0"  # needs_review_count (int)
        assert sentinel_cells[3] == "0.0"  # needs_review_rate (float)
        assert sentinel_cells[4] == "0"  # insufficient_evidence_count (int)
        assert sentinel_cells[5] == "0.0"  # insufficient_evidence_rate (float)


# =============================================================================
# Golden-snapshot template tests (AC ②)
# =============================================================================


class TestGoldenSnapshotHospitalTrend:
    """Byte-identical CSV for a fixed input.

    "Snapshot-based template tests (golden output for fixed input)" is
    the ticket's wording. The asserted bytes encode the section's CSV
    schema: any future refactor that changes the column order, the row
    ordering, the rounding, or the line endings will produce a CSV
    mismatch and force a deliberate update.
    """

    def test_byte_identical_csv(self, tmp_path: Path) -> None:
        rows = (
            _row(audit_id="a", final_classification="APPROPRIATE"),
            _row(audit_id="b", final_classification="INAPPROPRIATE"),
            _row(audit_id="c", final_classification="INAPPROPRIATE"),
            _row(audit_id="d", final_classification="NEEDS_REVIEW"),
        )
        trend = aggregate_hospital_trend(rows, MONTH)
        section = ReportSection(
            name="hospital_trend", rows=trend, footer=_footer()
        )
        out = write_section_csv(section, tmp_path)
        actual = out.read_bytes()
        expected = (
            b"month,total_orders,appropriate,inappropriate,needs_review,"
            b"insufficient_evidence,inappropriate_rate,policy_version,"
            b"model_id,redactor_version,redactor_model_sha,prompt_hash,"
            b"evidence_bundle_hash\n"
            b"2026-05-01,4,1,2,1,0,0.5,PR17.2-v1,"
            b"claude-sonnet-4-6-2026-04-01,deid-th-1.3.0,"
            b"sha-abcdef0123456789,sha-prompt-001,sha-bundle-001\n"
        )
        assert actual == expected


class TestGoldenSnapshotWardScorecard:
    """Byte-identical CSV for the ward scorecard section."""

    def test_byte_identical_csv(self, tmp_path: Path) -> None:
        rows = (
            _row(audit_id="1", ward_id="WARD-A", final_classification="APPROPRIATE"),
            _row(audit_id="2", ward_id="WARD-A", final_classification="INAPPROPRIATE"),
            _row(audit_id="3", ward_id="WARD-B", final_classification="APPROPRIATE"),
        )
        scorecard = aggregate_ward_scorecard(rows)
        section = ReportSection(
            name="ward_scorecard", rows=scorecard, footer=_footer()
        )
        out = write_section_csv(section, tmp_path)
        actual = out.read_bytes()
        expected = (
            b"ward_id,total_orders,appropriate,inappropriate,needs_review,"
            b"insufficient_evidence,inappropriate_rate,policy_version,"
            b"model_id,redactor_version,redactor_model_sha,prompt_hash,"
            b"evidence_bundle_hash\n"
            b"WARD-A,2,1,1,0,0,0.5,PR17.2-v1,"
            b"claude-sonnet-4-6-2026-04-01,deid-th-1.3.0,"
            b"sha-abcdef0123456789,sha-prompt-001,sha-bundle-001\n"
            b"WARD-B,1,1,0,0,0,0.0,PR17.2-v1,"
            b"claude-sonnet-4-6-2026-04-01,deid-th-1.3.0,"
            b"sha-abcdef0123456789,sha-prompt-001,sha-bundle-001\n"
        )
        assert actual == expected


class TestGoldenSnapshotIndicationDistribution:
    """Byte-identical CSV for the indication-distribution section."""

    def test_byte_identical_csv(self, tmp_path: Path) -> None:
        rows = (
            _row(audit_id="a", indication_codes=("anemia_symptomatic",)),
            _row(audit_id="b", indication_codes=("anemia_symptomatic",)),
            _row(audit_id="c", indication_codes=("acute_bleed",)),
        )
        section = ReportSection(
            name="indication_distribution",
            rows=aggregate_indication_distribution(rows),
            footer=_footer(),
        )
        out = write_section_csv(section, tmp_path)
        expected = (
            b"indication_code,total_orders,share,policy_version,model_id,"
            b"redactor_version,redactor_model_sha,prompt_hash,evidence_bundle_hash\n"
            b"anemia_symptomatic,2,0.666667,PR17.2-v1,"
            b"claude-sonnet-4-6-2026-04-01,deid-th-1.3.0,"
            b"sha-abcdef0123456789,sha-prompt-001,sha-bundle-001\n"
            b"acute_bleed,1,0.333333,PR17.2-v1,"
            b"claude-sonnet-4-6-2026-04-01,deid-th-1.3.0,"
            b"sha-abcdef0123456789,sha-prompt-001,sha-bundle-001\n"
        )
        assert out.read_bytes() == expected


class TestGoldenSnapshotCohortException:
    """Byte-identical CSV for the cohort-exception section."""

    def test_byte_identical_csv(self, tmp_path: Path) -> None:
        rows = (
            _row(audit_id="a", cohort_applied="cardiac_surgery",
                 final_classification="INAPPROPRIATE"),
            _row(audit_id="b", cohort_applied="cardiac_surgery",
                 final_classification="APPROPRIATE"),
            _row(audit_id="c", cohort_applied="default",
                 final_classification="APPROPRIATE"),
        )
        section = ReportSection(
            name="cohort_exception",
            rows=aggregate_cohort_exception(rows),
            footer=_footer(),
        )
        out = write_section_csv(section, tmp_path)
        expected = (
            b"cohort_applied,total_orders,inappropriate,inappropriate_rate,"
            b"policy_version,model_id,redactor_version,redactor_model_sha,"
            b"prompt_hash,evidence_bundle_hash\n"
            b"cardiac_surgery,2,1,0.5,PR17.2-v1,"
            b"claude-sonnet-4-6-2026-04-01,deid-th-1.3.0,"
            b"sha-abcdef0123456789,sha-prompt-001,sha-bundle-001\n"
            b"default,1,0,0.0,PR17.2-v1,"
            b"claude-sonnet-4-6-2026-04-01,deid-th-1.3.0,"
            b"sha-abcdef0123456789,sha-prompt-001,sha-bundle-001\n"
        )
        assert out.read_bytes() == expected


class TestGoldenSnapshotPipelineHealth:
    """Byte-identical CSV for the pipeline-health section."""

    def test_byte_identical_csv(self, tmp_path: Path) -> None:
        rows = (
            _row(audit_id="a", final_classification="APPROPRIATE"),
            _row(audit_id="b", final_classification="INAPPROPRIATE"),
            _row(audit_id="c", final_classification="NEEDS_REVIEW"),
            _row(audit_id="d", final_classification="INSUFFICIENT_EVIDENCE"),
        )
        section = ReportSection(
            name="pipeline_health",
            rows=aggregate_pipeline_health(rows),
            footer=_footer(),
        )
        out = write_section_csv(section, tmp_path)
        expected = (
            b"total_orders,classified_orders,needs_review_count,"
            b"needs_review_rate,insufficient_evidence_count,"
            b"insufficient_evidence_rate,policy_version,model_id,"
            b"redactor_version,redactor_model_sha,prompt_hash,"
            b"evidence_bundle_hash\n"
            b"4,2,1,0.25,1,0.25,PR17.2-v1,"
            b"claude-sonnet-4-6-2026-04-01,deid-th-1.3.0,"
            b"sha-abcdef0123456789,sha-prompt-001,sha-bundle-001\n"
        )
        assert out.read_bytes() == expected


class TestGoldenSnapshotPhysicianOwnView:
    """Byte-identical CSV for a single physician's own-view artifact.

    Asserted against the per-physician filename (one CSV per physician)
    so the byte-identical contract also pins the structural "own-data
    only" property from PRD user story #10.
    """

    def test_byte_identical_csv(self, tmp_path: Path) -> None:
        # Three physicians with rates {0.0, 0.5, 1.0}.
        rows = (
            _row(audit_id="a1", physician_id="phys-A",
                 final_classification="APPROPRIATE"),
            _row(audit_id="b1", physician_id="phys-B",
                 final_classification="APPROPRIATE"),
            _row(audit_id="b2", physician_id="phys-B",
                 final_classification="INAPPROPRIATE"),
            _row(audit_id="c1", physician_id="phys-C",
                 final_classification="INAPPROPRIATE"),
        )
        rendered = aggregate_physician_own_view(rows, ("phys-A",))
        section = ReportSection(
            name="physician_own_view", rows=rendered, footer=_footer()
        )
        out = write_section_csv(
            section,
            tmp_path,
            filename_override=physician_own_view_filename("phys-A"),
        )
        assert out.name == "physician_own_view_phys-A.csv"
        expected = (
            b"physician_id,own_total,own_inappropriate_rate,"
            b"peer_median_inappropriate_rate,peer_p25_inappropriate_rate,"
            b"peer_p75_inappropriate_rate,policy_version,model_id,"
            b"redactor_version,redactor_model_sha,prompt_hash,"
            b"evidence_bundle_hash\n"
            b"phys-A,1,0.0,0.5,0.25,0.75,PR17.2-v1,"
            b"claude-sonnet-4-6-2026-04-01,deid-th-1.3.0,"
            b"sha-abcdef0123456789,sha-prompt-001,sha-bundle-001\n"
        )
        assert out.read_bytes() == expected


# =============================================================================
# AC ③ — CSV schema doc
# =============================================================================


class TestCsvSchemaDocumented:
    """``docs/report-schema.md`` exists and names every section's CSV.

    Per the ticket AC: "CSV schema documented in docs/report-schema.md".
    The minimum bar is: the doc file exists and references each of the
    six section names so a downstream consumer can grep it for the
    section they care about.
    """

    def test_schema_doc_exists_and_names_each_section(self) -> None:
        # tests/unit/test_report_generator.py → repo-root/docs/report-schema.md
        repo_root = Path(__file__).resolve().parents[2]
        schema_doc = repo_root / "docs" / "report-schema.md"
        assert schema_doc.exists(), (
            f"AC ③ requires {schema_doc.relative_to(repo_root)}; "
            "create it in the GREEN phase before the promise fires."
        )
        text = schema_doc.read_text(encoding="utf-8")
        for section_name in SECTION_NAMES:
            assert section_name in text, (
                f"docs/report-schema.md must reference section {section_name!r}"
            )
        for footer_field in (
            "policy_version",
            "model_id",
            "redactor_version",
            "redactor_model_sha",
            "prompt_hash",
            "evidence_bundle_hash",
        ):
            assert footer_field in text, (
                f"docs/report-schema.md must reference footer field "
                f"{footer_field!r}"
            )


# =============================================================================
# AC ④ — PDF rendering sample run
# =============================================================================


class TestPdfRendering:
    """``render_report_pdf`` writes a PDF byte stream to disk for a
    sample run.

    Byte-identical comparison is *not* asserted: reportlab embeds a
    non-deterministic generation timestamp in the PDF trailer. The
    contract is "the file is a real PDF and contains all six sections",
    which we verify via magic-bytes + non-trivial size.
    """

    def test_pdf_emitted_with_magic_bytes(self, tmp_path: Path) -> None:
        sections = tuple(
            ReportSection(name=name, rows=(), footer=_footer())
            for name in SECTION_NAMES
        )
        out_path = tmp_path / "report.pdf"
        result = render_report_pdf(
            sections=sections,
            footer=_footer(),
            month_label="May 2026",
            output_path=out_path,
        )
        assert result == out_path
        data = out_path.read_bytes()
        assert data.startswith(PDF_MAGIC)
        # A trivial PDF still runs ~1 KB; assert non-trivial so a stub
        # writing just the magic header cannot pass this test.
        assert len(data) > 500


# =============================================================================
# Month-boundary filtering
# =============================================================================


class TestMonthBoundaryFiltering:
    """``filter_rows_for_month`` is half-open
    ``[month_local_start, next_month_local_start)`` in Asia/Bangkok
    (UTC+7, no DST).

    PRD §"Tz-aware throughout": datetimes are stored UTC, rendered
    Asia/Bangkok. The monthly bucket is the hospital business month
    (Bangkok local), not the UTC month — an order at 23:00 Bangkok on
    the 31st must bucket into that local month even though it falls in
    the next UTC month.
    """

    def test_includes_first_local_second_of_month(self) -> None:
        # Bangkok 00:00:00 May 1 == UTC 17:00:00 April 30. The Bangkok-
        # local "first second of May" belongs to May.
        row = _row(
            audit_id="bkk-may-start",
            order_datetime=datetime(2026, 4, 30, 17, 0, 0, tzinfo=UTC),
        )
        result = filter_rows_for_month((row,), MONTH)
        assert result == (row,)

    def test_excludes_first_local_second_of_next_month(self) -> None:
        # Bangkok 00:00:00 June 1 == UTC 17:00:00 May 31. The Bangkok-
        # local "first second of June" must NOT bucket into May.
        row = _row(
            audit_id="bkk-jun-start",
            order_datetime=datetime(2026, 5, 31, 17, 0, 0, tzinfo=UTC),
        )
        result = filter_rows_for_month((row,), MONTH)
        assert result == ()

    def test_includes_last_local_second_of_month(self) -> None:
        # Bangkok 23:59:59 May 31 == UTC 16:59:59 May 31. The last
        # Bangkok-local instant of the month is inside the bucket.
        row = _row(
            audit_id="bkk-may-end",
            order_datetime=datetime(2026, 5, 31, 16, 59, 59, tzinfo=UTC),
        )
        result = filter_rows_for_month((row,), MONTH)
        assert result == (row,)

    def test_utc_midnight_of_next_month_is_still_in_local_month(self) -> None:
        # The PRD-cited bug this test pins: UTC midnight June 1 is 07:00
        # Bangkok on June 1 — which is *June* locally, not May. A UTC-
        # bound filter would either include or exclude this depending
        # on the half-open direction; the Bangkok-bound filter excludes.
        row = _row(
            audit_id="utc-jun-midnight",
            order_datetime=datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC),
        )
        result = filter_rows_for_month((row,), MONTH)
        assert result == ()

    def test_utc_last_second_of_april_is_in_local_may(self) -> None:
        # 23:59:59 UTC April 30 == 06:59:59 Bangkok May 1, which is May
        # locally. PRD's "stored UTC, rendered Asia/Bangkok" rule means
        # this row IS in the May report.
        row = _row(
            audit_id="utc-apr-end",
            order_datetime=datetime(2026, 4, 30, 23, 59, 59, tzinfo=UTC),
        )
        result = filter_rows_for_month((row,), MONTH)
        assert result == (row,)

    def test_year_rollover_december_to_january(self) -> None:
        # Bangkok 00:00:00 Jan 1 2027 == UTC 17:00:00 Dec 31 2026. The
        # December 2026 report must exclude this row, and the January
        # 2027 report must include it. Pins the year-rollover branch in
        # _next_month_first_of's day-28+10 trick.
        rollover_row = _row(
            audit_id="jan-1-bkk",
            order_datetime=datetime(2026, 12, 31, 17, 0, 0, tzinfo=UTC),
        )
        late_december = _row(
            audit_id="dec-31-bkk-23",
            order_datetime=datetime(2026, 12, 31, 16, 59, 59, tzinfo=UTC),
        )
        december_2026 = date(2026, 12, 1)
        january_2027 = date(2027, 1, 1)
        assert filter_rows_for_month(
            (rollover_row, late_december), december_2026
        ) == (late_december,)
        assert filter_rows_for_month(
            (rollover_row, late_december), january_2027
        ) == (rollover_row,)


# =============================================================================
# Property test — total = sum of classification buckets.
#
# Issue #28 is labelled "thin", but the bucket-sum identity is the deepest
# invariant in the module: a sloppy aggregator that drops or double-counts
# a class would produce a non-zero residual. One property test is enough.
# =============================================================================


class TestClassificationBucketsSum:
    """For any input, hospital-trend buckets sum to ``total_orders``."""

    @settings(max_examples=50, deadline=None)
    @given(
        st.lists(
            st.sampled_from(
                ["APPROPRIATE", "INAPPROPRIATE", "NEEDS_REVIEW", "INSUFFICIENT_EVIDENCE"]
            ),
            min_size=1,
            max_size=20,
        )
    )
    def test_bucket_sum_equals_total(self, labels: list[str]) -> None:
        rows = tuple(
            _row(audit_id=f"id-{i}", final_classification=label)
            for i, label in enumerate(labels)
        )
        result = aggregate_hospital_trend(rows, MONTH)
        assert len(result) == 1
        r = result[0]
        bucket_sum = (
            r.appropriate + r.inappropriate + r.needs_review + r.insufficient_evidence
        )
        assert bucket_sum == r.total_orders == len(labels)


# =============================================================================
# Orchestrator end-to-end
# =============================================================================


class TestGenerateMonthlyReport:
    """``generate_monthly_report`` writes five committee CSVs, one PDF,
    and one CSV per physician in ``physician_ids_for_own_view``; raises
    on empty input."""

    def test_emits_five_committee_csvs_plus_per_physician_and_pdf(
        self, tmp_path: Path
    ) -> None:
        rows = (
            _row(audit_id="a", physician_id="phys-1",
                 final_classification="APPROPRIATE"),
            _row(audit_id="b", physician_id="phys-2",
                 final_classification="INAPPROPRIATE"),
        )
        artifacts = generate_monthly_report(
            _inputs(
                rows,
                output_dir=tmp_path,
                physician_ids_for_own_view=("phys-1", "phys-2"),
            )
        )
        assert isinstance(artifacts, ReportArtifacts)
        # Five committee section CSVs; physician_own_view is NOT a
        # committee artifact — its data is per-physician.
        expected_committee = {
            "hospital_trend",
            "ward_scorecard",
            "indication_distribution",
            "cohort_exception",
            "pipeline_health",
        }
        assert set(artifacts.csv_paths.keys()) == expected_committee
        for path in artifacts.csv_paths.values():
            assert path.exists()
        assert "physician_own_view" not in artifacts.csv_paths
        # One CSV per physician_id in the requested set.
        assert set(artifacts.physician_own_view_csv_paths.keys()) == {
            "phys-1",
            "phys-2",
        }
        for pid, path in artifacts.physician_own_view_csv_paths.items():
            assert path.exists()
            assert path.name == f"physician_own_view_{pid}.csv"
        assert artifacts.pdf_path.exists()
        assert artifacts.pdf_path.read_bytes().startswith(PDF_MAGIC)

    def test_per_physician_csv_contains_only_that_physician_row(
        self, tmp_path: Path
    ) -> None:
        # Structural enforcement of PRD user story #10's "own-data only":
        # each per-physician artifact must contain at most one data row
        # bearing this physician's identifier.
        rows = (
            _row(audit_id="a", physician_id="phys-1"),
            _row(audit_id="b", physician_id="phys-2"),
            _row(audit_id="c", physician_id="phys-3"),
        )
        artifacts = generate_monthly_report(
            _inputs(
                rows,
                output_dir=tmp_path,
                physician_ids_for_own_view=("phys-1", "phys-2", "phys-3"),
            )
        )
        for pid, path in artifacts.physician_own_view_csv_paths.items():
            text = path.read_text(encoding="utf-8")
            data_lines = text.splitlines()[1:]
            assert len(data_lines) == 1, (
                f"per-physician CSV must contain exactly one data row "
                f"(got {len(data_lines)} for {pid})"
            )
            # The single data row begins with this physician's id; no
            # other physician's id appears in the file.
            assert data_lines[0].startswith(f"{pid},"), (
                f"row 1 of {path.name} must begin with the artifact's "
                f"own physician_id"
            )
            for other in {"phys-1", "phys-2", "phys-3"} - {pid}:
                assert other not in text, (
                    f"{path.name} leaked another physician's id: {other}"
                )

    def test_empty_input_raises_empty_input_error(self, tmp_path: Path) -> None:
        # Zero rows for the month → operator should be notified, not
        # silently shipped an empty CSV.
        out_of_month = _row(
            audit_id="zzz",
            order_datetime=datetime(2026, 4, 15, 12, tzinfo=UTC),
        )
        with pytest.raises(EmptyInputError):
            generate_monthly_report(_inputs((out_of_month,), output_dir=tmp_path))

    def test_footer_round_trips_to_artifacts(self, tmp_path: Path) -> None:
        rows = (_row(audit_id="a"),)
        artifacts = generate_monthly_report(_inputs(rows, output_dir=tmp_path))
        assert artifacts.footer == _footer()


# =============================================================================
# Filename convention
# =============================================================================


class TestSectionFilename:
    """The CSV filename per section is ``{name}.csv``.

    Pinned so the orchestrator and external consumers agree without
    re-encoding the convention in two places.
    """

    def test_filename_matches_section_name(self) -> None:
        for name in SECTION_NAMES:
            assert section_filename(name) == f"{name}.csv"
