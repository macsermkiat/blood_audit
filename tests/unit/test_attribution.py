"""Tests for :mod:`bba.attribution` — doctor / department ordering scorecards.

Feature 2 (doctor / department top-10 ranking): the ordering doctor is
``BDVST.DCTREQ``; ``DCT.csv`` maps ``Dct`` → ``Deptlct`` / ``Deptname``.
The verdict source for this build is the 300-case human review workbook
(Sheet1 col J); the adapter is swappable (``VerdictSource``) so the next
build can rank on full-cohort pipeline verdicts without touching the
resolvers, aggregation, or ranking.

WHY these tests matter clinically: the ranking is committee-facing — a
doctor presented as "top inappropriate orderer" off a 1/1 = 100% rate
would be defamatory noise. The min-order threshold and the 3-bucket
collapse (Unresolved = NEEDS_REVIEW + INSUFFICIENT_EVIDENCE, per PRD
"Documentation absence is not INAPPROPRIATE") are the load-bearing rules.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import openpyxl
import pytest

from bba.attribution import (
    DEFAULT_MIN_ORDERS,
    HUMAN_LABEL_TO_CLASSIFICATION,
    UNATTRIBUTED_DEPARTMENT_ID,
    UNATTRIBUTED_DOCTOR_ID,
    DoctorRecord,
    GroupLabStats,
    OrderLabValue,
    RankedRow,
    aggregate_department_lab_stats,
    aggregate_doctor_lab_stats,
    build_department_scorecards,
    build_doctor_scorecards,
    build_rankings,
    human_label_verdict_source,
    load_dct_registry,
    load_order_labs,
    load_reqno_to_doctor,
    make_physician_resolver,
    make_ward_resolver,
    needs_review_verdict_projector,
    pipeline_verdict_source,
    rank_department_scorecards,
    rank_doctor_scorecards,
    rank_top_n,
    reconcile_verdict_sources,
    strict_verdict_projector,
    write_ranking_csv,
    write_rankings_html,
)
from bba.dashboard.models import PhysicianScorecard


_REPORT_LAB_COLUMNS = ("reqno", "component", "hb_value_g_dl", "hb_freshness")


def _write_report_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write a minimal report.csv carrying only the columns the lab loader
    reads (the real file is ~50 columns wide; DictReader ignores the rest)."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(_REPORT_LAB_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _red_cell(reqno: str, hb: str, *, freshness: str = "fresh") -> dict[str, str]:
    return {
        "reqno": reqno,
        "component": "red_cell",
        "hb_value_g_dl": hb,
        "hb_freshness": freshness,
    }


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


_DCT_HEADER = "Dct,Prefix,Fname,Lname,Deptlct,Deptname\n"


def _write_dct_csv(path: Path, rows: list[str]) -> Path:
    path.write_text(_DCT_HEADER + "".join(r + "\n" for r in rows), encoding="utf-8")
    return path


def _write_bdvst_csv(path: Path, rows: list[str]) -> Path:
    # EXTRA column proves the loader tolerates the real file's ~50 columns.
    path.write_text(
        "REQNO,EXTRA,DCTREQ\n" + "".join(r + "\n" for r in rows),
        encoding="utf-8",
    )
    return path


def _write_review_xlsx(
    path: Path, rows: list[tuple[object, object]], *, sheet: str = "Sheet1"
) -> Path:
    """Mimic the review workbook: two header rows, data from row 3,
    REQNO in column A (float-typed by Excel), verdict in column J."""
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = sheet
    ws.append(["ข้อมูลเคสผู้ป่วย"] + [None] * 8 + ["ความสมเหตุสมผล"])
    ws.append(["CaseNumber"] + [None] * 9)
    for reqno, verdict in rows:
        row: list[object] = [None] * 10
        row[0] = reqno
        row[9] = verdict
        ws.append(row)
    wb.save(path)
    return path


def _scorecard(
    pid: str,
    total: int,
    appropriate: int = 0,
    inappropriate: int = 0,
    needs_review: int = 0,
    insufficient: int = 0,
    name: str = "",
) -> PhysicianScorecard:
    return PhysicianScorecard(
        physician_id=pid,
        physician_name=name or pid,
        ward_id="d1",
        total_orders=total,
        appropriate_count=appropriate,
        inappropriate_count=inappropriate,
        needs_review_count=needs_review,
        insufficient_evidence_count=insufficient,
        average_confidence=0.0,
    )


# ---------------------------------------------------------------------------
# DCT registry loader
# ---------------------------------------------------------------------------


class TestLoadDctRegistry:
    def test_loads_records_keyed_by_dct(self, tmp_path: Path) -> None:
        csv_path = _write_dct_csv(
            tmp_path / "DCT.csv",
            [
                "302389,พญ.,ส*****,ว*****,1028000000,ฝ่ายอายุรศาสตร์",
                "302380,นพ.,ม*****,พ*****,1029000000,ฝ่ายศัลยศาสตร์",
            ],
        )
        registry = load_dct_registry(csv_path)
        assert set(registry) == {"302389", "302380"}
        rec = registry["302389"]
        assert isinstance(rec, DoctorRecord)
        assert rec.deptlct == "1028000000"
        assert rec.deptname == "ฝ่ายอายุรศาสตร์"

    def test_display_name_composes_masked_prefix_and_name(self, tmp_path: Path) -> None:
        csv_path = _write_dct_csv(
            tmp_path / "DCT.csv",
            ["302389,พญ.,ส*****,ว*****,1028000000,ฝ่ายอายุรศาสตร์"],
        )
        rec = load_dct_registry(csv_path)["302389"]
        assert "พญ." in rec.display_name
        assert "ส*****" in rec.display_name

    def test_blank_department_kept_as_empty_string(self, tmp_path: Path) -> None:
        csv_path = _write_dct_csv(tmp_path / "DCT.csv", ["304000,นพ.,ก*****,ข*****,,"])
        rec = load_dct_registry(csv_path)["304000"]
        assert rec.deptlct == ""
        assert rec.deptname == ""

    def test_missing_required_column_fails_loud(self, tmp_path: Path) -> None:
        bad = tmp_path / "DCT.csv"
        bad.write_text("Dct,Prefix\n302389,พญ.\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Deptlct"):
            load_dct_registry(bad)

    def test_duplicate_dct_key_fails_loud(self, tmp_path: Path) -> None:
        csv_path = _write_dct_csv(
            tmp_path / "DCT.csv",
            [
                "302389,พญ.,ส*****,ว*****,1028000000,ฝ่ายอายุรศาสตร์",
                "302389,นพ.,ม*****,พ*****,1029000000,ฝ่ายศัลยศาสตร์",
            ],
        )
        with pytest.raises(ValueError, match="302389"):
            load_dct_registry(csv_path)

    def test_utf8_bom_tolerated(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "DCT.csv"
        csv_path.write_text(
            "﻿" + _DCT_HEADER + "302389,พญ.,ส*****,ว*****,1028000000,อายุร\n",
            encoding="utf-8",
        )
        assert "302389" in load_dct_registry(csv_path)


# ---------------------------------------------------------------------------
# BDVST REQNO → DCTREQ loader
# ---------------------------------------------------------------------------


class TestLoadReqnoToDoctor:
    def test_maps_reqno_to_dctreq(self, tmp_path: Path) -> None:
        csv_path = _write_bdvst_csv(
            tmp_path / "BDVST.csv", ["68000001,x,302389", "68000002,y,302380"]
        )
        mapping = load_reqno_to_doctor(csv_path)
        assert mapping == {"68000001": "302389", "68000002": "302380"}

    def test_rows_without_dctreq_are_omitted(self, tmp_path: Path) -> None:
        csv_path = _write_bdvst_csv(
            tmp_path / "BDVST.csv", ["68000001,x,302389", "68000002,y,"]
        )
        mapping = load_reqno_to_doctor(csv_path)
        assert "68000002" not in mapping

    def test_missing_required_column_fails_loud(self, tmp_path: Path) -> None:
        bad = tmp_path / "BDVST.csv"
        bad.write_text("REQNO,OTHER\n68000001,x\n", encoding="utf-8")
        with pytest.raises(ValueError, match="DCTREQ"):
            load_reqno_to_doctor(bad)

    def test_conflicting_duplicate_reqno_fails_loud(self, tmp_path: Path) -> None:
        csv_path = _write_bdvst_csv(
            tmp_path / "BDVST.csv", ["68000001,x,302389", "68000001,y,999999"]
        )
        with pytest.raises(ValueError, match="68000001"):
            load_reqno_to_doctor(csv_path)

    def test_identical_duplicate_reqno_tolerated(self, tmp_path: Path) -> None:
        csv_path = _write_bdvst_csv(
            tmp_path / "BDVST.csv", ["68000001,x,302389", "68000001,y,302389"]
        )
        assert load_reqno_to_doctor(csv_path) == {"68000001": "302389"}


# ---------------------------------------------------------------------------
# Attribution resolvers (the M0/M1 seam implementations)
# ---------------------------------------------------------------------------


class TestResolvers:
    _REGISTRY = {
        "302389": DoctorRecord(
            dct="302389",
            display_name="พญ.ส***** ว*****",
            deptlct="1028000000",
            deptname="ฝ่ายอายุรศาสตร์",
        ),
        "304000": DoctorRecord(
            dct="304000", display_name="นพ.ก*****", deptlct="", deptname=""
        ),
    }
    _REQNO_TO_DOCTOR = {"68000001": "302389", "68000003": "304000"}

    def test_physician_resolver_returns_doctor_code(self) -> None:
        resolver = make_physician_resolver(self._REQNO_TO_DOCTOR)
        row = SimpleNamespace(reqno="68000001")
        assert resolver(row) == "302389"

    def test_physician_resolver_unknown_reqno_returns_sentinel(self) -> None:
        resolver = make_physician_resolver(self._REQNO_TO_DOCTOR)
        assert resolver(SimpleNamespace(reqno="99999999")) == UNATTRIBUTED_DOCTOR_ID

    def test_ward_resolver_returns_deptlct(self) -> None:
        resolver = make_ward_resolver(self._REQNO_TO_DOCTOR, self._REGISTRY)
        assert resolver(SimpleNamespace(reqno="68000001")) == "1028000000"

    def test_ward_resolver_doctor_without_department_returns_sentinel(self) -> None:
        resolver = make_ward_resolver(self._REQNO_TO_DOCTOR, self._REGISTRY)
        assert resolver(SimpleNamespace(reqno="68000003")) == UNATTRIBUTED_DEPARTMENT_ID

    def test_ward_resolver_unknown_reqno_returns_sentinel(self) -> None:
        resolver = make_ward_resolver(self._REQNO_TO_DOCTOR, self._REGISTRY)
        assert resolver(SimpleNamespace(reqno="99999999")) == UNATTRIBUTED_DEPARTMENT_ID


# ---------------------------------------------------------------------------
# Human-label verdict adapter
# ---------------------------------------------------------------------------


class TestHumanLabelVerdictSource:
    def test_maps_the_three_thai_labels(self, tmp_path: Path) -> None:
        xlsx = _write_review_xlsx(
            tmp_path / "review.xlsx",
            [
                (68000001.0, "สมเหตุสมผล"),
                (68000002.0, "ไม่สมเหตุสมผล"),
                (68000003.0, "ไม่สามารถสรุปได้"),
            ],
        )
        verdicts = human_label_verdict_source(xlsx)()
        assert verdicts == {
            "68000001": "APPROPRIATE",
            "68000002": "INAPPROPRIATE",
            "68000003": "NEEDS_REVIEW",
        }

    def test_float_reqno_normalizes_to_canonical_string(self, tmp_path: Path) -> None:
        xlsx = _write_review_xlsx(tmp_path / "review.xlsx", [(68049423.0, "สมเหตุสมผล")])
        assert "68049423" in human_label_verdict_source(xlsx)()

    def test_string_reqno_and_padded_label_accepted(self, tmp_path: Path) -> None:
        xlsx = _write_review_xlsx(
            tmp_path / "review.xlsx", [("68000001", " สมเหตุสมผล ")]
        )
        assert human_label_verdict_source(xlsx)() == {"68000001": "APPROPRIATE"}

    def test_unknown_label_fails_loud_naming_reqno_and_label(
        self, tmp_path: Path
    ) -> None:
        xlsx = _write_review_xlsx(tmp_path / "review.xlsx", [(68000001.0, "อื่นๆ")])
        with pytest.raises(ValueError, match="68000001"):
            human_label_verdict_source(xlsx)()

    def test_missing_verdict_for_present_reqno_fails_loud(self, tmp_path: Path) -> None:
        xlsx = _write_review_xlsx(tmp_path / "review.xlsx", [(68000001.0, None)])
        with pytest.raises(ValueError, match="68000001"):
            human_label_verdict_source(xlsx)()

    def test_rows_without_reqno_are_skipped(self, tmp_path: Path) -> None:
        xlsx = _write_review_xlsx(
            tmp_path / "review.xlsx",
            [(68000001.0, "สมเหตุสมผล"), (None, None)],
        )
        assert len(human_label_verdict_source(xlsx)()) == 1

    def test_duplicate_reqno_fails_loud(self, tmp_path: Path) -> None:
        xlsx = _write_review_xlsx(
            tmp_path / "review.xlsx",
            [(68000001.0, "สมเหตุสมผล"), (68000001.0, "ไม่สมเหตุสมผล")],
        )
        with pytest.raises(ValueError, match="68000001"):
            human_label_verdict_source(xlsx)()

    def test_label_map_covers_exactly_three_labels(self) -> None:
        # The 3-bucket collapse is a clinical decision (Unresolved =
        # cannot-conclude); a silent fourth label would corrupt totals.
        assert HUMAN_LABEL_TO_CLASSIFICATION == {
            "สมเหตุสมผล": "APPROPRIATE",
            "ไม่สมเหตุสมผล": "INAPPROPRIATE",
            "ไม่สามารถสรุปได้": "NEEDS_REVIEW",
        }


# ---------------------------------------------------------------------------
# Pipeline verdict adapter (the application's own per-order verdicts)
# ---------------------------------------------------------------------------


def _audit_row(reqno: object, final_classification: str) -> SimpleNamespace:
    """A structural stand-in for :class:`bba.audit_store.models.AuditRow` —
    only the two fields :class:`SupportsVerdict` reads."""
    return SimpleNamespace(reqno=reqno, final_classification=final_classification)


class TestPipelineVerdictSource:
    def test_maps_reqno_to_final_classification(self) -> None:
        rows = [
            _audit_row("68000001", "APPROPRIATE"),
            _audit_row("68000002", "INAPPROPRIATE"),
            _audit_row("68000003", "NEEDS_REVIEW"),
            _audit_row("68000004", "INSUFFICIENT_EVIDENCE"),
        ]
        assert pipeline_verdict_source(rows)() == {
            "68000001": "APPROPRIATE",
            "68000002": "INAPPROPRIATE",
            "68000003": "NEEDS_REVIEW",
            "68000004": "INSUFFICIENT_EVIDENCE",
        }

    def test_source_is_re_callable_over_a_generator(self) -> None:
        # A generator must be materialized so a second call is not empty —
        # the pilot's reconciliation reads the source more than once.
        source = pipeline_verdict_source(
            _audit_row(f"6800000{i}", "APPROPRIATE") for i in range(3)
        )
        first = source()
        assert source() == first
        assert len(first) == 3

    def test_identical_duplicate_reqno_tolerated(self) -> None:
        rows = [
            _audit_row("68000001", "APPROPRIATE"),
            _audit_row("68000001", "APPROPRIATE"),
        ]
        assert pipeline_verdict_source(rows)() == {"68000001": "APPROPRIATE"}

    def test_conflicting_duplicate_reqno_fails_loud(self) -> None:
        rows = [
            _audit_row("68000001", "APPROPRIATE"),
            _audit_row("68000001", "INAPPROPRIATE"),
        ]
        with pytest.raises(ValueError, match="68000001"):
            pipeline_verdict_source(rows)()

    def test_empty_reqno_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="empty REQNO"):
            pipeline_verdict_source([_audit_row("  ", "APPROPRIATE")])()

    def test_empty_cohort_yields_empty_mapping(self) -> None:
        assert pipeline_verdict_source([])() == {}

    def test_default_projector_fails_loud_on_potentially_inappropriate(self) -> None:
        # The audit store's fifth value has no confident bucket; the default
        # must refuse it rather than silently guess (clinical decision).
        rows = [_audit_row("68000001", "POTENTIALLY_INAPPROPRIATE")]
        with pytest.raises(ValueError, match="POTENTIALLY_INAPPROPRIATE"):
            pipeline_verdict_source(rows)()

    def test_needs_review_projector_pools_potentially_inappropriate(self) -> None:
        rows = [_audit_row("68000001", "POTENTIALLY_INAPPROPRIATE")]
        verdicts = pipeline_verdict_source(
            rows, projector=needs_review_verdict_projector
        )()
        assert verdicts == {"68000001": "NEEDS_REVIEW"}

    def test_strict_projector_names_preop_reservation_unconfirmed(self) -> None:
        with pytest.raises(ValueError, match="PREOP_RESERVATION_UNCONFIRMED"):
            strict_verdict_projector("PREOP_RESERVATION_UNCONFIRMED")

    def test_needs_review_projector_pools_preop_reservation_unconfirmed(
        self,
    ) -> None:
        rows = [_audit_row("68000001", "PREOP_RESERVATION_UNCONFIRMED")]
        verdicts = pipeline_verdict_source(
            rows, projector=needs_review_verdict_projector
        )()
        assert verdicts == {"68000001": "NEEDS_REVIEW"}

    def test_pooled_preop_reservation_totals_reconcile_as_unresolved(self) -> None:
        rows = [
            _audit_row("68000001", "APPROPRIATE"),
            _audit_row("68000002", "INAPPROPRIATE"),
            _audit_row("68000003", "PREOP_RESERVATION_UNCONFIRMED"),
        ]
        verdicts = pipeline_verdict_source(
            rows, projector=needs_review_verdict_projector
        )()

        result = build_rankings(
            verdicts=verdicts,
            reqno_to_doctor={},
            dct_registry={},
        )

        assert result.totals.appropriate == 1
        assert result.totals.inappropriate == 1
        assert result.totals.unresolved == 1
        assert result.totals.total == len(rows)

    def test_unknown_classification_fails_loud_under_both_projectors(self) -> None:
        rows = [_audit_row("68000001", "MADE_UP_LABEL")]
        with pytest.raises(ValueError, match="MADE_UP_LABEL"):
            pipeline_verdict_source(rows)()
        with pytest.raises(ValueError, match="MADE_UP_LABEL"):
            pipeline_verdict_source(rows, projector=needs_review_verdict_projector)()


# ---------------------------------------------------------------------------
# Verdict-source reconciliation (the pre-swap cross-check)
# ---------------------------------------------------------------------------


class TestReconcileVerdictSources:
    def test_counts_agreement_over_the_overlap_only(self) -> None:
        pipeline = {"1": "APPROPRIATE", "2": "APPROPRIATE", "3": "NEEDS_REVIEW"}
        human = {"1": "APPROPRIATE", "2": "INAPPROPRIATE", "4": "APPROPRIATE"}
        result = reconcile_verdict_sources(pipeline, human)
        assert result.overlap == 2  # reqnos 1 and 2
        assert result.agree == 1  # reqno 1
        assert result.disagree == 1  # reqno 2
        assert result.pipeline_only == 1  # reqno 3
        assert result.human_only == 1  # reqno 4

    def test_flags_pipeline_over_clear_of_a_human_inappropriate(self) -> None:
        # The peri-op danger: the human called it inappropriate, the pipeline
        # cleared it to appropriate. This count must stay ~0 to trust the swap.
        result = reconcile_verdict_sources({"1": "APPROPRIATE"}, {"1": "INAPPROPRIATE"})
        assert result.pipeline_over_clears == 1

    def test_needs_review_disagreement_is_not_an_over_clear(self) -> None:
        # Human INAPPROPRIATE vs pipeline NEEDS_REVIEW is a disagreement but
        # not a dangerous clear — the order stays out of the appropriate bucket.
        result = reconcile_verdict_sources(
            {"1": "NEEDS_REVIEW"}, {"1": "INAPPROPRIATE"}
        )
        assert result.disagree == 1
        assert result.pipeline_over_clears == 0


# ---------------------------------------------------------------------------
# Scorecard aggregation (reuses the dashboard's frozen scorecard models)
# ---------------------------------------------------------------------------


class TestBuildScorecards:
    _REGISTRY = {
        "302389": DoctorRecord(
            dct="302389",
            display_name="พญ.ส***** ว*****",
            deptlct="1028000000",
            deptname="ฝ่ายอายุรศาสตร์",
        ),
        "302380": DoctorRecord(
            dct="302380",
            display_name="นพ.ม***** พ*****",
            deptlct="1029000000",
            deptname="ฝ่ายศัลยศาสตร์",
        ),
    }
    _REQNO_TO_DOCTOR = {
        "r1": "302389",
        "r2": "302389",
        "r3": "302389",
        "r4": "302380",
    }
    _VERDICTS = {
        "r1": "APPROPRIATE",
        "r2": "INAPPROPRIATE",
        "r3": "NEEDS_REVIEW",
        "r4": "INSUFFICIENT_EVIDENCE",
        "r5": "APPROPRIATE",  # no attribution → unattributed bucket
    }

    def test_doctor_scorecards_count_all_four_classifications(self) -> None:
        cards = build_doctor_scorecards(
            self._VERDICTS, self._REQNO_TO_DOCTOR, self._REGISTRY
        )
        by_id = {c.physician_id: c for c in cards}
        card = by_id["302389"]
        assert card.total_orders == 3
        assert card.appropriate_count == 1
        assert card.inappropriate_count == 1
        assert card.needs_review_count == 1
        assert card.insufficient_evidence_count == 0
        assert card.physician_name == "พญ.ส***** ว*****"
        assert card.ward_id == "1028000000"

    def test_unattributed_orders_land_in_sentinel_bucket(self) -> None:
        cards = build_doctor_scorecards(
            self._VERDICTS, self._REQNO_TO_DOCTOR, self._REGISTRY
        )
        by_id = {c.physician_id: c for c in cards}
        assert by_id[UNATTRIBUTED_DOCTOR_ID].total_orders == 1

    def test_scorecard_totals_reconcile_with_verdict_count(self) -> None:
        cards = build_doctor_scorecards(
            self._VERDICTS, self._REQNO_TO_DOCTOR, self._REGISTRY
        )
        assert sum(c.total_orders for c in cards) == len(self._VERDICTS)

    def test_average_confidence_is_zero_for_human_labels(self) -> None:
        # Human labels carry no model confidence; the reused dashboard
        # model requires the field, and this feature never renders it.
        cards = build_doctor_scorecards(
            self._VERDICTS, self._REQNO_TO_DOCTOR, self._REGISTRY
        )
        assert all(c.average_confidence == 0.0 for c in cards)

    def test_doctor_not_in_registry_falls_back_to_code_and_sentinel_dept(
        self,
    ) -> None:
        cards = build_doctor_scorecards(
            {"r1": "APPROPRIATE"}, {"r1": "999999"}, self._REGISTRY
        )
        (card,) = cards
        assert card.physician_id == "999999"
        assert card.physician_name == "999999"
        assert card.ward_id == UNATTRIBUTED_DEPARTMENT_ID

    def test_department_scorecards_group_by_deptlct(self) -> None:
        cards = build_department_scorecards(
            self._VERDICTS, self._REQNO_TO_DOCTOR, self._REGISTRY
        )
        by_id = {c.ward_id: c for c in cards}
        assert by_id["1028000000"].total_orders == 3
        assert by_id["1028000000"].ward_name == "ฝ่ายอายุรศาสตร์"
        assert by_id["1029000000"].total_orders == 1
        assert by_id[UNATTRIBUTED_DEPARTMENT_ID].total_orders == 1
        assert sum(c.total_orders for c in cards) == len(self._VERDICTS)

    def test_output_sorted_by_group_id_for_determinism(self) -> None:
        cards = build_doctor_scorecards(
            self._VERDICTS, self._REQNO_TO_DOCTOR, self._REGISTRY
        )
        ids = [c.physician_id for c in cards]
        assert ids == sorted(ids)

    def test_returned_not_transfused_is_reported_but_excluded_from_rates(self) -> None:
        verdicts = {
            "r1": "INAPPROPRIATE",
            "r2": "NEEDS_REVIEW",
            "r3": "RETURNED_NOT_TRANSFUSED",
        }
        cards = build_doctor_scorecards(
            verdicts,
            {"r1": "302389", "r2": "302389", "r3": "302389"},
            self._REGISTRY,
        )
        (card,) = cards
        assert card.total_orders == 2
        assert card.inappropriate_count == 1
        assert card.needs_review_count == 1
        assert card.returned_not_transfused_count == 1
        (ranked,) = rank_top_n(
            cards,
            "inappropriate",
            group_id=lambda c: c.physician_id,
            group_name=lambda c: c.physician_name,
            min_orders=1,
        )
        assert ranked.bucket_rate == pytest.approx(0.5)
        assert ranked.returned_not_transfused_count == 1

    def test_periop_transfusion_exempt_is_reported_but_excluded_from_rates(
        self,
    ) -> None:
        verdicts = {
            "r1": "INAPPROPRIATE",
            "r2": "NEEDS_REVIEW",
            "r3": "PERIOP_TRANSFUSION_EXEMPT",
            "r4": "RETURNED_NOT_TRANSFUSED",
        }
        cards = build_doctor_scorecards(
            verdicts,
            {"r1": "302389", "r2": "302389", "r3": "302389", "r4": "302389"},
            self._REGISTRY,
        )
        (card,) = cards
        # Both excluded terminals are held out of the scorable denominator.
        assert card.total_orders == 2
        assert card.inappropriate_count == 1
        assert card.needs_review_count == 1
        assert card.periop_transfusion_exempt_count == 1
        assert card.returned_not_transfused_count == 1
        (ranked,) = rank_top_n(
            cards,
            "inappropriate",
            group_id=lambda c: c.physician_id,
            group_name=lambda c: c.physician_name,
            min_orders=1,
        )
        assert ranked.bucket_rate == pytest.approx(0.5)
        assert ranked.periop_transfusion_exempt_count == 1


# ---------------------------------------------------------------------------
# rank_top_n
# ---------------------------------------------------------------------------


class TestRankTopN:
    def _rank(
        self,
        cards: list[PhysicianScorecard],
        bucket: str = "inappropriate",
        **kwargs: object,
    ) -> tuple[RankedRow, ...]:
        return rank_top_n(
            cards,
            bucket,  # type: ignore[arg-type]
            group_id=lambda s: s.physician_id,
            group_name=lambda s: s.physician_name,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_thin_sample_cannot_outrank_qualified_rate(self) -> None:
        # The plan's canary: a 1/1 = 100% inappropriate doctor must NOT
        # be presented above a doctor with 3/6 = 50% on N >= min_orders.
        thin = _scorecard("d_thin", total=1, inappropriate=1)
        solid = _scorecard("d_solid", total=6, inappropriate=3)
        rows = self._rank([thin, solid], min_orders=5)
        assert rows[0].group_id == "d_solid"
        assert rows[0].meets_min_orders is True
        assert rows[1].group_id == "d_thin"
        assert rows[1].meets_min_orders is False

    def test_qualified_sorted_by_rate_desc(self) -> None:
        a = _scorecard("a", total=10, inappropriate=2)  # 0.2
        b = _scorecard("b", total=5, inappropriate=4)  # 0.8
        rows = self._rank([a, b], min_orders=5)
        assert [r.group_id for r in rows] == ["b", "a"]
        assert rows[0].rank == 1
        assert rows[0].bucket_rate == pytest.approx(0.8)

    def test_unqualified_sorted_by_count_desc(self) -> None:
        a = _scorecard("a", total=2, inappropriate=1)  # rate 0.5, count 1
        b = _scorecard("b", total=4, inappropriate=2)  # rate 0.5, count 2
        rows = self._rank([a, b], min_orders=5)
        assert [r.group_id for r in rows] == ["b", "a"]

    def test_ties_break_deterministically_by_group_id(self) -> None:
        a = _scorecard("a", total=6, inappropriate=3)
        b = _scorecard("b", total=6, inappropriate=3)
        rows = self._rank([b, a], min_orders=5)
        assert [r.group_id for r in rows] == ["a", "b"]

    def test_n_caps_the_output(self) -> None:
        cards = [_scorecard(f"d{i:02d}", total=1, inappropriate=1) for i in range(15)]
        rows = self._rank(cards, n=10)
        assert len(rows) == 10
        assert [r.rank for r in rows] == list(range(1, 11))

    def test_fewer_groups_than_n_returns_all(self) -> None:
        cards = [_scorecard("a", total=6, inappropriate=1)]
        rows = self._rank(cards, n=10)
        assert len(rows) == 1

    def test_unresolved_bucket_collapses_needs_review_and_insufficient(
        self,
    ) -> None:
        card = _scorecard("a", total=6, needs_review=2, insufficient=1)
        rows = self._rank([card], bucket="unresolved", min_orders=5)
        assert rows[0].bucket_count == 3
        assert rows[0].unresolved_count == 3
        assert rows[0].bucket_rate == pytest.approx(0.5)

    def test_appropriate_bucket(self) -> None:
        card = _scorecard("a", total=4, appropriate=3)
        rows = self._rank([card], bucket="appropriate")
        assert rows[0].bucket_count == 3

    def test_zero_bucket_count_groups_are_excluded(self) -> None:
        # A doctor with zero inappropriate orders cannot head the "top
        # inappropriate" table, even with qualifying N — on the real 300
        # the two only N>=5 doctors have zero inappropriate orders and
        # would otherwise outrank every actual finding.
        zero = _scorecard("a_qualified_clean", total=6, appropriate=6)
        one = _scorecard("b_thin", total=1, inappropriate=1)
        rows = self._rank([zero, one], min_orders=5)
        assert [r.group_id for r in rows] == ["b_thin"]

    def test_all_zero_counts_produce_empty_table(self) -> None:
        rows = self._rank([_scorecard("a", total=6, appropriate=6)])
        assert rows == ()

    def test_default_min_orders_is_five(self) -> None:
        # Frozen BEFORE scoring (overfit guard from the plan); changing it
        # is a clinical-review decision, not a tuning knob.
        assert DEFAULT_MIN_ORDERS == 5

    def test_dimension_helpers_expose_ids_and_names(self) -> None:
        doctor_rows = rank_doctor_scorecards(
            [_scorecard("302389", total=6, inappropriate=3, name="พญ.ส*****")],
            "inappropriate",
        )
        assert doctor_rows[0].group_id == "302389"
        assert doctor_rows[0].group_name == "พญ.ส*****"
        from bba.dashboard.models import WardScorecard

        dept_rows = rank_department_scorecards(
            [
                WardScorecard(
                    ward_id="1028000000",
                    ward_name="ฝ่ายอายุรศาสตร์",
                    total_orders=6,
                    appropriate_count=1,
                    inappropriate_count=3,
                    needs_review_count=1,
                    insufficient_evidence_count=1,
                    average_confidence=0.0,
                )
            ],
            "inappropriate",
        )
        assert dept_rows[0].group_id == "1028000000"
        assert dept_rows[0].group_name == "ฝ่ายอายุรศาสตร์"


# ---------------------------------------------------------------------------
# Outputs — CSV + standalone HTML
# ---------------------------------------------------------------------------


def _ranked_row(
    gid: str = "302389",
    name: str = "พญ.ส***** ว*****",
    rank: int = 1,
) -> RankedRow:
    return RankedRow(
        rank=rank,
        group_id=gid,
        group_name=name,
        total_orders=6,
        appropriate_count=2,
        inappropriate_count=3,
        unresolved_count=1,
        bucket="inappropriate",
        bucket_count=3,
        bucket_rate=0.5,
        meets_min_orders=True,
    )


class TestWriteRankingCsv:
    def test_writes_header_and_rows_with_unix_newlines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "bba.attribution.outputs.RETURNS_LEDGER_ENABLED", True, raising=False
        )
        out = write_ranking_csv((_ranked_row(),), tmp_path / "doctors.csv")
        text = out.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert lines[0] == (
            "rank,group_id,group_name,total_orders,appropriate,"
            "inappropriate,unresolved,returned_not_transfused,"
            "periop_transfusion_exempt,bucket,"
            "bucket_count,bucket_rate,"
            "meets_min_orders,mean_hb_g_dl,hb_order_n"
        )
        assert lines[1].startswith("1,302389,")
        assert "\r\n" not in text
        assert "0.5" in lines[1]
        assert "true" in lines[1]

    def test_flag_off_matches_pre_returns_csv_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "bba.attribution.outputs.RETURNS_LEDGER_ENABLED", False, raising=False
        )
        out = write_ranking_csv((_ranked_row(),), tmp_path / "doctors.csv")
        # The mean-trigger columns are appended unconditionally; a row with no
        # lab sample renders an empty mean cell and n=0 (spec #131/#133).
        assert (
            out.read_bytes()
            == (
                "rank,group_id,group_name,total_orders,appropriate,inappropriate,"
                "unresolved,bucket,bucket_count,bucket_rate,meets_min_orders,"
                "mean_hb_g_dl,hb_order_n\n"
                "1,302389,\u0e1e\u0e0d.\u0e2a***** \u0e27*****,6,2,3,1,inappropriate,3,0.5,true,,0\n"
            ).encode()
        )

    def test_rate_formatting_strips_trailing_zeros(self, tmp_path: Path) -> None:
        # Mirrors bba.report_generator.csv_writer float conventions so the
        # two report surfaces stay byte-consistent.
        row = _ranked_row()
        row = row.model_copy(update={"bucket_rate": 1.0 / 3.0})
        out = write_ranking_csv((row,), tmp_path / "r.csv")
        assert "0.333333" in out.read_text(encoding="utf-8")

    def test_header_and_row_have_equal_column_count_flag_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Header names and row values come from one column spec, so their
        # counts can never disagree — the guard a future appended column
        # relies on. Assert parity rather than a frozen count.
        monkeypatch.setattr(
            "bba.attribution.outputs.RETURNS_LEDGER_ENABLED", True, raising=False
        )
        out = write_ranking_csv((_ranked_row(),), tmp_path / "doctors.csv")
        header, data = out.read_text(encoding="utf-8").splitlines()[:2]
        assert len(header.split(",")) == len(data.split(","))

    def test_header_and_row_have_equal_column_count_flag_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "bba.attribution.outputs.RETURNS_LEDGER_ENABLED", False, raising=False
        )
        out = write_ranking_csv((_ranked_row(),), tmp_path / "doctors.csv")
        header, data = out.read_text(encoding="utf-8").splitlines()[:2]
        assert len(header.split(",")) == len(data.split(","))

    def test_none_cell_renders_empty(self) -> None:
        # A None value must render as an empty cell, not the literal
        # "None". No nullable ranking column exists yet (it arrives with
        # the mean-trigger columns), so the rendering contract is pinned
        # directly on the cell formatter.
        from bba.attribution.outputs import _format_cell

        assert _format_cell(None) == ""


class TestWriteRankingsHtml:
    def test_standalone_html_contains_both_tables_and_caveats(
        self, tmp_path: Path
    ) -> None:
        result = build_rankings(
            verdicts={"r1": "INAPPROPRIATE", "r2": "APPROPRIATE"},
            reqno_to_doctor={"r1": "302389", "r2": "302389"},
            dct_registry={
                "302389": DoctorRecord(
                    dct="302389",
                    display_name="พญ.ส*****",
                    deptlct="1028000000",
                    deptname="ฝ่ายอายุรศาสตร์",
                )
            },
        )
        out = write_rankings_html(
            result,
            tmp_path / "rankings.html",
            verdict_source_label="300-case human review (Sheet1 col J)",
        )
        html = out.read_text(encoding="utf-8")
        assert "<html" in html
        assert "ฝ่ายอายุรศาสตร์" in html
        assert "302389" in html
        assert str(DEFAULT_MIN_ORDERS) in html
        assert "300-case human review" in html
        # Bucket totals must be stated so a reader can reconcile.
        assert "1" in html and "2" in html

    def test_html_escapes_untrusted_names(self, tmp_path: Path) -> None:
        result = build_rankings(
            verdicts={"r1": "INAPPROPRIATE"},
            reqno_to_doctor={"r1": "302389"},
            dct_registry={
                "302389": DoctorRecord(
                    dct="302389",
                    display_name="<script>alert(1)</script>",
                    deptlct="1028000000",
                    deptname="dept",
                )
            },
        )
        out = write_rankings_html(
            result, tmp_path / "r.html", verdict_source_label="test"
        )
        html = out.read_text(encoding="utf-8")
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_flag_off_omits_returns_presentation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "bba.attribution.outputs.RETURNS_LEDGER_ENABLED", False, raising=False
        )
        result = build_rankings(
            verdicts={"r1": "INAPPROPRIATE"},
            reqno_to_doctor={"r1": "302389"},
            dct_registry={},
        )
        html = write_rankings_html(
            result, tmp_path / "rankings.html", verdict_source_label="test"
        ).read_text(encoding="utf-8")
        assert "Returned, not transfused" not in html
        assert "returned/not-transfused excluded" not in html
        assert "Peri-op transfusion (exempt)" not in html
        assert "peri-op-exempt excluded" not in html

    def test_flag_on_includes_returns_presentation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "bba.attribution.outputs.RETURNS_LEDGER_ENABLED", True, raising=False
        )
        result = build_rankings(
            verdicts={"r1": "INAPPROPRIATE"},
            reqno_to_doctor={"r1": "302389"},
            dct_registry={},
        )
        html = write_rankings_html(
            result, tmp_path / "rankings.html", verdict_source_label="test"
        ).read_text(encoding="utf-8")
        assert "Returned, not transfused" in html
        assert "returned/not-transfused excluded" in html
        assert "Peri-op transfusion (exempt)" in html
        assert "peri-op-exempt excluded" in html


# ---------------------------------------------------------------------------
# build_rankings — end-to-end over the swappable verdict source
# ---------------------------------------------------------------------------


class TestBuildRankings:
    def test_totals_reconcile_with_raw_verdict_counts(self) -> None:
        verdicts = {
            "r1": "APPROPRIATE",
            "r2": "APPROPRIATE",
            "r3": "INAPPROPRIATE",
            "r4": "NEEDS_REVIEW",
            "r5": "INSUFFICIENT_EVIDENCE",
        }
        result = build_rankings(
            verdicts=verdicts,  # type: ignore[arg-type]
            reqno_to_doctor={"r1": "d1", "r2": "d1", "r3": "d2", "r4": "d2"},
            dct_registry={},
        )
        assert result.totals.appropriate == 2
        assert result.totals.inappropriate == 1
        assert result.totals.unresolved == 2
        assert result.totals.total == 5
        # Ranked tables carry only bucket-positive groups; conservation
        # against the totals is over the scorecards (see
        # TestBuildScorecards and the real-data reconciliation), not the
        # capped tables.
        for table in (result.doctors, result.departments):
            assert table.rows, "the one inappropriate order must surface"
            assert all(r.inappropriate_count > 0 for r in table.rows)
            assert sum(r.inappropriate_count for r in table.rows) == 1

    def test_tables_carry_dimension_and_threshold_metadata(self) -> None:
        result = build_rankings(
            verdicts={"r1": "APPROPRIATE"},
            reqno_to_doctor={"r1": "d1"},
            dct_registry={},
        )
        assert result.doctors.dimension == "doctor"
        assert result.departments.dimension == "department"
        assert result.doctors.min_orders == DEFAULT_MIN_ORDERS
        assert result.doctors.bucket == "inappropriate"


# ---------------------------------------------------------------------------
# CLI seams — the concrete fix for the M0/M1 attribution blocker
# ---------------------------------------------------------------------------


class TestCliAttributionSeams:
    def test_ward_resolver_seam_wires_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from bba.cli.main import _get_ward_resolver

        bdvst = _write_bdvst_csv(tmp_path / "BDVST.csv", ["68000001,x,302389"])
        dct = _write_dct_csv(
            tmp_path / "DCT.csv",
            ["302389,พญ.,ส*****,ว*****,1028000000,ฝ่ายอายุรศาสตร์"],
        )
        monkeypatch.setenv("BBA_BDVST_CSV", str(bdvst))
        monkeypatch.setenv("BBA_DCT_CSV", str(dct))
        resolver = _get_ward_resolver()
        assert resolver(SimpleNamespace(reqno="68000001")) == "1028000000"

    def test_physician_resolver_seam_wires_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from bba.cli.main import _get_physician_resolver

        bdvst = _write_bdvst_csv(tmp_path / "BDVST.csv", ["68000001,x,302389"])
        monkeypatch.setenv("BBA_BDVST_CSV", str(bdvst))
        resolver = _get_physician_resolver()
        assert resolver(SimpleNamespace(reqno="68000001")) == "302389"

    def test_seams_fail_loud_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from bba.cli.exceptions import CliError
        from bba.cli.main import _get_physician_resolver, _get_ward_resolver

        monkeypatch.delenv("BBA_BDVST_CSV", raising=False)
        monkeypatch.delenv("BBA_DCT_CSV", raising=False)
        with pytest.raises(CliError, match="BBA_BDVST_CSV"):
            _get_physician_resolver()
        with pytest.raises(CliError, match="BBA_BDVST_CSV"):
            _get_ward_resolver()


# ---------------------------------------------------------------------------
# Integration — real 300-case workbook reconciliation (skipped when the
# review bundle is not on this machine)
# ---------------------------------------------------------------------------


_REVIEW_XLSX = Path.home() / "Downloads" / "Review การใช้เลือด.xlsx"
_BDVST_CSV = (
    Path(__file__).resolve().parents[3]
    / "Bloodbank"
    / "data"
    / "encrypted"
    / "BDVST.csv"
)
_DCT_CSV = (
    Path(__file__).resolve().parents[3] / "Bloodbank" / "data" / "raw" / "DCT.csv"
)

_real_data_available = (
    _REVIEW_XLSX.exists() and _BDVST_CSV.exists() and _DCT_CSV.exists()
)


@pytest.mark.skipif(
    not _real_data_available,
    reason="300-case review workbook / Bloodbank CSVs not present",
)
class TestRealDataReconciliation:
    """The two top-10 tables and the bucket totals must reconcile with the
    raw Sheet1 col-J counts: 162 appropriate / 32 inappropriate /
    106 unresolved. This is the feature's honest-metrics gate — a broken
    join or label map would silently redistribute counts."""

    def test_bucket_totals_match_raw_col_j_counts(self) -> None:
        verdicts = human_label_verdict_source(_REVIEW_XLSX)()
        assert len(verdicts) == 300
        result = build_rankings(
            verdicts=verdicts,
            reqno_to_doctor=load_reqno_to_doctor(_BDVST_CSV),
            dct_registry=load_dct_registry(_DCT_CSV),
        )
        assert result.totals.appropriate == 162
        assert result.totals.inappropriate == 32
        assert result.totals.unresolved == 106
        assert result.totals.total == 300

    def test_scorecard_sums_conserve_totals_in_both_dimensions(self) -> None:
        verdicts = human_label_verdict_source(_REVIEW_XLSX)()
        reqno_to_doctor = load_reqno_to_doctor(_BDVST_CSV)
        registry = load_dct_registry(_DCT_CSV)
        doctor_cards = build_doctor_scorecards(verdicts, reqno_to_doctor, registry)
        dept_cards = build_department_scorecards(verdicts, reqno_to_doctor, registry)
        for cards in (doctor_cards, dept_cards):
            assert sum(c.total_orders for c in cards) == 300
            assert sum(c.appropriate_count for c in cards) == 162
            assert sum(c.inappropriate_count for c in cards) == 32
            assert (
                sum(c.needs_review_count + c.insufficient_evidence_count for c in cards)
                == 106
            )

    def test_top_10_tables_are_capped_and_sequentially_ranked(self) -> None:
        verdicts = human_label_verdict_source(_REVIEW_XLSX)()
        reqno_to_doctor = load_reqno_to_doctor(_BDVST_CSV)
        registry = load_dct_registry(_DCT_CSV)
        result = build_rankings(
            verdicts=verdicts,
            reqno_to_doctor=reqno_to_doctor,
            dct_registry=registry,
        )
        # 32 inappropriate orders spread over well more than 10 doctors →
        # the doctor table is full.
        assert len(result.doctors.rows) == 10
        # Only ~8 departments appear in the 300 (+ the unattributed
        # sentinel), and zero-count groups are excluded from a bucket's
        # ranking, so the department table is smaller than n=10.
        dept_cards = build_department_scorecards(verdicts, reqno_to_doctor, registry)
        bucket_positive = sum(1 for c in dept_cards if c.inappropriate_count > 0)
        assert len(result.departments.rows) == min(10, bucket_positive)
        for table in (result.doctors, result.departments):
            assert [r.rank for r in table.rows] == list(range(1, len(table.rows) + 1))


# ---------------------------------------------------------------------------
# lab_stats — per-order lab join for the mean pre-transfusion trigger (#133)
# ---------------------------------------------------------------------------


class TestGroupLabStats:
    def test_empty_group_has_no_mean(self) -> None:
        stats = GroupLabStats()
        assert stats.mean_hb is None
        assert stats.hb_order_n == 0

    def test_populated_group_carries_mean_and_n(self) -> None:
        stats = GroupLabStats(mean_hb=8.4, hb_order_n=5)
        assert stats.mean_hb == 8.4
        assert stats.hb_order_n == 5

    def test_mean_without_sample_fails_loud(self) -> None:
        # n == 0 must imply mean is None — a 0.0 trigger on no data would
        # misrepresent the group as transfusing at an unknown-low threshold.
        with pytest.raises(ValueError, match="invariant"):
            GroupLabStats(mean_hb=8.0, hb_order_n=0)

    def test_sample_without_mean_fails_loud(self) -> None:
        with pytest.raises(ValueError, match="invariant"):
            GroupLabStats(mean_hb=None, hb_order_n=2)


class TestLoadOrderLabs:
    def test_loads_red_cell_hb_keyed_by_reqno(self, tmp_path: Path) -> None:
        path = _write_report_csv(
            tmp_path / "report.csv",
            [_red_cell("68000001", "8.4"), _red_cell("68000002", "9.1")],
        )
        labs = load_order_labs(path)
        assert labs["68000001"].hb_value_g_dl == 8.4
        assert labs["68000002"].hb_value_g_dl == 9.1

    def test_blank_or_unknown_component_contributes_no_hb(self, tmp_path: Path) -> None:
        # A plausible numeric Hb on a non-red-cell row must not count —
        # strict component match, not merely "row not empty".
        path = _write_report_csv(
            tmp_path / "report.csv",
            [
                {
                    "reqno": "68000001",
                    "component": "",
                    "hb_value_g_dl": "8.4",
                    "hb_freshness": "fresh",
                },
                {
                    "reqno": "68000002",
                    "component": "platelet",
                    "hb_value_g_dl": "9.1",
                    "hb_freshness": "fresh",
                },
            ],
        )
        labs = load_order_labs(path)
        assert labs["68000001"].hb_value_g_dl is None
        assert labs["68000002"].hb_value_g_dl is None

    def test_missing_freshness_sentinel_excluded(self, tmp_path: Path) -> None:
        path = _write_report_csv(
            tmp_path / "report.csv",
            [_red_cell("68000001", "8.4", freshness="missing")],
        )
        assert load_order_labs(path)["68000001"].hb_value_g_dl is None

    def test_out_of_range_value_excluded(self, tmp_path: Path) -> None:
        # Below 2 and above 25 g/dL are corrupt readings, not triggers.
        path = _write_report_csv(
            tmp_path / "report.csv",
            [_red_cell("68000001", "1.5"), _red_cell("68000002", "30.0")],
        )
        labs = load_order_labs(path)
        assert labs["68000001"].hb_value_g_dl is None
        assert labs["68000002"].hb_value_g_dl is None

    def test_unparseable_and_non_finite_tolerated_as_absent(
        self, tmp_path: Path
    ) -> None:
        path = _write_report_csv(
            tmp_path / "report.csv",
            [
                _red_cell("68000001", "n/a"),
                _red_cell("68000002", "inf"),
                _red_cell("68000003", "nan"),
            ],
        )
        labs = load_order_labs(path)
        assert labs["68000001"].hb_value_g_dl is None
        assert labs["68000002"].hb_value_g_dl is None
        assert labs["68000003"].hb_value_g_dl is None

    def test_missing_required_column_fails_loud(self, tmp_path: Path) -> None:
        bad = tmp_path / "report.csv"
        bad.write_text("reqno,component,hb_value_g_dl\n68000001,red_cell,8.4\n")
        with pytest.raises(ValueError, match="hb_freshness"):
            load_order_labs(bad)

    def test_conflicting_duplicate_reqno_fails_loud(self, tmp_path: Path) -> None:
        path = _write_report_csv(
            tmp_path / "report.csv",
            [_red_cell("68000001", "8.4"), _red_cell("68000001", "9.9")],
        )
        with pytest.raises(ValueError, match="two different lab records"):
            load_order_labs(path)

    def test_identical_duplicate_reqno_tolerated(self, tmp_path: Path) -> None:
        path = _write_report_csv(
            tmp_path / "report.csv",
            [_red_cell("68000001", "8.4"), _red_cell("68000001", "8.4")],
        )
        assert load_order_labs(path)["68000001"].hb_value_g_dl == 8.4

    def test_rows_without_reqno_are_skipped(self, tmp_path: Path) -> None:
        path = _write_report_csv(
            tmp_path / "report.csv",
            [_red_cell("", "8.4"), _red_cell("68000002", "9.1")],
        )
        labs = load_order_labs(path)
        assert "" not in labs
        assert labs["68000002"].hb_value_g_dl == 9.1


# A two-doctor / shared-department scenario exercising the scorable-cohort
# restriction, the unattributed fallback, and an out-of-cohort lab row.
def _mean_hb_scenario() -> tuple[
    dict[str, str], dict[str, str], dict[str, DoctorRecord], dict[str, OrderLabValue]
]:
    verdicts = {
        "r1": "INAPPROPRIATE",  # d1
        "r2": "APPROPRIATE",  # d1
        "r3": "INAPPROPRIATE",  # d2
        "r4": "RETURNED_NOT_TRANSFUSED",  # d1 — non-scorable, hb must not count
        "r5": "INAPPROPRIATE",  # unattributed doctor / department
    }
    reqno_to_doctor = {"r1": "d1", "r2": "d1", "r3": "d2", "r4": "d1"}
    dct_registry = {
        "d1": DoctorRecord(
            dct="d1", display_name="D1", deptlct="DEPT", deptname="Dept X"
        ),
        "d2": DoctorRecord(
            dct="d2", display_name="D2", deptlct="DEPT", deptname="Dept X"
        ),
    }
    order_labs = {
        "r1": OrderLabValue(reqno="r1", component="red_cell", hb_value_g_dl=8.0),
        "r2": OrderLabValue(reqno="r2", component="red_cell", hb_value_g_dl=10.0),
        "r3": OrderLabValue(reqno="r3", component="red_cell", hb_value_g_dl=7.0),
        "r4": OrderLabValue(reqno="r4", component="red_cell", hb_value_g_dl=12.0),
        "r5": OrderLabValue(reqno="r5", component="red_cell", hb_value_g_dl=6.0),
        "r99": OrderLabValue(reqno="r99", component="red_cell", hb_value_g_dl=25.0),
    }
    return verdicts, reqno_to_doctor, dct_registry, order_labs


class TestAggregateLabStats:
    def test_doctor_means_exclude_non_scorable_and_out_of_cohort(self) -> None:
        verdicts, reqno_to_doctor, _registry, order_labs = _mean_hb_scenario()
        stats = aggregate_doctor_lab_stats(verdicts, reqno_to_doctor, order_labs)
        # d1: r1(8) + r2(10); the returned r4(12) and out-of-cohort r99 excluded.
        assert stats["d1"].mean_hb == pytest.approx(9.0)
        assert stats["d1"].hb_order_n == 2
        assert stats["d2"].mean_hb == pytest.approx(7.0)
        assert stats["d2"].hb_order_n == 1
        assert stats[UNATTRIBUTED_DOCTOR_ID].mean_hb == pytest.approx(6.0)
        assert stats[UNATTRIBUTED_DOCTOR_ID].hb_order_n == 1

    def test_department_means_pool_over_the_shared_department(self) -> None:
        verdicts, reqno_to_doctor, registry, order_labs = _mean_hb_scenario()
        stats = aggregate_department_lab_stats(
            verdicts, reqno_to_doctor, registry, order_labs
        )
        # DEPT pools d1's r1,r2 and d2's r3 (returned r4 excluded): (8+10+7)/3.
        assert stats["DEPT"].mean_hb == pytest.approx(25.0 / 3.0)
        assert stats["DEPT"].hb_order_n == 3
        assert stats[UNATTRIBUTED_DEPARTMENT_ID].mean_hb == pytest.approx(6.0)
        assert stats[UNATTRIBUTED_DEPARTMENT_ID].hb_order_n == 1


class TestBuildRankingsMeanHb:
    def test_assembly_threads_mean_hb_onto_both_tables(self) -> None:
        verdicts, reqno_to_doctor, registry, order_labs = _mean_hb_scenario()
        result = build_rankings(
            verdicts=verdicts,
            reqno_to_doctor=reqno_to_doctor,
            dct_registry=registry,
            order_labs=order_labs,
            min_orders=1,
        )
        doctors = {row.group_id: row for row in result.doctors.rows}
        assert doctors["d1"].mean_hb == pytest.approx(9.0)
        assert doctors["d1"].hb_order_n == 2
        # n never exceeds Orders (N): d1's returned order is out of both.
        assert doctors["d1"].hb_order_n <= doctors["d1"].total_orders
        assert doctors["d2"].mean_hb == pytest.approx(7.0)
        assert doctors[UNATTRIBUTED_DOCTOR_ID].mean_hb == pytest.approx(6.0)

        departments = {row.group_id: row for row in result.departments.rows}
        assert departments["DEPT"].mean_hb == pytest.approx(25.0 / 3.0)
        assert departments["DEPT"].hb_order_n == 3

    def test_omitting_order_labs_leaves_mean_fields_defaulted(self) -> None:
        # Existing callers pass no lab mapping and must be unaffected.
        verdicts, reqno_to_doctor, registry, _labs = _mean_hb_scenario()
        result = build_rankings(
            verdicts=verdicts,
            reqno_to_doctor=reqno_to_doctor,
            dct_registry=registry,
            min_orders=1,
        )
        for table in (result.doctors, result.departments):
            for row in table.rows:
                assert row.mean_hb is None
                assert row.hb_order_n == 0

    def test_rank_top_n_without_group_stats_defaults_mean(self) -> None:
        card = PhysicianScorecard(
            physician_id="d1",
            physician_name="D1",
            ward_id="DEPT",
            total_orders=3,
            appropriate_count=1,
            inappropriate_count=2,
            needs_review_count=0,
            insufficient_evidence_count=0,
            average_confidence=0.0,
        )
        rows = rank_doctor_scorecards((card,), "inappropriate", min_orders=1)
        assert rows[0].mean_hb is None
        assert rows[0].hb_order_n == 0


class TestMeanHbOutput:
    def test_csv_appends_populated_mean_and_count(self, tmp_path: Path) -> None:
        verdicts, reqno_to_doctor, registry, order_labs = _mean_hb_scenario()
        result = build_rankings(
            verdicts=verdicts,
            reqno_to_doctor=reqno_to_doctor,
            dct_registry=registry,
            order_labs=order_labs,
            min_orders=1,
        )
        out = write_ranking_csv(result.doctors.rows, tmp_path / "doctors.csv")
        text = out.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert lines[0].endswith(",mean_hb_g_dl,hb_order_n")
        d1_line = next(line for line in lines[1:] if line.split(",")[1] == "d1")
        # d1 mean is 9.0 over 2 orders — raw values appended at the end.
        assert d1_line.endswith(",9.0,2")

    def test_html_renders_mean_with_n_and_emdash_and_caveat(
        self, tmp_path: Path
    ) -> None:
        verdicts, reqno_to_doctor, registry, order_labs = _mean_hb_scenario()
        # d2 orders only platelets? No — give a doctor with no usable Hb by
        # dropping its lab so the em-dash path renders.
        order_labs = dict(order_labs)
        del order_labs["r3"]  # d2 now has no usable Hb -> em-dash
        result = build_rankings(
            verdicts=verdicts,
            reqno_to_doctor=reqno_to_doctor,
            dct_registry=registry,
            order_labs=order_labs,
            min_orders=1,
        )
        html = write_rankings_html(
            result, tmp_path / "r.html", verdict_source_label="test"
        ).read_text(encoding="utf-8")
        assert "Mean Hb (g/dL)" in html
        assert "9.0 (n=2)" in html  # d1
        assert "&mdash;" in html  # d2, no usable Hb
        assert "mean pre-transfusion" in html  # caveat sentence
