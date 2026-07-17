"""Synthetic seam tests for the offline MSBOS name-match study."""

from __future__ import annotations

import csv
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType, ModuleType
from uuid import uuid4

import pytest

from bba.cohort_detector import OperativeEvent
from bba.preop_reservation.name_match import _index_from_rows

_REPORT_FIELDS = [
    "reqno",
    "an",
    "order_datetime_utc",
    "component",
    "classification",
    "msbos_reason",
    "msbos_reserved_units",
    "msbos_token",
    "msbos_recommended_units",
    "msbos_resolved_icd9",
    "msbos_reference_hash",
]

_EXPECTED_OUTPUT_FIELDS = [
    "row_kind",
    "reason",
    "source_icd9",
    "reqno",
    "an",
    "order_datetime_utc",
    "reserved_units",
    "icd10_diagnosis",
    "events_scope",
    "event_names",
    "tier",
    "match_status",
    "representative_operation",
    "matched_operations",
    "matched_event_name",
    "matched_event_datetime",
    "matched_event_hours_from_order",
    "matched_specialty",
    "matched_procedure_group",
    "recommendation_token",
    "recommendation_units",
    "would_be_reason",
    "would_be_is_over",
    "distinct_recommendation_count",
    "code_recommendation",
    "control_score",
    "tier2_confidence",
    "tier2_raw_suggestion",
    "reference_hash",
]


def _load_study(monkeypatch: pytest.MonkeyPatch, work: Path) -> ModuleType:
    monkeypatch.setenv("BBA_PILOT_WORK_DIR", str(work))
    pilot_dir = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
    if str(pilot_dir) not in sys.path:
        sys.path.insert(0, str(pilot_dir))
    path = pilot_dir / "msbos_name_match_study.py"
    spec = importlib.util.spec_from_file_location(
        f"pilot_msbos_name_match_study_{uuid4().hex}", path
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _reference_rows() -> list[dict[str, str]]:
    return [
        {
            "operation": "Synthetic alpha resection",
            "msbos": "G/M",
            "recommended_units": "2",
            "specialty": "Synthetic specialty A",
            "procedure_group": "Synthetic group 1",
        },
        {
            "operation": "Synthetic beta bypass",
            "msbos": "G/M",
            "recommended_units": "4",
            "specialty": "Synthetic specialty B",
            "procedure_group": "Synthetic group 2",
        },
        {
            "operation": "Synthetic gamma reconstruction",
            "msbos": "none",
            "recommended_units": "0",
            "specialty": "Synthetic specialty C",
            "procedure_group": "Synthetic group 3",
        },
    ]


def _study_reference(mod: ModuleType, content_hash: str = "synthetic-hash") -> object:
    rows = _reference_rows()
    return mod.StudyReference(
        index=_index_from_rows(rows, content_hash=content_hash),
        content_hash=content_hash,
        metadata=MappingProxyType(mod._metadata_from_rows(rows)),
    )


def _report_row(
    *,
    reqno: str = "SYN-REQ-1",
    an: str = "SYN-AN-1",
    reason: str = "unresolved_code",
    order_datetime: str = "2026-07-08T01:00:00+00:00",
    reserved_units: str = "3",
    token: str = "",
    recommended_units: str = "0",
    source_icd9: str = "SYN-CODE-X",
    content_hash: str = "synthetic-hash",
) -> dict[str, str]:
    return {
        "reqno": reqno,
        "an": an,
        "order_datetime_utc": order_datetime,
        "component": "red_cell",
        "classification": "RETURNED_NOT_TRANSFUSED",
        "msbos_reason": reason,
        "msbos_reserved_units": reserved_units,
        "msbos_token": token,
        "msbos_recommended_units": recommended_units,
        "msbos_resolved_icd9": source_icd9,
        "msbos_reference_hash": content_hash,
    }


def _valid_bundle(work: Path) -> None:
    bundle = work / "bundle"
    _write_csv(
        bundle / "IPTSUMOPRT.csv",
        [],
        ["An", "Icd9cm", "Indate", "Intime", "Orflag"],
    )
    _write_csv(bundle / "ICD9CM.csv", [], ["Icd9cm", "Name", "Orflag"])


def test_case_consumes_matcher_and_preserves_signed_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_study(monkeypatch, tmp_path)
    reference = _study_reference(mod)
    order = datetime(2026, 7, 8, 1, tzinfo=timezone.utc)
    upcoming = OperativeEvent(
        icd9="SYN-A",
        or_flag=True,
        operative_datetime=order + timedelta(hours=2),
        name="Synthetic alpha resection",
    )
    past = OperativeEvent(
        icd9="SYN-A",
        or_flag=True,
        operative_datetime=order - timedelta(hours=5),
        name="Synthetic alpha resection",
    )

    upcoming_row = mod._study_case(_report_row(), (upcoming,), (), reference=reference)
    past_row = mod._study_case(
        _report_row(reason="no_planned_op"), (past,), (), reference=reference
    )

    assert upcoming_row["match_status"] == "matched"
    assert upcoming_row["recommendation_token"] == "G/M"
    assert upcoming_row["recommendation_units"] == "2"
    assert upcoming_row["would_be_reason"] == "over_gm_excess"
    assert upcoming_row["would_be_is_over"] == "True"
    assert upcoming_row["matched_event_hours_from_order"] == "2.0"
    assert past_row["events_scope"] == "all_events"
    assert past_row["matched_event_hours_from_order"] == "-5.0"


@pytest.mark.parametrize(
    ("bad_hash", "message"),
    [
        ("stale-synthetic-hash", "stale-synthetic-hash"),
        ("", "missing provenance"),
    ],
)
def test_preflight_rejects_stale_and_blank_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_hash: str,
    message: str,
) -> None:
    mod = _load_study(monkeypatch, tmp_path)
    _valid_bundle(tmp_path)
    row = _report_row(content_hash=bad_hash)

    with pytest.raises(mod.StudyPreflightError, match=message):
        mod._preflight([row], _REPORT_FIELDS, expected_hash="synthetic-hash")


def test_preflight_rejects_missing_report_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_study(monkeypatch, tmp_path)
    _valid_bundle(tmp_path)
    fields = [field for field in _REPORT_FIELDS if field != "msbos_resolved_icd9"]

    with pytest.raises(mod.StudyPreflightError, match="msbos_resolved_icd9"):
        mod._preflight([_report_row()], fields, expected_hash="synthetic-hash")


def _write_end_to_end_bundle(work: Path) -> None:
    event_rows = [
        {
            "An": "SYN-AN-MATCH",
            "Icd9cm": "SYN1",
            "Indate": "July 7, 2026, 12:00 AM",
            "Intime": "100000",
            "Orflag": "1",
        },
        {
            "An": "SYN-AN-CONFLICT",
            "Icd9cm": "SYN2",
            "Indate": "July 8, 2026, 12:00 AM",
            "Intime": "100000",
            "Orflag": "1",
        },
        {
            "An": "SYN-AN-CONFLICT",
            "Icd9cm": "SYN3",
            "Indate": "July 8, 2026, 12:00 AM",
            "Intime": "110000",
            "Orflag": "1",
        },
        {
            "An": "SYN-AN-NONE",
            "Icd9cm": "SYN4",
            "Indate": "July 8, 2026, 12:00 AM",
            "Intime": "120000",
            "Orflag": "1",
        },
        {
            "An": "SYN-AN-AGREE",
            "Icd9cm": "SYN1",
            "Indate": "July 8, 2026, 12:00 AM",
            "Intime": "130000",
            "Orflag": "1",
        },
        {
            "An": "SYN-AN-DISAGREE",
            "Icd9cm": "SYN2",
            "Indate": "July 8, 2026, 12:00 AM",
            "Intime": "140000",
            "Orflag": "1",
        },
    ]
    _write_csv(
        work / "bundle" / "IPTSUMOPRT.csv",
        event_rows,
        ["An", "Icd9cm", "Indate", "Intime", "Orflag"],
    )
    _write_csv(
        work / "bundle" / "ICD9CM.csv",
        [
            {"Icd9cm": "SYN1", "Name": "Synthetic alpha resection", "Orflag": "1"},
            {"Icd9cm": "SYN2", "Name": "Synthetic beta bypass", "Orflag": "1"},
            {
                "Icd9cm": "SYN3",
                "Name": "Synthetic gamma reconstruction",
                "Orflag": "1",
            },
            {"Icd9cm": "SYN4", "Name": "Synthetic delta procedure", "Orflag": "1"},
        ],
        ["Icd9cm", "Name", "Orflag"],
    )
    _write_csv(
        work / "bundle" / "Diagnosis.csv",
        [
            {"AN": an, "ICD10": f"SYN-DX-{number}"}
            for number, an in enumerate(
                [
                    "SYN-AN-MATCH",
                    "SYN-AN-CONFLICT",
                    "SYN-AN-NONE",
                    "SYN-AN-AGREE",
                    "SYN-AN-DISAGREE",
                ],
                start=1,
            )
        ],
        ["AN", "ICD10"],
    )


def _write_end_to_end_report(work: Path) -> None:
    rows = [
        _report_row(
            reqno="SYN-REQ-MATCH",
            an="SYN-AN-MATCH",
            reason="no_planned_op",
        ),
        _report_row(
            reqno="SYN-REQ-CONFLICT",
            an="SYN-AN-CONFLICT",
            reason="ambiguous_code",
        ),
        _report_row(reqno="SYN-REQ-NONE", an="SYN-AN-NONE"),
        _report_row(
            reqno="SYN-REQ-AGREE",
            an="SYN-AN-AGREE",
            reason="within_recommendation",
            reserved_units="2",
            token="G/M",
            recommended_units="2",
            source_icd9="SYN-CODE-A",
        ),
        _report_row(
            reqno="SYN-REQ-DISAGREE",
            an="SYN-AN-DISAGREE",
            reason="over_none",
            reserved_units="1",
            token="none",
            recommended_units="0",
            source_icd9="SYN-CODE-B",
        ),
    ]
    _write_csv(work / "report.csv", rows, _REPORT_FIELDS)


def test_end_to_end_writes_exact_csv_and_scores_control_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_end_to_end_bundle(tmp_path)
    _write_end_to_end_report(tmp_path)
    mod = _load_study(monkeypatch, tmp_path)
    reference = _study_reference(mod)

    result = mod.run_study(reference=reference)
    output = mod.write_study_csv(result.rows)
    summary = mod.format_summary(result, output_path=output)

    with output.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = list(reader)
        assert reader.fieldnames == _EXPECTED_OUTPUT_FIELDS
    by_reqno = {row["reqno"]: row for row in written}
    assert len(written) == 5
    assert by_reqno["SYN-REQ-MATCH"]["match_status"] == "matched"
    assert by_reqno["SYN-REQ-MATCH"]["events_scope"] == "all_events"
    assert float(by_reqno["SYN-REQ-MATCH"]["matched_event_hours_from_order"]) < 0
    assert (
        by_reqno["SYN-REQ-MATCH"]["representative_operation"]
        == "Synthetic alpha resection"
    )
    assert by_reqno["SYN-REQ-CONFLICT"]["match_status"] == (
        "conflicting_recommendations"
    )
    assert by_reqno["SYN-REQ-CONFLICT"]["matched_operations"] == (
        "Synthetic beta bypass|Synthetic gamma reconstruction"
    )
    assert by_reqno["SYN-REQ-NONE"]["match_status"] == "no_match"
    assert by_reqno["SYN-REQ-AGREE"]["control_score"] == "agree"
    assert by_reqno["SYN-REQ-DISAGREE"]["control_score"] == "disagree"
    assert all(row["tier2_confidence"] == "" for row in written)
    assert all(row["tier2_raw_suggestion"] == "" for row in written)
    assert result.study_bucket_counts == {
        "ambiguous_code": 1,
        "ambiguous_planned_op": 0,
        "no_planned_op": 1,
        "unresolved_code": 1,
    }
    assert result.control_counts == {
        "agree": 1,
        "conflict": 0,
        "disagree": 1,
        "no_match": 0,
    }
    assert result.agreement_rate == 0.5
    assert result.gate_line == (
        "GATE: FAIL (rate=50.00%, n=2) -- LOW SAMPLE, not a robust gate"
    )
    assert result.gate_line in summary


def test_control_gate_is_na_when_matcher_declines_every_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_study(monkeypatch, tmp_path)

    line = mod.control_gate_line(
        {"agree": 0, "disagree": 0, "no_match": 2, "conflict": 1}
    )

    assert line == "GATE: N/A (0 name-matched control rows to score)"


def _event(name: str, order: datetime, *, hours: float) -> OperativeEvent:
    return OperativeEvent(
        icd9="SYN-CODE",
        or_flag=True,
        operative_datetime=order + timedelta(hours=hours),
        name=name,
    )


def test_upcoming_scope_excludes_past_events_for_non_no_planned_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Guards the scope distinction: an all-buckets-use-all_events regression would
    # let the past matching event surface and this must fail.
    mod = _load_study(monkeypatch, tmp_path)
    reference = _study_reference(mod)
    order = datetime(2026, 7, 8, 1, tzinfo=timezone.utc)
    past_match = _event("Synthetic alpha resection", order, hours=-5)
    upcoming_nonmatch = _event("Zulu unrelated procedure", order, hours=2)

    row = mod._study_case(
        _report_row(reason="unresolved_code"),
        (past_match, upcoming_nonmatch),
        (),
        reference=reference,
    )

    assert row["events_scope"] == "upcoming"
    assert row["match_status"] == "no_match"
    assert "Synthetic alpha resection" not in row["matched_operations"]


def test_duplicate_matched_name_picks_nearest_then_earliest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_study(monkeypatch, tmp_path)
    reference = _study_reference(mod)
    order = datetime(2026, 7, 8, 1, tzinfo=timezone.utc)

    near = _event("Synthetic alpha resection", order, hours=2)
    far = _event("Synthetic alpha resection", order, hours=5)
    nearest_row = mod._study_case(
        _report_row(reason="unresolved_code"), (far, near), (), reference=reference
    )
    assert nearest_row["matched_event_hours_from_order"] == "2.0"

    before = _event("Synthetic alpha resection", order, hours=-3)
    after = _event("Synthetic alpha resection", order, hours=3)
    tie_row = mod._study_case(
        _report_row(reason="no_planned_op"), (after, before), (), reference=reference
    )
    # Equal absolute distance -> earliest datetime wins the tie (a negative value).
    assert tie_row["matched_event_hours_from_order"] == "-3.0"


def _t3_reference_rows() -> list[dict[str, str]]:
    return [
        {
            "operation": "Alpha op",
            "msbos": "G/M",
            "recommended_units": "2",
            "specialty": "A",
            "procedure_group": "1",
        },
        {
            "operation": "Beta op",
            "msbos": "none",
            "recommended_units": "0",
            "specialty": "B",
            "procedure_group": "2",
        },
        {
            "operation": "Screen op",
            "msbos": "T/S",
            "recommended_units": "1",
            "specialty": "C",
            "procedure_group": "3",
        },
    ]


def _t3_reference(mod: ModuleType) -> object:
    rows = _t3_reference_rows()
    return mod.StudyReference(
        index=_index_from_rows(rows, content_hash="t3-hash"),
        content_hash="t3-hash",
        metadata=MappingProxyType(mod._metadata_from_rows(rows)),
    )


def test_agreement_rate_denominator_excludes_declines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Controls span agree (G/M + T/S-zero), disagree, no_match and conflict, so
    # total_controls (5) != agree+disagree (3). A regression to agree/total would
    # give 2/5=0.4 and fail these assertions.
    codes = {
        "C1": "Alpha op",
        "C2": "Beta op",
        "C3": "Screen op",
        "C4": "Zulu unrelated op",
    }
    event_rows = [
        {"An": "AN-AGREE-GM", "Icd9cm": "C1"},
        {"An": "AN-AGREE-TS", "Icd9cm": "C3"},
        {"An": "AN-DISAGREE", "Icd9cm": "C1"},
        {"An": "AN-NO", "Icd9cm": "C4"},
        {"An": "AN-CONFLICT", "Icd9cm": "C1"},
        {"An": "AN-CONFLICT", "Icd9cm": "C2"},
    ]
    for row in event_rows:
        row.update(
            {"Indate": "July 9, 2026, 12:00 AM", "Intime": "100000", "Orflag": "1"}
        )
    _write_csv(
        tmp_path / "bundle" / "IPTSUMOPRT.csv",
        event_rows,
        ["An", "Icd9cm", "Indate", "Intime", "Orflag"],
    )
    _write_csv(
        tmp_path / "bundle" / "ICD9CM.csv",
        [{"Icd9cm": code, "Name": name, "Orflag": "1"} for code, name in codes.items()],
        ["Icd9cm", "Name", "Orflag"],
    )
    report_rows = [
        _report_row(
            reqno="R-AGREE-GM",
            an="AN-AGREE-GM",
            reason="within_recommendation",
            reserved_units="2",
            token="G/M",
            recommended_units="2",
            content_hash="t3-hash",
        ),
        _report_row(
            reqno="R-AGREE-TS",
            an="AN-AGREE-TS",
            reason="type_and_screen_screen_only",
            reserved_units="0",
            token="T/S",
            recommended_units="0",
            content_hash="t3-hash",
        ),
        _report_row(
            reqno="R-DISAGREE",
            an="AN-DISAGREE",
            reason="over_none",
            reserved_units="1",
            token="none",
            recommended_units="0",
            content_hash="t3-hash",
        ),
        _report_row(
            reqno="R-NO",
            an="AN-NO",
            reason="over_gm_excess",
            reserved_units="9",
            token="G/M",
            recommended_units="1",
            content_hash="t3-hash",
        ),
        _report_row(
            reqno="R-CONFLICT",
            an="AN-CONFLICT",
            reason="over_gm_excess",
            reserved_units="9",
            token="G/M",
            recommended_units="1",
            content_hash="t3-hash",
        ),
    ]
    _write_csv(tmp_path / "report.csv", report_rows, _REPORT_FIELDS)
    mod = _load_study(monkeypatch, tmp_path)

    result = mod.run_study(reference=_t3_reference(mod))

    assert result.control_counts == {
        "agree": 2,
        "disagree": 1,
        "no_match": 1,
        "conflict": 1,
    }
    assert result.agreement_rate == pytest.approx(2 / 3)
    assert (
        result.gate_line
        == "GATE: FAIL (rate=66.67%, n=3) -- LOW SAMPLE, not a robust gate"
    )


def test_out_of_scope_rows_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _valid_bundle(tmp_path)
    rows = [
        _report_row(reqno="R-KEEP", an="AN-KEEP", reason="unresolved_code"),
        _report_row(reqno="R-NONTERMINAL", an="AN-1", reason="unresolved_code")
        | {"classification": "ISSUED_TRANSFUSED"},
        _report_row(reqno="R-LOOKUP", an="AN-2", reason="reservation_lookup_miss"),
        _report_row(reqno="R-OPUNRES", an="AN-3", reason="operation_unresolved"),
    ]
    _write_csv(tmp_path / "report.csv", rows, _REPORT_FIELDS)
    mod = _load_study(monkeypatch, tmp_path)

    result = mod.run_study(reference=_study_reference(mod))

    reqnos = {row["reqno"] for row in result.rows}
    assert reqnos == {"R-KEEP"}
    assert sum(result.study_bucket_counts.values()) == 1
    assert sum(result.control_counts.values()) == 0


def test_preflight_rejects_non_utc_and_naive_datetimes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_study(monkeypatch, tmp_path)
    _valid_bundle(tmp_path)
    for bad in ("2026-07-08T01:00:00", "2026-07-08T01:00:00+07:00"):
        with pytest.raises(mod.StudyPreflightError, match="timezone-aware UTC|invalid"):
            mod._preflight(
                [_report_row(order_datetime=bad)],
                _REPORT_FIELDS,
                expected_hash="synthetic-hash",
            )


def test_preflight_rejects_missing_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _valid_bundle(tmp_path)
    mod = _load_study(monkeypatch, tmp_path)
    with pytest.raises(mod.StudyPreflightError, match="report.csv"):
        mod.run_study(reference=_study_reference(mod))


def test_preflight_rejects_missing_bundle_and_required_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_study(monkeypatch, tmp_path)
    with pytest.raises(mod.StudyPreflightError, match="bundle directory"):
        mod._preflight([_report_row()], _REPORT_FIELDS, expected_hash="synthetic-hash")
    # Bundle dir present but ICD9CM.csv absent.
    _write_csv(
        tmp_path / "bundle" / "IPTSUMOPRT.csv",
        [],
        ["An", "Icd9cm", "Indate", "Intime", "Orflag"],
    )
    with pytest.raises(mod.StudyPreflightError, match="ICD9CM.csv"):
        mod._preflight([_report_row()], _REPORT_FIELDS, expected_hash="synthetic-hash")


def test_preflight_rejects_malformed_bundle_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_study(monkeypatch, tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    _write_csv(bundle / "ICD9CM.csv", [], ["Icd9cm", "Name", "Orflag"])
    # Valid header, corrupt body (invalid UTF-8) that a header-only check misses.
    (bundle / "IPTSUMOPRT.csv").write_bytes(
        b"An,Icd9cm,Indate,Intime,Orflag\nSYN,C1,\xff\xfe,100000,1\n"
    )
    with pytest.raises(mod.StudyPreflightError, match="malformed"):
        mod._preflight([_report_row()], _REPORT_FIELDS, expected_hash="synthetic-hash")
