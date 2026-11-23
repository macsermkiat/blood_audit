"""The deterministic pilot leg writes one report row per issue event (#276).

Each issue of blood is judged against the order that caused it, on the labs at
the moment it left the bank, so a REQNO the blood bank kept issuing under
yields one row per issue event. These tests run ``run_pipeline.main()`` over
small synthetic bundles shaped like the gold cases (case 21 = 68004408,
68000024) and read back report.csv and events_manifest.csv.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from bba.cohort_detector import OperativeEvent
from bba.platelet_lookup import PLATELET_LABEXM


def _load_run_pipeline() -> ModuleType:
    pilot_dir = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
    if str(pilot_dir) not in sys.path:
        sys.path.insert(0, str(pilot_dir))
    spec = importlib.util.spec_from_file_location(
        "pilot_run_pipeline_events_test", pilot_dir / "run_pipeline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RP = _load_run_pipeline()

T0 = datetime(2025, 1, 10, 8, 0)  # Bangkok wall clock of the first order
TABLES = {
    "BDVST.csv": "HN,REQNO,AN,BDVSTST,REQTYPE,CANCELDATE,REQDATE,REQTIME,"
    "BDVSTDATE,BDVSTTIME,PICKDATE,PICKTIME,ICD10,DIAGNOSIS",
    "BDVSTDT.csv": "REQNO,HN,BDVSTDATE,BDVSTTIME,USEDATE,USETIME,BDTYPE,ITEMNO,"
    "UNITAMT,USETYPE",
    "BDVSTTRANS.csv": "REQNO,HN,AN,BDTYPE,DNRNO,SEQNO,UNITSTAT,PAYDATE,PAYTIME,"
    "RTNDATE,RTNTIME,GIVEDATE,GIVETIME",
    "Lab.csv": "HN,AN,LVSTDATE,LVSTTIME,LABEXM,RESULT",
    "Diagnosis.csv": "HN,AN,ICD10",
    "Med.csv": "HN,AN,PRSCDATE,PRSCTIME,NAME_MEDITEM,NAME_GENERIC",
    "IPTSUMOPRT.csv": "An,Icd9cm,Orflag,Indate,Intime",
    "INCPT_OPRTACT.csv": "HN,AN,INCDATE,INCTIME,ORDERCODE,INCOME,CANCELDATE,"
    "INCGRP,O__OPRTACT",
    "ICD9CM.csv": "Icd9cm,Name,Orflag",
}


def _day(hours: float) -> str:
    return f"{(T0 + timedelta(hours=hours)):%Y-%m-%d} 00:00:00.000"


def _time(hours: float) -> str:
    return f"{(T0 + timedelta(hours=hours)):%H%M%S}"


def _csv_cell(value: str) -> str:
    return f'"{value}"' if "," in value else value


@dataclass
class Bundle:
    """Synthetic bundle rows; audited orders also go to BDVST.csv/BDVSTDT.csv.

    As ``sample_bundle.py`` cuts it, only BDVSTDT_RELATED.csv carries the
    emergency codes.
    """

    rows: dict[str, list[str]] = field(default_factory=dict)
    related: list[str] = field(default_factory=list)
    related_dt: list[str] = field(default_factory=list)
    doctors: dict[str, str] = field(default_factory=dict)
    pending: bool = True

    def __post_init__(self) -> None:
        if self.pending:
            self.order("P0", an="A0", status="1", audited=False)

    def add(self, table: str, *cells: str) -> None:
        self.rows.setdefault(table, []).append(",".join(map(_csv_cell, cells)))

    def order(
        self,
        reqno: str,
        *,
        an: str = "A1",
        hours: float = 0.0,
        status: str = "4",
        usetype: str = "1",
        product: str = "LPRC",
        unitamt: str = "1",
        doctor: str = "D1",
        emrtype: str = "",
        audited: bool = True,
    ) -> None:
        row = ",".join(
            ["H1", reqno, an, status, "P", "", _day(hours), _time(hours)]
            + [_day(hours), _time(hours), "", "", "", ""]
        )
        dt = ",".join(
            [reqno, "H1", _day(hours), _time(hours), "", "", product, "1"]
            + [unitamt, usetype]
        )
        self.related.append(row)
        self.related_dt.append(f"{dt},{emrtype},{'1' if emrtype else ''}")
        self.doctors[reqno] = doctor
        if audited:
            self.rows.setdefault("_BDVST", []).append(row)
            self.rows.setdefault("_BDVSTDT", []).append(dt)

    def unit(
        self,
        reqno: str,
        seq: str,
        hours: float,
        status: str = "2",
        an: str = "A1",
        bdtype: str = "LDPRC2",
        stray_return: bool = False,
    ) -> None:
        returned = status == "3" or stray_return
        self.add(
            "BDVSTTRANS.csv",
            reqno,
            "H1",
            an,
            bdtype,
            f"DN{reqno}{seq}",
            seq,
            status,
            _day(hours),
            _time(hours),
            _day(hours + 1) if returned else "",
            _time(hours + 1) if returned else "",
            "",
            "",
        )

    def hb(self, hours: float, value: str, an: str = "A1") -> None:
        self.add("Lab.csv", "H1", an, _day(hours), _time(hours), "290095", value)

    def operation(self, hours: float, icd9: str = "7934", an: str = "A1") -> None:
        day = f"{(T0 + timedelta(hours=hours)):%B %d, %Y}, 12:00 AM"
        self.add("IPTSUMOPRT.csv", an, icd9, "1", day, _time(hours))

    def charge(
        self, hours: float, group: str, oprtact: str, ordercode: str = "X1"
    ) -> None:
        self.add(
            "INCPT_OPRTACT.csv",
            "H1",
            "A1",
            _day(hours),
            _time(hours),
            ordercode,
            ordercode,
            "",
            group,
            oprtact,
        )

    def write(self, root: Path) -> Path:
        bundle = root / "bundle"
        bundle.mkdir(parents=True)
        for name, header in TABLES.items():
            body = self.rows.get(name, [])
            if name == "BDVST.csv":
                body = self.rows.get("_BDVST", [])
            if name == "BDVSTDT.csv":
                body = self.rows.get("_BDVSTDT", [])
            (bundle / name).write_text("\n".join([header, *body]) + "\n")
        (bundle / "BDVST_RELATED.csv").write_text(
            "\n".join([TABLES["BDVST.csv"], *self.related]) + "\n"
        )
        (bundle / "BDVSTDT_RELATED.csv").write_text(
            "\n".join([f"{TABLES['BDVSTDT.csv']},EMRTYPE,BDLEVELEMR", *self.related_dt])
            + "\n"
        )
        doctors = root / "BDVST_doctors.csv"
        doctors.write_text(
            "REQNO,DCTREQ\n" + "".join(f"{q},{d}\n" for q, d in self.doctors.items())
        )
        return doctors


@pytest.fixture
def pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ModuleType, Path]]:
    monkeypatch.delenv("BBA_BDVSTTRANS_CSV", raising=False)
    monkeypatch.setattr(RP, "WORK", tmp_path)
    monkeypatch.setattr(RP, "BUNDLE", tmp_path / "bundle")
    monkeypatch.setattr(RP, "BDVST_CSV", tmp_path / "BDVST_doctors.csv")
    monkeypatch.setattr(RP, "RETURNS_LEDGER_ENABLED", True)
    monkeypatch.setattr(RP, "DECLARED_USETYPE_PILOT_ENABLED", True)
    monkeypatch.setattr(RP, "DECLARED_USE_PREOP_EXEMPT_PILOT_ENABLED", True)
    monkeypatch.setattr(RP, "MSBOS_RESERVATION_PILOT_ENABLED", False)
    yield RP, tmp_path


def _run(pipeline: tuple[ModuleType, Path], bundle: Bundle) -> list[dict[str, str]]:
    module, root = pipeline
    bundle.write(root)
    module.main()
    with (root / "report.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _events(rows: list[dict[str, str]], reqno: str) -> list[dict[str, str]]:
    return [r for r in rows if r["reqno"] == reqno and r["event_seq"]]


def _case_21() -> Bundle:
    """68004408 shape: 4 single units, the first returned, three fresh orders."""
    b = Bundle()
    b.order("R1", doctor="D1")
    b.order("R2", hours=20, status="1", doctor="D2", audited=False)
    b.order("R3", hours=40, status="1", doctor="D3", audited=False)
    b.order(
        "R4", hours=76, status="1", usetype="2", doctor="D4", emrtype="2", audited=False
    )
    b.unit("R1", "1", 4.6, status="3")
    b.unit("R1", "2", 27.0)
    b.unit("R1", "3", 47.4)
    b.unit("R1", "4", 80.0)
    for hours, value in ((2, "11.3"), (25, "8.5"), (45, "7.6"), (78, "8.6")):
        b.hb(hours, value)
    b.charge(78.8, group="111", oprtact="")
    return b


def test_case_21_gives_one_row_per_issue_event_with_the_hb_at_its_issue(
    pipeline: tuple[ModuleType, Path],
) -> None:
    rows = _events(_run(pipeline, _case_21()), "R1")

    assert [r["event_seq"] for r in rows] == ["1", "2", "3", "4"]
    assert [float(r["hb_value_g_dl"]) for r in rows] == [11.3, 8.5, 7.6, 8.6]
    assert [r["linkage_rule"] for r in rows] == [
        "own",
        "fresh_order",
        "fresh_order",
        "fresh_order",
    ]
    assert [r["linked_reqno"] for r in rows] == ["R1", "R2", "R3", "R4"]
    assert [r["attributed_doctor_differs"] for r in rows] == [
        "False",
        "True",
        "True",
        "True",
    ]
    assert len({r["audit_id"] for r in rows}) == 4
    assert {r["parent_audit_id"] for r in rows} != {r["audit_id"] for r in rows}
    assert rows[0]["issue_datetime_local"] == "2025-01-10 12:36:00"
    assert rows[1]["order_to_issue_hours"] == "7.00"
    assert rows[3]["declared_use_applies"] == "True"
    assert rows[3]["classification"] == "PERIOP_TRANSFUSION_EXEMPT"
    assert [r["linked_order_emrtype"] for r in rows] == ["", "", "", "2"]


def test_case_21_event_1_all_returned_is_returned_not_transfused(
    pipeline: tuple[ModuleType, Path],
) -> None:
    rows = _events(_run(pipeline, _case_21()), "R1")

    assert rows[0]["classification"] == "RETURNED_NOT_TRANSFUSED"
    assert rows[0]["issue_units_returned"] == "1"
    assert rows[0]["returns_disposition"] == "not_transfused"
    assert rows[0]["returned_blood_datetime_local"] == "2025-01-10 13:36:00"
    assert rows[1]["returns_disposition"] == "transfused"
    assert rows[1]["returned_blood_datetime_local"] == ""


def test_stray_return_date_on_a_dispensed_unit_is_not_a_return(
    pipeline: tuple[ModuleType, Path],
) -> None:
    b = Bundle()
    b.order("R1")
    b.unit("R1", "1", 2.0, stray_return=True)
    rows = _events(_run(pipeline, b), "R1")

    assert rows[0]["returns_disposition"] == "transfused"
    assert rows[0]["returned_blood_datetime_local"] == ""


def test_later_event_hb_minimum_starts_after_the_last_transfusing_event(
    pipeline: tuple[ModuleType, Path],
) -> None:
    b = Bundle()
    b.order("R1")
    b.unit("R1", "1", 1.0, status="3")
    b.unit("R1", "2", 10.0)
    b.unit("R1", "3", 20.0)
    rows = _events(_run(pipeline, b), "R1")

    assert [r["prior_event_cutoff_local"] for r in rows] == [
        "",
        "",
        "2025-01-10 18:00:00",
    ]


def test_cutoff_pool_spans_orders_from_other_months_on_the_admission(
    pipeline: tuple[ModuleType, Path],
) -> None:
    b = Bundle()
    b.order("P1", hours=-30, audited=False)
    b.unit("P1", "1", -26.0)
    b.order("R1", hours=-10)
    b.unit("R1", "1", -6.0)
    rows = _events(_run(pipeline, b), "R1")

    assert rows[0]["prior_event_cutoff_local"] == "2025-01-09 06:00:00"


def test_declaring_reqno_exempts_event_1_but_judges_its_standing_event_on_hb(
    pipeline: tuple[ModuleType, Path],
) -> None:
    b = Bundle()
    b.order("S1", usetype="2")
    b.unit("S1", "1", 3.0)
    b.unit("S1", "2", 30.0)
    b.hb(2, "9.8")
    b.hb(29, "6.1")
    rows = _events(_run(pipeline, b), "S1")

    assert [r["linkage_rule"] for r in rows] == ["own", "standing_order"]
    assert [r["declared_use_applies"] for r in rows] == ["True", "False"]
    assert rows[0]["classification"] == "PERIOP_TRANSFUSION_EXEMPT"
    assert rows[1]["classification"] != "PERIOP_TRANSFUSION_EXEMPT"
    assert rows[1]["hb_value_g_dl"] == "6.1"


def _fresh_declaring(operation: str) -> Bundle:
    b = Bundle()
    b.order("F1")
    b.order("F2", hours=40, status="1", usetype="2", doctor="D2", audited=False)
    b.unit("F1", "1", 2.0)
    b.unit("F1", "2", 44.0)
    b.hb(1, "7.0")
    b.hb(43, "9.9")
    if operation == "iptsumoprt":
        b.operation(46.0)
    elif operation == "ga":
        b.charge(45.0, group="110", oprtact="AS057")
    elif operation == "ga_ordercode_only":
        b.charge(45.0, group="110", oprtact="", ordercode="AS057")
    return b


@pytest.mark.parametrize(
    ("operation", "exempt"),
    [
        ("iptsumoprt", True),
        ("ga", True),
        ("ga_ordercode_only", False),
        ("none", False),
    ],
)
def test_fresh_order_declaring_surgery_is_exempt_only_with_an_operation(
    pipeline: tuple[ModuleType, Path], operation: str, exempt: bool
) -> None:
    rows = _events(_run(pipeline, _fresh_declaring(operation)), "F1")

    assert rows[1]["linkage_rule"] == "fresh_order"
    assert rows[1]["declared_use_applies"] == str(exempt)
    assert (rows[1]["classification"] == "PERIOP_TRANSFUSION_EXEMPT") is exempt
    assert rows[1]["declared_use_code"] == "2"


def test_excluded_and_no_issue_reqnos_write_one_reqno_level_row(
    pipeline: tuple[ModuleType, Path],
) -> None:
    b = Bundle()
    b.order("E1", an="A9")
    b.add("Diagnosis.csv", "H1", "A9", "D56.1")
    b.unit("E1", "1", 2.0, an="A9")
    b.unit("E1", "2", 30.0, an="A9")
    b.order("N1", an="A8")
    module, root = pipeline
    rows = _run(pipeline, b)

    by_reqno = {r["reqno"]: r for r in rows}
    assert len(rows) == 2
    assert by_reqno["E1"]["classification"] == "excluded"
    assert by_reqno["E1"]["rationale"] == "hemoglobinopathy"
    assert by_reqno["E1"]["events_in_reqno"] == "2"
    assert by_reqno["E1"]["event_seq"] == ""
    assert by_reqno["E1"]["audit_id"] == by_reqno["E1"]["parent_audit_id"]
    assert by_reqno["N1"]["rationale"] == "no_issue_record"
    manifest = (root / "events_manifest.csv").read_text().splitlines()
    assert manifest == ["HN,REQNO,AN,component,audit_id,event_seq"]


def test_events_manifest_lists_every_event(pipeline: tuple[ModuleType, Path]) -> None:
    rows = _events(_run(pipeline, _case_21()), "R1")
    _, root = pipeline
    with (root / "events_manifest.csv").open(encoding="utf-8", newline="") as fh:
        manifest = list(csv.DictReader(fh))

    assert [(m["REQNO"], m["event_seq"], m["audit_id"]) for m in manifest] == [
        (r["reqno"], r["event_seq"], r["audit_id"]) for r in rows
    ]
    assert {m["component"] for m in manifest} == {"red_cell"}


def test_bundle_without_pending_orders_fails_loud(
    pipeline: tuple[ModuleType, Path],
) -> None:
    b = Bundle(pending=False)
    b.order("R1")
    b.unit("R1", "1", 2.0)

    with pytest.raises(SystemExit, match="status-1"):
        _run(pipeline, b)


def _msbos_bundle() -> Bundle:
    """Own order reserves 4 units for an operation MSBOS lists as 'none'."""
    b = Bundle()
    b.order("M1", usetype="2", unitamt="4")
    b.order("M2", hours=40, status="1", usetype="2", audited=False)
    b.unit("M1", "1", 3.0)
    b.unit("M1", "2", 44.0)
    b.operation(5.0, icd9="0205")
    b.operation(46.0, icd9="0205")
    b.hb(2, "12.0")
    b.hb(43, "11.0")
    return b


def test_preop_over_reservation_overlays_event_1_only(
    pipeline: tuple[ModuleType, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RP, "MSBOS_RESERVATION_PILOT_ENABLED", True)
    monkeypatch.setattr(RP, "MSBOS_PLANNED_OP_PICKER_V2_PILOT_ENABLED", False)
    rows = _events(_run(pipeline, _msbos_bundle()), "M1")

    assert [r["declared_use_applies"] for r in rows] == ["True", "True"]
    assert rows[0]["classification"] == "PREOP_OVER_RESERVATION"
    assert rows[1]["classification"] == "PERIOP_TRANSFUSION_EXEMPT"
    assert rows[1]["msbos_reason"] == ""


def test_msbos_review_gate_precedes_the_exemption_on_event_1(
    pipeline: tuple[ModuleType, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RP, "MSBOS_RESERVATION_PILOT_ENABLED", True)
    monkeypatch.setattr(RP, "MSBOS_PLANNED_OP_PICKER_V2_PILOT_ENABLED", True)
    b = Bundle()
    b.order("G1", usetype="2", unitamt="2")
    b.unit("G1", "1", 3.0)
    b.unit("G1", "2", 40.0)
    # The only planned procedure is on the non-blood list, so picker v2
    # excludes every candidate and routes the reservation to review.
    b.operation(6.0, icd9="0331")
    b.hb(2, "11.0")
    b.hb(39, "8.6")
    rows = _events(_run(pipeline, b), "G1")

    assert rows[0]["declared_use_applies"] == "True"
    assert rows[0]["classification"] == "NEEDS_REVIEW"
    assert rows[0]["rationale"] == "all_candidates_excluded"
    assert rows[1]["declared_use_applies"] == "False"
    assert rows[1]["classification"] != "PERIOP_TRANSFUSION_EXEMPT"
    assert rows[1]["msbos_reason"] == ""


def _returned_platelet_reservation(
    monkeypatch: pytest.MonkeyPatch, msbos: bool
) -> Bundle:
    """A platelet REQNO reserving 2 units for arthroplasty, all returned."""
    reference = SimpleNamespace(
        content_hash="platelet-reference-hash",
        groups_for=lambda code: ("Arthroplasty",) if code == "major" else (),
        codes=lambda: frozenset(),
        resolve=lambda code: None,
    )
    operation = OperativeEvent(
        icd9="major",
        or_flag=True,
        operative_datetime=datetime(2025, 1, 10, 5, 0, tzinfo=UTC),
    )
    monkeypatch.setattr(RP, "MSBOS_RESERVATION_PILOT_ENABLED", msbos)
    monkeypatch.setattr(RP, "load_msbos_reference", lambda: reference)
    monkeypatch.setattr(RP, "_build_op_events", lambda *_args: (operation,))
    b = Bundle()
    b.order("PL1", product="LDPPC", unitamt="2")
    b.unit("PL1", "1", 2.0, status="3", bdtype="LDPPC5")
    b.add("Lab.csv", "H1", "A1", _day(1), _time(1), PLATELET_LABEXM, "120")
    return b


def test_msbos_on_serializes_platelet_reservation_columns_on_event_1(
    pipeline: tuple[ModuleType, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _events(
        _run(pipeline, _returned_platelet_reservation(monkeypatch, msbos=True)),
        "PL1",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["component"] == "platelet"
    assert row["classification"] == "RETURNED_NOT_TRANSFUSED"
    assert {
        "msbos_plt_category": row["msbos_plt_category"],
        "msbos_plt_count_k_ul": row["msbos_plt_count_k_ul"],
        "msbos_plt_over_above_per_ul": row["msbos_plt_over_above_per_ul"],
        "msbos_plt_clinician_signed": row["msbos_plt_clinician_signed"],
        "msbos_reserved_units": row["msbos_reserved_units"],
        "msbos_reason": row["msbos_reason"],
        "msbos_is_over": row["msbos_is_over"],
        "msbos_reference_hash": row["msbos_reference_hash"],
        "msbos_resolved_icd9": row["msbos_resolved_icd9"],
    } == {
        "msbos_plt_category": "major_non_neuraxial",
        "msbos_plt_count_k_ul": "120.0",
        "msbos_plt_over_above_per_ul": "80000",
        "msbos_plt_clinician_signed": "True",
        "msbos_reserved_units": "2",
        "msbos_reason": "over_major_non_neuraxial",
        "msbos_is_over": "True",
        "msbos_reference_hash": "platelet-reference-hash",
        "msbos_resolved_icd9": "major",
    }
    assert row["msbos_token"] == ""
    assert row["msbos_recommended_units"] == ""


def test_msbos_off_keeps_the_msbos_columns_out_of_report_csv(
    pipeline: tuple[ModuleType, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _run(pipeline, _returned_platelet_reservation(monkeypatch, msbos=False))

    assert not any(name.startswith("msbos_") for name in rows[0])


def test_platelet_event_is_gated_on_the_count_at_its_issue(
    pipeline: tuple[ModuleType, Path],
) -> None:
    b = Bundle()
    b.order("PT1", product="LDPPC")
    b.unit("PT1", "1", 6.0, bdtype="LDPPC5")
    b.add("Lab.csv", "H1", "A1", _day(-1), _time(-1), PLATELET_LABEXM, "150")
    b.add("Lab.csv", "H1", "A1", _day(5), _time(5), PLATELET_LABEXM, "8")
    rows = _events(_run(pipeline, b), "PT1")

    assert rows[0]["platelet_count_k_ul"] == "8.0"
    assert rows[0]["platelet_anchor_datetime_local"] == "2025-01-10 14:00:00"
    assert rows[0]["returns_disposition"] == "transfused"


def test_event_context_is_read_at_the_issue_not_at_the_order(
    pipeline: tuple[ModuleType, Path],
) -> None:
    b = Bundle()
    b.order("C1")
    b.operation(4.0, icd9="8151")
    b.add("Lab.csv", "H1", "A1", _day(5), _time(5), "290093", "900")
    b.unit("C1", "1", 10.0)
    b.hb(9, "7.5")
    rows = _events(_run(pipeline, b), "C1")

    assert rows[0]["procedure_proximity_hours"] == "6.0"
    assert rows[0]["upcoming_procedure_hours"] == ""
    assert rows[0]["anc_value"] == "900"
