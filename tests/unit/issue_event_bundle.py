"""Synthetic pilot bundles shaped like the issue-event gold cases (#276).

``Bundle`` writes the CSVs ``sample_bundle.py`` cuts (BDVST_RELATED included)
plus the DCTREQ export, so a test can run either pilot leg's ``main()`` over
a case-21-shaped REQNO without real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from bba.audit_orders import AuditOrder, FilterResult
from bba.issue_events import (
    AuditEvent,
    AuditEvents,
    CandidateOrder,
    IssuedUnit,
    LedgerUnits,
    build_audit_events,
)

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
    "IPDNRFOCUSDT.csv": "AN,PROGRESSDATE,PROGRESSTIME,ACTION,RESPONSE",
    "IPDADMPROGRESS.csv": "AN,PROGDATE,FIRSTDATE,SUBJECTIVE,OBJECTIVE,ASSESSMENT,PLAN",
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

    def note(self, hours: float, text: str, an: str = "A1") -> None:
        self.add("IPDNRFOCUSDT.csv", an, _day(hours), _time(hours), "", text)

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


def one_event(order: AuditOrder, *, usetype: str | None = None) -> AuditEvent:
    """``order``'s single issue event: one unit, dispensed at the order time."""
    unit = IssuedUnit(
        dnrno="DN1",
        seqno="1",
        bdtype="LDPPC5" if order.component == "platelet" else "LDPRC2",
        status="2",
        issue_at=order.order_datetime,
        returned_at=None,
        given_at=None,
    )
    records = build_audit_events(
        FilterResult(included=(order,), excluded=()),
        {order.reqno: LedgerUnits(issued=(unit,), units_reserved_not_issued=0)},
        [
            CandidateOrder(
                reqno=order.reqno,
                an=order.an,
                component=order.component,
                status="4",
                placed_at=order.order_datetime,
            )
        ],
        usetypes={order.reqno: usetype},
        operations={},
    )
    return records.events[0]


def event_inputs(
    event: AuditEvent, *, msbos_reference: object, reserved_units: object
) -> tuple[object, ...]:
    """``run_llm_leg._build_inputs()``'s result for one event and empty tables."""
    return (
        AuditEvents(events=(event,), excluded=()),
        {event.order.an: [event]},
        [],
        [],
        [],
        [],
        [],
        {},
        {},
        [],
        [],
        {},
        {},
        None,
        msbos_reference,
        reserved_units,
    )
