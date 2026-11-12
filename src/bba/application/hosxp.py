"""HOSxP boundary parsing and admission-scoped joins for the application."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path

from bba.audit_orders import (
    AuditOrder,
    AuditOrdersConfig,
    BloodOrderInput,
    build_audit_orders,
)
from bba.cohort_detector import MedEvent, OperativeEvent
from bba.evidence_bundle_builder import FocusNote, ProgressNote
from bba.hb_lookup import HbObservation, parse_hb_value
from bba.ingest import RowTimestamp, all_tables, parse_hosxp_time
from bba.ingest.date_parser import parse_kcmh_english_date
from bba.ingest.models import CSVTable, ParsedTimeOfDay
from bba.ingest.normalize import normalize_header, normalize_rows
from bba.platelet_lookup import (
    PLATELET_LABEXM,
    PlateletObservation,
    parse_platelet_count,
)

Row = dict[str, str]


@dataclass(frozen=True)
class Tables:
    rows: dict[CSVTable, tuple[Row, ...]]
    by_admission: dict[tuple[CSVTable, str, str], tuple[Row, ...]]
    by_order: dict[tuple[CSVTable, str, str], tuple[Row, ...]]
    icd9: dict[str, Row]

    def __getitem__(self, table: CSVTable) -> tuple[Row, ...]:
        return self.rows[table]

    def encounter(self, table: CSVTable, hn: str, an: str) -> tuple[Row, ...]:
        return self.by_admission.get((table, hn, an), ()) + self.by_admission.get(
            (table, "", an), ()
        )

    def order_rows(self, table: CSVTable, hn: str, reqno: str) -> tuple[Row, ...]:
        return self.by_order.get((table, hn, reqno), ())


def read_tables(input_dir: Path) -> Tables:
    tables: dict[CSVTable, tuple[Row, ...]] = {}
    for table in all_tables():
        with (input_dir / f"{table}.csv").open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            kept = normalize_header(table, header).header
            tables[table] = tuple(
                dict(zip(kept, row.cells, strict=True))
                for row in normalize_rows(table, header, kept, reader)
            )
    admissions: dict[tuple[CSVTable, str, str], list[Row]] = defaultdict(list)
    ordered: dict[tuple[CSVTable, str, str], list[Row]] = defaultdict(list)
    for table, rows in tables.items():
        for row in rows:
            if row.get("AN"):
                admissions[(table, row.get("HN", ""), row["AN"])].append(row)
            if row.get("REQNO"):
                ordered[(table, row.get("HN", ""), row["REQNO"])].append(row)
    return Tables(
        tables,
        {k: tuple(v) for k, v in admissions.items()},
        {k: tuple(v) for k, v in ordered.items()},
        {r["ICD9CM"].replace(".", ""): r for r in tables["ICD9CM"]},
    )


def parse_date(raw: str) -> date | None:
    if not raw.strip():
        return None
    try:
        return date.fromisoformat(raw.strip()[:10])
    except ValueError:
        return parse_kcmh_english_date(raw).value


def moment(raw_date: str, raw_time: str = "") -> datetime | None:
    if not raw_time and len(raw_date) > 10 and raw_date[10] in {" ", "T"}:
        try:
            combined = datetime.fromisoformat(raw_date.strip())
        except ValueError:
            return None
        if combined.tzinfo is not None:
            return combined.astimezone(UTC)
        return RowTimestamp.from_parts(
            combined.date(),
            ParsedTimeOfDay(combined.hour, combined.minute, combined.second),
        ).utc.replace(microsecond=combined.microsecond)
    day = parse_date(raw_date)
    time = parse_hosxp_time(raw_time or raw_date).value
    if day is None or time is None:
        return None
    return RowTimestamp.from_parts(day, time).utc


def scoped(tables: Tables, table: CSVTable, order: AuditOrder) -> tuple[Row, ...]:
    return tables.encounter(table, order.hn, order.an)


def details(tables: Tables, hn: str, reqno: str) -> tuple[Row, ...]:
    return tables.order_rows("BDVSTDT", hn, reqno)


def orders(tables: Tables, version: str) -> tuple[AuditOrder, ...]:
    inputs = tuple(
        BloodOrderInput(
            hn=r["HN"],
            an=r.get("AN") or None,
            reqno=r["REQNO"],
            bdvstst=r["BDVSTST"],
            reqtype=r["REQTYPE"],
            canceldate=r["CANCELDATE"] or None,
            req_date=parse_date(r["REQDATE"]),
            req_time=parse_hosxp_time(r["REQTIME"]).value,
            bdvst_date=parse_date(r["BDVSTDATE"]),
            bdvst_time=parse_hosxp_time(r["BDVSTTIME"]).value,
            products=tuple(d["BDTYPE"] for d in details(tables, r["HN"], r["REQNO"])),
            diagnosis_codes=tuple(
                dict.fromkeys(
                    code.strip()
                    for raw in (
                        r["ICD10"],
                        *(
                            d["ICD10"] or d["ICD10WHO"]
                            for d in tables.encounter("Diagnosis", r["HN"], r["AN"])
                        ),
                    )
                    for code in raw.replace(";", ",").split(",")
                    if code.strip()
                )
            ),
        )
        for r in tables["BDVST"]
    )
    return build_audit_orders(inputs, AuditOrdersConfig(code_version=version)).included


def hb_observations(rows: tuple[Row, ...]) -> tuple[HbObservation, ...]:
    result = []
    for index, row in enumerate(rows):
        code = row["LABEXM"]
        value = parse_hb_value(row["RESULT"]) if code in {"290095", "500001"} else None
        at = moment(row["LVSTDATE"], row["LVSTTIME"])
        if value is not None and at is not None:
            result.append(
                HbObservation(
                    value_g_dl=value,
                    datetime_utc=at,
                    source="HEMATOLOGY" if code == "290095" else "POCT",
                    item_no=index,
                )
            )
    return tuple(result)


def platelet_observations(rows: tuple[Row, ...]) -> tuple[PlateletObservation, ...]:
    result = []
    for index, row in enumerate(rows):
        value = (
            parse_platelet_count(row["RESULT"])
            if row["LABEXM"] == PLATELET_LABEXM
            else None
        )
        at = moment(row["LVSTDATE"], row["LVSTTIME"])
        if value is not None and at is not None:
            result.append(
                PlateletObservation(
                    value_k_ul=value,
                    datetime_utc=at,
                    source="HEMATOLOGY",
                    item_no=index,
                )
            )
    return tuple(result)


def anc(rows: tuple[Row, ...], anchor: datetime) -> int | None:
    values = []
    for row in rows:
        if row["LABEXM"] != "290093":
            continue
        at = moment(row["LVSTDATE"], row["LVSTTIME"])
        try:
            value = float(row["RESULT"])
        except ValueError:
            continue
        if at is not None and at <= anchor and value >= 0:
            values.append((at, int(value)))
    return max(values)[1] if values else None


def medications(rows: tuple[Row, ...]) -> tuple[MedEvent, ...]:
    return tuple(
        MedEvent(
            timestamp=at,
            drug=" ".join(
                row[k]
                for k in (
                    "NAME_MEDITEM",
                    "NAME_GENERIC",
                    "STRENGTH",
                    "STRENGTHUNIT",
                    "MEDUSEQTY",
                )
                if row[k]
            ),
        )
        for row in rows
        if (at := moment(row["PRSCDATE"], row["PRSCTIME"])) is not None
    )


def notes(
    tables: Tables, order: AuditOrder
) -> tuple[tuple[ProgressNote, ...], tuple[FocusNote, ...]]:
    progress = tuple(
        ProgressNote(
            timestamp=at,
            text="\n".join(
                f"{section}: {r[section]}"
                for section in ("ASSESSMENT", "PLAN", "OBJECTIVE", "SUBJECTIVE")
                if r[section]
            ),
        )
        for r in scoped(tables, "IPDADMPROGRESS", order)
        if (at := moment(r["PROGDATE"])) is not None
    )
    focus = tuple(
        FocusNote(
            timestamp=at, text="\n".join(filter(None, (r["ACTION"], r["RESPONSE"])))
        )
        for r in scoped(tables, "IPDNRFOCUSDT", order)
        if (at := moment(r["PROGRESSDATE"], r["PROGRESSTIME"])) is not None
    )
    return progress, focus


def operations(tables: Tables, order: AuditOrder) -> tuple[OperativeEvent, ...]:
    dictionary = tables.icd9
    result = []
    for row in scoped(tables, "IPTSUMOPRT", order):
        at = moment(row["INDATE"], row["INTIME"])
        code = row["ICD9CM"].replace(".", "")
        if at is not None and code:
            meta = dictionary.get(code, {})
            result.append(
                OperativeEvent(
                    icd9=code,
                    or_flag=meta.get("ORFLAG") == "1",
                    operative_datetime=at,
                    name=meta.get("NAME") or None,
                )
            )
    for row in scoped(tables, "INCPT_OPRTACT", order):
        if row["CANCELDATE"] or row["INCGRP"] not in {"110", "111"}:
            continue
        at = moment(row["INCDATE"], row["INCTIME"])
        if at is None:
            continue
        codes = tuple(
            row[k].replace(".", "")
            for k in ("O__ICD9CM", "O__ICD9CMADD1", "O__ICD9CMADD2")
            if row[k]
        )
        for code in codes or (
            f"INCPT:{row['O__OPRTACT'] or row['INCOME'] or row['ORDERCODE']}",
        ):
            meta = dictionary.get(code, {})
            result.append(
                OperativeEvent(
                    icd9=code,
                    or_flag=meta.get("ORFLAG") == "1",
                    operative_datetime=at,
                    name=row["O__NAME_EN"]
                    or row["O__NAME"]
                    or meta.get("NAME")
                    or None,
                )
            )
    return tuple(result)
