"""Loader for the order → ordering-doctor mapping (``BDVST.DCTREQ``).

The ordering doctor lives on the BDVST order header as ``DCTREQ``
(filled on ~99.8% of orders; the separate ``DCT`` column in the same
export is empty and must not be used). This loader accepts any CSV that
carries at least ``REQNO`` and ``DCTREQ`` — the raw encrypted export
today, or a projection of the ingested store once the ``DCTREQ`` schema
column (see :mod:`bba.ingest.schemas`) has flowed through a re-ingest.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

from bba.attribution.csv_support import require_columns
from bba.attribution.models import (
    UNATTRIBUTED_DOCTOR_ID,
    DoctorRecord,
    EventVerdict,
)


_REQUIRED_COLUMNS: tuple[str, ...] = ("REQNO", "DCTREQ")


def load_reqno_to_doctor(path: Path) -> Mapping[str, str]:
    """Read a BDVST-shaped CSV into a ``REQNO`` → ``DCTREQ`` mapping.

    Orders without a ``DCTREQ`` are omitted — the resolvers translate a
    miss into the explicit unattributed sentinel, so the gap stays
    visible instead of being papered over here. Identical duplicate
    ``REQNO`` rows are tolerated; a ``REQNO`` re-appearing with a
    *different* doctor fails loud (the 2025 export is verified unique,
    so a conflict means a corrupted or concatenated input).
    """
    mapping: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        require_columns(reader.fieldnames, _REQUIRED_COLUMNS, path)
        for row in reader:
            reqno = (row["REQNO"] or "").strip()
            dctreq = (row["DCTREQ"] or "").strip()
            if not reqno or not dctreq:
                continue
            existing = mapping.get(reqno)
            if existing is not None and existing != dctreq:
                raise ValueError(
                    f"{path} maps REQNO {reqno!r} to two different doctors "
                    f"({existing!r} and {dctreq!r}); REQNO must be unique — "
                    "refusing to attribute on a corrupted export"
                )
            mapping[reqno] = dctreq
    return mapping


def attributed_doctor(event: EventVerdict, reqno_to_doctor: Mapping[str, str]) -> str:
    """The doctor an event is credited to: ``DCTREQ`` of its linked order.

    Plan 2.3: a fresh order placed between two issues of one REQNO carries the
    later event to that order's doctor, not to the REQNO's. A linked order
    with no ``DCTREQ`` lands on :data:`UNATTRIBUTED_DOCTOR_ID`.
    """
    return reqno_to_doctor.get(event.linked_reqno, UNATTRIBUTED_DOCTOR_ID)


def attributed_department(
    event: EventVerdict,
    reqno_to_doctor: Mapping[str, str],
    dct_registry: Mapping[str, DoctorRecord],
) -> DoctorRecord | None:
    """The registry record of the event's attributed doctor, when it names a
    department; ``None`` sends the event to the unattributed department."""
    doctor = reqno_to_doctor.get(event.linked_reqno)
    record = dct_registry.get(doctor) if doctor is not None else None
    return record if record is not None and record.deptlct else None
