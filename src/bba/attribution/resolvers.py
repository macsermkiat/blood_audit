"""Attribution resolvers — the concrete fix for the M0/M1 blocker.

:mod:`bba.report_generator.builder` and :mod:`bba.dashboard.models`
declare ``Callable[[AuditRow], str]`` resolver seams for physician and
ward attribution; until now the CLI seams raised because no attribution
table existed. ``BDVST.DCTREQ`` → ``DCT.csv`` supplies it.

The factories here close over plain mappings and read only ``.reqno``
from the row (typed as the :class:`SupportsReqno` protocol), so the
returned callables satisfy both declared resolver shapes without this
package importing the audit-store's runtime dependencies.

Department-as-ward: the existing grouping key in the report / dashboard
layers is "ward"; the department (``Deptlct``) is supplied through the
same seam rather than a forked aggregator, per the feature plan.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from bba.attribution.models import (
    UNATTRIBUTED_DEPARTMENT_ID,
    UNATTRIBUTED_DOCTOR_ID,
    DoctorRecord,
)


class SupportsReqno(Protocol):
    """Anything carrying the order's ``reqno`` — satisfied by
    :class:`bba.audit_store.AuditRow` structurally."""

    reqno: str


def make_physician_resolver(
    reqno_to_doctor: Mapping[str, str],
) -> Callable[[SupportsReqno], str]:
    """Return a resolver mapping a row to its ordering-doctor code.

    Unknown ``reqno`` (no BDVST row, or BDVST row without ``DCTREQ``)
    resolves to :data:`UNATTRIBUTED_DOCTOR_ID` — never an exception, so
    one unattributed order cannot abort a whole report run.
    """

    def resolve(row: SupportsReqno) -> str:
        return reqno_to_doctor.get(row.reqno, UNATTRIBUTED_DOCTOR_ID)

    return resolve


def make_ward_resolver(
    reqno_to_doctor: Mapping[str, str],
    dct_registry: Mapping[str, DoctorRecord],
) -> Callable[[SupportsReqno], str]:
    """Return a resolver mapping a row to its department (``Deptlct``).

    Resolution chain: ``reqno`` → ``DCTREQ`` → registry ``Deptlct``. Any
    broken link (unknown order, unknown doctor, doctor without a
    department) resolves to :data:`UNATTRIBUTED_DEPARTMENT_ID`.
    """

    def resolve(row: SupportsReqno) -> str:
        doctor = reqno_to_doctor.get(row.reqno)
        if doctor is None:
            return UNATTRIBUTED_DEPARTMENT_ID
        record = dct_registry.get(doctor)
        if record is None or not record.deptlct:
            return UNATTRIBUTED_DEPARTMENT_ID
        return record.deptlct

    return resolve
