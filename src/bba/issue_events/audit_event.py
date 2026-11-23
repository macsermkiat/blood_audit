"""Per-event audit records (blood_audit #276, plan 4.1-4.2 and 8.15).

Each issue event becomes one audit record with its own audit_id, judged
against the order it is linked to. A REQNO that ``audit_orders`` excludes, or
that issued no unit, keeps one REQNO-level record under the legacy
``build_audit_id(hn, reqno)`` and yields no event records.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from bba.audit_orders import (
    AuditOrder,
    Component,
    ExclusionReason,
    FilterResult,
    build_audit_id,
    build_event_audit_id,
)
from bba.declared_use import DECLARED_SURGICAL_LABELS, label_for
from bba.issue_events.cluster import cluster_events
from bba.issue_events.link import CandidateOrder, EventLink, LinkTag, link_events
from bba.issue_events.models import LedgerUnits

# Owner decision 8.15 (c): a fresh order's declared surgical use exempts its
# event only when an operation follows that order within this window of the
# issue, both ends inclusive.
PERIOP_BEFORE_ISSUE = timedelta(hours=6)
PERIOP_AFTER_ISSUE = timedelta(hours=72)

ReqnoExclusionReason = Literal[ExclusionReason, "no_issue_record"]


class AuditEvent(BaseModel):
    """One issue event as an audit record.

    ``order`` is the REQNO's audit_orders row (its identity, order time and
    product family); ``link`` is the event and the order it is judged against.
    ``linked_usetype`` is the linked order's collapsed BDVSTDT USETYPE.
    ``declared_use_applies`` says whether that declaration counts for this
    event (owner decision 8.15 (c)); whether it exempts the event stays the
    classifier's call, under its flags and gates, for either product family.
    Built by :func:`build_audit_events`.
    """

    model_config = ConfigDict(frozen=True)

    order: AuditOrder
    link: EventLink
    event_seq: int = Field(ge=1)
    events_in_reqno: int
    units_reserved_not_issued: int = Field(ge=0)
    linked_usetype: str | None
    declared_use_applies: bool

    @property
    def audit_id(self) -> str:
        first = self.link.event.units[0]
        return build_event_audit_id(
            self.order.hn, self.order.reqno, (first.dnrno, first.seqno, first.bdtype)
        )

    @property
    def parent_audit_id(self) -> str:
        return self.order.audit_id

    @property
    def issue_datetime(self) -> AwareDatetime:
        return self.link.event.issue_datetime

    @property
    def issue_units(self) -> int:
        return len(self.link.event.units)

    @property
    def issue_units_returned(self) -> int:
        return self.link.event.units_returned

    @property
    def issue_unit_statuses(self) -> tuple[str, ...]:
        return tuple(unit.status for unit in self.link.event.units)

    @property
    def linkage_rule(self) -> LinkTag:
        return self.link.tag

    @property
    def linked_reqno(self) -> str:
        return self.link.linked.reqno

    @property
    def linked_order_datetime(self) -> AwareDatetime:
        return self.link.linked.placed_at

    @property
    def linked_order_status(self) -> str:
        return self.link.linked.status

    @property
    def linked_reqno_latest_before(self) -> str | None:
        before = self.link.latest_before
        return before.reqno if before else None

    @property
    def order_to_issue_hours(self) -> float:
        return (self.issue_datetime - self.linked_order_datetime).total_seconds() / 3600

    @property
    def unit_family_mismatch(self) -> bool:
        """A unit's product code names another family than the order's."""
        return any(
            unit_family(unit.bdtype) != self.order.component
            for unit in self.link.event.units
        )


class ExcludedReqno(BaseModel):
    """The one REQNO-level record of a REQNO that has no event records.

    ``events_in_reqno`` is the measured event count, 0 when the REQNO is not
    a linkage candidate (refused or cancelled, no admission, another product
    family, or no order time).
    """

    model_config = ConfigDict(frozen=True)

    hn: str
    reqno: str
    reason: ReqnoExclusionReason
    detail: str | None
    events_in_reqno: int = Field(ge=0)
    units_reserved_not_issued: int = Field(ge=0)

    @property
    def audit_id(self) -> str:
        return build_audit_id(self.hn, self.reqno)

    @property
    def parent_audit_id(self) -> str:
        return self.audit_id

    @property
    def event_seq(self) -> None:
        return None


class AuditEvents(BaseModel):
    """Event records and REQNO-level excluded records; audit_ids are unique."""

    model_config = ConfigDict(frozen=True)

    events: tuple[AuditEvent, ...]
    excluded: tuple[ExcludedReqno, ...]

    @model_validator(mode="after")
    def _unique_audit_ids(self) -> AuditEvents:
        ids = Counter(
            [*(e.audit_id for e in self.events), *(r.audit_id for r in self.excluded)]
        )
        if any(count > 1 for count in ids.values()):
            raise ValueError(
                "two audit records share an audit_id; events of one REQNO must "
                "start with distinct units"
            )
        return self


def unit_family(bdtype: str) -> Component | None:
    """The product family a BDVSTTRANS unit code names, by marker (plan 1.1).

    Unit codes are variants (LDPRC2, LDPPC5I, SDPFI4) outside the order
    allow-lists, so only the family marker is read.
    """
    code = bdtype.upper()
    if "PRC" in code or code.startswith("SDR"):
        return "red_cell"
    if "PC" in code or code.startswith("SDP"):
        return "platelet"
    return None


def build_audit_events(
    orders: FilterResult,
    ledgers: Mapping[str, LedgerUnits],
    candidates: Iterable[CandidateOrder],
    *,
    usetypes: Mapping[str, str | None],
    operations: Mapping[str, Sequence[datetime]],
) -> AuditEvents:
    """Turn audit_orders output into per-event audit records.

    ``ledgers`` holds each REQNO's issued units on its own admission (empty
    when it has no ledger rows); ``candidates`` holds every order that could
    have caused an issue, the audited ones included. ``usetypes`` maps a REQNO
    to its collapsed USETYPE (absent means no declaration). ``operations``
    maps an admission to its operation times: the caller decides what counts
    as an operation (owner decision 8.15 (c)).
    """
    if any(t.tzinfo is None for times in operations.values() for t in times):
        raise ValueError("operation times must be timezone-aware")
    by_reqno: dict[str, CandidateOrder] = {}
    by_scope: dict[tuple[str, Component], list[CandidateOrder]] = defaultdict(list)
    for candidate in candidates:
        if by_reqno.setdefault(candidate.reqno, candidate) != candidate:
            raise ValueError(f"REQNO {candidate.reqno} has two different candidates")
        by_scope[(candidate.an, candidate.component)].append(candidate)

    def measured(reqno: str) -> tuple[tuple[EventLink, ...], int]:
        """The REQNO's linked events and reserved-not-issued units."""
        own = by_reqno.get(reqno)
        if own is None:
            return (), 0
        ledger = ledgers.get(reqno)
        if ledger is None:
            raise ValueError(f"REQNO {reqno} has no ledger units (pass an empty one)")
        scope = by_scope[(own.an, own.component)]
        events = cluster_events(ledger.issued, [c.placed_at for c in scope])
        return link_events(own, events, scope), ledger.units_reserved_not_issued

    excluded: list[ExcludedReqno] = []
    for record in orders.excluded:
        links, reserved = measured(record.reqno)
        excluded.append(
            ExcludedReqno(
                hn=record.hn,
                reqno=record.reqno,
                reason=record.reason,
                detail=record.detail,
                events_in_reqno=len(links),
                units_reserved_not_issued=reserved,
            )
        )
    events: list[AuditEvent] = []
    for order in orders.included:
        _require_own_candidate(order, by_reqno.get(order.reqno))
        links, reserved = measured(order.reqno)
        if not links:
            excluded.append(
                ExcludedReqno(
                    hn=order.hn,
                    reqno=order.reqno,
                    reason="no_issue_record",
                    detail=None,
                    events_in_reqno=0,
                    units_reserved_not_issued=reserved,
                )
            )
            continue
        operation_times = operations.get(order.an, ())
        for seq, link in enumerate(links, start=1):
            usetype = usetypes.get(link.linked.reqno)
            events.append(
                AuditEvent(
                    order=order,
                    link=link,
                    event_seq=seq,
                    events_in_reqno=len(links),
                    units_reserved_not_issued=reserved,
                    linked_usetype=usetype,
                    declared_use_applies=_declared_use_applies(
                        link, usetype, operation_times
                    ),
                )
            )
    return AuditEvents(events=tuple(events), excluded=tuple(excluded))


def _require_own_candidate(order: AuditOrder, own: CandidateOrder | None) -> None:
    """The audited order must be linked as the same order it was audited as."""
    if own is None:
        raise ValueError(f"audited REQNO {order.reqno} is not among the candidates")
    if (own.an, own.component, own.placed_at) != (
        order.an,
        order.component,
        order.order_datetime,
    ):
        raise ValueError(f"candidate {order.reqno} disagrees with its audit order")


def _declared_use_applies(
    link: EventLink, usetype: str | None, operation_times: Iterable[datetime]
) -> bool:
    """Owner decision 8.15 (c), amended 2026-11-22.

    The first event keeps today's rule (its own order's declaration); a
    standing-order event never inherits one; a fresh order's declaration
    applies only when an operation follows that order within the periop
    window of the issue.
    """
    if usetype is None or label_for(usetype) not in DECLARED_SURGICAL_LABELS:
        return False
    if link.tag == "own":
        return True
    if link.tag == "standing_order":
        return False
    issued = link.event.issue_datetime
    return any(
        link.linked.placed_at < moment
        and issued - PERIOP_BEFORE_ISSUE <= moment <= issued + PERIOP_AFTER_ISSUE
        for moment in operation_times
    )
