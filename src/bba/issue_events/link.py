"""Link each issue event to the order that caused it (plan 2.1-2.2).

The blood bank books later units on the first open REQNO, so the REQNO names
the right order only for its first event. A later event links to the latest
order placed since the previous event (a fresh decision); with none, it is
more blood on the standing REQNO. The linked order's DCTREQ is the attributed
doctor; callers resolve it from ``linked.reqno``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

from bba.audit_orders import BloodOrderInput, resolve_anchor
from bba.audit_orders.models import Component
from bba.audit_orders.rules import check_rbc_product, is_rbc_product
from bba.issue_events.models import IssueEvent

# Refused (6), cancelled (7) and cancelled doctor orders (11, 12) never caused
# an issue. Every other status, including pending (1), is a candidate.
NON_CANDIDATE_STATUSES = frozenset({"6", "7", "11", "12"})

LinkTag = Literal["own", "fresh_order", "standing_order"]


class CandidateOrder(BaseModel):
    """An order that could have caused an issue on its admission and family.

    Built only by :func:`candidate_order`, which owns the filter; the same
    candidates split events (``cluster_events`` order times) and link them.
    """

    model_config = ConfigDict(frozen=True)

    reqno: str
    an: str
    component: Component
    status: str
    placed_at: AwareDatetime


class EventLink(BaseModel):
    """One event and the order it is judged against and attributed to.

    ``latest_before`` is the #276 rule (latest candidate placed at or before
    the issue), kept beside the default for the January v3 comparison.
    """

    model_config = ConfigDict(frozen=True)

    event: IssueEvent
    tag: LinkTag
    linked: CandidateOrder
    latest_before: CandidateOrder | None

    @property
    def order_after_issue(self) -> bool:
        """The linked order was placed after the issue it is said to cause."""
        return self.linked.placed_at > self.event.issue_datetime


def candidate_order(record: BloodOrderInput) -> CandidateOrder | None:
    """``record`` as a linkage candidate, or None when it cannot be one.

    Not a candidate: a refused or cancelled status, no admission, a product
    family outside red cell and platelet, or no placement time (REQDATE +
    REQTIME, else BDVSTDATE + BDVSTTIME, as the audit order anchor).
    """
    if record.bdvstst in NON_CANDIDATE_STATUSES or not record.an:
        return None
    if check_rbc_product(record) is not None:
        return None
    anchor = resolve_anchor(record).anchor
    if anchor is None:
        return None
    component: Component = (
        "red_cell" if any(is_rbc_product(p) for p in record.products) else "platelet"
    )
    return CandidateOrder(
        reqno=record.reqno,
        an=record.an,
        component=component,
        status=record.bdvstst,
        placed_at=anchor.utc,
    )


def link_events(
    own: CandidateOrder,
    events: Sequence[IssueEvent],
    candidates: Iterable[CandidateOrder],
) -> tuple[EventLink, ...]:
    """Link ``own``'s events, in issue order, per plan 2.2.

    Event 1 links to ``own``. Event k > 1 links to the latest candidate placed
    in (event k-1 issue, event k issue], else to ``own`` as a standing order.
    Candidates on another admission or family are ignored; ties on placement
    time break on REQNO so the link is deterministic.
    """
    anchors = [e.issue_datetime for e in events]
    if anchors != sorted(anchors):
        raise ValueError("events must be in issue order to be linked")
    scoped = [c for c in candidates if (c.an, c.component) == (own.an, own.component)]
    links: list[EventLink] = []
    previous: datetime | None = None
    for event in events:
        issued = event.issue_datetime
        tag: LinkTag = "own"
        linked = own
        if previous is not None:
            fresh = _latest(c for c in scoped if previous < c.placed_at <= issued)
            tag, linked = ("fresh_order", fresh) if fresh else ("standing_order", own)
        links.append(
            EventLink(
                event=event,
                tag=tag,
                linked=linked,
                latest_before=_latest(c for c in scoped if c.placed_at <= issued),
            )
        )
        previous = issued
    return tuple(links)


def _latest(orders: Iterable[CandidateOrder]) -> CandidateOrder | None:
    return max(orders, key=lambda c: (c.placed_at, c.reqno), default=None)
