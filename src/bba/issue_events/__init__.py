"""Issue events: the audit unit of blood_audit #276.

A REQNO's physical units (BDVSTTRANS) issued no more than 4 h apart, with no
same-admission, same-family order placed in between, form one issue event,
anchored on the first unit's issue time. Each event links to the order that
caused it, whose doctor it is attributed to. Pure: no I/O, no model calls.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.issue_events.cluster import cluster_events
from bba.issue_events.ledger import units_from_ledger
from bba.issue_events.link import (
    CandidateOrder,
    EventLink,
    LinkTag,
    candidate_order,
    link_events,
)
from bba.issue_events.models import (
    ISSUE_EVENT_GAP,
    IssuedUnit,
    IssueEvent,
    LedgerUnits,
)

__all__: Sequence[str] = (
    "ISSUE_EVENT_GAP",
    "CandidateOrder",
    "EventLink",
    "IssueEvent",
    "IssuedUnit",
    "LedgerUnits",
    "LinkTag",
    "candidate_order",
    "cluster_events",
    "link_events",
    "units_from_ledger",
)
