"""Cluster one REQNO's issued units into issue events (plan 1.2)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

from bba.issue_events.models import ISSUE_EVENT_GAP, IssuedUnit, IssueEvent


def cluster_events(
    units: Iterable[IssuedUnit],
    order_times: Iterable[datetime],
    gap: timedelta = ISSUE_EVENT_GAP,
) -> tuple[IssueEvent, ...]:
    """Group units, in issue order, into events.

    A unit joins the current event when it was issued at most ``gap`` after
    the previous unit and no order in ``order_times`` was placed in the
    interval (previous unit issue, this unit issue]. ``order_times`` are the
    placement times of the candidate orders: same admission and family as the
    REQNO, any status but refused or cancelled; the caller selects them.
    """
    orders = tuple(order_times)
    clusters: list[list[IssuedUnit]] = []
    for unit in sorted(units, key=lambda u: u.issue_order):
        if clusters:
            previous = clusters[-1][-1].issue_at
            new_order = any(previous < placed <= unit.issue_at for placed in orders)
            if not new_order and unit.issue_at - previous <= gap:
                clusters[-1].append(unit)
                continue
        clusters.append([unit])
    return tuple(IssueEvent(units=tuple(cluster)) for cluster in clusters)
