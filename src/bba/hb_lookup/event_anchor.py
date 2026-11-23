"""Hb anchors and the #249 transfusion cutoff for one issue event (#276).

An issue event is judged on the labs at the moment its blood left the bank,
so its lab anchor and its evidence anchor are both the issue time. The
REQNO-level resolvers in :mod:`bba.hb_lookup.anchor` stay for the per-order
pilot path until it switches to events.
"""

from __future__ import annotations

from collections.abc import Iterable

from bba.hb_lookup.anchor import DEFAULT_REANCHOR_THRESHOLD, EvidenceAnchor
from bba.hb_lookup.lookup import _MIN_WINDOW
from bba.issue_events import AuditEvent

# Issue #249: a transfusion inside the 24 h Hb minimum's window may already
# have corrected the low, so the minimum starts at it.
PRIOR_TRANSFUSION_WINDOW = _MIN_WINDOW


def resolve_event_anchors(event: AuditEvent) -> EvidenceAnchor:
    """The one moment ``event``'s Hb, platelets and evidence are read from.

    ``anchor_utc`` is the issue time. ``gap_hours`` is the time from the
    linked order to the issue. When that gap reaches the re-anchor threshold
    of :func:`resolve_evidence_anchor` (blood reserved ahead), ``reason`` is
    ``"issue_reanchor"``, the signal that widens the bundle's Hb window past
    the issue; otherwise it is ``"issue_datetime"``. The lab anchor reason of
    an event is always ``"issue_datetime"``. ``display`` is empty: local-time
    formatting belongs to the report writer.
    """
    gap = event.issue_datetime - event.linked_order_datetime
    return EvidenceAnchor(
        anchor_utc=event.issue_datetime,
        reason=(
            "issue_reanchor" if gap >= DEFAULT_REANCHOR_THRESHOLD else "issue_datetime"
        ),
        gap_hours=gap.total_seconds() / 3600,
        display="",
    )


def prior_transfusion_cutoff(
    events_on_an: Iterable[AuditEvent], event: AuditEvent
) -> AuditEvent | None:
    """The transfusion that bounds ``event``'s 24 h Hb minimum, or None.

    The latest red-cell event in ``events_on_an`` on the same admission as
    ``event`` (under any REQNO; a record on another admission is ignored), issued
    strictly before ``event`` and at most 24 h before it, that transfused at
    least one unit. An event whose units all came back (status 3 or 7) gave
    no blood and is skipped (#264 option A). A platelet event reads no Hb
    and has no cutoff. Pass its ``issue_datetime`` to ``lookup_hb`` as
    ``not_before_utc``. Simultaneous priors break the tie on REQNO, then
    event_seq, so the reported cutoff is deterministic.
    """
    if event.order.component != "red_cell":
        return None
    issued = event.issue_datetime
    priors = [
        e
        for e in events_on_an
        if e.order.component == "red_cell"
        and e.order.an == event.order.an
        and e.issue_datetime < issued
        and issued - e.issue_datetime <= PRIOR_TRANSFUSION_WINDOW
        and not e.link.event.all_returned
    ]
    return max(
        priors,
        key=lambda e: (e.issue_datetime, e.order.reqno, e.event_seq),
        default=None,
    )
