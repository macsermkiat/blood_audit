"""Anchor-datetime resolution for :mod:`bba.audit_orders`.

Per issue #4: anchor datetime = ``REQDATE + REQTIME``, with
``BDVSTDATE + BDVSTTIME`` as the fallback. The fallback must be flagged
on the output row (``AuditOrder.anchor_imputed = True``).

A "usable" pair is one where both the date and the time-of-day are non-null
(time was parsed cleanly by :func:`bba.ingest.parse_hosxp_time` upstream).
Per PRD §1, the strict parser never silently shifts; an unrecognized HOSxP
time is a ``parse_warning`` with ``value=None``, which surfaces here as
``time is None`` and disqualifies that pair from being the anchor.

Returns a :class:`RowTimestamp` so the persisted moment is always tz-aware
UTC. The conversion lives in one place (PRD §"Tz-aware throughout").
"""

from __future__ import annotations

from dataclasses import dataclass

from bba.ingest.row_timestamp import RowTimestamp

from bba.audit_orders.models import BloodOrderInput


@dataclass(frozen=True, slots=True)
class AnchorResolution:
    """Outcome of :func:`resolve_anchor` for one :class:`BloodOrderInput`.

    Invariant: when ``anchor`` is non-None, ``imputed`` is False iff the
    primary (REQ) pair supplied it; True iff the fallback (BDVST) pair did.
    When ``anchor`` is None, neither pair was usable and the caller must
    raise :class:`bba.audit_orders.exceptions.UnrecoverableAnchorError`
    before persisting the row.
    """

    anchor: RowTimestamp | None
    imputed: bool


def resolve_anchor(
    record: BloodOrderInput,
    tz_source: str = "Asia/Bangkok",
) -> AnchorResolution:
    """Pick the best anchor datetime for ``record``.

    Resolution order:

    1. If ``req_date`` and ``req_time`` are both non-null → primary anchor,
       ``imputed=False``.
    2. Else if ``bdvst_date`` and ``bdvst_time`` are both non-null →
       fallback anchor, ``imputed=True``.
    3. Else → ``AnchorResolution(anchor=None, imputed=False)``; the caller
       decides whether to raise or quarantine.

    ``tz_source`` is the wall-clock zone of the HOSxP date/time columns.
    """
    raise NotImplementedError


__all__ = ("AnchorResolution", "resolve_anchor")
