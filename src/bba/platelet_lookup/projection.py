"""Straight-line 24 h platelet projection (issue #237).

Clinician ruling 2026-09-21: the policy clause "expected to drop below
10,000 /uL within 24 hours" means a straight-line projection only. Take the
last two platelet counts before the order and extend the line 24 h past the
LATEST COUNT; the clause holds iff the projected value is below 10,000 /uL.
(A 30%-per-day reading was considered and dropped.)

Edge cases, fixed here so the guardrail and the evidence line agree:

* Window — the same strict 7-day pre-order lookback as
  :func:`bba.platelet_lookup.lookup_platelet`, so the projection never reads a
  count the deterministic gate cannot see. Post-order counts are ignored.
* Same timestamp — two Lab rows at one datetime are a correction, not two
  draws; the highest ``item_no`` wins (``lookup_platelet``'s tie-break).
* Gap — draws more than :data:`MAX_DRAW_GAP` apart give no projection (the
  #237 analysis used 72 h). Draws less than :data:`MIN_DRAW_GAP` apart give
  none either: a recheck minutes later is assay noise, and its slope would
  project 0 and read as a supported fall. SEED value (6 h) pending a hematology
  ruling; on the 2026-09-21 hematology cohort 6 of 315 orders fall under it.
* Fewer than two usable draws — ``None``; callers treat that as "the trend does
  not support the clause", never as a fall.
* A line that crosses zero is reported as ``0.0``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from bba.platelet_lookup.lookup import _LOOKBACK
from bba.platelet_lookup.models import PlateletObservation

PROJECTION_HORIZON = timedelta(hours=24)
MAX_DRAW_GAP = timedelta(hours=72)
MIN_DRAW_GAP = timedelta(hours=6)


def project_24h(
    *,
    observations: Sequence[PlateletObservation],
    anchor_utc: datetime,
) -> float | None:
    """Projected count (×10³/µL) 24 h past the latest pre-order draw, or ``None``.

    ``observations`` must be for a single patient; order is unimportant.
    ``anchor_utc`` must be tz-aware UTC.
    """
    corrected_by_datetime: dict[datetime, PlateletObservation] = {}
    for o in observations:
        if not (
            o.datetime_utc <= anchor_utc and anchor_utc - o.datetime_utc < _LOOKBACK
        ):
            continue
        kept = corrected_by_datetime.get(o.datetime_utc)
        if kept is None or o.item_no > kept.item_no:
            corrected_by_datetime[o.datetime_utc] = o
    draws = sorted(corrected_by_datetime.values(), key=lambda o: o.datetime_utc)
    if len(draws) < 2:
        return None
    previous, latest = draws[-2], draws[-1]
    gap = latest.datetime_utc - previous.datetime_utc
    if gap > MAX_DRAW_GAP or gap < MIN_DRAW_GAP:
        return None
    slope_per_hour = (latest.value_k_ul - previous.value_k_ul) / (
        gap.total_seconds() / 3600.0
    )
    projected = latest.value_k_ul + slope_per_hour * (
        PROJECTION_HORIZON.total_seconds() / 3600.0
    )
    return max(projected, 0.0)


__all__ = ("MAX_DRAW_GAP", "MIN_DRAW_GAP", "PROJECTION_HORIZON", "project_24h")
