"""Timezone normalization helpers.

PRD §1, fix E31: all timestamps are persisted UTC in Parquet and rendered as
``Asia/Bangkok`` at boundaries (dashboard, reports). Naive ``datetime`` clock
reads inside this module are banned (lint rule + a structural test scan) —
always pass an explicit ``tzinfo`` when constructing a fresh "now".
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def to_utc(dt_local: datetime, tz: str = "Asia/Bangkok") -> datetime:
    """Convert a naive local datetime in ``tz`` to a tz-aware UTC datetime.

    Aware inputs are re-anchored to UTC. The return value is always tz-aware
    with ``tzinfo`` equal to UTC.
    """
    source = ZoneInfo(tz)
    if dt_local.tzinfo is None:
        dt_local = dt_local.replace(tzinfo=source)
    return dt_local.astimezone(timezone.utc)
