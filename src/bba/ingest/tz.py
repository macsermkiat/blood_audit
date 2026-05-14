"""Timezone normalization helpers.

PRD §1, fix E31: all timestamps are stored UTC in Parquet and rendered as
``Asia/Bangkok`` at boundaries (dashboard, reports). A lint rule bans naive
``datetime.now()`` / ``datetime.utcnow()`` calls in this module.
"""

from __future__ import annotations

from datetime import datetime


def to_utc(dt_local: datetime, tz: str = "Asia/Bangkok") -> datetime:
    """Convert a *naive* local datetime in ``tz`` to a tz-aware UTC datetime.

    Aware-datetime inputs MAY be re-anchored to UTC. The return value is always
    tz-aware with ``tzinfo`` equal to UTC.
    """
    raise NotImplementedError
