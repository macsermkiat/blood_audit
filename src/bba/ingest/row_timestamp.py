"""Row-level wall-clock moments as tz-aware UTC datetimes.

Per PRD §1 fix E31: every persisted timestamp is UTC; ``Asia/Bangkok`` is only
the *source* zone and the *display* zone. :class:`RowTimestamp` is the one
place in the codebase that combines a row's date column with its parsed time
into a tz-aware UTC datetime — there is no other path from a (date, time, tz)
triple to a stored timestamp, so the normalization rule cannot be bypassed
by accident.

This module subsumes the formerly-standalone ``to_utc`` helper: it had no real
call site of its own and the row-write path is where the conversion actually
needs to live.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from bba.ingest.models import ParsedTimeOfDay


@dataclass(frozen=True, slots=True)
class RowTimestamp:
    """A row's wall-clock moment as a tz-aware UTC datetime.

    The ``utc`` field always carries ``tzinfo == UTC``. Construct via
    :meth:`from_parts` — never assemble naive datetimes elsewhere.
    """

    utc: datetime

    @classmethod
    def from_parts(
        cls,
        date: _date,
        time: ParsedTimeOfDay,
        tz: str = "Asia/Bangkok",
    ) -> RowTimestamp:
        """Combine a date + time-of-day in zone ``tz`` and normalize to UTC.

        The intermediate local datetime is tz-aware (never naive), and the
        return value is tz-aware UTC, so subsequent arithmetic cannot drift.
        """
        local = datetime(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=time.hour,
            minute=time.minute,
            second=time.second,
            tzinfo=ZoneInfo(tz),
        )
        return cls(utc=local.astimezone(timezone.utc))
