"""Postgres-backed persistence for monitoring alarms + sample manifests.

Single table (``monitoring_alarms``) for alarm rows raised by the three
monitors (SPRT, sentinel-κ, golden-set drift). Weekly reviewer sample
manifests are persisted in a sibling table (``monitoring_sample_manifests``)
so historical audit can re-derive who reviewed what week.

Storage choice matches :mod:`bba.review_actions`: Postgres with append-only
semantics (REVOKE UPDATE/DELETE + trigger guard at the DB layer). The
Python layer is a thin wrapper around ``INSERT ... RETURNING`` and
``SELECT`` queries.

Schema migrations live alongside the review_actions migrations under
``<repo_root>/migrations/`` (alembic). A monitoring-specific revision is
added when this module's GREEN-phase implementation lands.
"""

from __future__ import annotations

from datetime import datetime

from bba.monitoring.models import (
    AlarmKind,
    MonitoringAlarm,
    MonitoringAlarmInput,
    MonitoringConfig,
    WeeklyReviewerSample,
)


class MonitoringStore:
    """Postgres-backed store for ``monitoring_alarms`` and weekly sample manifests.

    Construct once per process; the store owns a connection pool (lazy
    open, matching :class:`bba.review_actions.ReviewActionsStore`). Methods
    are thread-safe — multiple cron-driven monitors can write concurrently
    without application-level locking.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        self._config = config

    @property
    def config(self) -> MonitoringConfig:
        return self._config

    # -- Public API -----------------------------------------------------------

    def record_alarm(self, alarm: MonitoringAlarmInput) -> MonitoringAlarm:
        """Persist one alarm row; return the DB-assigned record."""
        raise NotImplementedError

    def list_alarms(
        self,
        *,
        kind: AlarmKind | None = None,
        since: datetime | None = None,
    ) -> tuple[MonitoringAlarm, ...]:
        """List alarms, optionally filtered by ``kind`` and/or ``raised_at >= since``.

        Both filters AND together. Ordered by ``(raised_at, alarm_id)``
        ascending — the chronological alarm timeline.
        """
        raise NotImplementedError

    def persist_sample_manifest(self, sample: WeeklyReviewerSample) -> None:
        """Persist a weekly reviewer sample manifest.

        Idempotent on ``(week_iso, sample_size, seed)``: a re-run of the
        same week's draw with the same parameters is a no-op. This
        preserves the "same week → same manifest" determinism without
        adding duplicate rows for cron retries.
        """
        raise NotImplementedError

    def list_sample_manifests(
        self,
        *,
        week_iso: str | None = None,
    ) -> tuple[WeeklyReviewerSample, ...]:
        """List sample manifests, optionally filtered by ISO week."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the connection pool. Idempotent."""
        raise NotImplementedError


__all__ = ("MonitoringStore",)
