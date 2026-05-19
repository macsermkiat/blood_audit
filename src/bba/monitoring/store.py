"""In-memory persistence for monitoring alarms + sample manifests (Phase 1).

Phase 1 contract (issue #27 AC ④ — "Alerting integration stub"): storage
is an in-process dict. This is a deliberate scope choice, NOT a
regression vs the ticket — the AC asks for a stub, and the in-memory
implementation satisfies it without adding an alembic migration or a
testcontainers dependency to the unit-test surface.

Phase 1.5 will substitute a Postgres-backed implementation that matches
:class:`bba.review_actions.ReviewActionsStore`'s append-only contract
(REVOKE UPDATE/DELETE + trigger guard). The interface stays unchanged
across the substitution — callers depend on ``MonitoringStore`` as a
single class, and the swap is internal. Tests parameterized on
:class:`MonitoringConfig` will then spin up a testcontainer; today's
tests use a placeholder DSN that the in-memory implementation ignores.

Thread-safety: a single lock guards both stores. The cron-driven
monitors (#29 will wire them) will run sequentially per cadence; the
lock exists so multi-threaded test fixtures and future concurrent-cadence
deployments are safe. The TestConcurrentAlarmWrites regression test
verifies that 8 worker threads writing 50 alarms each produce 400 rows
with unique ``alarm_id`` values and no lost writes.
"""

from __future__ import annotations

from datetime import datetime
from threading import Lock

from bba.monitoring.models import (
    AlarmKind,
    MonitoringAlarm,
    MonitoringAlarmInput,
    MonitoringConfig,
    WeeklyReviewerSample,
    ensure_utc,
)


class MonitoringStore:
    """In-memory store for alarms + weekly sample manifests (Phase 1).

    Construct once per process. ``MonitoringStore(config)`` accepts the
    same :class:`MonitoringConfig` the Phase 1.5 Postgres-backed
    implementation will take, so callers do not change when the swap
    happens.
    """

    def __init__(self, config: MonitoringConfig) -> None:
        self._config = config
        self._lock = Lock()
        self._alarms: list[MonitoringAlarm] = []
        self._manifests: dict[tuple[str, int, int], WeeklyReviewerSample] = {}
        self._next_alarm_id: int = 1
        self._closed = False

    @property
    def config(self) -> MonitoringConfig:
        return self._config

    # -- Public API -----------------------------------------------------------

    def record_alarm(self, alarm: MonitoringAlarmInput) -> MonitoringAlarm:
        """Persist one alarm row; return the assigned record.

        The persisted record's ``alarm_id`` is a monotonically-increasing
        in-process counter. Phase 1.5 will substitute a Postgres bigserial.
        """
        self._ensure_open()
        with self._lock:
            alarm_id = self._next_alarm_id
            self._next_alarm_id += 1
            persisted = MonitoringAlarm(
                alarm_id=alarm_id,
                kind=alarm.kind,
                signal=alarm.signal,
                raised_at=alarm.raised_at,
                detail=alarm.detail,
            )
            self._alarms.append(persisted)
            return persisted

    def list_alarms(
        self,
        *,
        kind: AlarmKind | None = None,
        since: datetime | None = None,
    ) -> tuple[MonitoringAlarm, ...]:
        """List alarms, optionally filtered by ``kind`` and/or ``raised_at >= since``.

        Both filters AND together. Ordered by ``(raised_at, alarm_id)``
        ascending — the chronological alarm timeline.

        ``since`` MUST be tz-aware. A naive ``since`` would silently
        TypeError at the ``>=`` comparison against the alarm's tz-aware
        ``raised_at``; instead we normalize through :func:`ensure_utc`
        which raises :class:`ValueError` at the boundary with a clear
        message.
        """
        self._ensure_open()
        normalized_since = ensure_utc(since) if since is not None else None
        with self._lock:
            result = list(self._alarms)
        if kind is not None:
            result = [a for a in result if a.kind == kind]
        if normalized_since is not None:
            result = [a for a in result if a.raised_at >= normalized_since]
        result.sort(key=lambda a: (a.raised_at, a.alarm_id))
        return tuple(result)

    def persist_sample_manifest(self, sample: WeeklyReviewerSample) -> None:
        """Persist a weekly reviewer sample manifest.

        Idempotent on ``(week_iso, sample_size, seed)``: a re-run of the
        same week's draw with the same parameters is a no-op. This
        preserves the "same week → same manifest" determinism without
        adding duplicate rows for cron retries.
        """
        self._ensure_open()
        key = (sample.week_iso, sample.sample_size, sample.seed)
        with self._lock:
            self._manifests.setdefault(key, sample)

    def list_sample_manifests(
        self,
        *,
        week_iso: str | None = None,
    ) -> tuple[WeeklyReviewerSample, ...]:
        """List sample manifests, optionally filtered by ISO week."""
        self._ensure_open()
        with self._lock:
            items = list(self._manifests.values())
        if week_iso is not None:
            items = [m for m in items if m.week_iso == week_iso]
        items.sort(key=lambda m: (m.week_iso, m.sample_size, m.seed))
        return tuple(items)

    def close(self) -> None:
        """Release in-memory storage. Idempotent."""
        with self._lock:
            self._closed = True
            self._alarms.clear()
            self._manifests.clear()

    def __enter__(self) -> MonitoringStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    # -- Internal helpers ----------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MonitoringStore is closed")


__all__ = ("MonitoringStore",)
