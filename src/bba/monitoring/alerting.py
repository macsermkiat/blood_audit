"""Alerting stub for bba.monitoring (Phase 1 contract).

Two channels in Phase 1, both passive:

1. **Structured log event** via stdlib ``logging`` (logger name
   ``bba.monitoring.alarms``, WARNING level). The structured payload
   travels as ``extra={"monitoring_alarm_kind": ..., "monitoring_alarm_signal":
   ..., "monitoring_alarm_detail": ...}`` — log-aggregation tools
   (Datadog, Elastic, Loki) consume the structured fields directly,
   plain-text greps still locate the event via the formatted message.
   Stdlib logging is the deliberate Phase 1 choice: it covers the AC's
   "log + structured output" requirement without adding a new runtime
   dependency. A Phase 1.5 follow-up may migrate to structlog if a
   richer processor pipeline is needed.
2. **Alarm row** via :class:`bba.monitoring.MonitoringStore`. The store
   is in-memory in Phase 1; a Postgres-backed implementation mirroring
   :class:`bba.review_actions.ReviewActionsStore`'s append-only contract
   is the documented Phase 1.5 follow-up.

Explicitly OUT OF SCOPE for Phase 1 (tracked as Phase 1.5 follow-ups):

* Slack / Teams webhooks
* PagerDuty / OpsGenie / Atlassian Alerts integration
* Email digest
* Any synchronous notification surface

The test suite enforces this boundary: a regression check verifies the
module does NOT import ``slack_sdk``, ``smtplib``, ``pagerduty``, or any
similar transport. The stub-only contract keeps Phase 1 deployable
without integration credentials.
"""

from __future__ import annotations

import logging

from bba.monitoring.models import MonitoringAlarm, MonitoringAlarmInput


_alarm_log = logging.getLogger("bba.monitoring.alarms")
"""Module-level logger for alarm events.

Operators subscribe to ``bba.monitoring.alarms`` to surface alarms in
their log-aggregation tool of choice (no synchronous transport — see
module docstring)."""


def emit_alarm(
    alarm: MonitoringAlarm | MonitoringAlarmInput,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Emit a structured log event for ``alarm``.

    Does NOT persist the alarm — that is the caller's responsibility (via
    :meth:`MonitoringStore.record_alarm`). The split is deliberate: the
    persistence layer is transactional, the log emission is fire-and-forget,
    and coupling them would force one to wait on the other.

    ``logger`` defaults to ``bba.monitoring.alarms``; tests inject a
    captured logger for assertions on the structured record.

    Emits at ``WARNING`` level so default operator-log configs surface
    alarms without re-tuning. The ``extra`` dict carries the structured
    fields aggregation tools (Datadog, Elastic, Loki) consume directly;
    the formatted message itself includes ``kind`` so plain-text greps
    still locate the event.
    """
    target = logger if logger is not None else _alarm_log
    signal_repr = alarm.signal if alarm.signal is not None else "none"
    detail_dict = dict(alarm.detail)
    target.warning(
        "monitoring.alarm kind=%s signal=%s detail=%s",
        alarm.kind,
        signal_repr,
        detail_dict,
        extra={
            "monitoring_alarm_kind": alarm.kind,
            "monitoring_alarm_signal": signal_repr,
            "monitoring_alarm_detail": detail_dict,
        },
    )


__all__ = ("emit_alarm",)
