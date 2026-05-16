"""Alerting stub for bba.monitoring (Phase 1 contract).

Two channels in Phase 1, both passive:

1. **Structured log event** via the stdlib ``logging`` module (logger name
   ``bba.monitoring.alarms``). Operators tail / aggregate this logger.
2. **Postgres ``monitoring_alarms`` row** via
   :class:`bba.monitoring.MonitoringStore`. The dashboard (#26) renders
   the table; the report generator (#28) can cite it.

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
    """
    raise NotImplementedError


__all__ = ("emit_alarm",)
