"""bba.monitoring — drift detection, sentinel-κ, golden-set probe (issue #27).

PRD §18 Implementation Decisions defines three monitor cadences:

* **Weekly clinical-reviewer sample** — 50–75 random audit rows per ISO
  week for human inspection. Deterministic by ``(week_iso, sample_size,
  seed)``; no alarm. See :func:`draw_weekly_reviewer_sample`.
* **Continuous SPRT drift detection** — Wald's Sequential Probability
  Ratio Test on the quote-grounding failure rate and the NEEDS_REVIEW
  rate. Target ARL₀ ≥ 500. See :class:`WaldSprtMonitor`.
* **Weekly intra-model κ** — 200-case sentinel set re-run weekly.
  Alarms when Cohen's κ vs the previous week drops below 0.90. See
  :func:`evaluate_sentinel_run`.
* **Quarterly golden-set drift probe** — 100-row fixed cohort re-run
  against the same Anthropic snapshot ID. Alarms on >5% classification
  drift or >10% indication drift. See :func:`evaluate_golden_set_drift`.

Alerting in Phase 1 is structured-log + an in-memory
``monitoring_alarms`` store (stub per issue #27 AC ④). Slack / email /
paging integration AND a Postgres-backed store are Phase 1.5 follow-ups,
deliberately out of scope for #27. See :mod:`bba.monitoring.alerting`
and :mod:`bba.monitoring.store`.

Cadence scheduling is NOT wired here. Every monitor exposes a callable
function or class; issue #29 (``bba.cli``) wires them to a scheduler
(cron / systemd-timer / APScheduler — the cli ticket chooses).

This module is OPERATIONAL, not clinical. It reads classification +
verifier outputs off :mod:`bba.audit_store` rows and never re-derives the
underlying clinical reasoning. Inter-rater agreement coefficients
(κ, AC1) are imported from :mod:`bba.eval_harness.agreement` — NOT
re-implemented here.
"""

from bba.monitoring.alerting import emit_alarm
from bba.monitoring.drift_sprt import (
    WaldSprtMonitor,
    run_sprt_on_window,
    synthetic_drift_stream,
    wald_bounds,
)
from bba.monitoring.exceptions import (
    GoldenSetMismatchError,
    InsufficientHistoryError,
    MonitoringError,
)
from bba.monitoring.golden_set import evaluate_golden_set_drift
from bba.monitoring.models import (
    GOLDEN_SET_CLASSIFICATION_DRIFT_THRESHOLD,
    GOLDEN_SET_INDICATION_DRIFT_THRESHOLD,
    GOLDEN_SET_SIZE,
    SENTINEL_KAPPA_ALARM_THRESHOLD,
    SENTINEL_SET_SEED,
    SENTINEL_SET_SIZE,
    SPRT_DEFAULT_ALPHA,
    SPRT_DEFAULT_BETA,
    SPRT_DEFAULT_MIN_N,
    SPRT_TARGET_ARL0,
    WEEKLY_REVIEWER_SAMPLE_MAX,
    WEEKLY_REVIEWER_SAMPLE_MIN,
    AlarmKind,
    DriftSignal,
    GoldenSetDriftReport,
    GoldenSetEntry,
    GoldenSetRowDelta,
    MonitoringAlarm,
    MonitoringAlarmInput,
    MonitoringConfig,
    SafeId,
    SentinelComparison,
    SentinelManifest,
    SprtConfig,
    SprtState,
    SprtVerdict,
    UTCDatetime,
    WeekIso,
    WeeklyReviewerSample,
    ensure_utc,
)
from bba.monitoring.sampling import draw_weekly_reviewer_sample
from bba.monitoring.sentinel import build_sentinel_manifest, evaluate_sentinel_run
from bba.monitoring.store import MonitoringStore


__all__ = [
    "GOLDEN_SET_CLASSIFICATION_DRIFT_THRESHOLD",
    "GOLDEN_SET_INDICATION_DRIFT_THRESHOLD",
    "GOLDEN_SET_SIZE",
    "SENTINEL_KAPPA_ALARM_THRESHOLD",
    "SENTINEL_SET_SEED",
    "SENTINEL_SET_SIZE",
    "SPRT_DEFAULT_ALPHA",
    "SPRT_DEFAULT_BETA",
    "SPRT_DEFAULT_MIN_N",
    "SPRT_TARGET_ARL0",
    "WEEKLY_REVIEWER_SAMPLE_MAX",
    "WEEKLY_REVIEWER_SAMPLE_MIN",
    "AlarmKind",
    "DriftSignal",
    "GoldenSetDriftReport",
    "GoldenSetEntry",
    "GoldenSetMismatchError",
    "GoldenSetRowDelta",
    "InsufficientHistoryError",
    "MonitoringAlarm",
    "MonitoringAlarmInput",
    "MonitoringConfig",
    "MonitoringError",
    "MonitoringStore",
    "SafeId",
    "SentinelComparison",
    "SentinelManifest",
    "SprtConfig",
    "SprtState",
    "SprtVerdict",
    "UTCDatetime",
    "WaldSprtMonitor",
    "WeekIso",
    "WeeklyReviewerSample",
    "build_sentinel_manifest",
    "draw_weekly_reviewer_sample",
    "emit_alarm",
    "ensure_utc",
    "evaluate_golden_set_drift",
    "evaluate_sentinel_run",
    "run_sprt_on_window",
    "synthetic_drift_stream",
    "wald_bounds",
]
