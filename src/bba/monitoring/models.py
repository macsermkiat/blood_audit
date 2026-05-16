"""Frozen Pydantic v2 models for the bba.monitoring contract.

This module is operational, not clinical: every persisted record references
artifacts produced by the audit pipeline (``audit_id``, ``run_id``,
classification outputs) and never re-derives a clinical fact. The shape of
that boundary is captured here so downstream consumers (#26 dashboard,
#29 cli) can depend on a stable interface even as the implementation in
sibling modules evolves.

Three monitor cadences (PRD §18 Implementation Decisions / issue #27):

* **Weekly clinical-reviewer sample** — 50–75 random audit rows for human
  review. Deterministic by ``(week_iso, sample_size, seed)``. No alarm.
* **Continuous SPRT drift detection** — Wald's Sequential Probability Ratio
  Test on the quote-grounding failure rate and the NEEDS_REVIEW rate.
  Target ARL₀ ≥ 500 (≈1 false alarm/year at expected throughput). Alarms
  on H1 rejection.
* **Weekly intra-model κ** — 200-case sentinel set re-run weekly through
  the audit pipeline. Alarms when Cohen's κ vs the previous week drops
  below 0.90. Computed via :mod:`bba.eval_harness.agreement` (NOT
  re-implemented).
* **Quarterly golden-set drift probe** — 100-row fixed cohort re-run
  against the same Anthropic snapshot ID. Alarms when >5% of rows change
  classification or >10% change cited indications.

Alerting in Phase 1 is structured-log + Postgres ``monitoring_alarms``
table only. Slack / email / paging integration is a Phase 1.5 follow-up.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict

from bba.audit_store import Classification


# =============================================================================
# Constants (PRD §18 / issue #27 defaults)
# =============================================================================


WEEKLY_REVIEWER_SAMPLE_MIN: int = 50
"""Minimum weekly reviewer sample size (PRD §18)."""


WEEKLY_REVIEWER_SAMPLE_MAX: int = 75
"""Maximum weekly reviewer sample size (PRD §18)."""


SENTINEL_SET_SIZE: int = 200
"""Fixed 200-case sentinel set size (PRD §18)."""


SENTINEL_SET_SEED: int = 42
"""Deterministic sentinel-construction seed (issue #27)."""


SENTINEL_KAPPA_ALARM_THRESHOLD: float = 0.90
"""Cohen's κ below this triggers a sentinel alarm (PRD §18)."""


GOLDEN_SET_SIZE: int = 100
"""Quarterly golden-set size (PRD §18)."""


GOLDEN_SET_CLASSIFICATION_DRIFT_THRESHOLD: float = 0.05
"""Alarm if >5% of golden-set rows changed classification vs baseline."""


GOLDEN_SET_INDICATION_DRIFT_THRESHOLD: float = 0.10
"""Alarm if >10% of golden-set rows changed cited indications vs baseline."""


SPRT_DEFAULT_ALPHA: float = 0.05
"""Type-I error rate for Wald's SPRT (false alarm probability)."""


SPRT_DEFAULT_BETA: float = 0.05
"""Type-II error rate for Wald's SPRT (missed-drift probability)."""


SPRT_TARGET_ARL0: int = 500
"""Target average run length under the null (≈1 false alarm/year)."""


# =============================================================================
# Literal types
# =============================================================================


AlarmKind = Literal[
    "drift_sprt",
    "sentinel_kappa",
    "golden_set_classification",
    "golden_set_indication",
]
"""Tag for the monitor that raised an alarm.

Weekly reviewer-sample draws are NOT an alarm class — they are operator
work items, not deviation signals. Persistence still uses ``MonitoringStore``
for historical audit (``persist_sample_manifest``), but no row lands in
``monitoring_alarms``.
"""


DriftSignal = Literal[
    "quote_grounding_failure_rate",
    "needs_review_rate",
]
"""The two binomial rates the SPRT watches (PRD §18)."""


SprtVerdict = Literal["continue", "reject_null", "accept_null"]
"""Wald's SPRT terminal verdict.

* ``continue`` — the cumulative log-likelihood is inside ``(B, A)``;
  observation continues.
* ``reject_null`` — crossed the upper bound ``A``; drift detected. An
  alarm SHOULD fire and the monitor SHOULD reset before next window.
* ``accept_null`` — crossed the lower bound ``B``; no drift. Monitor
  resets and continues.
"""


# =============================================================================
# Validators (shared with sibling modules' SafeId / UTCDatetime conventions)
# =============================================================================


_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_safe_id(value: str) -> str:
    """Reject identifiers that would be unsafe in log lines or file paths.

    Mirrors :func:`bba.audit_store.models._validate_safe_id` — same allow-list
    so the monitoring module's IDs are interchangeable with audit-store IDs
    without re-validation at the boundary.
    """
    if not value:
        raise ValueError("identifier must not be empty")
    if not _SAFE_ID_PATTERN.match(value):
        raise ValueError(
            f"identifier must match [A-Za-z0-9._-]+ (got {value!r})"
        )
    if value in {".", ".."}:
        raise ValueError(
            f"identifier must not be a path-traversal segment (got {value!r})"
        )
    return value


SafeId = Annotated[str, AfterValidator(_validate_safe_id)]


def _ensure_utc(dt: datetime) -> datetime:
    """Reject naive datetimes; normalize aware non-UTC to UTC.

    Mirrors the audit_store / review_actions invariant. An alarm
    ``raised_at`` that is naive would compare incorrectly against the
    audit row's tz-aware ``order_datetime`` in dashboard queries.
    """
    if dt.tzinfo is None:
        raise ValueError(
            "datetime must be tz-aware; naive datetimes are forbidden in "
            "monitoring (see CONTEXT.md 'tz-aware UTC')"
        )
    return dt.astimezone(UTC)


UTCDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]


_WEEK_ISO_PATTERN = re.compile(r"^\d{4}-W\d{2}$")


def _validate_week_iso(value: str) -> str:
    """Reject any ``week_iso`` not in ISO ``YYYY-Www`` form (e.g., ``2026-W21``).

    The week tag participates in the deterministic sampling seed. A typo
    like ``2026-21`` or ``2026/W21`` would produce a different RNG seed
    and silently break the "same week → same audit_ids" invariant; the
    structural check at the model boundary stops that class of bug.
    """
    if not _WEEK_ISO_PATTERN.match(value):
        raise ValueError(
            f"week_iso must be ISO week format YYYY-Www (got {value!r})"
        )
    return value


WeekIso = Annotated[str, AfterValidator(_validate_week_iso)]


# =============================================================================
# Weekly reviewer sample
# =============================================================================


class WeeklyReviewerSample(BaseModel):
    """One week's deterministic clinical-reviewer sample manifest.

    The same ``(week_iso, sample_size, seed)`` always produces the same
    ``audit_ids`` — the determinism property the operator relies on for
    historical audit (re-deriving last quarter's sample from the manifest
    only requires the inputs, not the RNG state).
    """

    model_config = ConfigDict(frozen=True)

    week_iso: WeekIso
    sample_size: int
    seed: int
    audit_ids: tuple[SafeId, ...]
    drawn_at: UTCDatetime


# =============================================================================
# SPRT drift detection (Wald's Sequential Probability Ratio Test)
# =============================================================================


class SprtConfig(BaseModel):
    """Wald's SPRT parameters for one binomial drift signal.

    ``p_null`` is the long-run baseline rate (e.g., baseline quote-grounding
    failure rate measured over the last quarter). ``p_alt`` is the
    minimum-detectable shift the operator wants to flag (e.g., baseline +
    5pp). ``alpha``/``beta`` set the Wald bounds:

        A = log((1 - beta) / alpha)        # upper bound — reject H0
        B = log(beta / (1 - alpha))        # lower bound — accept H0

    Under H0 the average run length to a false alarm is ≈ -log(alpha) /
    (p_null * log(p_alt/p_null) + (1 - p_null) * log((1-p_alt)/(1-p_null))).
    Pick ``alpha`` to target the desired ARL₀ (PRD §18: ARL₀ ≥ 500).
    """

    model_config = ConfigDict(frozen=True)

    signal: DriftSignal
    p_null: float
    p_alt: float
    alpha: float = SPRT_DEFAULT_ALPHA
    beta: float = SPRT_DEFAULT_BETA
    min_n: int = 30


class SprtState(BaseModel):
    """Running state of one SPRT monitor.

    ``log_lr`` is the cumulative log-likelihood ratio; ``verdict`` is
    ``"continue"`` until ``log_lr`` crosses ``upper_bound`` (then
    ``"reject_null"``) or ``lower_bound`` (then ``"accept_null"``).
    """

    model_config = ConfigDict(frozen=True)

    signal: DriftSignal
    n_observations: int
    n_successes: int
    log_lr: float
    upper_bound: float
    lower_bound: float
    verdict: SprtVerdict


# =============================================================================
# Sentinel (weekly 200-case κ probe)
# =============================================================================


class SentinelManifest(BaseModel):
    """The 200-case sentinel set.

    Constructed ONCE with a deterministic seed (default 42). Re-running
    construction with the same population + seed MUST yield the same
    audit_ids; if it doesn't, the manifest is stale (the population
    changed) and :class:`SentinelStaleError` is raised by the constructor.
    """

    model_config = ConfigDict(frozen=True)

    size: int
    seed: int
    audit_ids: tuple[SafeId, ...]
    built_at: UTCDatetime


class SentinelComparison(BaseModel):
    """Result of comparing two sentinel-set runs.

    ``kappa`` is Cohen's κ over the paired classifications (computed via
    :func:`bba.eval_harness.agreement.cohen_kappa`); ``gwet_ac1`` is the
    prevalence-resistant companion. ``alarm_fired`` is True iff
    ``kappa < kappa_threshold``.
    """

    model_config = ConfigDict(frozen=True)

    n_paired: int
    cohen_kappa: float
    gwet_ac1: float
    kappa_threshold: float
    alarm_fired: bool


# =============================================================================
# Quarterly golden-set drift probe
# =============================================================================


class GoldenSetEntry(BaseModel):
    """One row in a golden-set run.

    Captures the two fields the quarterly drift probe compares:
    ``classification`` (the pipeline's final classification) and
    ``indications`` (the cited indication tags). A row in the baseline run
    is paired by ``audit_id`` to a row in the current run; missing pairs
    raise :class:`GoldenSetMismatchError`.
    """

    model_config = ConfigDict(frozen=True)

    audit_id: SafeId
    classification: Classification
    indications: tuple[str, ...]


class GoldenSetRowDelta(BaseModel):
    """Per-row delta in a golden-set drift comparison."""

    model_config = ConfigDict(frozen=True)

    audit_id: SafeId
    classification_changed: bool
    indications_changed: bool
    baseline_classification: Classification
    current_classification: Classification
    baseline_indications: tuple[str, ...]
    current_indications: tuple[str, ...]


class GoldenSetDriftReport(BaseModel):
    """Aggregate golden-set drift result.

    Two independent alarms: classification drift (``classification_changed_pct
    > GOLDEN_SET_CLASSIFICATION_DRIFT_THRESHOLD``) and indication drift
    (``indications_changed_pct > GOLDEN_SET_INDICATION_DRIFT_THRESHOLD``).
    Each fires independently — both can be True simultaneously.
    """

    model_config = ConfigDict(frozen=True)

    n_rows: int
    classification_changed_pct: float
    indications_changed_pct: float
    classification_alarm_fired: bool
    indications_alarm_fired: bool
    deltas: tuple[GoldenSetRowDelta, ...]


# =============================================================================
# Alarm record + persistence config
# =============================================================================


class MonitoringAlarmInput(BaseModel):
    """Input to :meth:`MonitoringStore.record_alarm`.

    ``detail`` is a free-form JSON-shaped mapping of monitor-specific
    fields (SPRT: ``log_lr``, ``n_observations``; sentinel: ``kappa``,
    ``n_paired``; golden_set: ``changed_pct``). The mapping is shallow on
    purpose — deep nesting would push complexity into the alarms-table
    query layer.
    """

    model_config = ConfigDict(frozen=True)

    kind: AlarmKind
    signal: DriftSignal | None
    raised_at: UTCDatetime
    detail: Mapping[str, str | int | float | bool]


class MonitoringAlarm(BaseModel):
    """A persisted alarm record. Append-only.

    ``alarm_id`` is the DB-assigned bigserial PK; ``raised_at`` is
    server-assigned ``now()`` at INSERT so the row's clock is always the
    Postgres server's.
    """

    model_config = ConfigDict(frozen=True)

    alarm_id: int
    kind: AlarmKind
    signal: DriftSignal | None
    raised_at: UTCDatetime
    detail: Mapping[str, str | int | float | bool]


class MonitoringConfig(BaseModel):
    """Postgres connection configuration for :class:`MonitoringStore`.

    Mirrors :class:`bba.review_actions.ReviewActionsConfig` field shape —
    secrets MUST NOT be hardcoded; the caller resolves credentials from
    env vars or a secret manager and constructs the config object.
    ``app_name`` is reported back via Postgres ``application_name`` so the
    DBA can correlate connections to the monitoring layer.
    """

    model_config = ConfigDict(frozen=True)

    dsn: str
    app_name: str = "bba.monitoring"


__all__: Sequence[str] = (
    "GOLDEN_SET_CLASSIFICATION_DRIFT_THRESHOLD",
    "GOLDEN_SET_INDICATION_DRIFT_THRESHOLD",
    "GOLDEN_SET_SIZE",
    "SENTINEL_KAPPA_ALARM_THRESHOLD",
    "SENTINEL_SET_SEED",
    "SENTINEL_SET_SIZE",
    "SPRT_DEFAULT_ALPHA",
    "SPRT_DEFAULT_BETA",
    "SPRT_TARGET_ARL0",
    "WEEKLY_REVIEWER_SAMPLE_MAX",
    "WEEKLY_REVIEWER_SAMPLE_MIN",
    "AlarmKind",
    "DriftSignal",
    "GoldenSetDriftReport",
    "GoldenSetEntry",
    "GoldenSetRowDelta",
    "MonitoringAlarm",
    "MonitoringAlarmInput",
    "MonitoringConfig",
    "SafeId",
    "SentinelComparison",
    "SentinelManifest",
    "SprtConfig",
    "SprtState",
    "SprtVerdict",
    "UTCDatetime",
    "WeekIso",
    "WeeklyReviewerSample",
)
