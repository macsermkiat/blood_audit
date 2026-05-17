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

Alerting in Phase 1 is structured-log only (stdlib ``logging`` with
structured ``extra`` fields, surfaced via the ``bba.monitoring.alarms``
logger). The alarm record is also captured in :class:`MonitoringStore`
— an in-memory store in Phase 1; a Postgres-backed implementation that
matches :mod:`bba.review_actions`'s append-only contract is the
documented Phase 1.5 follow-up. Slack / email / paging integration is
also Phase 1.5.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator

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


SPRT_DEFAULT_MIN_N: int = 30
"""Minimum observations before the SPRT can return a non-``continue`` verdict.

The min-N gate prevents single-observation alarms when the random walk
happens to start with a long success run. PRD §18 does not fix a specific
value; 30 is the deployment default and is exported as a constant so
operators tune it explicitly rather than via a magic number in
:class:`SprtConfig`."""


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
        raise ValueError(f"identifier must match [A-Za-z0-9._-]+ (got {value!r})")
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

    Also rejects datetimes carrying a ``tzinfo`` whose ``utcoffset()``
    returns ``None`` — Python admits this shape (e.g., a custom
    ``tzinfo`` subclass) but it is operationally naive: comparisons with
    a true UTC-aware datetime still raise ``TypeError`` at runtime.
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(
            "datetime must be tz-aware (with a concrete utcoffset); "
            "naive datetimes are forbidden in monitoring "
            "(see CONTEXT.md 'tz-aware UTC')"
        )
    return dt.astimezone(UTC)


def ensure_utc(dt: datetime) -> datetime:
    """Public re-export of :func:`_ensure_utc` for callers that need to
    normalize a caller-supplied datetime (e.g., :meth:`MonitoringStore.list_alarms`
    on its ``since`` filter)."""
    return _ensure_utc(dt)


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
        raise ValueError(f"week_iso must be ISO week format YYYY-Www (got {value!r})")
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
    min_n: int = SPRT_DEFAULT_MIN_N

    @model_validator(mode="after")
    def _validate_sprt_parameters(self) -> SprtConfig:
        """Reject nonsense rates / error bounds at construction time.

        The hot path computes ``math.log(p_alt / p_null)`` and
        ``math.log((1 - p_alt) / (1 - p_null))`` — both blow up for
        boundary values. Field-level validation here makes the failure
        mode loud at config time instead of silent inside the SPRT loop.

        Invariants:

        * ``0 < p_null < 1`` and ``0 < p_alt < 1`` — both rates are
          proper Bernoulli probabilities.
        * ``p_null < p_alt`` — the alternative must be larger than the
          null (otherwise the per-step log-LR would have the wrong sign
          and the SPRT would point in the opposite direction).
        * ``0 < alpha < 1`` and ``0 < beta < 1`` — Wald bounds require
          both error rates strictly inside the open unit interval; the
          boundary values would push the bounds to ±∞.
        * ``min_n >= 1`` — the min-N gate is a positive integer count.
        """
        if not 0.0 < self.p_null < 1.0:
            raise ValueError(f"p_null must be in (0, 1), got {self.p_null!r}")
        if not 0.0 < self.p_alt < 1.0:
            raise ValueError(f"p_alt must be in (0, 1), got {self.p_alt!r}")
        if self.p_null >= self.p_alt:
            raise ValueError(
                f"p_null ({self.p_null}) must be strictly less than "
                f"p_alt ({self.p_alt}); same/inverted rates produce a "
                f"log-LR with the wrong sign"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha!r}")
        if not 0.0 < self.beta < 1.0:
            raise ValueError(f"beta must be in (0, 1), got {self.beta!r}")
        if self.min_n < 1:
            raise ValueError(f"min_n must be >= 1, got {self.min_n!r}")
        return self


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
    construction with the same population + seed yields the same
    audit_ids; if the underlying population has changed between calls,
    the manifest's audit_ids may no longer all exist in the new
    population. Staleness detection is a Phase 1.5 follow-up (the
    operator currently re-runs :func:`build_sentinel_manifest` once per
    deployment and persists the manifest via
    :class:`MonitoringStore.persist_sample_manifest` for re-use).
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

    ``alarm_id`` is a monotonically-increasing assigned integer (in-process
    counter in Phase 1; Phase 1.5 will substitute a Postgres bigserial).
    ``raised_at`` is captured by the caller at the moment the alarm is
    raised — the store does not overwrite it (so an out-of-order replay
    keeps its original timestamp).
    """

    model_config = ConfigDict(frozen=True)

    alarm_id: int
    kind: AlarmKind
    signal: DriftSignal | None
    raised_at: UTCDatetime
    detail: Mapping[str, str | int | float | bool]


class MonitoringConfig(BaseModel):
    """Connection configuration for :class:`MonitoringStore`.

    The shape mirrors :class:`bba.review_actions.ReviewActionsConfig` so a
    Phase 1.5 Postgres-backed swap doesn't break callers: ``dsn`` and
    ``app_name`` are the two fields a libpq-style backend would consume.
    Phase 1's in-memory store ignores both fields; they're carried on the
    config so the transition is a substitution rather than a re-typing
    exercise.

    Secrets MUST NOT be hardcoded; the caller resolves credentials from
    env vars or a secret manager and constructs the config object.
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
    "SPRT_DEFAULT_MIN_N",
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
    "ensure_utc",
)
