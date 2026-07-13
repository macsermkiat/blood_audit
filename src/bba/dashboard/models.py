"""Frozen pydantic models for the dashboard contract.

The dashboard composes two persistence dependencies declared in PRD §17:

* :class:`bba.audit_store.AuditStore` — Parquet-backed audit-row store,
  consumed via :class:`bba.audit_store.SnapshotView` for daily-rotated
  read-only views (#19).
* :class:`bba.review_actions.ReviewActionsStore` — Postgres-backed reviewer
  decisions + ``phi_access_log`` (#25).

Models declared here are the view-layer DTOs the route handlers return;
they mirror persisted shapes from the two stores but never carry mutable
references — every model is frozen, every nested container is a tuple.

The same ``UTCDatetime`` / ``SafeId`` / ``NonEmptyStr`` invariants from
``bba.audit_store`` and ``bba.review_actions`` apply here at the input
boundary so a naive timestamp or a path-traversal segment cannot leak past
construction.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict

from bba.audit_store import AuditRow, AuditStore, Classification
from bba.review_actions import ReviewActionsStore


# ---------------------------------------------------------------------------
# Injectable resolvers — the dashboard does not own un-redacted PHI nor
# ward / physician attribution. Production wires real implementations
# (HIS / de-id-twin store, ward registry); tests provide fakes.
# ---------------------------------------------------------------------------


UnredactedPhiResolver = Callable[[str, str], tuple[str, str]]
"""Resolve un-redacted ``(hn, an)`` from their hashes.

Signature: ``(hn_hash: str, an_hash: str) -> (hn: str, an: str)``.
The dashboard never holds un-redacted PHI in its own storage; this
callable is the integration point with the HIS or de-id-twin store.
"""


WardAttributionResolver = Callable[[AuditRow], str]
"""Resolve the ward attribution for an :class:`AuditRow`.

Signature: ``(row: AuditRow) -> ward_id: str``.
Wired to the ward registry in production; the audit_store schema does not
carry ward_id directly so attribution is computed at view time.
"""


PhysicianAttributionResolver = Callable[[AuditRow], str]
"""Resolve the attending-physician attribution for an :class:`AuditRow`.

Signature: ``(row: AuditRow) -> physician_id: str``.
Wired to the physician registry in production. The per-physician own-view
route guard reads :attr:`Reviewer.physician_id` against the URL path
parameter; this resolver is the *data*-side counterpart used by the
scorecard aggregations.
"""


# ---------------------------------------------------------------------------
# Shared annotated types (mirroring audit_store + review_actions).
# ---------------------------------------------------------------------------


def _ensure_utc(dt: datetime) -> datetime:
    """Reject naive datetimes; normalize aware non-UTC to UTC.

    The dashboard composes datetimes from two stores that both enforce
    tz-aware UTC. Re-applying the invariant at the view boundary keeps a
    future caller that constructs a model directly (e.g., test fixtures)
    from leaking naive timestamps into the rendered HTML.
    """
    if dt.tzinfo is None:
        raise ValueError(
            "datetime must be tz-aware; naive datetimes are forbidden in "
            "bba.dashboard (see CONTEXT.md 'tz-aware UTC')"
        )
    return dt.astimezone(UTC)


UTCDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]


_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_safe_id(value: str) -> str:
    """Reject identifiers that would be unsafe in path / log contexts."""
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


def _validate_nonempty(value: str) -> str:
    """Reject empty / whitespace-only strings on required-text fields."""
    if not value or not value.strip():
        raise ValueError("text must not be empty or whitespace-only")
    return value


NonEmptyStr = Annotated[str, AfterValidator(_validate_nonempty)]


# ---------------------------------------------------------------------------
# Identity / role models.
# ---------------------------------------------------------------------------


ReviewerRole = Literal["reviewer", "physician", "senior_reviewer", "admin"]


class Reviewer(BaseModel):
    """The viewing identity for a dashboard session.

    ``care_team_memberships`` enumerates the audit_id values whose
    care-team-of-record this reviewer is on. Membership is the gate on
    un-redacted case-detail access in the no-break-glass path; an empty
    tuple means the reviewer can only access cases via break-glass.

    ``physician_id`` is set when the reviewer's primary role is
    ``physician`` so the own-view route guard can compare it against the
    URL's ``{physician_id}`` path parameter.
    """

    model_config = ConfigDict(frozen=True)

    reviewer_id: SafeId
    name: str
    role: ReviewerRole
    physician_id: SafeId | None = None
    care_team_memberships: tuple[SafeId, ...] = ()


class Physician(BaseModel):
    """A physician identity for scorecards."""

    model_config = ConfigDict(frozen=True)

    physician_id: SafeId
    name: str
    ward_id: SafeId


class Ward(BaseModel):
    """A ward identity for scorecards."""

    model_config = ConfigDict(frozen=True)

    ward_id: SafeId
    name: str


# ---------------------------------------------------------------------------
# View-layer DTOs.
# ---------------------------------------------------------------------------


QueueSortKey = Literal["order_datetime", "hb_value", "confidence", "audit_id"]
SortDirection = Literal["asc", "desc"]


class QueueItem(BaseModel):
    """One row in the NEEDS_REVIEW queue view.

    Carries only the redacted projection of the audit row — the queue is a
    triage surface, not an un-redacted view. ``hn_hash`` / ``an_hash`` are
    surfaced (not the raw HN/AN) because the queue is rendered without a
    break-glass justification.
    """

    model_config = ConfigDict(frozen=True)

    audit_id: SafeId
    run_id: SafeId
    order_datetime: UTCDatetime
    hb_value: float
    # ``hb_freshness == "missing"`` marks the 0.0 sentinel emitted by the
    # missing-Hb positive-evidence bypass (MTP / peri-procedural). Templates
    # MUST consult this before formatting ``hb_value`` so reviewers do not
    # see "0.0" as if it were a real measurement.
    hb_freshness: str
    confidence: float
    final_classification: Classification
    review_reason: str | None
    ward_id: SafeId
    hn_hash: str
    an_hash: str


class CaseDetail(BaseModel):
    """The case-detail view payload.

    ``unredacted`` is ``True`` only when the reviewer cleared either the
    care-team gating or the break-glass justification path. When ``False``,
    ``raw_hn`` / ``raw_an`` MUST be ``None`` and the rendered template falls
    back to the redacted projection.
    """

    model_config = ConfigDict(frozen=True)

    audit_id: SafeId
    run_id: SafeId
    final_classification: Classification
    rule_classification: Classification
    confidence: float
    reasoning_summary_en: str
    reasoning_summary_thai: str
    indications_json: tuple[dict[str, object], ...]
    negative_evidence_json: tuple[dict[str, object], ...]
    needs_human_review: bool
    review_reason: str | None
    hb_value: float
    # See :class:`QueueItem.hb_freshness` — same sentinel-detection contract.
    hb_freshness: str
    order_datetime: UTCDatetime
    unredacted: bool
    raw_hn: str | None
    raw_an: str | None


class WardScorecard(BaseModel):
    """Per-ward aggregations rendered as the ward-scorecard view."""

    model_config = ConfigDict(frozen=True)

    ward_id: SafeId
    ward_name: str
    total_orders: int
    appropriate_count: int
    inappropriate_count: int
    needs_review_count: int
    insufficient_evidence_count: int
    returned_not_transfused_count: int = 0
    periop_transfusion_exempt_count: int = 0
    average_confidence: float


class PhysicianScorecard(BaseModel):
    """Per-physician own-view scorecard."""

    model_config = ConfigDict(frozen=True)

    physician_id: SafeId
    physician_name: str
    ward_id: SafeId
    total_orders: int
    appropriate_count: int
    inappropriate_count: int
    needs_review_count: int
    insufficient_evidence_count: int
    returned_not_transfused_count: int = 0
    periop_transfusion_exempt_count: int = 0
    average_confidence: float


class PipelineHealth(BaseModel):
    """Pipeline-health metrics.

    Surfaces the rates the operator monitors: NEEDS_REVIEW rate, verifier
    pass rate, Sonnet→Opus escalation rate, average classification latency.
    """

    model_config = ConfigDict(frozen=True)

    total_audits: int
    needs_review_rate: float
    verifier_pass_rate: float
    escalation_rate: float
    average_latency_ms: float
    snapshot_as_of: UTCDatetime


# ---------------------------------------------------------------------------
# Request / context models.
# ---------------------------------------------------------------------------


class BreakGlassRequest(BaseModel):
    """POST body for the break-glass un-redacted-view endpoint.

    ``justification`` is the reviewer-supplied free-form rationale that gets
    written to ``phi_access_log.break_glass_justification`` BEFORE any
    un-redacted text is surfaced. ``NonEmptyStr`` enforces non-empty at the
    boundary; an empty justification is rejected with HTTP 400 before any
    audit-log row is written.
    """

    model_config = ConfigDict(frozen=True)

    justification: NonEmptyStr


class RouteContext(BaseModel):
    """The current request's resolved identity + access context.

    Constructed by the auth dependency at the top of every route. Carries
    the reviewer, the optional subject physician (for own-view checks), and
    the optional ward filter.
    """

    model_config = ConfigDict(frozen=True)

    reviewer: Reviewer
    subject_physician_id: SafeId | None = None
    ward_id: SafeId | None = None


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------


class DashboardConfig(BaseModel):
    """Configuration for the FastAPI app factory.

    The dashboard does not own persistence; it composes the audit-store and
    review-actions-store handles passed in by the caller. ``snapshot_dir``
    is informational — the actual snapshot rotation is performed inside
    :class:`bba.audit_store.SnapshotView`.

    ``default_reviewer`` is the fallback reviewer identity for requests
    that have not been resolved by an upstream auth dependency. Production
    deployments wire a real auth dependency via FastAPI's
    ``app.dependency_overrides`` mechanism; for dev / unit tests the
    default keeps routes reachable without a session.

    ``unredacted_phi_resolver`` is the injectable callable that maps an
    ``(hn_hash, an_hash)`` pair back to its un-redacted ``(hn, an)``. The
    dashboard refuses to surface un-redacted PHI when this is ``None``
    (raises :class:`UnredactedSourceUnavailableError`) — the integration
    point with the HIS / de-id-twin store is explicit, not implicit.

    ``ward_attribution_resolver`` / ``physician_attribution_resolver``
    map an audit row to its ward and physician identifiers. The
    audit_store schema does not carry these directly; production wires
    real registries, tests provide deterministic fakes.

    ``template_dir`` overrides the default Jinja2 template directory
    (``src/bba/dashboard/templates/``). Used by tests that ship custom
    templates and by deployments that themed the dashboard.

    ``arbitrary_types_allowed`` is required because :class:`AuditStore` and
    :class:`ReviewActionsStore` are not pydantic models themselves; they're
    composed as runtime handles.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    audit_store: AuditStore
    review_actions_store: ReviewActionsStore
    snapshot_dir: Path
    template_dir: Path | None = None
    default_reviewer: Reviewer | None = None
    unredacted_phi_resolver: UnredactedPhiResolver | None = None
    ward_attribution_resolver: WardAttributionResolver | None = None
    physician_attribution_resolver: PhysicianAttributionResolver | None = None


__all__: Sequence[str] = (
    "BreakGlassRequest",
    "CaseDetail",
    "DashboardConfig",
    "NonEmptyStr",
    "Physician",
    "PhysicianAttributionResolver",
    "PhysicianScorecard",
    "PipelineHealth",
    "QueueItem",
    "QueueSortKey",
    "Reviewer",
    "ReviewerRole",
    "RouteContext",
    "SafeId",
    "SortDirection",
    "UTCDatetime",
    "UnredactedPhiResolver",
    "Ward",
    "WardAttributionResolver",
    "WardScorecard",
)
