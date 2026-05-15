"""FastAPI app factory for the reviewer dashboard.

Five views, all rendered via FastAPI + HTMX + Tailwind:

* ``GET /queue`` — NEEDS_REVIEW queue (sortable).
* ``GET /case/{audit_id}`` — case-detail (un-redacted gated on
  care-team-of-record membership).
* ``POST /case/{audit_id}/break-glass`` — break-glass un-redacted access
  capture; writes a ``phi_access_log`` row BEFORE surfacing un-redacted
  text.
* ``GET /scorecard/ward/{ward_id}`` — per-ward scorecard.
* ``GET /scorecard/physician/{physician_id}`` — per-physician own-view
  scorecard (route guard: requesting physician = subject physician).
* ``GET /pipeline-health`` — pipeline-health dashboard.

This module declares the factory contract only. RED-phase scaffolding —
every body is :func:`NotImplementedError` so the test suite drives the
implementation in GREEN.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bba.dashboard.models import (
    BreakGlassRequest,
    CaseDetail,
    DashboardConfig,
    PhysicianScorecard,
    PipelineHealth,
    QueueItem,
    QueueSortKey,
    Reviewer,
    RouteContext,
    SortDirection,
    WardScorecard,
)


if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app(config: DashboardConfig) -> FastAPI:
    """Construct and return the FastAPI application for the reviewer UI.

    The factory wires the five views above and the break-glass endpoint to
    the audit-store and review-actions-store handles inside ``config``.
    Sessions resolve reviewer identity via the auth dependency; the
    dependency is injectable for tests.
    """
    raise NotImplementedError("bba.dashboard.create_app is not yet implemented")


def list_queue(
    config: DashboardConfig,
    context: RouteContext,
    *,
    sort_key: QueueSortKey = "order_datetime",
    sort_direction: SortDirection = "desc",
) -> tuple[QueueItem, ...]:
    """Return the sortable NEEDS_REVIEW queue for the reviewer's scope.

    Reads via :class:`bba.audit_store.SnapshotView` (daily-rotated) so an
    in-flight batch write does not produce mid-query inconsistency.
    """
    raise NotImplementedError("list_queue is not yet implemented")


def get_case_detail(
    config: DashboardConfig,
    context: RouteContext,
    audit_id: str,
) -> CaseDetail:
    """Return the case-detail view for ``audit_id``.

    Care-team-of-record gating: if the reviewer is on the case's care team,
    ``CaseDetail.unredacted=True`` and ``raw_hn``/``raw_an`` are populated.
    Otherwise ``unredacted=False`` and the un-redacted fields are ``None``.
    The break-glass path (:func:`record_break_glass_access`) is the only
    way to obtain un-redacted access without care-team membership.
    """
    raise NotImplementedError("get_case_detail is not yet implemented")


def record_break_glass_access(
    config: DashboardConfig,
    context: RouteContext,
    audit_id: str,
    request: BreakGlassRequest,
) -> CaseDetail:
    """Write a ``phi_access_log`` row carrying ``request.justification``,
    THEN return the un-redacted :class:`CaseDetail`.

    The log row is committed BEFORE the un-redacted text is surfaced
    (structural via :meth:`ReviewActionsStore.access_phi` context manager).
    A break-glass request with an empty / whitespace-only justification is
    rejected at the model boundary by :data:`NonEmptyStr`.
    """
    raise NotImplementedError(
        "record_break_glass_access is not yet implemented"
    )


def get_ward_scorecard(
    config: DashboardConfig,
    context: RouteContext,
    ward_id: str,
) -> WardScorecard:
    """Return the per-ward aggregation scorecard for ``ward_id``."""
    raise NotImplementedError("get_ward_scorecard is not yet implemented")


def get_physician_scorecard(
    config: DashboardConfig,
    context: RouteContext,
    physician_id: str,
) -> PhysicianScorecard:
    """Return the per-physician own-view scorecard.

    Route guard: raises :class:`PhysicianAccessDeniedError` when
    ``context.reviewer.physician_id != physician_id``. This is the
    own-view-only contract from PRD §17.
    """
    raise NotImplementedError(
        "get_physician_scorecard is not yet implemented"
    )


def get_pipeline_health(
    config: DashboardConfig,
    context: RouteContext,
) -> PipelineHealth:
    """Return pipeline-health aggregations (NEEDS_REVIEW rate, verifier
    pass rate, escalation rate, average latency, snapshot age)."""
    raise NotImplementedError(
        "get_pipeline_health is not yet implemented"
    )


def resolve_reviewer(
    config: DashboardConfig,
    reviewer_id: str,
) -> Reviewer:
    """Resolve a reviewer identity from the request session.

    The default implementation is a placeholder that route handlers can
    override via FastAPI's dependency-injection system in production.
    """
    raise NotImplementedError("resolve_reviewer is not yet implemented")


__all__ = (
    "create_app",
    "get_case_detail",
    "get_physician_scorecard",
    "get_pipeline_health",
    "get_ward_scorecard",
    "list_queue",
    "record_break_glass_access",
    "resolve_reviewer",
)
