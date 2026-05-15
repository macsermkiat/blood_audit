"""FastAPI app factory + handler functions for the reviewer dashboard.

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

The handler functions are exposed for direct unit-test invocation (they
raise typed :class:`DashboardError` subclasses). The route layer wraps
each handler and translates :class:`DashboardError` into HTTP responses.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi import Request as FastApiRequest

from bba.audit_store import AuditRow, Classification, SnapshotView
from bba.dashboard.exceptions import (
    AuditNotFoundError,
    DashboardError,
    MissingJustificationError,
    PhysicianAccessDeniedError,
)
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
from bba.review_actions import PhiAccessInput


# ---------------------------------------------------------------------------
# Defaults — the dashboard does not (yet) own a ward / physician registry.
# PRD §17 attribution-by-registry is out of scope for #26; the defaults
# below keep the view-layer wiring sound until the registry lands. A future
# ticket (#XX) wires real attribution; #26 only needs to render the
# stable surface.
# ---------------------------------------------------------------------------


_DEFAULT_WARD_ID = "default-ward"
_DEFAULT_WARD_NAME = "Default Ward"
_DEFAULT_PHYSICIAN_NAME = "Default Physician"


# ---------------------------------------------------------------------------
# Pure helpers — projection from AuditRow into view-layer DTOs.
# ---------------------------------------------------------------------------


def _ward_id_for_row(row: AuditRow) -> str:
    """Resolve the ward attribution for an audit row.

    Placeholder until the ward registry (separate ticket) is wired. Every
    row maps to ``default-ward`` so the queue / ward-scorecard surface
    renders. Replacing this function is the integration point for the
    real registry; nothing else in the dashboard needs to change.
    """
    return _DEFAULT_WARD_ID


def _physician_id_for_row(row: AuditRow) -> str:
    """Resolve the physician attribution for an audit row.

    Placeholder until the physician registry is wired. Returns the same
    fixed id for all rows so the physician-scorecard surface renders
    while the registry is still pending.
    """
    return "default-physician"


def _audit_row_to_queue_item(row: AuditRow) -> QueueItem:
    return QueueItem(
        audit_id=row.audit_id,
        run_id=row.run_id,
        order_datetime=row.order_datetime,
        hb_value=row.hb_value,
        confidence=row.confidence,
        final_classification=row.final_classification,
        review_reason=row.review_reason,
        ward_id=_ward_id_for_row(row),
        hn_hash=row.hn_hash,
        an_hash=row.an_hash,
    )


def _audit_row_to_case_detail(
    row: AuditRow, *, unredacted: bool
) -> CaseDetail:
    """Project an AuditRow into a CaseDetail.

    Un-redacted HN/AN are synthetic placeholders for #26 (the un-redacted
    text source is the de-id store, a separate dependency to be wired in a
    follow-up). The placeholder is stable per audit_id so view-layer tests
    can assert structural shape without depending on the real source.
    """
    if unredacted:
        raw_hn: str | None = f"hn-actual-{row.audit_id}"
        raw_an: str | None = f"an-actual-{row.audit_id}"
    else:
        raw_hn = None
        raw_an = None
    return CaseDetail(
        audit_id=row.audit_id,
        run_id=row.run_id,
        final_classification=row.final_classification,
        rule_classification=row.rule_classification,
        confidence=row.confidence,
        reasoning_summary_en=row.reasoning_summary_en,
        reasoning_summary_thai=row.reasoning_summary_thai,
        indications_json=tuple(dict(item) for item in row.indications_json),
        negative_evidence_json=tuple(
            dict(item) for item in row.negative_evidence_json
        ),
        needs_human_review=row.needs_human_review,
        review_reason=row.review_reason,
        hb_value=row.hb_value,
        order_datetime=row.order_datetime,
        unredacted=unredacted,
        raw_hn=raw_hn,
        raw_an=raw_an,
    )


def _read_snapshot_rows(config: DashboardConfig) -> tuple[AuditRow, ...]:
    """Read audit rows via the daily-rotated DuckDB snapshot view.

    Keyed by ``datetime.now(UTC).date()`` — same-day reads share the
    materialized snapshot, mid-batch writes are isolated. The first read
    of a given day materializes the view; subsequent reads return the
    same frozen result set.
    """
    today = datetime.now(UTC).date()
    view = SnapshotView.open(config.audit_store, today)
    return view.read_audit_results()


def _find_audit_row(config: DashboardConfig, audit_id: str) -> AuditRow:
    for row in _read_snapshot_rows(config):
        if row.audit_id == audit_id:
            return row
    raise AuditNotFoundError(
        f"audit_id {audit_id!r} not found in today's snapshot"
    )


def _count_classification(
    rows: tuple[AuditRow, ...], classification: Classification
) -> int:
    return sum(1 for r in rows if r.final_classification == classification)


def _aggregate_classifications(
    rows: tuple[AuditRow, ...],
) -> dict[str, int | float]:
    """Compute the count + average-confidence aggregations for a row set.

    Used by both ward and physician scorecards (same aggregation shape).
    """
    total = len(rows)
    avg_confidence = (
        sum(r.confidence for r in rows) / total if total > 0 else 0.0
    )
    return {
        "total_orders": total,
        "appropriate_count": _count_classification(rows, "APPROPRIATE"),
        "inappropriate_count": _count_classification(rows, "INAPPROPRIATE"),
        "needs_review_count": _count_classification(rows, "NEEDS_REVIEW"),
        "insufficient_evidence_count": _count_classification(
            rows, "INSUFFICIENT_EVIDENCE"
        ),
        "average_confidence": avg_confidence,
    }


# ---------------------------------------------------------------------------
# Handler functions — called directly from unit tests, and indirectly from
# the route layer in :func:`create_app`. Raise typed :class:`DashboardError`
# subclasses; the route layer translates to HTTP.
# ---------------------------------------------------------------------------


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
    rows = _read_snapshot_rows(config)
    review_rows = tuple(
        r for r in rows if r.final_classification == "NEEDS_REVIEW"
    )
    items = [_audit_row_to_queue_item(r) for r in review_rows]
    sort_fn: dict[str, Callable[[QueueItem], Any]] = {
        "order_datetime": lambda i: i.order_datetime,
        "hb_value": lambda i: i.hb_value,
        "confidence": lambda i: i.confidence,
        "audit_id": lambda i: i.audit_id,
    }
    items.sort(key=sort_fn[sort_key], reverse=(sort_direction == "desc"))
    return tuple(items)


def get_case_detail(
    config: DashboardConfig,
    context: RouteContext,
    audit_id: str,
) -> CaseDetail:
    """Return the case-detail view for ``audit_id``.

    Care-team-of-record gating: when ``audit_id`` is in the reviewer's
    ``care_team_memberships`` the response is un-redacted. Outsiders see
    the redacted projection (``unredacted=False``,
    ``raw_hn``/``raw_an=None``) — the break-glass flow
    (:func:`record_break_glass_access`) is the only way to obtain
    un-redacted access without care-team membership.
    """
    row = _find_audit_row(config, audit_id)
    unredacted = audit_id in context.reviewer.care_team_memberships
    return _audit_row_to_case_detail(row, unredacted=unredacted)


def record_break_glass_access(
    config: DashboardConfig,
    context: RouteContext,
    audit_id: str,
    request: BreakGlassRequest,
) -> CaseDetail:
    """Write a ``phi_access_log`` row carrying ``request.justification``,
    THEN return the un-redacted :class:`CaseDetail`.

    Defense in depth: the justification is re-checked at the handler
    boundary even though :data:`NonEmptyStr` enforces non-empty at
    construction. The test bypasses model validation via
    ``BreakGlassRequest.model_construct`` to simulate a malformed request
    that reached the handler.

    The log row is committed BEFORE the un-redacted text is surfaced
    (structurally via :meth:`ReviewActionsStore.access_phi` context
    manager — write happens at ``__enter__``, not on body success).
    """
    if not request.justification or not request.justification.strip():
        raise MissingJustificationError(
            "break-glass justification must be a non-empty, "
            "non-whitespace string; un-redacted access is not granted "
            "without a reviewer-supplied rationale"
        )
    row = _find_audit_row(config, audit_id)
    access = PhiAccessInput(
        reviewer_id=context.reviewer.reviewer_id,
        audit_id=audit_id,
        hn_hash=row.hn_hash,
        an_hash=row.an_hash,
        break_glass_justification=request.justification,
    )
    with config.review_actions_store.access_phi(access):
        return _audit_row_to_case_detail(row, unredacted=True)


def get_ward_scorecard(
    config: DashboardConfig,
    context: RouteContext,
    ward_id: str,
) -> WardScorecard:
    """Return the per-ward aggregation scorecard for ``ward_id``.

    An empty result (ward not represented in today's snapshot) returns a
    scorecard with zero counts — the operator may legitimately check a
    ward that has no audited orders today.
    """
    rows = _read_snapshot_rows(config)
    ward_rows = tuple(r for r in rows if _ward_id_for_row(r) == ward_id)
    aggs = _aggregate_classifications(ward_rows)
    return WardScorecard(
        ward_id=ward_id,
        ward_name=_DEFAULT_WARD_NAME,
        total_orders=int(aggs["total_orders"]),
        appropriate_count=int(aggs["appropriate_count"]),
        inappropriate_count=int(aggs["inappropriate_count"]),
        needs_review_count=int(aggs["needs_review_count"]),
        insufficient_evidence_count=int(aggs["insufficient_evidence_count"]),
        average_confidence=float(aggs["average_confidence"]),
    )


def get_physician_scorecard(
    config: DashboardConfig,
    context: RouteContext,
    physician_id: str,
) -> PhysicianScorecard:
    """Return the per-physician own-view scorecard.

    Route guard: a reviewer whose ``role == 'physician'`` may ONLY view
    their own scorecard. A request for another physician's id raises
    :class:`PhysicianAccessDeniedError` before any data retrieval —
    PRD §17 makes this the route layer's responsibility, not the UI's.

    Non-physician roles (senior reviewer, admin) are permitted to view
    arbitrary physician scorecards; the guard is "physician cannot view
    OTHER's", not "only physicians can view".
    """
    if (
        context.reviewer.role == "physician"
        and context.reviewer.physician_id != physician_id
    ):
        raise PhysicianAccessDeniedError(
            f"physician {context.reviewer.physician_id!r} cannot view "
            f"another physician's scorecard ({physician_id!r}); the "
            f"per-physician view is own-view-only"
        )
    rows = _read_snapshot_rows(config)
    phys_rows = tuple(
        r for r in rows if _physician_id_for_row(r) == physician_id
    )
    aggs = _aggregate_classifications(phys_rows)
    return PhysicianScorecard(
        physician_id=physician_id,
        physician_name=_DEFAULT_PHYSICIAN_NAME,
        ward_id=_DEFAULT_WARD_ID,
        total_orders=int(aggs["total_orders"]),
        appropriate_count=int(aggs["appropriate_count"]),
        inappropriate_count=int(aggs["inappropriate_count"]),
        needs_review_count=int(aggs["needs_review_count"]),
        insufficient_evidence_count=int(aggs["insufficient_evidence_count"]),
        average_confidence=float(aggs["average_confidence"]),
    )


def get_pipeline_health(
    config: DashboardConfig,
    context: RouteContext,
) -> PipelineHealth:
    """Return pipeline-health aggregations.

    Rates are computed from the snapshot's row set:

    * ``needs_review_rate`` — fraction of audits flagged NEEDS_REVIEW.
    * ``verifier_pass_rate`` — fraction with ``verifier_pass=True``.
    * ``escalation_rate`` — fraction with ``escalated_to_opus=True``.
    * ``average_latency_ms`` — placeholder (the audit_store snapshot does
      not include ``LlmCall`` latency rows in the daily view; pulling
      those in is a follow-up).
    """
    rows = _read_snapshot_rows(config)
    total = len(rows)
    if total == 0:
        return PipelineHealth(
            total_audits=0,
            needs_review_rate=0.0,
            verifier_pass_rate=0.0,
            escalation_rate=0.0,
            average_latency_ms=0.0,
            snapshot_as_of=datetime.now(UTC),
        )
    needs_review = _count_classification(rows, "NEEDS_REVIEW")
    verifier_pass = sum(1 for r in rows if r.verifier_pass)
    escalated = sum(1 for r in rows if r.escalated_to_opus)
    return PipelineHealth(
        total_audits=total,
        needs_review_rate=needs_review / total,
        verifier_pass_rate=verifier_pass / total,
        escalation_rate=escalated / total,
        average_latency_ms=0.0,
        snapshot_as_of=datetime.now(UTC),
    )


def resolve_reviewer(
    config: DashboardConfig,
    reviewer_id: str,
) -> Reviewer:
    """Resolve a reviewer identity from a session id.

    Production wires this to a real auth backend via FastAPI's
    ``app.dependency_overrides``. The default implementation is a fallthrough
    to :attr:`DashboardConfig.default_reviewer` (set by tests / dev configs).
    """
    if config.default_reviewer is None:
        raise DashboardError(
            f"reviewer_id {reviewer_id!r} could not be resolved; no "
            f"default reviewer configured and no auth dependency overrode "
            f"this function"
        )
    return config.default_reviewer


# ---------------------------------------------------------------------------
# FastAPI app factory.
# ---------------------------------------------------------------------------


def create_app(config: DashboardConfig) -> FastAPI:
    """Construct and return the FastAPI application for the reviewer UI.

    Routes wrap the module-level handler functions and translate
    :class:`DashboardError` subclasses into HTTP responses:

    * :class:`AuditNotFoundError` → 404
    * :class:`PhysicianAccessDeniedError` → 403
    * :class:`MissingJustificationError` → 400
    * :class:`DashboardError` (catch-all) → 500

    Reviewer identity is resolved via a FastAPI dependency
    (:func:`_route_context_dep`). The dependency returns the configured
    ``default_reviewer`` (for dev / smoke tests) or raises 401. Production
    wires a real auth dependency via ``app.dependency_overrides``.
    """
    app = FastAPI(title="bba.dashboard", version="0.1.0")

    def _route_context_dep(request: FastApiRequest) -> RouteContext:
        # ``request`` is unused; accepted so production overrides can
        # inspect headers / cookies without a signature change.
        del request
        if config.default_reviewer is None:
            raise HTTPException(
                status_code=401,
                detail=(
                    "reviewer not resolved; no default reviewer configured "
                    "and no auth dependency overrode this route"
                ),
            )
        return RouteContext(reviewer=config.default_reviewer)

    @app.get("/queue")
    def _queue_route(
        context: RouteContext = Depends(_route_context_dep),
    ) -> dict[str, Any]:
        items = list_queue(config, context)
        return {"items": [item.model_dump(mode="json") for item in items]}

    @app.get("/case/{audit_id}")
    def _case_detail_route(
        audit_id: str,
        context: RouteContext = Depends(_route_context_dep),
    ) -> dict[str, Any]:
        try:
            detail = get_case_detail(config, context, audit_id)
        except AuditNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return detail.model_dump(mode="json")

    @app.post("/case/{audit_id}/break-glass")
    def _break_glass_route(
        audit_id: str,
        request: BreakGlassRequest,
        context: RouteContext = Depends(_route_context_dep),
    ) -> dict[str, Any]:
        try:
            detail = record_break_glass_access(
                config, context, audit_id, request
            )
        except MissingJustificationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AuditNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return detail.model_dump(mode="json")

    @app.get("/scorecard/ward/{ward_id}")
    def _ward_scorecard_route(
        ward_id: str,
        context: RouteContext = Depends(_route_context_dep),
    ) -> dict[str, Any]:
        sc = get_ward_scorecard(config, context, ward_id)
        return sc.model_dump(mode="json")

    @app.get("/scorecard/physician/{physician_id}")
    def _physician_scorecard_route(
        physician_id: str,
        context: RouteContext = Depends(_route_context_dep),
    ) -> dict[str, Any]:
        try:
            sc = get_physician_scorecard(config, context, physician_id)
        except PhysicianAccessDeniedError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return sc.model_dump(mode="json")

    @app.get("/pipeline-health")
    def _pipeline_health_route(
        context: RouteContext = Depends(_route_context_dep),
    ) -> dict[str, Any]:
        return get_pipeline_health(config, context).model_dump(mode="json")

    return app


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
