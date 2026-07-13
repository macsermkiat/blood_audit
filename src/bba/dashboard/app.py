"""FastAPI app factory + handler functions for the reviewer dashboard.

Five views, all rendered via FastAPI + HTMX + Tailwind (Jinja2 templates
under ``src/bba/dashboard/templates/``):

* ``GET /queue`` — NEEDS_REVIEW queue (sortable via ``sort_key`` /
  ``sort_direction`` query params; HTMX swaps the table in-place).
* ``GET /case/{audit_id}`` — case-detail (un-redacted gated on
  care-team-of-record; ``phi_access_log`` row written on every
  un-redacted surfacing, care-team or break-glass).
* ``POST /case/{audit_id}/break-glass`` — break-glass un-redacted access
  capture; writes a ``phi_access_log`` row carrying the justification
  BEFORE surfacing un-redacted text.
* ``GET /scorecard/ward/{ward_id}`` — per-ward scorecard (attribution
  via :data:`WardAttributionResolver`).
* ``GET /scorecard/physician/{physician_id}`` — per-physician own-view
  scorecard (route guard: requesting physician = subject physician).
* ``GET /pipeline-health`` — pipeline-health dashboard (NEEDS_REVIEW
  rate, verifier pass rate, Sonnet→Opus escalation rate, average LLM
  latency).

The handler functions are exposed at module-level for direct unit-test
invocation; they raise typed :class:`DashboardError` subclasses. The
route layer wraps each handler and translates :class:`DashboardError`
into HTTP responses.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Form, HTTPException
from fastapi import Request as FastApiRequest
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from bba.audit_store import AuditRow, Classification, LlmCall, SnapshotView
from bba.dashboard.exceptions import (
    AuditNotFoundError,
    DashboardError,
    MissingJustificationError,
    PhysicianAccessDeniedError,
    UnredactedSourceUnavailableError,
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
from bba.feature_flags import RETURNS_LEDGER_ENABLED
from bba.review_actions import PhiAccessInput


# ---------------------------------------------------------------------------
# Attribution defaults — used only when the caller has not configured a
# resolver. Codex round 1 flagged that hardcoded defaults silently masked
# attribution bugs; production deployments MUST inject the real registries.
# The defaults below are clearly named so a misconfigured deploy shows up
# in the rendered view as ``unattributed-*`` rather than a plausible-looking
# placeholder.
# ---------------------------------------------------------------------------


_UNATTRIBUTED_WARD_ID = "unattributed-ward"
_UNATTRIBUTED_PHYSICIAN_ID = "unattributed-physician"
_DEFAULT_WARD_NAME = "Default Ward"
_DEFAULT_PHYSICIAN_NAME = "Default Physician"
_DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


# ---------------------------------------------------------------------------
# Resolver lookups (with safe fallbacks for un-configured deploys).
# ---------------------------------------------------------------------------


def _ward_id_for_row(config: DashboardConfig, row: AuditRow) -> str:
    """Resolve the ward attribution for an audit row.

    Delegates to ``config.ward_attribution_resolver`` when configured;
    otherwise returns :data:`_UNATTRIBUTED_WARD_ID` so a misconfigured
    deploy is visibly broken rather than silently misattributing.
    """
    if config.ward_attribution_resolver is None:
        return _UNATTRIBUTED_WARD_ID
    return config.ward_attribution_resolver(row)


def _physician_id_for_row(config: DashboardConfig, row: AuditRow) -> str:
    """Resolve the physician attribution for an audit row.

    Same pattern as :func:`_ward_id_for_row`. The ``unattributed-physician``
    sentinel cannot collide with any real :data:`SafeId` from production
    (real ids never carry the literal substring ``unattributed-``).
    """
    if config.physician_attribution_resolver is None:
        return _UNATTRIBUTED_PHYSICIAN_ID
    return config.physician_attribution_resolver(row)


def _resolve_unredacted_phi(config: DashboardConfig, row: AuditRow) -> tuple[str, str]:
    """Resolve un-redacted ``(hn, an)`` for an audit row, or raise.

    The dashboard refuses to surface un-redacted PHI without an explicit
    resolver — fabricated placeholder values would silently violate the
    PRD §17 acceptance criterion. A deploy that forgets the resolver
    fails LOUD via :class:`UnredactedSourceUnavailableError`.
    """
    if config.unredacted_phi_resolver is None:
        raise UnredactedSourceUnavailableError(
            f"un-redacted PHI requested for audit_id {row.audit_id!r} but "
            f"no UnredactedPhiResolver is configured; the dashboard refuses "
            f"to fabricate un-redacted values"
        )
    return config.unredacted_phi_resolver(row.hn_hash, row.an_hash)


# ---------------------------------------------------------------------------
# Pure projections — AuditRow → view-layer DTO.
# ---------------------------------------------------------------------------


def _audit_row_to_queue_item(config: DashboardConfig, row: AuditRow) -> QueueItem:
    return QueueItem(
        audit_id=row.audit_id,
        run_id=row.run_id,
        order_datetime=row.order_datetime,
        hb_value=row.hb_value,
        hb_freshness=row.hb_freshness,
        confidence=row.confidence,
        final_classification=row.final_classification,
        review_reason=row.review_reason,
        ward_id=_ward_id_for_row(config, row),
        hn_hash=row.hn_hash,
        an_hash=row.an_hash,
    )


def _audit_row_to_case_detail(
    config: DashboardConfig, row: AuditRow, *, unredacted: bool
) -> CaseDetail:
    """Project an AuditRow into a CaseDetail.

    When ``unredacted=True``, the un-redacted ``(hn, an)`` is resolved via
    :func:`_resolve_unredacted_phi`. The dashboard never fabricates
    placeholder PHI — a deploy without a resolver fails loud.
    """
    if unredacted:
        raw_hn, raw_an = _resolve_unredacted_phi(config, row)
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
        negative_evidence_json=tuple(dict(item) for item in row.negative_evidence_json),
        needs_human_review=row.needs_human_review,
        review_reason=row.review_reason,
        hb_value=row.hb_value,
        hb_freshness=row.hb_freshness,
        order_datetime=row.order_datetime,
        unredacted=unredacted,
        raw_hn=raw_hn,
        raw_an=raw_an,
    )


def _read_snapshot_rows(config: DashboardConfig) -> tuple[AuditRow, ...]:
    """Read RBC audit rows via the daily-rotated DuckDB snapshot view.

    Keyed by ``datetime.now(UTC).date()`` — same-day reads share the
    materialized snapshot, mid-batch writes are isolated. The first read
    of a given day materializes the view; subsequent reads return the
    same frozen result set.

    Only ``component == "red_cell"`` rows are returned. Platelet rows use
    different clinical thresholds (platelet count vs Hb) and are excluded
    from every RBC dashboard view (queue, ward/physician scorecard, pipeline
    health). This is the single choke point: filtering here keeps every
    downstream handler free of per-component branching.
    """
    today = datetime.now(UTC).date()
    view = SnapshotView.open(config.audit_store, today)
    return tuple(r for r in view.read_audit_results() if r.component == "red_cell")


def _find_audit_row(config: DashboardConfig, audit_id: str) -> AuditRow:
    for row in _read_snapshot_rows(config):
        if row.audit_id == audit_id:
            return row
    raise AuditNotFoundError(f"audit_id {audit_id!r} not found in today's snapshot")


def _count_classification(
    rows: tuple[AuditRow, ...], classification: Classification
) -> int:
    return sum(1 for r in rows if r.final_classification == classification)


def _aggregate_classifications(
    rows: tuple[AuditRow, ...],
) -> dict[str, int | float]:
    """Compute the count + average-confidence aggregations for a row set.

    Used by both ward and physician scorecards (same aggregation shape).
    An empty input returns zeros for every count and ``0.0`` for the
    confidence average (rather than raising) — operators legitimately
    check wards / physicians with no audited orders today.
    """
    scorable_rows = tuple(
        r for r in rows if r.final_classification != "RETURNED_NOT_TRANSFUSED"
    )
    total = len(scorable_rows)
    avg_confidence = (
        sum(r.confidence for r in scorable_rows) / total if total > 0 else 0.0
    )
    return {
        "total_orders": total,
        "appropriate_count": _count_classification(rows, "APPROPRIATE"),
        "inappropriate_count": _count_classification(rows, "INAPPROPRIATE"),
        "needs_review_count": _count_classification(rows, "NEEDS_REVIEW"),
        "insufficient_evidence_count": _count_classification(
            rows, "INSUFFICIENT_EVIDENCE"
        ),
        "returned_not_transfused_count": _count_classification(
            rows, "RETURNED_NOT_TRANSFUSED"
        ),
        "average_confidence": avg_confidence,
    }


def _average_llm_latency_ms(calls: tuple[LlmCall, ...]) -> float:
    if not calls:
        return 0.0
    return sum(c.latency_ms for c in calls) / len(calls)


def _phi_access_input(
    context: RouteContext, row: AuditRow, *, justification: str | None
) -> PhiAccessInput:
    """Build a ``PhiAccessInput`` for the active reviewer + audit row.

    ``justification=None`` is the care-team path (no break-glass override);
    a non-None justification is the break-glass path. Both are PHI access
    events and BOTH must land in ``phi_access_log`` (PRD §17).
    """
    return PhiAccessInput(
        reviewer_id=context.reviewer.reviewer_id,
        audit_id=row.audit_id,
        hn_hash=row.hn_hash,
        an_hash=row.an_hash,
        break_glass_justification=justification,
    )


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
    """Return the sortable NEEDS_REVIEW queue.

    Reads via :class:`bba.audit_store.SnapshotView` (daily-rotated) so an
    in-flight batch write does not produce mid-query inconsistency.
    Filters to ``final_classification == 'NEEDS_REVIEW'``; the queue is a
    triage surface, not a general browser.
    """
    del context  # queue scope is global (PRD §17 — no per-reviewer slice)
    rows = _read_snapshot_rows(config)
    review_rows = tuple(r for r in rows if r.final_classification == "NEEDS_REVIEW")
    items = [_audit_row_to_queue_item(config, r) for r in review_rows]
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
    ``care_team_memberships`` the response is un-redacted AND a
    ``phi_access_log`` row is written (PRD §17 "every un-redacted access
    writes to phi_access_log" — that applies to the care-team path too,
    not just break-glass). The write happens BEFORE the un-redacted text
    is surfaced via :meth:`ReviewActionsStore.access_phi`.

    Outsiders see the redacted projection (``unredacted=False``,
    ``raw_hn``/``raw_an=None``). The break-glass flow
    (:func:`record_break_glass_access`) is the only way to obtain
    un-redacted access without care-team membership.
    """
    row = _find_audit_row(config, audit_id)
    if audit_id in context.reviewer.care_team_memberships:
        access = _phi_access_input(context, row, justification=None)
        with config.review_actions_store.access_phi(access):
            return _audit_row_to_case_detail(config, row, unredacted=True)
    return _audit_row_to_case_detail(config, row, unredacted=False)


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
    ``BreakGlassRequest.model_construct`` to simulate a malformed
    request that reached the handler.

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
    access = _phi_access_input(context, row, justification=request.justification)
    with config.review_actions_store.access_phi(access):
        return _audit_row_to_case_detail(config, row, unredacted=True)


def get_ward_scorecard(
    config: DashboardConfig,
    context: RouteContext,
    ward_id: str,
) -> WardScorecard:
    """Return the per-ward aggregation scorecard for ``ward_id``.

    Attribution via :data:`WardAttributionResolver`. An empty result
    (ward not represented in today's snapshot) returns a scorecard with
    zero counts — the operator may legitimately check a ward that has no
    audited orders today.
    """
    del context
    rows = _read_snapshot_rows(config)
    ward_rows = tuple(r for r in rows if _ward_id_for_row(config, r) == ward_id)
    aggs = _aggregate_classifications(ward_rows)
    return WardScorecard(
        ward_id=ward_id,
        ward_name=_DEFAULT_WARD_NAME,
        total_orders=int(aggs["total_orders"]),
        appropriate_count=int(aggs["appropriate_count"]),
        inappropriate_count=int(aggs["inappropriate_count"]),
        needs_review_count=int(aggs["needs_review_count"]),
        insufficient_evidence_count=int(aggs["insufficient_evidence_count"]),
        returned_not_transfused_count=int(aggs["returned_not_transfused_count"]),
        average_confidence=float(aggs["average_confidence"]),
    )


def get_physician_scorecard(
    config: DashboardConfig,
    context: RouteContext,
    physician_id: str,
) -> PhysicianScorecard:
    """Return the per-physician own-view scorecard.

    Route guard: strict own-view — a reviewer may ONLY view the
    scorecard whose ``physician_id`` matches their own
    ``reviewer.physician_id``. Mismatched ids (including ``None``)
    raise :class:`PhysicianAccessDeniedError` before any data
    retrieval.

    PRD §17 names this an "own-view-only" surface; an ordinary
    reviewer (``role='reviewer'``) without an attached physician
    identity must not be able to enumerate per-physician metrics
    through this route. Cross-physician views (admin / senior
    reviewer aggregation) belong on a separate, explicitly
    privileged route, not behind a role check on this endpoint
    (codex round 4 P1: role-gated guard let non-physician roles slip
    through).
    """
    if context.reviewer.physician_id != physician_id:
        raise PhysicianAccessDeniedError(
            f"reviewer {context.reviewer.reviewer_id!r} "
            f"(role={context.reviewer.role!r}, "
            f"physician_id={context.reviewer.physician_id!r}) cannot "
            f"view physician {physician_id!r}'s scorecard; the "
            f"per-physician view is strict own-view"
        )
    rows = _read_snapshot_rows(config)
    phys_rows = tuple(
        r for r in rows if _physician_id_for_row(config, r) == physician_id
    )
    aggs = _aggregate_classifications(phys_rows)
    return PhysicianScorecard(
        physician_id=physician_id,
        physician_name=_DEFAULT_PHYSICIAN_NAME,
        ward_id=_UNATTRIBUTED_WARD_ID,
        total_orders=int(aggs["total_orders"]),
        appropriate_count=int(aggs["appropriate_count"]),
        inappropriate_count=int(aggs["inappropriate_count"]),
        needs_review_count=int(aggs["needs_review_count"]),
        insufficient_evidence_count=int(aggs["insufficient_evidence_count"]),
        returned_not_transfused_count=int(aggs["returned_not_transfused_count"]),
        average_confidence=float(aggs["average_confidence"]),
    )


def get_pipeline_health(
    config: DashboardConfig,
    context: RouteContext,
) -> PipelineHealth:
    """Return pipeline-health aggregations.

    All metrics are computed on the SAME snapshot frame: rates come
    from the snapshot's audit rows; the latency aggregation is
    restricted to ``llm_calls`` whose ``audit_id`` appears in that
    snapshot. The audit_store snapshot is audit-results-only, so the
    call records are read live via :meth:`AuditStore.read_llm_calls`
    and then filtered to the snapshot's audit ids.

    Filtering matters during in-flight batch writes: ``llm_calls`` is
    persisted BEFORE its matching ``audit_results`` row commits (PRD
    §10 transactional ordering), so a live read of ``llm_calls`` can
    include calls whose audit row is not yet in the snapshot. Without
    the filter, ``average_latency_ms`` would include those calls
    while ``total_audits`` did not — codex round 4 P2.
    """
    del context
    rows = _read_snapshot_rows(config)
    total = len(rows)
    snapshot_audit_ids = {r.audit_id for r in rows}
    all_calls = config.audit_store.read_llm_calls()
    snapshot_calls = tuple(c for c in all_calls if c.audit_id in snapshot_audit_ids)
    avg_latency = _average_llm_latency_ms(snapshot_calls)
    if total == 0:
        return PipelineHealth(
            total_audits=0,
            needs_review_rate=0.0,
            verifier_pass_rate=0.0,
            escalation_rate=0.0,
            average_latency_ms=avg_latency,
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
        average_latency_ms=avg_latency,
        snapshot_as_of=datetime.now(UTC),
    )


def resolve_reviewer(
    config: DashboardConfig,
    reviewer_id: str,
) -> Reviewer:
    """Resolve a reviewer identity from a session id.

    Production wires this to a real auth backend via FastAPI's
    ``app.dependency_overrides``. The default implementation falls
    through to :attr:`DashboardConfig.default_reviewer` (set by tests /
    dev configs); the ``reviewer_id`` argument is currently unused
    because the default-reviewer path is a fixed identity.
    """
    del reviewer_id
    if config.default_reviewer is None:
        raise DashboardError(
            "reviewer could not be resolved: no default reviewer "
            "configured and no auth dependency overrode this function"
        )
    return config.default_reviewer


def get_route_context(request: FastApiRequest) -> RouteContext:
    """Module-level FastAPI dependency that resolves the active
    :class:`RouteContext` for a request.

    Stored at module level (not a closure inside :func:`create_app`)
    so production can replace it via
    ``app.dependency_overrides[get_route_context] = my_auth_dep``.
    The dependency reads :class:`DashboardConfig` from
    ``request.app.state.dashboard_config`` and falls back to
    :attr:`DashboardConfig.default_reviewer` — production overrides
    this dependency entirely; tests can leave the default in place.

    Raises HTTP 401 when neither path resolves a reviewer.
    """
    config: DashboardConfig | None = getattr(
        request.app.state, "dashboard_config", None
    )
    if config is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "request.app.state.dashboard_config is unset; the app was "
                "not built via bba.dashboard.create_app"
            ),
        )
    if config.default_reviewer is None:
        raise HTTPException(
            status_code=401,
            detail=(
                "reviewer not resolved; no default reviewer configured "
                "and no auth dependency overrode this route"
            ),
        )
    return RouteContext(reviewer=config.default_reviewer)


# ---------------------------------------------------------------------------
# FastAPI app factory.
# ---------------------------------------------------------------------------


def create_app(config: DashboardConfig) -> FastAPI:
    """Construct and return the FastAPI application for the reviewer UI.

    Routes render Jinja2 templates under
    ``src/bba/dashboard/templates/`` (Tailwind via CDN, HTMX via CDN —
    sufficient for the Phase 1 acceptance criteria; production builds
    compile Tailwind into a static stylesheet). Each route wraps the
    corresponding module-level handler and translates
    :class:`DashboardError` subclasses into HTTP responses:

    * :class:`AuditNotFoundError` → 404
    * :class:`PhysicianAccessDeniedError` → 403
    * :class:`MissingJustificationError` → 400
    * :class:`UnredactedSourceUnavailableError` → 503

    Reviewer identity is resolved via :func:`get_route_context` (a
    module-level dependency so production can override it through
    ``app.dependency_overrides[get_route_context] = my_auth_dep``).
    The default returns ``config.default_reviewer`` (for dev / smoke
    tests) or raises 401 if no default is configured.
    """
    app = FastAPI(title="bba.dashboard", version="0.1.0")
    # Stash the config on app.state so the module-level
    # ``get_route_context`` dependency can read it. Production
    # replaces the dependency entirely via
    # ``app.dependency_overrides[get_route_context] = ...``; the
    # default uses ``config.default_reviewer`` for dev / smoke runs.
    app.state.dashboard_config = config
    template_dir = config.template_dir or _DEFAULT_TEMPLATE_DIR
    templates = Jinja2Templates(directory=str(template_dir))

    @app.get("/queue", response_class=HTMLResponse)
    def _queue_route(
        request: FastApiRequest,
        context: RouteContext = Depends(get_route_context),
        sort_key: QueueSortKey = "order_datetime",
        sort_direction: SortDirection = "desc",
    ) -> HTMLResponse:
        items = list_queue(
            config, context, sort_key=sort_key, sort_direction=sort_direction
        )
        context_dict = {
            "items": items,
            "sort_key": sort_key,
            "sort_direction": sort_direction,
        }
        # HTMX requests carry the ``HX-Request: true`` header. Render
        # only the table fragment so the swap target (#queue-table)
        # receives table-shaped HTML, not a full document. Plain
        # browser navigations get the full page.
        if request.headers.get("HX-Request") == "true":
            return templates.TemplateResponse(
                request, "_queue_table.html", context_dict
            )
        return templates.TemplateResponse(request, "queue.html", context_dict)

    @app.get("/case/{audit_id}", response_class=HTMLResponse)
    def _case_detail_route(
        request: FastApiRequest,
        audit_id: str,
        context: RouteContext = Depends(get_route_context),
    ) -> HTMLResponse:
        try:
            detail = get_case_detail(config, context, audit_id)
        except AuditNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except UnredactedSourceUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return templates.TemplateResponse(
            request, "case_detail.html", {"detail": detail}
        )

    @app.post("/case/{audit_id}/break-glass", response_class=HTMLResponse)
    def _break_glass_route(
        request: FastApiRequest,
        audit_id: str,
        justification: str = Form(...),
        context: RouteContext = Depends(get_route_context),
    ) -> HTMLResponse:
        # Accept the justification as form data (what an HTMX-driven
        # form sends); construct the validated request object inside
        # the handler so a malformed empty value still surfaces via
        # the typed MissingJustificationError path.
        try:
            break_glass_request = BreakGlassRequest(justification=justification)
        except Exception as exc:  # pydantic ValidationError → 400
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            detail = record_break_glass_access(
                config, context, audit_id, break_glass_request
            )
        except MissingJustificationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AuditNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except UnredactedSourceUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return templates.TemplateResponse(
            request, "case_detail.html", {"detail": detail}
        )

    @app.get("/scorecard/ward/{ward_id}", response_class=HTMLResponse)
    def _ward_scorecard_route(
        request: FastApiRequest,
        ward_id: str,
        context: RouteContext = Depends(get_route_context),
    ) -> HTMLResponse:
        scorecard = get_ward_scorecard(config, context, ward_id)
        return templates.TemplateResponse(
            request,
            "ward_scorecard.html",
            {
                "scorecard": scorecard,
                "returns_ledger_enabled": RETURNS_LEDGER_ENABLED,
            },
        )

    @app.get("/scorecard/physician/{physician_id}", response_class=HTMLResponse)
    def _physician_scorecard_route(
        request: FastApiRequest,
        physician_id: str,
        context: RouteContext = Depends(get_route_context),
    ) -> HTMLResponse:
        try:
            scorecard = get_physician_scorecard(config, context, physician_id)
        except PhysicianAccessDeniedError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return templates.TemplateResponse(
            request,
            "physician_scorecard.html",
            {
                "scorecard": scorecard,
                "returns_ledger_enabled": RETURNS_LEDGER_ENABLED,
            },
        )

    @app.get("/pipeline-health", response_class=HTMLResponse)
    def _pipeline_health_route(
        request: FastApiRequest,
        context: RouteContext = Depends(get_route_context),
    ) -> HTMLResponse:
        health = get_pipeline_health(config, context)
        return templates.TemplateResponse(
            request, "pipeline_health.html", {"health": health}
        )

    return app


__all__ = (
    "create_app",
    "get_case_detail",
    "get_physician_scorecard",
    "get_pipeline_health",
    "get_route_context",
    "get_ward_scorecard",
    "list_queue",
    "record_break_glass_access",
    "resolve_reviewer",
)
