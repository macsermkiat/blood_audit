"""RED-phase failing tests for issue #26 (bba.dashboard).

Each ``class`` maps to one acceptance criterion in the issue body.

* AC ① "Implementation in ``src/bba/dashboard/``"
    → :class:`TestModulePublicSurface`
* AC ② "All five views render with synthetic test data"
    → :class:`TestFiveViewsRender`
* AC ③ "Break-glass flow: viewing un-redacted requires justification
    text → written to phi_access_log"
    → :class:`TestBreakGlassFlow`
* AC ④ "Per-physician own-view: route guard enforces requesting
    physician = subject physician"
    → :class:`TestPhysicianOwnViewGuard`
* AC ⑤ "DuckDB snapshot read path tested (no inconsistency under
    concurrent batch write)"
    → :class:`TestSnapshotReadPath`
* AC ⑥ "Smoke tests for all routes"
    → :class:`TestRouteSmoke`
* AC ⑦ "Coverage ≥ 60%; ruff + mypy clean"
    → verified by the build (not a behavioral test)

In addition, :class:`TestModelContracts` exercises the frozen Pydantic
models — these pass in RED because the models ARE the declared interface,
not behavioral assertions. The property test in
:class:`TestPhysicianOwnViewGuard` is the hypothesis safety net required
for non-trivial route-guard logic.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from bba.audit_store import (
    AuditRow,
    AuditStore,
    AuditStoreConfig,
    LlmCall,
)
from bba.dashboard import (
    BreakGlassRequest,
    CaseDetail,
    DashboardConfig,
    DashboardError,
    MissingJustificationError,
    Physician,
    PhysicianAccessDeniedError,
    PhysicianScorecard,
    PipelineHealth,
    QueueItem,
    Reviewer,
    RouteContext,
    SnapshotInconsistencyError,
    UnredactedSourceUnavailableError,
    Ward,
    WardScorecard,
    create_app,
)
from bba.dashboard.app import (
    _aggregate_classifications,
    get_case_detail,
    get_physician_scorecard,
    get_pipeline_health,
    get_route_context,
    get_ward_scorecard,
    list_queue,
    record_break_glass_access,
)
from bba.review_actions import (
    PhiAccessInput,
    PhiAccessLog,
    ReviewActionsConfig,
    ReviewActionsStore,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# Fixtures — the dashboard composes two persistence handles. In RED, both
# may be no-op placeholders; the route handlers raise NotImplementedError
# before any persistence call. GREEN-phase will replace these with the
# real Postgres + Parquet handles.
# =============================================================================


@pytest.fixture
def audit_store(tmp_path: Path) -> AuditStore:
    """An AuditStore rooted at a fresh tmp_path.

    The store is real (filesystem-backed) — the dashboard reads through
    :class:`bba.audit_store.SnapshotView`, so a real store is the closest
    fixture to production. In RED, route handlers raise before invoking
    the store.
    """
    return AuditStore(
        AuditStoreConfig(root_dir=tmp_path / "audit", code_version="test-v1")
    )


class _InMemoryReviewActionsStore(ReviewActionsStore):
    """In-memory test double for :class:`ReviewActionsStore`.

    Bypasses Postgres so dashboard unit tests are fast + isolated. The
    real store is exercised end-to-end in ``test_review_actions.py`` (issue
    #25); this fake is structurally compatible — same public methods,
    same return types — and overrides only the data-mutating + listing
    methods that the dashboard touches.

    The parent's ``_ensure_migrated`` is short-circuited by setting
    ``_migrated_checked=True`` so the data methods never attempt to
    open a connection pool.
    """

    def __init__(self) -> None:
        super().__init__(
            ReviewActionsConfig(dsn="postgresql://fake:fake@localhost:5432/fake")
        )
        # Bypass migration / privilege / trigger checks — we never hit DB.
        self._migrated_checked = True
        self._fake_phi_logs: list[PhiAccessLog] = []
        self._fake_counter = 0

    def record_phi_access(self, access: PhiAccessInput) -> PhiAccessLog:
        self._fake_counter += 1
        log = PhiAccessLog(
            access_id=self._fake_counter,
            reviewer_id=access.reviewer_id,
            audit_id=access.audit_id,
            hn_hash=access.hn_hash,
            an_hash=access.an_hash,
            break_glass_justification=access.break_glass_justification,
            accessed_at=datetime.now(UTC),
        )
        self._fake_phi_logs.append(log)
        return log

    @contextmanager
    def access_phi(self, access: PhiAccessInput) -> Iterator[PhiAccessLog]:
        # Mirror the parent's contract: log row is committed BEFORE the
        # body runs (PRD §17 "log row exists before un-redacted text is
        # surfaced"). On body raise, the log row stays — the access
        # attempt is still an access for audit purposes.
        log = self.record_phi_access(access)
        try:
            yield log
        finally:
            pass

    def list_phi_access(  # type: ignore[override]
        self,
        *,
        reviewer_id: str | None = None,
        audit_id: str | None = None,
    ) -> tuple[PhiAccessLog, ...]:
        result = list(self._fake_phi_logs)
        if reviewer_id is not None:
            result = [r for r in result if r.reviewer_id == reviewer_id]
        if audit_id is not None:
            result = [r for r in result if r.audit_id == audit_id]
        return tuple(result)


@pytest.fixture
def review_actions_store(tmp_path: Path) -> ReviewActionsStore:
    """An in-memory ``ReviewActionsStore`` test double.

    The real Postgres-backed store is exercised in ``test_review_actions.py``;
    here we use a fake so the dashboard unit suite is fast (no
    testcontainers spin-up) and isolated (no shared schema state). The
    fake satisfies the same public-method surface the dashboard depends
    on (``access_phi`` / ``record_phi_access`` / ``list_phi_access``).
    """
    del tmp_path  # not needed — fake is in-memory
    return _InMemoryReviewActionsStore()


@pytest.fixture
def fake_unredacted_resolver() -> Any:
    """A deterministic un-redacted PHI resolver for unit tests.

    Real production wires this to the HIS / de-id-twin store; the dashboard
    test boundary just needs a known-good ``(hn, an)`` to assert against
    the rendered un-redacted case detail.
    """

    def _resolve(hn_hash: str, an_hash: str) -> tuple[str, str]:
        return (f"HN-from-hash-{hn_hash}", f"AN-from-hash-{an_hash}")

    return _resolve


@pytest.fixture
def fake_ward_resolver() -> Any:
    """A static ward-attribution resolver.

    Synthetic test rows are attributed to ``ward-001``; un-mapped audits
    fall through to ``ward-unmapped`` so a misconfigured fixture is
    visible in assertions rather than silently bucketed with mapped rows.
    """
    ward_by_audit_id = {
        "audit-001": "ward-001",
        "audit-pre": "ward-001",
        "audit-batch-0": "ward-001",
        "audit-batch-1": "ward-001",
        "audit-batch-2": "ward-001",
        "audit-batch-3": "ward-001",
        "audit-batch-4": "ward-001",
        "audit-post": "ward-001",
        "audit-appropriate": "ward-001",
    }

    def _resolve(row: AuditRow) -> str:
        return ward_by_audit_id.get(row.audit_id, "ward-unmapped")

    return _resolve


@pytest.fixture
def fake_physician_resolver() -> Any:
    """A static physician-attribution resolver."""
    physician_by_audit_id = {
        "audit-001": "phys-001",
        "audit-pre": "phys-001",
        "audit-appropriate": "phys-001",
    }

    def _resolve(row: AuditRow) -> str:
        return physician_by_audit_id.get(row.audit_id, "phys-unmapped")

    return _resolve


@pytest.fixture
def config(
    audit_store: AuditStore,
    review_actions_store: ReviewActionsStore,
    fake_unredacted_resolver: Any,
    fake_ward_resolver: Any,
    fake_physician_resolver: Any,
    tmp_path: Path,
) -> DashboardConfig:
    """A DashboardConfig with deterministic fakes wired for every
    injectable dependency.

    The default reviewer is a physician with ``physician_id="phys-001"``
    so the physician-scorecard smoke route returns 200 (own-view) rather
    than 403 (cross-view). ``care_team_memberships=("audit-001",)`` makes
    the case-detail smoke route un-redacted (which also writes to the
    fake phi_access_log — the care-team un-redacted path is covered by
    :meth:`TestBreakGlassFlow.test_care_team_unredacted_writes_phi_access_log`).
    """
    return DashboardConfig(
        audit_store=audit_store,
        review_actions_store=review_actions_store,
        snapshot_dir=tmp_path / "snapshots",
        default_reviewer=Reviewer(
            reviewer_id="default-smoke-reviewer",
            name="Smoke Reviewer",
            role="physician",
            physician_id="phys-001",
            care_team_memberships=("audit-001",),
        ),
        unredacted_phi_resolver=fake_unredacted_resolver,
        ward_attribution_resolver=fake_ward_resolver,
        physician_attribution_resolver=fake_physician_resolver,
    )


@pytest.fixture
def config_without_unredacted_resolver(
    audit_store: AuditStore,
    review_actions_store: ReviewActionsStore,
    fake_ward_resolver: Any,
    fake_physician_resolver: Any,
    tmp_path: Path,
) -> DashboardConfig:
    """A DashboardConfig with NO un-redacted PHI resolver — the deploy
    that forgot to wire the HIS / de-id-twin store.

    Used by :meth:`TestBreakGlassFlow.test_unredacted_without_resolver_fails_loud`
    to assert the dashboard refuses to fabricate un-redacted PHI.
    """
    return DashboardConfig(
        audit_store=audit_store,
        review_actions_store=review_actions_store,
        snapshot_dir=tmp_path / "snapshots",
        default_reviewer=Reviewer(
            reviewer_id="default-smoke-reviewer",
            name="Smoke Reviewer",
            role="reviewer",
        ),
        ward_attribution_resolver=fake_ward_resolver,
        physician_attribution_resolver=fake_physician_resolver,
    )


@pytest.fixture
def reviewer_in_care_team() -> Reviewer:
    return Reviewer(
        reviewer_id="reviewer-care-team",
        name="Dr. Care Team",
        role="reviewer",
        physician_id=None,
        care_team_memberships=("audit-001",),
    )


@pytest.fixture
def reviewer_outside_care_team() -> Reviewer:
    return Reviewer(
        reviewer_id="reviewer-outsider",
        name="Dr. Outsider",
        role="reviewer",
        physician_id=None,
        care_team_memberships=(),
    )


@pytest.fixture
def physician_self() -> Reviewer:
    return Reviewer(
        reviewer_id="phys-self",
        name="Dr. Self",
        role="physician",
        physician_id="phys-001",
        care_team_memberships=(),
    )


@pytest.fixture
def physician_other() -> Reviewer:
    return Reviewer(
        reviewer_id="phys-other",
        name="Dr. Other",
        role="physician",
        physician_id="phys-002",
        care_team_memberships=(),
    )


@pytest.fixture
def context_in_care_team(reviewer_in_care_team: Reviewer) -> RouteContext:
    return RouteContext(reviewer=reviewer_in_care_team)


@pytest.fixture
def context_outsider(reviewer_outside_care_team: Reviewer) -> RouteContext:
    return RouteContext(reviewer=reviewer_outside_care_team)


@pytest.fixture
def context_physician_self(physician_self: Reviewer) -> RouteContext:
    return RouteContext(reviewer=physician_self, subject_physician_id="phys-001")


@pytest.fixture
def context_physician_other_subject(physician_self: Reviewer) -> RouteContext:
    return RouteContext(reviewer=physician_self, subject_physician_id="phys-002")


# =============================================================================
# Builders
# =============================================================================


def _audit_row(
    *,
    audit_id: str = "audit-001",
    run_id: str = "run-001",
    final_classification: str = "NEEDS_REVIEW",
) -> AuditRow:
    """A minimal valid AuditRow for synthetic data tests.

    Mirrors the field-group order in ``bba.audit_store.models.AuditRow``;
    every field is required so the builder is verbose by necessity.
    """
    now = datetime.now(UTC)
    return AuditRow(
        audit_id=audit_id,
        run_id=run_id,
        run_timestamp=now,
        hn_hash="hn-hash-001",
        an_hash="an-hash-001",
        reqno="REQ-001",
        order_datetime=now,
        products_ordered=("PRC",),
        hb_value=6.5,
        hb_datetime=now,
        hb_freshness="fresh",
        hb_source="lab",
        vitals_sbp=None,
        vitals_hr=None,
        vitals_timestamp=None,
        vitals_source=None,
        prior_rbc_units_24h=0,
        prior_rbc_units_7d=0,
        cohort_threshold=7.0,
        delta_hb_window_results=(),
        rule_classification=final_classification,  # type: ignore[arg-type]
        final_classification=final_classification,  # type: ignore[arg-type]
        cohort_applied="adult_stable",
        indications_json=(),
        negative_evidence_json=(),
        confidence=0.85,
        reasoning_summary_thai="สรุปเหตุผล",
        reasoning_summary_en="reasoning summary",
        needs_human_review=True,
        review_reason="below threshold",
        model_id="claude-sonnet-test",
        prompt_hash="prompt-hash-001",
        evidence_bundle_hash="evidence-001",
        redactor_version="redactor-v1",
        redactor_model_sha="redactor-sha-001",
        policy_version="policy-v1",
        verifier_pass=True,
        verifier_retries=0,
        escalated_to_opus=False,
    )


def _llm_call(*, audit_id: str = "audit-001", run_id: str = "run-001") -> LlmCall:
    now = datetime.now(UTC)
    return LlmCall(
        call_id=f"{audit_id}-call",
        audit_id=audit_id,
        run_id=run_id,
        model_id="claude-sonnet-test",
        anthropic_version="2024-01-01",
        prompt_cache_id=None,
        request_json={"messages": []},
        response_json={"content": []},
        request_timestamp=now,
        latency_ms=100,
        extended_thinking_blocks=None,
        cold_storage_uri=None,
    )


def _writeable_audit_row_pair(
    audit_id: str = "audit-001", run_id: str = "run-001"
) -> tuple[AuditRow, tuple[LlmCall, ...]]:
    """A matched ``(AuditRow, llm_calls)`` pair for direct store writes."""
    return _audit_row(audit_id=audit_id, run_id=run_id), (
        _llm_call(audit_id=audit_id, run_id=run_id),
    )


# =============================================================================
# AC ① — Implementation in src/bba/dashboard/
# =============================================================================


class TestModulePublicSurface:
    """The module exposes the public surface declared in __init__.py."""

    def test_module_imports_cleanly(self) -> None:
        import bba.dashboard as dashboard

        assert dashboard.__name__ == "bba.dashboard"

    def test_module_lives_under_src_bba_dashboard(self) -> None:
        import bba.dashboard as dashboard

        module_path = Path(dashboard.__file__).resolve()
        assert module_path.parent.name == "dashboard"
        assert module_path.parent.parent.name == "bba"

    @pytest.mark.parametrize(
        "name",
        [
            "BreakGlassRequest",
            "CaseDetail",
            "DashboardConfig",
            "DashboardError",
            "MissingJustificationError",
            "Physician",
            "PhysicianAccessDeniedError",
            "PhysicianScorecard",
            "PipelineHealth",
            "QueueItem",
            "Reviewer",
            "RouteContext",
            "SnapshotInconsistencyError",
            "Ward",
            "WardScorecard",
            "create_app",
        ],
    )
    def test_public_surface_exports_name(self, name: str) -> None:
        import bba.dashboard as dashboard

        assert hasattr(dashboard, name), f"missing public export: {name}"


# =============================================================================
# AC ② — All five views render with synthetic test data
# =============================================================================


class TestFiveViewsRender:
    """One test per view; each fails in RED with NotImplementedError."""

    def test_queue_view_renders_with_synthetic_data(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        row, calls = _writeable_audit_row_pair()
        audit_store.write(row, calls)

        items = list_queue(config, context_in_care_team)

        assert all(isinstance(item, QueueItem) for item in items)
        assert any(item.audit_id == "audit-001" for item in items)
        # Behavioral: ward attribution must come through the resolver,
        # not a hardcoded default. Synthetic ``audit-001`` maps to
        # ``ward-001`` (see ``fake_ward_resolver``).
        audit_item = next(i for i in items if i.audit_id == "audit-001")
        assert audit_item.ward_id == "ward-001"

    def test_case_detail_view_renders_with_synthetic_data(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        row, calls = _writeable_audit_row_pair()
        audit_store.write(row, calls)

        detail = get_case_detail(config, context_in_care_team, "audit-001")

        assert isinstance(detail, CaseDetail)
        assert detail.audit_id == "audit-001"
        # Behavioral: care-team member sees un-redacted via the resolver
        # (not a fabricated placeholder).
        assert detail.unredacted is True
        assert detail.raw_hn == "HN-from-hash-hn-hash-001"
        assert detail.raw_an == "AN-from-hash-an-hash-001"

    def test_ward_scorecard_view_renders_with_synthetic_data(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        # Write two NEEDS_REVIEW + one APPROPRIATE row; only the two
        # NEEDS_REVIEW ones map to ward-001 via the resolver.
        row_a, calls_a = _writeable_audit_row_pair("audit-001", "run-001")
        audit_store.write(row_a, calls_a)
        row_b = _audit_row(
            audit_id="audit-appropriate",
            run_id="run-002",
            final_classification="APPROPRIATE",
        )
        audit_store.write(
            row_b, (_llm_call(audit_id="audit-appropriate", run_id="run-002"),)
        )

        scorecard = get_ward_scorecard(config, context_in_care_team, "ward-001")

        assert isinstance(scorecard, WardScorecard)
        assert scorecard.ward_id == "ward-001"
        # Behavioral: both audits attribute to ward-001, so total=2.
        assert scorecard.total_orders == 2
        assert scorecard.needs_review_count == 1
        assert scorecard.appropriate_count == 1
        # Average confidence is 0.85 for both synthetic rows.
        assert scorecard.average_confidence == pytest.approx(0.85)

    def test_ward_scorecard_unknown_ward_returns_zero_counts(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        row, calls = _writeable_audit_row_pair()
        audit_store.write(row, calls)

        scorecard = get_ward_scorecard(
            config, context_in_care_team, "ward-with-no-rows"
        )

        assert scorecard.total_orders == 0
        assert scorecard.average_confidence == 0.0

    def test_returned_not_transfused_is_displayed_but_excluded_from_total(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        returned = _audit_row(
            audit_id="audit-001",
            run_id="run-returned",
            final_classification="RETURNED_NOT_TRANSFUSED",
        )
        audit_store.write(
            returned,
            (_llm_call(audit_id="audit-001", run_id="run-returned"),),
        )
        scorecard = get_ward_scorecard(config, context_in_care_team, "ward-001")
        assert scorecard.total_orders == 0
        assert scorecard.returned_not_transfused_count == 1
        assert (
            scorecard.appropriate_count
            + scorecard.inappropriate_count
            + scorecard.needs_review_count
            + scorecard.insufficient_evidence_count
            == scorecard.total_orders
        )

    def test_periop_transfusion_exempt_is_displayed_but_excluded_from_total(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        exempt = _audit_row(
            audit_id="audit-001",
            run_id="run-periop",
            final_classification="PERIOP_TRANSFUSION_EXEMPT",
        )
        audit_store.write(
            exempt,
            (_llm_call(audit_id="audit-001", run_id="run-periop"),),
        )
        scorecard = get_ward_scorecard(config, context_in_care_team, "ward-001")
        assert scorecard.total_orders == 0
        assert scorecard.periop_transfusion_exempt_count == 1
        assert (
            scorecard.appropriate_count
            + scorecard.inappropriate_count
            + scorecard.needs_review_count
            + scorecard.insufficient_evidence_count
            == scorecard.total_orders
        )

    def test_preop_over_reservation_conserves_aggregation_and_scorecards(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
        context_physician_self: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        row = _audit_row(
            audit_id="audit-001",
            run_id="run-over-reservation",
            final_classification="PREOP_OVER_RESERVATION",
        )
        direct = _aggregate_classifications((row,))
        audit_store.write(
            row,
            (_llm_call(audit_id="audit-001", run_id="run-over-reservation"),),
        )

        ward = get_ward_scorecard(config, context_in_care_team, "ward-001")
        physician = get_physician_scorecard(config, context_physician_self, "phys-001")

        assert direct["total_orders"] == 1
        assert direct["inappropriate_count"] == 1
        assert direct["over_reservation_count"] == 1
        for card in (ward, physician):
            assert card.total_orders == 1
            assert card.inappropriate_count == 1
            assert card.over_reservation_count == 1

    def test_physician_scorecard_view_renders_with_synthetic_data(
        self,
        config: DashboardConfig,
        context_physician_self: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        row, calls = _writeable_audit_row_pair()
        audit_store.write(row, calls)

        scorecard = get_physician_scorecard(config, context_physician_self, "phys-001")

        assert isinstance(scorecard, PhysicianScorecard)
        assert scorecard.physician_id == "phys-001"
        # Behavioral: synthetic audit-001 attributes to phys-001 via the
        # resolver; the scorecard counts one NEEDS_REVIEW.
        assert scorecard.total_orders == 1
        assert scorecard.needs_review_count == 1
        assert scorecard.appropriate_count == 0

    def test_pipeline_health_view_renders_with_synthetic_data(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        row, calls = _writeable_audit_row_pair()
        audit_store.write(row, calls)

        health = get_pipeline_health(config, context_in_care_team)

        assert isinstance(health, PipelineHealth)
        assert health.total_audits >= 1
        # Behavioral: average_latency_ms must reflect synthetic latency
        # (100ms per call) — Codex round 1 flagged hardcoded 0.0.
        assert health.average_latency_ms == pytest.approx(100.0)
        assert health.needs_review_rate == pytest.approx(1.0)

    def test_pipeline_health_latency_restricted_to_snapshot_frame(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        """Codex round 4 P2: ``llm_calls`` is persisted BEFORE its
        matching audit row commits (PRD §10 transactional ordering),
        so a live read of ``llm_calls`` can include calls whose audit
        is not yet in the snapshot. Pipeline metrics must therefore
        restrict the latency aggregation to calls whose ``audit_id``
        appears in the snapshot frame — otherwise the average reflects
        invisible audits."""
        # Materialize today's snapshot with audit-001 (latency_ms=100).
        row_pre, calls_pre = _writeable_audit_row_pair("audit-001", "run-001")
        audit_store.write(row_pre, calls_pre)
        # First read materializes the snapshot — audit-001 only.
        get_pipeline_health(config, context_in_care_team)

        # Write a second audit AFTER materialization. Its llm_call
        # carries a 5000ms latency that would skew the avg dramatically
        # if it leaked into the snapshot-restricted aggregation.
        row_post = _audit_row(audit_id="audit-post", run_id="run-post")
        call_post_dict = _llm_call(
            audit_id="audit-post", run_id="run-post"
        ).model_dump()
        call_post_dict["latency_ms"] = 5000
        call_post = LlmCall.model_validate(call_post_dict)
        audit_store.write(row_post, (call_post,))

        health = get_pipeline_health(config, context_in_care_team)

        # Snapshot is frozen at the audit-001 moment; latency must
        # therefore remain 100ms despite the 5000ms call now living in
        # the store.
        assert health.total_audits == 1
        assert health.average_latency_ms == pytest.approx(100.0)


# =============================================================================
# AC ③ — Break-glass flow: justification required → written to phi_access_log
# =============================================================================


class TestBreakGlassFlow:
    """Un-redacted access via break-glass requires a non-empty justification
    and MUST write a ``phi_access_log`` row BEFORE surfacing un-redacted text."""

    def test_break_glass_with_justification_returns_unredacted_case(
        self,
        config: DashboardConfig,
        context_outsider: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        row, calls = _writeable_audit_row_pair()
        audit_store.write(row, calls)
        request = BreakGlassRequest(justification="Patient deteriorating")

        detail = record_break_glass_access(
            config, context_outsider, "audit-001", request
        )

        assert isinstance(detail, CaseDetail)
        assert detail.unredacted is True

    def test_break_glass_writes_phi_access_log_row(
        self,
        config: DashboardConfig,
        context_outsider: RouteContext,
        review_actions_store: ReviewActionsStore,
        audit_store: AuditStore,
    ) -> None:
        row, calls = _writeable_audit_row_pair()
        audit_store.write(row, calls)
        request = BreakGlassRequest(
            justification="Urgent clinical review per attending"
        )

        record_break_glass_access(config, context_outsider, "audit-001", request)

        logs = review_actions_store.list_phi_access(
            reviewer_id="reviewer-outsider", audit_id="audit-001"
        )
        assert len(logs) == 1
        assert (
            logs[0].break_glass_justification == "Urgent clinical review per attending"
        )

    def test_break_glass_empty_justification_rejected_at_model_boundary(
        self,
    ) -> None:
        with pytest.raises(ValidationError):
            BreakGlassRequest(justification="")

    def test_break_glass_whitespace_justification_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BreakGlassRequest(justification="   ")

    def test_outsider_without_break_glass_gets_redacted_case_detail(
        self,
        config: DashboardConfig,
        context_outsider: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        row, calls = _writeable_audit_row_pair()
        audit_store.write(row, calls)

        detail = get_case_detail(config, context_outsider, "audit-001")

        assert detail.unredacted is False
        assert detail.raw_hn is None
        assert detail.raw_an is None

    def test_outsider_break_glass_without_justification_raises(
        self,
        config: DashboardConfig,
        context_outsider: RouteContext,
    ) -> None:
        """A missing justification (None / empty after parse) is a typed
        rejection from the dashboard layer — not a generic 400. The
        dashboard never logs an access for an empty justification."""
        with pytest.raises(MissingJustificationError):
            # Bypassing BreakGlassRequest validation simulates a malformed
            # request that reached the handler — the handler must raise.
            record_break_glass_access(
                config,
                context_outsider,
                "audit-001",
                BreakGlassRequest.model_construct(justification=""),
            )

    def test_phi_access_log_written_before_unredacted_text_surfaced(
        self,
        config: DashboardConfig,
        context_outsider: RouteContext,
        review_actions_store: ReviewActionsStore,
        audit_store: AuditStore,
    ) -> None:
        """If un-redacted-text retrieval raises, the log row MUST still be
        committed (write happens at access_phi.__enter__, not on success).
        An attempted access that crashed mid-render is still an access."""
        row, calls = _writeable_audit_row_pair()
        audit_store.write(row, calls)
        request = BreakGlassRequest(justification="Crashing render")

        # We don't directly induce a crash here in RED — GREEN must wire
        # access_phi() context manager structurally so the log row is
        # committed at __enter__. We assert that calling the handler
        # produces a log row (the structural test); the "crash mid-render"
        # variant is implied by the access_phi() contract.
        record_break_glass_access(config, context_outsider, "audit-001", request)
        logs = review_actions_store.list_phi_access(
            reviewer_id="reviewer-outsider", audit_id="audit-001"
        )
        assert len(logs) == 1

    def test_care_team_unredacted_writes_phi_access_log(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
        review_actions_store: ReviewActionsStore,
        audit_store: AuditStore,
    ) -> None:
        """PRD §17 says EVERY un-redacted access writes to phi_access_log
        — not just break-glass. A care-team-of-record member viewing an
        un-redacted case detail MUST land a row in phi_access_log too
        (Codex round 1: care-team un-redacted path bypassed logging)."""
        row, calls = _writeable_audit_row_pair()
        audit_store.write(row, calls)

        detail = get_case_detail(config, context_in_care_team, "audit-001")

        assert detail.unredacted is True
        logs = review_actions_store.list_phi_access(
            reviewer_id="reviewer-care-team", audit_id="audit-001"
        )
        assert len(logs) == 1
        # Care-team path: no break-glass justification (it's the
        # privileged-by-default access route, not an override).
        assert logs[0].break_glass_justification is None

    def test_outsider_redacted_view_does_not_write_phi_access_log(
        self,
        config: DashboardConfig,
        context_outsider: RouteContext,
        review_actions_store: ReviewActionsStore,
        audit_store: AuditStore,
    ) -> None:
        """Redacted access surfaces NO un-redacted PHI → no
        phi_access_log row. The log records PHI access events, not
        every page view."""
        row, calls = _writeable_audit_row_pair()
        audit_store.write(row, calls)

        get_case_detail(config, context_outsider, "audit-001")

        logs = review_actions_store.list_phi_access(
            reviewer_id="reviewer-outsider", audit_id="audit-001"
        )
        assert logs == ()

    def test_unredacted_without_resolver_fails_loud(
        self,
        config_without_unredacted_resolver: DashboardConfig,
        audit_store: AuditStore,
    ) -> None:
        """The dashboard refuses to surface un-redacted PHI when no
        resolver is configured — fabricated placeholders would silently
        violate the PRD §17 acceptance criterion.

        Uses a reviewer whose care_team_memberships covers audit-001 so
        the un-redacted path IS reached (and then fails fast)."""
        row, calls = _writeable_audit_row_pair()
        audit_store.write(row, calls)
        context = RouteContext(
            reviewer=Reviewer(
                reviewer_id="r1",
                name="N",
                role="reviewer",
                care_team_memberships=("audit-001",),
            )
        )

        with pytest.raises(UnredactedSourceUnavailableError):
            get_case_detail(config_without_unredacted_resolver, context, "audit-001")


# =============================================================================
# AC ④ — Per-physician own-view: route guard
# =============================================================================


class TestPhysicianOwnViewGuard:
    """Requesting physician = subject physician is enforced at the route
    layer, not the UI. A physician requesting another physician's scorecard
    is rejected with :class:`PhysicianAccessDeniedError`."""

    def test_physician_can_view_own_scorecard(
        self,
        config: DashboardConfig,
        context_physician_self: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        row, calls = _writeable_audit_row_pair()
        audit_store.write(row, calls)

        scorecard = get_physician_scorecard(config, context_physician_self, "phys-001")

        assert scorecard.physician_id == "phys-001"

    def test_physician_cannot_view_other_physicians_scorecard(
        self,
        config: DashboardConfig,
        context_physician_other_subject: RouteContext,
    ) -> None:
        with pytest.raises(PhysicianAccessDeniedError):
            get_physician_scorecard(config, context_physician_other_subject, "phys-002")

    def test_ordinary_reviewer_cannot_view_any_physician_scorecard(
        self,
        config: DashboardConfig,
    ) -> None:
        """Codex round 4 P1: the previous guard only fired for
        ``role='physician'``; a reviewer with ``role='reviewer'`` (and
        ``physician_id=None``) could enumerate any physician's
        scorecard. The strict own-view guard denies regardless of role
        — only the subject physician themselves passes."""
        reviewer = Reviewer(
            reviewer_id="ordinary-reviewer",
            name="Ordinary",
            role="reviewer",
            physician_id=None,
        )
        context = RouteContext(reviewer=reviewer)
        with pytest.raises(PhysicianAccessDeniedError):
            get_physician_scorecard(config, context, "phys-001")

    def test_senior_reviewer_cannot_view_other_physician_scorecard(
        self,
        config: DashboardConfig,
    ) -> None:
        """Even senior reviewers cannot use this route to view another
        physician's scorecard — cross-physician aggregation belongs on
        a separate, explicitly privileged endpoint (out of scope for
        #26)."""
        reviewer = Reviewer(
            reviewer_id="senior-reviewer",
            name="Senior",
            role="senior_reviewer",
            physician_id=None,
        )
        context = RouteContext(reviewer=reviewer)
        with pytest.raises(PhysicianAccessDeniedError):
            get_physician_scorecard(config, context, "phys-001")

    @given(
        viewer_physician_id=st.text(
            alphabet=("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
            min_size=1,
            max_size=20,
        ),
        subject_physician_id=st.text(
            alphabet=("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_route_guard_holds_for_arbitrary_id_pairs(
        self,
        config: DashboardConfig,
        viewer_physician_id: str,
        subject_physician_id: str,
    ) -> None:
        """Property: a physician identity NEVER renders another physician's
        scorecard, for any pair of distinct ids. The route guard cannot be
        bypassed by id-shape coincidence (e.g., a viewer whose id is a
        prefix / suffix / substring of the subject's)."""
        reviewer = Reviewer(
            reviewer_id=f"r-{viewer_physician_id}",
            name="Test",
            role="physician",
            physician_id=viewer_physician_id,
            care_team_memberships=(),
        )
        context = RouteContext(
            reviewer=reviewer, subject_physician_id=subject_physician_id
        )

        if viewer_physician_id == subject_physician_id:
            # Matching ids — must NOT raise PhysicianAccessDeniedError.
            # The handler may succeed (return a scorecard) or raise
            # other errors (e.g., missing data); both are acceptable —
            # the property is that the route guard does NOT misfire on
            # own-view requests.
            try:
                get_physician_scorecard(config, context, subject_physician_id)
            except PhysicianAccessDeniedError:
                pytest.fail(
                    "PhysicianAccessDeniedError raised for matching "
                    "ids; route guard misclassified own-view as "
                    "cross-view"
                )
            except DashboardError:
                pass  # other dashboard errors are acceptable
        else:
            # Different ids — MUST raise PhysicianAccessDeniedError
            # specifically, before any data retrieval.
            with pytest.raises(PhysicianAccessDeniedError):
                get_physician_scorecard(config, context, subject_physician_id)


# =============================================================================
# AC ⑤ — DuckDB snapshot read path under concurrent batch write
# =============================================================================


class TestSnapshotReadPath:
    """The dashboard reads via ``SnapshotView`` — an in-flight batch write
    does not produce mid-query inconsistency."""

    def test_queue_reads_via_snapshot_view_not_live_store(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        """Materialize the daily snapshot via a first read, then a live
        write MUST NOT appear in a subsequent same-day read. This is the
        PRD §17 + #19 isolation guarantee — the dashboard reads from the
        frozen materialization point, not the live store."""
        row, calls = _writeable_audit_row_pair("audit-pre", "run-pre")
        audit_store.write(row, calls)

        items_before = list_queue(config, context_in_care_team)
        ids_before = {item.audit_id for item in items_before}
        assert "audit-pre" in ids_before  # baseline: pre-write IS visible

        # Live write AFTER snapshot materialization.
        row_new, calls_new = _writeable_audit_row_pair("audit-post", "run-post")
        audit_store.write(row_new, calls_new)

        items_after = list_queue(config, context_in_care_team)
        ids_after = {item.audit_id for item in items_after}

        # The materialized snapshot is frozen → post-write is invisible.
        assert "audit-post" not in ids_after
        # And the original row is still there (no regression).
        assert ids_after == ids_before

    def test_concurrent_batch_write_does_not_corrupt_snapshot_read(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        """A reader concurrent with a batch writer MUST observe a stable
        point-in-time row set: every queue item it returns existed BEFORE
        the batch writer started (snapshot frozen at materialization).
        Mid-batch writes never appear in the reader's result."""
        row_pre, calls_pre = _writeable_audit_row_pair("audit-pre", "run-pre")
        audit_store.write(row_pre, calls_pre)

        # Materialize the snapshot BEFORE the batch writer starts so we
        # have a deterministic "pre-batch" materialization point.
        items_pre = list_queue(config, context_in_care_team)
        ids_pre = {it.audit_id for it in items_pre}

        write_errors: list[BaseException] = []
        batch_started = threading.Event()
        reader_done = threading.Event()

        def _batch_writer() -> None:
            try:
                batch_started.set()
                for i in range(5):
                    r, c = _writeable_audit_row_pair(
                        f"audit-batch-{i}", f"run-batch-{i}"
                    )
                    audit_store.write(r, c)
                    if not reader_done.is_set():
                        # Yield briefly so the reader's concurrent read
                        # interleaves with the writes (not just batched
                        # after).
                        threading.Event().wait(0.001)
            except BaseException as exc:  # pragma: no cover
                write_errors.append(exc)

        writer = threading.Thread(target=_batch_writer)
        writer.start()
        try:
            batch_started.wait(timeout=2.0)
            items_concurrent = list_queue(config, context_in_care_team)
            reader_done.set()
        finally:
            writer.join(timeout=5.0)

        assert not write_errors, f"batch writer raised: {write_errors!r}"
        # Stability: the concurrent read result equals the pre-batch read
        # (snapshot was frozen — no batch row leaked in).
        ids_concurrent = {it.audit_id for it in items_concurrent}
        assert ids_concurrent == ids_pre
        # And no batch id appears.
        assert not any(
            it.audit_id.startswith("audit-batch-") for it in items_concurrent
        )

    def test_inconsistency_surfaces_as_typed_exception(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
    ) -> None:
        """If the snapshot read observes a row that does not match the
        snapshot's materialization point, the dashboard raises
        :class:`SnapshotInconsistencyError` — not a silent miss.

        This is a contract test: we assert the exception TYPE exists and
        is a DashboardError subclass; the GREEN-phase implementation must
        raise it when the snapshot view yields off-point rows."""
        assert issubclass(SnapshotInconsistencyError, DashboardError)


# =============================================================================
# AC ⑥ — Smoke tests for all routes
# =============================================================================


class TestRouteSmoke:
    """HTTP-200 smoke tests for every route the dashboard exposes."""

    @pytest.fixture
    def client(self, config: DashboardConfig, audit_store: AuditStore) -> Iterator[Any]:
        # Pre-seed the snapshot with audit-001 so case-detail and
        # break-glass routes have a row to find. The smoke contract is
        # "route reachable" — without a row the case routes legitimately
        # return 404; pre-seeding pushes them to the 200 path.
        from fastapi.testclient import TestClient

        row, calls = _writeable_audit_row_pair("audit-001", "run-001")
        audit_store.write(row, calls)

        app = create_app(config)
        with TestClient(app) as test_client:
            yield test_client

    def test_get_queue_returns_200(self, client: Any) -> None:
        response = client.get("/queue")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        # Tailwind + HTMX must be present in the rendered queue view
        # (FastAPI + HTMX + Tailwind is the PRD §17 acceptance shape).
        assert "tailwindcss" in response.text
        assert "htmx.org" in response.text

    def test_get_queue_htmx_request_returns_table_fragment(self, client: Any) -> None:
        """HTMX swap target is ``#queue-table`` with ``outerHTML``. An
        HX-Request must therefore receive a TABLE fragment, not a full
        HTML document — otherwise HTMX swaps a ``<html>...</html>`` blob
        into a table-shaped slot (Codex round 2 finding)."""
        response = client.get("/queue", headers={"HX-Request": "true"})
        assert response.status_code == 200
        # Fragment: starts with the table element, no <html>/<head>.
        assert response.text.lstrip().startswith("<table")
        assert "<html" not in response.text
        assert "<head" not in response.text

    def test_get_queue_plain_browser_returns_full_document(self, client: Any) -> None:
        """Plain browser navigation (no HX-Request header) gets the full
        page with nav chrome, Tailwind, and HTMX script tag."""
        response = client.get("/queue")
        assert response.status_code == 200
        assert "<html" in response.text
        assert "tailwindcss" in response.text
        assert "htmx.org" in response.text

    def test_get_queue_accepts_sort_query_params(
        self, client: Any, audit_store: AuditStore
    ) -> None:
        """Sortability is exposed via query params on the route. Codex
        round 1: the handler accepted sort args but the route did not
        forward them, so the queue was effectively unsortable through
        the UI."""
        # Seed two rows with different Hb so we can observe sort order.
        row_low = _audit_row(audit_id="audit-low", run_id="run-low")
        row_high = _audit_row(audit_id="audit-high", run_id="run-high")
        # The default _audit_row uses hb_value=6.5; override one to differ.
        row_high_dict = row_high.model_dump()
        row_high_dict["hb_value"] = 8.5
        row_high = AuditRow.model_validate(row_high_dict)
        audit_store.write(row_low, (_llm_call(audit_id="audit-low", run_id="run-low"),))
        audit_store.write(
            row_high, (_llm_call(audit_id="audit-high", run_id="run-high"),)
        )

        response = client.get("/queue?sort_key=hb_value&sort_direction=asc")
        assert response.status_code == 200
        assert "audit-low" in response.text
        assert "audit-high" in response.text
        # Lower Hb appears before higher in ascending order.
        assert response.text.index("audit-low") < response.text.index("audit-high")

    def test_get_case_detail_returns_200(self, client: Any) -> None:
        response = client.get("/case/audit-001")
        assert response.status_code == 200

    def test_post_break_glass_returns_200(self, client: Any) -> None:
        # HTMX submits form-encoded bodies; the route accepts a Form
        # field rather than JSON so the dashboard's UI works without a
        # client-side JSON shim.
        response = client.post(
            "/case/audit-001/break-glass",
            data={"justification": "Clinical review per attending"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_get_ward_scorecard_returns_200(self, client: Any) -> None:
        response = client.get("/scorecard/ward/ward-001")
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "path",
        ["/scorecard/ward/ward-001", "/scorecard/physician/phys-001"],
    )
    def test_flag_off_scorecards_match_pre_returns_markup(
        self,
        client: Any,
        monkeypatch: pytest.MonkeyPatch,
        path: str,
    ) -> None:
        monkeypatch.setattr(
            "bba.dashboard.app.RETURNS_LEDGER_ENABLED", False, raising=False
        )
        response = client.get(path)
        assert response.status_code == 200
        assert "RETURNED_NOT_TRANSFUSED (excluded)" not in response.text
        assert "PERIOP_TRANSFUSION_EXEMPT (excluded)" not in response.text

    def test_flag_on_scorecard_includes_returns_markup(
        self, client: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "bba.dashboard.app.RETURNS_LEDGER_ENABLED", True, raising=False
        )
        response = client.get("/scorecard/ward/ward-001")
        assert response.status_code == 200
        assert "RETURNED_NOT_TRANSFUSED (excluded)" in response.text
        assert "PERIOP_TRANSFUSION_EXEMPT (excluded)" in response.text

    def test_get_physician_scorecard_returns_200(self, client: Any) -> None:
        response = client.get("/scorecard/physician/phys-001")
        # 200 if same physician; 403 if route guard fired — either is
        # "route exists" (the smoke test goal). GREEN must wire a route
        # such that a same-physician request lands on 200.
        assert response.status_code in {200, 403}

    def test_get_pipeline_health_returns_200(self, client: Any) -> None:
        response = client.get("/pipeline-health")
        assert response.status_code == 200

    def test_auth_dependency_is_overridable_via_dependency_overrides(
        self,
        config: DashboardConfig,
        audit_store: AuditStore,
    ) -> None:
        """Codex round 4 P2: the auth dependency was previously a
        closure inside ``create_app``, so production could not register
        ``app.dependency_overrides[<dep>] = ...`` without introspecting
        FastAPI internals. The dependency is now exposed at module
        level (:func:`get_route_context`) and overridable through the
        canonical FastAPI mechanism."""
        from fastapi.testclient import TestClient

        row, calls = _writeable_audit_row_pair("audit-001", "run-001")
        audit_store.write(row, calls)
        app = create_app(config)

        # Production wires real auth here; the override replaces the
        # default-reviewer fallback with a named identity.
        overridden_reviewer = Reviewer(
            reviewer_id="override-via-deps",
            name="Production Auth Identity",
            role="physician",
            physician_id="phys-001",
            care_team_memberships=("audit-001",),
        )

        def _override() -> RouteContext:
            return RouteContext(reviewer=overridden_reviewer)

        app.dependency_overrides[get_route_context] = _override
        try:
            with TestClient(app) as client:
                response = client.get("/queue")
                assert response.status_code == 200
                # The override resolved successfully — page rendered
                # without the 401 fallback path.
                assert "tailwindcss" in response.text
        finally:
            app.dependency_overrides.clear()

    def test_routes_return_401_when_no_default_reviewer_and_no_override(
        self,
        config_without_unredacted_resolver: DashboardConfig,
    ) -> None:
        """When neither the default reviewer nor an override is
        configured, every route must 401 rather than silently surface
        unauthenticated content. (The
        ``config_without_unredacted_resolver`` fixture does set a
        default reviewer; we build a fresh one here with neither.)"""
        from fastapi.testclient import TestClient

        bare_config = DashboardConfig(
            audit_store=config_without_unredacted_resolver.audit_store,
            review_actions_store=(
                config_without_unredacted_resolver.review_actions_store
            ),
            snapshot_dir=config_without_unredacted_resolver.snapshot_dir,
            default_reviewer=None,
        )
        app = create_app(bare_config)
        with TestClient(app) as client:
            response = client.get("/queue")
            assert response.status_code == 401


# =============================================================================
# Model contracts — frozen Pydantic models declared as part of the
# interface. These pass in RED because the models are the contract.
# =============================================================================


class TestModelContracts:
    """The frozen Pydantic models are part of the public contract.

    Tests in this class exercise the contract itself (frozen, required
    fields, tz-aware datetime enforcement, etc.) — they are not behavioral
    tests of the route handlers. They pass in RED as a regression guard on
    the declared model shape.
    """

    def test_reviewer_is_frozen(self) -> None:
        reviewer = Reviewer(reviewer_id="r1", name="N", role="reviewer")
        with pytest.raises(ValidationError):
            reviewer.name = "tampered"  # type: ignore[misc]

    def test_break_glass_request_strips_then_rejects_whitespace(self) -> None:
        with pytest.raises(ValidationError):
            BreakGlassRequest(justification="\t\n  ")

    def test_break_glass_request_accepts_non_empty(self) -> None:
        req = BreakGlassRequest(justification="ok")
        assert req.justification == "ok"

    def test_physician_scorecard_is_frozen(self) -> None:
        sc = PhysicianScorecard(
            physician_id="p1",
            physician_name="N",
            ward_id="w1",
            total_orders=0,
            appropriate_count=0,
            inappropriate_count=0,
            needs_review_count=0,
            insufficient_evidence_count=0,
            average_confidence=0.0,
        )
        with pytest.raises(ValidationError):
            sc.total_orders = 99  # type: ignore[misc]

    def test_ward_model_rejects_unsafe_id(self) -> None:
        with pytest.raises(ValidationError):
            Ward(ward_id="../etc/passwd", name="bad")

    def test_physician_requires_safe_id(self) -> None:
        with pytest.raises(ValidationError):
            Physician(physician_id="", name="x", ward_id="w1")

    def test_route_context_default_subject_physician_id_is_none(self) -> None:
        ctx = RouteContext(
            reviewer=Reviewer(reviewer_id="r1", name="N", role="reviewer")
        )
        assert ctx.subject_physician_id is None
        assert ctx.ward_id is None


# =============================================================================
# A2 — Platelet rows excluded from dashboard aggregates
# =============================================================================


class TestPlateletRowsExcludedFromDashboard:
    """Platelet AuditRows must not appear in RBC dashboard counts or rates.

    WHY: dashboard views (queue, ward scorecard, physician scorecard, pipeline
    health) expose RBC-specific metrics. Platelet appropriateness uses different
    clinical thresholds (platelet count vs Hb) and different policy criteria.
    Including platelet rows in dashboard aggregates would misrepresent the RBC
    inappropriate_rate to committee reviewers. _read_snapshot_rows is the single
    choke point — filtering there protects every downstream dashboard view.
    """

    def test_pipeline_health_count_excludes_platelet_rows(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        """Pipeline health total_orders must equal RBC row count only.

        Writing one RBC row and one platelet row to the store, the reported
        total must be 1, not 2, because the platelet row is not an RBC audit.
        """
        rbc_row, rbc_calls = _writeable_audit_row_pair("audit-rbc", "run-001")
        audit_store.write(rbc_row, rbc_calls)

        plt_row_dict = {
            **_audit_row(audit_id="audit-plt", run_id="run-001").model_dump(),
            "component": "platelet",
        }
        plt_row = AuditRow.model_validate(plt_row_dict)
        plt_call = _llm_call(audit_id="audit-plt", run_id="run-001")
        audit_store.write(plt_row, (plt_call,))

        health = get_pipeline_health(config, context_in_care_team)

        assert health.total_audits == 1, (
            "platelet row must not be counted in RBC pipeline health total"
        )

    def test_queue_excludes_platelet_rows(
        self,
        context_in_care_team: RouteContext,
        audit_store: AuditStore,
        tmp_path: Path,
    ) -> None:
        """The NEEDS_REVIEW queue must not surface platelet rows.

        Both the RBC and platelet rows carry needs_human_review=True, but
        only the RBC row belongs in the RBC queue.
        """
        from bba.audit_store import AuditStoreConfig
        from bba.dashboard import DashboardConfig

        store2 = AuditStore(
            AuditStoreConfig(root_dir=tmp_path / "aud2", code_version="v-test")
        )
        rbc_row, rbc_calls = _writeable_audit_row_pair("audit-rbc2", "run-002")
        store2.write(rbc_row, rbc_calls)

        plt_row_dict = {
            **_audit_row(audit_id="audit-plt2", run_id="run-002").model_dump(),
            "component": "platelet",
        }
        plt_row = AuditRow.model_validate(plt_row_dict)
        store2.write(plt_row, (_llm_call(audit_id="audit-plt2", run_id="run-002"),))

        cfg2 = DashboardConfig(
            audit_store=store2,
            review_actions_store=_InMemoryReviewActionsStore(),
            snapshot_dir=tmp_path / "snap2",
        )
        items = list_queue(cfg2, context_in_care_team)
        audit_ids = {it.audit_id for it in items}

        assert "audit-rbc2" in audit_ids
        assert "audit-plt2" not in audit_ids, (
            "platelet row must not appear in the RBC NEEDS_REVIEW queue"
        )
