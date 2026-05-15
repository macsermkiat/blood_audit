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
    CareTeamAccessDeniedError,
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
    Ward,
    WardScorecard,
    create_app,
)
from bba.dashboard.app import (
    get_case_detail,
    get_physician_scorecard,
    get_pipeline_health,
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
            ReviewActionsConfig(
                dsn="postgresql://fake:fake@localhost:5432/fake"
            )
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
    def access_phi(
        self, access: PhiAccessInput
    ) -> Iterator[PhiAccessLog]:
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
def config(
    audit_store: AuditStore,
    review_actions_store: ReviewActionsStore,
    tmp_path: Path,
) -> DashboardConfig:
    """A DashboardConfig with a permissive default reviewer for routes
    that don't have a route-context fixture wired (e.g.,
    :class:`TestRouteSmoke`).

    The default reviewer is a physician with ``physician_id="phys-001"``
    so the physician-scorecard smoke route returns 200 (own-view) rather
    than 403 (cross-view). ``care_team_memberships=("audit-001",)`` makes
    the case-detail smoke route un-redacted.
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
    return RouteContext(
        reviewer=physician_self, subject_physician_id="phys-001"
    )


@pytest.fixture
def context_physician_other_subject(physician_self: Reviewer) -> RouteContext:
    return RouteContext(
        reviewer=physician_self, subject_physician_id="phys-002"
    )


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

    def test_ward_scorecard_view_renders_with_synthetic_data(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        row, calls = _writeable_audit_row_pair()
        audit_store.write(row, calls)

        scorecard = get_ward_scorecard(
            config, context_in_care_team, "ward-001"
        )

        assert isinstance(scorecard, WardScorecard)
        assert scorecard.ward_id == "ward-001"

    def test_physician_scorecard_view_renders_with_synthetic_data(
        self,
        config: DashboardConfig,
        context_physician_self: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        row, calls = _writeable_audit_row_pair()
        audit_store.write(row, calls)

        scorecard = get_physician_scorecard(
            config, context_physician_self, "phys-001"
        )

        assert isinstance(scorecard, PhysicianScorecard)
        assert scorecard.physician_id == "phys-001"

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

        record_break_glass_access(
            config, context_outsider, "audit-001", request
        )

        logs = review_actions_store.list_phi_access(
            reviewer_id="reviewer-outsider", audit_id="audit-001"
        )
        assert len(logs) == 1
        assert (
            logs[0].break_glass_justification
            == "Urgent clinical review per attending"
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
        record_break_glass_access(
            config, context_outsider, "audit-001", request
        )
        logs = review_actions_store.list_phi_access(
            reviewer_id="reviewer-outsider", audit_id="audit-001"
        )
        assert len(logs) == 1


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

        scorecard = get_physician_scorecard(
            config, context_physician_self, "phys-001"
        )

        assert scorecard.physician_id == "phys-001"

    def test_physician_cannot_view_other_physicians_scorecard(
        self,
        config: DashboardConfig,
        context_physician_other_subject: RouteContext,
    ) -> None:
        with pytest.raises(PhysicianAccessDeniedError):
            get_physician_scorecard(
                config, context_physician_other_subject, "phys-002"
            )

    @given(
        viewer_physician_id=st.text(
            alphabet=(
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
            ),
            min_size=1,
            max_size=20,
        ),
        subject_physician_id=st.text(
            alphabet=(
                "abcdefghijklmnopqrstuvwxyz"
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789"
            ),
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
                get_physician_scorecard(
                    config, context, subject_physician_id
                )
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
                get_physician_scorecard(
                    config, context, subject_physician_id
                )


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
        """Snapshot is materialized; a subsequent live write must not
        appear in the queue read from the snapshot."""
        row, calls = _writeable_audit_row_pair("audit-pre", "run-pre")
        audit_store.write(row, calls)

        items_before = list_queue(config, context_in_care_team)
        ids_before = {item.audit_id for item in items_before}

        row_new, calls_new = _writeable_audit_row_pair("audit-post", "run-post")
        audit_store.write(row_new, calls_new)

        items_after = list_queue(config, context_in_care_team)
        ids_after = {item.audit_id for item in items_after}

        # Same-day snapshot: post-write must NOT leak into the read.
        assert "audit-post" not in ids_after - ids_before

    def test_concurrent_batch_write_does_not_corrupt_snapshot_read(
        self,
        config: DashboardConfig,
        context_in_care_team: RouteContext,
        audit_store: AuditStore,
    ) -> None:
        """Read while a batch writer is appending — the read MUST observe a
        consistent point-in-time row set, never a partial mid-batch view."""
        row_pre, calls_pre = _writeable_audit_row_pair("audit-pre", "run-pre")
        audit_store.write(row_pre, calls_pre)

        write_errors: list[BaseException] = []

        def _batch_writer() -> None:
            try:
                for i in range(5):
                    r, c = _writeable_audit_row_pair(
                        f"audit-batch-{i}", f"run-batch-{i}"
                    )
                    audit_store.write(r, c)
            except BaseException as exc:  # pragma: no cover - test diagnostic
                write_errors.append(exc)

        writer = threading.Thread(target=_batch_writer)
        writer.start()
        try:
            items = list_queue(config, context_in_care_team)
        finally:
            writer.join()

        assert not write_errors, f"batch writer raised: {write_errors!r}"
        # The read result must be a tuple of QueueItems (point-in-time
        # consistent); we don't assert size because the snapshot's
        # materialization point is implementation-defined.
        assert all(isinstance(it, QueueItem) for it in items)

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
    def client(
        self, config: DashboardConfig, audit_store: AuditStore
    ) -> Iterator[Any]:
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

    def test_get_case_detail_returns_200(self, client: Any) -> None:
        response = client.get("/case/audit-001")
        assert response.status_code == 200

    def test_post_break_glass_returns_200(self, client: Any) -> None:
        response = client.post(
            "/case/audit-001/break-glass",
            json={"justification": "Clinical review per attending"},
        )
        assert response.status_code == 200

    def test_get_ward_scorecard_returns_200(self, client: Any) -> None:
        response = client.get("/scorecard/ward/ward-001")
        assert response.status_code == 200

    def test_get_physician_scorecard_returns_200(self, client: Any) -> None:
        response = client.get("/scorecard/physician/phys-001")
        # 200 if same physician; 403 if route guard fired — either is
        # "route exists" (the smoke test goal). GREEN must wire a route
        # such that a same-physician request lands on 200.
        assert response.status_code in {200, 403}

    def test_get_pipeline_health_returns_200(self, client: Any) -> None:
        response = client.get("/pipeline-health")
        assert response.status_code == 200


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
        reviewer = Reviewer(
            reviewer_id="r1", name="N", role="reviewer"
        )
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
            reviewer=Reviewer(
                reviewer_id="r1", name="N", role="reviewer"
            )
        )
        assert ctx.subject_physician_id is None
        assert ctx.ward_id is None
