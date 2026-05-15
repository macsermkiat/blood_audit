"""bba.dashboard — FastAPI + HTMX reviewer UI.

See issue #26 for acceptance criteria. PRD §17 Implementation Decisions:

* FastAPI + HTMX + Tailwind reviewer dashboard.
* Five views: NEEDS_REVIEW queue (sortable), case-detail (un-redacted with
  care-team-of-record gating + break-glass justification capture), per-ward
  scorecard, per-physician own-view scorecard, pipeline-health dashboard.
* Every un-redacted access writes to ``phi_access_log`` via #25.
* Daily DuckDB snapshot view from #19 for ``audit_results`` reads (so
  in-flight batch writes do not produce inconsistent reads mid-query).

This module is the human-facing surface for the audit pipeline.
"""

from bba.dashboard.app import create_app
from bba.dashboard.exceptions import (
    CareTeamAccessDeniedError,
    DashboardError,
    MissingJustificationError,
    PhysicianAccessDeniedError,
    SnapshotInconsistencyError,
)
from bba.dashboard.models import (
    BreakGlassRequest,
    CaseDetail,
    DashboardConfig,
    Physician,
    PipelineHealth,
    PhysicianScorecard,
    QueueItem,
    Reviewer,
    RouteContext,
    Ward,
    WardScorecard,
)


__all__ = [
    "BreakGlassRequest",
    "CareTeamAccessDeniedError",
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
]
