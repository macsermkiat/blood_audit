"""Typed exceptions for the dashboard module.

Every dashboard-specific failure surfaces as a subclass of
:class:`DashboardError` so callers can ``except DashboardError`` and not have
to enumerate concrete types. The FastAPI app converts these into HTTP error
responses at the route boundary.
"""

from __future__ import annotations


class DashboardError(Exception):
    """Base class for every dashboard-layer error."""


class MissingJustificationError(DashboardError):
    """Raised when a break-glass un-redacted-view request lacks justification.

    PRD §17: viewing un-redacted text requires reviewer-supplied justification
    text (free-form, non-empty) so the post-hoc audit trail can trace WHY the
    redaction policy was overridden. A request without one is rejected BEFORE
    any un-redacted text is surfaced and BEFORE any ``phi_access_log`` row is
    written.
    """


class PhysicianAccessDeniedError(DashboardError):
    """Raised when the requesting physician != the subject physician.

    PRD §17 (and #26 AC④): the per-physician own-view scorecard is a route
    guard, not a UI affordance. A physician requesting ``/scorecard/physician/
    {other_id}`` is rejected at the route layer — the dashboard never renders
    another physician's scorecard, regardless of role.
    """


class CareTeamAccessDeniedError(DashboardError):
    """Raised when the reviewer is not on the case's care-team-of-record.

    PRD §17: un-redacted case detail is gated on care-team-of-record
    membership. A reviewer outside the care team can only view the case
    through the break-glass flow (which itself requires a justification and
    writes to ``phi_access_log``).
    """


class SnapshotInconsistencyError(DashboardError):
    """Raised when the DuckDB snapshot read observes mid-batch-write rows.

    PRD §17 + #19: the dashboard reads from a daily-rotated DuckDB snapshot
    view to insulate readers from in-flight batch writes. If the snapshot
    handle yields a row that does not match the snapshot's materialization
    point, that is a structural bug; this exception makes the failure mode
    loud rather than silent.
    """
