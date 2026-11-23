"""Value models for issue events (blood_audit #276).

An issue event is the audit unit: the physical blood units one REQNO issued
close together, judged once at the first unit's issue time. Units and events
are frozen Pydantic models like the rest of the pipeline's value types; every
flag on them is derived from the units, so an event cannot disagree with its
own units.
"""

from __future__ import annotations

from datetime import timedelta

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from bba.returns_ledger import INCOMPAT_STATUS, RETURNED_STATUS

# Plan 1.2: the last gap bucket where most consecutive red-cell issues carry no
# new Hb draw. Part of the code identity (plan 4.1): changing it re-keys events.
ISSUE_EVENT_GAP = timedelta(hours=4)

# A unit that left the bank and came back unused is returned or
# crossmatch-incompatible (never given). Dispensed (2) and transfused (5)
# units count as transfused.
NOT_TRANSFUSED_STATUSES = frozenset({RETURNED_STATUS, INCOMPAT_STATUS})
TRANSFUSED_STATUSES = frozenset({"2", "5"})


class IssuedUnit(BaseModel):
    """One physical unit (``DNRNO``, ``SEQNO``, ``BDTYPE``) that left the bank.

    ``status`` is the terminal ``UNITSTAT`` of the unit's lifecycle rows.
    ``issue_at`` is the earliest PAYDATE/PAYTIME among them; ``returned_at``
    and ``given_at`` are the earliest return and administration times, kept
    for display only (an issue proves issue, not administration).
    """

    model_config = ConfigDict(frozen=True)

    dnrno: str
    seqno: str
    bdtype: str
    status: str
    issue_at: AwareDatetime
    returned_at: AwareDatetime | None
    given_at: AwareDatetime | None

    @property
    def issue_order(self) -> tuple[AwareDatetime, str, str, str]:
        """Plan 1.1 tie-break: the event's first unit fixes its audit_id."""
        return (self.issue_at, self.dnrno, self.seqno, self.bdtype)

    @property
    def returned(self) -> bool:
        """Returned to the bank; keyed on status, never on a return date."""
        return self.status == RETURNED_STATUS

    @property
    def not_transfused(self) -> bool:
        return self.status in NOT_TRANSFUSED_STATUSES


class LedgerUnits(BaseModel):
    """One REQNO's BDVSTTRANS units on the order's admission.

    ``issued`` is in issue order. Units with no issue time never left the bank
    (crossmatch holds); they form no event and are only counted.
    """

    model_config = ConfigDict(frozen=True)

    issued: tuple[IssuedUnit, ...]
    units_reserved_not_issued: int


class IssueEvent(BaseModel):
    """Units of one REQNO issued as one transfusion decision, in issue order."""

    model_config = ConfigDict(frozen=True)

    units: tuple[IssuedUnit, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _units_in_issue_order(self) -> IssueEvent:
        orders = [unit.issue_order for unit in self.units]
        if orders != sorted(orders):
            raise ValueError("issue event units must be in issue order")
        return self

    @property
    def issue_datetime(self) -> AwareDatetime:
        """The event anchor: the first unit's issue time."""
        return self.units[0].issue_at

    @property
    def units_returned(self) -> int:
        """Units with terminal status 3, as the returns ledger counts them."""
        return sum(1 for unit in self.units if unit.returned)

    @property
    def all_returned(self) -> bool:
        """No unit was transfused: the event is RETURNED_NOT_TRANSFUSED."""
        return all(unit.not_transfused for unit in self.units)
