"""Value model for the BDVSTTRANS returns-ledger aggregation (spec #119).

Returns come from structured blood-bank ledger rows, NOT from ``VitalsNote``
free text, so this lives in its own package rather than ``vitals_extractor``
(whose ``AdministrationSummary`` carries a binding facts-only guardrail that
forbids a "not transfused" representation). A ledger row is a different source:
a physical blood unit with an explicit disposition status code.

:class:`ReturnsSummary` is a frozen Pydantic model mirroring the immutable
summary contract used elsewhere in the pipeline (:class:`PeriopSummary`,
:class:`AdministrationSummary`). The ``disposition`` is derived, never stored,
so a summary cannot be constructed in an inconsistent state.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Disposition = Literal["not_transfused", "transfused", "inconclusive"]


class ReturnsSummary(BaseModel):
    """Per-order aggregation over one REQNO's BDVSTTRANS physical units.

    ``units_total`` counts DISTINCT physical units, not raw ledger rows: the
    complete production export records a unit's lifecycle as multiple rows
    (dispense + return, aliquots) keyed apart by ``SEQNO``, so
    :func:`summarize_returns` collapses rows sharing ``(DNRNO, SEQNO)`` to one
    unit at its terminal status before counting (spec #119 complete-ledger
    ingest). The summary counts returned (terminal ``Unitstat==3``), transfused
    (``==5``) and crossmatch-incompatible (``==7``) units, records the ordered
    unit amount summed from the order's BDVSTDT ``UNITAMT`` lines, and reports
    whether the ledger COVERS the order (``ledger_complete`` =
    ``units_total >= ordered_unit_amount``).

    ``disposition`` is derived (see :meth:`disposition`):

    - ``not_transfused`` when the ledger is complete and every unit reached a
      non-transfusion terminal (returned OR incompatible — an Incompat unit was
      never given);
    - ``transfused`` when the ledger is complete and at least one unit is
      neither returned nor incompatible (dispensed/presumed-given, or an
      explicit transfused-status unit);
    - ``inconclusive`` otherwise (fail-closed).

    "Returned" keys on the returned status code, never on the presence of a
    return date: a transfused unit can also carry a stray return timestamp.
    """

    model_config = ConfigDict(frozen=True)

    units_total: int = 0
    units_returned: int = 0
    units_transfused: int = 0
    # Crossmatch-incompatible units (terminal ``Unitstat==7``). Never
    # transfused, so they count toward the not-transfused screen alongside
    # returned units. Internal to the disposition; not a report column.
    units_incompat: int = 0
    ordered_unit_amount: int | None = None
    ledger_complete: bool = False

    @property
    def disposition(self) -> Disposition:
        """Derived disposition; ``inconclusive`` unless the ledger is complete.

        With a guaranteed-complete ledger an over-dispensed all-returned order
        genuinely has ALL its units returned — the patient received nothing — so
        it screens ``not_transfused`` (spec #119 complete-ledger go-live relaxes
        the earlier NARROW exact-count guard, which only existed to hedge against
        a hidden transfused replacement unit on a PARTIAL export). A unit that
        was crossmatch-incompatible (``Unitstat==7``) was never transfused, so it
        counts toward the not-transfused screen alongside returned units.
        """
        if not self.ledger_complete:
            return "inconclusive"
        if self.units_returned + self.units_incompat == self.units_total:
            return "not_transfused"
        return "transfused"
