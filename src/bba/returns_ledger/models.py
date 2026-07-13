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
    """Per-order aggregation over one REQNO's BDVSTTRANS unit rows.

    Each ledger row is one physical, dispensed blood unit. The summary counts
    returned (``Unitstat==3``) and transfused (``Unitstat==5``) units, records
    the ordered unit amount summed from the order's BDVSTDT ``UNITAMT`` lines,
    and reports whether the ledger completely covers the order.

    ``disposition`` is derived (see :meth:`disposition`):

    - ``not_transfused`` only when the ledger is complete and every unit was
      returned;
    - ``transfused`` when the ledger is complete and at least one unit is
      non-returned (or any unit is explicitly transfused);
    - ``inconclusive`` otherwise (fail-closed).

    "Returned" keys on the returned status code, never on the presence of a
    return date: a transfused unit can also carry a stray return timestamp.
    """

    model_config = ConfigDict(frozen=True)

    units_total: int = 0
    units_returned: int = 0
    units_transfused: int = 0
    ordered_unit_amount: int | None = None
    ledger_complete: bool = False

    @property
    def disposition(self) -> Disposition:
        """Derived disposition; ``inconclusive`` unless the ledger is complete."""
        if not self.ledger_complete:
            return "inconclusive"
        if self.units_returned == self.units_total:
            return "not_transfused"
        return "transfused"
