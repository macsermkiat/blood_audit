"""BDVSTTRANS returns-ledger disposition aggregation (spec #119, ticket #120).

Public surface: the pure :func:`summarize_returns` aggregator and the frozen
:class:`ReturnsSummary` value it produces. The disposition router that consumes
the summary (RETURNED_NOT_TRANSFUSED / PERIOP_TRANSFUSION_EXEMPT) lands in later
tickets; this package is read-only plumbing.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.returns_ledger.models import Disposition, ReturnsSummary
from bba.returns_ledger.summarize import (
    INCOMPAT_STATUS,
    RETURNED_STATUS,
    physical_unit_rows,
    physical_units,
    rows_for_admission,
    summarize_event_returns,
    summarize_returns,
    terminal_status,
)

__all__: Sequence[str] = (
    "INCOMPAT_STATUS",
    "RETURNED_STATUS",
    "Disposition",
    "ReturnsSummary",
    "physical_unit_rows",
    "physical_units",
    "rows_for_admission",
    "summarize_event_returns",
    "summarize_returns",
    "terminal_status",
)
