"""Per-table normalize layer — preprocesses CSV headers before drift detection.

The bundle's CSVs do not match the audit's pandera schemas one-to-one: they
are wider than the schema (extra columns the audit ignores), some use
Title-Case naming (``IPTSUMOPRT``, ``ICD9CM``), and one carries duplicate
column names (``IPDADMPROGRESS`` has two ``HN`` and two ``AN`` columns —
positions 1 + 30 and 2 + 3 respectively).

Without this layer, :func:`bba.ingest.schemas.validate_header` would raise
``SchemaDriftError`` on every real bundle because of the extras. The lock
spec (2026-05-19; see ``docs/ingest-mapping.md``) chooses policy *(a)*:
project the raw header down to the schema's declared columns, log the
dropped extras to the run audit, and pass the projected header to
``validate_header`` for the strict-drift check.

Applies in fixed order per table:

1. **Case-normalize** (IPTSUMOPRT, ICD9CM only): uppercase all column
   names so ``An`` / ``Icd9cm`` / ``Indate`` line up with the
   ALL-CAPS schema declarations.
2. **Dedupe**: drop columns whose name already appeared earlier in the
   (post-case-normalize) header. First occurrence wins. For
   IPDADMPROGRESS this drops the second ``HN`` (col 30) and second
   ``AN`` (col 3) per the locked spec. Applied universally so an
   unexpected duplicate in any future export shows up in the dropped
   list rather than corrupting drift detection.
3. **Project**: keep columns the schema declares; route everything else
   to the dropped list.

**Phase-1 scope: header-only.** Row-level normalize rules — year-filter
for IPDADMPROGRESS and IPDNRFOCUSDT (``PROGDATE`` / ``PROGRESSDATE`` in
2025), INDATE date-parse via :func:`bba.ingest.parse_iptsumoprt_date` for
IPTSUMOPRT, and positional row-cell dropping that mirrors the header
dedupe for IPDADMPROGRESS — are owned by the per-row ingest tickets and
not implemented here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from bba.ingest.models import CSVTable
from bba.ingest.schemas import get_schema

logger = logging.getLogger(__name__)


# Tables whose CSV headers use Title-Case naming and must be uppercased
# before any drift detection. The other 9 tables already use ALL-CAPS in
# the bundle's exports and need no case-normalization.
_CASE_NORMALIZE_TABLES: frozenset[CSVTable] = frozenset(
    {
        "IPTSUMOPRT",
        "ICD9CM",
    }
)


@dataclass(frozen=True, slots=True)
class NormalizedHeader:
    """Result of normalizing one CSV table's header.

    ``header`` is the projected, case-normalized, deduplicated header — a
    subset of the raw CSV columns that the schema declares. Pass it to
    :func:`bba.ingest.schemas.validate_header` for the strict drift check
    (which now only fires on *required* columns missing from the file,
    since extras have been projected away).

    ``dropped`` lists the raw columns that did not survive (duplicates and
    schema-undeclared extras). Order preserves the raw column order so an
    operator reading the audit log can correlate dropped names with their
    file-position context.
    """

    header: list[str]
    dropped: list[str]


def normalize_header(
    table: CSVTable, raw_header: Sequence[str]
) -> NormalizedHeader:
    """Apply the per-table normalize rules and project to declared columns.

    Returns a :class:`NormalizedHeader`. Does not raise on
    schema-undeclared columns (that is the point: they land in ``dropped``)
    nor on missing required columns (callers run
    :func:`validate_header` separately, which raises ``SchemaDriftError``
    on missing required columns).
    """
    declared = set(get_schema(table).columns)

    if table in _CASE_NORMALIZE_TABLES:
        cols: list[str] = [col.upper() for col in raw_header]
    else:
        cols = list(raw_header)

    seen: set[str] = set()
    kept: list[str] = []
    dropped: list[str] = []

    for col in cols:
        if col in seen:
            dropped.append(col)
            continue
        seen.add(col)
        if col in declared:
            kept.append(col)
        else:
            dropped.append(col)

    return NormalizedHeader(header=kept, dropped=dropped)


__all__ = ("NormalizedHeader", "normalize_header")
