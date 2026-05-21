"""Per-table normalize layer — preprocesses CSV headers and rows before drift detection.

The bundle's CSVs do not match the audit's pandera schemas one-to-one: they
are wider than the schema (extra columns the audit ignores), some use
Title-Case naming (``IPTSUMOPRT``, ``INCPT``, ``ICD9CM``), and one carries duplicate
column names (``IPDADMPROGRESS`` has two ``HN`` and two ``AN`` columns —
positions 1 + 30 and 2 + 3 respectively).

Without this layer, :func:`bba.ingest.schemas.validate_header` would raise
``SchemaDriftError`` on every real bundle because of the extras. The lock
spec (2026-05-19; see ``docs/ingest-mapping.md``) chooses policy *(a)*:
project the raw header down to the schema's declared columns, log the
dropped extras to the run audit, and pass the projected header to
``validate_header`` for the strict-drift check.

**Header pass** — :func:`normalize_header`. Applies in fixed order:

1. **Case-normalize** (IPTSUMOPRT, INCPT, ICD9CM only): uppercase all column
   names so ``An`` / ``Hn`` / ``Icd9cm`` / ``Indate`` line up with the
   ALL-CAPS schema declarations.
2. **Dedupe**: drop columns whose name already appeared earlier in the
   (post-case-normalize) header. First occurrence wins.
3. **Project**: keep columns the schema declares; route everything else
   to the dropped list.

**Row pass** — :func:`normalize_rows`. Applies after :func:`validate_header`
clears the projected header, before any Parquet write:

1. **Project + positional dedupe**: for each kept column, read the cell at
   its first-occurrence position in the raw row. Drops the duplicate-AN
   cell at position 3 and duplicate-HN cell at position 30 of
   IPDADMPROGRESS by construction (same rule the header pass applies,
   just at the row level).
2. **Year-filter** (IPDADMPROGRESS, IPDNRFOCUSDT only): drop rows whose
   date column does not match :data:`COHORT_YEAR`. Saves ~7% of the
   2.7M / 16.9M raw rows from reaching the Parquet writer.
3. **Date-parse** (procedure-family tables): convert ``IPTSUMOPRT.INDATE``
   / ``INCPT.INCDATE`` from English-locale form to ISO ``YYYY-MM-DD`` via
   :func:`bba.ingest.parse_kcmh_english_date`. Failures
   produce a ``parse_warning`` carried on the row (mirrors
   :func:`bba.ingest.parse_hosxp_time` semantics; the row is **not**
   dropped on parse failure — PRD §1 fix E35).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from bba.ingest.date_parser import parse_kcmh_english_date
from bba.ingest.models import CSVTable
from bba.ingest.schemas import get_schema

logger = logging.getLogger(__name__)


# Tables whose CSV headers use Title-Case naming and must be uppercased
# before any drift detection. The other tables already use ALL-CAPS in
# the bundle's exports and need no case-normalization.
_CASE_NORMALIZE_TABLES: frozenset[CSVTable] = frozenset(
    {
        "IPTSUMOPRT",
        "INCPT",
        "ICD9CM",
    }
)


# Per-table row-level rules. Single point of change for the cohort year so
# a future run with a different scope (e.g., 2026 audit) only needs to
# update one constant; rules that key off the table-to-column mapping live
# in the per-rule dictionaries below.
COHORT_YEAR: int = 2025

# Year-filter rule: drop rows whose date column's year does not match the
# cohort year. The HOSxP standard format is ``"2025-05-19 00:00:00.000"``,
# but real exports can also carry a UTF-8 BOM on the first cell and/or
# leading whitespace. The match pattern strips both, then requires
# **exactly 4** leading digits followed by a literal hyphen (the HOSxP
# date separator) so malformed cells are dropped along with wrong-year
# rows. Two classes of false-positive this catches:
#   - clobbered 5-digit year prefix: ``"20259-..."``
#   - wrong separator: ``"2025X05-19"`` / ``"2025/05/19"``
# See issue #63 (Codex P2.A.3, plus the GitHub Codex P2 follow-up that
# tightened the separator check on PR #65).
_YEAR_FILTER_RE: re.Pattern[str] = re.compile(r"^[﻿\s]*(\d{4})-")
_YEAR_FILTER_COLUMN: dict[CSVTable, str] = {
    "IPDADMPROGRESS": "PROGDATE",
    "IPDNRFOCUSDT": "PROGRESSDATE",
}

# Date-parse rule: replace the cell with the ISO-8601 form of the parsed
# date. Procedure-family exports use English-locale date strings here; all
# other date columns in the bundle use the HOSxP standard format and are
# passed through unchanged.
_DATE_PARSE_COLUMN: dict[CSVTable, str] = {
    "IPTSUMOPRT": "INDATE",
    "INCPT": "INCDATE",
}

# Rule-order guard: no table currently has both year-filter and date-parse
# rules, and :func:`normalize_row` applies year-filter before date-parse
# on the *raw* cell value. If a future table needs both, that ordering
# decision must be made explicitly (e.g., parse the date first then
# compare against the cohort year). Adding a table to both dicts without
# revisiting the order is a silent bug — fail at module import so the
# invariant cannot drift. See issue #63 (Codex P2.A.4).
#
# Using ``raise`` rather than ``assert`` so the guard survives
# ``python -O`` (assertions are stripped under optimized builds).
_rule_overlap = set(_YEAR_FILTER_COLUMN) & set(_DATE_PARSE_COLUMN)
if _rule_overlap:
    raise RuntimeError(
        f"no table may appear in both _YEAR_FILTER_COLUMN and "
        f"_DATE_PARSE_COLUMN without an explicit precedence decision in "
        f"normalize_row(); overlap: {sorted(_rule_overlap)}"
    )
del _rule_overlap


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


@dataclass(frozen=True, slots=True)
class NormalizedRow:
    """One CSV row after the row-level normalize pass.

    ``cells`` is parallel to the :class:`NormalizedHeader` ``header`` —
    same length, same column order. A consumer can build a dict via
    ``dict(zip(header, row.cells, strict=True))`` if it prefers
    name-based access.

    ``parse_warnings`` is the list of column-level parse failures (e.g.,
    a malformed ``INDATE``) attached to *this* row. The row is yielded
    anyway — strict parser philosophy (PRD §1 fix E35) keeps the row but
    surfaces the failure so the run audit can record it. Empty tuple is
    the clean case.
    """

    cells: tuple[str, ...]
    parse_warnings: tuple[tuple[str, str], ...]


def normalize_header(table: CSVTable, raw_header: Sequence[str]) -> NormalizedHeader:
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


def _row_positions(
    table: CSVTable,
    raw_header: Sequence[str],
    kept_header: Sequence[str],
) -> tuple[int, ...]:
    """Return the raw-row indices to read for each column in ``kept_header``.

    The returned tuple is parallel to ``kept_header``: position ``i`` in
    the result is the index into ``raw_header`` (and therefore the raw
    row) from which the ``i``-th kept column should be read.

    Handles case-normalize for the procedure-family tables and the
    first-occurrence-wins dedupe rule. Mirrors :func:`normalize_header`
    so that the row pass aligns 1:1 with the header pass.
    """
    case_normalize = table in _CASE_NORMALIZE_TABLES
    first_index: dict[str, int] = {}
    for i, name in enumerate(raw_header):
        canonical = name.upper() if case_normalize else name
        if canonical not in first_index:
            first_index[canonical] = i
    return tuple(first_index[col] for col in kept_header)


RAGGED_ROW_WARNING_KEY: str = "__row__"
"""Sentinel ``column_name`` used in :attr:`NormalizedRow.parse_warnings` for
row-level warnings (e.g., a ragged row) that are not tied to any single
column. Real column warnings (such as INDATE parse failures) use the
column's own name so consumers can route them by column."""


def normalize_row(
    table: CSVTable,
    raw_row: Sequence[str],
    positions: Sequence[int],
    kept_header: Sequence[str],
    cohort_year: int = COHORT_YEAR,
) -> NormalizedRow | None:
    """Apply per-table row rules to a single raw row.

    Returns ``None`` if the row is filtered out by a year-filter rule, or
    a :class:`NormalizedRow` whose cells align with ``kept_header`` and
    whose ``parse_warnings`` capture any column-level parse failures.

    ``positions`` is the output of :func:`_row_positions` for the same
    ``(table, raw_header, kept_header)``; caller computes it once per
    table and passes it in to avoid recomputing on every row.

    **Ragged rows.** A raw row that has fewer cells than
    ``max(positions) + 1`` (e.g., a truncated export line) is *not* a
    fatal error: missing cells are filled with ``""`` and a row-level
    warning is attached under :data:`RAGGED_ROW_WARNING_KEY`. The
    strict-parser philosophy (PRD §1 fix E35) prefers surfacing the
    issue over crashing the entire ingest. Downstream year-filter and
    date-parse rules then run on the filled row — if year-filter drops
    it (because the date cell was the missing one), the warning is
    lost; the ``rows_filtered`` count in
    :func:`bba.ingest.pipeline._drain_normalize_rows` still records the
    drop indirectly.
    """
    cells: list[str] = []
    ragged_columns: list[str] = []
    for col_name, idx in zip(kept_header, positions, strict=True):
        if idx < len(raw_row):
            cells.append(raw_row[idx])
        else:
            cells.append("")
            ragged_columns.append(col_name)

    warnings: list[tuple[str, str]] = []
    if ragged_columns:
        warnings.append(
            (
                RAGGED_ROW_WARNING_KEY,
                f"ragged row: {len(raw_row)} cells, "
                f"missing {len(ragged_columns)} for columns {ragged_columns}",
            )
        )

    year_col = _YEAR_FILTER_COLUMN.get(table)
    if year_col is not None:
        idx = kept_header.index(year_col)
        match = _YEAR_FILTER_RE.match(cells[idx])
        if match is None or int(match.group(1)) != cohort_year:
            return None

    parse_col = _DATE_PARSE_COLUMN.get(table)
    if parse_col is not None:
        idx = kept_header.index(parse_col)
        parsed = parse_kcmh_english_date(cells[idx])
        if parsed.value is None:
            assert parsed.parse_warning is not None  # invariant on DateParseResult
            warnings.append((parse_col, parsed.parse_warning))
        else:
            cells[idx] = parsed.value.isoformat()

    return NormalizedRow(cells=tuple(cells), parse_warnings=tuple(warnings))


def normalize_rows(
    table: CSVTable,
    raw_header: Sequence[str],
    kept_header: Sequence[str],
    raw_rows: Iterable[Sequence[str]],
    cohort_year: int = COHORT_YEAR,
) -> Iterator[NormalizedRow]:
    """Stream a raw CSV's rows through the per-table row-level pipeline.

    Computes the position map once from ``(raw_header, kept_header)`` and
    applies :func:`normalize_row` to each row. Drops rows the year-filter
    rejects; yields the rest in input order. Parse warnings attach to
    individual rows — caller iterates ``r.parse_warnings`` to surface
    them to the run audit.

    Year-filter accounting (rows-in vs rows-yielded) is the caller's
    responsibility: this generator does not expose statistics directly,
    but a simple counter wrapping the iteration suffices.
    """
    positions = _row_positions(table, raw_header, kept_header)
    for raw_row in raw_rows:
        result = normalize_row(table, raw_row, positions, kept_header, cohort_year)
        if result is not None:
            yield result


__all__ = (
    "COHORT_YEAR",
    "NormalizedHeader",
    "NormalizedRow",
    "RAGGED_ROW_WARNING_KEY",
    "normalize_header",
    "normalize_row",
    "normalize_rows",
)
