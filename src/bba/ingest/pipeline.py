"""End-to-end ingest orchestration: CSV → header validation → run-identity → noop-or-mark.

Public entry point: :func:`ingest`. Per PRD §1:

* discover the 13 HOSxP CSVs in ``config.input_dir`` by file stem (per the
  2026-05-19 schema lock; see ``docs/ingest-mapping.md``);
* fail loud (:class:`IncompleteInputError`) if the input dir is missing or
  any canonical CSV is absent;
* validate each header against its pandera schema, raising
  :class:`SchemaDriftError` on unknown or missing columns;
* derive a :class:`RunIdentity` from the input content hashes + schema
  fingerprint + code version;
* short-circuit when the identity reports itself complete on disk
  (``skipped_idempotent=True``).

The Phase-1 implementation focuses on header validation and the idempotency
boundary; per-row Parquet writes are owned by subsequent tickets that consume
the validated dataframes.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast, get_args

from bba.ingest.hashing import content_hash
from bba.ingest.models import CSVTable, IngestConfig, IngestResult
from bba.ingest.normalize import normalize_header, normalize_rows
from bba.ingest.run_identity import RunIdentity
from bba.ingest.schemas import (
    IncompleteInputError,
    schema_fingerprint,
    validate_header,
)

logger = logging.getLogger(__name__)


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            return next(reader)
        except StopIteration:
            return []


def _stream_csv_rows(path: Path) -> Iterator[list[str]]:
    """Yield raw rows from ``path``, skipping the header line.

    Uses an explicit ``try/finally`` so the file handle closes
    deterministically even if the consumer raises mid-iteration. The
    functionally-equivalent ``with`` form would rely on generator
    finalization (``.close()`` on GC) for that case, which is correct in
    CPython but not guaranteed across Python implementations — Codex
    P2.B.1 on PR #62 flagged the GC dependency as advisory; this is the
    explicit version. See issue #63.
    """
    fh = path.open("r", encoding="utf-8", newline="")
    try:
        reader = csv.reader(fh)
        next(reader, None)  # skip header
        yield from reader
    finally:
        fh.close()


# Per-column cap on the number of distinct example warning messages
# logged after a drain. Aggregating by ``(column, count)`` keeps the log
# bounded even when 16.9M rows each carry a unique warning message; the
# examples give an operator enough surface to diagnose the failure
# pattern without flooding the audit.
_PARSE_WARNING_EXAMPLES_PER_COLUMN: int = 3


def _drain_normalize_rows(
    table: CSVTable,
    raw_header: list[str],
    kept_header: list[str],
    csv_path: Path,
) -> None:
    """Stream all rows of ``csv_path`` through :func:`normalize_rows`,
    discard the output, and log aggregate stats.

    Phase 1 has no Parquet writer yet; the drain exists to (a) verify the
    row-level pipeline runs end-to-end on the real bundle without
    crashing, (b) surface year-filter drop counts to the run audit, and
    (c) emit per-row parse warnings (e.g., unparseable IPTSUMOPRT INDATE
    values or ragged rows) so an operator can spot regressions in the
    export shape.

    Parse warnings are aggregated by column and logged at WARNING level
    after the drain: one line per column with a count plus up to
    :data:`_PARSE_WARNING_EXAMPLES_PER_COLUMN` example messages. This
    keeps the log bounded for high-cardinality unique messages while
    preserving enough detail to identify which column / parser changed.

    The next ticket replaces "discard" with the Parquet write.
    """
    rows_in = 0

    def counting_input() -> Iterator[list[str]]:
        # ``rows_in`` accuracy depends on the outer drain loop exhausting
        # ``normalize_rows()`` to completion. If a future caller short-
        # circuits the iteration (e.g., a row-limit kwarg), this counter
        # undercounts silently — wrap the limiter with its own explicit
        # accounting rather than relying on the closure here. See
        # issue #63 (Codex P2.B.2).
        nonlocal rows_in
        for row in _stream_csv_rows(csv_path):
            rows_in += 1
            yield row

    rows_kept = 0
    rows_with_warnings = 0
    warning_counts: Counter[str] = Counter()
    warning_examples: dict[str, list[str]] = {}

    for normalized in normalize_rows(table, raw_header, kept_header, counting_input()):
        rows_kept += 1
        if normalized.parse_warnings:
            rows_with_warnings += 1
            for col, msg in normalized.parse_warnings:
                warning_counts[col] += 1
                examples = warning_examples.setdefault(col, [])
                if len(examples) < _PARSE_WARNING_EXAMPLES_PER_COLUMN:
                    examples.append(msg)

    rows_filtered = rows_in - rows_kept
    if rows_in > 0:
        logger.info(
            "normalize: table=%s rows_in=%d rows_kept=%d rows_filtered=%d "
            "rows_with_warnings=%d",
            table,
            rows_in,
            rows_kept,
            rows_filtered,
            rows_with_warnings,
        )
        for col, count in warning_counts.most_common():
            logger.warning(
                "normalize: table=%s parse_warning column=%s count=%d examples=%s",
                table,
                col,
                count,
                warning_examples[col],
            )


@dataclass(frozen=True, slots=True)
class _TableRef:
    """Internal: state carried from the discovery/header phase of
    :func:`ingest` to the row-drain phase. Built once per CSV in phase 1
    so phase 2 can iterate without re-reading the header or recomputing
    the normalized projection. Not part of the public surface — exists
    purely as a structured tuple for the two-phase loop introduced in
    issue #64 (idempotency-first reorder)."""

    table: CSVTable
    raw_header: list[str]
    kept_header: list[str]
    csv_path: Path


def ingest(config: IngestConfig) -> IngestResult:
    """Ingest the configured CSV directory.

    Two-phase contract (issue #64): an already-complete re-run (matching
    ``run_id`` marker on disk) returns ``skipped_idempotent=True`` without
    streaming a single row through the normalize pipeline. The drain
    phase only runs when the identity check says the work is genuinely
    new.

    * Phase 1 — discover every CSV, validate its header against the
      pandera schema, compute its ``content_hash``. Raises before any
      side-effect on schema drift or missing tables. Side-effect-free
      apart from logging the per-file dropped-column summary, so the
      ``IncompleteInputError`` / ``SchemaDriftError`` paths never write
      a completion marker.
    * Identity gate — compute :class:`RunIdentity` from the phase-1
      hashes + schema fingerprint + code version. If the identity reports
      itself complete on disk, return ``skipped_idempotent=True``
      immediately. **The drain phase is skipped entirely on this path.**
    * Phase 2 — only on a non-idempotent run, stream each CSV's rows
      through :func:`_drain_normalize_rows` to surface row-level audit
      events (filter counts, parse warnings). Phase 1's per-file state
      is replayed from the cached :class:`_TableRef`s; no header re-read.

    On the production-sized bundle (2.1 GB IPDNRFOCUSDT), an idempotent
    re-run previously scanned every row before reaching the
    ``is_complete()`` check. The new order saves that scan on retries.
    """
    known_tables: tuple[CSVTable, ...] = cast(
        "tuple[CSVTable, ...]", get_args(CSVTable)
    )

    if not config.input_dir.exists() or not config.input_dir.is_dir():
        raise IncompleteInputError(
            f"input_dir {str(config.input_dir)!r} is missing or not a directory; "
            f"a complete HOSxP export of {len(known_tables)} CSVs is required"
        )

    # Phase 1: discovery + header validation + content hashing.
    per_file_hashes: dict[CSVTable, str] = {}
    table_refs: list[_TableRef] = []

    for csv_path in sorted(config.input_dir.glob("*.csv")):
        stem = csv_path.stem
        if stem not in known_tables:
            # Unknown filename — skip silently; the canonical set is exactly
            # the 13 HOSxP tables and operators may stage extra artefacts.
            continue
        table = stem
        raw_header = _read_csv_header(csv_path)
        normalized = normalize_header(table, raw_header)
        if normalized.dropped:
            # Policy (a) per docs/ingest-mapping.md: project + log dropped.
            # An operator reviewing the run audit can diff the dropped set
            # across runs to notice newly-arrived columns the schema doesn't
            # yet declare.
            logger.info(
                "normalize: table=%s dropped %d columns: %s",
                table,
                len(normalized.dropped),
                sorted(set(normalized.dropped)),
            )
        validate_header(table, normalized.header)
        per_file_hashes[table] = content_hash(csv_path)
        table_refs.append(
            _TableRef(
                table=table,
                raw_header=raw_header,
                kept_header=normalized.header,
                csv_path=csv_path,
            )
        )

    validated = [ref.table for ref in table_refs]
    missing_tables = sorted(set(known_tables) - set(validated))
    if missing_tables:
        raise IncompleteInputError(
            f"input_dir is missing {len(missing_tables)} of {len(known_tables)} "
            f"required HOSxP tables: {missing_tables}"
        )

    # Identity gate: short-circuit before any row scan on idempotent re-runs.
    identity = RunIdentity.from_inputs(
        per_file_hashes, schema_fingerprint(), config.code_version
    )
    tables_written = tuple(validated)

    if identity.is_complete(config.output_dir):
        return IngestResult(
            run_id=identity.run_id,
            rows_written=0,
            tables_written=tables_written,
            skipped_idempotent=True,
        )

    # Phase 2: row drain — only reached when the run is genuinely new.
    # Per-row Parquet writes land here in subsequent tickets (#4–#7). For
    # now the drain surfaces filter counts + parse warnings to the run
    # audit and the idempotency marker stamps the completion.
    for ref in table_refs:
        _drain_normalize_rows(ref.table, ref.raw_header, ref.kept_header, ref.csv_path)

    identity.mark_complete(config.output_dir)

    return IngestResult(
        run_id=identity.run_id,
        rows_written=0,
        tables_written=tables_written,
        skipped_idempotent=False,
    )
