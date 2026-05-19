"""End-to-end ingest orchestration: CSV → header validation → run-identity → noop-or-mark.

Public entry point: :func:`ingest`. Per PRD §1:

* discover the 11 HOSxP CSVs in ``config.input_dir`` by file stem (per the
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
from pathlib import Path
from typing import cast, get_args

from bba.ingest.hashing import content_hash
from bba.ingest.models import CSVTable, IngestConfig, IngestResult
from bba.ingest.normalize import normalize_header
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


def ingest(config: IngestConfig) -> IngestResult:
    """Ingest the configured CSV directory.

    See module docstring for the contract. Raises before any side-effect on
    drift or incomplete input — so a malformed input dir never writes a
    completion marker.
    """
    known_tables: tuple[CSVTable, ...] = cast(
        "tuple[CSVTable, ...]", get_args(CSVTable)
    )

    if not config.input_dir.exists() or not config.input_dir.is_dir():
        raise IncompleteInputError(
            f"input_dir {str(config.input_dir)!r} is missing or not a directory; "
            f"a complete HOSxP export of {len(known_tables)} CSVs is required"
        )

    per_file_hashes: dict[CSVTable, str] = {}
    validated: list[CSVTable] = []

    for csv_path in sorted(config.input_dir.glob("*.csv")):
        stem = csv_path.stem
        if stem not in known_tables:
            # Unknown filename — skip silently; the canonical set is exactly
            # the 11 HOSxP tables and operators may stage extra artefacts.
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
        validated.append(table)

    missing_tables = sorted(set(known_tables) - set(validated))
    if missing_tables:
        raise IncompleteInputError(
            f"input_dir is missing {len(missing_tables)} of {len(known_tables)} "
            f"required HOSxP tables: {missing_tables}"
        )

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

    # Per-row Parquet writes land here in subsequent tickets (#4–#7). For now
    # the idempotency boundary is the marker file — once written, this run_id
    # is considered complete.
    identity.mark_complete(config.output_dir)

    return IngestResult(
        run_id=identity.run_id,
        rows_written=0,
        tables_written=tables_written,
        skipped_idempotent=False,
    )
