"""End-to-end ingest orchestration: CSV → pandera validation → DuckDB+Parquet.

Public entry point: :func:`ingest`. Per PRD §1:

* discover the 10 HOSxP CSVs in ``config.input_dir`` by file stem;
* validate each header against its pandera schema, raising
  :class:`~bba.ingest.schemas.SchemaDriftError` loudly on any unknown column;
* derive ``run_id`` from the input content hashes + schema fingerprint +
  code version;
* short-circuit when the writer reports the ``run_id`` already complete
  (``skipped_idempotent=True``).

The Phase-1 implementation focuses on header validation and the idempotency
boundary; per-row Parquet writes are owned by subsequent tickets that consume
the validated dataframes.
"""

from __future__ import annotations

import hashlib
import csv
from pathlib import Path
from typing import cast, get_args

from bba.ingest.hashing import compute_run_id, content_hash
from bba.ingest.models import CSVTable, IngestConfig, IngestResult
from bba.ingest.schemas import (
    SchemaDriftError,
    get_schema,
    schema_fingerprint,
)
from bba.ingest.writer import is_run_complete, mark_run_complete


def _aggregate_input_hash(per_file_hashes: dict[CSVTable, str]) -> str:
    h = hashlib.sha256()
    for table in sorted(per_file_hashes):
        h.update(table.encode("utf-8"))
        h.update(b":")
        h.update(per_file_hashes[table].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            return next(reader)
        except StopIteration:
            return []


def ingest(config: IngestConfig) -> IngestResult:
    """Ingest the configured CSV directory into the Parquet store.

    See module docstring for the contract. On any header drift the function
    raises :class:`SchemaDriftError` before mutating ``output_dir``.
    """
    known_tables: tuple[CSVTable, ...] = cast("tuple[CSVTable, ...]", get_args(CSVTable))

    per_file_hashes: dict[CSVTable, str] = {}
    validated: list[CSVTable] = []

    for csv_path in sorted(config.input_dir.glob("*.csv")):
        stem = csv_path.stem
        if stem not in known_tables:
            # Unknown filename — skip silently; the canonical set is exactly
            # the 10 HOSxP tables and operators may stage extra artefacts.
            continue
        table = stem
        schema = get_schema(table)
        header = _read_csv_header(csv_path)
        unknown = [c for c in header if c not in schema.columns]
        if unknown:
            raise SchemaDriftError(
                f"schema drift in table {table!r}: unknown columns {unknown} "
                f"(declared columns: {sorted(schema.columns)})"
            )
        per_file_hashes[table] = content_hash(csv_path)
        validated.append(table)

    input_csv_hash = _aggregate_input_hash(per_file_hashes)
    run_id = compute_run_id(input_csv_hash, schema_fingerprint(), config.code_version)

    if is_run_complete(config.output_dir, run_id):
        return IngestResult(
            run_id=run_id,
            rows_written=0,
            tables_written=validated,
            skipped_idempotent=True,
        )

    # Per-row Parquet writes land here in subsequent tickets (#4–#7). For now
    # the idempotency boundary is the marker file — once written, this run_id
    # is considered complete.
    mark_run_complete(config.output_dir, run_id)

    return IngestResult(
        run_id=run_id,
        rows_written=0,
        tables_written=validated,
        skipped_idempotent=False,
    )
