"""bba.ingest — HOSxP CSV → DuckDB+Parquet ingestion.

See issue #3 for acceptance criteria. Implementation Decisions §1 in the PRD
defines the strict time parser, schema-drift detection, tz normalization, and
run_id idempotency contract.

This module is the foundation for #4, #5, #6, #7, #9, #12.
"""

from bba.ingest.hashing import compute_run_id, content_hash
from bba.ingest.models import (
    CSVTable,
    IngestConfig,
    IngestResult,
    ParseResult,
)
from bba.ingest.pipeline import ingest
from bba.ingest.schemas import (
    IncompleteInputError,
    SchemaDriftError,
    all_tables,
    get_schema,
    schema_fingerprint,
)
from bba.ingest.time_parser import parse_hosxp_time
from bba.ingest.tz import to_utc
from bba.ingest.writer import is_run_complete

__all__ = [
    "CSVTable",
    "IncompleteInputError",
    "IngestConfig",
    "IngestResult",
    "ParseResult",
    "SchemaDriftError",
    "all_tables",
    "compute_run_id",
    "content_hash",
    "get_schema",
    "ingest",
    "is_run_complete",
    "parse_hosxp_time",
    "schema_fingerprint",
    "to_utc",
]
