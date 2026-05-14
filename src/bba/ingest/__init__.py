"""bba.ingest — HOSxP CSV → DuckDB+Parquet ingestion.

See issue #3 for acceptance criteria. Implementation Decisions §1 in the PRD
defines the strict time parser, schema-drift detection, tz normalization, and
run_id idempotency contract.

This module is the foundation for #4, #5, #6, #7, #9, #12.
"""

from bba.ingest.hashing import content_hash
from bba.ingest.models import (
    CSVTable,
    IngestConfig,
    IngestResult,
    ParsedTimeOfDay,
    ParseResult,
)
from bba.ingest.pipeline import ingest
from bba.ingest.row_timestamp import RowTimestamp
from bba.ingest.run_identity import RunIdentity
from bba.ingest.schemas import (
    IncompleteInputError,
    SchemaDriftError,
    all_tables,
    get_schema,
    schema_fingerprint,
    validate_header,
)
from bba.ingest.time_parser import parse_hosxp_time

__all__ = [
    "CSVTable",
    "IncompleteInputError",
    "IngestConfig",
    "IngestResult",
    "ParseResult",
    "ParsedTimeOfDay",
    "RowTimestamp",
    "RunIdentity",
    "SchemaDriftError",
    "all_tables",
    "content_hash",
    "get_schema",
    "ingest",
    "parse_hosxp_time",
    "schema_fingerprint",
    "validate_header",
]
