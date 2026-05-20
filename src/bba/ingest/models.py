"""Pydantic v2 models for the ingest module — inputs, outputs, parse results.

All models are immutable (frozen=True). Per PRD §1, the strict time parser
NEVER silently shifts: an unrecognized format must yield ``ParseResult(value=None,
parse_warning=<reason>)``, never a wrong-but-plausible datetime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

CSVTable = Literal[
    "BDVST",
    "BDVSTDT",
    "BDVSTST",
    "BDTYPE",
    "Diagnosis",
    "Lab",
    "Med",
    "IPDADMPROGRESS",
    "IPDNRFOCUSDT",
    "IPTSUMOPRT",
    "INCPT",
    "ICD9CM",
]


@dataclass(frozen=True, slots=True)
class ParsedTimeOfDay:
    """A clock time-of-day (no date) parsed from a HOSxP time column.

    Time alone does not pin a moment. Callers MUST combine a ``ParsedTimeOfDay``
    with the row's date column via :class:`~bba.ingest.row_timestamp.RowTimestamp`
    before persisting; this type's purpose is to prevent the date from being
    invented at parse time (which previously meant a sentinel ``1900-01-01``
    that callers had to remember to ignore).
    """

    hour: int
    minute: int
    second: int


class ParseResult(BaseModel):
    """Result of strict HOSxP time parsing.

    Invariant: exactly one of ``value`` and ``parse_warning`` is non-None.
    Unrecognized inputs produce ``value=None, parse_warning="…"`` — never a
    silently shifted time. ``value`` is a :class:`ParsedTimeOfDay`, not a
    ``datetime`` — see that type's docstring for why.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    value: ParsedTimeOfDay | None
    parse_warning: str | None
    raw: str


class IngestConfig(BaseModel):
    """Top-level configuration for a single ingest run."""

    model_config = ConfigDict(frozen=True)

    input_dir: Path
    output_dir: Path
    code_version: str
    tz_source: str = "Asia/Bangkok"


class IngestResult(BaseModel):
    """Outcome of a single ingest run.

    ``skipped_idempotent=True`` indicates the writer detected an already-complete
    ``run_id`` and no-op'd per PRD §1 (run-level idempotency).

    ``tables_written`` is a tuple — ``frozen=True`` only prevents reassigning the
    field, not mutating a nested ``list``. A tuple makes the public output
    contract genuinely immutable.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    rows_written: int
    tables_written: tuple[CSVTable, ...]
    skipped_idempotent: bool
