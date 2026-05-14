"""Pydantic v2 models for the ingest module — inputs, outputs, parse results.

All models are immutable (frozen=True). Per PRD §1, the strict time parser
NEVER silently shifts: an unrecognized format must yield ``ParseResult(value=None,
parse_warning=<reason>)``, never a wrong-but-plausible datetime.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

CSVTable = Literal[
    "BDVST",
    "BDVSTDT",
    "BDTYPE",
    "BDVSTST",
    "Diagnosis",
    "Lab",
    "MED",
    "IPDADMPROGRESS",
    "IPDNRFOCUSDT",
    "UnUSE_Patient_Background",
]


class ParseResult(BaseModel):
    """Result of strict HOSxP time/datetime parsing.

    Invariant: exactly one of ``value`` and ``parse_warning`` is non-None.
    Unrecognized inputs produce ``value=None, parse_warning="…"`` — never a
    silently shifted datetime.
    """

    model_config = ConfigDict(frozen=True)

    value: datetime | None
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
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    rows_written: int
    tables_written: list[CSVTable]
    skipped_idempotent: bool
