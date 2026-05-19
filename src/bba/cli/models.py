"""Frozen Pydantic v2 models for the six ``bba`` subcommand inputs.

Every subcommand parses ``click`` options into one of these models before
delegating to the underlying module. The models exist so the CLI's input
surface has a stable, validatable contract independent of click's
positional argument order — and so a future programmatic caller (e.g. a
test harness) can construct inputs without going through ``argv``.

PRD §20 (Implementation Decisions) names the six entrypoints; this file
mirrors that list one-to-one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


SchemaVersion = Literal["v1"]
"""HOSxP schema version. Phase 1 ships v1 only — v2 is a #3-follow-up."""


ReportFormat = Literal["html", "pdf", "json"]
"""Supported ``bba report --format`` values."""


SentinelCadence = Literal["weekly", "quarterly"]
"""``bba sentinel`` cadence flag — exactly one must be selected."""


class _FrozenModel(BaseModel):
    """Shared base: frozen, ``extra='forbid'``, strict.

    Frozen so the CLI surface can pass instances through layers without
    worrying about late mutation. ``extra='forbid'`` so a typo in a kwarg
    is loud, not silent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=False)


class IngestCommandInput(_FrozenModel):
    """Inputs for ``bba ingest <input.csv> --schema-version vN``."""

    input_csv: Path
    schema_version: SchemaVersion = "v1"


class AuditCommandInput(_FrozenModel):
    """Inputs for ``bba audit (--input X | --run-id Y) [--force]``.

    Exactly one of ``input_csv`` / ``run_id`` must be supplied. The model
    rejects both-set and neither-set at validation time so the click layer
    never reaches the pipeline with an ambiguous request.
    """

    input_csv: Path | None = None
    run_id: str | None = None
    force: bool = False

    @model_validator(mode="after")
    def _xor_input_or_run_id(self) -> Self:
        provided = (self.input_csv is not None, self.run_id is not None)
        if provided == (False, False):
            raise ValueError("bba audit requires exactly one of --input or --run-id")
        if provided == (True, True):
            raise ValueError(
                "bba audit accepts only one of --input or --run-id, not both"
            )
        return self


class EvaluateCommandInput(_FrozenModel):
    """Inputs for ``bba evaluate --run-id ...``."""

    run_id: str = Field(min_length=1)


class ReportCommandInput(_FrozenModel):
    """Inputs for ``bba report --run-id ... --format html|pdf|json``."""

    run_id: str = Field(min_length=1)
    format: ReportFormat = "html"


class ServeDashboardInput(_FrozenModel):
    """Inputs for ``bba serve-dashboard --port N``."""

    port: int = Field(default=8000, ge=1, le=65535)


class SentinelCommandInput(_FrozenModel):
    """Inputs for ``bba sentinel --weekly|--quarterly``."""

    cadence: SentinelCadence
