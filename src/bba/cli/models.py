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

import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


SchemaVersion = Literal["v1"]
"""HOSxP schema version. Phase 1 ships v1 only — v2 is a #3-follow-up."""


ReportFormat = Literal["html", "pdf", "json"]
"""Supported ``bba report --format`` values."""


SentinelCadence = Literal["weekly", "quarterly"]
"""``bba sentinel`` cadence flag — exactly one must be selected."""


_SAFE_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_safe_run_id(value: str) -> str:
    """Reject ``run_id`` values that would be unsafe to interpolate into
    a filesystem path component.

    The CLI composes ``$BBA_DATA_DIR/reports/<run_id>`` (and analogous
    paths) by raw interpolation. An upstream string containing ``/``,
    ``\\``, or a path-traversal segment would let writes escape the
    intended dataset directory. Mirrors the defense at
    :data:`bba.audit_store.models.SafeId` /
    :data:`bba.report_generator.models.SafeFsId`; defined locally so
    :mod:`bba.cli.models` does not transitively import either heavy
    dependency (pyarrow / duckdb).

    Allow-list: non-empty, ``[A-Za-z0-9._-]+``, not exactly ``.`` or
    ``..``. The 16-char hex ``run_id`` produced by :func:`compute_run_id`
    satisfies this pattern, so production runs are unaffected."""
    if not value:
        raise ValueError("run_id must not be empty")
    if not _SAFE_RUN_ID_PATTERN.match(value):
        raise ValueError(
            f"run_id must match [A-Za-z0-9._-]+ to be a safe filesystem "
            f"path component (got {value!r}); path-traversal segments "
            "and special characters are rejected so report artifacts "
            "stay inside BBA_DATA_DIR/reports"
        )
    if value in {".", ".."}:
        raise ValueError(f"run_id must not be a path-traversal segment (got {value!r})")
    return value


SafeRunId = Annotated[str, AfterValidator(_validate_safe_run_id)]
"""A ``str`` constrained to a filesystem-safe ``run_id`` shape.

Use on every CLI-input ``run_id`` field that flows into a filesystem
path. Adopted by :class:`ReportCommandInput`; symmetric tightening for
:class:`AuditCommandInput` / :class:`EvaluateCommandInput` is a
follow-up tracked outside this PR (those flows also interpolate
``run_id`` into paths but were not flagged by the Codex review on
PR #71)."""


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
    """Inputs for ``bba report --run-id ... --format html|pdf|json``.

    ``run_id`` is constrained to :data:`SafeRunId` because the CLI
    interpolates it into ``$BBA_DATA_DIR/reports/<run_id>``; a value
    like ``../../tmp/pwn`` would otherwise let report artifacts escape
    the data directory (Codex P1 review on PR #71)."""

    run_id: SafeRunId
    format: ReportFormat = "html"


class ServeDashboardInput(_FrozenModel):
    """Inputs for ``bba serve-dashboard --port N``."""

    port: int = Field(default=8000, ge=1, le=65535)


class SentinelCommandInput(_FrozenModel):
    """Inputs for ``bba sentinel --weekly|--quarterly``."""

    cadence: SentinelCadence
