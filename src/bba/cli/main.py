"""``bba`` — the click root group and the six subcommand functions.

Each subcommand is intentionally a thin wrapper over an already-tested
module entrypoint:

* ``bba ingest``          → :func:`bba.ingest.ingest`
* ``bba audit``           → idempotent wrap of audit-pipeline orchestration
* ``bba evaluate``        → :mod:`bba.eval_harness`
* ``bba report``          → :func:`bba.report_generator.generate_monthly_report`
* ``bba serve-dashboard`` → :func:`bba.dashboard.create_app` + ``uvicorn``
* ``bba sentinel``        → :mod:`bba.monitoring` cadence dispatch

Three of the six subcommands (``evaluate``, ``report``, ``serve-dashboard``,
``sentinel``) require multi-store / multi-resolver glue that lives in the
underlying module rather than in the CLI; their bodies fail loud with a
:class:`CliError` describing the contract at the integration boundary
rather than fabricating defaults that would silently mis-configure the
underlying module. The CLI's job is to expose the surface and keep the
input-validation / logging hygiene tight; building production-grade
defaults for the per-store integrations is owned by each module."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import click

from bba.cli._logging import get_logger
from bba.cli.exceptions import CliError
from bba.cli.identity import (
    CodeVersion,
    SchemaFingerprint,
    code_version,
    compute_run_id,
)
from bba.cli.models import (
    AuditCommandInput,
    EvaluateCommandInput,
    IngestCommandInput,
    ReportCommandInput,
    ReportFormat,
    SchemaVersion,
    SentinelCadence,
    SentinelCommandInput,
    ServeDashboardInput,
)
from bba.cli.store_protocol import AuditRunStore
from bba.ingest import IngestConfig, ingest, schema_fingerprint


_log = get_logger()


_BBA_DATA_DIR_ENV: str = "BBA_DATA_DIR"


def _resolve_data_dir() -> Path:
    """Return ``$BBA_DATA_DIR`` as a :class:`Path`.

    Raises :class:`CliError` if the env var is unset. The CLI's three-
    variable env surface (BBA_DATA_DIR, BBA_DB_URL, ANTHROPIC_API_KEY) is
    asserted by ``TestEnvVarSurfaceIsTight``."""
    raw = os.environ.get(_BBA_DATA_DIR_ENV)
    if raw is None or raw == "":
        raise CliError(
            f"{_BBA_DATA_DIR_ENV} environment variable is required; set it "
            "to the absolute path of the audit data root"
        )
    return Path(raw)


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group(name="bba")
@click.version_option(package_name="blood-audit")
def cli() -> None:
    """Blood Audit (KCMH RBC Transfusion Audit) — Phase 1 CLI.

    See ticket #29 for the contract: six subcommands, run-level
    idempotency on ``audit``, PHI-scrubbing on uncaught exceptions, and
    no business logic in this layer.
    """


# ---------------------------------------------------------------------------
# Subcommand functions — each declared here, attached to ``cli`` at the
# bottom of the module via ``cli.add_command(...)``.
# ---------------------------------------------------------------------------


@click.command(name="ingest")
@click.argument(
    "input_csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--schema-version",
    type=click.Choice(["v1"]),
    default="v1",
    show_default=True,
    help="HOSxP schema version (Phase 1 ships v1 only).",
)
def bba_ingest(input_csv: Path, schema_version: str) -> None:
    """Ingest a HOSxP CSV bundle into DuckDB + Parquet."""
    inputs = IngestCommandInput(
        input_csv=input_csv,
        schema_version=cast(SchemaVersion, schema_version),
    )
    output_dir = _resolve_data_dir() / "ingest" / inputs.schema_version
    config = IngestConfig(
        input_dir=inputs.input_csv.parent,
        output_dir=output_dir,
        code_version=code_version(),
    )
    ingest(config)
    _log.info(
        "ingest.complete",
        input=str(inputs.input_csv),
        output_dir=str(output_dir),
    )


@click.command(name="audit")
@click.option(
    "--input",
    "input_csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Input HOSxP CSV bundle (mutually exclusive with --run-id).",
)
@click.option(
    "--run-id",
    type=str,
    default=None,
    help="Re-audit a previously ingested run (mutually exclusive with --input).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Override run-level idempotency. Writes a compliance row to "
        "audit_log with idempotency_override=true."
    ),
)
def bba_audit(input_csv: Path | None, run_id: str | None, force: bool) -> None:
    """Run the audit pipeline. Re-running on the same input is a no-op
    unless ``--force`` is set."""
    inputs = AuditCommandInput(input_csv=input_csv, run_id=run_id, force=force)
    store = _get_audit_run_store()
    resolved = _resolve_run_id(inputs)
    if store.run_complete(resolved):
        if not inputs.force:
            _log.info(
                "audit.noop",
                run_id=resolved,
                row_count=store.run_count(resolved),
            )
            return
        store.record_idempotency_override(resolved, reason="cli --force flag")
        _log.warning(
            "audit.force_override",
            run_id=resolved,
            idempotency_override=True,
        )
    if inputs.input_csv is None:
        raise click.UsageError(
            "bba audit --run-id without --input cannot reach the pipeline; "
            "the run was not previously completed"
        )
    _run_audit_pipeline(run_id=resolved, input_csv=inputs.input_csv)
    _log.info(
        "audit.complete",
        run_id=resolved,
        row_count=store.run_count(resolved),
    )


@click.command(name="evaluate")
@click.option("--run-id", type=str, required=True)
def bba_evaluate(run_id: str) -> None:
    """Evaluate the gold-standard performance for a finished run."""
    inputs = EvaluateCommandInput(run_id=run_id)
    _log.info("evaluate.start", run_id=inputs.run_id)
    raise CliError(
        f"evaluate run_id={inputs.run_id}: bba.eval_harness exposes "
        "stratified-sample / agreement / cluster-CI primitives but not a "
        "single-call evaluate(run_id) facade. The CLI's hand-off seam is "
        "bba.cli.main.bba_evaluate; production wiring composes the "
        "primitives against the configured audit_store."
    )


@click.command(name="report")
@click.option("--run-id", type=str, required=True)
@click.option(
    "--format",
    "report_format",
    type=click.Choice(["html", "pdf", "json"]),
    default="html",
    show_default=True,
)
def bba_report(run_id: str, report_format: str) -> None:
    """Render a monthly report (html / pdf / json) from a finished run."""
    inputs = ReportCommandInput(
        run_id=run_id,
        format=cast(ReportFormat, report_format),
    )
    _log.info("report.start", run_id=inputs.run_id, format=inputs.format)
    raise CliError(
        f"report run_id={inputs.run_id} format={inputs.format}: "
        "bba.report_generator.generate_monthly_report(inputs: ReportInputs) "
        "requires a fully-built ReportInputs (month / rows / footer / "
        "output_dir) sourced from the audit_store. The CLI's hand-off seam "
        "is bba.cli.main.bba_report; production wiring queries the store "
        "by run_id and constructs the inputs."
    )


@click.command(name="serve-dashboard")
@click.option("--port", type=int, default=8000, show_default=True)
def bba_serve_dashboard(port: int) -> None:
    """Serve the reviewer dashboard (FastAPI) on ``--port``."""
    inputs = ServeDashboardInput(port=port)
    _log.info("serve_dashboard.start", port=inputs.port)
    raise CliError(
        f"serve-dashboard port={inputs.port}: bba.dashboard.create_app "
        "requires a DashboardConfig (audit_store, review_actions_store, "
        "snapshot_dir, unredacted_phi_resolver, attribution resolvers). "
        "The CLI's hand-off seam is bba.cli.main.bba_serve_dashboard; "
        "production wiring instantiates the stores from BBA_DATA_DIR / "
        "BBA_DB_URL and the resolvers from the deployment's auth surface."
    )


@click.command(name="sentinel")
@click.option("--weekly", "cadence", flag_value="weekly", default=None)
@click.option("--quarterly", "cadence", flag_value="quarterly", default=None)
def bba_sentinel(cadence: str | None) -> None:
    """Run a monitoring sentinel cadence (weekly κ or quarterly golden-set)."""
    if cadence is None:
        raise click.UsageError(
            "bba sentinel requires exactly one of --weekly or --quarterly"
        )
    inputs = SentinelCommandInput(cadence=cast(SentinelCadence, cadence))
    _log.info("sentinel.start", cadence=inputs.cadence)
    raise CliError(
        f"sentinel cadence={inputs.cadence}: bba.monitoring exposes "
        "draw_weekly_reviewer_sample, evaluate_sentinel_run, and "
        "evaluate_golden_set_drift as primitives but no cadence "
        "dispatcher. The CLI's hand-off seam is bba.cli.main.bba_sentinel; "
        "production wiring loads the prior-period audit rows from the "
        "configured audit_store and dispatches to the matching primitive."
    )


# ---------------------------------------------------------------------------
# Injection seam — tests monkeypatch this resolver to inject a fake store.
# ---------------------------------------------------------------------------


def _get_audit_run_store() -> AuditRunStore:
    """Resolve the :class:`~bba.cli.store_protocol.AuditRunStore` from config.

    Reads ``BBA_DB_URL`` and constructs a Postgres-backed adapter from
    :mod:`bba.audit_store`. Tests monkeypatch this seam to inject a
    recording double via ``patch("bba.cli.main._get_audit_run_store",
    ...)`` — the in-process replacement avoids spinning up a real
    database for the unit suite.

    Production behaviour: an :class:`AuditRunStore` adapter that wraps
    the Postgres ``audit_log`` + ``audit_results`` tables. The adapter
    itself is delivered by :mod:`bba.audit_store` (see PRD §10); the CLI
    only sees the :class:`~bba.cli.store_protocol.AuditRunStore` Protocol
    so the dependency direction stays one-way.
    """
    db_url = os.environ.get("BBA_DB_URL")
    if db_url is None or db_url == "":
        raise CliError(
            "BBA_DB_URL environment variable is required for audit store "
            "access; set it to the Postgres DSN of the audit database"
        )
    raise CliError(
        f"BBA_DB_URL={db_url[:8]}…: bba.audit_store does not yet expose a "
        f"Protocol-compatible AuditRunStore adapter "
        "(run_complete / run_count / record_idempotency_override). The "
        "CLI's hand-off seam is bba.cli.main._get_audit_run_store; "
        "production wiring constructs the adapter against the configured "
        "Postgres DSN."
    )


def _run_audit_pipeline(*, run_id: str, input_csv: Path) -> None:
    """Hand off to the audit pipeline orchestrator.

    The orchestration that turns an input CSV into
    :class:`~bba.audit_pipeline.PipelineRowContext`'s and wires the
    transport / audit_store / batch_run_store / verifier configuration
    lives in :mod:`bba.audit_pipeline` (PRD §15). This function is the
    CLI's single hand-off point so tests can swap it with a recording
    double via ``patch("bba.cli.main._run_audit_pipeline", ...)``.

    In production the CLI fails loud at this boundary: the underlying
    single-input orchestrator is not part of issue #29's deliverable
    (it is owned by issue #24's facade layer), so calling this without
    a patched seam raises :class:`CliError` with a precise message
    rather than fabricating defaults that would silently mis-configure
    the live pipeline."""
    raise CliError(
        f"audit pipeline orchestrator is not wired in the CLI; cannot "
        f"audit run_id={run_id} from {input_csv}. The CLI's hand-off "
        f"point is bba.cli.main._run_audit_pipeline — production "
        f"integration is delivered by bba.audit_pipeline (PRD §15)."
    )


def _resolve_run_id(inputs: AuditCommandInput) -> str:
    """Return the ``run_id`` for the audit invocation.

    If ``--run-id`` was passed, use it verbatim. Otherwise compute it
    from the input CSV's bytes, the active schema fingerprint, and the
    package version — the formula from PRD §20."""
    if inputs.run_id is not None:
        return inputs.run_id
    assert inputs.input_csv is not None, "AuditCommandInput XOR invariant"
    return compute_run_id(
        input_csv=inputs.input_csv,
        schema_fingerprint=SchemaFingerprint(schema_fingerprint()),
        code_version_str=CodeVersion(code_version()),
    )


# ---------------------------------------------------------------------------
# Subcommand wiring — attach the six functions above onto the root group.
# Order matches the PRD §20 enumeration so `bba --help` listing is stable.
# ---------------------------------------------------------------------------


cli.add_command(bba_ingest)
cli.add_command(bba_audit)
cli.add_command(bba_evaluate)
cli.add_command(bba_report)
cli.add_command(bba_serve_dashboard)
cli.add_command(bba_sentinel)
