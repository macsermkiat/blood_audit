"""``bba`` — the click root group and the six subcommand functions.

Each subcommand is intentionally a thin (≤20-line) wrapper over an
already-tested module entrypoint:

* ``bba ingest``         → :func:`bba.ingest.ingest`
* ``bba audit``          → :func:`bba.audit_pipeline.run_pipeline`
                            (with run-level idempotency around it)
* ``bba evaluate``       → :mod:`bba.eval_harness` (a single
                            ``evaluate(run_id, ...)`` facade to land in
                            GREEN; eval_harness already exports the
                            sub-primitives)
* ``bba report``         → :func:`bba.report_generator.generate_monthly_report`
* ``bba serve-dashboard``→ :func:`bba.dashboard.create_app`
                            served via ``uvicorn``
* ``bba sentinel``       → :mod:`bba.monitoring` cadence dispatcher

GREEN phase wires :func:`cli.add_command` for each function below. RED
phase leaves them detached so ``bba --help`` lists *no* subcommands and
the behavioural tests fail.
"""

from __future__ import annotations

from pathlib import Path

import click

from bba.cli.store_protocol import AuditRunStore


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
# Subcommand functions — declared here, attached by GREEN phase.
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
    raise NotImplementedError("bba ingest — GREEN phase wires bba.ingest.ingest")


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
    raise NotImplementedError(
        "bba audit — GREEN phase wires compute_run_id + audit_pipeline.run_pipeline"
    )


@click.command(name="evaluate")
@click.option("--run-id", type=str, required=True)
def bba_evaluate(run_id: str) -> None:
    """Evaluate the gold-standard performance for a finished run."""
    raise NotImplementedError("bba evaluate — GREEN phase wires bba.eval_harness")


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
    raise NotImplementedError(
        "bba report — GREEN phase wires bba.report_generator.generate_monthly_report"
    )


@click.command(name="serve-dashboard")
@click.option("--port", type=int, default=8000, show_default=True)
def bba_serve_dashboard(port: int) -> None:
    """Serve the reviewer dashboard (FastAPI) on ``--port``."""
    raise NotImplementedError(
        "bba serve-dashboard — GREEN phase wires bba.dashboard.create_app + uvicorn"
    )


@click.command(name="sentinel")
@click.option("--weekly", "cadence", flag_value="weekly", default=None)
@click.option("--quarterly", "cadence", flag_value="quarterly", default=None)
def bba_sentinel(cadence: str | None) -> None:
    """Run a monitoring sentinel cadence (weekly κ or quarterly golden-set)."""
    raise NotImplementedError(
        "bba sentinel — GREEN phase wires bba.monitoring cadence dispatch"
    )


# ---------------------------------------------------------------------------
# Injection seam — tests monkeypatch this resolver to inject a fake store.
# ---------------------------------------------------------------------------


def _get_audit_run_store() -> AuditRunStore:
    """Resolve the :class:`~bba.cli.store_protocol.AuditRunStore` from config.

    GREEN-phase implementation reads ``BBA_DB_URL`` and constructs a
    Postgres-backed adapter from :mod:`bba.audit_store`. Until then this
    is a NotImplementedError stub; tests inject a double via
    ``unittest.mock.patch``.
    """
    raise NotImplementedError(
        "_get_audit_run_store — GREEN phase wires bba.audit_store adapter"
    )


def _run_audit_pipeline(*, run_id: str, input_csv: Path) -> None:
    """Hand off to :func:`bba.audit_pipeline.run_pipeline`.

    Separated from :func:`bba_audit` so tests can patch the pipeline call
    without monkeypatching the whole click command. GREEN-phase signature
    will likely grow a ``config`` argument; the CLI test suite cares only
    about call-count.
    """
    raise NotImplementedError(
        "_run_audit_pipeline — GREEN phase wires audit_pipeline.run_pipeline"
    )
