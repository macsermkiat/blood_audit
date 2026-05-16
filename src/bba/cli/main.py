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

from bba.cli._logging import get_logger
from bba.cli.identity import code_version, compute_run_id
from bba.cli.models import AuditCommandInput
from bba.cli.store_protocol import AuditRunStore
from bba.ingest import schema_fingerprint


_log = get_logger()


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
    without monkeypatching the whole click command.

    The body delegates straight to :func:`bba.audit_pipeline.run_pipeline`;
    the CLI deliberately holds no business logic past this hand-off.
    """
    from bba.audit_pipeline import run_pipeline

    run_pipeline(run_id=run_id, input_csv=input_csv)


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
        schema_fingerprint=schema_fingerprint(),
        code_version_str=code_version(),
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
