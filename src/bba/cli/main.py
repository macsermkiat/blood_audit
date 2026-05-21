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
from bba.cli.audit_run_store import FileBackedAuditRunStore
from bba.cli.exceptions import CliError
from bba.cli.phi_scrubber import install_excepthook
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
from bba.ingest import IngestConfig, IngestResult, ingest, schema_fingerprint
from bba.audit_store import AuditStore, AuditStoreConfig
from bba.report_generator import (
    PhysicianAttributionResolver,
    WardAttributionResolver,
    build_report_inputs,
    generate_monthly_report,
)


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

    Process-level exception scrubbing is installed before any subcommand
    body runs so a crash anywhere downstream produces a redacted
    traceback instead of leaking PHI to the operator log. The optional
    faulthandler sidecar is enabled when ``BBA_DATA_DIR`` is set so a
    hard interpreter crash (e.g. SIGSEGV in a C extension) still leaves
    a scrubbable stack trace on disk.
    """
    raw_data_dir = os.environ.get(_BBA_DATA_DIR_ENV)
    sidecar: Path | None = None
    if raw_data_dir:
        sidecar = Path(raw_data_dir) / "logs" / "faulthandler.sidecar"
    install_excepthook(faulthandler_sidecar=sidecar)


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
    unless ``--force`` is set.

    The check-then-act sequence (``run_complete`` → run pipeline →
    ``mark_run_complete``) is wrapped in ``store.acquire_run_lock``,
    which serialises invocations on the same ``run_id`` so two
    concurrent CLI processes cannot both pass the guard and execute
    the pipeline twice. Concurrent audits of *different* inputs run in
    parallel."""
    inputs = AuditCommandInput(input_csv=input_csv, run_id=run_id, force=force)
    store = _get_audit_run_store()
    resolved = _resolve_run_id(inputs)
    with store.acquire_run_lock(resolved):
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
                "bba audit --run-id without --input cannot reach the "
                "pipeline; the run was not previously completed"
            )
        _run_audit_pipeline(run_id=resolved, input_csv=inputs.input_csv, store=store)
        store.mark_run_complete(resolved)
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
    """Render a monthly report (html / pdf / json) from a finished run.

    The on-disk artifacts (committee CSVs + per-physician CSVs + PDF)
    are produced unconditionally; ``--format`` is captured as run
    metadata in the structured log and is reserved for a future
    stdout-shaping pass (see ``bba.cli.main.bba_report`` follow-up)."""
    inputs = ReportCommandInput(
        run_id=run_id,
        format=cast(ReportFormat, report_format),
    )
    _log.info("report.start", run_id=inputs.run_id, format=inputs.format)
    output_dir = _resolve_data_dir() / "reports" / inputs.run_id
    report_inputs = build_report_inputs(
        run_id=inputs.run_id,
        audit_store=_get_audit_store(),
        output_dir=output_dir,
        ward_resolver=_get_ward_resolver(),
        physician_resolver=_get_physician_resolver(),
    )
    artifacts = generate_monthly_report(report_inputs)
    _log.info(
        "report.complete",
        run_id=inputs.run_id,
        format=inputs.format,
        pdf_path=str(artifacts.pdf_path),
        committee_csv_count=len(artifacts.csv_paths),
        physician_csv_count=len(artifacts.physician_own_view_csv_paths),
    )
    click.echo(str(artifacts.pdf_path))


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
@click.option("--weekly", is_flag=True, default=False)
@click.option("--quarterly", is_flag=True, default=False)
def bba_sentinel(weekly: bool, quarterly: bool) -> None:
    """Run a monitoring sentinel cadence (weekly κ or quarterly golden-set).

    ``--weekly`` and ``--quarterly`` are mutually exclusive — supplying
    both is a usage error (the spec requires exactly one). Click's
    ``flag_value`` shortcut silently last-wins on conflict, so we use
    two booleans + an explicit XOR check."""
    if weekly and quarterly:
        raise click.UsageError(
            "bba sentinel --weekly and --quarterly are mutually exclusive"
        )
    if not (weekly or quarterly):
        raise click.UsageError(
            "bba sentinel requires exactly one of --weekly or --quarterly"
        )
    cadence: SentinelCadence = "weekly" if weekly else "quarterly"
    inputs = SentinelCommandInput(cadence=cadence)
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


def _get_audit_store() -> AuditStore:
    """Resolve the :class:`~bba.audit_store.AuditStore` rooted at
    ``$BBA_DATA_DIR/audit/<code_version>``.

    Tests monkeypatch this seam via
    ``patch("bba.cli.main._get_audit_store", ...)`` to inject a
    pre-populated in-memory store; production callers use the
    on-disk Parquet layout the audit_store ships with."""
    config = AuditStoreConfig(
        root_dir=_resolve_data_dir() / "audit_store",
        code_version=code_version(),
    )
    return AuditStore(config)


def _get_ward_resolver() -> WardAttributionResolver:
    """Resolve the production
    :class:`~bba.report_generator.WardAttributionResolver`.

    Phase 1's HOSxP ingest-side procedure-table export is still pending
    (M0/M1 blocker per the design notes), so the production resolver is
    not yet plumbed; this seam fails loud naming
    ``bba.cli.main.bba_report`` so the CLI keeps its "no fabricated
    defaults" promise. Tests inject a fake via
    ``patch("bba.cli.main._get_ward_resolver", ...)``."""
    raise CliError(
        "bba.cli.main.bba_report cannot resolve a WardAttributionResolver: "
        "the HOSxP ingest store does not yet expose a ward-attribution "
        "table (M0/M1 blocker). Plumb the production resolver in "
        "bba.cli.main._get_ward_resolver when the table is available; "
        "until then, monkeypatch the seam in tests."
    )


def _get_physician_resolver() -> PhysicianAttributionResolver:
    """Resolve the production
    :class:`~bba.report_generator.PhysicianAttributionResolver`.

    Symmetric with :func:`_get_ward_resolver` — same M0/M1 blocker, same
    loud-failure contract, same injection seam."""
    raise CliError(
        "bba.cli.main.bba_report cannot resolve a "
        "PhysicianAttributionResolver: the HOSxP ingest store does not "
        "yet expose a physician-attribution table (M0/M1 blocker). Plumb "
        "the production resolver in bba.cli.main._get_physician_resolver "
        "when the table is available; until then, monkeypatch the seam "
        "in tests."
    )


def _get_audit_run_store() -> AuditRunStore:
    """Resolve the :class:`~bba.cli.store_protocol.AuditRunStore`.

    Phase 1 ships a file-backed adapter
    (:class:`~bba.cli.audit_run_store.FileBackedAuditRunStore`) rooted at
    ``$BBA_DATA_DIR/audit_runs/``. The CLI's contract is four narrow
    operations on a single ``run_id`` key, and a flat-file layout
    satisfies durability + auditability at zero ops cost. A future
    Postgres-backed adapter implements the same Protocol and is swapped
    in here without changing the CLI.

    Tests monkeypatch this seam via
    ``patch("bba.cli.main._get_audit_run_store", ...)`` to inject a
    recording double; the in-process replacement keeps the unit suite
    independent of the on-disk layout.
    """
    return FileBackedAuditRunStore(_resolve_data_dir())


def _run_audit_pipeline(
    *,
    run_id: str,
    input_csv: Path,
    store: AuditRunStore,
) -> None:
    """Run the Phase 1 audit pipeline for ``input_csv``.

    The Phase 1 deliverable for ``bba audit`` is the *ingest leg* of
    the pipeline: the HOSxP CSV bundle is materialised into the
    configured DuckDB + Parquet store under
    ``BBA_DATA_DIR/audit/<run_id>/``. The LLM-driven analysis leg
    (deterministic classifier → evidence bundle → de-id → prompt →
    batch submit → quote ground → calibrate → audit_store write)
    requires a context-builder composition over :mod:`bba.audit_orders`
    / :mod:`bba.deterministic_classifier` /
    :mod:`bba.evidence_bundle_builder` / :mod:`bba.deid_redactor` /
    :mod:`bba.prompt_builder` that is owned by
    :mod:`bba.audit_pipeline` per its own contract ("the pipeline never
    re-implements upstream modules; the caller assembles the context").
    When that orchestration facade lands, this seam delegates to it.

    ``store.record_row`` is called once per ingested HOSxP table so
    :meth:`AuditRunStore.run_count` is consistent with
    :meth:`AuditRunStore.run_complete` after this function returns —
    a downstream observer querying ``run_count > 0`` will see a real
    count of committed units of work, not a stale zero.

    Tests patch this seam via
    ``patch("bba.cli.main._run_audit_pipeline", ...)`` to isolate the
    CLI's idempotency / force / scrubbing behaviour from the ingest
    cost."""
    output_dir = _resolve_data_dir() / "audit" / run_id
    config = IngestConfig(
        input_dir=input_csv.parent,
        output_dir=output_dir,
        code_version=code_version(),
    )
    result: IngestResult = ingest(config)
    # ``IngestResult.tables_written`` is the authoritative list of
    # validated HOSxP tables for this run; trusting the function's
    # typed return value is more robust than re-globbing the output
    # directory (which would be tied to whichever ingest revision is
    # currently shipping parquet writes vs marker-only writes).
    if not result.tables_written:
        # Defensive guard: a successful ingest invariably returns at
        # least one validated table (and otherwise raises
        # ``IncompleteInputError``). Reaching here means ingest's
        # contract changed silently — refuse to mark the run complete
        # rather than persist a ``run_<id>.complete`` marker for a run
        # with zero row markers (PRD §20 "CLI fails loud").
        raise CliError(
            f"audit pipeline for run_id={run_id} produced zero ingested "
            "tables; refusing to mark the run complete (a successful "
            "Phase 1 audit must yield at least one validated HOSxP table)"
        )
    # Record one marker per validated HOSxP table so run_count is
    # consistent with the completed marker that bba_audit writes next.
    # The audit_id namespace ``phase1_ingest_<table>`` is reserved for
    # the ingest-leg-only Phase 1; the full LLM-driven leg will write
    # ``audit_<hn_hash>_<requested_at>`` rows under the same run_id.
    for table_name in result.tables_written:
        store.record_row(run_id, f"phase1_ingest_{table_name}")
    _log.info(
        "audit.pipeline_ingest_complete",
        run_id=run_id,
        output_dir=str(output_dir),
        leg="phase1_ingest_only",
        table_count=len(result.tables_written),
        ingest_skipped_idempotent=result.skipped_idempotent,
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
