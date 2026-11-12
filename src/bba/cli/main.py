"""Supported command-line entry points and deployment configuration boundaries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast
import json
import hashlib

from pydantic import SecretStr

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
from bba.attribution import (
    load_dct_registry,
    load_reqno_to_doctor,
    make_physician_resolver,
    make_ward_resolver,
)
from bba.ingest import IngestConfig, ingest, schema_fingerprint
from bba.application.audit import audit_bundle
from bba.application.config import load_config
from bba.application.operations import (
    MonitoringManifest,
    evaluate,
    persist_analysis,
    run_rows,
    sentinel,
    write_report,
)
from bba.audit_store import AuditStore, AuditStoreConfig
from bba.report_generator import (
    PhysicianAttributionResolver,
    WardAttributionResolver,
    build_report_inputs,
)


_log = get_logger()


_BBA_DATA_DIR_ENV: str = "BBA_DATA_DIR"


def _resolve_data_dir() -> Path:
    """Return ``$BBA_DATA_DIR`` as a :class:`Path`.

    Raises :class:`CliError` if the env var is unset. The CLI's env
    surface (BBA_DATA_DIR, BBA_DB_URL, ANTHROPIC_API_KEY, plus the
    attribution inputs BBA_BDVST_CSV / BBA_DCT_CSV) is asserted by
    ``TestEnvVarSurfaceIsTight``."""
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
        source = inputs.input_csv
        if source is None:
            manifest_path = (
                _resolve_data_dir() / "audit_runs" / f"{resolved}.input.json"
            )
            if not manifest_path.exists():
                raise click.UsageError("no recorded source for this run; use --input")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source = Path(manifest["input_csv"])
            if _resolve_run_id(AuditCommandInput(input_csv=source)) != resolved:
                raise CliError(
                    "source bundle or application configuration changed; use --input for a new run"
                )
        _run_audit_pipeline(run_id=resolved, input_csv=source, store=store)
        store.mark_run_complete(resolved)
        _log.info(
            "audit.complete",
            run_id=resolved,
            row_count=store.run_count(resolved),
        )


@click.command(name="evaluate")
@click.option("--run-id", type=str, required=True)
@click.option(
    "--labels",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--code-version", "selected_version", type=str, default=None)
def bba_evaluate(run_id: str, labels: Path, selected_version: str | None) -> None:
    """Compare committed classifications against explicit reviewed CSV labels."""
    inputs = EvaluateCommandInput(run_id=run_id)
    version = selected_version or code_version()
    result = evaluate(
        run_rows(_get_audit_store(), inputs.run_id, version),
        labels,
    )
    target = persist_analysis(
        _resolve_data_dir() / "evaluations",
        {"run_id": inputs.run_id, "code_version": version, **result},
    )
    click.echo(str(target))


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
    """Render a committed run as HTML, PDF or JSON."""
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
    path = write_report(report_inputs, inputs.format)
    _log.info(
        "report.complete", run_id=inputs.run_id, format=inputs.format, path=str(path)
    )
    click.echo(str(path))


@click.command(name="serve-dashboard")
@click.option("--port", type=int, default=8000, show_default=True)
@click.option("--run-id", type=str, default=None)
@click.option("--code-version", "selected_version", type=str, default=None)
def bba_serve_dashboard(
    port: int, run_id: str | None, selected_version: str | None
) -> None:
    """Serve behind an HTTPS auth proxy that issues signed bba_session cookies."""
    import uvicorn
    from bba.dashboard import DashboardConfig, create_app
    from bba.review_actions import ReviewActionsConfig, ReviewActionsStore

    inputs = ServeDashboardInput(port=port)
    dsn = os.environ.get("BBA_DB_URL")
    secret = os.environ.get("BBA_DASHBOARD_AUTH_SECRET")
    origin = os.environ.get("BBA_DASHBOARD_PUBLIC_ORIGIN")
    if not dsn or not secret or not origin:
        raise CliError(
            "serve-dashboard requires BBA_DB_URL, BBA_DASHBOARD_AUTH_SECRET and BBA_DASHBOARD_PUBLIC_ORIGIN"
        )
    reviews = ReviewActionsStore(ReviewActionsConfig(dsn=dsn))
    try:
        config = DashboardConfig(
            audit_store=_get_audit_store(),
            review_actions_store=reviews,
            snapshot_dir=_resolve_data_dir() / "snapshots",
            code_version=selected_version or code_version(),
            run_id=run_id,
            auth_secret=SecretStr(secret),
            public_origin=origin,
            ward_attribution_resolver=_get_ward_resolver(),
            physician_attribution_resolver=_get_physician_resolver(),
        )
        uvicorn.run(create_app(config), host="127.0.0.1", port=inputs.port)
    finally:
        reviews.close()


@click.command(name="sentinel")
@click.option("--weekly", is_flag=True, default=False)
@click.option("--quarterly", is_flag=True, default=False)
@click.option(
    "--manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
)
def bba_sentinel(weekly: bool, quarterly: bool, manifest: Path | None) -> None:
    """Compare explicit baseline/current committed runs using a fixed manifest."""
    if weekly == quarterly:
        raise click.UsageError(
            "bba sentinel requires exactly one of --weekly or --quarterly; they are mutually exclusive"
        )
    if manifest is None:
        raise click.UsageError(
            "bba sentinel requires --manifest with baseline/current runs, versions and sentinel membership"
        )
    cadence: SentinelCadence = "weekly" if weekly else "quarterly"
    inputs = SentinelCommandInput(cadence=cadence)
    selection = MonitoringManifest.model_validate_json(
        manifest.read_text(encoding="utf-8")
    )
    result = sentinel(_get_audit_store(), selection, quarterly=quarterly)
    target = persist_analysis(
        _resolve_data_dir() / "monitoring",
        {
            **result.model_dump(mode="json"),
            "cadence": inputs.cadence,
            "comparison": selection.model_dump(mode="json"),
        },
    )
    click.echo(str(target))


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


_BBA_BDVST_CSV_ENV: str = "BBA_BDVST_CSV"
_BBA_DCT_CSV_ENV: str = "BBA_DCT_CSV"


def _require_env_path(name: str, purpose: str) -> Path:
    """Return the env var ``name`` as a :class:`Path`, failing loud.

    The attribution seams keep the CLI's "no fabricated defaults"
    promise: an unset variable names the seam and the expected file
    instead of guessing a location."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        raise CliError(
            f"bba.cli.main.bba_report cannot resolve a "
            f"WardAttributionResolver / PhysicianAttributionResolver: "
            f"{name} environment variable is required ({purpose}). "
            f"The former M0/M1 attribution blocker is resolved by "
            f"BDVST.DCTREQ -> DCT; set {_BBA_BDVST_CSV_ENV} (BDVST export "
            f"carrying REQNO + DCTREQ) and {_BBA_DCT_CSV_ENV} (DCT.csv "
            f"doctor registry) to enable attribution."
        )
    return Path(raw)


def _get_ward_resolver() -> WardAttributionResolver:
    """Resolve the production
    :class:`~bba.report_generator.WardAttributionResolver`.

    Wired via :mod:`bba.attribution`: ``REQNO`` → ``BDVST.DCTREQ`` →
    ``DCT.csv`` ``Deptlct`` (the department stands in for the ward —
    the HOSxP export has no separate ward table). Requires
    ``BBA_BDVST_CSV`` and ``BBA_DCT_CSV``; unset fails loud. Tests
    inject a fake via ``patch("bba.cli.main._get_ward_resolver", ...)``."""
    bdvst_csv = _require_env_path(
        _BBA_BDVST_CSV_ENV, "BDVST export with REQNO + DCTREQ columns"
    )
    dct_csv = _require_env_path(
        _BBA_DCT_CSV_ENV, "DCT.csv doctor registry with Deptlct/Deptname"
    )
    return make_ward_resolver(
        load_reqno_to_doctor(bdvst_csv), load_dct_registry(dct_csv)
    )


def _get_physician_resolver() -> PhysicianAttributionResolver:
    """Resolve the production
    :class:`~bba.report_generator.PhysicianAttributionResolver`.

    Wired via :mod:`bba.attribution`: ``REQNO`` → ``BDVST.DCTREQ``
    (the ordering-doctor code). Requires ``BBA_BDVST_CSV``; unset fails
    loud. Symmetric with :func:`_get_ward_resolver`."""
    bdvst_csv = _require_env_path(
        _BBA_BDVST_CSV_ENV, "BDVST export with REQNO + DCTREQ columns"
    )
    return make_physician_resolver(load_reqno_to_doctor(bdvst_csv))


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
    """Compose the supported application and count committed AuditRows."""
    raw_config = os.environ.get("BBA_APPLICATION_CONFIG")
    config = load_config(Path(raw_config) if raw_config else None)
    audit_ids = audit_bundle(
        input_csv=input_csv,
        data_dir=_resolve_data_dir(),
        version=code_version(),
        run_id=run_id,
        audit_store=_get_audit_store(),
        config=config,
    )
    if not audit_ids:
        raise CliError("audit emitted no committed rows; refusing to record completion")
    for audit_id in audit_ids:
        store.record_row(run_id, audit_id)


def _resolve_run_id(inputs: AuditCommandInput) -> str:
    """Return the ``run_id`` for the audit invocation.

    If ``--run-id`` was passed, use it verbatim. Otherwise compute it
    from the input CSV's bytes, the active schema fingerprint, and the
    package version — the formula from PRD §20."""
    if inputs.run_id is not None:
        return inputs.run_id
    assert inputs.input_csv is not None, "AuditCommandInput XOR invariant"
    base_run_id = compute_run_id(
        input_csv=inputs.input_csv,
        schema_fingerprint=SchemaFingerprint(schema_fingerprint()),
        code_version_str=CodeVersion(code_version()),
    )
    raw_config = os.environ.get("BBA_APPLICATION_CONFIG")
    if raw_config:
        config = load_config(Path(raw_config))
        return hashlib.sha256(
            f"{base_run_id}|{config.model_dump_json()}".encode()
        ).hexdigest()[:16]
    return base_run_id


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
