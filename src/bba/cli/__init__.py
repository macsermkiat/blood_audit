"""``bba.cli`` — user-facing surface of Phase 1 (issue #29).

Thin glue: every subcommand is a ≤20-line wrapper over an already-tested
module. The only real logic that lives here is run-level idempotency
(``compute_run_id`` + ``AuditRunStore`` checks) and process-level
exception scrubbing (``install_excepthook`` + faulthandler redirect).

See ``src/bba/cli/main.py`` for the click root group, and the module
docstring of each submodule for the contract it owns.
"""

from bba.cli.exceptions import (
    CliError,
    IdempotencyError,
    MutuallyExclusiveOptionError,
    RunNotFoundError,
)
from bba.cli.identity import (
    RUN_ID_LENGTH,
    CodeVersion,
    InputCsvHash,
    RunId,
    SchemaFingerprint,
    code_version,
    compute_run_id,
)
from bba.cli.main import (
    bba_audit,
    bba_evaluate,
    bba_ingest,
    bba_report,
    bba_sentinel,
    bba_serve_dashboard,
    cli,
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
from bba.cli.phi_scrubber import (
    PHI_LOCAL_NAME_REGEX,
    PHI_REGEXES,
    install_excepthook,
    scrub_traceback,
)
from bba.cli.store_protocol import AuditRunStore

__all__ = [
    "AuditCommandInput",
    "AuditRunStore",
    "CliError",
    "CodeVersion",
    "EvaluateCommandInput",
    "IdempotencyError",
    "IngestCommandInput",
    "InputCsvHash",
    "MutuallyExclusiveOptionError",
    "PHI_LOCAL_NAME_REGEX",
    "PHI_REGEXES",
    "RUN_ID_LENGTH",
    "ReportCommandInput",
    "ReportFormat",
    "RunId",
    "RunNotFoundError",
    "SchemaFingerprint",
    "SchemaVersion",
    "SentinelCadence",
    "SentinelCommandInput",
    "ServeDashboardInput",
    "bba_audit",
    "bba_evaluate",
    "bba_ingest",
    "bba_report",
    "bba_sentinel",
    "bba_serve_dashboard",
    "cli",
    "code_version",
    "compute_run_id",
    "install_excepthook",
    "scrub_traceback",
]
