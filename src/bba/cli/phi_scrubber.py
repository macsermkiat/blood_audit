"""Log exception locations without rendering clinical data or credentials.

Names and type/length markers identify frame bindings. Values and exception
messages stay out of logs because neither PHI nor secrets have a complete
textual pattern that a blacklist can safely recognize.
"""

from __future__ import annotations

import faulthandler
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Protocol, cast

from bba.cli._logging import get_logger
from bba.deid_redactor import PHI_REGEXES as _DEID_PHI_REGEXES


# ---------------------------------------------------------------------------
# Constants — the regex set is part of the CLI's safety contract. The values
# are unit-tested in ``TestPhiSurface`` and a change here is a behavioural
# change that requires a corresponding test update.
# ---------------------------------------------------------------------------


PHI_LOCAL_NAME_REGEX: Final[re.Pattern[str]] = re.compile(
    r"^(bundle|patient|note|hn|an|encounter)",
    re.IGNORECASE,
)
"""Frame-local variable names whose ``repr`` must be redacted.

Prefix match (``^...``) so ``hn_digits``, ``patient_age``,
``encounter_id`` are all caught. Case-insensitive so ``HN`` / ``Patient``
/ ``BUNDLE`` are caught too. Lives here (not in
:mod:`bba.deid_redactor`) because frame-local naming is a Python /
traceback concern; the redactor's interest in PHI is purely textual."""


PHI_REGEXES: Final[tuple[re.Pattern[str], ...]] = _DEID_PHI_REGEXES
"""Regexes that match PHI inside string values, re-exported from
:data:`bba.deid_redactor.PHI_REGEXES` so the scrubber and the redactor
share one source of truth. Any pattern added there is automatically
applied by the traceback scrubber on the next run."""


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class StructlogLogger(Protocol):
    """Minimal contract for the structlog logger the scrubber emits to.

    Declaring a Protocol (instead of importing ``structlog.BoundLogger``)
    keeps the CLI scrubber importable in environments that bind a custom
    logger (e.g. tests that inject a recording double)."""

    def error(self, event: str, /, **kwargs: object) -> None: ...


_REDACTED_VALUE_MARKER: Final[str] = "<REDACTED:phi>"
"""String inserted in place of a regex-matched PHI substring."""


def _redact_by_name(value: object) -> str:
    """Return the ``<REDACTED:type=X len=N>`` marker for a PHI-named local.

    ``len`` is the string-length of the value's ``repr`` if available,
    or the value's ``__len__`` otherwise; falling back to ``-1`` for
    objects with no notion of length keeps the marker non-throwing for
    arbitrary frame locals."""
    typename = type(value).__name__
    try:
        length = len(value)  # type: ignore[arg-type]
    except TypeError:
        length = -1
    return f"<REDACTED:type={typename} len={length}>"


def _redact_phi_in_string(text: str) -> str:
    """Apply :data:`PHI_REGEXES` to ``text``; each match → ``<REDACTED:phi>``."""
    for pattern in PHI_REGEXES:
        text = pattern.sub(_REDACTED_VALUE_MARKER, text)
    return text


def _scrub_frame_dict(scope: Mapping[str, Any]) -> dict[str, Any]:
    """Return binding names and opaque markers without rendering their values."""
    out: dict[str, Any] = {}
    for name, value in scope.items():
        if name.startswith("__") and name.endswith("__"):
            continue
        out[name] = _redact_by_name(value)
    return out


def scrub_traceback(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: TracebackType | None,
) -> str:
    """Render exception type and frame locations with opaque local markers."""
    lines: list[str] = [f"{exc_type.__name__}: <REDACTED:exception-message>"]
    tb: TracebackType | None = exc_tb
    while tb is not None:
        frame = tb.tb_frame
        code = frame.f_code
        lines.append(
            f'  File "{code.co_filename}", line {tb.tb_lineno}, in {code.co_name}'
        )
        scrubbed_locals = _scrub_frame_dict(frame.f_locals)
        for name, value in scrubbed_locals.items():
            lines.append(f"    {name} = {value}")
        tb = tb.tb_next
    return "\n".join(lines)


def install_excepthook(
    *,
    logger: StructlogLogger | None = None,
    faulthandler_sidecar: Path | None = None,
) -> None:
    """Install the PHI-scrubbing ``sys.excepthook`` + ``faulthandler`` sidecar.

    Idempotent — a second call replaces the prior hook so tests can
    re-install per-fixture without leaking state.

    ``logger`` defaults to ``structlog.get_logger("bba.cli")`` (bridged
    to stdlib logging via :mod:`bba.cli._logging`). ``faulthandler_sidecar``
    defaults to ``None`` (no sidecar); when set, :mod:`faulthandler` is
    enabled and redirects to that file's ``fd`` so a hard crash dumps a
    stack trace which a downstream reader scrubs before display.
    """
    # ``get_logger()`` returns a structlog BoundLogger; cast satisfies
    # the structural Protocol contract without dragging the concrete
    # structlog type into the signature.
    active_logger: StructlogLogger = (
        logger if logger is not None else cast(StructlogLogger, get_logger())
    )

    def _hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None:
        scrubbed = scrub_traceback(exc_type, exc_value, exc_tb)
        active_logger.error(
            "uncaught_exception",
            exc_type=exc_type.__name__,
            traceback=scrubbed,
        )

    sys.excepthook = _hook

    if faulthandler_sidecar is not None:
        faulthandler_sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar_fh = faulthandler_sidecar.open("a", encoding="utf-8")
        faulthandler.enable(file=sidecar_fh, all_threads=True)
