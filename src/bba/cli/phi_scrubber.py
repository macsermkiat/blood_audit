"""Process-level exception scrubbing for ``bba`` (PRD §20 / ticket #29).

A long-running ``bba audit`` reads PHI in memory (HOSxP bundles, free-
text notes, ``hn`` / ``an`` identifiers). An uncaught exception would
default-format its frame locals into the traceback, leaking PHI into
the operator's log surface. The scrubber:

1. Installs a :data:`sys.excepthook` that captures the traceback object
   and walks its frame locals + globals.
2. For any binding whose **name** matches :data:`PHI_LOCAL_NAME_REGEX`
   (``bundle`` / ``patient`` / ``note`` / ``hn`` / ``an`` /
   ``encounter``, case-insensitive prefix), the value's ``repr`` is
   replaced with ``<REDACTED:type=<typename> len=<len>>``.
3. For every string value, any match against :data:`PHI_REGEXES`
   (HN-shaped digit runs, Western honorific + Capitalized name, Thai
   honorifics) is replaced with ``<REDACTED:phi>``.
4. The scrubbed traceback is emitted through ``structlog`` so the log
   pipeline (file / SIEM / Loki) only ever sees redacted content.
5. :mod:`faulthandler` is redirected to a sidecar file that gets
   scrubbed *on read* (the write path is too hot — see PRD §20).

:data:`PHI_REGEXES` is owned here for now; a follow-up to issue #17 will
migrate it to :data:`bba.deid_redactor.PHI_REGEXES` so the redactor and
the scrubber share one source of truth. Until then this module is the
authority — see :class:`TestPhiRegexesProvenance` in
``tests/unit/test_cli.py`` for the alignment check.
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
/ ``BUNDLE`` are caught too."""


_THAI_HONORIFICS: Final[str] = r"นาย|นาง|นางสาว|เด็กชาย|เด็กหญิง"
"""Five Thai honorifics common in HOSxP free-text notes. Matching the
honorific alone is enough to redact the surrounding string — we do not
try to grab the trailing given name token, which would mis-fire on
benign sentences."""


PHI_REGEXES: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\b\d{7,10}\b"),  # HN / AN digit runs
    re.compile(
        r"\b(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
    ),  # Western honorific + 1–N capitalised name tokens
    re.compile(_THAI_HONORIFICS),
)
"""Regexes that match PHI inside string values.

Order matters: digit-run is tried first because the cost of evaluating
a digit-run regex is lower than the alpha-token regexes.
"""


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
    """Return a scrubbed copy of a frame's locals / globals.

    Two passes:
    1. PHI-named keys are replaced with the type/len marker regardless
       of value type (a ``patient`` dict, a ``bundle`` Mapping, a
       ``note`` str — all collapse to one marker).
    2. Remaining string values are regex-swept against
       :data:`PHI_REGEXES`; matched substrings collapse to
       ``<REDACTED:phi>``.

    Built-in module bindings (``__builtins__``, ``__name__``, ...) are
    dropped — they are noisy and never PHI."""
    out: dict[str, Any] = {}
    for name, value in scope.items():
        if name.startswith("__") and name.endswith("__"):
            continue
        if PHI_LOCAL_NAME_REGEX.match(name):
            out[name] = _redact_by_name(value)
            continue
        if isinstance(value, str):
            out[name] = _redact_phi_in_string(value)
            continue
        out[name] = value
    return out


def scrub_traceback(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: TracebackType | None,
) -> str:
    """Return a scrubbed rendering of ``(exc_type, exc_value, exc_tb)``.

    Walks the traceback frame-by-frame and emits, for each frame:

    * the file / line / function header (``traceback.format_list`` form);
    * the source line (already redacted by Python's traceback module);
    * a ``locals:`` block of ``name = repr`` entries built from the
      scrubbed copy of the frame's locals.

    The output is multi-line; tests assert the *absence* of PHI
    substrings and the *presence* of ``<REDACTED``.
    """
    scrubbed_message = _redact_phi_in_string(str(exc_value))
    lines: list[str] = [f"{exc_type.__name__}: {scrubbed_message}"]
    tb: TracebackType | None = exc_tb
    while tb is not None:
        frame = tb.tb_frame
        code = frame.f_code
        lines.append(
            f'  File "{code.co_filename}", line {tb.tb_lineno}, '
            f"in {code.co_name}"
        )
        scrubbed_locals = _scrub_frame_dict(frame.f_locals)
        for name, value in scrubbed_locals.items():
            try:
                rendered = repr(value)
            except Exception:  # noqa: BLE001 — repr must not blow up the hook
                rendered = "<repr-failed>"
            lines.append(f"    {name} = {_redact_phi_in_string(rendered)}")
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
