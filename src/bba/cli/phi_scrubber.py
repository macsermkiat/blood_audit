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

import re
from pathlib import Path
from types import TracebackType
from typing import Final, Protocol


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


def scrub_traceback(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: TracebackType | None,
) -> str:
    """Return a scrubbed, human-readable rendering of ``(exc_type, exc_value, exc_tb)``.

    GREEN-phase steps:
    1. Walk ``exc_tb`` frame-by-frame.
    2. For each frame, copy ``frame.f_locals`` / ``frame.f_globals`` into
       a working dict; replace PHI-named bindings with
       ``"<REDACTED:type=<typename> len=<len>>"``.
    3. For every string in the working dict, apply :data:`PHI_REGEXES`
       and replace each match with ``"<REDACTED:phi>"``.
    4. Render the scrubbed frame chain into a multi-line string using
       :mod:`traceback.format_list`-style formatting.

    The returned string must satisfy: contains ``"<REDACTED"`` for every
    redacted slot; contains no substring from the original PHI values.
    """
    raise NotImplementedError(
        "scrub_traceback — GREEN phase wires frame-locals walk + regex sweep"
    )


def install_excepthook(
    *,
    logger: StructlogLogger | None = None,
    faulthandler_sidecar: Path | None = None,
) -> None:
    """Install the PHI-scrubbing ``sys.excepthook`` and ``faulthandler`` redirect.

    Idempotent: a second call replaces the prior hook (so tests can
    re-install per-fixture without leaking state). Does *not* clobber an
    upstream hook chained via ``sys.__excepthook__`` — the prior hook is
    captured and may be invoked on a scrubbed copy if the GREEN-phase
    design chooses to chain.

    ``logger`` defaults to ``structlog.get_logger("bba.cli")``;
    ``faulthandler_sidecar`` defaults to
    ``$BBA_DATA_DIR/_faulthandler.sidecar``.
    """
    raise NotImplementedError(
        "install_excepthook — GREEN phase wires sys.excepthook + faulthandler"
    )
