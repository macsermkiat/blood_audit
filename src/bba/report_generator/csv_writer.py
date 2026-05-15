"""CSV writer for monthly report sections (issue #28).

Each section's CSV is a *self-contained* file: column header, data rows,
and the reproducibility footer stamped on every data row. Stamping per row
(rather than only in a trailing footer line) means a downstream consumer
can grep / filter / join the CSV without losing the reproducibility chain.

Output uses ``\\n`` line endings (Unix) and UTF-8 encoding without a BOM:
the file is consumed by Python tooling and the dashboard's web view, not
opened in Excel-on-Windows; forcing CRLF would produce noisy diffs in the
golden snapshot tests.
"""

from __future__ import annotations

from pathlib import Path

from bba.report_generator.models import ReportSection, SectionName


CSV_NEWLINE = "\n"
"""Line terminator for every CSV emitted by this module. Locked to ``\\n``
so the golden-snapshot tests are byte-stable across platforms.
"""

CSV_ENCODING = "utf-8"
"""Encoding for every CSV emitted by this module. No BOM."""


def write_section_csv(section: ReportSection, output_dir: Path) -> Path:
    """Write ``section`` to ``output_dir / f"{section.name}.csv"`` and return
    the resulting :class:`Path`.

    The output path is derived deterministically from ``section.name`` so a
    second call with the same section overwrites the first (idempotent
    re-run is a project-wide contract; see ``bba.audit_store``).

    Raises :class:`FileNotFoundError` if ``output_dir`` does not exist; the
    caller (the orchestrator) is responsible for creating the directory.
    """
    raise NotImplementedError


def section_filename(name: SectionName) -> str:
    """Return the canonical CSV filename for ``name``.

    Exposed so the orchestrator and the test fixtures agree on the filename
    without re-encoding the convention in two places.
    """
    raise NotImplementedError
