"""PDF renderer for the monthly report (issue #28).

One PDF per monthly run: a cover page (month + footer), then one page per
section. The renderer takes the already-aggregated :class:`ReportSection`
list so it has no opinion about how the numbers were computed — only how
they are laid out on the page.

PDF rendering is the most fragile part of this module (font handling,
binary-stability across reportlab patch releases). The "PDF rendering
tested with a sample run" acceptance criterion (issue #28) is satisfied
by asserting the bytes start with the PDF magic header and that the file
is non-trivial in size; byte-identical PDF output is not promised because
reportlab embeds a generation timestamp.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from bba.report_generator.models import ReportFooter, ReportSection


PDF_MAGIC = b"%PDF-"
"""The four-byte magic header every PDF file starts with. Used by the
acceptance test instead of byte-identical comparison because reportlab
embeds a non-deterministic generation timestamp.
"""


def render_report_pdf(
    sections: Sequence[ReportSection],
    footer: ReportFooter,
    month_label: str,
    output_path: Path,
) -> Path:
    """Render ``sections`` to a PDF at ``output_path`` and return the path.

    ``month_label`` is a human-readable string ("May 2026") used on the
    cover page. The footer (policy_version / model_id / redactor_version)
    appears on every page so a printed-then-detached page still carries
    the reproducibility chain.
    """
    raise NotImplementedError
