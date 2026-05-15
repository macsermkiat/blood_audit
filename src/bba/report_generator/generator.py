"""Top-level orchestrator for the monthly report (issue #28).

:func:`generate_monthly_report` is the only entry point the monthly CLI
(``bba report``) and the dashboard's "download" handler need to call. It
filters by month, aggregates each section, writes CSVs, renders the PDF,
and returns the resulting :class:`ReportArtifacts`.
"""

from __future__ import annotations

from bba.report_generator.models import ReportArtifacts, ReportInputs


def generate_monthly_report(inputs: ReportInputs) -> ReportArtifacts:
    """Produce the six section CSVs and the PDF for ``inputs.month``.

    Side effects: writes seven files into ``inputs.output_dir`` (six CSVs
    + one PDF). The directory is created if it does not exist. The
    returned :class:`ReportArtifacts` has absolute paths so downstream
    consumers (email distribution, SSO portal upload) do not need to
    re-resolve relative paths.

    Raises :class:`EmptyInputError` if no row falls inside the month;
    see the exception's docstring for why this is not silent.
    """
    raise NotImplementedError
