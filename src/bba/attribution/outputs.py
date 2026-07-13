"""Artifact writers for the ranking deliverable — CSV + standalone HTML.

CSV conventions are shared with :mod:`bba.report_generator.csv_writer`
(same ``\\n`` line endings, UTF-8 without BOM, same float rendering) so
the two report surfaces stay byte-consistent; the section-registry
writer itself is closed over a ``SectionName`` literal and is not
extended here.

The HTML artifact is deliberately standalone (inline CSS, no external
assets) so it can be shared with the transfusion committee as a single
file. Every dynamic string is escaped — doctor display names come from
an external export and are untrusted.
"""

from __future__ import annotations

import csv
import html
import io
from collections.abc import Sequence
from pathlib import Path

from bba.attribution.models import RankedRow, RankingResult, RankingTable
from bba.feature_flags import RETURNS_LEDGER_ENABLED
from bba.report_generator.csv_writer import CSV_ENCODING, CSV_NEWLINE


_BASE_RANKING_CSV_COLUMNS: tuple[str, ...] = (
    "rank",
    "group_id",
    "group_name",
    "total_orders",
    "appropriate",
    "inappropriate",
    "unresolved",
    "bucket",
    "bucket_count",
    "bucket_rate",
    "meets_min_orders",
)
RANKING_CSV_COLUMNS: tuple[str, ...] = (
    _BASE_RANKING_CSV_COLUMNS[:7]
    + (("returned_not_transfused",) if RETURNS_LEDGER_ENABLED else ())
    + _BASE_RANKING_CSV_COLUMNS[7:]
)


def _ranking_csv_columns() -> tuple[str, ...]:
    return (
        _BASE_RANKING_CSV_COLUMNS[:7]
        + (("returned_not_transfused",) if RETURNS_LEDGER_ENABLED else ())
        + _BASE_RANKING_CSV_COLUMNS[7:]
    )


def _format_cell(value: object) -> str:
    """Render one CSV cell, mirroring the report CSV writer's float and
    bool conventions (``0.5`` not ``0.500000``; lowercase bools)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        formatted = f"{value:.6f}"
        if "." in formatted:
            formatted = formatted.rstrip("0")
            if formatted.endswith("."):
                formatted += "0"
        return formatted
    return str(value)


def _row_cells(row: RankedRow) -> list[str]:
    values: list[object] = [
        row.rank,
        row.group_id,
        row.group_name,
        row.total_orders,
        row.appropriate_count,
        row.inappropriate_count,
        row.unresolved_count,
    ]
    if RETURNS_LEDGER_ENABLED:
        values.append(row.returned_not_transfused_count)
    values.extend(
        (row.bucket, row.bucket_count, row.bucket_rate, row.meets_min_orders)
    )
    return [_format_cell(value) for value in values]


def write_ranking_csv(rows: Sequence[RankedRow], path: Path) -> Path:
    """Write one ranking table to ``path`` and return it."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator=CSV_NEWLINE)
    writer.writerow(_ranking_csv_columns())
    for row in rows:
        writer.writerow(_row_cells(row))
    path.write_text(buf.getvalue(), encoding=CSV_ENCODING, newline="")
    return path


_STYLE = """
body { font-family: system-ui, 'Sarabun', sans-serif; margin: 2rem auto;
       max-width: 64rem; color: #1a202c; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.15rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { border: 1px solid #cbd5e0; padding: 0.35rem 0.6rem;
         text-align: left; }
th { background: #edf2f7; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.below-threshold td { color: #718096; }
p.caveat { background: #fffbea; border: 1px solid #ecc94b;
           padding: 0.6rem 0.9rem; font-size: 0.9rem; }
p.totals { font-size: 0.9rem; }
"""


def _render_table(table: RankingTable) -> str:
    returns_header = (
        '<th class="num">Returned, not transfused</th>'
        if RETURNS_LEDGER_ENABLED
        else ""
    )
    header_cells = (
        '<th>Rank</th><th>Code</th><th>Name</th><th class="num">Orders (N)</th>'
        '<th class="num">Appropriate</th><th class="num">Inappropriate</th>'
        '<th class="num">Unresolved</th>'
        f"{returns_header}"
        f'<th class="num">{html.escape(table.bucket)} rate</th>'
        f"<th>N &ge; {table.min_orders}</th>"
    )
    body_rows: list[str] = []
    for row in table.rows:
        css_class = "" if row.meets_min_orders else ' class="below-threshold"'
        threshold_mark = "yes" if row.meets_min_orders else "no"
        returns_cell = (
            f'<td class="num">{row.returned_not_transfused_count}</td>'
            if RETURNS_LEDGER_ENABLED
            else ""
        )
        body_rows.append(
            f"<tr{css_class}>"
            f'<td class="num">{row.rank}</td>'
            f"<td>{html.escape(row.group_id)}</td>"
            f"<td>{html.escape(row.group_name)}</td>"
            f'<td class="num">{row.total_orders}</td>'
            f'<td class="num">{row.appropriate_count}</td>'
            f'<td class="num">{row.inappropriate_count}</td>'
            f'<td class="num">{row.unresolved_count}</td>'
            f"{returns_cell}"
            f'<td class="num">{_format_cell(row.bucket_rate)}</td>'
            f"<td>{threshold_mark}</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{header_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def write_rankings_html(
    result: RankingResult,
    path: Path,
    *,
    verdict_source_label: str,
    title: str = "Blood-order appropriateness — top ordering doctors and departments",
) -> Path:
    """Write the two ranking tables as one standalone HTML file.

    The caveat block states the verdict source, the ranked bucket, and
    the minimum-order threshold — the plan requires these visible in the
    artifact so a 300-sample table cannot be over-read.
    """
    totals = result.totals
    bucket = html.escape(result.doctors.bucket)
    min_orders = result.doctors.min_orders
    caveat = (
        f"Verdict source: {html.escape(verdict_source_label)}. "
        f"Tables are ranked by {bucket} rate among groups with "
        f"N &ge; {min_orders} orders; groups below that threshold are "
        "listed after the qualified rows, ranked by count, and shown "
        "greyed — a rate computed on 1&ndash;4 orders is not comparable. "
        f"Groups with zero {bucket} orders are omitted entirely. "
        "Unresolved = needs-review + insufficient-evidence; cohort "
        "totals below cover all groups, including omitted ones."
    )
    returns_total = (
        f"; {totals.returned_not_transfused} returned/not-transfused excluded"
        if RETURNS_LEDGER_ENABLED
        else ""
    )
    totals_line = (
        f"Cohort totals: {totals.total} orders &mdash; "
        f"{totals.appropriate} appropriate / "
        f"{totals.inappropriate} inappropriate / "
        f"{totals.unresolved} unresolved{returns_total}."
    )
    document = (
        "<!DOCTYPE html>\n"
        '<html lang="th">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{_STYLE}</style>\n</head>\n<body>\n"
        f"<h1>{html.escape(title)}</h1>\n"
        f'<p class="caveat">{caveat}</p>\n'
        f'<p class="totals">{totals_line}</p>\n'
        f"<h2>Top {result.doctors.n} ordering doctors</h2>\n"
        f"{_render_table(result.doctors)}\n"
        f"<h2>Top {result.departments.n} departments</h2>\n"
        f"{_render_table(result.departments)}\n"
        "</body>\n</html>\n"
    )
    path.write_text(document, encoding="utf-8")
    return path
