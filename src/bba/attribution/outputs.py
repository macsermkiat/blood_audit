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
from collections.abc import Callable, Sequence
from pathlib import Path

from bba.attribution.models import RankedRow, RankingResult, RankingTable
from bba.feature_flags import RETURNS_LEDGER_ENABLED
from bba.report_generator.csv_writer import CSV_ENCODING, CSV_NEWLINE


# A ranking CSV column: its header name paired with the accessor that
# reads the matching value off a row. The header row and every data row
# are both produced by walking this one ordered spec, so an appended
# column can never desync the header from the values (count or order).
_RankingColumn = tuple[str, Callable[[RankedRow], object]]

_LEADING_COLUMNS: tuple[_RankingColumn, ...] = (
    ("rank", lambda row: row.rank),
    ("group_id", lambda row: row.group_id),
    ("group_name", lambda row: row.group_name),
    ("total_orders", lambda row: row.total_orders),
    ("appropriate", lambda row: row.appropriate_count),
    ("inappropriate", lambda row: row.inappropriate_count),
    ("unresolved", lambda row: row.unresolved_count),
)
_RETURNS_COLUMNS: tuple[_RankingColumn, ...] = (
    ("returned_not_transfused", lambda row: row.returned_not_transfused_count),
    ("periop_transfusion_exempt", lambda row: row.periop_transfusion_exempt_count),
)
_TRAILING_COLUMNS: tuple[_RankingColumn, ...] = (
    ("bucket", lambda row: row.bucket),
    ("bucket_count", lambda row: row.bucket_count),
    ("bucket_rate", lambda row: row.bucket_rate),
    ("meets_min_orders", lambda row: row.meets_min_orders),
)
# Mean pre-transfusion triggers, appended after the thin-sample marker as
# raw mean + count per component (spec #131) — an empty cell when a mean is
# absent. Kept strictly separate: Hb from red-cell orders, platelet count
# from platelet orders.
_MEAN_COLUMNS: tuple[_RankingColumn, ...] = (
    ("mean_hb_g_dl", lambda row: row.mean_hb),
    ("hb_order_n", lambda row: row.hb_order_n),
    ("mean_platelet_k_ul", lambda row: row.mean_platelet),
    ("platelet_order_n", lambda row: row.platelet_order_n),
)


def _ranking_columns() -> tuple[_RankingColumn, ...]:
    """The ordered CSV column spec. The returns-disposition columns are
    included only when the returns ledger is enabled; evaluated per call
    so a runtime flag flip (and the tests that monkeypatch it) is
    honoured, matching the prior helper's behaviour."""
    return (
        _LEADING_COLUMNS
        + (_RETURNS_COLUMNS if RETURNS_LEDGER_ENABLED else ())
        + _TRAILING_COLUMNS
        + _MEAN_COLUMNS
    )


def _ranking_csv_columns() -> tuple[str, ...]:
    return tuple(name for name, _ in _ranking_columns())


RANKING_CSV_COLUMNS: tuple[str, ...] = _ranking_csv_columns()


def _format_cell(value: object) -> str:
    """Render one CSV cell, mirroring the report CSV writer's float and
    bool conventions (``0.5`` not ``0.500000``; lowercase bools). A
    ``None`` renders as an empty cell, not the literal ``"None"``."""
    if value is None:
        return ""
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
    return [_format_cell(accessor(row)) for _, accessor in _ranking_columns()]


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


def _format_mean(mean: float | None, n: int) -> str:
    """Render a trigger cell: ``mean (n=k)`` to one decimal when the group
    has a usable sample, otherwise an em-dash so absence of data is never
    mistaken for a low trigger."""
    if n > 0 and mean is not None:
        return f"{mean:.1f} (n={n})"
    return "&mdash;"


def _render_table(table: RankingTable) -> str:
    returns_header = (
        '<th class="num">Returned, not transfused</th>'
        '<th class="num">Peri-op transfusion (exempt)</th>'
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
        '<th class="num">Mean Hb (g/dL)</th>'
        '<th class="num">Mean platelet (&times;10&sup3;/&micro;L)</th>'
    )
    body_rows: list[str] = []
    for row in table.rows:
        css_class = "" if row.meets_min_orders else ' class="below-threshold"'
        threshold_mark = "yes" if row.meets_min_orders else "no"
        returns_cell = (
            f'<td class="num">{row.returned_not_transfused_count}</td>'
            f'<td class="num">{row.periop_transfusion_exempt_count}</td>'
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
            f'<td class="num">{_format_mean(row.mean_hb, row.hb_order_n)}</td>'
            f'<td class="num">'
            f"{_format_mean(row.mean_platelet, row.platelet_order_n)}</td>"
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
        "totals below cover all groups, including omitted ones. "
        "Mean Hb (g/dL) is the mean pre-transfusion haemoglobin over the "
        "group's scorable red-cell orders, and Mean platelet "
        "(&times;10&sup3;/&micro;L) the mean pre-transfusion platelet count "
        "over its scorable platelet orders; n is how many of those orders "
        "carried a usable value, and a column shows &mdash; when none did."
    )
    returns_total = (
        f"; {totals.returned_not_transfused} returned/not-transfused excluded"
        f"; {totals.periop_transfusion_exempt} peri-op-exempt excluded"
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
