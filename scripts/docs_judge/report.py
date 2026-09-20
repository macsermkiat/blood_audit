"""Render judgments as a Markdown report plus a JSON dump."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from .judge import PARITY_THRESHOLD, BlockJudgment, PageSummary, ParityJudgment

_SNIPPET = 200


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= _SNIPPET else flat[: _SNIPPET - 1] + "…"


def _page_sort_key(page: PageSummary) -> tuple[int, int, float]:
    return (-page.counts["rewrite"], -page.counts["light_edit"], page.readability_mean)


def _nonzero(counts: Mapping[str, int]) -> str:
    items = [f"{k}={v}" for k, v in counts.items() if v]
    return ", ".join(items) if items else "-"


def _summary_table(pages: Sequence[PageSummary]) -> list[str]:
    lines = [
        "| Page | Lang | Blocks | rewrite | light_edit | reads_human | "
        "Readability (0-2) | Top tells | Mechanical tells |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for page in sorted(pages, key=_page_sort_key):
        tells = ", ".join(f"{t} x{n}" for t, n in page.top_tells[:3]) or "-"
        lines.append(
            f"| {page.path} | {page.language} | {page.n_blocks} | "
            f"{page.counts['rewrite']} | {page.counts['light_edit']} | "
            f"{page.counts['reads_human']} | {page.readability_mean:.2f} | "
            f"{tells} | {_nonzero(page.precheck_totals)} |"
        )
    return lines


def _language_totals(pages: Sequence[PageSummary]) -> list[str]:
    lines = [
        "| Lang | Pages | Blocks | rewrite | light_edit | reads_human |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lang in ("en", "th"):
        subset = [p for p in pages if p.language == lang]
        if not subset:
            continue
        lines.append(
            f"| {lang} | {len(subset)} | {sum(p.n_blocks for p in subset)} | "
            f"{sum(p.counts['rewrite'] for p in subset)} | "
            f"{sum(p.counts['light_edit'] for p in subset)} | "
            f"{sum(p.counts['reads_human'] for p in subset)} |"
        )
    return lines


def _flagged(j: BlockJudgment) -> bool:
    return j.verdict != "reads_human" or bool(j.tells) or j.readability < 1.5


def _block_lines(judgments: Sequence[BlockJudgment]) -> list[str]:
    lines: list[str] = []
    by_page: dict[str, list[BlockJudgment]] = {}
    for j in judgments:
        by_page.setdefault(j.block.path, []).append(j)
    for path, items in by_page.items():
        flagged = [j for j in items if _flagged(j)]
        if not flagged:
            continue
        lines.append(f"### {path}")
        lines.append("")
        for j in sorted(flagged, key=lambda x: x.block.line_start):
            tells = ", ".join(f"{t} {p:.2f}" for t, p in j.tells) or "-"
            lines.append(
                f"- L{j.block.line_start}-{j.block.line_end} ({j.block.kind}, "
                f"heading: {j.block.heading or '-'}): **{j.verdict}** "
                f"p={j.verdict_p:.2f}; readability {j.readability:.2f}; "
                f"tells: {tells}; mechanical: {_nonzero(j.prechecks)}"
            )
            lines.append(f"  > {_snippet(j.block.text)}")
        lines.append("")
    return lines


CONFIDENT_ALIGN = 0.8


def _parity_lines(parity: Sequence[ParityJudgment]) -> list[str]:
    low = [p for p in parity if p.parity < PARITY_THRESHOLD]
    drift = sorted(
        (p for p in low if p.align_probability >= CONFIDENT_ALIGN),
        key=lambda p: p.parity,
    )
    weak = [p for p in low if p.align_probability < CONFIDENT_ALIGN]
    lines = [
        "The Thai pages are a deliberately abridged operator-facing summary "
        "(ruling of 2026-09-20); the English page is authoritative for full "
        "detail, and every Thai page carries a banner saying so. A Thai block "
        "that states fewer facts than its English counterpart is therefore "
        "expected coverage information, not a defect.",
        "",
        f"{len(parity)} EN/TH block pairs (Jev-aligned) judged. {len(low)} "
        f"differ on facts: {len(drift)} on pairs aligned at "
        f"{CONFIDENT_ALIGN:.2f} or better (the Thai block is abridged), "
        f"{len(weak)} on weakly aligned pairs (the blocks probably do not "
        "correspond at all, so read nothing into those).",
        "",
        "### Where the Thai text carries fewer facts",
        "",
    ]
    if not drift:
        lines.append("None.")
    for p in drift:
        lines.append(
            f"- parity {p.parity:.2f} (aligned {p.align_probability:.2f}): "
            f"{p.pair.en.path} L{p.pair.en.line_start} "
            f"vs {p.pair.th.path} L{p.pair.th.line_start}"
        )
        lines.append(f"  > EN: {_snippet(p.pair.en.text)}")
        lines.append(f"  > TH: {_snippet(p.pair.th.text)}")
    return lines


def render_markdown(
    *,
    pages: Sequence[PageSummary],
    judgments: Sequence[BlockJudgment],
    parity: Sequence[ParityJudgment],
    calibration: Mapping[str, Any] | None,
) -> str:
    tokens = sum(j.input_tokens for j in judgments)
    out = [
        "# Docs judge report",
        "",
        "Judge: TypeSafe Jev (typed questions per block). Verdicts are one "
        "Choice per block, pressure-tested with three refutation Nouls when "
        "not reads_human (a fully refuted rewrite becomes light_edit); tells "
        "are Nouls at or above the threshold; mechanical tells are counted in "
        "code. No composite score.",
        "",
        f"Blocks judged: {len(judgments)}. Input tokens: {tokens}.",
        "",
        "## Totals by language",
        "",
        *_language_totals(pages),
        "",
        "## Pages ranked by rewrite count",
        "",
        *_summary_table(pages),
        "",
    ]
    title_case = [(p.path, h) for p in pages for h in p.title_case_headings]
    if title_case:
        out += ["## Title-case headings", ""]
        out += [f"- {path}: {heading}" for path, heading in title_case]
        out.append("")
    if calibration is not None:
        out += [
            "## Calibration",
            "",
            "```json",
            json.dumps(calibration, ensure_ascii=False, indent=1),
            "```",
            "",
        ]
    if parity:
        out += ["## EN/TH parity", "", *_parity_lines(parity), ""]
    out += ["## Flagged blocks", "", *_block_lines(judgments)]
    return "\n".join(out).rstrip() + "\n"


def to_json(
    *,
    pages: Sequence[PageSummary],
    judgments: Sequence[BlockJudgment],
    parity: Sequence[ParityJudgment],
    calibration: Mapping[str, Any] | None,
) -> str:
    return json.dumps(
        {
            "pages": [asdict(p) for p in pages],
            "judgments": [asdict(j) for j in judgments],
            "parity": [asdict(p) for p in parity],
            "calibration": calibration,
        },
        ensure_ascii=False,
        indent=1,
    )
