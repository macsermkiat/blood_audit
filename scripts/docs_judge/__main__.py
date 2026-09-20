"""CLI: judge the docs corpus with Jev and write a report.

    uv run python -m docs_judge run --out DIR [--limit N] [--dry-run] [--no-parity]
    uv run python -m docs_judge calibrate --out DIR

Run from the repo root with PYTHONPATH=scripts. Reads TYPESAFE_API_KEY from
the environment, falling back to a .env file in the working directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .blocks import Block, split_blocks
from .calibrate import run_calibration
from .jev import JevClient, TransportResponse
from .judge import (
    BlockJudgment,
    PageSummary,
    ParityJudgment,
    audience_for,
    judge_blocks,
    judge_parity,
    language_for,
    pair_blocks,
    rollup,
)
from .prechecks import heading_title_case
from .pressure import pressure_test
from .report import render_markdown, to_json

DOCS_ROOT = Path("docs/src/content/docs")
EXTRA_FILES = (Path("README.md"),)


def _load_env_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("TYPESAFE_API_KEY not set and not found in .env")


def _dry_transport(payload: bytes) -> TransportResponse:
    request = json.loads(payload)
    answers: dict[str, Any] = {}
    for qid, q in request["questions"].items():
        if q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": 0.0}
        elif q["type"] == "score":
            n = len(q["criteria"])
            answers[qid] = {
                "type": "score",
                "score": float(n - 1),
                "confidence": 1.0,
                "probabilities": {
                    str(i): (1.0 if i == n - 1 else 0.0) for i in range(n)
                },
            }
        else:
            first = next(iter(q["criteria"]))
            answers[qid] = {
                "type": "choice",
                "choice": first,
                "confidence": 1.0,
                "probabilities": {
                    k: (1.0 if k == first else 0.0) for k in q["criteria"]
                },
            }
    body = {"model": "dry-run", "answers": answers, "usage": {"input_tokens": 0}}
    return TransportResponse(200, json.dumps(body).encode(), {})


def _corpus_files(limit: int | None) -> list[Path]:
    files = sorted(DOCS_ROOT.rglob("*.mdx")) + [p for p in EXTRA_FILES if p.exists()]
    return files[:limit] if limit else files


def _split_all(files: Sequence[Path]) -> dict[str, tuple[Block, ...]]:
    return {str(f): split_blocks(f.read_text(encoding="utf-8"), str(f)) for f in files}


def _judge_corpus(
    client: JevClient, blocks_by_page: Mapping[str, Sequence[Block]], workers: int
) -> tuple[list[PageSummary], list[BlockJudgment]]:
    pages: list[PageSummary] = []
    judgments: list[BlockJudgment] = []
    for path, blocks in blocks_by_page.items():
        language = language_for(path)
        judged = tuple(
            j
            for j, _ in pressure_test(
                client,
                judge_blocks(
                    client, blocks, language, audience_for(path), max_workers=workers
                ),
                max_workers=workers,
            )
        )
        title_case = heading_title_case(Path(path).read_text(encoding="utf-8"))
        pages.append(rollup(path, judged, title_case))
        judgments.extend(judged)
        print(f"judged {path}: {len(judged)} blocks", file=sys.stderr)
    return pages, judgments


def _parity(
    client: JevClient, blocks_by_page: Mapping[str, Sequence[Block]], workers: int
) -> list[ParityJudgment]:
    out: list[ParityJudgment] = []
    for path, en_blocks in blocks_by_page.items():
        if "/en/" not in path:
            continue
        th_path = path.replace("/en/", "/th/")
        th_blocks = blocks_by_page.get(th_path)
        if th_blocks is None:
            continue
        pairs, _ = pair_blocks(en_blocks, th_blocks)
        out.extend(judge_parity(client, pairs, max_workers=workers))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="docs_judge")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--no-parity", action="store_true")
    run.add_argument("--no-calibration", action="store_true")
    run.add_argument("--workers", type=int, default=8)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--out", type=Path, required=True)
    cal.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    api_key = "dry" if args.dry_run else _load_env_key()
    client = JevClient(
        api_key,
        cache_path=args.out / "jev_cache.jsonl",
        transport=_dry_transport if args.dry_run else None,
    )
    blocks_by_page = _split_all(_corpus_files(getattr(args, "limit", None)))

    if args.command == "calibrate":
        result = run_calibration(client, blocks_by_page)
        (args.out / "calibration.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return 0

    calibration = (
        None if args.no_calibration else run_calibration(client, blocks_by_page)
    )
    pages, judgments = _judge_corpus(client, blocks_by_page, args.workers)
    parity = [] if args.no_parity else _parity(client, blocks_by_page, args.workers)
    (args.out / "report.md").write_text(
        render_markdown(
            pages=pages, judgments=judgments, parity=parity, calibration=calibration
        ),
        encoding="utf-8",
    )
    (args.out / "report.json").write_text(
        to_json(
            pages=pages, judgments=judgments, parity=parity, calibration=calibration
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out / 'report.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
