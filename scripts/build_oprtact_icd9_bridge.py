"""Build the packaged OPRTACT->ICD9 bridge CSV for the pre-op reservation arm.

Reads the curated ``Mapping INCPT ICD9.csv`` export (one row per OPRTACT
billing code with three score-ranked ICD-9 candidates and a 0-indexed
``Human suggestion`` selector) and emits the small packaged
``oprtact_icd9_bridge.csv`` consumed by ``bba.preop_reservation.bridge``.

Selection policy (spec #196, user rulings 2026-07-18): the bridge code is the
highest-score **First Choice** cell, never the ``Human suggestion`` pick — the
Human column produces clinically wrong MSBOS hits that would hard-flip real
orders. The human pick is recorded as provenance (``human_index`` plus
``human_agreed``) so the verdict-gate's disagreement guard can consult it.

Row rules:
  * rows with a blank ``First Choice`` are skipped (no bridge candidate)
  * rows with a blank ``OPRTACT`` key are excluded from output (reported, not
    fatal — the real export contains exactly one)
  * duplicate OPRTACT keys collapse to the first occurrence when their
    First-Choice (icd9, raw score string) agree; a genuine disagreement fails
    the build loudly
  * any First-Choice cell not matching ``<icd9>::<name> (score=<float>)``
    fails the build loudly (guards future re-exports)

Run:  uv run python scripts/build_oprtact_icd9_bridge.py
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

RAW_FILENAME = "Mapping INCPT ICD9.csv"
OUT_FILENAME = "oprtact_icd9_bridge.csv"
REQUIRED_COLUMNS = ("OPRTACT", "First Choice", "Human suggestion")
OUTPUT_COLUMNS = (
    "oprtact",
    "icd9",
    "icd9_nodot",
    "score",
    "human_index",
    "human_agreed",
    "name",
)

_FIRST_CHOICE_RE = re.compile(r"\(score=([^()]*)\)\s*$")


class BridgeBuildError(ValueError):
    """The raw mapping export is malformed and the bridge cannot be built."""


@dataclass
class BuildCounts:
    """Stderr-reported audit counters for one build run."""

    eligible: int = 0
    blank_key_skipped: int = 0
    collapsed_duplicates: int = 0
    conflicts: list[str] = field(default_factory=list)


def parse_first_choice(cell: str) -> tuple[str, str, str]:
    """Parse a ``<icd9>::<name> (score=<float>)`` cell to its three parts.

    The raw score string is returned verbatim (no float round-trip) so the
    emitted CSV is byte-stable across rebuilds.
    """
    icd9, sep, rest = cell.partition("::")
    if not sep:
        raise BridgeBuildError(f"First Choice cell missing '::': {cell!r}")
    icd9 = icd9.strip()
    if not icd9:
        raise BridgeBuildError(f"First Choice cell has empty icd9: {cell!r}")
    score_match = _FIRST_CHOICE_RE.search(rest)
    if score_match is None:
        raise BridgeBuildError(f"First Choice cell missing (score=...): {cell!r}")
    score = score_match.group(1).strip()
    try:
        value = float(score)
    except ValueError as exc:
        raise BridgeBuildError(
            f"First Choice cell has non-float score: {cell!r}"
        ) from exc
    if not math.isfinite(value):
        raise BridgeBuildError(f"First Choice cell has non-finite score: {cell!r}")
    name = rest[: score_match.start()].strip()
    return icd9, name, score


def build_rows(
    records: Iterable[Mapping[str, str]],
) -> tuple[list[dict[str, str]], BuildCounts]:
    """Apply the row rules to raw records; return sorted output rows + counts."""
    counts = BuildCounts()
    by_key: dict[str, dict[str, str]] = {}
    for record in records:
        first_choice = (record.get("First Choice") or "").strip()
        if not first_choice:
            continue
        oprtact = (record.get("OPRTACT") or "").strip()
        if not oprtact:
            counts.blank_key_skipped += 1
            continue
        counts.eligible += 1
        icd9, name, score = parse_first_choice(first_choice)
        human_index = (record.get("Human suggestion") or "").strip()
        row = {
            "oprtact": oprtact,
            "icd9": icd9,
            "icd9_nodot": icd9.replace(".", ""),
            "score": score,
            "human_index": human_index,
            "human_agreed": "true" if human_index == "0" else "false",
            "name": name,
        }
        existing = by_key.get(oprtact)
        if existing is None:
            by_key[oprtact] = row
            continue
        if (existing["icd9"], existing["score"]) == (icd9, score):
            counts.collapsed_duplicates += 1
            continue
        raise BridgeBuildError(
            f"duplicate OPRTACT {oprtact} with conflicting First Choice: "
            f"kept ({existing['icd9']}, score={existing['score']}) vs "
            f"new ({icd9}, score={score})"
        )
    return [by_key[key] for key in sorted(by_key)], counts


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=repo_root.parent / "Bloodbank" / "data" / "raw",
        help=f"Folder holding {RAW_FILENAME} (default: Bloodbank/data/raw).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "src" / "bba" / "preop_reservation" / "data",
        help="Where to write the packaged CSV (default: the package data dir).",
    )
    args = ap.parse_args()

    src = args.raw_dir / RAW_FILENAME
    if not src.is_file():
        sys.stderr.write(f"ERROR: raw mapping export not found: {src}\n")
        sys.exit(1)

    with src.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            sys.stderr.write(f"ERROR: missing required columns: {missing}\n")
            sys.exit(1)
        try:
            rows, counts = build_rows(reader)
        except BridgeBuildError as exc:
            sys.stderr.write(f"ERROR: {exc}\n")
            sys.exit(1)

    out = args.out_dir / OUT_FILENAME
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"[eligible]   rows with OPRTACT + First Choice: {counts.eligible}",
        file=sys.stderr,
    )
    print(
        f"[blank-key]  rows excluded for blank OPRTACT: {counts.blank_key_skipped}",
        file=sys.stderr,
    )
    print(
        f"[collapsed]  identical duplicate keys collapsed: {counts.collapsed_duplicates}",
        file=sys.stderr,
    )
    print(f"[output]     bridge rows written: {len(rows)}", file=sys.stderr)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
