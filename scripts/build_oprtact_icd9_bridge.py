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
ICD9CM_FILENAME = "ICD9CM.csv"
OUT_FILENAME = "oprtact_icd9_bridge.csv"
REQUIRED_COLUMNS = (
    "OPRTACT",
    "First Choice",
    "Second choice",
    "Third choice",
    "Human suggestion",
)
# Which raw choice column a disagreeing Human suggestion index selects. Index
# "0" is agreement (the First Choice itself); indexes beyond Top-3 ("3"/"4")
# point at nothing and are treated as no-selection.
SELECTED_CHOICE_COLUMNS = {"1": "Second choice", "2": "Third choice"}
OUTPUT_COLUMNS = (
    "oprtact",
    "icd9",
    "icd9_nodot",
    "score",
    "human_index",
    "human_agreed",
    "human_icd9",
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
    leading_zero_restored: int = 0
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


def canonical_nodot(code: str, name: str, icd9cm_names: Mapping[str, str]) -> str:
    """Restore a 2-digit-category leading zero the raw export stripped.

    ``Mapping INCPT ICD9.csv`` stores codes dotless (``'309'``), so a code whose
    true category is ``0X`` (03.09 -> ``'0309'``) arrives with the zero gone
    (``'309'``, which reads as 30.9). The dotless string alone is ambiguous, so
    the operation NAME disambiguates against the ICD-9-CM master: prefer the
    zero-padded form only when IT matches the name and the raw form does not.
    Legitimate 3-digit codes (55.4 -> ``'554'``, ``'Partial nephrectomy'``) are
    left untouched. Absent an ICD-9-CM map (unit tests), the code is unchanged.
    """
    raw = code.strip()
    if not raw or not icd9cm_names:
        return raw
    name_norm = name.strip().casefold()
    padded = "0" + raw
    padded_ok = icd9cm_names.get(padded, "").strip().casefold() == name_norm
    raw_ok = icd9cm_names.get(raw, "").strip().casefold() == name_norm
    if padded_ok and not raw_ok:
        return padded
    return raw


def load_icd9cm_names(path: Path) -> dict[str, str]:
    """Map dotless ICD-9-CM procedure code -> canonical name (first wins)."""
    names: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for record in csv.DictReader(f):
            code = (record.get("Icd9cm") or "").strip()
            if code:
                names.setdefault(code, (record.get("Name") or "").strip())
    return names


def build_rows(
    records: Iterable[Mapping[str, str]],
    icd9cm_names: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, str]], BuildCounts]:
    """Apply the row rules to raw records; return sorted output rows + counts.

    ``icd9cm_names`` maps dotless ICD-9 code -> canonical name; when supplied it
    drives :func:`canonical_nodot` to restore leading zeros the raw export lost.
    Omitted (unit tests), codes pass through verbatim.
    """
    icd9cm_names = icd9cm_names or {}
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
        nodot = canonical_nodot(icd9.replace(".", ""), name, icd9cm_names)
        if nodot != icd9.replace(".", ""):
            counts.leading_zero_restored += 1
        # The raw export is dotless, so the canonical code IS the nodot form; a
        # (test-only) dotted source keeps its dotted icd9 while nodot normalizes.
        icd9_out = icd9 if "." in icd9 else nodot
        human_index = (record.get("Human suggestion") or "").strip()
        human_icd9 = ""
        selected_column = SELECTED_CHOICE_COLUMNS.get(human_index)
        if selected_column is not None:
            selected_cell = (record.get(selected_column) or "").strip()
            if selected_cell:
                # Malformed non-blank cells fail loud via the shared grammar;
                # a blank cell means the index points at nothing (no-selection,
                # same handling as out-of-range indexes).
                h_icd9, h_name, _ = parse_first_choice(selected_cell)
                human_icd9 = canonical_nodot(
                    h_icd9.replace(".", ""), h_name, icd9cm_names
                )
        row = {
            "oprtact": oprtact,
            "icd9": icd9_out,
            "icd9_nodot": nodot,
            "score": score,
            "human_index": human_index,
            "human_agreed": "true" if human_index == "0" else "false",
            "human_icd9": human_icd9,
            "name": name,
        }
        existing = by_key.get(oprtact)
        if existing is None:
            by_key[oprtact] = row
            continue
        if (existing["icd9"], existing["score"]) == (icd9_out, score):
            counts.collapsed_duplicates += 1
            continue
        raise BridgeBuildError(
            f"duplicate OPRTACT {oprtact} with conflicting First Choice: "
            f"kept ({existing['icd9']}, score={existing['score']}) vs "
            f"new ({icd9_out}, score={score})"
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

    icd9cm_src = args.raw_dir / ICD9CM_FILENAME
    if not icd9cm_src.is_file():
        sys.stderr.write(
            f"ERROR: ICD-9-CM master not found (needed to restore leading zeros): "
            f"{icd9cm_src}\n"
        )
        sys.exit(1)
    icd9cm_names = load_icd9cm_names(icd9cm_src)

    with src.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            sys.stderr.write(f"ERROR: missing required columns: {missing}\n")
            sys.exit(1)
        try:
            rows, counts = build_rows(reader, icd9cm_names)
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
    print(
        f"[zero-fix]   category leading zeros restored: {counts.leading_zero_restored}",
        file=sys.stderr,
    )
    print(f"[output]     bridge rows written: {len(rows)}", file=sys.stderr)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
