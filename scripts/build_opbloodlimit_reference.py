"""Build the OPBloodLimit MSBOS reference CSVs for the pre-op blood-reservation arm.

Reads the four specialty sheets of ``OPBloodLimit.xlsx`` (the hospital's Maximum
Surgical Blood Ordering Schedule) and emits two CSVs beside the source:

  * ``OPBloodLimit.csv``          one row per operation; ICD-9 codes joined with "; "
  * ``OPBloodLimit_by_icd9.csv``  exploded to one row per (operation, ICD-9 code),
                                  keyed on the dotless ``icd9_code_nodot`` for
                                  joining against ``ICD9CM.csv`` / the operative
                                  tables (which store codes dotless, e.g. ``0602``).

The raw ``.xlsx`` is treated as an immutable hospital export: source quirks and the
two known data-entry errors are corrected *here*, in code, with an audit trail --
never by mutating the workbook.

Source quirks handled:
  * merged procedure-group cells (forward-fill down col A)
  * the extra Thai header row in the ``Sx`` sheet
  * ICD-9 codes spread across many trailing columns
  * ICD-9 codes stored as floats (OB-Gyn) with spurious trailing zeros
  * sub-10 codes that lost their leading zero (a numeric 6.02, or the string
    "4.73") are re-padded to a two-digit integer part before the dotless join
    key is formed -- otherwise 6.02 -> "602" would collide with the code 60.2
  * two Ortho rows where Excel mis-parsed a typed "1-2" unit range into a date

A supplement CSV (default: the vendored ``OPBloodLimit_supplement.csv`` beside
the package copy of the exploded file) adds surgeon-confirmed ICD-9 codes to
EXISTING operation rows -- codes the hospital workbook does not list but a
surgeon has mapped to one of its operations. Each supplement row names the
target operation (``sheet``, ``procedure_group``, ``operation``) and must agree
with that row's MSBOS token (and, for G/M rows, its units); a row that matches
no operation, or disagrees, aborts the build. A blank ``procedure_group`` means
"every group listing this operation", allowed only when they all share one
recommendation. The workbook itself is never edited.

Operations with no ICD-9 code in the source are KEPT in the exploded file with
blank code columns (they cannot be matched by code -- the arm must match them by
operation description). ``ICD9CM.csv`` coverage is reported for information only
and is never a reason to drop a row: that master may itself be incomplete.

Run:  uv run python scripts/build_opbloodlimit_reference.py
"""

from __future__ import annotations

import argparse
import csv
import datetime
import sys
from collections import defaultdict
from pathlib import Path
from typing import cast

import openpyxl

SPECIALTY = {
    "Sx": "General Surgery",
    "Ortho": "Orthopedics",
    "ENT": "ENT (Otolaryngology)",
    "OB-Gyn": "Obstetrics & Gynecology",
}
MEANING = {
    "T/S": "Type and Screen",
    "G/M": "Group and Match (crossmatch)",
    "none": "No blood preparation",
}
HEADER_OPS = {"OPERATION", "หัตถการ"}  # English + the extra Sx Thai header row

# Verified data-entry correction, applied to the canonical dotted code produced by
# fmt_icd. The target is a *structurally invalid* code (not merely absent from the
# ICD-9-CM master), cross-checked against ICD9CM.csv:
#   06.40 -> 06.4   No 4-digit 06.40 exists; 06.4 = "Complete thyroidectomy", and
#                   the ENT sheet already writes this same code correctly as 06.4.
# (The former "4.73" -> "04.73" fix -- a dropped leading zero -- is now handled
# generally by fmt_icd's leading-zero canonicalisation, so it is not special-cased.)
ICD_CORRECTIONS = {"06.40": "06.4"}

DEFAULT_SUPPLEMENT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "bba"
    / "preop_reservation"
    / "data"
    / "OPBloodLimit_supplement.csv"
)

GROUPED_COLS = [
    "specialty",
    "sheet",
    "procedure_group",
    "operation",
    "msbos",
    "msbos_meaning",
    "recommended_units",
    "icd9_codes",
]
EXPLODED_COLS = [
    "icd9_code",
    "icd9_code_nodot",
    "specialty",
    "sheet",
    "procedure_group",
    "operation",
    "msbos",
    "msbos_meaning",
    "recommended_units",
]


def fmt_icd(v: object) -> str:
    """Render a raw ICD-9 cell to a canonical dotted code string.

    Floats (OB-Gyn stores codes numerically) are capped at two decimals -- the
    ICD-9 maximum -- then stripped of spurious trailing zeros so a genuine
    3-digit code like 74.1 does not masquerade as the 4-digit 74.10. A single
    significant ``.0`` is kept, though: 74.0 (Classical cesarean) is a distinct
    procedure code, not the category 74.

    ICD-9 procedure codes always carry a two-digit integer part, so a 1-digit
    integer part means a leading zero was lost -- a numeric 6.02 for 06.02, or the
    string "4.73" for 04.73. It is zero-padded back to two digits so the downstream
    dotless key is correct; otherwise 6.02 -> "602" would collide with the distinct
    code 60.2. Idempotent on already-canonical values (06.02 stays 06.02).
    """
    if isinstance(v, bool):
        return ""
    if isinstance(v, float):
        s = ("%.2f" % v).rstrip("0")
        if s.endswith("."):
            s += "0"  # keep a significant .0 (74.0 is a code, distinct from 74)
    elif isinstance(v, int):
        s = str(v)
    else:
        s = str(v).strip()
    if not s:
        return ""
    intpart, dot, frac = s.partition(".")
    if intpart.isdigit() and len(intpart) < 2:
        intpart = intpart.zfill(2)
    return f"{intpart}{dot}{frac}"


def fmt_units(v: object) -> str:
    """Render the (unlabeled) recommended-units cell."""
    if v is None:
        return ""
    if isinstance(v, (datetime.datetime, datetime.date)):
        # Excel mis-parsed a typed "N-M" unit range into a date (day=N, month=M).
        return f"{v.day}-{v.month}"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else str(v)
    return str(v).strip()


def norm_msbos(v: object) -> str:
    """Normalise the MSBOS token; fold the 'None'/'none' variants to 'none'."""
    if v is None:
        return ""
    s = str(v).strip()
    return "none" if s.lower() == "none" else s


def parse(src: Path, applied: dict[str, int]) -> list[dict[str, object]]:
    """Parse every sheet into per-operation records with de-duped dotted codes.

    ``applied`` is mutated to count how many times each ICD_CORRECTIONS key fired,
    so a correction that no longer matches the source (fixed upstream) is visible.
    """
    wb = openpyxl.load_workbook(src, read_only=True, data_only=True)
    ops: list[dict[str, object]] = []
    for sheet in wb.sheetnames:
        group = ""
        for r in wb[sheet].iter_rows(min_row=1, values_only=True):
            op = r[1]
            if op is None or str(op).strip() == "" or str(op).strip() in HEADER_OPS:
                continue
            if r[0] is not None and str(r[0]).strip() != "":
                group = str(r[0]).strip()
            codes: list[str] = []
            for c in range(4, len(r)):
                if r[c] is not None and str(r[c]).strip() != "":
                    dc = fmt_icd(r[c])
                    if dc in ICD_CORRECTIONS:
                        applied[dc] = applied.get(dc, 0) + 1
                        dc = ICD_CORRECTIONS[dc]
                    if dc and dc not in codes:
                        codes.append(dc)
            msbos = norm_msbos(r[2])
            ops.append(
                {
                    "specialty": SPECIALTY.get(sheet, sheet),
                    "sheet": sheet,
                    "procedure_group": group,
                    "operation": str(op).strip(),
                    "msbos": msbos,
                    "msbos_meaning": MEANING.get(msbos, ""),
                    "recommended_units": fmt_units(r[3]),
                    "codes": codes,
                }
            )
    return ops


SUPPLEMENT_REQUIRED = frozenset(
    {"sheet", "procedure_group", "operation", "icd9_code", "msbos", "recommended_units"}
)


class SupplementError(ValueError):
    """A supplement row cannot be applied; the build must not silently continue."""


def load_supplement(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = SUPPLEMENT_REQUIRED.difference(reader.fieldnames or ())
        if missing:
            raise SupplementError(
                f"{path.name}: missing columns {', '.join(sorted(missing))}"
            )
        return [{k: (v or "").strip() for k, v in row.items()} for row in reader]


def _supplement_targets(
    ops: list[dict[str, object]], row: dict[str, str], where: str
) -> list[int]:
    """Indices of the operation rows a supplement row applies to (fail loud)."""
    hits = [
        i
        for i, o in enumerate(ops)
        if o["sheet"] == row["sheet"]
        and o["operation"] == row["operation"]
        and (
            not row["procedure_group"] or o["procedure_group"] == row["procedure_group"]
        )
    ]
    if not hits:
        raise SupplementError(f"{where}: no operation matches {row!r}")
    if len(hits) > 1 and row["procedure_group"]:
        raise SupplementError(f"{where}: operation is not unique: {row!r}")
    recs = {(ops[i]["msbos"], ops[i]["recommended_units"]) for i in hits}
    if len(recs) > 1:
        raise SupplementError(
            f"{where}: matched groups disagree on (msbos, units) {sorted(recs)}: {row!r}"
        )
    msbos, units = recs.pop()
    if norm_msbos(row["msbos"]) != msbos:
        raise SupplementError(
            f"{where}: msbos {row['msbos']!r} disagrees with reference {msbos!r}: {row!r}"
        )
    # T/S units are meaningless and ignored by the loader; every other token
    # must agree on units so the supplement cannot quietly change a tariff.
    if msbos != "T/S" and row["recommended_units"] != units:
        raise SupplementError(
            f"{where}: units {row['recommended_units']!r} disagree with "
            f"reference {units!r}: {row!r}"
        )
    return hits


def apply_supplement(
    ops: list[dict[str, object]], rows: list[dict[str, str]]
) -> tuple[list[dict[str, object]], int]:
    """Return a new ops list with the supplement codes added, plus the count added."""
    codes = [list(cast(list[str], o["codes"])) for o in ops]
    added = 0
    for n, row in enumerate(rows, start=2):
        where = f"supplement row {n}"
        code = fmt_icd(row["icd9_code"])
        if not code:
            raise SupplementError(f"{where}: blank icd9_code")
        for i in _supplement_targets(ops, row, where):
            if code in codes[i]:
                raise SupplementError(
                    f"{where}: {code} already listed on {ops[i]['operation']!r}"
                )
            codes[i].append(code)
            added += 1
    return [{**o, "codes": c} for o, c in zip(ops, codes, strict=True)], added


def write_grouped(ops: list[dict[str, object]], out: Path) -> None:
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=GROUPED_COLS)
        w.writeheader()
        for o in ops:
            row = {k: o[k] for k in GROUPED_COLS if k != "icd9_codes"}
            row["icd9_codes"] = "; ".join(o["codes"])  # type: ignore[arg-type]
            w.writerow(row)


def build_exploded(ops: list[dict[str, object]]) -> list[dict[str, str]]:
    """Explode to one row per (operation, code); no-ICD operations kept with blanks."""
    rows: list[dict[str, str]] = []
    for o in ops:
        base: dict[str, str] = {
            "specialty": str(o["specialty"]),
            "sheet": str(o["sheet"]),
            "procedure_group": str(o["procedure_group"]),
            "operation": str(o["operation"]),
            "msbos": str(o["msbos"]),
            "msbos_meaning": str(o["msbos_meaning"]),
            "recommended_units": str(o["recommended_units"]),
        }
        codes: list[str] = o["codes"]  # type: ignore[assignment]
        if codes:
            for dc in codes:
                rows.append(
                    {"icd9_code": dc, "icd9_code_nodot": dc.replace(".", ""), **base}
                )
        else:
            rows.append({"icd9_code": "", "icd9_code_nodot": "", **base})
    # Coded rows first (sorted by dotless key), blank-code rows last.
    rows.sort(
        key=lambda x: (x["icd9_code_nodot"] == "", x["icd9_code_nodot"], x["operation"])
    )
    return rows


def write_exploded(rows: list[dict[str, str]], out: Path) -> None:
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=EXPLODED_COLS)
        w.writeheader()
        w.writerows(rows)


def load_ref_codes(ref: Path) -> set[str]:
    codes: set[str] = set()
    with ref.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = (row.get("Icd9cm") or "").strip()
            if c:
                codes.add(c)
    return codes


def report(
    ops: list[dict[str, object]],
    exploded: list[dict[str, str]],
    ref_codes: set[str],
    applied: dict[str, int],
) -> None:
    coded = [x for x in exploded if x["icd9_code_nodot"]]
    blank = [x for x in exploded if not x["icd9_code_nodot"]]
    distinct = {x["icd9_code_nodot"] for x in coded}

    # correction audit (fail loud if a mapping went stale)
    for key in ICD_CORRECTIONS:
        n = applied.get(key, 0)
        tag = "applied" if n else "STALE (no longer in source -- consider removing)"
        print(
            f"[correction] {key} -> {ICD_CORRECTIONS[key]}: {n}x {tag}", file=sys.stderr
        )

    # coverage: informational only -- never drops a row
    unmatched = sorted(
        {
            (x["icd9_code"], x["icd9_code_nodot"])
            for x in coded
            if x["icd9_code_nodot"] not in ref_codes
        }
    )
    print(
        f"[coverage] codes absent from ICD9CM.csv (kept anyway): {len(unmatched)}",
        file=sys.stderr,
    )
    for dotted, nodot in unmatched:
        print(f"    KEEP {nodot} ({dotted})", file=sys.stderr)

    # same code -> conflicting recommendation (arm must disambiguate by operation)
    bycode: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for x in coded:
        bycode[x["icd9_code_nodot"]].add((x["msbos"], x["recommended_units"]))
    conflicts = {k: v for k, v in bycode.items() if len(v) > 1}

    print(
        f"[blank]    operations with no ICD-9 code (kept, match by name): {len(blank)}",
        file=sys.stderr,
    )
    for x in blank:
        print(f"    {x['sheet']}: {x['operation']}", file=sys.stderr)
    print(
        f"[conflicts] codes with >1 (msbos,units): {len(conflicts)} of {len(distinct)} distinct",
        file=sys.stderr,
    )
    print(f"grouped rows : {len(ops)}", file=sys.stderr)
    print(
        f"exploded rows: {len(exploded)} ({len(coded)} coded + {len(blank)} blank)",
        file=sys.stderr,
    )


def main() -> None:
    default_raw = (
        Path(__file__).resolve().parents[1].parent / "Bloodbank" / "data" / "raw"
    )
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw,
        help="Folder holding OPBloodLimit.xlsx + ICD9CM.csv, and where the CSVs "
        "are written (default: Bloodbank/data/raw).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write the CSVs (default: same as --raw-dir).",
    )
    ap.add_argument(
        "--supplement",
        type=Path,
        default=DEFAULT_SUPPLEMENT,
        help="Surgeon-confirmed code additions applied to existing operation rows "
        "(default: the vendored OPBloodLimit_supplement.csv). Pass a missing "
        "path to build without it.",
    )
    args = ap.parse_args()

    raw_dir: Path = args.raw_dir
    out_dir: Path = args.out_dir or raw_dir
    src = raw_dir / "OPBloodLimit.xlsx"
    ref = raw_dir / "ICD9CM.csv"
    if not src.is_file():
        sys.stderr.write(f"ERROR: source workbook not found: {src}\n")
        sys.exit(1)

    applied: dict[str, int] = {}
    ops = parse(src, applied)
    if args.supplement.is_file():
        try:
            ops, added = apply_supplement(ops, load_supplement(args.supplement))
        except SupplementError as exc:
            sys.stderr.write(f"ERROR: {exc}\n")
            sys.exit(1)
        print(f"[supplement] {args.supplement}: {added} codes added", file=sys.stderr)
    else:
        print(f"[supplement] none ({args.supplement} not found)", file=sys.stderr)
    write_grouped(ops, out_dir / "OPBloodLimit.csv")
    exploded = build_exploded(ops)
    write_exploded(exploded, out_dir / "OPBloodLimit_by_icd9.csv")

    ref_codes = load_ref_codes(ref) if ref.is_file() else set()
    if not ref_codes:
        sys.stderr.write(
            f"WARN: ICD9CM.csv not found at {ref}; coverage check skipped\n"
        )
    report(ops, exploded, ref_codes, applied)
    print(
        f"wrote {out_dir / 'OPBloodLimit.csv'} and {out_dir / 'OPBloodLimit_by_icd9.csv'}"
    )


if __name__ == "__main__":
    main()
