"""Sample N RBC orders from an encrypted HOSxP bundle and write a mini bundle.

Strategy:

1. Read BDVST + BDVSTDT from ``$BBA_PILOT_RAW_DIR``. Keep orders whose
   line items contain at least one RBC product (BDTYPE in
   {LPRC, LDPRC, SDR}), with BDVSTST in {4, 5}, REQTYPE == 'P',
   CANCELDATE NULL, AN non-null.
2. Random-sample N (HN, REQNO) keys (seed configurable so reruns are
   reproducible).
3. Project each table to rows that match the sampled keys, into
   ``$BBA_PILOT_WORK_DIR/bundle/``. Blood-bank order tables keep only
   sampled REQNOs so pipeline outputs remain manifest-scoped; review-only
   sidecar tables carry same-admission rows for related platelet/FFP/PRC
   requests under neighboring REQNOs.
4. Carry the dictionaries (BDVSTST, BDTYPE, ICD9CM) whole — they are
   small and the audit pipeline needs them for lookups.
5. Write the sample manifest alongside the bundle for traceability.

Environment variables:

* ``BBA_PILOT_RAW_DIR`` — directory containing the encrypted HOSxP
  CSVs (default: ``../Bloodbank/data/encrypted`` relative to repo root).
* ``BBA_PILOT_WORK_DIR`` — output directory (default: ``/tmp/bba_mini``).
* ``BBA_PILOT_SAMPLE_N`` — number of orders to sample (default: 10).
* ``BBA_PILOT_SAMPLE_SEED`` — RNG seed (default: 20260519).
"""

from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path

SRC = Path(
    os.environ.get(
        "BBA_PILOT_RAW_DIR",
        str(
            Path(__file__).resolve().parents[2].parent
            / "Bloodbank"
            / "data"
            / "encrypted"
        ),
    )
)
RAW_SRC = SRC.parent / "raw"
WORK = Path(os.environ.get("BBA_PILOT_WORK_DIR", "/tmp/bba_mini"))
DST = WORK / "bundle"
N = int(os.environ.get("BBA_PILOT_SAMPLE_N", "10"))
SEED = int(os.environ.get("BBA_PILOT_SAMPLE_SEED", "20260519"))

RBC = {"LPRC", "LDPRC", "SDR"}
ELIGIBLE_STATUS = {"4", "5"}

csv.field_size_limit(sys.maxsize)


BDVST_COLS = [
    "HN",
    "REQNO",
    "AN",
    "BDVSTST",
    "REQTYPE",
    "CANCELDATE",
    "REQDATE",
    "REQTIME",
    "BDVSTDATE",
    "BDVSTTIME",
    "PICKDATE",
    "PICKTIME",
    "ICD10",
    "DIAGNOSIS",
]
BDVSTDT_COLS = [
    "REQNO",
    "HN",
    "BDVSTDATE",
    "BDVSTTIME",
    "USEDATE",
    "USETIME",
    "BDTYPE",
    "ITEMNO",
    "UNITAMT",
]
BDVSTTRANS_COLS = [
    "HN",
    "AN",
    "BDTYPE",
    "GIVEDATE",
    "GIVETIME",
    "PAYDATE",
    "PAYTIME",
    "RTNDATE",
    "RTNTIME",
    "OFFDATE",
    "QTYUSE",
    "UNITSTAT",
    "PAYOUTFLAG",
    "PAYCOMM",
]


def _filter(src_name: str, dst_name: str, predicate, *, cols=None) -> int:
    n_in = n_out = 0
    with (SRC / src_name).open(encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        in_header = reader.fieldnames or []
        out_header = cols if cols is not None else in_header
        with (DST / dst_name).open("w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_header, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                n_in += 1
                if predicate(row):
                    n_out += 1
                    writer.writerow({c: row.get(c, "") for c in out_header})
    print(f"  {src_name:32s} -> {dst_name:24s} {n_out:>8d}/{n_in:<8d}")
    return n_out


def _copy(src_name: str, dst_name: str) -> int:
    n = 0
    with (
        (SRC / src_name).open(encoding="utf-8", newline="") as fin,
        (DST / dst_name).open("w", encoding="utf-8", newline="") as fout,
    ):
        for line in fin:
            fout.write(line)
            n += 1
    print(f"  {src_name:32s} -> {dst_name:24s} {n - 1:>8d} rows")
    return n - 1


def _copy_first_available(src_names: tuple[Path, ...], dst_name: str) -> int:
    for src_path in src_names:
        if not src_path.exists():
            continue
        n = 0
        with (
            src_path.open(encoding="utf-8", newline="") as fin,
            (DST / dst_name).open("w", encoding="utf-8", newline="") as fout,
        ):
            for line in fin:
                fout.write(line)
                n += 1
        print(f"  {src_path.name:32s} -> {dst_name:24s} {n - 1:>8d} rows")
        return n - 1
    return 0


def main() -> None:
    if not SRC.exists():
        sys.exit(f"BBA_PILOT_RAW_DIR not found: {SRC}")
    DST.mkdir(parents=True, exist_ok=True)
    print(f"raw  : {SRC}")
    print(f"work : {WORK}")
    print(f"N={N}, seed={SEED}")

    # Pass 1 — index BDVSTDT REQNOs that carry at least one RBC line item.
    rbc_reqnos: set[str] = set()
    with (SRC / "BDVSTDT.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("BDTYPE", "").strip().upper() in RBC:
                rbc_reqnos.add(row["REQNO"])
    print(f"BDVSTDT: {len(rbc_reqnos)} REQNOs carry at least one RBC line item")

    # Pass 2 — scan BDVST, keep eligible RBC orders.
    candidates: list[tuple[str, str, str]] = []
    with (SRC / "BDVST.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["REQNO"] not in rbc_reqnos:
                continue
            if row.get("REQTYPE", "").strip() != "P":
                continue
            if row.get("BDVSTST", "").strip() not in ELIGIBLE_STATUS:
                continue
            if row.get("CANCELDATE", "").strip():
                continue
            an = (row.get("AN") or "").strip()
            hn = (row.get("HN") or "").strip()
            if not an or not hn:
                continue
            candidates.append((hn, row["REQNO"], an))
    print(f"BDVST: {len(candidates)} eligible RBC orders")
    if len(candidates) < N:
        sys.exit(f"only {len(candidates)} candidates < N={N}")

    rng = random.Random(SEED)
    sample = rng.sample(candidates, N)
    sample_reqnos = {r for _, r, _ in sample}
    sample_pairs = {(h, a) for h, _, a in sample}
    sample_hns = {hn for hn, _, _ in sample}
    sample_ans = {a for _, _, a in sample}
    related_reqnos = set(sample_reqnos)
    with (SRC / "BDVST.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            hn = (row.get("HN") or "").strip()
            an = (row.get("AN") or "").strip()
            if (hn, an) in sample_pairs:
                related_reqnos.add(row["REQNO"])
    print("\nSampled (HN, REQNO, AN):")
    for s in sample:
        print(" ", s)

    print("\nWriting mini bundle:")
    _filter(
        "BDVST.csv",
        "BDVST.csv",
        lambda r: r["REQNO"] in sample_reqnos,
        cols=BDVST_COLS,
    )
    _filter(
        "BDVSTDT.csv",
        "BDVSTDT.csv",
        lambda r: r["REQNO"] in sample_reqnos,
        cols=BDVSTDT_COLS,
    )
    _filter(
        "BDVST.csv",
        "BDVST_RELATED.csv",
        lambda r: r["REQNO"] in related_reqnos,
        cols=BDVST_COLS,
    )
    _filter(
        "BDVSTDT.csv",
        "BDVSTDT_RELATED.csv",
        lambda r: r["REQNO"] in related_reqnos,
        cols=BDVSTDT_COLS,
    )
    _filter("Diagnosis.csv", "Diagnosis.csv", lambda r: r.get("AN") in sample_ans)
    _filter("Lab.csv", "Lab.csv", lambda r: r.get("AN") in sample_ans)
    _filter("Med.csv", "Med.csv", lambda r: r.get("AN") in sample_ans)
    _filter(
        "IPDADMPROGRESS.csv", "IPDADMPROGRESS.csv", lambda r: r.get("AN") in sample_ans
    )
    _filter("IPDNRFOCUSDT.csv", "IPDNRFOCUSDT.csv", lambda r: r.get("AN") in sample_ans)
    # Procedure-family exports have arrived as both Title-Case and ALL-CAPS.
    _filter(
        "IPTSUMOPRT.csv",
        "IPTSUMOPRT.csv",
        lambda r: (r.get("An") or r.get("AN")) in sample_ans,
    )
    if (SRC / "IPDDCHSUMOPRT.csv").exists():
        _filter(
            "IPDDCHSUMOPRT.csv",
            "IPDDCHSUMOPRT.csv",
            lambda r: (r.get("An") or r.get("AN")) in sample_ans,
        )
    if (SRC / "INCPT_OPRTACT.csv").exists():
        _filter(
            "INCPT_OPRTACT.csv",
            "INCPT_OPRTACT.csv",
            lambda r: (
                (
                    (r.get("Hn") or r.get("HN") or ""),
                    (r.get("An") or r.get("AN") or ""),
                )
                in sample_pairs
            ),
        )
    elif (SRC / "INCPT.csv").exists():
        _filter(
            "INCPT.csv",
            "INCPT.csv",
            lambda r: (
                (
                    (r.get("Hn") or r.get("HN") or ""),
                    (r.get("An") or r.get("AN") or ""),
                )
                in sample_pairs
            ),
        )
    if (SRC / "BDVSTTRANS.csv").exists():
        _filter(
            "BDVSTTRANS.csv",
            "BDVSTTRANS.csv",
            lambda r: (r.get("AN") or r.get("An")) in sample_ans
            or (r.get("HN") or r.get("Hn")) in sample_hns,
            cols=BDVSTTRANS_COLS,
        )

    _copy("BDTYPE.csv", "BDTYPE.csv")
    _copy("BDVSTST.csv", "BDVSTST.csv")
    _copy("ICD9CM.csv", "ICD9CM.csv")
    _copy_first_available(
        (SRC / "OPRTACT.csv", RAW_SRC / "OPRTACT.csv"),
        "OPRTACT.csv",
    )

    manifest = WORK / "sample_manifest.csv"
    with manifest.open("w", encoding="utf-8") as fh:
        fh.write("HN,REQNO,AN\n")
        for s in sample:
            fh.write(",".join(s) + "\n")
    print(f"\nManifest: {manifest}")


if __name__ == "__main__":
    main()
