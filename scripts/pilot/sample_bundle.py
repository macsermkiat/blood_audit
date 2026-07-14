"""Sample N RBC orders from an encrypted HOSxP bundle and write a mini bundle.

Strategy:

1. Read BDVST + BDVSTDT from ``$BBA_PILOT_RAW_DIR``. Keep orders whose
   line items contain at least one RBC product (BDTYPE in
   {LPRC, LDPRC, SDR}), with BDVSTST in {4, 5}, REQTYPE == 'P',
   CANCELDATE NULL, AN non-null.
2. Random-sample N (HN, REQNO) keys (seed configurable so reruns are
   reproducible).
3. Optionally sample an additional M platelet orders using a SEPARATE
   ``random.Random`` instance seeded by ``BBA_PILOT_PLATELET_SEED``.
   The platelet RNG never touches the RBC RNG — the two instances are
   completely independent, so the RBC sample is byte-identical whether
   platelet sampling is active or not.
4. Project each table to rows that match the sampled keys (RBC union
   platelet), into ``$BBA_PILOT_WORK_DIR/bundle/``. Blood-bank order
   tables keep only sampled REQNOs so pipeline outputs remain
   manifest-scoped; review-only sidecar tables carry same-admission rows
   for related requests under neighboring REQNOs.
5. Carry the dictionaries (BDVSTST, BDTYPE, ICD9CM) whole — they are
   small and the audit pipeline needs them for lookups.
6. Write the sample manifest alongside the bundle, component-tagged
   (``rbc`` / ``platelet``) with each row's seed for traceability.

Environment variables:

* ``BBA_PILOT_RAW_DIR`` — directory containing the encrypted HOSxP
  CSVs (default: ``../Bloodbank/data/encrypted`` relative to repo root).
* ``BBA_PILOT_WORK_DIR`` — output directory (default: ``/tmp/bba_mini``).
* ``BBA_PILOT_SAMPLE_N`` — number of RBC orders to sample (default: 10).
* ``BBA_PILOT_SAMPLE_SEED`` — RBC RNG seed (default: 20260519).
* ``BBA_PILOT_PLATELET_SAMPLE_N`` — number of platelet orders to sample
  (default: 0, opt-in so the bundle is RBC-only unless explicitly set).
* ``BBA_PILOT_PLATELET_SEED`` — platelet RNG seed, independent from the
  RBC seed (default: 20260520).
"""

from __future__ import annotations

import csv
import os
import random
import sys
from pathlib import Path

from bba.component_map import is_platelet_product

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
PLATELET_N = int(os.environ.get("BBA_PILOT_PLATELET_SAMPLE_N", "0"))
PLATELET_SEED = int(os.environ.get("BBA_PILOT_PLATELET_SEED", "20260520"))

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
# BDVSTTRANS is a unit-level returns ledger keyed by REQNO. The complete export
# also carries HN/AN identifiers, per-unit DNRNO/SEQNO (the physical-unit key
# summarize_returns dedups lifecycle rows on), and GIVEDATE/GIVETIME (the
# transfusion time, present iff UNITSTAT=5). Names match the raw UPPERCASE
# header; _filter resolves them case-insensitively so a header-casing change
# does not silently blank the columns.
BDVSTTRANS_COLS = [
    "REQNO",
    "HN",
    "AN",
    "BDTYPE",
    "DNRNO",
    "SEQNO",
    "UNITSTAT",
    "PAYDATE",
    "PAYTIME",
    "RTNDATE",
    "RTNTIME",
    "GIVEDATE",
    "GIVETIME",
    "OFFDATE",
    "QTYUSE",
    "PAYOUTFLAG",
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
                    # Resolve each requested column case-insensitively so a col
                    # list survives a raw-export header casing change (the
                    # complete BDVSTTRANS ships UPPERCASE headers).
                    lower = {k.lower(): v for k, v in row.items()}
                    writer.writerow(
                        {c: row.get(c, lower.get(c.lower(), "")) for c in out_header}
                    )
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
    sample_ans = {a for _, _, a in sample}
    print("\nSampled RBC (HN, REQNO, AN):")
    for s in sample:
        print(" ", s)

    # --- Platelet sampling (ADDITIVE; separate rng, never touches the RBC rng) ---
    # When PLATELET_N=0 (default) the block below is skipped entirely, so the RBC
    # rng's call sequence — and therefore its output — is byte-identical to a
    # pre-platelet run.  When PLATELET_N>0 a completely independent
    # random.Random(PLATELET_SEED) is constructed and used; it never shares state
    # with `rng`.
    platelet_sample: list[tuple[str, str, str]] = []
    if PLATELET_N > 0:
        # Pass 1p — index BDVSTDT REQNOs where ALL line items are platelet
        # products.  Mixed RBC+platelet REQNOs carry at least one RBC code
        # and are tagged component="red_cell" by build_audit_orders — they
        # must not appear in the platelet stream.  Only a REQNO whose every
        # BDTYPE passes is_platelet_product() (and has at least one item)
        # matches the intake predicate that tags component="platelet".
        plt_bdtypes_by_reqno: dict[str, list[str]] = {}
        with (SRC / "BDVSTDT.csv").open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                reqno = row["REQNO"]
                bdtype = row.get("BDTYPE", "").strip().upper()
                plt_bdtypes_by_reqno.setdefault(reqno, []).append(bdtype)

        plt_reqnos: set[str] = set()
        for reqno, bdtypes in plt_bdtypes_by_reqno.items():
            if bdtypes and all(is_platelet_product(bt) for bt in bdtypes):
                plt_reqnos.add(reqno)
        print(f"BDVSTDT: {len(plt_reqnos)} REQNOs are platelet-ONLY orders")

        # Pass 2p — scan BDVST, keep eligible platelet orders.
        plt_candidates: list[tuple[str, str, str]] = []
        with (SRC / "BDVST.csv").open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["REQNO"] not in plt_reqnos:
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
                plt_candidates.append((hn, row["REQNO"], an))
        print(f"BDVST: {len(plt_candidates)} eligible platelet orders")
        if len(plt_candidates) < PLATELET_N:
            sys.exit(
                f"only {len(plt_candidates)} platelet candidates < PLATELET_N={PLATELET_N}"
            )

        # Separate rng — NEVER draws from the RBC rng.
        plt_rng = random.Random(PLATELET_SEED)
        platelet_sample = plt_rng.sample(plt_candidates, PLATELET_N)
        print("\nSampled platelet (HN, REQNO, AN):")
        for s in platelet_sample:
            print(" ", s)

    # --- Build combined key sets (RBC union platelet) ---
    # When PLATELET_N=0 all plt_* sets are empty so the unions collapse to
    # the RBC-only sets — behaviour is byte-identical to the original code.
    plt_reqnos_sampled: set[str] = {r for _, r, _ in platelet_sample}
    plt_ans: set[str] = {a for _, _, a in platelet_sample}
    plt_pairs: set[tuple[str, str]] = {(h, a) for h, _, a in platelet_sample}

    all_reqnos = sample_reqnos | plt_reqnos_sampled
    all_ans = sample_ans | plt_ans
    all_pairs = sample_pairs | plt_pairs

    related_reqnos = set(all_reqnos)
    with (SRC / "BDVST.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            hn = (row.get("HN") or "").strip()
            an = (row.get("AN") or "").strip()
            if (hn, an) in all_pairs:
                related_reqnos.add(row["REQNO"])

    print("\nWriting mini bundle:")
    _filter(
        "BDVST.csv",
        "BDVST.csv",
        lambda r: r["REQNO"] in all_reqnos,
        cols=BDVST_COLS,
    )
    _filter(
        "BDVSTDT.csv",
        "BDVSTDT.csv",
        lambda r: r["REQNO"] in all_reqnos,
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
    _filter("Diagnosis.csv", "Diagnosis.csv", lambda r: r.get("AN") in all_ans)
    _filter("Lab.csv", "Lab.csv", lambda r: r.get("AN") in all_ans)
    _filter("Med.csv", "Med.csv", lambda r: r.get("AN") in all_ans)
    _filter(
        "IPDADMPROGRESS.csv", "IPDADMPROGRESS.csv", lambda r: r.get("AN") in all_ans
    )
    _filter("IPDNRFOCUSDT.csv", "IPDNRFOCUSDT.csv", lambda r: r.get("AN") in all_ans)
    # Procedure-family exports have arrived as both Title-Case and ALL-CAPS.
    _filter(
        "IPTSUMOPRT.csv",
        "IPTSUMOPRT.csv",
        lambda r: (r.get("An") or r.get("AN")) in all_ans,
    )
    if (SRC / "IPDDCHSUMOPRT.csv").exists():
        _filter(
            "IPDDCHSUMOPRT.csv",
            "IPDDCHSUMOPRT.csv",
            lambda r: (r.get("An") or r.get("AN")) in all_ans,
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
                in all_pairs
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
                in all_pairs
            ),
        )
    if (SRC / "BDVSTTRANS.csv").exists():
        _filter(
            "BDVSTTRANS.csv",
            "BDVSTTRANS.csv",
            lambda r: (r.get("Reqno") or r.get("REQNO")) in all_reqnos,
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
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["HN", "REQNO", "AN", "component", "seed"])
        for s in sample:
            writer.writerow([s[0], s[1], s[2], "rbc", SEED])
        for s in platelet_sample:
            writer.writerow([s[0], s[1], s[2], "platelet", PLATELET_SEED])
    print(
        f"\nManifest: {manifest} ({len(sample)} rbc, {len(platelet_sample)} platelet)"
    )


if __name__ == "__main__":
    main()
