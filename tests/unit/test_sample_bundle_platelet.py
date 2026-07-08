"""Tests for the additive platelet sampling stream in scripts/pilot/sample_bundle.py.

WHY these tests exist:
- The platelet stream shares the same output bundle files as the RBC stream.
  If the two random.Random instances were accidentally shared or interleaved, the
  RBC sample would change when PLATELET_N is toggled — breaking reproducibility of
  all existing pilot runs. Each test below encodes a specific failure mode that
  would surface such a regression.

Test matrix:
  (a) test_rbc_reqnos_unchanged_when_platelet_sampling_active
      — the RBC sample is byte-identical whether PLATELET_N=0 or PLATELET_N>0.
      Failure would indicate the platelet rng draw perturbed the RBC rng.

  (b) test_platelet_orders_sampled_and_projected_when_n_positive
      — platelet REQNOs appear in the output bundle (BDVST.csv + BDVSTDT.csv)
      and every sampled BDTYPE is a platelet-family product per bba.component_map.

  (c) test_platelet_seed_changes_platelet_sample_not_rbc
      — different PLATELET_SEED values change which platelet orders are drawn but
      leave the RBC sample untouched. Failure would indicate the rng objects are
      not truly independent.

  (d) test_manifest_records_component_and_seed_for_both_streams
      — the sample_manifest.csv is component-tagged so RBC vs platelet counts are
      distinguishable and the seed used for each row is traceable.

  (e) test_zero_platelet_n_produces_no_platelet_rows_in_bundle
      — when PLATELET_N=0 (the default), the bundle contains only RBC rows and the
      manifest contains only rbc-tagged entries.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE_COUNTER = 0


def _load_sample_bundle() -> ModuleType:
    """Load scripts/pilot/sample_bundle.py fresh.

    Uses a unique module name on every call so Python's module cache is bypassed
    and module-level code (which reads env vars for SRC, N, SEED, etc.) is
    re-executed with whatever env vars are current at call time.
    """
    global _MODULE_COUNTER
    _MODULE_COUNTER += 1
    mod_name = f"_sample_bundle_test_{_MODULE_COUNTER}"

    if str(PILOT_DIR) not in sys.path:
        sys.path.insert(0, str(PILOT_DIR))

    path = PILOT_DIR / "sample_bundle.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Synthetic data constants — small enough for tests, large enough for 2-of-N sampling.
# RBC orders: BDTYPE LPRC (red cell).  Platelet orders: BDTYPE LPPC / LDPPC / SDPF / PC.
_RBC_ORDERS = [
    ("H1", "RBC001", "A1"),
    ("H2", "RBC002", "A2"),
    ("H3", "RBC003", "A3"),
]
_PLT_ORDERS = [
    ("H4", "PLT001", "A4", "LPPC"),
    ("H5", "PLT002", "A5", "LDPPC"),
    ("H6", "PLT003", "A6", "SDPF"),
    ("H7", "PLT004", "A7", "PC"),
]
# Mixed order: same REQNO carries both an RBC and a platelet line item.
# Intake tags this as component="red_cell" (mixed → red_cell), so the
# platelet sampler must NOT include it in the platelet stream.
_MIXED_HN = "H8"
_MIXED_REQNO = "MIX001"
_MIXED_AN = "A8"

# Determined empirically: random.Random(42).sample(rbc_candidates, 2)
# rbc_candidates = _RBC_ORDERS + [_MIXED_REQNO] (4 total; MIX001 carries LPRC).
_EXPECTED_RBC_REQNOS_SEED42 = {"MIX001", "RBC001"}

# random.Random(90).sample(_PLT_ORDERS, 2) and random.Random(91).sample(...)
# give DIFFERENT sets — confirmed in pre-commit verification.
_PLT_SEED_A = 90
_PLT_SEED_B = 91


def _make_raw_dir(base: Path) -> Path:
    """Create a minimal fake raw dir with all CSV files sample_bundle.main() needs."""
    raw = base
    raw.mkdir(parents=True, exist_ok=True)

    # --- BDVST.csv (all orders: RBC + platelet + mixed) ---
    bdvst_rows = []
    for hn, reqno, an in _RBC_ORDERS:
        bdvst_rows.append(
            {
                "HN": hn,
                "REQNO": reqno,
                "AN": an,
                "BDVSTST": "4",
                "REQTYPE": "P",
                "CANCELDATE": "",
                "REQDATE": "2026-01-01",
                "REQTIME": "080000",
                "BDVSTDATE": "2026-01-01",
                "BDVSTTIME": "090000",
                "PICKDATE": "",
                "PICKTIME": "",
                "ICD10": f"ICD-{reqno}",
                "DIAGNOSIS": f"diag-{reqno}",
            }
        )
    for hn, reqno, an, _bdtype in _PLT_ORDERS:
        bdvst_rows.append(
            {
                "HN": hn,
                "REQNO": reqno,
                "AN": an,
                "BDVSTST": "4",
                "REQTYPE": "P",
                "CANCELDATE": "",
                "REQDATE": "2026-01-01",
                "REQTIME": "080000",
                "BDVSTDATE": "2026-01-01",
                "BDVSTTIME": "090000",
                "PICKDATE": "",
                "PICKTIME": "",
                "ICD10": f"ICD-{reqno}",
                "DIAGNOSIS": f"diag-{reqno}",
            }
        )
    # Mixed order: contains both LPRC (RBC) and LPPC (platelet) — must be
    # excluded from the platelet stream but admitted via the RBC stream.
    bdvst_rows.append(
        {
            "HN": _MIXED_HN,
            "REQNO": _MIXED_REQNO,
            "AN": _MIXED_AN,
            "BDVSTST": "4",
            "REQTYPE": "P",
            "CANCELDATE": "",
            "REQDATE": "2026-01-01",
            "REQTIME": "080000",
            "BDVSTDATE": "2026-01-01",
            "BDVSTTIME": "090000",
            "PICKDATE": "",
            "PICKTIME": "",
            "ICD10": f"ICD-{_MIXED_REQNO}",
            "DIAGNOSIS": f"diag-{_MIXED_REQNO}",
        }
    )
    _write_csv(raw / "BDVST.csv", bdvst_rows)

    # --- BDVSTDT.csv ---
    bdvstdt_rows = []
    for hn, reqno, _an in _RBC_ORDERS:
        bdvstdt_rows.append(
            {
                "REQNO": reqno,
                "HN": hn,
                "BDVSTDATE": "2026-01-01",
                "BDVSTTIME": "090000",
                "USEDATE": "2026-01-01",
                "USETIME": "100000",
                "BDTYPE": "LPRC",
                "ITEMNO": "1",
                "UNITAMT": "1",
            }
        )
    for hn, reqno, _an, bdtype in _PLT_ORDERS:
        bdvstdt_rows.append(
            {
                "REQNO": reqno,
                "HN": hn,
                "BDVSTDATE": "2026-01-01",
                "BDVSTTIME": "090000",
                "USEDATE": "2026-01-01",
                "USETIME": "100000",
                "BDTYPE": bdtype,
                "ITEMNO": "1",
                "UNITAMT": "1",
            }
        )
    # Mixed order has TWO line items: one RBC (LPRC) and one platelet (LPPC).
    for bdtype in ("LPRC", "LPPC"):
        bdvstdt_rows.append(
            {
                "REQNO": _MIXED_REQNO,
                "HN": _MIXED_HN,
                "BDVSTDATE": "2026-01-01",
                "BDVSTTIME": "090000",
                "USEDATE": "2026-01-01",
                "USETIME": "100000",
                "BDTYPE": bdtype,
                "ITEMNO": "1" if bdtype == "LPRC" else "2",
                "UNITAMT": "1",
            }
        )
    _write_csv(raw / "BDVSTDT.csv", bdvstdt_rows)

    # --- Tables filtered by AN (empty content is fine for the join logic) ---
    for name, fields in [
        ("Diagnosis.csv", ["AN", "ICD10", "DIAG"]),
        ("Lab.csv", ["AN", "LABEXM", "LABRESULT"]),
        ("Med.csv", ["AN", "MEDCODE", "MEDNAME"]),
        ("IPDADMPROGRESS.csv", ["AN", "PROGDATE", "PROGTEXT"]),
        ("IPDNRFOCUSDT.csv", ["AN", "FOCUSDATE", "FOCUSTEXT"]),
        ("IPTSUMOPRT.csv", ["AN", "ICD9CM", "ORFLAG"]),
    ]:
        _write_csv(raw / name, [], fieldnames=fields)

    # --- Dictionary tables (copied whole; must be non-empty) ---
    _write_csv(raw / "BDTYPE.csv", [{"CODE": "LPRC", "NAME": "Leukodepleted PRC"}])
    _write_csv(raw / "BDVSTST.csv", [{"CODE": "4", "NAME": "Issued"}])
    _write_csv(raw / "ICD9CM.csv", [{"CODE": "9999", "NAME": "Test"}])

    return raw


def _write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    elif fieldnames is None:
        fieldnames = []
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv_col(path: Path, col: str) -> list[str]:
    """Read a single column from a CSV as a list."""
    with path.open(encoding="utf-8", newline="") as fh:
        return [row[col] for row in csv.DictReader(fh)]


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rbc_reqnos_unchanged_when_platelet_sampling_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(a) RBC rng is not perturbed by the platelet rng.

    With the same BBA_PILOT_SAMPLE_SEED, the set of RBC REQNOs written to
    BDVST.csv is identical regardless of whether PLATELET_N is 0 or >0.
    If the two Random instances shared state or the platelet draw interleaved
    with the RBC draw, the samples would diverge.
    """
    raw = _make_raw_dir(tmp_path / "raw")

    def _run(platelet_n: int) -> set[str]:
        work = tmp_path / f"work_{platelet_n}"
        monkeypatch.setenv("BBA_PILOT_RAW_DIR", str(raw))
        monkeypatch.setenv("BBA_PILOT_WORK_DIR", str(work))
        monkeypatch.setenv("BBA_PILOT_SAMPLE_N", "2")
        monkeypatch.setenv("BBA_PILOT_SAMPLE_SEED", "42")
        monkeypatch.setenv("BBA_PILOT_PLATELET_SAMPLE_N", str(platelet_n))
        monkeypatch.setenv("BBA_PILOT_PLATELET_SEED", "90")
        mod = _load_sample_bundle()
        mod.main()
        manifest_rows = _read_manifest(work / "sample_manifest.csv")
        return {r["REQNO"] for r in manifest_rows if r.get("component") == "rbc"}

    rbc_without_platelet = _run(0)
    rbc_with_platelet = _run(2)

    # The RBC set must be byte-identical regardless of the platelet stream.
    assert rbc_without_platelet == rbc_with_platelet
    # Sanity-check: these are the expected REQNOs for seed=42 on our synthetic data.
    assert rbc_without_platelet == _EXPECTED_RBC_REQNOS_SEED42


def test_platelet_orders_sampled_and_projected_when_n_positive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(b) Platelet orders actually appear in the bundle when PLATELET_N>0.

    Checks that:
    - The sampled platelet REQNOs appear in output BDVST.csv.
    - Their BDTYPE values in output BDVSTDT.csv are platelet-family per
      bba.component_map.is_platelet_product (the authoritative allow-list).
    - The manifest has platelet-tagged entries equal in count to PLATELET_N.
    """
    from bba.component_map import is_platelet_product

    raw = _make_raw_dir(tmp_path / "raw")
    work = tmp_path / "work"
    monkeypatch.setenv("BBA_PILOT_RAW_DIR", str(raw))
    monkeypatch.setenv("BBA_PILOT_WORK_DIR", str(work))
    monkeypatch.setenv("BBA_PILOT_SAMPLE_N", "2")
    monkeypatch.setenv("BBA_PILOT_SAMPLE_SEED", "42")
    monkeypatch.setenv("BBA_PILOT_PLATELET_SAMPLE_N", "2")
    monkeypatch.setenv("BBA_PILOT_PLATELET_SEED", "90")
    mod = _load_sample_bundle()
    mod.main()

    bundle = work / "bundle"
    manifest = _read_manifest(work / "sample_manifest.csv")

    plt_manifest_rows = [r for r in manifest if r.get("component") == "platelet"]
    assert len(plt_manifest_rows) == 2, (
        "manifest must list exactly PLATELET_N platelet rows"
    )

    plt_reqnos_in_manifest = {r["REQNO"] for r in plt_manifest_rows}

    # Verify the sampled REQNOs appear in the output BDVST.csv.
    bdvst_reqnos = set(_read_csv_col(bundle / "BDVST.csv", "REQNO"))
    assert plt_reqnos_in_manifest.issubset(bdvst_reqnos), (
        "platelet REQNOs from manifest must be present in output BDVST.csv"
    )

    # Every platelet REQNO in the output BDVSTDT.csv must map to a platelet product.
    plt_bdtypes: set[str] = set()
    with (bundle / "BDVSTDT.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("REQNO") in plt_reqnos_in_manifest:
                plt_bdtypes.add(row["BDTYPE"].strip().upper())

    assert plt_bdtypes, "no BDTYPE found for platelet REQNOs in output BDVSTDT.csv"
    for bdtype in plt_bdtypes:
        assert is_platelet_product(bdtype), (
            f"BDTYPE {bdtype!r} in the platelet sample is not a platelet product"
            " per bba.component_map.is_platelet_product — allow-list mismatch"
        )


def test_platelet_seed_changes_platelet_sample_not_rbc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(c) Changing PLATELET_SEED changes the platelet sample but not the RBC sample.

    This is the independence proof: if the two RNG objects were coupled in any way,
    changing only the platelet seed would inevitably alter the RBC sample too.
    """
    raw = _make_raw_dir(tmp_path / "raw")

    def _run(platelet_seed: int) -> tuple[set[str], set[str]]:
        work = tmp_path / f"work_pseed_{platelet_seed}"
        monkeypatch.setenv("BBA_PILOT_RAW_DIR", str(raw))
        monkeypatch.setenv("BBA_PILOT_WORK_DIR", str(work))
        monkeypatch.setenv("BBA_PILOT_SAMPLE_N", "2")
        monkeypatch.setenv("BBA_PILOT_SAMPLE_SEED", "42")
        monkeypatch.setenv("BBA_PILOT_PLATELET_SAMPLE_N", "2")
        monkeypatch.setenv("BBA_PILOT_PLATELET_SEED", str(platelet_seed))
        mod = _load_sample_bundle()
        mod.main()
        manifest = _read_manifest(work / "sample_manifest.csv")
        rbc_reqnos = {r["REQNO"] for r in manifest if r.get("component") == "rbc"}
        plt_reqnos = {r["REQNO"] for r in manifest if r.get("component") == "platelet"}
        return rbc_reqnos, plt_reqnos

    rbc_a, plt_a = _run(_PLT_SEED_A)
    rbc_b, plt_b = _run(_PLT_SEED_B)

    # RBC sample must be identical (seed=42 is fixed; platelet seed must not bleed over).
    assert rbc_a == rbc_b == _EXPECTED_RBC_REQNOS_SEED42
    # Platelet sample must differ — proves the seeds are truly independent.
    assert plt_a != plt_b, (
        f"seeds {_PLT_SEED_A} and {_PLT_SEED_B} produced identical platelet samples"
        " — use a different pair of seeds in the test constants"
    )


def test_manifest_records_component_and_seed_for_both_streams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(d) sample_manifest.csv is component-tagged and seed-tagged for traceability.

    Each row must carry a 'component' column ('rbc' or 'platelet') and a 'seed'
    column so a reader can reconstruct exactly which rng produced each entry.
    """
    raw = _make_raw_dir(tmp_path / "raw")
    work = tmp_path / "work"
    monkeypatch.setenv("BBA_PILOT_RAW_DIR", str(raw))
    monkeypatch.setenv("BBA_PILOT_WORK_DIR", str(work))
    monkeypatch.setenv("BBA_PILOT_SAMPLE_N", "2")
    monkeypatch.setenv("BBA_PILOT_SAMPLE_SEED", "42")
    monkeypatch.setenv("BBA_PILOT_PLATELET_SAMPLE_N", "2")
    monkeypatch.setenv("BBA_PILOT_PLATELET_SEED", "90")
    mod = _load_sample_bundle()
    mod.main()

    rows = _read_manifest(work / "sample_manifest.csv")
    assert len(rows) == 4  # 2 rbc + 2 platelet

    for row in rows:
        assert "component" in row, "manifest must have a 'component' column"
        assert "seed" in row, "manifest must have a 'seed' column"
        assert row["component"] in {"rbc", "platelet"}

    rbc_rows = [r for r in rows if r["component"] == "rbc"]
    plt_rows = [r for r in rows if r["component"] == "platelet"]
    assert len(rbc_rows) == 2
    assert len(plt_rows) == 2

    assert all(r["seed"] == "42" for r in rbc_rows), "rbc rows must record the RBC seed"
    assert all(r["seed"] == "90" for r in plt_rows), (
        "platelet rows must record the platelet seed"
    )


def test_zero_platelet_n_produces_no_platelet_rows_in_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(e) Default PLATELET_N=0 keeps the bundle RBC-only (backwards compatibility).

    When the env var is unset (or 0), no platelet rows must appear in BDVST.csv
    and the manifest must contain only rbc-tagged entries.
    """
    raw = _make_raw_dir(tmp_path / "raw")
    work = tmp_path / "work"
    monkeypatch.setenv("BBA_PILOT_RAW_DIR", str(raw))
    monkeypatch.setenv("BBA_PILOT_WORK_DIR", str(work))
    monkeypatch.setenv("BBA_PILOT_SAMPLE_N", "2")
    monkeypatch.setenv("BBA_PILOT_SAMPLE_SEED", "42")
    monkeypatch.setenv("BBA_PILOT_PLATELET_SAMPLE_N", "0")
    monkeypatch.delenv("BBA_PILOT_PLATELET_SEED", raising=False)
    mod = _load_sample_bundle()
    mod.main()

    manifest = _read_manifest(work / "sample_manifest.csv")

    plt_rows = [r for r in manifest if r.get("component") == "platelet"]
    assert plt_rows == [], (
        "with PLATELET_N=0 the manifest must not contain platelet rows"
    )

    rbc_reqnos = {r["REQNO"] for r in manifest if r.get("component") == "rbc"}
    # Only known-RBC REQNOs should appear in the bundle
    known_plt_reqnos = {r[1] for r in _PLT_ORDERS}
    bdvst_reqnos = set(_read_csv_col(work / "bundle" / "BDVST.csv", "REQNO"))
    assert not (bdvst_reqnos & known_plt_reqnos), (
        "platelet REQNOs must not appear in the bundle when PLATELET_N=0"
    )
    assert rbc_reqnos == _EXPECTED_RBC_REQNOS_SEED42


def test_mixed_order_excluded_from_platelet_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(f) A REQNO that mixes RBC and platelet line items must NOT appear in
    the platelet sample.

    WHY: build_audit_orders tags an order as component="platelet" only when
    ALL its BDTYPE codes are platelet products — mixed orders get tagged
    component="red_cell". The sampler predicate must match this contract
    (all-platelet), not the looser "has at least one platelet" predicate.
    If it does not, a mixed order would appear in both the platelet manifest
    AND the RBC bundle, causing it to be classified twice (once as RBC, once
    as platelet) and inflating the platelet count.
    """
    from bba.component_map import is_platelet_product

    raw = _make_raw_dir(tmp_path / "raw")
    work = tmp_path / "work"
    monkeypatch.setenv("BBA_PILOT_RAW_DIR", str(raw))
    monkeypatch.setenv("BBA_PILOT_WORK_DIR", str(work))
    monkeypatch.setenv("BBA_PILOT_SAMPLE_N", "2")
    monkeypatch.setenv("BBA_PILOT_SAMPLE_SEED", "42")
    monkeypatch.setenv("BBA_PILOT_PLATELET_SAMPLE_N", "2")
    monkeypatch.setenv("BBA_PILOT_PLATELET_SEED", "90")
    mod = _load_sample_bundle()
    mod.main()

    manifest = _read_manifest(work / "sample_manifest.csv")
    plt_reqnos_in_manifest = {
        r["REQNO"] for r in manifest if r.get("component") == "platelet"
    }

    # The mixed order must NOT appear in the platelet manifest.
    assert _MIXED_REQNO not in plt_reqnos_in_manifest, (
        f"Mixed order {_MIXED_REQNO!r} (LPRC+LPPC) must not appear in the platelet "
        "manifest — intake tags it as red_cell, so the platelet sampler predicate "
        "(all-platelet) must exclude it"
    )

    # Sanity: the sampled platelet REQNOs must all be purely platelet-only.
    bundle = work / "bundle"
    with (bundle / "BDVSTDT.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("REQNO") in plt_reqnos_in_manifest:
                bdtype = row["BDTYPE"].strip().upper()
                assert is_platelet_product(bdtype), (
                    f"BDTYPE {bdtype!r} in a sampled platelet order is not a "
                    "platelet product — all-platelet predicate was not enforced"
                )
