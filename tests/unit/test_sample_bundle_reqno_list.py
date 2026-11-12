"""Tests for the explicit REQNO-list mode of scripts/pilot/sample_bundle.py.

WHY these tests exist:
- A human-reviewed pilot (the 300-case review workbook) is keyed by REQNO. A
  seeded draw reproduces a sample only on the exact BDVST snapshot it came from;
  on a newer export the same seed silently picks different orders (the 2026-09
  rebuild attempt overlapped 6/300). ``BBA_PILOT_REQNO_FILE`` rebuilds the bundle
  from the reviewed list instead.
- Each test encodes a way that rebuild could go wrong without anyone noticing:
  (a) the bundle must hold exactly the listed orders, in list order, whatever
      BBA_PILOT_SAMPLE_N / BBA_PILOT_SAMPLE_SEED say;
  (b) a listed order that is missing, cancelled, or not an RBC order must stop
      the run, not silently shrink the reviewed set; so must a REQNO shared by
      two patients (picking one could put the wrong patient in the bundle);
      a failed run must write nothing;
  (c) a duplicated REQNO must stop the run (the verdict source rejects
      duplicates too), and so must an empty or missing list file;
  (d) the manifest must mark list-mode rows so nobody mistakes them for a
      seeded draw;
  (e) with the variable unset or blank, the seeded draw must be unchanged.
"""

from __future__ import annotations

import csv
import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType

import pytest

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"

# Eligible RBC orders, in BDVST file order.
_RBC = [
    ("H1", "RBC001", "A1"),
    ("H2", "RBC002", "A2"),
    ("H3", "RBC003", "A3"),
    ("H4", "RBC004", "A4"),
    ("H5", "RBC005", "A5"),
]
_CANCELLED = ("H6", "CAN001", "A6")  # RBC line item, but CANCELDATE set
_PLATELET = ("H7", "PLT001", "A7")  # platelet-only order

_MODULE_COUNTER = 0


def _load_sample_bundle() -> ModuleType:
    """Load sample_bundle.py fresh so module-level env reads see the current env."""
    global _MODULE_COUNTER
    _MODULE_COUNTER += 1
    if str(PILOT_DIR) not in sys.path:
        sys.path.insert(0, str(PILOT_DIR))
    spec = importlib.util.spec_from_file_location(
        f"_sample_bundle_reqno_test_{_MODULE_COUNTER}", PILOT_DIR / "sample_bundle.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_raw_dir(raw: Path) -> Path:
    raw.mkdir(parents=True, exist_ok=True)
    orders = [(*o, "") for o in _RBC] + [(*_CANCELLED, "2026-01-02"), (*_PLATELET, "")]
    _write_csv(
        raw / "BDVST.csv",
        ["HN", "REQNO", "AN", "BDVSTST", "REQTYPE", "CANCELDATE"],
        [
            {
                "HN": h,
                "REQNO": r,
                "AN": a,
                "BDVSTST": "4",
                "REQTYPE": "P",
                "CANCELDATE": c,
            }
            for h, r, a, c in orders
        ],
    )
    line_items = [(h, r, "LPRC") for h, r, _ in [*_RBC, _CANCELLED]] + [
        (_PLATELET[0], _PLATELET[1], "LPPC")
    ]
    _write_csv(
        raw / "BDVSTDT.csv",
        ["REQNO", "HN", "BDTYPE"],
        [{"REQNO": r, "HN": h, "BDTYPE": t} for h, r, t in line_items],
    )
    for name in (
        "Diagnosis.csv",
        "Lab.csv",
        "Med.csv",
        "IPDADMPROGRESS.csv",
        "IPDNRFOCUSDT.csv",
        "IPTSUMOPRT.csv",
    ):
        _write_csv(raw / name, ["AN"], [])
    _write_csv(raw / "BDTYPE.csv", ["CODE"], [{"CODE": "LPRC"}])
    _write_csv(raw / "BDVSTST.csv", ["CODE"], [{"CODE": "4"}])
    _write_csv(raw / "ICD9CM.csv", ["CODE"], [{"CODE": "9999"}])
    return raw


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    reqno_file: str | None,
    n: str = "2",
    seed: str = "42",
    tag: str = "run",
) -> Path:
    raw = tmp_path / "raw"
    if not raw.exists():
        _make_raw_dir(raw)
    work = tmp_path / tag
    monkeypatch.setenv("BBA_PILOT_RAW_DIR", str(raw))
    monkeypatch.setenv("BBA_PILOT_WORK_DIR", str(work))
    monkeypatch.setenv("BBA_PILOT_SAMPLE_N", n)
    monkeypatch.setenv("BBA_PILOT_SAMPLE_SEED", seed)
    monkeypatch.setenv("BBA_PILOT_PLATELET_SAMPLE_N", "0")
    if reqno_file is None:
        monkeypatch.delenv("BBA_PILOT_REQNO_FILE", raising=False)
    else:
        monkeypatch.setenv("BBA_PILOT_REQNO_FILE", reqno_file)
    _load_sample_bundle().main()
    return work


def _manifest(work: Path) -> list[dict[str, str]]:
    with (work / "sample_manifest.csv").open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _bundle_reqnos(work: Path) -> set[str]:
    with (work / "bundle" / "BDVST.csv").open(encoding="utf-8", newline="") as fh:
        return {row["REQNO"] for row in csv.DictReader(fh)}


def _list_file(tmp_path: Path, text: str) -> str:
    path = tmp_path / "reqnos.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_bundle_holds_exactly_the_listed_orders_whatever_seed_and_n(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(a) The reviewed set comes back exactly; seed and N cannot change it.

    N=99 exceeds the 5 candidates, which would stop a seeded run, so a pass also
    proves list mode does not fall through to the random draw.
    """
    listed = _list_file(tmp_path, "# reviewed cases\nRBC004\n\n  RBC002  \n")
    first = _run(monkeypatch, tmp_path, reqno_file=listed, n="2", seed="1", tag="a")
    second = _run(monkeypatch, tmp_path, reqno_file=listed, n="99", seed="999", tag="b")

    for work in (first, second):
        assert [r["REQNO"] for r in _manifest(work)] == ["RBC004", "RBC002"]
        assert _bundle_reqnos(work) == {"RBC004", "RBC002"}


@pytest.mark.parametrize(
    "bad",
    [_CANCELLED[1], _PLATELET[1], "NOPE999"],
    ids=["cancelled", "platelet-only", "unknown"],
)
def test_listed_order_that_is_not_an_eligible_rbc_order_stops_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad: str
) -> None:
    """(b) Fail loud, naming the REQNO, and write nothing.

    A silently dropped case would shrink the reviewed set, and partial output
    could be mistaken for a complete rebuild.
    """
    listed = _list_file(tmp_path, f"RBC001\n{bad}\n")
    with pytest.raises(SystemExit, match=bad):
        _run(monkeypatch, tmp_path, reqno_file=listed, tag="bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    ("text", "message"),
    [("RBC001\nRBC002\nRBC001\n", "RBC001"), ("# nothing here\n\n", "no REQNO")],
    ids=["duplicate", "empty"],
)
def test_duplicate_or_empty_list_stops_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, text: str, message: str
) -> None:
    """(c) The verdict source rejects duplicate REQNOs; an empty list is a mistake."""
    listed = _list_file(tmp_path, text)
    with pytest.raises(SystemExit, match=message):
        _run(monkeypatch, tmp_path, reqno_file=listed)


def test_missing_list_file_stops_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(c) A typo in the path must not fall back to a random draw."""
    with pytest.raises(SystemExit, match="BBA_PILOT_REQNO_FILE"):
        _run(monkeypatch, tmp_path, reqno_file=str(tmp_path / "missing.txt"))
    assert not (tmp_path / "run").exists()


def test_manifest_marks_list_rows_instead_of_a_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(d) A reader must be able to tell a listed sample from a seeded one."""
    listed = _list_file(tmp_path, "RBC003\nRBC005\n")
    work = _run(monkeypatch, tmp_path, reqno_file=listed)
    rows = _manifest(work)
    assert [(r["component"], r["seed"]) for r in rows] == [
        ("rbc", "list"),
        ("rbc", "list"),
    ]


@pytest.mark.parametrize(
    "reqno_file", [None, "", "   "], ids=["unset", "empty", "blank"]
)
def test_unset_or_blank_variable_keeps_the_seeded_draw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reqno_file: str | None
) -> None:
    """(e) Existing pilot runs stay reproducible: seeded draw over eligible orders."""
    work = _run(monkeypatch, tmp_path, reqno_file=reqno_file, n="2", seed="42")
    expected = [r for _, r, _ in random.Random(42).sample(_RBC, 2)]
    rows = _manifest(work)
    assert [r["REQNO"] for r in rows] == expected
    assert {r["seed"] for r in rows} == {"42"}


def test_reqno_shared_by_two_patients_stops_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(b) A reused REQNO must not silently pick one patient's order.

    RBC003 is eligible for H3/A3 and, reused, for H9/A9. Keeping either one
    would put a patient in the bundle who may not be the reviewed case.
    """
    raw = _make_raw_dir(tmp_path / "raw")
    with (raw / "BDVST.csv").open("a", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow(["H9", "RBC003", "A9", "4", "P", ""])
    listed = _list_file(tmp_path, "RBC001\nRBC003\n")
    with pytest.raises(SystemExit, match="RBC003"):
        _run(monkeypatch, tmp_path, reqno_file=listed)
    assert not (tmp_path / "run").exists()


def test_list_saved_with_a_byte_order_mark_is_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(a) Excel writes a BOM; it must not stick to the first REQNO."""
    path = tmp_path / "reqnos.txt"
    path.write_text("\ufeffRBC004\nRBC002\n", encoding="utf-8")
    work = _run(monkeypatch, tmp_path, reqno_file=str(path))
    assert [r["REQNO"] for r in _manifest(work)] == ["RBC004", "RBC002"]


def test_seeded_draw_still_stops_when_n_exceeds_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """(e) The pre-existing guard survives the refactor, and now writes nothing."""
    with pytest.raises(SystemExit, match="N=99"):
        _run(monkeypatch, tmp_path, reqno_file=None, n="99")
    assert not (tmp_path / "run").exists()


def test_sampler_rbc_set_is_the_audit_gate_allow_list() -> None:
    """The sampler must sample exactly the orders the pipeline will audit.

    If the two sets drift (as they did before irradiated PRC was added), a
    cohort that orders the missing product is silently under-sampled while
    the pipeline would have judged those orders fine.
    """
    from bba.audit_orders import RBC_PRODUCTS

    mod = _load_sample_bundle()
    assert set(mod.RBC) == set(RBC_PRODUCTS)
    assert {"LDPRCI", "LPRCI"} <= set(mod.RBC)
