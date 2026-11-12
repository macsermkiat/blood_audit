"""The bundle's BDVSTTRANS copy must cover every order on a sampled admission.

WHY this test exists:
- The returns pre-flight (ticket #125) attributes an in-window "gave blood"
  note on an all-returned order to a NOT-returned unit of a sibling order on the
  same admission. That attribution can only see units present in the ledger it
  reads, and it prefers the bundle's own ``BDVSTTRANS.csv`` when one exists.
- Cutting that copy to the sampled REQNOs alone hides every sibling that was not
  itself sampled. On the 2026-09 rebuild of the 300-case pilot this turned a
  real sibling transfusion (68019920 -> 68020779) into a false "hidden
  transfusion" and a spurious HOLD verdict.
- So the ledger copy is cut by admission, like ``BDVST_RELATED.csv``: a sibling
  on a sampled (HN, AN) is kept; an order on another admission is not. Every
  pipeline consumer joins the ledger by REQNO exactly, so the extra rows cannot
  change a verdict.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"

_SAMPLED = ("H1", "RBC001", "A1")
_SIBLING = ("H1", "SIB001", "A1")  # same admission, never sampled
_OTHER = ("H2", "RBC002", "A2")  # different admission


def _load_sample_bundle() -> ModuleType:
    if str(PILOT_DIR) not in sys.path:
        sys.path.insert(0, str(PILOT_DIR))
    spec = importlib.util.spec_from_file_location(
        "_sample_bundle_ledger_admission_test", PILOT_DIR / "sample_bundle.py"
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


def _make_raw_dir(raw: Path) -> None:
    raw.mkdir(parents=True)
    orders = [_SAMPLED, _SIBLING, _OTHER]
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
                "CANCELDATE": "",
            }
            for h, r, a in orders
        ],
    )
    _write_csv(
        raw / "BDVSTDT.csv",
        ["REQNO", "HN", "BDTYPE"],
        [{"REQNO": r, "HN": h, "BDTYPE": "LPRC"} for h, r, _ in orders],
    )
    _write_csv(
        raw / "BDVSTTRANS.csv",
        ["REQNO", "HN", "AN", "BDTYPE", "DNRNO", "SEQNO", "UNITSTAT"],
        [
            {
                "REQNO": r,
                "HN": h,
                "AN": a,
                "BDTYPE": "LPRC",
                "DNRNO": f"D{r}",
                "SEQNO": "0",
                "UNITSTAT": "2",
            }
            for h, r, a in orders
        ],
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


def _bundle_ledger_reqnos(work: Path) -> set[str]:
    with (work / "bundle" / "BDVSTTRANS.csv").open(encoding="utf-8", newline="") as fh:
        return {row["REQNO"] for row in csv.DictReader(fh)}


def test_bundle_ledger_keeps_same_admission_siblings_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw = tmp_path / "raw"
    _make_raw_dir(raw)
    work = tmp_path / "work"
    listed = tmp_path / "reqnos.txt"
    listed.write_text(f"{_SAMPLED[1]}\n", encoding="utf-8")
    monkeypatch.setenv("BBA_PILOT_RAW_DIR", str(raw))
    monkeypatch.setenv("BBA_PILOT_WORK_DIR", str(work))
    monkeypatch.setenv("BBA_PILOT_REQNO_FILE", str(listed))
    monkeypatch.setenv("BBA_PILOT_PLATELET_SAMPLE_N", "0")

    _load_sample_bundle().main()

    assert _bundle_ledger_reqnos(work) == {_SAMPLED[1], _SIBLING[1]}
