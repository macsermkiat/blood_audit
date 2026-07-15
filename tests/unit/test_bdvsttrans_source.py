"""Unit tests for the pilot driver's BDVSTTRANS source resolver (spec #119).

The pilot scripts are the de-facto production classification driver; this seam is
how a production run points them at the canonical complete ledger
(``$BBA_BDVSTTRANS_CSV``) instead of a staged bundle slice. Getting the
precedence and the fail-open-to-empty behaviour right keeps a flag-on run from
silently reading the wrong ledger or crashing on an absent one.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_source_module() -> ModuleType:
    pilot_dir = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
    if str(pilot_dir) not in sys.path:
        sys.path.insert(0, str(pilot_dir))
    spec = importlib.util.spec_from_file_location(
        "pilot_bdvsttrans_source_test", pilot_dir / "_bdvsttrans_source.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SRC = _load_source_module()


def _write_csv(path: Path, header: str, *rows: str) -> None:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")


def test_override_wins_over_bundle(tmp_path: Path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_csv(bundle / "BDVSTTRANS.csv", "REQNO,UNITSTAT", "B1,3")
    override = tmp_path / "canonical.csv"
    _write_csv(override, "Reqno,Unitstat", "C1,5")  # title-case -> uppercased
    monkeypatch.setenv("BBA_BDVSTTRANS_CSV", str(override))

    assert SRC.resolve_bdvsttrans_path(bundle) == override
    rows = SRC.load_bdvsttrans_rows(bundle)
    assert rows == [{"REQNO": "C1", "UNITSTAT": "5"}]  # override, keys uppercased


def test_bundle_used_when_no_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("BBA_BDVSTTRANS_CSV", raising=False)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_csv(bundle / "BDVSTTRANS.csv", "REQNO,UNITSTAT", "B1,3")

    assert SRC.resolve_bdvsttrans_path(bundle) == bundle / "BDVSTTRANS.csv"
    assert SRC.load_bdvsttrans_rows(bundle) == [{"REQNO": "B1", "UNITSTAT": "3"}]


def test_canonical_default_when_no_override_or_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    # Zero-config default: no override and no bundle copy -> the canonical
    # data/encrypted export, so a production run needs no env var.
    monkeypatch.delenv("BBA_BDVSTTRANS_CSV", raising=False)
    canonical = tmp_path / "canonical.csv"
    _write_csv(canonical, "REQNO,UNITSTAT", "K1,3")
    monkeypatch.setattr(SRC, "_CANONICAL_DEFAULT", canonical)
    bundle = tmp_path / "bundle"
    bundle.mkdir()  # no BDVSTTRANS.csv in the bundle

    assert SRC.resolve_bdvsttrans_path(bundle) == canonical
    assert SRC.load_bdvsttrans_rows(bundle) == [{"REQNO": "K1", "UNITSTAT": "3"}]


def test_bundle_wins_over_canonical_default(tmp_path: Path, monkeypatch) -> None:
    # A bundle-staged slice takes precedence over the full canonical export.
    monkeypatch.delenv("BBA_BDVSTTRANS_CSV", raising=False)
    canonical = tmp_path / "canonical.csv"
    _write_csv(canonical, "REQNO,UNITSTAT", "K1,3")
    monkeypatch.setattr(SRC, "_CANONICAL_DEFAULT", canonical)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_csv(bundle / "BDVSTTRANS.csv", "REQNO,UNITSTAT", "B1,5")

    assert SRC.resolve_bdvsttrans_path(bundle) == bundle / "BDVSTTRANS.csv"


def test_absent_ledger_returns_empty(tmp_path: Path, monkeypatch) -> None:
    # No override, no bundle copy, AND no canonical export -> empty, never a
    # crash (the driver then falls through to the legacy pipeline).
    monkeypatch.delenv("BBA_BDVSTTRANS_CSV", raising=False)
    monkeypatch.setattr(SRC, "_CANONICAL_DEFAULT", tmp_path / "no_canonical.csv")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    assert SRC.resolve_bdvsttrans_path(bundle) is None
    assert SRC.load_bdvsttrans_rows(bundle) == []


def test_blank_override_falls_back_to_bundle(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BBA_BDVSTTRANS_CSV", "   ")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_csv(bundle / "BDVSTTRANS.csv", "REQNO,UNITSTAT", "B1,3")
    assert SRC.resolve_bdvsttrans_path(bundle) == bundle / "BDVSTTRANS.csv"


def test_missing_override_path_returns_empty(tmp_path: Path, monkeypatch) -> None:
    # An override pointing at a non-existent file loads empty rather than raising.
    monkeypatch.setenv("BBA_BDVSTTRANS_CSV", str(tmp_path / "nope.csv"))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    assert SRC.load_bdvsttrans_rows(bundle) == []
