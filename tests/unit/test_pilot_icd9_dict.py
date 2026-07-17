"""Focused unit test for the pilot ICD-9 dictionary loader helper (ticket #186).

Prefactor: `_icd9_dict_from_rows` is a pure module-level helper extracted
verbatim from `main()`'s inline dict comprehension. These tests pin its exact
key/strip/replace/get semantics with fully synthetic rows (no PHI).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"


def _load_pilot_module(filename: str, module_name: str) -> ModuleType:
    if str(PILOT_DIR) not in sys.path:
        sys.path.insert(0, str(PILOT_DIR))
    spec = importlib.util.spec_from_file_location(module_name, PILOT_DIR / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_run_pipeline(module_name: str) -> ModuleType:
    return _load_pilot_module("run_pipeline.py", module_name)


def test_icd9_dict_stores_dotted_code_dotless() -> None:
    # Arrange
    module = _load_run_pipeline("run_pipeline_icd9_dotless")
    rows = [{"Icd9cm": "79.31", "Name": "Open reduction", "Orflag": "Y"}]

    # Act
    result = module._icd9_dict_from_rows(rows)

    # Assert
    assert "7931" in result
    assert "79.31" not in result
    assert result["7931"] == {"NAME": "Open reduction", "ORFLAG": "Y"}


def test_icd9_dict_strips_name_and_orflag() -> None:
    # Arrange
    module = _load_run_pipeline("run_pipeline_icd9_strip")
    rows = [
        {"Icd9cm": "  8151  ", "Name": "  Total hip replacement  ", "Orflag": "  N  "}
    ]

    # Act
    result = module._icd9_dict_from_rows(rows)

    # Assert: code stripped (and dotless), NAME/ORFLAG stripped and carried through
    assert result["8151"] == {"NAME": "Total hip replacement", "ORFLAG": "N"}


def test_icd9_dict_blank_code_maps_to_empty_key() -> None:
    # Arrange: a blank/missing Icd9cm collapses to the "" key (verbatim behavior)
    module = _load_run_pipeline("run_pipeline_icd9_blank")
    rows = [
        {"Icd9cm": "   ", "Name": "No-code op", "Orflag": "Y"},
        {"Name": "Missing-key op", "Orflag": "N"},
    ]

    # Act
    result = module._icd9_dict_from_rows(rows)

    # Assert: both rows target the "" key; the last one wins (comprehension order)
    assert "" in result
    assert result[""] == {"NAME": "Missing-key op", "ORFLAG": "N"}


def test_icd9_dict_missing_name_and_orflag_default_empty() -> None:
    # Arrange: the (r.get(...) or "") semantics turn absent Name/Orflag into ""
    module = _load_run_pipeline("run_pipeline_icd9_defaults")
    rows = [{"Icd9cm": "3891"}]

    # Act
    result = module._icd9_dict_from_rows(rows)

    # Assert
    assert result["3891"] == {"NAME": "", "ORFLAG": ""}
