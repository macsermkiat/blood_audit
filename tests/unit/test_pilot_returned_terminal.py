from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

from bba.cohort_detector import CohortAssignment, CohortLabel
from bba.deterministic_classifier import ClassifierInputs, classify
from bba.hb_lookup import HbLookupResult
from bba.returns_ledger import ReturnsSummary


def _load_pilot() -> ModuleType:
    pilot_dir = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
    if str(pilot_dir) not in sys.path:
        sys.path.insert(0, str(pilot_dir))
    spec = importlib.util.spec_from_file_location(
        "pilot_run_pipeline_returned_test", pilot_dir / "run_pipeline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pilot_classifier_path_emits_returned_terminal(monkeypatch) -> None:
    pilot = _load_pilot()
    monkeypatch.setattr(pilot, "RETURNS_LEDGER_ENABLED", True)
    summary = ReturnsSummary(
        units_total=2,
        units_returned=2,
        ordered_unit_amount=2,
        ledger_complete=True,
    )
    result = classify(
        ClassifierInputs(
            audit_id="pilot-returned",
            hb_result=HbLookupResult(
                value_g_dl=None,
                datetime_utc=None,
                source=None,
                freshness="missing",
                delta_hb_bypass=False,
                delta_hb_windows=(),
                needs_review_single_low_hb=False,
            ),
            cohort_assignment=CohortAssignment(
                label=CohortLabel.DEFAULT,
                threshold=7.0,
                evidence_code=None,
                evidence_name=None,
            ),
            order_datetime=datetime(2026, 7, 13, tzinfo=UTC),
            procedure_proximity_hours=None,
            crystalloid_liters_prior_4h=0.0,
            returns_disposition=pilot._returns_disposition_for_classifier(summary),
        )
    )
    assert result.classification == "RETURNED_NOT_TRANSFUSED"


def test_pilot_flag_off_forces_inconclusive(monkeypatch) -> None:
    pilot = _load_pilot()
    monkeypatch.setattr(pilot, "RETURNS_LEDGER_ENABLED", False)
    summary = ReturnsSummary(
        units_total=1,
        units_returned=1,
        ordered_unit_amount=1,
        ledger_complete=True,
    )
    assert pilot._returns_disposition_for_classifier(summary) == "inconclusive"
