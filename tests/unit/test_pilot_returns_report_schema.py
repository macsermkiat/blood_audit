"""Guard the pilot report schema so a flag-off run stays byte-identical.

The returns-ledger columns are appended to ``REPORT_FIELDNAMES`` only when
``RETURNS_LEDGER_ENABLED`` is on. This test freezes the base schema (so an
accidental edit that would change today's report.csv header fails loudly) and
pins the exact returns columns.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

# The exact report.csv schema as of ticket #120 (flag OFF). Freezing it here
# guards the "flag-off reproduces today's report byte-for-byte" acceptance
# criterion against accidental base-column edits.
EXPECTED_BASE_FIELDNAMES = [
    "reqno",
    "an",
    "order_datetime_utc",
    "anchor_imputed",
    "evidence_anchor_reason",
    "evidence_anchor_datetime_local",
    "reanchor_gap_hours",
    "products_ordered",
    "diagnosis_codes_n",
    "hb_anchor_datetime_local",
    "hb_anchor_reason",
    "hb_value_g_dl",
    "hb_freshness",
    "hb_source",
    "hb_delta_bypass",
    "hb_needs_review_single_low",
    "cohort_label",
    "cohort_threshold",
    "cohort_evidence_code",
    "cohort_evidence_name",
    "procedure_proximity_hours",
    "upcoming_procedure_hours",
    "crystalloid_liters_prior_4h",
    "anc_value",
    "dispense_datetime_local",
    "use_datetime_local",
    "returned_blood_datetime_local",
    "classification",
    "rationale",
    "bypass_reason",
    "component",
    "platelet_count_k_ul",
    "platelet_freshness",
]

EXPECTED_RETURNS_FIELDNAMES = [
    "returns_disposition",
    "returns_units_total",
    "returns_units_returned",
    "returns_units_transfused",
    "returns_ordered_unit_amount",
    "returns_ledger_complete",
]


def _load_run_pipeline() -> ModuleType:
    pilot_dir = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
    if str(pilot_dir) not in sys.path:
        sys.path.insert(0, str(pilot_dir))
    path = pilot_dir / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("pilot_run_pipeline", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_base_report_schema_is_frozen() -> None:
    mod = _load_run_pipeline()
    assert mod.REPORT_FIELDNAMES == EXPECTED_BASE_FIELDNAMES


def test_returns_fieldnames_are_exact() -> None:
    mod = _load_run_pipeline()
    assert mod.RETURNS_LEDGER_FIELDNAMES == EXPECTED_RETURNS_FIELDNAMES


def test_returns_and_base_fieldnames_are_disjoint() -> None:
    mod = _load_run_pipeline()
    assert set(mod.REPORT_FIELDNAMES).isdisjoint(mod.RETURNS_LEDGER_FIELDNAMES)


def test_base_schema_carries_no_returns_columns() -> None:
    # Flag-off byte identity: none of the returns_* columns may leak into the
    # base schema.
    mod = _load_run_pipeline()
    assert not any(name.startswith("returns_") for name in mod.REPORT_FIELDNAMES)
    assert "returned_blood_datetime_local" in mod.REPORT_FIELDNAMES
