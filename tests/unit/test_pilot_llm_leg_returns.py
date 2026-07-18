"""Ticket #124: the pilot LLM leg (``run_llm_leg.py``) must treat both
returns-ledger terminals as final and never submit them to the model, wired in
lockstep with the deterministic leg / production composer (spec #119).

The seam under test is the module-level ``DETERMINISTIC_FINAL`` set plus the two
gated helpers the inline ``ClassifierInputs`` uses. A returned/exempt row that
reaches ``classify()`` must land in ``DETERMINISTIC_FINAL`` so the leg's
``if cres.classification not in DETERMINISTIC_FINAL: llm_contexts.append(ctx)``
guard drops it from the model work list.
"""

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


def _load_llm_leg(module_name: str = "pilot_run_llm_leg_returns_test") -> ModuleType:
    pilot_dir = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
    if str(pilot_dir) not in sys.path:
        sys.path.insert(0, str(pilot_dir))
    spec = importlib.util.spec_from_file_location(
        module_name, pilot_dir / "run_llm_leg.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hb_missing() -> HbLookupResult:
    return HbLookupResult(
        value_g_dl=None,
        datetime_utc=None,
        source=None,
        freshness="missing",
        delta_hb_bypass=False,
        delta_hb_windows=(),
        needs_review_single_low_hb=False,
    )


def _hb_present() -> HbLookupResult:
    return HbLookupResult(
        value_g_dl=11.0,
        datetime_utc=datetime(2026, 7, 13, tzinfo=UTC),
        source="HEMATOLOGY",
        freshness="fresh",
        delta_hb_bypass=False,
        delta_hb_windows=(),
        needs_review_single_low_hb=False,
    )


_COHORT = CohortAssignment(
    label=CohortLabel.DEFAULT,
    threshold=7.0,
    evidence_code=None,
    evidence_name=None,
)


def test_llm_leg_deterministic_final_includes_returns_terminals() -> None:
    """Both returns terminals join the leg's deterministic-final set beside the
    original three — otherwise a returned/exempt row earns an LLM verdict."""
    leg = _load_llm_leg()
    assert leg.DETERMINISTIC_FINAL >= {
        "APPROPRIATE",
        "INSUFFICIENT_EVIDENCE",
        "INAPPROPRIATE",
        "RETURNED_NOT_TRANSFUSED",
        "PERIOP_TRANSFUSION_EXEMPT",
    }


def test_llm_leg_periop_surgical_gate_env_matrix(monkeypatch) -> None:
    """The LLM leg resolves the periop use-type gate from
    BBA_PILOT_PERIOP_EXEMPT_SURGICAL (default: the library flag) and folds a
    distinct ``+periopgate`` code identity when ON — the audit store is
    idempotent on (run_id, audit_id, code_version), so the verdict-affecting
    gate must not silently reuse a flag-off run's committed rows (PR #194
    Codex P1: the gate was previously inert outside the det pilot leg)."""
    monkeypatch.setenv("BBA_PILOT_PERIOP_EXEMPT_SURGICAL", "0")
    off = _load_llm_leg("pilot_run_llm_leg_periop_gate_off")
    monkeypatch.setenv("BBA_PILOT_PERIOP_EXEMPT_SURGICAL", "1")
    on = _load_llm_leg("pilot_run_llm_leg_periop_gate_on")

    assert off.PERIOP_EXEMPT_REQUIRE_SURGICAL_USETYPE_PILOT is False
    assert on.PERIOP_EXEMPT_REQUIRE_SURGICAL_USETYPE_PILOT is True
    assert "+periopgate" not in off.CODE_VERSION
    assert "+periopgate" in on.CODE_VERSION


def test_llm_leg_returned_summary_is_terminal_and_skips_model(monkeypatch) -> None:
    """An all-returned order routes to RETURNED_NOT_TRANSFUSED, which is in the
    leg's DETERMINISTIC_FINAL set, so the submission guard drops it — the exact
    bug this ticket closes (returned row was previously LLM-submitted)."""
    leg = _load_llm_leg()
    monkeypatch.setattr(leg, "RETURNS_LEDGER_ENABLED", True)
    summary = ReturnsSummary(
        units_total=2, units_returned=2, ordered_unit_amount=2, ledger_complete=True
    )
    result = classify(
        ClassifierInputs(
            audit_id="llm-returned",
            hb_result=_hb_missing(),
            cohort_assignment=_COHORT,
            order_datetime=datetime(2026, 7, 13, tzinfo=UTC),
            procedure_proximity_hours=None,
            crystalloid_liters_prior_4h=0.0,
            returns_disposition=leg._returns_disposition_for_classifier(summary),
            returns_periop_context=leg._returns_periop_context_for_classifier(
                summary,
                surgical_context=False,
                intraop_transfusion=False,
                procedure_proximity_hours=None,
                upcoming_procedure_hours=None,
            ),
        )
    )
    assert result.classification == "RETURNED_NOT_TRANSFUSED"
    assert result.classification in leg.DETERMINISTIC_FINAL


def test_llm_leg_periop_transfused_is_terminal_and_skips_model(monkeypatch) -> None:
    """A confirmed transfusion inside the peri-op envelope routes to
    PERIOP_TRANSFUSION_EXEMPT, also terminal in the leg — never LLM-submitted."""
    leg = _load_llm_leg()
    monkeypatch.setattr(leg, "RETURNS_LEDGER_ENABLED", True)
    summary = ReturnsSummary(
        units_total=2,
        units_returned=0,
        units_transfused=2,
        ordered_unit_amount=2,
        ledger_complete=True,
    )
    result = classify(
        ClassifierInputs(
            audit_id="llm-periop-exempt",
            hb_result=_hb_present(),
            cohort_assignment=_COHORT,
            order_datetime=datetime(2026, 7, 13, tzinfo=UTC),
            procedure_proximity_hours=None,
            upcoming_procedure_hours=48.0,
            crystalloid_liters_prior_4h=0.0,
            returns_disposition=leg._returns_disposition_for_classifier(summary),
            returns_periop_context=leg._returns_periop_context_for_classifier(
                summary,
                surgical_context=False,
                intraop_transfusion=False,
                procedure_proximity_hours=None,
                upcoming_procedure_hours=48.0,
            ),
        )
    )
    assert result.classification == "PERIOP_TRANSFUSION_EXEMPT"
    assert result.classification in leg.DETERMINISTIC_FINAL


def test_llm_leg_flag_off_disposition_inconclusive_not_screened(monkeypatch) -> None:
    """Flag-off byte identity: the helper forces ``inconclusive`` even for an
    all-returned summary, so classify never emits a returns terminal and the
    leg's submission set is unchanged from today."""
    leg = _load_llm_leg()
    monkeypatch.setattr(leg, "RETURNS_LEDGER_ENABLED", False)
    summary = ReturnsSummary(
        units_total=2, units_returned=2, ordered_unit_amount=2, ledger_complete=True
    )
    assert leg._returns_disposition_for_classifier(summary) == "inconclusive"
    result = classify(
        ClassifierInputs(
            audit_id="llm-flag-off",
            hb_result=_hb_present(),
            cohort_assignment=_COHORT,
            order_datetime=datetime(2026, 7, 13, tzinfo=UTC),
            procedure_proximity_hours=None,
            crystalloid_liters_prior_4h=0.0,
            returns_disposition=leg._returns_disposition_for_classifier(summary),
        )
    )
    assert result.classification not in {
        "RETURNED_NOT_TRANSFUSED",
        "PERIOP_TRANSFUSION_EXEMPT",
    }


def test_llm_leg_code_version_is_flag_sensitive(monkeypatch) -> None:
    """Ticket #124 run identity: enabling the returns ledger must yield a
    DISTINCT code identity so a re-run cannot silently reuse a flag-off run's
    committed audit_store rows. Flag-off keeps the original identity (byte
    identity). The flag is captured at import, so each fresh load reflects it."""
    import bba.feature_flags as feature_flags

    # Isolate the returns flag: force the other now-ON default seams off so they
    # do not add their own CODE_VERSION tokens ("+declared", "+msbos5",
    # "+periopgate").
    monkeypatch.setenv("BBA_PILOT_DECLARED_USETYPE", "0")
    monkeypatch.setenv("BBA_PILOT_MSBOS_RESERVATION", "0")
    monkeypatch.setenv("BBA_PILOT_PERIOP_EXEMPT_SURGICAL", "0")

    monkeypatch.setattr(feature_flags, "RETURNS_LEDGER_ENABLED", True)
    assert _load_llm_leg().CODE_VERSION == "pilot-mini+returns"

    monkeypatch.setattr(feature_flags, "RETURNS_LEDGER_ENABLED", False)
    assert _load_llm_leg().CODE_VERSION == "pilot-mini"


def test_llm_leg_periop_envelope_is_windowed(monkeypatch) -> None:
    """Regression guard for #123's P1 in the LLM leg: a remote surgery (well
    beyond the 6h/72h windows) must NOT establish peri-op context, so it cannot
    exempt an unrelated transfusion."""
    leg = _load_llm_leg()
    monkeypatch.setattr(leg, "RETURNS_LEDGER_ENABLED", True)
    summary = ReturnsSummary(
        units_total=1,
        units_returned=0,
        units_transfused=1,
        ordered_unit_amount=1,
        ledger_complete=True,
    )
    assert (
        leg._returns_periop_context_for_classifier(
            summary,
            surgical_context=False,
            intraop_transfusion=False,
            procedure_proximity_hours=999.0,
            upcoming_procedure_hours=999.0,
        )
        is False
    )
    assert (
        leg._returns_periop_context_for_classifier(
            summary,
            surgical_context=False,
            intraop_transfusion=False,
            procedure_proximity_hours=None,
            upcoming_procedure_hours=48.0,
        )
        is True
    )
