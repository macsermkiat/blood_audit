"""Follow-up to #124: the platelet-returns short-circuit must be wired in BOTH
pilot legs (``run_pipeline.py`` deterministic + ``run_llm_leg.py`` model) in
lockstep, mirroring ``pipeline._RETURNS_TERMINAL_CLASSIFICATIONS`` (spec #119).

Before this change both legs ``continue``d the platelet path BEFORE the returns
logic, so an all-returned platelet order was classified by the platelet gate (or
LLM-submitted) instead of short-circuiting to ``RETURNED_NOT_TRANSFUSED`` like
the RBC path already does. Production (``pipeline.run_pipeline``) was already
safe via ``_RETURNS_TERMINAL_CLASSIFICATIONS``; the two pilot legs were the gap.

Returns are component-agnostic, so the same terminal that fires for an
all-returned RBC order fires for an all-returned platelet order. Peri-op IS fed
to the platelet check — matching production's tested contract
(``test_platelet_dispatch.test_platelet_periop_transfusion_is_terminal`` feeds a
``periop_summary`` and expects ``PERIOP_TRANSFUSION_EXEMPT``) — so BOTH terminals
are reachable and, crucially, the hard intra-op/EBL contradiction guard stays
active (an all-returned platelet whose notes chart an intra-op transfusion or
EBL >= PERIOP_MIN_EBL_ML falls through instead of being falsely cleared).

The two legs feed peri-op from different windows (the deterministic leg scans
admission-wide — the accepted #123 Risk #3 — while the model leg passes the
bundle's windowed ``periop_summary``, mirroring production), but the pure
``classify()`` decision is identical given the same peri-op. The seam under test
is each leg's module-level ``_platelet_returns_result`` helper plus its
``_RETURNS_TERMINAL_CLASSIFICATIONS`` set; both legs are exercised against
identical (directly-constructed ``PeriopSummary``) cases so the shared decision
logic stays in lockstep and a future edit to one leg's helper fails here.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

from bba.deterministic_classifier import PERIOP_MIN_EBL_ML
from bba.returns_ledger import ReturnsSummary
from bba.vitals_extractor import PeriopSummary

_PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
_ORDER_DT = datetime(2026, 7, 13, tzinfo=UTC)


def _load(module_name: str, filename: str) -> ModuleType:
    if str(_PILOT_DIR) not in sys.path:
        sys.path.insert(0, str(_PILOT_DIR))
    spec = importlib.util.spec_from_file_location(module_name, _PILOT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec_module: the pilot scripts carry frozen dataclasses
    # whose string annotations resolve via sys.modules[cls.__module__].
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_det_leg() -> ModuleType:
    return _load("pilot_run_pipeline_platelet_returns_test", "run_pipeline.py")


def _load_model_leg() -> ModuleType:
    return _load("pilot_run_llm_leg_platelet_returns_test", "run_llm_leg.py")


_LEG_LOADERS = {"det": _load_det_leg, "model": _load_model_leg}


def _all_returned() -> ReturnsSummary:
    return ReturnsSummary(
        units_total=2, units_returned=2, ordered_unit_amount=2, ledger_complete=True
    )


def _transfused() -> ReturnsSummary:
    """A not-all-returned, ledger-complete order (one unit not returned)."""
    return ReturnsSummary(
        units_total=2,
        units_returned=1,
        units_transfused=1,
        ordered_unit_amount=2,
        ledger_complete=True,
    )


def _incomplete() -> ReturnsSummary:
    """Ledger missing lines -> disposition ``inconclusive`` -> no short-circuit."""
    return ReturnsSummary(
        units_total=1, units_returned=1, ordered_unit_amount=2, ledger_complete=False
    )


def _call(leg: ModuleType, summary, periop):
    return leg._platelet_returns_result(
        audit_id="plt",
        order_datetime=_ORDER_DT,
        returns_summary=summary,
        periop=periop,
    )


@pytest.mark.parametrize("leg_key", ["det", "model"])
def test_both_legs_expose_returns_terminal_set(leg_key: str) -> None:
    """Each leg mirrors ``pipeline._RETURNS_TERMINAL_CLASSIFICATIONS`` exactly —
    the two returns terminals, never APPROPRIATE / INAPPROPRIATE."""
    leg = _LEG_LOADERS[leg_key]()
    assert leg._RETURNS_TERMINAL_CLASSIFICATIONS == frozenset(
        {"RETURNED_NOT_TRANSFUSED", "PERIOP_TRANSFUSION_EXEMPT"}
    )


@pytest.mark.parametrize("leg_key", ["det", "model"])
def test_platelet_short_circuit_threads_periop_surgical_gate(
    leg_key: str, monkeypatch
) -> None:
    """Both legs thread their pilot periop use-type gate constant into the
    short-circuit's ClassifierInputs — the det/model legs run the same pure
    ``classify()`` decision and must stay in lockstep on its inputs (spec #119;
    PR #194 Codex P1 found the model leg missing the field)."""
    leg = _LEG_LOADERS[leg_key]()
    monkeypatch.setattr(leg, "RETURNS_LEDGER_ENABLED", True)
    captured = []
    real_classify = leg.classify

    def capture(inputs):  # noqa: ANN001, ANN202 - mirrors leg.classify
        captured.append(inputs)
        return real_classify(inputs)

    monkeypatch.setattr(leg, "classify", capture)
    _call(leg, _transfused(), PeriopSummary())
    (inputs,) = captured
    assert inputs.require_surgical_use_for_periop_exempt is (
        leg.PERIOP_EXEMPT_REQUIRE_SURGICAL_USETYPE_PILOT
    )


@pytest.mark.parametrize("leg_key", ["det", "model"])
def test_all_returned_platelet_short_circuits(leg_key: str, monkeypatch) -> None:
    """An all-returned platelet order (no contradicting peri-op) returns the
    RETURNED_NOT_TRANSFUSED terminal (component-agnostic) so the platelet gate /
    LLM leg is bypassed."""
    leg = _LEG_LOADERS[leg_key]()
    monkeypatch.setattr(leg, "RETURNS_LEDGER_ENABLED", True)
    result = _call(leg, _all_returned(), PeriopSummary())
    assert result is not None
    assert result.classification == "RETURNED_NOT_TRANSFUSED"


@pytest.mark.parametrize("leg_key", ["det", "model"])
def test_all_returned_with_intraop_transfusion_falls_through(
    leg_key: str, monkeypatch
) -> None:
    """Finding 2 guard: an all-returned ledger that CONTRADICTS a charted intra-op
    transfusion must NOT be cleared as RETURNED_NOT_TRANSFUSED — it falls through
    (helper returns None) so the order is judged, never falsely excluded."""
    leg = _LEG_LOADERS[leg_key]()
    monkeypatch.setattr(leg, "RETURNS_LEDGER_ENABLED", True)
    periop = PeriopSummary(intraop_transfusion=True)
    assert _call(leg, _all_returned(), periop) is None


@pytest.mark.parametrize("leg_key", ["det", "model"])
def test_all_returned_with_major_blood_loss_falls_through(
    leg_key: str, monkeypatch
) -> None:
    """Finding 2 guard (EBL arm): major charted blood loss also contradicts an
    all-returned ledger, so the order falls through instead of being cleared."""
    leg = _LEG_LOADERS[leg_key]()
    monkeypatch.setattr(leg, "RETURNS_LEDGER_ENABLED", True)
    periop = PeriopSummary(blood_loss_ml=PERIOP_MIN_EBL_ML)
    assert _call(leg, _all_returned(), periop) is None


@pytest.mark.parametrize("leg_key", ["det", "model"])
def test_transfused_platelet_in_periop_is_exempt(leg_key: str, monkeypatch) -> None:
    """Finding 3: a confirmed-transfused platelet order inside the peri-op
    envelope (surgical context) short-circuits to PERIOP_TRANSFUSION_EXEMPT —
    matching production's tested contract."""
    leg = _LEG_LOADERS[leg_key]()
    monkeypatch.setattr(leg, "RETURNS_LEDGER_ENABLED", True)
    periop = PeriopSummary(surgical_context=True)
    result = _call(leg, _transfused(), periop)
    assert result is not None
    assert result.classification == "PERIOP_TRANSFUSION_EXEMPT"


@pytest.mark.parametrize("leg_key", ["det", "model"])
def test_transfused_platelet_without_periop_falls_through(
    leg_key: str, monkeypatch
) -> None:
    """A confirmed-transfused platelet order with NO peri-op envelope cannot fire
    PERIOP_TRANSFUSION_EXEMPT — the helper returns None and the order falls
    through to the platelet gate."""
    leg = _LEG_LOADERS[leg_key]()
    monkeypatch.setattr(leg, "RETURNS_LEDGER_ENABLED", True)
    assert _call(leg, _transfused(), PeriopSummary()) is None


@pytest.mark.parametrize("leg_key", ["det", "model"])
def test_incomplete_ledger_falls_through(leg_key: str, monkeypatch) -> None:
    """An incomplete ledger is ``inconclusive`` — no short-circuit."""
    leg = _LEG_LOADERS[leg_key]()
    monkeypatch.setattr(leg, "RETURNS_LEDGER_ENABLED", True)
    assert _call(leg, _incomplete(), PeriopSummary()) is None


@pytest.mark.parametrize("leg_key", ["det", "model"])
def test_flag_off_never_short_circuits(leg_key: str, monkeypatch) -> None:
    """Flag-off byte identity: even an all-returned summary with a peri-op
    context yields None, so the platelet path is unchanged from today."""
    leg = _LEG_LOADERS[leg_key]()
    monkeypatch.setattr(leg, "RETURNS_LEDGER_ENABLED", False)
    assert _call(leg, _all_returned(), PeriopSummary(surgical_context=True)) is None


@pytest.mark.parametrize("leg_key", ["det", "model"])
def test_none_summary_falls_through(leg_key: str, monkeypatch) -> None:
    """No ledger coverage (summary is None) -> None even with the flag on."""
    leg = _LEG_LOADERS[leg_key]()
    monkeypatch.setattr(leg, "RETURNS_LEDGER_ENABLED", True)
    assert _call(leg, None, None) is None
