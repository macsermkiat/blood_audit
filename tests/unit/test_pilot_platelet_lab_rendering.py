"""Platelet count evidence must reach the platelet LLM prompt legibly.

The bundle emits platelet counts under the ``Lab`` source with
``test == "platelet_count"``. The first real-data platelet LLM run
(2026-09-20) rendered every one of them with the Hb template, i.e. the blank
string ``Hb  g/dL () at``, so the model judged platelet orders without the
trigger count and answered INSUFFICIENT_EVIDENCE / INAPPROPRIATE for orders
whose count was on file.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"

_ANCHOR = datetime(2026, 7, 16, 4, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def llm_leg() -> ModuleType:
    if str(PILOT_DIR) not in sys.path:
        sys.path.insert(0, str(PILOT_DIR))
    spec = importlib.util.spec_from_file_location(
        "pilot_run_llm_leg_plt_lab_rendering", PILOT_DIR / "run_llm_leg.py"
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _platelet_payload(value_k_ul: float) -> dict[str, object]:
    return {
        "test": "platelet_count",
        "value_k_ul": value_k_ul,
        "unit": "10^3/uL",
        "lab_source": "HEMATOLOGY",
        "item_no": "1",
    }


def test_platelet_lab_payload_renders_the_count_not_a_blank_hb_line(
    llm_leg: ModuleType,
) -> None:
    text = llm_leg._render_payload("Lab", _platelet_payload(11.0))

    assert "Hb" not in text
    assert "Platelet count" in text
    # The prompt states thresholds per uL (10,000 /uL), so the per-uL figure
    # must be explicit rather than left to the model's unit conversion.
    assert "11,000 /uL" in text


def test_hb_lab_payload_rendering_is_unchanged(llm_leg: ModuleType) -> None:
    # RBC prompts must stay byte-identical: the RBC verdicts on file were
    # produced from this exact string.
    payload = {"value_g_dl": 7.8, "lab_source": "HEMATOLOGY", "item_no": "1"}

    assert llm_leg._render_payload("Lab", payload) == "Hb 7.8 g/dL () at "


def test_closest_pre_order_count_is_flagged_with_its_age(llm_leg: ModuleType) -> None:
    # The count that triggered the order is the one the threshold rule applies
    # to; older counts are trend context only.
    text = llm_leg._annotate_platelet_lab(
        "Platelet count 11 10^3/uL (11,000 /uL)",
        timestamp_utc=_ANCHOR - timedelta(hours=3),
        anchor_utc=_ANCHOR,
        is_closest=True,
    )

    assert text.endswith("[closest pre-order platelet count; 3.0h before order]")


def test_older_count_carries_age_only(llm_leg: ModuleType) -> None:
    text = llm_leg._annotate_platelet_lab(
        "Platelet count 25 10^3/uL (25,000 /uL)",
        timestamp_utc=_ANCHOR - timedelta(hours=30),
        anchor_utc=_ANCHOR,
        is_closest=False,
    )

    assert text.endswith("[30.0h before order]")
    assert "closest" not in text
