"""Merged verdicts for the ranking (scripts/pilot/rank_merged.py).

The audit store holds LLM-leg rows only, so a ranking read from it counts no
deterministic clear, return or exemption and its denominators are wrong. The
merge takes the deterministic verdict for every scored order and lets the LLM
leg's final verdict replace it where that leg ran.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "pilot" / "rank_merged.py"


@pytest.fixture(scope="module")
def rank() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pilot_rank_merged", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _work(tmp_path: Path) -> Path:
    (tmp_path / "report.csv").write_text(
        "reqno,classification,component\n"
        "R1,APPROPRIATE,red_cell\n"  # deterministic clear, no LLM row
        "R2,NEEDS_REVIEW,red_cell\n"  # routed to the LLM
        "R3,NEEDS_REVIEW,platelet\n"  # routed to the LLM
        "R4,excluded,red_cell\n"  # out of scope
        "R5,RETURNED_NOT_TRANSFUSED,platelet\n"
        "R6,PERIOP_TRANSFUSION_EXEMPT,red_cell\n",
        encoding="utf-8",
    )
    (tmp_path / "llm_report.json").write_text(
        json.dumps(
            [
                {"reqno": "R2", "llm_final": {"final_classification": "INAPPROPRIATE"}},
                {"reqno": "R3", "llm_final": {"final_classification": "APPROPRIATE"}},
                # A stale LLM row for an order the deterministic leg finalised
                # (Codex on the PR): the terminal wins, the row is ignored.
                {"reqno": "R5", "llm_final": {"final_classification": "INAPPROPRIATE"}},
                {"reqno": "R6", "llm_final": {"final_classification": "INAPPROPRIATE"}},
                {"reqno": "R9", "llm_final": None},
            ]
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_llm_final_replaces_the_deterministic_routing_verdict(
    rank: ModuleType, tmp_path: Path
) -> None:
    assert rank.merged_verdicts(_work(tmp_path), "red_cell") == {
        "R1": "APPROPRIATE",
        "R2": "INAPPROPRIATE",
        "R6": "PERIOP_TRANSFUSION_EXEMPT",
    }


def test_deterministic_terminals_stay_and_excluded_orders_drop(
    rank: ModuleType, tmp_path: Path
) -> None:
    assert rank.merged_verdicts(_work(tmp_path), "platelet") == {
        "R3": "APPROPRIATE",
        "R5": "RETURNED_NOT_TRANSFUSED",
    }


def test_components_never_mix(rank: ModuleType, tmp_path: Path) -> None:
    # The trigger columns are Hb for one and platelet count for the other, so
    # a mixed table would average unlike numbers.
    work = _work(tmp_path)
    assert not set(rank.merged_verdicts(work, "red_cell")) & set(
        rank.merged_verdicts(work, "platelet")
    )


def test_an_llm_verdict_for_an_order_missing_from_the_report_fails_loud(
    rank: ModuleType, tmp_path: Path
) -> None:
    # Codex on the PR: iterating the deterministic keys would silently drop an
    # LLM-judged order that a stale report.csv omits, and the ranking would
    # look complete.
    work = _work(tmp_path)
    (work / "llm_report.json").write_text(
        json.dumps(
            [{"reqno": "R99", "llm_final": {"final_classification": "INAPPROPRIATE"}}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="R99"):
        rank.merged_verdicts(work, "red_cell")
