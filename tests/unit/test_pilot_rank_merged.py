"""Merged verdicts for the ranking (scripts/pilot/rank_merged.py).

The audit store holds LLM-leg rows only, so a ranking read from it counts no
deterministic clear, return or exemption and its denominators are wrong. The
merge takes the deterministic verdict for every scored order and lets the LLM
leg's final verdict replace it where that leg ran. Records are per issue
event: they join on audit_id, never on REQNO, because one REQNO can hold
events with different verdicts and different linked orders (blood_audit #276).
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


_REPORT_HEADER = "audit_id,reqno,linked_reqno,event_seq,classification,component\n"


def _work(tmp_path: Path) -> Path:
    (tmp_path / "report.csv").write_text(
        _REPORT_HEADER
        + "a1,R1,R1,1,APPROPRIATE,red_cell\n"  # deterministic clear, no LLM row
        "a2,R2,R2,1,NEEDS_REVIEW,red_cell\n"  # routed to the LLM
        "a3,R3,R3,1,NEEDS_REVIEW,platelet\n"  # routed to the LLM
        "p4,R4,,,excluded,\n"  # REQNO-level row: listed, not scored
        "a5,R5,R5,1,RETURNED_NOT_TRANSFUSED,platelet\n"
        "a6,R6,R6,1,PERIOP_TRANSFUSION_EXEMPT,red_cell\n",
        encoding="utf-8",
    )
    (tmp_path / "llm_report.json").write_text(
        json.dumps(
            [
                {
                    "audit_id": "a2",
                    "reqno": "R2",
                    "llm_final": {"final_classification": "INAPPROPRIATE"},
                },
                {
                    "audit_id": "a3",
                    "reqno": "R3",
                    "llm_final": {"final_classification": "APPROPRIATE"},
                },
                # A stale LLM row for an event the deterministic leg finalised
                # (Codex on the PR): the terminal wins, the row is ignored.
                {
                    "audit_id": "a5",
                    "reqno": "R5",
                    "llm_final": {"final_classification": "INAPPROPRIATE"},
                },
                {
                    "audit_id": "a6",
                    "reqno": "R6",
                    "llm_final": {"final_classification": "INAPPROPRIATE"},
                },
                {"audit_id": "a9", "reqno": "R9", "llm_final": None},
            ]
        ),
        encoding="utf-8",
    )
    return tmp_path


def _by_id(records: tuple) -> dict[str, str]:
    return {r.audit_id: r.verdict for r in records}


def test_llm_final_replaces_the_deterministic_routing_verdict(
    rank: ModuleType, tmp_path: Path
) -> None:
    assert _by_id(rank.merged_verdicts(_work(tmp_path), "red_cell")) == {
        "a1": "APPROPRIATE",
        "a2": "INAPPROPRIATE",
        "a6": "PERIOP_TRANSFUSION_EXEMPT",
    }


def test_deterministic_terminals_stay_and_excluded_rows_drop(
    rank: ModuleType, tmp_path: Path
) -> None:
    assert _by_id(rank.merged_verdicts(_work(tmp_path), "platelet")) == {
        "a3": "APPROPRIATE",
        "a5": "RETURNED_NOT_TRANSFUSED",
    }


def test_components_never_mix(rank: ModuleType, tmp_path: Path) -> None:
    # The trigger columns are Hb for one and platelet count for the other, so
    # a mixed table would average unlike numbers.
    work = _work(tmp_path)
    assert not set(_by_id(rank.merged_verdicts(work, "red_cell"))) & set(
        _by_id(rank.merged_verdicts(work, "platelet"))
    )


def test_an_llm_verdict_for_an_event_missing_from_the_report_fails_loud(
    rank: ModuleType, tmp_path: Path
) -> None:
    # Codex on the PR: iterating the deterministic keys would silently drop an
    # LLM-judged event that a stale report.csv omits, and the ranking would
    # look complete.
    work = _work(tmp_path)
    (work / "llm_report.json").write_text(
        json.dumps(
            [
                {
                    "audit_id": "a99",
                    "reqno": "R99",
                    "llm_final": {"final_classification": "INAPPROPRIATE"},
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="a99"):
        rank.merged_verdicts(work, "red_cell")


def _case_21_work(tmp_path: Path) -> Path:
    """Four events of one REQNO: event 1 returned under its own order, events
    2-4 caused by fresh orders. Only event 3 went to the LLM."""
    (tmp_path / "report.csv").write_text(
        _REPORT_HEADER + "e1,R21,R21,1,RETURNED_NOT_TRANSFUSED,red_cell\n"
        "e2,R21,F2,2,NEEDS_REVIEW,red_cell\n"
        "e3,R21,F3,3,NEEDS_REVIEW,red_cell\n"
        "e4,R21,F4,4,PERIOP_TRANSFUSION_EXEMPT,red_cell\n",
        encoding="utf-8",
    )
    (tmp_path / "llm_report.json").write_text(
        json.dumps(
            [
                {
                    "audit_id": "e3",
                    "reqno": "R21",
                    "llm_final": {"final_classification": "APPROPRIATE"},
                }
            ]
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_every_event_of_a_reqno_is_kept_with_its_own_verdict_and_link(
    rank: ModuleType, tmp_path: Path
) -> None:
    # Before #276 the merge keyed on REQNO and silently kept the last event.
    records = rank.merged_verdicts(_case_21_work(tmp_path), "red_cell")

    assert [(r.audit_id, r.linked_reqno, r.verdict) for r in records] == [
        ("e1", "R21", "RETURNED_NOT_TRANSFUSED"),
        ("e2", "F2", "NEEDS_REVIEW"),
        ("e3", "F3", "APPROPRIATE"),
        ("e4", "F4", "PERIOP_TRANSFUSION_EXEMPT"),
    ]


def test_a_repeated_audit_id_in_the_report_fails_loud(
    rank: ModuleType, tmp_path: Path
) -> None:
    work = _case_21_work(tmp_path)
    with (work / "report.csv").open("a", encoding="utf-8") as fh:
        fh.write("e2,R21,F2,2,APPROPRIATE,red_cell\n")

    with pytest.raises(ValueError, match="e2"):
        rank.merged_verdicts(work, "red_cell")


def test_a_reqno_level_report_without_audit_ids_fails_loud(
    rank: ModuleType, tmp_path: Path
) -> None:
    (tmp_path / "report.csv").write_text(
        "reqno,classification,component\nR1,APPROPRIATE,red_cell\n",
        encoding="utf-8",
    )
    (tmp_path / "llm_report.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="audit_id"):
        rank.merged_verdicts(tmp_path, "red_cell")
