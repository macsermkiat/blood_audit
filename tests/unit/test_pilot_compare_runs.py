"""v2 (per REQNO) against v3 (per issue event), counts only (plan 4.3, 6.2).

The owner decides the linkage and declared-use defaults from these counts on
January v3, so each count must mean what it says: every v2 row maps to v3
events or to one REQNO-level row, and the re-ask list names exactly the events
whose linked order the other linkage rule would change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import compare_runs
from bba.audit_orders import build_audit_id

_V3_HEADER = (
    "audit_id,parent_audit_id,reqno,event_seq,events_in_reqno,linked_reqno,"
    "linked_reqno_latest_before,linkage_rule,attributed_doctor_differs,"
    "declared_use_applies,hb_value_g_dl,classification,rationale,component\n"
)


def _v2(root: Path) -> Path:
    root.mkdir()
    (root / "sample_manifest.csv").write_text(
        "HN,REQNO,AN,component\nH1,R21,A1,rbc\nH2,R7,A2,rbc\nH3,R5,A3,rbc\n",
        encoding="utf-8",
    )
    (root / "report.csv").write_text(
        "reqno,hb_value_g_dl,classification,component\n"
        "R21,11.3,NEEDS_REVIEW,red_cell\n"
        "R7,6.5,APPROPRIATE,red_cell\n"
        "R5,,POTENTIALLY_INAPPROPRIATE,platelet\n",
        encoding="utf-8",
    )
    (root / "llm_report.json").write_text(
        json.dumps(
            [{"reqno": "R21", "llm_final": {"final_classification": "INAPPROPRIATE"}}]
        ),
        encoding="utf-8",
    )
    return root


def _v3(root: Path) -> Path:
    root.mkdir()
    p21, p7, p5 = (
        build_audit_id(h, r) for h, r in (("H1", "R21"), ("H2", "R7"), ("H3", "R5"))
    )
    (root / "report.csv").write_text(
        _V3_HEADER
        + f"e1,{p21},R21,1,3,R21,F0,own,False,True,11.3,RETURNED_NOT_TRANSFUSED,x,red_cell\n"
        f"e2,{p21},R21,2,3,F2,F2,fresh_order,True,False,8.5,NEEDS_REVIEW,x,red_cell\n"
        f"e3,{p21},R21,3,3,R21,F2,standing_order,False,False,7.6,APPROPRIATE,x,red_cell\n"
        f"e7,{p7},R7,1,1,R7,R7,own,False,False,6.5,APPROPRIATE,x,red_cell\n"
        f"{p5},{p5},R5,,0,,,,,,,excluded,no_issue_record,\n",
        encoding="utf-8",
    )
    (root / "llm_report.json").write_text(
        json.dumps(
            [{"audit_id": "e2", "llm_final": {"final_classification": "APPROPRIATE"}}]
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def result(tmp_path: Path) -> compare_runs.Comparison:
    return compare_runs.compare(_v2(tmp_path / "v2"), _v3(tmp_path / "v3"))


def test_every_v2_row_maps_to_events_or_a_reqno_level_row(
    result: compare_runs.Comparison,
) -> None:
    c = result.counts
    assert c["v2_rows"] == 3
    assert c["v2_rows_mapped_to_events"] == 2
    assert c["v2_rows_mapped_to_reqno_level_row"] == 1
    assert "v2_rows_unmapped" not in c


def test_event_counts_and_linkage_tags(result: compare_runs.Comparison) -> None:
    c = result.counts
    assert c["v3_event_rows"] == 4
    assert c["v3_reqnos_with_events"] == 2
    assert c["v3_reqno_level:no_issue_record"] == 1
    assert c["events_per_reqno:3"] == 1
    assert c["events_per_reqno:1"] == 1
    assert (
        c["linkage:own"],
        c["linkage:fresh_order"],
        c["linkage:standing_order"],
    ) == (
        2,
        1,
        1,
    )
    assert c["attributed_doctor_differs:True"] == 1


def test_re_ask_list_names_events_the_other_linkage_rule_relinks(
    result: compare_runs.Comparison,
) -> None:
    # e1 (own, latest-before F0) and e3 (standing, latest-before F2) change
    # linked order under the #276 rule; e2 and e7 do not.
    assert result.relink_audit_ids == ["e1", "e3"]
    assert result.counts["events_relinked_under_latest_before"] == 2


def test_first_event_is_compared_with_the_v2_verdict_and_hb(
    result: compare_runs.Comparison,
) -> None:
    by_reqno = {r["reqno"]: r for r in result.per_reqno}
    assert by_reqno["R21"]["v2_final"] == "INAPPROPRIATE"
    assert (
        by_reqno["R21"]["v3_finals"]
        == "RETURNED_NOT_TRANSFUSED|APPROPRIATE|APPROPRIATE"
    )
    assert by_reqno["R21"]["v3_hbs"] == "11.3|8.5|7.6"
    assert by_reqno["R5"]["v3_finals"] == "excluded"
    assert result.counts["first_event_hb_equals_v2"] == 2
    assert result.counts["first_event_verdict_equals_v2"] == 1


def test_an_unmapped_v2_row_makes_the_script_exit_non_zero(tmp_path: Path) -> None:
    v2, v3 = _v2(tmp_path / "v2"), _v3(tmp_path / "v3")
    with (v2 / "sample_manifest.csv").open("a", encoding="utf-8") as fh:
        fh.write("H9,R9,A9,rbc\n")
    with (v2 / "report.csv").open("a", encoding="utf-8") as fh:
        fh.write("R9,7.0,APPROPRIATE,red_cell\n")

    assert compare_runs.main([str(v2), str(v3)]) == 1
    assert (v3 / compare_runs.LINKAGE_LIST).read_text() == "e1,e3"
    assert not (v2 / compare_runs.PER_REQNO_CSV).exists()
