"""The review page per issue event (blood_audit #276, plan 4.3).

A clinician reviewing a REQNO with two issues needs two cases: each issue was
a separate decision, judged at its own Hb and credited to its own linked
order. Each case shows the REQNO's whole unit timeline so the reviewer sees
which units the event covers and which order caused it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from test_pilot_build_review import _load_build_review, _render_review_with_rows

_MANIFEST = "HN,REQNO,AN,component\nHNX,R21,AN1,rbc\nHNY,R5,AN2,rbc\n"
_EVENTS = (
    "HN,REQNO,AN,component,audit_id,event_seq\n"
    "HNX,R21,AN1,red_cell,e1,1\n"
    "HNX,R21,AN1,red_cell,e2,2\n"
)
_REPORT_HEADER = (
    "audit_id,parent_audit_id,reqno,event_seq,events_in_reqno,issue_datetime_local,"
    "issue_units,issue_units_returned,linked_reqno,linkage_rule,"
    "attributed_doctor_differs,declared_use_applies,declared_use_code,"
    "declared_use_label,hb_value_g_dl,classification,rationale,component\n"
)
_REPORT = _REPORT_HEADER + (
    "e1,p21,R21,1,2,2025-01-05 10:00:00,1,1,R21,own,False,True,2,surgery,"
    "11.3,RETURNED_NOT_TRANSFUSED,returned_not_transfused,red_cell\n"
    "e2,p21,R21,2,2,2025-01-07 09:00:00,1,0,F2,fresh_order,True,False,1,ward,"
    "8.5,NEEDS_REVIEW,hb_7_to_10,red_cell\n"
    "p5,p5,R5,,0,,,,,,,,,,,excluded,no_issue_record,\n"
)
_TRANS = (
    "REQNO,AN,DNRNO,SEQNO,BDTYPE,UNITSTAT,PAYDATE,PAYTIME,RTNDATE,RTNTIME\n"
    "R21,AN1,D1,1,LDPRC,3,2025-01-05,100000,2025-01-05,120000\n"
    "R21,AN1,D2,1,LDPRC,2,2025-01-07,90000,,\n"
)


def _render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: object
) -> str:
    kwargs = {
        "manifest_csv": _MANIFEST,
        "report_csv": _REPORT,
        "llm_json": "[]",
        "events_csv": _EVENTS,
        "trans_csv": _TRANS,
        **overrides,
    }
    return _render_review_with_rows(
        _load_build_review(), tmp_path, monkeypatch, **kwargs
    ).decode()


def _case(html: str, number: int) -> str:
    start = html.index(f"id='case-{number}'")
    end = html.find("<section class='case'", start)
    return html[start : end if end != -1 else len(html)]


def test_two_events_of_one_reqno_render_as_two_numbered_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = _render(tmp_path, monkeypatch)

    assert html.count("<section class='case'") == 3
    assert "<h3>Case 1 — REQNO R21 · event 1 of 2</h3>" in html
    assert "<h3>Case 2 — REQNO R21 · event 2 of 2</h3>" in html
    assert "<h3>Case 3 — REQNO R5</h3>" in html


def test_each_case_judges_its_own_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = _render(tmp_path, monkeypatch)
    first, second = _case(html, 1), _case(html, 2)

    assert "Issue datetime (event anchor):</b> 2025-01-05 10:00:00" in first
    assert "cls-returned_not_transfused" in first
    assert "Issue datetime (event anchor):</b> 2025-01-07 09:00:00" in second
    assert "Linked order:</b> F2 (fresh_order" in second
    assert "cls-needs_review" in second


def test_every_event_case_carries_the_reqno_unit_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = _render(tmp_path, monkeypatch)
    second = _case(html, 2)
    start = second.index("REQNO unit timeline")
    timeline = second[start : second.index("</table>", start)]
    rows = re.findall(r"<tr(?: class='this-event')?><td>.*?</tr>", timeline)

    assert len(rows) == 2
    assert "2025-01-05 12:00:00" in rows[0]
    assert "1 of 2" in rows[0] and "own" in rows[0]
    assert rows[1].startswith("<tr class='this-event'>")
    assert "F2" in rows[1] and "fresh_order" in rows[1]
    assert "<td>yes</td><td>no</td>" in rows[1]
    assert second.index("REQNO unit timeline") < second.index("Source data")


def test_pick_is_shown_but_labelled_not_an_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = _render(tmp_path, monkeypatch)

    assert "PICKTIME (not an anchor)" in html
    assert "Dispense datetime (BDVST PICK)" not in html


def test_a_reqno_level_row_is_listed_without_a_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    third = _case(_render(tmp_path, monkeypatch), 3)

    assert "no issue event judged for this REQNO: no_issue_record; 0 measured" in third
    assert "cls-excluded" in third


def test_a_ledger_that_disagrees_with_the_report_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _REPORT.replace("2025-01-07 09:00:00,1,0,F2", "2025-01-07 09:00:00,2,0,F2")

    with pytest.raises(ValueError, match="report.csv says 2"):
        _render(tmp_path, monkeypatch, report_csv=report)


def test_an_event_outside_the_sample_manifest_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="missing from sample_manifest"):
        _render(tmp_path, monkeypatch, manifest_csv="HN,REQNO,AN,component\n")


def test_the_llm_verdict_joins_its_own_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Both events share REQNO R21; a join on REQNO would show event 2's model
    # verdict on event 1 as well.
    llm = json.dumps(
        [
            {
                "audit_id": "e2",
                "reqno": "R21",
                "llm_final": {
                    "final_classification": "INAPPROPRIATE",
                    "confidence": 0.8,
                    "model": "test",
                    "review_reason": "model_verdict",
                    "indications": [],
                    "negative_evidence": [],
                    "reasoning_en": "x",
                    "reasoning_th": "x",
                },
            }
        ]
    )
    html = _render(tmp_path, monkeypatch, llm_json=llm)

    assert "(LLM not run" in _case(html, 1)
    assert "cls-inappropriate" in _case(html, 2)


def test_evidence_windows_read_from_the_issue_not_the_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The REQNO was ordered in December; event 2 was issued on 7 January. An
    # Hb drawn on 6 January is inside the 7-day window before the issue and
    # outside the one before the order: both legs judged event 2 on it, so the
    # page must show it.
    bundle = {
        "BDVST.csv": "REQNO,HN,AN,REQDATE,REQTIME,BDVSTDATE,BDVSTTIME\n"
        "R21,HNX,AN1,2024-12-20,0800,2024-12-20,0800\n",
        "Lab.csv": "AN,LABEXM,NAME_LABEXM,LVSTDATE,LVSTTIME,RESULT\n"
        "AN1,290095,Hb,2025-01-06,1000,8.5\n",
    }
    html = _render(tmp_path, monkeypatch, bundle_files=bundle)

    assert "Hb history (1 rows" in _case(html, 2)
    assert "Hb history (0 rows" in _case(html, 1)


def test_an_event_row_missing_from_the_events_manifest_fails_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Otherwise the event would silently get no case on the page.
    events = "\n".join(_EVENTS.splitlines()[:2]) + "\n"

    with pytest.raises(ValueError, match="missing from events_manifest"):
        _render(tmp_path, monkeypatch, events_csv=events)
