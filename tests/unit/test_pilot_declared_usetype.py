"""Focused coverage for the pilot declared-use seam (ticket #151)."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

from bba.declared_use import DeclaredUse, collapse_usetype

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"


def _load_pilot_module(filename: str, module_name: str) -> ModuleType:
    if str(PILOT_DIR) not in sys.path:
        sys.path.insert(0, str(PILOT_DIR))
    spec = importlib.util.spec_from_file_location(module_name, PILOT_DIR / filename)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_run_pipeline(module_name: str = "pilot_run_pipeline_declared") -> ModuleType:
    return _load_pilot_module("run_pipeline.py", module_name)


def _load_run_llm_leg(module_name: str = "pilot_run_llm_leg_declared") -> ModuleType:
    return _load_pilot_module("run_llm_leg.py", module_name)


def _aggregate_usetype(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], list[str]]:
    values: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        reqno = row["REQNO"]
        values.setdefault(((row.get("HN") or "").strip(), reqno), []).append(
            (row.get("USETYPE") or "").strip()
        )
    return values


def test_hn_reqno_aggregation_mixed_collapses_without_routing(
    monkeypatch, caplog
) -> None:
    monkeypatch.setenv("BBA_PILOT_DECLARED_USETYPE", "1")
    mod = _load_run_pipeline("pilot_run_pipeline_declared_mixed")
    values = _aggregate_usetype(
        [
            {"HN": "AAA", "REQNO": "R1", "USETYPE": "2"},
            {"HN": "AAA", "REQNO": "R1", "USETYPE": "3"},
        ]
    )

    with caplog.at_level(logging.WARNING):
        collapsed = collapse_usetype(values[("AAA", "R1")])

    assert collapsed is None
    assert caplog.records
    assert mod._declared_use_label_for_classifier(collapsed) is None


def test_hn_reqno_aggregation_prevents_cross_hn_leak() -> None:
    reqno = "REUSED"
    values = _aggregate_usetype(
        [
            {"HN": "AAA", "REQNO": reqno, "USETYPE": "2"},
            {"HN": "BBB", "REQNO": reqno, "USETYPE": ""},
        ]
    )

    assert collapse_usetype(values[("BBB", reqno)]) is None
    assert collapse_usetype(["2", ""]) == "2"


def test_run_pipeline_flag_off_report_schema_is_frozen(monkeypatch) -> None:
    monkeypatch.delenv("BBA_PILOT_DECLARED_USETYPE", raising=False)
    mod = _load_run_pipeline("pilot_run_pipeline_declared_schema_off")

    assert mod.DECLARED_USETYPE_FIELDNAMES == [
        "declared_use_code",
        "declared_use_label",
    ]
    assert not any(name.startswith("declared_use_") for name in mod.REPORT_FIELDNAMES)
    assert set(mod.DECLARED_USETYPE_FIELDNAMES).isdisjoint(
        mod.REPORT_FIELDNAMES + mod.RETURNS_LEDGER_FIELDNAMES
    )


def test_run_pipeline_declared_label_helper_gates_on_import_time_env(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BBA_PILOT_DECLARED_USETYPE", "1")
    enabled = _load_run_pipeline("pilot_run_pipeline_declared_helper_on")
    monkeypatch.delenv("BBA_PILOT_DECLARED_USETYPE")
    disabled = _load_run_pipeline("pilot_run_pipeline_declared_helper_off")

    assert enabled._declared_use_label_for_classifier("2") == "surgery"
    assert enabled._declared_use_label_for_classifier("5") == "unknown"
    assert enabled._declared_use_label_for_classifier(None) is None
    assert disabled._declared_use_label_for_classifier("2") is None
    assert disabled._declared_use_label_for_classifier("5") is None
    assert disabled._declared_use_label_for_classifier(None) is None


def test_run_llm_declared_record_is_gated_and_mapped_only(monkeypatch) -> None:
    mod = _load_run_llm_leg("pilot_run_llm_leg_declared_record")
    monkeypatch.setattr(mod.feature_flags, "DECLARED_USETYPE_ENABLED", True)

    assert mod._declared_use_record("2") == DeclaredUse(code="2", label="surgery")
    assert mod._declared_use_record("5") is None
    assert mod._declared_use_record(None) is None

    monkeypatch.setattr(mod.feature_flags, "DECLARED_USETYPE_ENABLED", False)
    assert mod._declared_use_record("2") is None
    assert mod._declared_use_record("5") is None
    assert mod._declared_use_record(None) is None


def test_render_periop_declared_only_is_nonempty_and_hedged() -> None:
    mod = _load_run_llm_leg("pilot_run_llm_leg_declared_render_only")
    rendered = mod._render_periop({"declared_use": {"code": "2", "label": "surgery"}})

    assert "DECLARED INTENT AT ORDER TIME" in rendered
    assert "clinician coded use = surgery" in rendered
    assert "code 2" in rendered
    assert rendered.strip()
    assert "PERI-OP SIGNALS" not in rendered


def test_render_periop_merged_keeps_declared_fact_on_separate_line() -> None:
    mod = _load_run_llm_leg("pilot_run_llm_leg_declared_render_merged")
    rendered = mod._render_periop(
        {
            "surgical_context": True,
            "findings": [
                {
                    "snippet": "underwent laparotomy",
                    "source": "IPDADMPROGRESS",
                    "lag_min": -30,
                }
            ],
            "declared_use": {"code": "3", "label": "type_screen"},
        }
    )
    lines = rendered.splitlines()

    assert len(lines) == 2
    assert lines[0].startswith("PERI-OP SIGNALS: surgery=YES")
    assert "DECLARED INTENT AT ORDER TIME" not in lines[0]
    assert lines[1].startswith("DECLARED INTENT AT ORDER TIME")


def test_render_periop_note_only_output_is_unchanged() -> None:
    mod = _load_run_llm_leg("pilot_run_llm_leg_declared_render_note")
    payload = {
        "surgical_context": True,
        "blood_loss_ml": 450,
        "intraop_transfusion": True,
        "findings": [
            {
                "snippet": "EBL 450 ml",
                "source": "IPDADMPROGRESS",
                "lag_min": -15,
            }
        ],
    }

    assert mod._render_periop(payload) == (
        "PERI-OP SIGNALS: surgery=YES, blood_loss=450 ml, "
        'intra-op transfusion=YES | evidence: "EBL 450 ml" '
        "(IPDADMPROGRESS, 15 min before order)"
    )


def test_render_periop_findings_without_signal_parts_render_empty() -> None:
    # Structural flag-off guard: with no surgical/EBL/intra-op part, the note
    # portion renders nothing (old contract ``if not parts: return ""``) — the
    # evidence quotes must not surface on their own. scan_periop never emits
    # findings without a flag, but the render must not depend on that invariant.
    mod = _load_run_llm_leg("pilot_run_llm_leg_declared_render_findings_only")
    payload = {
        "findings": [
            {"snippet": "orphan note", "source": "IPDADMPROGRESS", "lag_min": -20}
        ]
    }

    assert mod._render_periop(payload) == ""


def test_note_derived_periop_hint_gate_excludes_declared_only() -> None:
    mod = _load_run_llm_leg("pilot_run_llm_leg_declared_hint")

    assert not mod._has_note_derived_periop(
        {"declared_use": {"code": "2", "label": "surgery"}}
    )
    assert mod._has_note_derived_periop({"surgical_context": True})
    assert mod._has_note_derived_periop({"findings": [{"snippet": "op"}]})
    assert mod._has_note_derived_periop({"blood_loss_ml": 100})
    assert mod._has_note_derived_periop({"intraop_transfusion": True})


def test_declared_deferral_is_in_reserve_ahead_dispatch_set() -> None:
    mod = _load_run_llm_leg("pilot_run_llm_leg_declared_dispatch")

    assert mod._RESERVE_AHEAD_RATIONALES == frozenset(
        {"preop_defer_llm", "preop_defer_llm_declared"}
    )


def test_sample_bundle_carries_usetype() -> None:
    mod = _load_pilot_module("sample_bundle.py", "pilot_sample_bundle_declared_usetype")

    assert "USETYPE" in mod.BDVSTDT_COLS
