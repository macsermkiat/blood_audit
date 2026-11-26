"""The LLM pilot leg asks about one issue event at a time (#276, plan PR 7).

Each issue of blood is judged against the order that caused it, on the labs at
the moment it left the bank. These tests run ``run_llm_leg.main()`` over small
synthetic bundles: in privacy-prepare mode they read the exact prompt requests
the leg would submit; without a privacy policy every submission is refused
admission locally, so the store and llm_report.json fill without any model
call.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest

import bba.feature_flags as feature_flags
from bba.audit_orders import build_audit_id, build_event_audit_id

from bba.platelet_lookup import PLATELET_LABEXM

from issue_event_bundle import Bundle, _day, _time
from privacy_requests import PrivacyRequests, load_requests

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
OLD_CODE_VERSION = "pilot-mini+returns+declared+msbos5+opbound2+usetypeonly"
INTRAOP_TRANSFUSION_NOTE = "intraop LPRC 2 units given"


def _load(filename: str, name: str) -> ModuleType:
    if str(PILOT_DIR) not in sys.path:
        sys.path.insert(0, str(PILOT_DIR))
    spec = importlib.util.spec_from_file_location(name, PILOT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LEG = _load("run_llm_leg.py", "pilot_run_llm_leg_events_test")
RP = _load("run_pipeline.py", "pilot_run_pipeline_parity_test")


@pytest.fixture
def leg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    for flag in (
        "RESERVE_AHEAD_ROUTER_ENABLED",
        "DECLARED_USETYPE_ENABLED",
        "MSBOS_RESERVATION_ENABLED",
        "MSBOS_PLANNED_OP_PICKER_V2_ENABLED",
        "DECLARED_USE_PREOP_EXEMPT_ENABLED",
        "PLATELET_TREND_GUARDRAIL_ENABLED",
        "PLATELET_SPECIALIST_TARGET_GUARDRAIL_ENABLED",
    ):
        monkeypatch.setattr(feature_flags, flag, getattr(feature_flags, flag))
    for env in ("BBA_BDVSTTRANS_CSV", "BBA_PRIVACY_ARTIFACTS", "BBA_PRIVACY_POLICY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv("BBA_PILOT_PRIVACY_PREPARE", raising=False)
    monkeypatch.setenv("BBA_PILOT_PLATELET_LLM", "1")
    monkeypatch.setattr(LEG, "WORK", tmp_path)
    monkeypatch.setattr(LEG, "BUNDLE", tmp_path / "bundle")
    monkeypatch.setattr(LEG, "AUDIT_STORE_ROOT", tmp_path / "data" / "audit_store")
    monkeypatch.setattr(LEG, "RUN_ID", "run-events-1")
    monkeypatch.setattr(LEG, "ONLY_REQNOS", frozenset())
    monkeypatch.setattr(LEG, "ONLY_AUDIT_IDS", frozenset())
    monkeypatch.setattr(LEG, "RETURNS_LEDGER_ENABLED", True)
    monkeypatch.setattr(LEG, "DECLARED_USETYPE_PILOT_ENABLED", True)
    monkeypatch.setattr(LEG, "DECLARED_USE_PREOP_EXEMPT_PILOT_ENABLED", True)
    monkeypatch.setattr(LEG, "MSBOS_RESERVATION_PILOT_ENABLED", False)
    yield LEG


def _prepare(
    leg: ModuleType, monkeypatch: pytest.MonkeyPatch, bundle: Bundle
) -> PrivacyRequests:
    bundle.write(leg.WORK)
    path = leg.WORK / "requests.json"
    monkeypatch.setenv("BBA_PILOT_PRIVACY_PREPARE", str(path))
    with pytest.raises(SystemExit) as stop:
        leg.main()
    assert stop.value.code == 0
    monkeypatch.delenv("BBA_PILOT_PRIVACY_PREPARE")
    return load_requests(path)


def _run(leg: ModuleType, bundle: Bundle | None = None) -> list[dict[str, object]]:
    if bundle is not None:
        bundle.write(leg.WORK)
    leg.main()
    return json.loads((leg.WORK / "llm_report.json").read_text())


def _audit_id(reqno: str, seq: str, bdtype: str = "LDPRC2") -> str:
    """The event's identity: its first unit (Bundle.unit names it DN<REQNO><seq>)."""
    return build_event_audit_id("H1", reqno, (f"DN{reqno}{seq}", seq, bdtype))


def _text(requests: PrivacyRequests, audit_id: str, source: str) -> list[str]:
    chunks = requests.cases[audit_id].prompt_request.evidence_chunks
    return [c.text for c in chunks if c.source == source]


def _two_reviewed_events() -> Bundle:
    """One REQNO issuing twice, 28 h apart, each on a gray-zone Hb."""
    b = Bundle()
    b.order("R1")
    b.unit("R1", "1", 2.0)
    b.unit("R1", "2", 30.0)
    b.hb(1, "8.0")
    b.hb(29, "8.2")
    return b


def test_privacy_day_0_is_the_issue_date_not_the_order_date(
    leg: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 8.16: the date shift is anchored on the day the blood was issued."""
    b = Bundle()
    b.order("R1", hours=14)  # 22:00 on 10 January, Bangkok
    b.unit("R1", "1", 19.0)  # issued 03:00 on 11 January
    b.hb(18, "8.0")

    requests = _prepare(leg, monkeypatch, b)

    assert requests.cases[_audit_id("R1", "1")].anchor_date == date(2025, 1, 11)


def test_two_events_of_one_reqno_are_two_submissions_with_distinct_audit_ids(
    leg: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = _prepare(leg, monkeypatch, _two_reviewed_events())

    assert set(requests.cases) == {_audit_id("R1", "1"), _audit_id("R1", "2")}
    assert build_audit_id("H1", "R1") not in requests.cases
    assert _text(requests, _audit_id("R1", "2"), "Issue_Event") == [
        "ISSUE EVENT (blood bank record): this audit judges issue 2 of 2 under "
        "this blood request. Units issued in this event: 1; returned to the "
        "blood bank unused: 0. Linked order: this request's order; no newer "
        "order was placed; issued 30.0 h after that order. Declared surgical "
        "use applies to this issue: no."
    ]


def test_each_event_prompt_counts_its_own_returned_units(
    leg: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owner decision 1.3: returns are counted per event, not per REQNO."""
    b = Bundle()
    b.order("R1", unitamt="3")
    b.unit("R1", "1", 2.0)
    b.unit("R1", "2", 2.5, status="3")
    b.unit("R1", "3", 30.0)
    b.hb(1, "8.0")
    b.hb(29, "8.2")

    requests = _prepare(leg, monkeypatch, b)

    first = _text(requests, _audit_id("R1", "1"), "Issue_Event")[0]
    second = _text(requests, _audit_id("R1", "3"), "Issue_Event")[0]
    assert (
        "Units issued in this event: 2; returned to the blood bank unused: 1." in first
    )
    assert (
        "Units issued in this event: 1; returned to the blood bank unused: 0." in second
    )


def test_standing_order_event_of_a_declaring_reqno_is_judged_without_the_declaration(
    leg: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """8.15: the declared surgery exempts event 1 (never submitted); the later
    standing-order event goes to the model with no declared intent in it."""
    b = Bundle()
    b.order("R1", usetype="2")
    b.unit("R1", "1", 2.0)
    b.unit("R1", "2", 40.0)
    b.hb(1, "8.0")
    b.hb(39, "8.2")

    requests = _prepare(leg, monkeypatch, b)

    standing = _audit_id("R1", "2")
    assert set(requests.cases) == {standing}
    assert _text(requests, standing, "Issue_Event")[0].endswith(
        "Declared surgical use applies to this issue: no."
    )
    assert not any(
        "DECLARED INTENT" in text for text in _text(requests, standing, "Periop")
    )


def test_fresh_order_event_prompt_names_the_newer_order(
    leg: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    b = Bundle()
    b.order("R1")
    b.order("R2", hours=20, status="1", audited=False)
    b.unit("R1", "1", 2.0)
    b.unit("R1", "2", 27.0)
    b.hb(1, "8.0")
    b.hb(26, "8.2")

    requests = _prepare(leg, monkeypatch, b)

    assert (
        "Linked order: a newer order placed after the previous issue; issued "
        "7.0 h after that order."
    ) in _text(requests, _audit_id("R1", "2"), "Issue_Event")[0]


@pytest.mark.parametrize(
    ("prior_reqno", "prior_status", "low_is_minimum"),
    [("R1", "2", False), ("R0", "2", False), ("R1", "3", True)],
    ids=["same-reqno", "other-month", "returned-prior"],
)
def test_prior_transfusion_bounds_the_24h_minimum_the_model_is_shown(
    leg: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    prior_reqno: str,
    prior_status: str,
    low_is_minimum: bool,
) -> None:
    """#249: a low the previous transfusion already corrected (under this REQNO
    or one outside the run on the same admission) is not the 24 h minimum; an
    issue whose units all came back gave no blood and corrects nothing."""
    b = Bundle()
    b.order("R1")
    if prior_reqno == "R0":
        b.order("R0", hours=-1, audited=False)
    b.unit(prior_reqno, "1", 2.0, status=prior_status)
    b.unit("R1", "2", 20.0)
    b.hb(1, "6.5")
    b.hb(19, "8.5")

    requests = _prepare(leg, monkeypatch, b)

    labs = _text(requests, _audit_id("R1", "2"), "Lab")
    low = next(t for t in labs if t.startswith("Hb 6.5"))
    later = next(t for t in labs if t.startswith("Hb 8.5"))
    assert ("minimum in 24h pre-dispense" in low) is low_is_minimum
    assert ("minimum in 24h pre-dispense" in later) is not low_is_minimum


def test_hb_is_read_at_the_issue_not_at_the_order(
    leg: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A draw between the order and the issue is the one the event is judged on."""
    b = Bundle()
    b.order("R1")
    b.unit("R1", "1", 10.0)
    b.hb(-1, "6.0")
    b.hb(9, "8.4")

    report_requests = _prepare(leg, monkeypatch, b)

    labs = _text(report_requests, _audit_id("R1", "1"), "Lab")
    assert any(t.startswith("Hb 8.4") and "closest pre-dispense Hb" in t for t in labs)


def _seed_stale_rows(leg: ModuleType) -> None:
    """Commit, under the leg's run id, rows an older code identity wrote: one
    under the REQNO-level (v2) audit_id and one under the event's audit_id."""
    store_cls, config_cls = leg.AuditStore, leg.AuditStoreConfig
    fresh = store_cls(
        config_cls(root_dir=leg.AUDIT_STORE_ROOT, code_version=leg.CODE_VERSION)
    )
    row = fresh.read_audit_results(run_id=leg.RUN_ID)[0]
    calls = fresh.read_llm_calls(run_id=leg.RUN_ID)
    for path in leg.AUDIT_STORE_ROOT.rglob("*"):
        if path.is_file():
            path.unlink()
    stale = store_cls(
        config_cls(root_dir=leg.AUDIT_STORE_ROOT, code_version=OLD_CODE_VERSION)
    )
    for audit_id in (row.audit_id, build_audit_id("H1", "R1")):
        stale.write(
            row.model_copy(
                update={
                    "audit_id": audit_id,
                    "final_classification": "APPROPRIATE",
                    "review_reason": "stale-v2-row",
                }
            ),
            [
                c.model_copy(
                    update={"audit_id": audit_id, "call_id": f"{audit_id}-{i}"}
                )
                for i, c in enumerate(calls)
            ],
        )


def test_rows_of_another_code_version_are_not_served(leg: ModuleType) -> None:
    b = Bundle()
    b.order("R1")
    b.unit("R1", "1", 2.0)
    b.hb(1, "8.0")
    _run(leg, b)
    _seed_stale_rows(leg)

    report = _run(leg)

    assert [rec["audit_id"] for rec in report] == [_audit_id("R1", "1")]
    final = report[0]["llm_final"]
    assert isinstance(final, dict)
    assert final["review_reason"] != "stale-v2-row"
    assert final["final_classification"] == "NEEDS_REVIEW"


@pytest.mark.parametrize("knob", ["ONLY_AUDIT_IDS", "ONLY_REQNOS"])
def test_filtered_rerun_replaces_only_the_selected_events_by_audit_id(
    leg: ModuleType, monkeypatch: pytest.MonkeyPatch, knob: str
) -> None:
    """The re-ask list names events; BBA_PILOT_ONLY_REQNO names all of a REQNO's."""
    b = _two_reviewed_events()
    b.order("S1", an="A2")
    b.unit("S1", "1", 2.0, an="A2")
    b.hb(1, "8.0", an="A2")
    first = _run(leg, b)
    assert {rec["audit_id"] for rec in first} == {
        _audit_id("R1", "1"),
        _audit_id("R1", "2"),
        _audit_id("S1", "1"),
    }
    target = {
        "ONLY_AUDIT_IDS": frozenset({_audit_id("R1", "2")}),
        "ONLY_REQNOS": frozenset({"R1"}),
    }[knob]
    monkeypatch.setattr(leg, knob, target)
    monkeypatch.setattr(leg, "RUN_ID", "run-events-2")
    lab = leg.WORK / "bundle" / "Lab.csv"
    lab.write_text(lab.read_text().replace(",8.2\n", ",8.6\n"))

    merged = _run(leg)

    by_id = {rec["audit_id"]: rec for rec in merged}
    assert len(merged) == len(by_id) == 3
    deterministic = by_id[_audit_id("R1", "2")]["deterministic"]
    assert isinstance(deterministic, dict) and deterministic["hb"] == 8.6
    store = leg.AuditStore(
        leg.AuditStoreConfig(
            root_dir=leg.AUDIT_STORE_ROOT, code_version=leg.CODE_VERSION
        )
    )
    asked = {r.audit_id for r in store.read_audit_results(run_id="run-events-2")}
    expected = {_audit_id("R1", "2")} | (
        {_audit_id("R1", "1")} if knob == "ONLY_REQNOS" else set()
    )
    assert asked == expected


def test_unknown_audit_id_in_the_reask_list_stops_the_run(
    leg: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(leg, "ONLY_AUDIT_IDS", frozenset({"0" * 32}))
    with pytest.raises(SystemExit, match="no issue event"):
        _run(leg, _two_reviewed_events())


def test_llm_report_entry_carries_the_event_fields(leg: ModuleType) -> None:
    b = Bundle()
    b.order("R1")
    b.unit("R1", "1", 2.0)
    b.unit("R1", "2", 20.0)
    b.hb(1, "8.0")
    b.hb(19, "8.2")
    report = _run(leg, b)

    second = next(rec for rec in report if rec["event_seq"] == 2)
    assert second["parent_audit_id"] == build_audit_id("H1", "R1")
    assert second["linked_reqno"] == "R1"
    assert second["linkage_rule"] == "standing_order"
    assert second["issue_datetime_local"] == "2025-01-11 04:00:00"
    assert second["declared_use_applies"] is False
    assert second["prior_event_cutoff_local"] == "2025-01-10 10:00:00"


# --- the all-returned notes guard: both legs, one window ---------------------


def _all_returned_second_event(
    product: str, bdtype: str, second_issue: float, note_hours: float
) -> Bundle:
    b = Bundle()
    b.order("W1", product=product)
    b.unit("W1", "1", 2.0, bdtype=bdtype)
    b.unit("W1", "2", second_issue, status="3", bdtype=bdtype)
    b.hb(1, "8.0")
    b.hb(second_issue - 1, "8.2")
    for hours in (1, second_issue - 1):
        b.add("Lab.csv", "H1", "A1", _day(hours), _time(hours), PLATELET_LABEXM, "40")
    b.note(note_hours, INTRAOP_TRANSFUSION_NOTE)
    return b


def _deterministic_classifications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bundle: Bundle
) -> dict[str, str]:
    root = tmp_path / "det"
    doctors = bundle.write(root)
    monkeypatch.setattr(RP, "WORK", root)
    monkeypatch.setattr(RP, "BUNDLE", root / "bundle")
    monkeypatch.setattr(RP, "BDVST_CSV", doctors)
    monkeypatch.setattr(RP, "RETURNS_LEDGER_ENABLED", True)
    monkeypatch.setattr(RP, "DECLARED_USETYPE_PILOT_ENABLED", True)
    monkeypatch.setattr(RP, "DECLARED_USE_PREOP_EXEMPT_PILOT_ENABLED", True)
    monkeypatch.setattr(RP, "MSBOS_RESERVATION_PILOT_ENABLED", False)
    RP.main()
    rows = (root / "report.csv").read_text().splitlines()
    return {
        r["audit_id"]: r["classification"]
        for r in csv.DictReader(rows)
        if r["event_seq"]
    }


@pytest.mark.parametrize(
    ("product", "bdtype"), [("LPRC", "LDPRC2"), ("LDPPC", "LDPPC5")], ids=["rbc", "plt"]
)
@pytest.mark.parametrize(
    ("second_issue", "note_hours", "released"),
    [(30.0, 3.0, True), (10.0, 5.0, True), (30.0, 31.0, False)],
    ids=["note-in-event-1-window", "note-on-the-same-day", "note-in-own-window"],
)
def test_both_legs_release_an_all_returned_event_on_the_same_notes_window(
    leg: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    product: str,
    bdtype: str,
    second_issue: float,
    note_hours: float,
    released: bool,
) -> None:
    """Owner decision 2026-11-24, in both legs: only a transfusion charted in
    the all-returned event's own window keeps it from RETURNED_NOT_TRANSFUSED,
    and the LLM leg submits it exactly when the deterministic leg holds it. A
    note charted the same day for the earlier issue sits inside the bundle's
    window but outside the event's, so only the event window releases it."""
    returned = _audit_id("W1", "2", bdtype)
    det = _deterministic_classifications(
        tmp_path,
        monkeypatch,
        _all_returned_second_event(product, bdtype, second_issue, note_hours),
    )
    requests = _prepare(
        leg,
        monkeypatch,
        _all_returned_second_event(product, bdtype, second_issue, note_hours),
    )

    assert (det[returned] == "RETURNED_NOT_TRANSFUSED") is released
    assert (returned not in requests.cases) is released
