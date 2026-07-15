"""Unit coverage for the declared-USETYPE go-live preflight."""

from __future__ import annotations

import csv
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import pytest

from bba import feature_flags
from bba.cohort_detector import CohortAssignment, CohortLabel, OperativeEvent
from bba.deterministic_classifier import classify
from bba.deterministic_classifier.models import (
    BypassReason,
    ClassifierInputs,
    ClassifierResult,
)
from bba.hb_lookup import DeltaHbWindow, HbLookupResult

ORDER_DT = datetime(2026, 1, 10, 8, tzinfo=timezone.utc)


def _load_preflight() -> ModuleType:
    pilot_dir = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
    if str(pilot_dir) not in sys.path:
        sys.path.insert(0, str(pilot_dir))
    path = pilot_dir / "preflight_declared_usetype.py"
    spec = importlib.util.spec_from_file_location(
        "pilot_preflight_declared_usetype", path
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _delta_windows(triggered: bool) -> tuple[DeltaHbWindow, ...]:
    return tuple(
        DeltaHbWindow(
            window_hours=hours,
            threshold_g_dl=threshold,
            prior_value_g_dl=10.0 if triggered and hours == 6 else None,
            prior_datetime_utc=(
                ORDER_DT - timedelta(hours=5) if triggered and hours == 6 else None
            ),
            drop_g_dl=2.0 if triggered and hours == 6 else None,
            triggered=triggered and hours == 6,
        )
        for hours, threshold in ((6, 2.0), (12, 2.5), (24, 3.0))
    )


def _inputs(hb_value: float, *, delta_bypass: bool = False) -> ClassifierInputs:
    return ClassifierInputs(
        audit_id="audit-preflight-test",
        hb_result=HbLookupResult(
            value_g_dl=hb_value,
            datetime_utc=ORDER_DT - timedelta(hours=2),
            source="HEMATOLOGY",
            freshness="fresh",
            delta_hb_bypass=delta_bypass,
            delta_hb_windows=_delta_windows(delta_bypass),
            needs_review_single_low_hb=False,
        ),
        cohort_assignment=CohortAssignment(
            label=CohortLabel.DEFAULT,
            threshold=7.0,
            evidence_code=None,
            evidence_name=None,
        ),
        order_datetime=ORDER_DT,
        procedure_proximity_hours=None,
        upcoming_procedure_hours=None,
        crystalloid_liters_prior_4h=0.0,
        declared_use=None,
    )


@pytest.mark.parametrize(
    ("hb_value", "delta_bypass", "off_rationale", "bucket"),
    [
        (8.0, False, "hb_7_to_10", "bucket_rationale_rename"),
        (11.0, False, "hb_ge_10", "bucket_highhb_to_defer"),
        (8.0, True, "bypass_delta_hb", "bucket_deltahb_preempt"),
    ],
)
def test_expected_flip_buckets_use_real_classifier_transitions(
    hb_value: float,
    delta_bypass: bool,
    off_rationale: str,
    bucket: str,
) -> None:
    mod = _load_preflight()
    inputs = _inputs(hb_value, delta_bypass=delta_bypass)
    res_off = classify(inputs)
    res_on = classify(inputs.model_copy(update={"declared_use": "surgery"}))

    finding = mod.bucket_flip(
        reqno="REQ-1",
        hn="HN-1",
        label="surgery",
        structured_upcoming=False,
        res_off=res_off,
        res_on=res_on,
    )

    assert res_off.rationale == off_rationale
    assert (res_on.classification, res_on.rationale) == (
        "NEEDS_REVIEW",
        "preop_defer_llm_declared",
    )
    assert finding is not None
    assert finding.bucket == bucket
    assert finding.unexpected_reason is None


def test_unexpected_flip_is_recorded_and_forces_hold() -> None:
    mod = _load_preflight()
    off = ClassifierResult(
        classification="APPROPRIATE",
        bypass_reason=BypassReason.NONE,
        cohort_threshold=7.0,
        rationale="hb_lt_threshold",
    )
    on = ClassifierResult(
        classification="NEEDS_REVIEW",
        bypass_reason=BypassReason.NONE,
        cohort_threshold=7.0,
        rationale="preop_defer_llm_declared",
    )

    finding = mod.bucket_flip(
        reqno="REQ-BAD",
        hn="HN-1",
        label="surgery",
        structured_upcoming=False,
        res_off=off,
        res_on=on,
    )

    assert finding is not None
    assert finding.bucket == "unexpected"
    assert finding.unexpected_reason
    text = mod.recommendation(
        audited_orders=1,
        declared_surgical_orders=1,
        mixed_count=0,
        unexpected_flip_count=1,
    )
    assert text.startswith("HOLD")


@pytest.mark.parametrize(
    ("values", "expected"),
    [(["2", "3"], 1), (["2", "2"], 0), (["2", ""], 0)],
)
def test_mixed_detection(values: list[str], expected: int) -> None:
    mod = _load_preflight()
    summary = mod.summarize_usetype(
        audited_keys=[("HN", "REQ")],
        values_by_hn_reqno={("HN", "REQ"): values},
    )
    assert len(summary.mixed_orders) == expected


def test_cross_hn_collision_reports_only_multi_hn_reqnos() -> None:
    mod = _load_preflight()
    collisions = mod.cross_hn_collisions(
        [
            {"HN": "AAA", "REQNO": "R1"},
            {"HN": "BBB", "REQNO": "R1"},
            {"HN": "AAA", "REQNO": "R2"},
            {"HN": "", "REQNO": "R2"},
        ]
    )
    assert [(item.reqno, item.hns) for item in collisions] == [("R1", ("AAA", "BBB"))]


def test_incremental_signal_counts_surgical_order_without_blood_requiring_op() -> None:
    mod = _load_preflight()
    blood_requiring = OperativeEvent(
        icd9="8154",
        or_flag=True,
        operative_datetime=ORDER_DT + timedelta(hours=2),
        name="Total knee replacement",
    )
    orders = [
        mod.OrderDeclaredUse("HN-1", "R-NO-OP", "AN-1", "surgery"),
        mod.OrderDeclaredUse("HN-2", "R-OP", "AN-2", "type_screen"),
        mod.OrderDeclaredUse("HN-3", "R-WARD", "AN-3", "ward"),
    ]

    reqnos = mod.incremental_signal_reqnos(
        orders,
        {"AN-1": (), "AN-2": (blood_requiring,), "AN-3": ()},
    )

    assert reqnos == ["R-NO-OP"]


def test_distribution_includes_unknown_and_none_and_raw_codes() -> None:
    mod = _load_preflight()
    keys = [("H1", "R1"), ("H2", "R2"), ("H3", "R3"), ("H4", "R4")]
    summary = mod.summarize_usetype(
        audited_keys=keys,
        values_by_hn_reqno={
            keys[0]: ["2"],
            keys[1]: ["5"],
            keys[2]: [""],
            keys[3]: ["2", "2"],
        },
    )
    assert summary.distribution == {"none": 1, "surgery": 2, "unknown": 1}
    assert summary.raw_code_frequency == {"2": 3, "5": 1}


def test_recommendation_names_remaining_requirements_and_flag_stays_off() -> None:
    mod = _load_preflight()
    before = feature_flags.DECLARED_USETYPE_ENABLED

    inputs = _inputs(11.0)
    mod.bucket_flip(
        reqno="REQ-1",
        hn="HN-1",
        label="surgery",
        structured_upcoming=False,
        res_off=classify(inputs),
        res_on=classify(inputs.model_copy(update={"declared_use": "surgery"})),
    )
    text = mod.recommendation(
        audited_orders=1,
        declared_surgical_orders=1,
        mixed_count=0,
        unexpected_flip_count=0,
    )

    assert text.startswith("GO")
    assert "flag-on LLM-leg comparison" in text
    assert "attribution/dashboard delta" in text
    assert "clinician sign-off" in text
    assert "hb_ge_10" in text
    assert "bypass_delta_hb" in text
    assert feature_flags.DECLARED_USETYPE_ENABLED is before  # read-only: unchanged


def test_run_preflight_uses_real_order_inputs_without_mutating_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_preflight()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_csv(
        bundle / "BDVST.csv",
        [
            {
                "HN": "HN-1",
                "AN": "AN-1",
                "REQNO": "REQ-1",
                "BDVSTST": "4",
                "REQTYPE": "P",
                "CANCELDATE": "",
                "REQDATE": "2026-01-01",
                "REQTIME": "080000",
                "BDVSTDATE": "2026-01-01",
                "BDVSTTIME": "080000",
                "PICKDATE": "",
                "PICKTIME": "",
                "ICD10": "",
            }
        ],
        [
            "HN",
            "AN",
            "REQNO",
            "BDVSTST",
            "REQTYPE",
            "CANCELDATE",
            "REQDATE",
            "REQTIME",
            "BDVSTDATE",
            "BDVSTTIME",
            "PICKDATE",
            "PICKTIME",
            "ICD10",
        ],
    )
    _write_csv(
        bundle / "BDVSTDT.csv",
        [
            {
                "HN": "HN-1",
                "REQNO": "REQ-1",
                "BDTYPE": "LPRC",
                "UNITAMT": "1",
                "USETYPE": "2",
                "BDVSTDATE": "2026-01-01",
                "BDVSTTIME": "080000",
                "USEDATE": "2026-01-01",
                "USETIME": "090000",
            }
        ],
        [
            "HN",
            "REQNO",
            "BDTYPE",
            "UNITAMT",
            "USETYPE",
            "BDVSTDATE",
            "BDVSTTIME",
            "USEDATE",
            "USETIME",
        ],
    )
    _write_csv(bundle / "Diagnosis.csv", [], ["AN", "ICD10"])
    _write_csv(
        bundle / "Lab.csv",
        [
            {
                "AN": "AN-1",
                "LABEXM": "290095",
                "RESULT": "11.0",
                "LVSTDATE": "2026-01-01",
                "LVSTTIME": "070000",
            }
        ],
        ["AN", "LABEXM", "RESULT", "LVSTDATE", "LVSTTIME"],
    )
    _write_csv(bundle / "Med.csv", [], ["AN", "PRSCDATE", "PRSCTIME"])
    _write_csv(
        bundle / "IPTSUMOPRT.csv",
        [],
        ["AN", "ICD9CM", "INDATE", "INTIME", "OPRTTEXT", "ORFLAG"],
    )
    _write_csv(bundle / "ICD9CM.csv", [], ["Icd9cm", "Name", "Orflag"])
    monkeypatch.setattr(mod, "BUNDLE", bundle)
    monkeypatch.setattr(mod, "WORK", tmp_path)
    before = feature_flags.DECLARED_USETYPE_ENABLED

    result = mod.run_preflight()

    assert result.orders_audited == 1
    assert result.flip_bucket_counts == {"bucket_highhb_to_defer": 1}
    assert result.incremental_signal_reqnos == ["REQ-1"]
    assert result.recommendation.startswith("GO")
    assert feature_flags.DECLARED_USETYPE_ENABLED is before  # read-only: unchanged
