from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType


def _load_run_pipeline() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "scripts" / "pilot" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("pilot_run_pipeline", path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ipddchsumoprt_rows_feed_operation_events() -> None:
    mod = _load_run_pipeline()

    events = mod._build_op_events(
        [],
        [
            {
                "AN": "PHI_case",
                "ICD9CM": "0159",
                "ORFLAG": "1",
                "INDATE": "2025-11-20 00:00:00.000",
                "INTIME": "090000",
                "OPRTTEXT": "Awake craniotomy and temporal lesionectomy",
            }
        ],
        [],
        {},
        {"0159": {"NAME": "Other brain operation", "ORFLAG": ""}},
        "PHI_case",
    )

    assert len(events) == 1
    assert events[0].icd9 == "0159"
    assert events[0].or_flag is True
    assert events[0].operative_datetime == datetime(2025, 11, 20, 2, tzinfo=UTC)
    assert events[0].name == "Awake craniotomy and temporal lesionectomy"


def test_ipddchsumoprt_rows_ignore_other_admissions() -> None:
    mod = _load_run_pipeline()

    events = mod._build_op_events(
        [],
        [
            {
                "AN": "PHI_other",
                "ICD9CM": "0159",
                "INDATE": "2025-11-20 00:00:00.000",
                "INTIME": "090000",
            }
        ],
        [],
        {},
        {},
        "PHI_case",
    )

    assert events == ()


def test_incpt_uses_optract_icd9cm_bridge_when_available() -> None:
    mod = _load_run_pipeline()

    events = mod._build_op_events(
        [],
        [],
        [
            {
                "AN": "PHI_case",
                "INCGRP": "111",
                "INCOME": "P1141",
                "INCDATE": "2025-11-20",
                "INCTIME": "090000",
                "CANCELDATE": "",
            }
        ],
        {
            "P1141": {
                "ICD9CM": "7863",
                "NAME EN": "Removal of internal fixation device, radius and ulna",
            }
        },
        {"7863": {"NAME": "Removal of implanted device", "ORFLAG": "1"}},
        "PHI_case",
    )

    assert len(events) == 1
    assert events[0].icd9 == "7863"
    assert events[0].or_flag is True
    assert events[0].operative_datetime == datetime(2025, 11, 20, 2, tzinfo=UTC)
    assert events[0].name == "Removal of internal fixation device, radius and ulna"


def test_joined_incpt_optract_row_uses_row_level_icd9cm() -> None:
    mod = _load_run_pipeline()

    events = mod._build_op_events(
        [],
        [],
        [
            {
                "AN": "PHI_case",
                "INCGRP": "111",
                "INCOME": "P1141",
                "O__OPRTACT": "P1141",
                "O__ICD9CM": "7863",
                "O__NAME EN": "Removal of internal fixation device, radius and ulna",
                "INCDATE": "2025-11-20 00:00:00.000",
                "INCTIME": "090000",
                "CANCELDATE": "",
            }
        ],
        {},
        {"7863": {"NAME": "Removal of implanted device", "ORFLAG": "1"}},
        "PHI_case",
    )

    assert len(events) == 1
    assert events[0].icd9 == "7863"
    assert events[0].or_flag is True
    assert events[0].operative_datetime == datetime(2025, 11, 20, 2, tzinfo=UTC)
    assert events[0].name == "Removal of internal fixation device, radius and ulna"


def test_incpt_without_optract_mapping_keeps_timing_fallback() -> None:
    mod = _load_run_pipeline()

    events = mod._build_op_events(
        [],
        [],
        [
            {
                "AN": "PHI_case",
                "INCGRP": "110",
                "INCOME": "AS061",
                "INCDATE": "2025-11-20",
                "INCTIME": "090000",
                "CANCELDATE": "",
            }
        ],
        {},
        {},
        "PHI_case",
    )

    assert len(events) == 1
    assert events[0].icd9 == "INCPT:AS061"
    assert events[0].or_flag is False


def test_transfusion_issue_time_zero_pads_hosxp_time() -> None:
    mod = _load_run_pipeline()

    assert mod._fmt_hosxp_time("80000") == "08:00:00"
    assert mod._fmt_hosxp_time("123456") == "12:34:56"
