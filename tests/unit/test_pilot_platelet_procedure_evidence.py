"""Platelet orders see the procedure charges the RBC path already sees.

Clinician review 2026-11-09: "CVC is a procedure". Bedside CVC / DLC
insertions are rarely coded as procedures but are billed (INCPT group 110,
e.g. "การใส่ Double lumen catheter (CVP)"). In 2025, 240 of 3,696 eligible
platelet orders had such a charge within 24 h, yet the platelet LLM leg never
received INCPT evidence, so the CVC indication could only be found in notes.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"


def _leg():
    if str(PILOT_DIR) not in sys.path:
        sys.path.insert(0, str(PILOT_DIR))
    spec = importlib.util.spec_from_file_location(
        "pilot_run_llm_leg_plt_procedures", PILOT_DIR / "run_llm_leg.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ORDER = datetime(2025, 8, 17, 15, 0, tzinfo=timezone.utc)  # 22:00 Bangkok
DISPENSE = ORDER + timedelta(hours=1)  # 23:00 Bangkok
WINDOW = (
    datetime(2025, 8, 15, 17, 0, tzinfo=timezone.utc),
    datetime(2025, 8, 17, 17, 0, tzinfo=timezone.utc),
)


def _charge(date: str, time: str, *, cancelled: str = "") -> dict[str, str]:
    return {
        "AN": "AN1",
        "INCDATE": date,
        "INCTIME": time,
        "INCGRP": "110",
        "INCGRP__NAME": "ค่าทำหัตถการ และวิสัญญี",
        "INCOME": "X1",
        "ORDERCODE": "",
        "O__OPRTACT": "",
        "O__ICD9CM": "",
        "O__NAME_EN": "Venous catheterization,not elsewhere classified",
        "O__NAME": "การใส่ Double lumen catheter,ครั้ง",
        "CANCELDATE": cancelled,
    }


def test_billed_dlc_insertion_reaches_platelet_evidence() -> None:
    leg = _leg()
    chunks = leg._platelet_procedure_chunks(
        [_charge("2025-08-17", "200000")],
        {},
        an="AN1",
        order_utc=ORDER,
        dispense_utc=DISPENSE,
        window_start=WINDOW[0],
        notes_window_end=WINDOW[1],
    )
    assert len(chunks) == 1
    assert chunks[0].source == "INCPT"
    # Real 2025 rows carry no ICD-9 code; the English name is generic and the
    # Thai name is the one that says "Double lumen catheter", so both show.
    assert "Venous catheterization" in chunks[0].text
    assert "Double lumen catheter" in chunks[0].text
    assert "-2.0h vs order" in chunks[0].text


def test_cancelled_and_out_of_window_charges_stay_out() -> None:
    leg = _leg()
    chunks = leg._platelet_procedure_chunks(
        [
            _charge("2025-08-17", "200000", cancelled="2025-08-17"),
            _charge("2025-08-14", "100000"),
        ],
        {},
        an="AN1",
        order_utc=ORDER,
        dispense_utc=DISPENSE,
        window_start=WINDOW[0],
        notes_window_end=WINDOW[1],
    )
    assert chunks == ()


def test_insertion_billed_the_day_after_the_order_is_shown() -> None:
    # The insertion usually follows the platelets: in 2025, 98 of the 240
    # billed CVC / DLC insertions fell after midnight of the order day, all
    # within 24 h of dispense (e.g. 68058967, billed 20 h after the order).
    # The notes window ends at that midnight; the procedure window must not.
    leg = _leg()
    chunks = leg._platelet_procedure_chunks(
        [_charge("2025-08-18", "180000")],  # 18:00 Bangkok next day
        {},
        an="AN1",
        order_utc=ORDER,
        dispense_utc=DISPENSE,
        window_start=WINDOW[0],
        notes_window_end=WINDOW[1],
    )
    assert len(chunks) == 1
    assert "+20.0h vs order" in chunks[0].text


def test_insertion_more_than_24h_after_dispense_stays_out() -> None:
    leg = _leg()
    chunks = leg._platelet_procedure_chunks(
        [_charge("2025-08-18", "233000")],  # dispense + 24.5 h
        {},
        an="AN1",
        order_utc=ORDER,
        dispense_utc=DISPENSE,
        window_start=WINDOW[0],
        notes_window_end=WINDOW[1],
    )
    assert chunks == ()
