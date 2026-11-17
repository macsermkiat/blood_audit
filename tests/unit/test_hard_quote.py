"""Hard-indication quote screens (owner decision 2026-11-17, PR #274).

A grounded HEMODYNAMIC_INSTABILITY or ACS citation now clears an over-clear
with no human review, so the quote itself must document the condition now.
"""

from __future__ import annotations

from datetime import date

import pytest

from bba.audit_pipeline.hard_quote import (
    quote_documents_acs,
    quote_documents_instability,
)

ORDER = date(2026, 5, 16)


@pytest.mark.parametrize(
    "quote",
    [
        "ABP 50/30 on norepinephrine",
        "BP 79/50, MAP 46",
        "septic shock, start Levophed",
        "HR 134, BP 100/60",
        "pulse 128 irregular",
        "ความดันต่ำ ให้ยากระตุ้นความดัน",
        "ช็อก ให้ norepinephrine",
        "hypotension despite 2 L NSS",
    ],
)
def test_instability_documented(quote: str) -> None:
    assert quote_documents_instability(quote, ORDER)


@pytest.mark.parametrize(
    "quote",
    [
        "BP 120/70, no signs of shock",
        "not hypotensive, off pressors",
        "r/o septic shock, lactate 1.1",
        "history of cardiogenic shock in 2019",
        "Hx. 1/5/69: septic shock on Levophed",
        "BP 130/80, HR 96",
        "follow up 30/5/68 clinic",
        "ไม่มีภาวะช็อก",
    ],
)
def test_instability_not_documented(quote: str) -> None:
    assert not quote_documents_instability(quote, ORDER)


@pytest.mark.parametrize(
    "quote",
    [
        "NSTEMI with ongoing chest pain",
        "acute coronary syndrome, troponin rising",
        "ACS: start heparin and DAPT",
        "anterior wall myocardial infarction this morning",
        "สงสัยกล้ามเนื้อหัวใจขาดเลือด EKG ST depression",
    ],
)
def test_acs_documented(quote: str) -> None:
    assert quote_documents_acs(quote, ORDER)


@pytest.mark.parametrize(
    "quote",
    [
        "R/O ACS, troponin negative x2",
        "old MI 2015 s/p PCI, ACS 2019",
        "no ischemic change on ECG",
        "s/p STEMI with stent",
        "history of acute coronary syndrome",
        "pacs report pending",
    ],
)
def test_acs_not_documented(quote: str) -> None:
    assert not quote_documents_acs(quote, ORDER)
