"""Does a hard-indication quote document the condition it is labelled with?

WHY: since the owner decision of 2026-11-17 a grounded, high-confidence
HEMODYNAMIC_INSTABILITY or ACS citation lets the model's APPROPRIATE stand
with no human review (:func:`bba.audit_pipeline.replay._grounded_hard_indication`).
Grounding proves the quote exists in the notes, not that it says what the
label claims: "BP 120/70, no signs of shock", "R/O ACS, troponin negative"
and "old MI 2015 s/p PCI, ACS 2019" all ground. These checks require
positive, current evidence in the quote itself, using the same negator
windows and stale-date masking as the bleeding exemption
(:mod:`bba.audit_pipeline.bleeding`). Fail-closed: a quote that does not
pass is not trusted, and the over-clear is asserted as before.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Final

from bba.audit_pipeline.bleeding import (
    marker_occurrence_negated,
    mask_stale_dated_spans,
)

_INSTABILITY_TERMS: Final[tuple[str, ...]] = (
    "shock",
    "hypotension",
    "hypotensive",
    "vasopressor",
    "inotrope",
    "norepinephrine",
    "norepi",
    "levophed",
    "adrenaline",
    "epinephrine",
    "dopamine",
    "vasopressin",
    "ช็อก",  # shock
    "ช็อค",  # shock (common spelling)
    "ความดันต่ำ",  # low blood pressure
    "ความดันโลหิตต่ำ",  # low blood pressure
)

_ACS_TERMS: Final[tuple[str, ...]] = (
    "acute coronary",
    "stemi",  # also matches NSTEMI
    "unstable angina",
    "myocardial infarction",
    "ischemi",
    "ischaemi",
    "หัวใจขาดเลือด",  # myocardial ischemia
)
_ACS_ABBREVIATION_RE: Final = re.compile(r"(?<![a-z])(?:acs|ami)(?![a-z])")

# Prompt definition of HEMODYNAMIC_INSTABILITY: SBP < 90 or HR > 120.
# MAP < 65 is the same hypotension stated as a mean pressure.
_SBP_LIMIT: Final = 90
_HR_LIMIT: Final = 120
_MAP_LIMIT: Final = 65
_BP_RE: Final = re.compile(r"(?<![\d/.])(\d{2,3})\s*/\s*(\d{2,3})(?![\d/])")
_HR_RE: Final = re.compile(r"(?<![a-z])(?:hr|pr|pulse)\s*[:=]?\s*(\d{2,3})(?!\d)")
_MAP_RE: Final = re.compile(r"(?<![a-z])map\s*[:=]?\s*(\d{2,3})(?!\d)")
_PLAUSIBLE_SBP: Final = range(30, 251)
_PLAUSIBLE_HR: Final = range(20, 251)

# History markers that make an otherwise un-negated term a past event
# ("old MI", "s/p NSTEMI", "prior shock"). Scanned in the same clause before
# the term; "history of" / "hx of" are already bleeding.py negators.
_HISTORY_TOKENS: Final[tuple[str, ...]] = (
    "old ",
    "prior ",
    "previous ",
    "past ",
    "s/p",
    "hx",
    "known ",
    "เคย",  # "has had before"
)
_HISTORY_WINDOW_CHARS: Final = 30
_CLAUSE_BOUNDARIES: Final[tuple[str, ...]] = (";", ".", ",", "\n")
_YEAR_RE: Final = re.compile(r"(?<!\d)(19\d{2}|20\d{2}|24\d{2}|25\d{2}|26\d{2})(?!\d)")
_BE_CE_OFFSET_YEARS: Final = 543


def _clause_after(lowered: str, end: int) -> str:
    post = lowered[end : end + _HISTORY_WINDOW_CHARS]
    cut = min(
        (i for b in _CLAUSE_BOUNDARIES if (i := post.find(b)) != -1),
        default=len(post),
    )
    return post[:cut]


def _clause_before(lowered: str, start: int) -> str:
    pre = lowered[max(0, start - _HISTORY_WINDOW_CHARS) : start]
    cut = max((pre.rfind(b) for b in _CLAUSE_BOUNDARIES), default=-1)
    return pre[cut + 1 :]


def _dated_before_order_year(clause: str, order_date: date) -> bool:
    for match in _YEAR_RE.finditer(clause):
        year = int(match.group(1))
        ce_year = year - _BE_CE_OFFSET_YEARS if year >= 2400 else year
        if ce_year < order_date.year:
            return True
    return False


def _current_occurrence(lowered: str, start: int, end: int, order_date: date) -> bool:
    if marker_occurrence_negated(lowered, start, end):
        return False
    before = _clause_before(lowered, start)
    if any(token in before for token in _HISTORY_TOKENS):
        return False
    return not _dated_before_order_year(
        before + _clause_after(lowered, end), order_date
    )


def _has_current_term(lowered: str, terms: tuple[str, ...], order_date: date) -> bool:
    for term in terms:
        start = 0
        while (idx := lowered.find(term, start)) != -1:
            if _current_occurrence(lowered, idx, idx + len(term), order_date):
                return True
            start = idx + len(term)
    return False


def _unstable_vital(lowered: str) -> bool:
    for match in _BP_RE.finditer(lowered):
        sbp, dbp = int(match.group(1)), int(match.group(2))
        if sbp in _PLAUSIBLE_SBP and dbp < sbp and sbp < _SBP_LIMIT:
            return True
    for match in _HR_RE.finditer(lowered):
        hr = int(match.group(1))
        if hr in _PLAUSIBLE_HR and hr > _HR_LIMIT:
            return True
    return any(int(m.group(1)) < _MAP_LIMIT for m in _MAP_RE.finditer(lowered))


def quote_documents_instability(quote: str, order_date: date) -> bool:
    """True iff the quote, after stale-dated spans are blanked, shows an
    unstable vital sign (SBP < 90, HR > 120, MAP < 65) or a current,
    un-negated shock / hypotension / vasopressor term."""
    lowered = mask_stale_dated_spans(quote, order_date).lower()
    return _unstable_vital(lowered) or _has_current_term(
        lowered, _INSTABILITY_TERMS, order_date
    )


def quote_documents_acs(quote: str, order_date: date) -> bool:
    """True iff the quote, after stale-dated spans are blanked, names a
    current, un-negated acute coronary syndrome or active ischemia."""
    lowered = mask_stale_dated_spans(quote, order_date).lower()
    if _has_current_term(lowered, _ACS_TERMS, order_date):
        return True
    return any(
        _current_occurrence(lowered, m.start(), m.end(), order_date)
        for m in _ACS_ABBREVIATION_RE.finditer(lowered)
    )


__all__ = ("quote_documents_acs", "quote_documents_instability")
