"""Crystalloid totaling helper for the hemodilution bypass.

The deterministic classifier's hemodilution rule (Round 1 B5) requires
knowing how much crystalloid (NSS, LRS, RLS, Plasmalyte, D5W, …) was
administered in the 4 h before the order anchor. The orchestrator
derives that scalar from the MED.csv stream via
:func:`total_crystalloid_liters` and passes it as
``ClassifierInputs.crystalloid_liters_prior_4h``.

This helper is intentionally minimal: a thin sum-and-window utility that
parses a unit-bearing dose string (``"1000 mL"`` / ``"1 L"`` / ``"500 cc"``)
into liters and filters the events to the [order_datetime - window,
order_datetime] interval. Drug-name → "is crystalloid?" classification
lives upstream in the MED-table reader; this helper trusts that the
``med_events`` passed in are already crystalloid-only.

See PRD §"Implementation Decisions §6" for the 4-h window and the 2 L
threshold; both are constants on :mod:`bba.deterministic_classifier.classifier`.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from bba.cohort_detector import MedEvent

CRYSTALLOID_WINDOW_HOURS: float = 4.0
"""PRD §6 hemodilution lookback window (hours): we sum crystalloid
administrations in [order_datetime - 4 h, order_datetime]."""


_DOSE_PATTERN = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(mL|cc|L)\b(?!\s*/)",
    re.IGNORECASE,
)
"""Capture the first unit-bearing numeric dose on a HOSxP drug string.

The grouping is intentionally permissive on whitespace and case so
``"NSS 1000 mL"``, ``"RLS 1 L"``, ``"D5W 500 cc"``, ``"NSS1000ML"`` all
parse to the same numeric value. Thousands separators are tolerated
(``"NSS 1,000 mL"`` -> 1.0 L); the comma is stripped before ``float``.
Without ``[\\d,]`` in the group, ``"1,000 mL"`` would match only the
``"000 mL"`` tail and silently parse as 0 L, undercounting the
hemodilution total.

First-match (not last) is deliberate: in an additive string such as
``"D5W 1000 mL + KCl 20 mL"`` the first unit-bearing number is the
crystalloid volume, so taking the last match would wrongly sum the
20 mL additive instead.

Infusion-rate strings (``"NSS 500 mL/h"``, ``"D5W 200 cc/hour"``,
``"RLS 1 L/hr"``) are excluded by the negative lookahead ``(?!\\s*/)``:
``\\b`` alone matches at the L→/ boundary because ``/`` is non-alphanumeric,
which would incorrectly count a rate as a delivered bolus (PR #52 Codex P2).
The lookahead is whitespace-tolerant so ``"500 mL / h"`` is also rejected.
"""


def total_crystalloid_liters(
    med_events: tuple[MedEvent, ...],
    order_datetime: datetime,
    window_hours: float = CRYSTALLOID_WINDOW_HOURS,
) -> float:
    """Return total crystalloid volume (liters) administered in the window.

    Inputs:

    * ``med_events`` — crystalloid-only MED rows for this patient. Each
      :class:`bba.cohort_detector.MedEvent` carries a free-text ``drug``
      string from HOSxP whose tail contains the dose (e.g.,
      ``"NSS 1000 mL"``, ``"RLS 1 L"``, ``"D5W 500 cc"``). The parser
      accepts mL / cc / L (case-insensitive) and returns 0.0 for rows
      whose dose cannot be parsed.

    * ``order_datetime`` — tz-aware UTC order anchor. Events strictly
      AFTER ``order_datetime`` are excluded. Events at-or-before are
      included iff they fall inside the window.

    * ``window_hours`` — defaults to :data:`CRYSTALLOID_WINDOW_HOURS`
      (4 h). Exposed for testing only.

    Returns a non-negative float in liters. The deterministic classifier
    compares this scalar against
    :data:`bba.deterministic_classifier.HEMODILUTION_CRYSTALLOID_LITERS`
    (2.0 L) to decide whether the hemodilution bypass fires.
    """
    window_start = order_datetime - timedelta(hours=window_hours)
    total = 0.0
    for event in med_events:
        if event.timestamp > order_datetime:
            continue
        if event.timestamp < window_start:
            continue
        match = _DOSE_PATTERN.search(event.drug)
        if match is None:
            continue
        value = float(match.group(1).replace(",", ""))
        unit = match.group(2).lower()
        if unit in ("ml", "cc"):
            total += value / 1000.0
        else:
            total += value
    return total


__all__ = ("CRYSTALLOID_WINDOW_HOURS", "total_crystalloid_liters")
