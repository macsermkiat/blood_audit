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

from datetime import datetime

from bba.cohort_detector import MedEvent

CRYSTALLOID_WINDOW_HOURS: float = 4.0
"""PRD §6 hemodilution lookback window (hours): we sum crystalloid
administrations in [order_datetime - 4 h, order_datetime]."""


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
    raise NotImplementedError


__all__ = ("CRYSTALLOID_WINDOW_HOURS", "total_crystalloid_liters")
