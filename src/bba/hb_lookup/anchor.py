"""Shared Hb-anchor resolution policy.

The deterministic report (``scripts/pilot/run_pipeline.py``) and the LLM
gate (``scripts/pilot/run_llm_leg.py``) must resolve the Hb anchor
identically. When they did not, a case could classify one way in the
report and a different way at the gate — the divergence bug documented in
``docs/handoff-hb-anchor-unification.md`` (case 7 / REQNO 68066907, where
the gate anchored on the order REQTIME only, found no Hb, and silently
dropped the case from LLM adjudication). This module is the single
canonical resolver so the two paths cannot drift again.

Pure and unit-testable: no HOSxP-row or bundle knowledge. Callers build
the ordered ``AnchorCandidate`` list (see
``scripts/pilot/_anchor_candidates.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from bba.hb_lookup.lookup import lookup_hb
from bba.hb_lookup.models import HbLookupResult, HbObservation


@dataclass(frozen=True)
class AnchorCandidate:
    """A fallback Hb anchor with display text + provenance for the report.

    ``anchor_utc`` is the tz-aware UTC datetime to look the Hb back from.
    ``display`` is the local-time string surfaced in the review page, and
    ``reason`` records which fallback fired (e.g. ``"issue_datetime"`` or
    ``"blood_bank_visit_fallback"``).
    """

    anchor_utc: datetime
    display: str
    reason: str


def resolve_hb_with_fallback(
    *,
    observations: Sequence[HbObservation],
    order_datetime: datetime,
    candidates: Sequence[AnchorCandidate],
) -> tuple[HbLookupResult, str, str]:
    """Resolve the most-recent Hb, falling back through ``candidates``.

    The primary anchor is always ``order_datetime``. On a miss, each
    candidate is tried in the order given; a candidate strictly *before*
    the order is skipped — a fallback anchor may be slightly after the
    order (labs drawn minutes post-REQTIME) but never before it. The first
    candidate that yields a non-missing Hb wins. If none do, the original
    order-time (missing) result is returned.

    Returns ``(hb_result, anchor_display, anchor_reason)``. For an
    order-time hit (and for all-miss) ``anchor_display`` is ``""`` and
    ``anchor_reason`` is ``"order_datetime"``.
    """
    primary = lookup_hb(observations=observations, anchor_utc=order_datetime)
    if primary.value_g_dl is not None:
        return primary, "", "order_datetime"

    for candidate in candidates:
        if candidate.anchor_utc < order_datetime:
            continue
        fallback = lookup_hb(observations=observations, anchor_utc=candidate.anchor_utc)
        if fallback.value_g_dl is not None:
            return fallback, candidate.display, candidate.reason

    return primary, "", "order_datetime"
