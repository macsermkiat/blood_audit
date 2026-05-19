"""Most-recent Hb lookup with freshness tiers and tiered delta-Hb bypass.

Per PRD §3 and Round 2 E3:

* Most-recent Hb at or before the order anchor wins. Source preference:
  ``HEMATOLOGY`` (LABEXM 290095) beats ``POCT`` (LABEXM 500001) when at
  least one HEMATOLOGY observation is in the 7-day lookback. Ties on
  datetime resolve to highest ``item_no`` (the later-inserted /
  corrected-result row in HOSxP semantics).
* Freshness tiers, measured from anchor to chosen-current's datetime:
  <24h ``fresh``, [24h, 72h) ``stale_24_72h``, [72h, 7d) ``stale_3_7d``,
  otherwise ``missing``.
* Tiered delta-Hb trigger (Round 2 E3 clinical thresholds): drop
  ≥ 1.5 g/dL in 6h, ≥ 2.0 g/dL in 12h, ≥ 2.5 g/dL in 24h. The "prior" in
  each window is the *highest* Hb strictly before current — the most
  conservative bleed signal. Any window triggering sets
  ``delta_hb_bypass = True`` for the deterministic classifier.
* ``needs_review_single_low_hb`` fires when the chosen current Hb is
  < 8 g/dL and no prior observation sits in the 24h window before it —
  an isolated low value with no trend to interpret it against.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from bba.hb_lookup.models import (
    DeltaHbWindow,
    HbFreshness,
    HbLookupResult,
    HbObservation,
)

_LOOKBACK = timedelta(days=7)
_FRESH = timedelta(hours=24)
_STALE_24_72 = timedelta(hours=72)
_LOW_HB_REVIEW_THRESHOLD = 8.0

# Window-hours and the corresponding required drop (g/dL) for a trigger.
# Ordered shortest-window-first to keep the public output canonical.
_DELTA_HB_SPECS: tuple[tuple[int, float], ...] = (
    (6, 1.5),
    (12, 2.0),
    (24, 2.5),
)


def lookup_hb(
    *,
    observations: Sequence[HbObservation],
    anchor_utc: datetime,
) -> HbLookupResult:
    """Return the most-recent Hb at or before ``anchor_utc`` with freshness + delta.

    ``observations`` must be the Hb observations for a single patient
    (caller filters by HN). ``anchor_utc`` must be tz-aware UTC.

    Order of ``observations`` is unimportant — the function sorts
    internally. When no observation lies in the 7-day lookback at or
    before ``anchor_utc``, returns ``HbLookupResult`` with
    ``freshness="missing"`` and all value fields ``None``.
    """
    before_anchor = [o for o in observations if o.datetime_utc <= anchor_utc]
    in_lookback = [o for o in before_anchor if anchor_utc - o.datetime_utc < _LOOKBACK]
    if not in_lookback:
        return _missing_result()

    current = _select_current(in_lookback)
    freshness = _freshness_tier(anchor_utc - current.datetime_utc)
    windows = _compute_delta_windows(current=current, all_priors=before_anchor)
    bypass = any(w.triggered for w in windows)
    needs_review = _single_low_hb(current=current, all_priors=before_anchor)

    return HbLookupResult(
        value_g_dl=current.value_g_dl,
        datetime_utc=current.datetime_utc,
        source=current.source,
        freshness=freshness,
        delta_hb_bypass=bypass,
        delta_hb_windows=windows,
        needs_review_single_low_hb=needs_review,
    )


def _select_current(in_lookback: Sequence[HbObservation]) -> HbObservation:
    """Pick the most-recent Hb after applying HEMATOLOGY-preferred source.

    Within the chosen source group, sort by (datetime desc, item_no desc)
    and take the first — that resolves the same-datetime tie to the
    highest ``item_no`` (the later-inserted / amended Lab row).
    """
    hematology = [o for o in in_lookback if o.source == "HEMATOLOGY"]
    pool: Sequence[HbObservation] = hematology if hematology else in_lookback
    return max(pool, key=lambda o: (o.datetime_utc, o.item_no))


def _freshness_tier(age: timedelta) -> HbFreshness:
    if age < _FRESH:
        return "fresh"
    if age < _STALE_24_72:
        return "stale_24_72h"
    if age < _LOOKBACK:
        return "stale_3_7d"
    # Unreachable: the in_lookback filter already excluded >=7d, but keep
    # an explicit branch so the function is total over its annotation.
    return "missing"


def _compute_delta_windows(
    *,
    current: HbObservation,
    all_priors: Sequence[HbObservation],
) -> tuple[DeltaHbWindow, ...]:
    """One ``DeltaHbWindow`` per (6h, 12h, 24h) threshold spec."""
    return tuple(
        _delta_window_for(
            current=current, all_priors=all_priors, hours=hours, threshold=threshold
        )
        for hours, threshold in _DELTA_HB_SPECS
    )


def _delta_window_for(
    *,
    current: HbObservation,
    all_priors: Sequence[HbObservation],
    hours: int,
    threshold: float,
) -> DeltaHbWindow:
    window = timedelta(hours=hours)
    priors_in_window = [
        o
        for o in all_priors
        if o.datetime_utc < current.datetime_utc
        and current.datetime_utc - o.datetime_utc <= window
    ]
    if not priors_in_window:
        return DeltaHbWindow(
            window_hours=hours,
            threshold_g_dl=threshold,
            prior_value_g_dl=None,
            prior_datetime_utc=None,
            drop_g_dl=None,
            triggered=False,
        )
    # The "peak prior" — bigger drop is the more conservative bleed signal.
    # Tie-break (same Hb value) by most-recent datetime, then highest
    # item_no — matches _select_current's convention so the public output
    # is deterministic regardless of input ordering.
    peak = max(
        priors_in_window,
        key=lambda o: (o.value_g_dl, o.datetime_utc, o.item_no),
    )
    drop = peak.value_g_dl - current.value_g_dl
    return DeltaHbWindow(
        window_hours=hours,
        threshold_g_dl=threshold,
        prior_value_g_dl=peak.value_g_dl,
        prior_datetime_utc=peak.datetime_utc,
        drop_g_dl=drop,
        triggered=drop >= threshold,
    )


def _single_low_hb(
    *,
    current: HbObservation,
    all_priors: Sequence[HbObservation],
) -> bool:
    """True iff current is < 8 g/dL and no prior sits in its 24h window."""
    if current.value_g_dl >= _LOW_HB_REVIEW_THRESHOLD:
        return False
    twenty_four_hours = timedelta(hours=24)
    has_prior_in_24h = any(
        o.datetime_utc < current.datetime_utc
        and current.datetime_utc - o.datetime_utc <= twenty_four_hours
        for o in all_priors
    )
    return not has_prior_in_24h


def _missing_result() -> HbLookupResult:
    return HbLookupResult(
        value_g_dl=None,
        datetime_utc=None,
        source=None,
        freshness="missing",
        delta_hb_bypass=False,
        delta_hb_windows=tuple(
            DeltaHbWindow(
                window_hours=hours,
                threshold_g_dl=threshold,
                prior_value_g_dl=None,
                prior_datetime_utc=None,
                drop_g_dl=None,
                triggered=False,
            )
            for hours, threshold in _DELTA_HB_SPECS
        ),
        needs_review_single_low_hb=False,
    )
