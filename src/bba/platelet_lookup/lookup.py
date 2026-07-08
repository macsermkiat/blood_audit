"""Most-recent platelet count lookup with freshness tiers.

Platelet counterpart to :func:`bba.hb_lookup.lookup_hb`, deliberately simpler:
the §5.1 gate is a plain count-vs-ceiling rule, so there is no delta bypass and
no single-low-no-trend flag. The selection contract matches Hb's:

* Most-recent count at or before the anchor wins, within a 7-day lookback.
* Source is HEMATOLOGY-only (LABEXM 290078) today; ties on datetime resolve to
  the highest ``item_no`` (the later-inserted / corrected Lab row).
* Freshness measured from anchor to the chosen count's datetime.

Not a shared-engine extraction from ``hb_lookup`` (docs plan §5.3 stage 2
proposed that): a focused mirror avoids refactoring the RBC-critical Hb path.
The freshness helper is small enough that duplication is cheaper than the
coupling risk; a later refactor can hoist it if a third component appears.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta

from bba.platelet_lookup.models import (
    PlateletFreshness,
    PlateletLookupResult,
    PlateletObservation,
)

_LOOKBACK = timedelta(days=7)
_FRESH = timedelta(hours=24)
_STALE_24_72 = timedelta(hours=72)


def lookup_platelet(
    *,
    observations: Sequence[PlateletObservation],
    anchor_utc: datetime,
) -> PlateletLookupResult:
    """Return the most-recent platelet count at or before ``anchor_utc``.

    ``observations`` must be for a single patient (caller filters by HN).
    ``anchor_utc`` must be tz-aware UTC. Order is unimportant — sorted
    internally. When no count lies in the 7-day lookback at or before the
    anchor, returns ``freshness="missing"`` with all value fields ``None``.
    """
    in_lookback = [
        o
        for o in observations
        if o.datetime_utc <= anchor_utc and anchor_utc - o.datetime_utc < _LOOKBACK
    ]
    if not in_lookback:
        return PlateletLookupResult(
            value_k_ul=None,
            datetime_utc=None,
            source=None,
            freshness="missing",
        )

    current = max(in_lookback, key=lambda o: (o.datetime_utc, o.item_no))
    return PlateletLookupResult(
        value_k_ul=current.value_k_ul,
        datetime_utc=current.datetime_utc,
        source=current.source,
        freshness=_freshness_tier(anchor_utc - current.datetime_utc),
    )


def _freshness_tier(age: timedelta) -> PlateletFreshness:
    if age < _FRESH:
        return "fresh"
    if age < _STALE_24_72:
        return "stale_24_72h"
    if age < _LOOKBACK:
        return "stale_3_7d"
    # Unreachable: the lookback filter already excluded >=7d, but keep an
    # explicit branch so the function is total over its annotation.
    return "missing"


__all__ = ("lookup_platelet",)
