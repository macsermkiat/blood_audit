"""Most-recent Hb lookup with freshness tiers and tiered delta-Hb bypass.

Per PRD §3 and Round 2 E3:

* Most-recent Hb before the order anchor wins. Source preference:
  ``HEMATOLOGY`` (LABEXM 290095) beats ``POCT`` (LABEXM 500001) when both
  have observations in the 7-day lookback. Ties on datetime + source resolve
  to highest ``item_no``.
* Freshness tiers measured from anchor: <24h ``fresh``, [24h, 72h)
  ``stale_24_72h``, [72h, 7d) ``stale_3_7d``, otherwise ``missing``.
* Tiered delta-Hb trigger (Round 2 E3, clinical thresholds):
  drop ≥ 1.5 g/dL in 6h, ≥ 2.0 g/dL in 12h, ≥ 2.5 g/dL in 24h. Any window
  triggering sets ``delta_hb_bypass = True`` for the deterministic
  classifier.
* ``needs_review_single_low_hb`` fires when the current Hb is < 8 g/dL and
  no prior observation sits in the 24h window before it (no trend to
  interpret an isolated low value).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from bba.hb_lookup.models import HbLookupResult, HbObservation


def lookup_hb(
    *,
    observations: Sequence[HbObservation],
    anchor_utc: datetime,
) -> HbLookupResult:
    """Return the most-recent Hb before ``anchor_utc`` with freshness + delta.

    ``observations`` must be the Hb observations for a single patient
    (caller filters by HN). ``anchor_utc`` must be tz-aware UTC.

    Order is unimportant — the function sorts internally. When no
    observation lies in the 7-day lookback before ``anchor_utc``, returns
    ``HbLookupResult`` with ``freshness="missing"`` and all value fields
    ``None``.
    """

    raise NotImplementedError("lookup_hb: implement in GREEN phase")
