"""Platelet lab config, value parsing, and the validated observation model.

Phase 2 counterpart to :mod:`bba.hb_lookup`'s value layer. The recent-value
selection engine is shared and extracted in a follow-up (docs plan §5.3
stage 2); this package currently exposes the config + parser + observation.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.platelet_lookup.lookup import lookup_platelet
from bba.platelet_lookup.models import (
    PLATELET_LABEXM,
    PLATELET_UNIT,
    PlateletFreshness,
    PlateletLookupResult,
    PlateletObservation,
    PlateletSource,
)
from bba.platelet_lookup.parse import (
    MAX_PLATELET,
    MIN_PLATELET,
    parse_platelet_count,
)

__all__: Sequence[str] = (
    "MAX_PLATELET",
    "MIN_PLATELET",
    "PLATELET_LABEXM",
    "PLATELET_UNIT",
    "PlateletFreshness",
    "PlateletLookupResult",
    "PlateletObservation",
    "PlateletSource",
    "lookup_platelet",
    "parse_platelet_count",
)
