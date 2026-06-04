"""bba.hb_lookup — most-recent Hb before an order anchor.

See issue #5 for acceptance criteria. Implementation Decisions §3 in the PRD
defines the freshness tiers, source preference, numeric-range validation,
multi-Hb tie-breaking, and the tiered delta-Hb bypass contract.

The module feeds :mod:`bba.deterministic_classifier` (#8): the
``delta_hb_bypass`` flag and the freshness tier directly drive Hb-tier rules.
"""

from bba.hb_lookup.anchor import (
    AnchorCandidate,
    EvidenceAnchor,
    resolve_evidence_anchor,
    resolve_hb_with_fallback,
)
from bba.hb_lookup.lookup import lookup_hb
from bba.hb_lookup.models import (
    DeltaHbWindow,
    HbFreshness,
    HbLookupResult,
    HbObservation,
    HbSource,
)
from bba.hb_lookup.parse import parse_hb_value

__all__ = [
    "AnchorCandidate",
    "DeltaHbWindow",
    "EvidenceAnchor",
    "HbFreshness",
    "HbLookupResult",
    "HbObservation",
    "HbSource",
    "lookup_hb",
    "parse_hb_value",
    "resolve_evidence_anchor",
    "resolve_hb_with_fallback",
]
