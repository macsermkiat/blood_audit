"""Clinical-salience ranker for evidence-bundle truncation (issue #76).

A PURE category-to-bucket mapping (issue #76 User Story 13). When the char cap
forces whole-item drops, the order items are emitted in decides what survives:
the truncation pass drops the LAST item first, so a lower bucket value must
sort earlier and therefore survive longer. This module assigns a coarse,
static salience bucket to a MED drug string so decision-critical evidence
(vasopressors, inotropes, blood products) outranks maintenance fluids
(crystalloids, saline flushes, irrigation) when cap pressure forces a drop.

This is the channel that was starved in Case 2 / REQNO 68012352: the LLM's
medication list was "reduced to saline / irrigation / omeprazole" because the
cap dropped items in arrival order, shedding the pressor while keeping flushes.

BINDING GUARDRAIL: salience is an ORDERING signal only. It never gates,
classifies, or weights a transfusion decision — it decides emission order under
truncation, nothing else. A bucket is not a clinical verdict; all weighing
stays with the LLM (the deterministic classifier has no salience gate). The
buckets are a static keyword seed with clinical sign-off, the same posture as
the cohort_detector allow-lists.

Ambiguity guardrail (mirrors :mod:`bba.vitals_extractor.hemodynamic`): the
genuinely ambiguous abbreviations (``NA`` = sodium, ``NAD`` = no acute
distress) are deliberately ABSENT from the critical keywords — promoting them
to the vasopressor bucket would fabricate pressor support.
"""

from __future__ import annotations

import re
from enum import IntEnum


class SalienceBucket(IntEnum):
    """Coarse clinical-salience tiers, ordered so lower survives truncation.

    Emission sorts ASCENDING and the truncation pass drops the LAST item, so a
    lower value is emitted earlier and shed later. The integer values are part
    of the contract — they are used directly as a sort key.
    """

    CRITICAL = 0
    """Vasopressors, inotropes, and blood products — drop last."""

    ROUTINE = 1
    """The default: anything not recognised as critical or maintenance."""

    MAINTENANCE = 2
    """Crystalloids, saline flushes, irrigation — drop first under cap pressure."""


# Vasopressors / inotropes (full generic names + the Levophed brand) and blood
# products. Matched case-insensitively with word boundaries. Full names only:
# a structured MED ``drug`` field charts "Norepinephrine", never the bare "NE"
# alias, so the ambiguous short tokens (NA/NAD) are simply absent and cannot be
# misread as a pressor. Blood products are included so a unit of PRC/FFP/
# platelets outranks a saline carrier under the cap (issue US8).
_CRITICAL_RE = re.compile(
    r"\b(?:"
    r"nor[\- ]?adrenalin(?:e)?|norepinephrine|levophed|"
    r"epinephrine|adrenalin(?:e)?|dopamine|dobutamine|vasopressin|"
    r"phenylephrine|milrinone|"
    r"l?prc|rbc|packed\s+red|whole\s+blood|"
    r"ffp|fresh\s+frozen\s+plasma|cryoprecipitate|cryo|platelets?"
    r")\b",
    re.IGNORECASE,
)

# Crystalloids / saline / irrigation / flush. ``plasma-lyte`` (a balanced
# crystalloid) lives here, NOT in the blood-product critical set, so the bare
# token "plasma" is deliberately excluded from _CRITICAL_RE to avoid promoting
# Plasma-Lyte. Checked only AFTER the critical set, so a pressor diluted in
# saline ("Norepinephrine in 0.9% NSS") stays CRITICAL — the drug is the signal,
# not its carrier.
_MAINTENANCE_RE = re.compile(
    r"\b(?:"
    r"nss|saline|nacl|"
    r"ringer(?:'?s)?|rls|lactated|acetar|plasma[\- ]?lyte|"
    r"dextrose|d5w?|d10w?|"
    r"sterile\s+water|irrigation|irrigate|flush"
    r")\b",
    re.IGNORECASE,
)


def med_salience(drug: str) -> SalienceBucket:
    """Map a MED ``drug`` string to its :class:`SalienceBucket`.

    Precedence is CRITICAL → MAINTENANCE → ROUTINE: the critical set is checked
    first so a pressor or blood product carried in saline is never demoted to
    maintenance by its diluent. Anything matching neither set is ROUTINE.
    """
    if _CRITICAL_RE.search(drug):
        return SalienceBucket.CRITICAL
    if _MAINTENANCE_RE.search(drug):
        return SalienceBucket.MAINTENANCE
    return SalienceBucket.ROUTINE


__all__ = (
    "SalienceBucket",
    "med_salience",
)
