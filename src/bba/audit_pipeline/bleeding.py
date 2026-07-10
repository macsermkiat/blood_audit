"""Bleeding-analysis helpers for the RBC over-clear guardrail (spec #89 / T1).

WHY: the over-clear guardrail (:mod:`bba.audit_pipeline.replay`) asserts
``INAPPROPRIATE`` when the LLM clears a gray-zone order the deterministic
leg withheld and no *structured* hard signal justifies it. Active bleeding
is the one soft, prose-only indication the committee agreed to trust — but
only when it is a genuine major bleed, never a 2x2 cm gauze stain. This
module is the sole place that trust is granted, kept small, pure, and
side-effect-free so the guardrail stays deterministic and replay-stable.

Locked decision (spec #89 #2): a clear survives only when a single
active-bleeding indication has its own confidence >= 0.8 AND either a
documented volume strictly > 300 mL OR an explicit life-threatening /
uncontrolled marker. Small or qualitative bleeds do not clear.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

__all__ = [
    "LLM_OVERCLEAR_MIN_BLEED_CONFIDENCE",
    "LLM_OVERCLEAR_MIN_BLEED_ML",
    "has_life_threatening_marker",
    "parse_max_volume_ml",
    "qualified_bleeding_exempt",
]

LLM_OVERCLEAR_MIN_BLEED_ML: float = 300.0
"""Documented blood loss (mL) that must be *strictly exceeded* for an
active-bleeding prose indication to keep a gray-zone clear APPROPRIATE.
Exactly 300 mL does not qualify (user-chosen threshold, spec #89 #2)."""

LLM_OVERCLEAR_MIN_BLEED_CONFIDENCE: float = 0.8
"""The active-bleeding indication's *own* confidence must be at least this
for the guardrail to trust its prose. Model confidence alone cannot clear a
small bleed — the volume / life-threatening test still has to pass."""

# Volume tokens. Litres scale x1000; every other unit is already millilitres
# (cc, ml, and the Thai มล./ซีซี). Case-sensitive by design (spec #89 §3.c
# regex) so the litre "L" alternative cannot swallow a stray lowercase "l".
#
# Correct AND linear on this replay-critical path — two properties, both load-
# bearing (each was a separate stop-time review finding):
#
#   * NO MID-NUMBER RESTART (correctness). A bounded or naive scan can restart
#     inside a long number and read a spurious sub-value — e.g.
#     "0.0000000000000350 mL" misread as 350 mL, which would falsely qualify a
#     vanishingly tiny bleed. The negative lookbehinds pin a match to a genuine
#     number START: ``(?<![0-9])`` never mid-digit-run; ``(?<![0-9][,.])``
#     never at a grouping-comma / decimal-point continuation (so "1,100" and
#     "12.5" read whole). A comma/period used as a plain delimiter (preceded by
#     a non-digit, e.g. "gauze,500 mL") still starts a fresh match. Result is
#     byte-identical to a naive unbounded scan on every input.
#   * LINEAR TIME. The naive scan is O(n^2): ``finditer`` re-scans the tail from
#     every offset of a long digit run. ``(?<![0-9])`` fails in O(1) at every
#     interior digit offset, so real work happens only at number starts; the
#     possessive ``[0-9,]*+`` / ``[0-9]++`` stop the one real attempt from
#     backtracking. Measured: 400k adversarial chars in <25 ms (naive: ~3 s at
#     only 10k). No exponential ReDoS.
#
# Case-sensitive by design (see the "Volume tokens" note above).
_VOLUME_RE = re.compile(
    r"(?<![0-9])(?<![0-9][,.])([0-9][0-9,]*+(?:\.[0-9]++)?)"
    r"\s*(mL|ml|cc|L|liters|liter|มล\.?|ซีซี)"
)

_MAX_SCAN_CHARS: int = 10_000
"""Hard upper bound on characters scanned for a volume — belt-and-suspenders
now that :data:`_VOLUME_RE` is linear-time. Real clinical quotes are a short
note field, far below this; the cap only guards against a future regex edit
reintroducing super-linear scanning on an attacker-influenced quote length."""

_LITRE_UNITS: frozenset[str] = frozenset({"l", "liter", "liters"})

# Conservative, high-confidence-gated prose markers of a life-threatening or
# uncontrolled bleed. Deliberately narrow (spec #89 risk note): benign
# bleeding language ("oozing", "bleeding precaution") must never match. Thai
# phrases are unambiguous ("bleeding won't stop", "hemorrhagic shock").
_LIFE_THREATENING_MARKERS: tuple[str, ...] = (
    "life-threatening",
    "life threatening",
    "uncontrolled",
    "exsanguinat",  # exsanguinating / exsanguination
    "hemorrhagic shock",
    "haemorrhagic shock",
    "active hemorrhage",
    "active haemorrhage",
    "เลือดออกไม่หยุด",  # bleeding does not stop (uncontrolled)
    "ช็อกจากการเสียเลือด",  # shock from blood loss (hemorrhagic shock)
)

# Negation qualifiers that disqualify an otherwise ACTIVE_BLEEDING-prefixed
# code: a risk of / history of / not-active bleed is not an active bleed.
_ACTIVE_BLEEDING_PREFIX = "ACTIVE_BLEEDING"
_ACTIVE_BLEEDING_NEGATIONS: tuple[str, ...] = ("RISK", "HISTORY", "NOT_ACTIVE")

# Prose negators that turn a life-threatening marker into a documented
# ABSENCE of the emergency ("no active hemorrhage"). Scanned in a short,
# clause-bounded window before each marker occurrence (Codex PR #97 P2).
# The screen deliberately errs fail-closed: a false negation hit merely
# withholds the exemption (the assert stands); it can never auto-clear.
_MARKER_NEGATION_TOKENS: tuple[str, ...] = (
    "no ",
    "not ",
    "non-",
    "without ",
    "denies ",
    "denied ",
    "negative for ",
    "ไม่มี",  # "there is no ..."
    "ไม่พบ",  # "... not found"
    "ปฏิเสธ",  # "denies ..."
)
_MARKER_NEGATION_WINDOW_CHARS = 30
# A clause boundary cuts the lookback so an unrelated negation earlier in
# the sentence ("no fever today; active hemorrhage") cannot suppress a
# genuine marker in the next clause.
_MARKER_CLAUSE_BOUNDARIES: tuple[str, ...] = (";", ".", ",", ":", "\n")


def parse_max_volume_ml(text: str) -> float | None:
    """Return the largest documented blood-loss volume in ``text`` as mL.

    Reads English (``ml`` / ``cc`` / ``L``) and Thai (``มล.`` / ``ซีซี``)
    units, strips comma thousands, and scales litres x1000. When several
    figures appear, the maximum is returned (the guardrail judges the worst
    documented bleed). Returns ``None`` when no volume token is present.

    Only the first :data:`_MAX_SCAN_CHARS` characters are scanned so an
    unbounded quote length cannot stall this replay-critical path.
    """
    best: float | None = None
    for match in _VOLUME_RE.finditer(text[:_MAX_SCAN_CHARS]):
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:  # pragma: no cover - regex guarantees a number
            continue
        unit = match.group(2).lower()
        millilitres = value * 1000.0 if unit in _LITRE_UNITS else value
        if best is None or millilitres > best:
            best = millilitres
    return best


def has_life_threatening_marker(text: str) -> bool:
    """True iff ``text`` names an explicit life-threatening / uncontrolled bleed.

    A conservative case-insensitive keyword scan (spec #89 §3.c). Benign
    bleeding language must not fire — this is the un-quantified path that can
    clear an order on prose alone, so it stays deliberately narrow.

    Each marker occurrence is screened for a preceding negator within a
    clause-bounded window ("no active hemorrhage" documents the ABSENCE of
    the emergency, Codex PR #97 P2). The screen errs fail-closed: a false
    negation hit only withholds the exemption, never auto-clears.
    """
    lowered = text.lower()
    for marker in _LIFE_THREATENING_MARKERS:
        start = 0
        while (idx := lowered.find(marker, start)) != -1:
            if not _marker_occurrence_negated(lowered, idx):
                return True
            start = idx + len(marker)
    return False


def _marker_occurrence_negated(lowered: str, marker_index: int) -> bool:
    """True iff a negator precedes ``marker_index`` within the same clause."""
    window = lowered[
        max(0, marker_index - _MARKER_NEGATION_WINDOW_CHARS) : marker_index
    ]
    cut = max(
        (window.rfind(boundary) for boundary in _MARKER_CLAUSE_BOUNDARIES),
        default=-1,
    )
    if cut != -1:
        window = window[cut + 1 :]
    return any(token in window for token in _MARKER_NEGATION_TOKENS)


def qualified_bleeding_exempt(indications: Iterable[Mapping[str, object]]) -> bool:
    """True iff a genuine major active bleed justifies keeping an order clear.

    Fires when *one* indication satisfies all of: an
    ``ACTIVE_BLEEDING``-prefixed ``code`` with no negation qualifier
    (``RISK`` / ``HISTORY`` / ``NOT_ACTIVE``); its own ``confidence`` >=
    :data:`LLM_OVERCLEAR_MIN_BLEED_CONFIDENCE`; and either a parsed volume
    strictly > :data:`LLM_OVERCLEAR_MIN_BLEED_ML` OR a life-threatening
    marker in its ``quote``.

    Every field is read defensively (``.get`` + ``isinstance``): the RBC
    replay path does not fully schema-validate the indication list, so a
    missing key, a non-string code/quote, or a non-numeric, out-of-range
    ([0,1]) or NaN confidence must never raise and must never accidentally
    exempt an over-clear. The per-indication confidence is used, never a
    min-across aggregate.
    """
    for indication in indications:
        if not isinstance(indication, Mapping):
            continue

        code = indication.get("code")
        if not isinstance(code, str):
            continue
        upper = code.upper()
        if not upper.startswith(_ACTIVE_BLEEDING_PREFIX):
            continue
        if any(negation in upper for negation in _ACTIVE_BLEEDING_NEGATIONS):
            continue

        confidence = indication.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            continue
        # The tool schema means confidence as a [0,1] probability; an
        # out-of-range value (2.0, inf) or NaN is schema drift and must read
        # as malformed — never as "very sure" (spec #89 story 15). The chained
        # comparison rejects NaN because NaN fails every comparison.
        if not (0.0 <= confidence <= 1.0):
            continue
        if confidence < LLM_OVERCLEAR_MIN_BLEED_CONFIDENCE:
            continue

        quote = indication.get("quote")
        if not isinstance(quote, str):
            continue
        # Known limitation (spec #89 §3.c locks the parser as a plain mL
        # reader): parse_max_volume_ml is context-free, so an IV-fluid /
        # urine-output / irrigation volume co-located in the same
        # ACTIVE_BLEEDING quote can satisfy the >300 mL test on a
        # non-blood-loss figure. Bounded by the ACTIVE_BLEEDING code family +
        # the >=0.8 confidence gate above; tracked as a follow-up, not fixed
        # here (a context-aware parser is out of scope for T1).
        volume = parse_max_volume_ml(quote)
        if (
            volume is not None and volume > LLM_OVERCLEAR_MIN_BLEED_ML
        ) or has_life_threatening_marker(quote):
            return True

    return False
