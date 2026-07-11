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
    "is_active_bleeding_code",
    "marker_occurrence_negated",
    "parse_max_volume_ml",
    "qualified_bleeding_exempt",
    "quote_negates_bleeding",
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
# Matched as whole _-delimited tokens (NOT_ACTIVE as an adjacent pair) so a
# genuine qualifier like ACTIVE_BLEEDING_BRISK is not misread as RISK
# (Codex PR #97 round 3).
_ACTIVE_BLEEDING_PREFIX = "ACTIVE_BLEEDING"
_ACTIVE_BLEEDING_NEGATION_TOKENS: frozenset[str] = frozenset({"RISK", "HISTORY"})


def _code_negation_qualified(upper_code: str) -> bool:
    """True iff the _-delimited code carries a RISK / HISTORY / NOT_ACTIVE
    qualifier as whole tokens."""
    segments = upper_code.split("_")
    if any(segment in _ACTIVE_BLEEDING_NEGATION_TOKENS for segment in segments):
        return True
    return any(
        first == "NOT" and second == "ACTIVE"
        for first, second in zip(segments, segments[1:])
    )


def is_active_bleeding_code(code: str) -> bool:
    """True iff ``code`` is ACTIVE_BLEEDING-family with no negation qualifier.

    Shared with the replay guardrail's hemodynamic-accompaniment check
    (owner ruling on #98): a hypotension citation only floors to review when
    real bleeding evidence accompanies it, judged by the same code-family
    rules as the exemption itself."""
    upper = code.upper()
    return upper.startswith(_ACTIVE_BLEEDING_PREFIX) and not _code_negation_qualified(
        upper
    )


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
    # Pre-marker exclusion phrases (Codex PR #97 round 6): "ruled out
    # active hemorrhage" and history/risk-of prose are documented
    # non-events, not active bleeds.
    "ruled out",
    "rule out",
    "r/o",  # bare form so "r/o:" also matches (round 7)
    "absence of",
    "history of",
    "hx of",
    "risk of",
    "ไม่มี",  # "there is no ..."
    "ไม่พบ",  # "... not found"
    "ปฏิเสธ",  # "denies ..."
    "ประวัติ",  # "history of ..."
)
_MARKER_NEGATION_WINDOW_CHARS = 30
# A clause boundary cuts the lookback so an unrelated negation earlier in
# the sentence ("no fever today; active hemorrhage") cannot suppress a
# genuine marker in the next clause. ":" is NOT a boundary on either side:
# exclusion formatting binds across it ("r/o: active hemorrhage", round 7),
# mirroring the post-marker "shock: absent" reasoning. A benign label
# before a colon carries no negator, so markers still flag.
_MARKER_CLAUSE_BOUNDARIES: tuple[str, ...] = (";", ".", ",", "\n")

# Trailing negators that void the marker AFTER it in the same clause
# ("active hemorrhage is not present", "uncontrolled bleeding denied" —
# Codex PR #97 round 2). The post-marker window keeps ":" inside the clause
# ("hemorrhagic shock: absent" is one label:value statement) but stops at
# ";.,\n" so a negation in the next clause cannot void a standing marker.
_MARKER_POST_NEGATION_TOKENS: tuple[str, ...] = (
    "not ",
    "no longer",
    "denie",  # denied / denies
    "resolved",
    "ruled out",
    "absent",
    "negative",
    "ไม่",  # Thai negator ("... has stopped / is absent")
)
_MARKER_POST_CLAUSE_BOUNDARIES: tuple[str, ...] = (";", ".", ",", "\n")

# Markers that are generic clinical intensifiers rather than bleed terms —
# "uncontrolled pain", "life-threatening arrhythmia" must not exempt an
# over-clear (Codex PR #97 round 4). These require a bleeding term in the
# same clause; the remaining markers name the bleed inherently
# (hemorrhagic shock, active hemorrhage, exsanguinat-, the Thai phrases).
_CONTEXT_REQUIRED_MARKERS: frozenset[str] = frozenset(
    {"life-threatening", "life threatening", "uncontrolled"}
)
_BLEEDING_CONTEXT_TERMS: tuple[str, ...] = (
    "bleed",  # bleed / bleeding / bleeds
    "hemorrhag",  # hemorrhage / hemorrhagic
    "haemorrhag",
    "blood loss",
    "exsanguinat",
    "เลือด",  # Thai blood/bleeding compounds
)
_MARKER_CONTEXT_WINDOW_CHARS = 40

# Bleed-synonym terms recognised ONLY by the negation screen
# (:func:`quote_negates_bleeding`, Codex PR #99 round 4): "denies melena"
# miscoded as ACTIVE_BLEEDING must not pass as accompaniment just because
# the generic terms miss the synonym. Kept OUT of
# :data:`_BLEEDING_CONTEXT_TERMS` deliberately — adding them there would
# widen the life-threatening-marker exemption ("uncontrolled melena"
# auto-clearing), and prose auto-clear surface is a committee decision.
# A false hit here only withholds the hemodynamic floor (assert stands).
_BLEEDING_NEGATION_SCREEN_TERMS: tuple[str, ...] = _BLEEDING_CONTEXT_TERMS + (
    "melena",
    "hematemesis",
    "haematemesis",
    "hematochezia",
    "haematochezia",
    "hemoptysis",
    "haemoptysis",
    "epistaxis",
    "ถ่ายดำ",  # melena ("black stool")
)

# Clause boundaries for the accompaniment screen ONLY (Codex PR #99 round
# 5): a clinical denial distributes across a comma list ("denies bleeding,
# melena"), so the comma must not shield later items from the negator.
# The marker screens above deliberately KEEP the comma boundary — there a
# false negation hit would wrongly assert a genuine major bleed ("no fever
# today, hemorrhagic shock"), while here it merely withholds the
# hemodynamic floor and the ruling's assert stands. The window is wider
# than the marker screens' for the same reason: a three-item denial list
# ("no hematemesis, hematochezia, or melena") outruns 30 chars.
_DENIAL_LIST_BOUNDARIES: tuple[str, ...] = (";", ".", "\n")
_DENIAL_LIST_WINDOW_CHARS = 60

# Contrastive connectors cut the PRE side only (Codex PR #99 round 6):
# in "denies hematemesis but melena ongoing" the denial binds up to the
# connector — what follows is a documented ACTIVE bleed the floor must
# see. Asymmetric by design: a contrastive AFTER a term still lets a
# trailing negator void it ("melena noted but now resolved" is a bleed
# documented as over), so the post side keeps flowing across "but".
_DENIAL_PRE_BOUNDARIES: tuple[str, ...] = _DENIAL_LIST_BOUNDARIES + (
    " but ",
    "however",
    "แต่",  # Thai "but"
)

# Post-side double negatives that mean the bleed is ONGOING, not absent
# (Codex PR #99 round 7): in "bleeding not controlled" the negator binds
# the control verb, not the bleeding. When one of these appears in the
# clause-bounded post window, the post negation is a false read and the
# term stays visible to the floor. Accompaniment screen only — applying
# these to the marker screens would widen the exemption / auto-clear
# surface, which is committee-owned.
_POST_STILL_ACTIVE_PATTERNS: tuple[str, ...] = (
    "not controlled",
    "not yet controlled",
    "not under control",
    "not stopped",
    "not stopping",
    "not resolved",
    "no longer controlled",
    "ไม่หยุด",  # "... does not stop" (also matches ยังไม่หยุด)
)


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
    the emergency, Codex PR #97 P2). Generic intensifier markers
    ("uncontrolled", "life-threatening") additionally require a bleeding
    term in the same clause — "uncontrolled pain" is not a bleed (round 4).
    Both screens err fail-closed: a false hit only withholds the exemption,
    never auto-clears.
    """
    lowered = text.lower()
    for marker in _LIFE_THREATENING_MARKERS:
        start = 0
        while (idx := lowered.find(marker, start)) != -1:
            end = idx + len(marker)
            if not marker_occurrence_negated(lowered, idx, end) and (
                marker not in _CONTEXT_REQUIRED_MARKERS
                or _bleeding_context_near(lowered, idx, end)
            ):
                return True
            start = end
    return False


def _bleeding_context_near(lowered: str, start: int, end: int) -> bool:
    """True iff a bleeding term appears near ``lowered[start:end]`` within
    the same clause (either side)."""
    pre = lowered[max(0, start - _MARKER_CONTEXT_WINDOW_CHARS) : start]
    cut = max(
        (pre.rfind(boundary) for boundary in _MARKER_CLAUSE_BOUNDARIES),
        default=-1,
    )
    if cut != -1:
        pre = pre[cut + 1 :]
    post = lowered[end : end + _MARKER_CONTEXT_WINDOW_CHARS]
    cut = min(
        (
            found
            for boundary in _MARKER_POST_CLAUSE_BOUNDARIES
            if (found := post.find(boundary)) != -1
        ),
        default=-1,
    )
    if cut != -1:
        post = post[:cut]
    return any(term in pre or term in post for term in _BLEEDING_CONTEXT_TERMS)


def marker_occurrence_negated(lowered: str, start: int, end: int) -> bool:
    """True iff a negator voids the ``lowered[start:end]`` marker occurrence —
    either preceding it or trailing it within the same clause.

    Public seam shared by this module's marker/quote screens and the replay
    guardrail's fluid-refractory check (Codex PR #99 round 4: "not
    refractory after IV fluids" must not read as refractoriness), so every
    negation read uses the same negator tokens and clause windows."""
    return _occurrence_negated(
        lowered, start, end, _MARKER_CLAUSE_BOUNDARIES, _MARKER_POST_CLAUSE_BOUNDARIES
    )


def _occurrence_negated(
    lowered: str,
    start: int,
    end: int,
    pre_boundaries: tuple[str, ...],
    post_boundaries: tuple[str, ...],
    window: int = _MARKER_NEGATION_WINDOW_CHARS,
    post_rescue: tuple[str, ...] = (),
) -> bool:
    """Negator scan around ``lowered[start:end]`` with caller-chosen clause
    boundaries — the marker screens keep the comma boundary, the
    accompaniment screen drops it (see :data:`_DENIAL_LIST_BOUNDARIES`).
    A ``post_rescue`` pattern in the post window overrides a post-side
    negator hit (still-active double negatives, round 7)."""
    pre = lowered[max(0, start - window) : start]
    cut = max(
        (pre.rfind(boundary) for boundary in pre_boundaries),
        default=-1,
    )
    if cut != -1:
        pre = pre[cut + 1 :]
    if any(token in pre for token in _MARKER_NEGATION_TOKENS):
        return True

    post = lowered[end : end + window]
    cut = min(
        (found for boundary in post_boundaries if (found := post.find(boundary)) != -1),
        default=-1,
    )
    if cut != -1:
        post = post[:cut]
    if not any(token in post for token in _MARKER_POST_NEGATION_TOKENS):
        return False
    return not any(pattern in post for pattern in post_rescue)


def quote_negates_bleeding(quote: str) -> bool:
    """True iff ``quote`` mentions bleeding only in negation-voided form —
    it documents the ABSENCE of a bleed ("no active hemorrhage", "r/o GI
    bleed", "ไม่มีเลือดออก"), not evidence of one.

    WHY (Codex PR #99 round 2 / owner ruling #98): the replay guardrail's
    hemodynamic-accompaniment check trusts an ACTIVE_BLEEDING-family
    citation as qualifier-(1) bleeding evidence on code + confidence alone;
    a mislabeled citation whose grounded quote actually negates bleeding
    must not float a bare-hypotension over-clear to review. Reuses the same
    negator / clause-window machinery as the life-threatening-marker screen
    so exemption and accompaniment read prose identically, over the
    synonym-extended term list (:data:`_BLEEDING_NEGATION_SCREEN_TERMS` —
    "denies melena" counts, Codex PR #99 round 4) with denial-list
    boundaries (round 5: the negator distributes across "denies bleeding,
    melena"; ";", ".", "\\n" still cut). A quote with no
    bleeding term returns ``False`` — this screen only rejects positively
    negated prose; code-level trust stays the caller's decision. One
    non-negated bleeding mention keeps the quote usable (a note often
    clears one site while another still bleeds). Errs fail-closed for the
    floor: a false hit merely withholds it (the assert stands); it can
    never auto-clear.

    Post-side double negatives whose negator binds a control verb
    ("bleeding not controlled", "เลือดออกไม่หยุด" — the bleed is ONGOING)
    are rescued via :data:`_POST_STILL_ACTIVE_PATTERNS` (round 7) and stay
    visible to the floor.
    """
    lowered = quote.lower()
    found_negated = False
    for term in _BLEEDING_NEGATION_SCREEN_TERMS:
        start = 0
        while (idx := lowered.find(term, start)) != -1:
            end = idx + len(term)
            if not _occurrence_negated(
                lowered,
                idx,
                end,
                _DENIAL_PRE_BOUNDARIES,
                _DENIAL_LIST_BOUNDARIES,
                window=_DENIAL_LIST_WINDOW_CHARS,
                post_rescue=_POST_STILL_ACTIVE_PATTERNS,
            ):
                return False
            found_negated = True
            start = end
    return found_negated


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
        if _code_negation_qualified(upper):
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
