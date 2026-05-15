"""Role-token mapping for the deid_redactor post-processing wrapper.

PRD §8: "Emits role-preserving tokens [ATTENDING] / [NURSE] / [PATIENT] /
[FAMILY] (post-processing wrapper, not a fork of the redactor)." The
underlying ``thai-medical-deid`` backend tags every PHI person span with
the generic ``[PERSON]`` token; the wrapper inspects the original text
plus surrounding context for each span and upgrades the placeholder to a
role-specific token.

The mapping is a small lexicon + window-context rule, not an LLM. The
classifier reads the ``ROLE_CONTEXT_WINDOW`` characters of original text
immediately before AND after the span. Cues are searched in that window;
first matching family in priority order wins (ATTENDING > NURSE >
PATIENT > FAMILY). No cue → returns ``None`` and the wrapper preserves
the generic ``[PERSON]`` token.

Determinism: the lexicon is a module-level constant, the search is a
case-insensitive substring scan. Pure function; no I/O; deterministic
output for the bundle-hash stability AC.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from bba.deid_redactor.exceptions import BackendRedactionError
from bba.deid_redactor.models import RedactionSpan, RoleClassifier, RoleToken


ROLE_CONTEXT_WINDOW: int = 40
"""Half-window (in characters) of original-text context around a span."""


ATTENDING_CUES: tuple[str, ...] = (
    "dr.",
    "dr",
    "md",
    "physician",
    "attending",
    "นพ.",
    "พญ.",
    "อาจารย์หมอ",
    "อาจารย์แพทย์",
    "หมอ",
)
"""Substring cues that mark the surrounding context as physician-spoken.

ASCII cues are matched with word-boundary regex (``\\b``) so ``"dr"``
matches ``"Dr. Smith"`` and ``"Dr Smith"`` but not ``"drainage"``. Thai
cues are matched as raw substrings because Thai script does not use
inter-word spaces and ``\\b`` does not align with Thai word boundaries.
"""


NURSE_CUES: tuple[str, ...] = (
    "nurse",
    "rn",
    "พยาบาล",
)


PATIENT_CUES: tuple[str, ...] = (
    "patient",
    "pt",
    "ผู้ป่วย",
    "คนไข้",
)


FAMILY_CUES: tuple[str, ...] = (
    "mother",
    "father",
    "spouse",
    "wife",
    "husband",
    "daughter",
    "son",
    "parent",
    "family",
    "relative",
    "แม่",
    "พ่อ",
    "สามี",
    "ภรรยา",
    "ลูก",
    "ญาติ",
)


_ROLE_PRIORITY: tuple[tuple[RoleToken, tuple[str, ...]], ...] = (
    (RoleToken.ATTENDING, ATTENDING_CUES),
    (RoleToken.NURSE, NURSE_CUES),
    (RoleToken.PATIENT, PATIENT_CUES),
    (RoleToken.FAMILY, FAMILY_CUES),
)


def _is_ascii(text: str) -> bool:
    """Whether ``text`` is pure ASCII — boundary semantics differ by script."""
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def _compile_role_patterns() -> tuple[tuple[RoleToken, re.Pattern[str]], ...]:
    """Compile per-role alternation regex from the lexicons.

    ASCII cues are word-boundary anchored (``\\b...\\b``) so ``"dr"`` does
    not match inside ``"drainage"`` or ``"hydrate"``. Thai cues are bare
    substrings — the Unicode ``\\b`` definition does not align with Thai
    word boundaries (Thai script has no inter-word spaces by convention),
    and substring search is the standard approach in Thai NLP for short
    lexicon lookups.

    Cues containing a trailing ``.`` (e.g. ``"dr."``, ``"นพ."``) have the
    period included in the literal; the regex escape preserves it. Such
    cues use ``\\b`` only on the leading side (the trailing ``.`` is its
    own non-word-character boundary).
    """
    compiled: list[tuple[RoleToken, re.Pattern[str]]] = []
    for role, cues in _ROLE_PRIORITY:
        alternatives: list[str] = []
        for cue in cues:
            if _is_ascii(cue):
                escaped = re.escape(cue)
                if cue.endswith("."):
                    # trailing period IS the trailing boundary
                    alternatives.append(rf"\b{escaped}")
                else:
                    alternatives.append(rf"\b{escaped}\b")
            else:
                alternatives.append(re.escape(cue))
        compiled.append(
            (role, re.compile("|".join(alternatives), re.IGNORECASE))
        )
    return tuple(compiled)


_ROLE_PATTERNS: tuple[tuple[RoleToken, re.Pattern[str]], ...] = (
    _compile_role_patterns()
)


_HONORIFICS: tuple[tuple[RoleToken, tuple[str, ...]], ...] = (
    (
        RoleToken.ATTENDING,
        (
            "dr.",
            "dr",
            "md",
            "physician",
            "attending",
            "นพ.",
            "พญ.",
            "อาจารย์หมอ",
            "อาจารย์แพทย์",
        ),
    ),
    (RoleToken.NURSE, ("rn", "rn.")),
    (RoleToken.PATIENT, ("pt", "pt.")),
    # FAMILY intentionally has no honorifics — every family cue ("son",
    # "mother", "wife", "ลูก", etc.) doubles as a common given/surname
    # in real KCMH demographics, so a span whose original_text equals
    # one of these terms cannot be unambiguously classified by the
    # in-span pass. The proximity layer handles FAMILY based on the
    # surrounding cue placement instead.
)
"""Narrow lexicon used for span-internal cue lookup.

The full :data:`_ROLE_PRIORITY` lexicon is too permissive for span-text
matching: many family/patient cue words are also common names
("son", "mother", "patient", "ผู้ป่วย" → all real surnames in the KCMH
population). Restricting the in-span pass to titles + unambiguous
abbreviations (codex GitHub review on PR #40 round 2) ensures a
name/cue collision falls through to the proximity layer rather than
short-circuiting on the wrong role.
"""


def _compile_honorific_patterns() -> tuple[tuple[RoleToken, re.Pattern[str]], ...]:
    """Compile :data:`_HONORIFICS` with the same boundary rules as roles."""
    compiled: list[tuple[RoleToken, re.Pattern[str]]] = []
    for role, cues in _HONORIFICS:
        alternatives: list[str] = []
        for cue in cues:
            if _is_ascii(cue):
                escaped = re.escape(cue)
                if cue.endswith("."):
                    alternatives.append(rf"\b{escaped}")
                else:
                    alternatives.append(rf"\b{escaped}\b")
            else:
                alternatives.append(re.escape(cue))
        compiled.append(
            (role, re.compile("|".join(alternatives), re.IGNORECASE))
        )
    return tuple(compiled)


_HONORIFIC_PATTERNS: tuple[tuple[RoleToken, re.Pattern[str]], ...] = (
    _compile_honorific_patterns()
)


def classify_honorific_in_span(span_text: str) -> RoleToken | None:
    """Return the role of the first matching honorific in ``span_text``.

    Used by :func:`default_role_classifier` for the in-span pass. The
    narrower :data:`_HONORIFICS` lexicon is searched here (not the full
    role lexicon) so a name that happens to equal a family/patient cue
    word (``"Son"``, ``"Mother"``) does not bypass the proximity layer.
    """
    haystack = unicodedata.normalize("NFC", span_text)
    for role, pattern in _HONORIFIC_PATTERNS:
        if pattern.search(haystack):
            return role
    return None


def extract_context(
    *,
    original_text: str,
    span: RedactionSpan,
    window: int = ROLE_CONTEXT_WINDOW,
) -> str:
    """Slice the original-text context around ``span`` (±``window`` chars).

    Returns ``before + " " + after`` so the cue search is one ``str.find``
    instead of two. The two halves are joined by a single space so a cue
    straddling the span boundary is not glued into a false match.
    """
    pre_start = max(0, span.start - window)
    pre_end = span.start
    post_start = span.end
    post_end = min(len(original_text), span.end + window)
    before = original_text[pre_start:pre_end]
    after = original_text[post_start:post_end]
    return f"{before} {after}"


def classify_role_by_cues(context: str) -> RoleToken | None:
    """Return the role implied by cue presence in ``context``, or ``None``.

    Walks the role-priority list in order; the first role whose
    word-boundary-anchored regex finds a match in the NFC-normalized
    context wins. ``None`` means no cue → leave the span as
    :attr:`RoleToken.PERSON`.
    """
    haystack = unicodedata.normalize("NFC", context)
    for role, pattern in _ROLE_PATTERNS:
        if pattern.search(haystack):
            return role
    return None


def _classify_by_proximity(*, before: str, after: str) -> RoleToken | None:
    """Pick the role whose cue match is closest to the span boundary.

    The span boundary is the end of ``before`` (equivalently, the start
    of ``after``). For each role, scan both halves with the role's
    regex and find the smallest distance to the boundary:

    * a match in ``before`` sits at distance ``len(before) - match.end()``
    * a match in ``after`` sits at distance ``match.start()``

    Across roles, the smallest distance wins; ties resolve by global
    priority (``_ROLE_PRIORITY`` iteration order, ATTENDING first).
    Returns ``None`` when no role has any match.

    Proximity rather than global priority is required because clinical
    prose routinely names multiple actors in a single sentence
    (``"Dr. Smith saw patient John Doe"``). A pure-priority scan would
    label the patient span as ATTENDING because ``dr.`` outranks
    ``patient`` regardless of distance; proximity correctly attaches
    the role of the nearest cue to the redacted name.
    """
    before_n = unicodedata.normalize("NFC", before)
    after_n = unicodedata.normalize("NFC", after)
    before_len = len(before_n)

    best_role: RoleToken | None = None
    best_distance = -1  # placeholder; replaced on first match

    for role, pattern in _ROLE_PATTERNS:
        role_distance = -1
        for m in pattern.finditer(before_n):
            d = before_len - m.end()
            if role_distance == -1 or d < role_distance:
                role_distance = d
        for m in pattern.finditer(after_n):
            d = m.start()
            if role_distance == -1 or d < role_distance:
                role_distance = d
        if role_distance == -1:
            continue
        # Strict ``<`` so equal-distance ties fall to the higher-priority
        # role (which iterates first); flips of priority order would
        # silently re-label every classification on a deploy.
        if best_role is None or role_distance < best_distance:
            best_role = role
            best_distance = role_distance
    return best_role


def default_role_classifier(
    *,
    original_text: str,
    context: str,
    span: RedactionSpan,
) -> RoleToken | None:
    """The wrapper's built-in :class:`RoleClassifier` implementation.

    Resolution order:

    1. The span's own ``original_text``, matched against the
       :data:`_HONORIFICS` narrow lexicon via
       :func:`classify_honorific_in_span`. Honorifics ("Dr.", "MD",
       "นพ.") often live inside the redacted name span itself; matching
       them in-span is unambiguous. The narrow lexicon excludes family
       and patient cue words ("son", "mother", "patient") that double
       as common names — a name/cue collision must fall through to
       proximity, not short-circuit on the wrong role (codex GitHub
       review on PR #40 round 2).
    2. The original-text window around the span, classified with
       :func:`_classify_by_proximity` — the role whose cue is nearest
       to the span wins. Prevents misclassifying spans in multi-actor
       sentences (``"Dr. Smith saw patient John Doe"`` — codex GitHub
       review on PR #40 round 1).
    3. The caller-supplied ``context`` as a final fallback, using
       priority-only classification. Used when the caller passes a
       custom context window that the span-derived slice cannot
       reproduce.
    """
    span_text = span.original_text or original_text[span.start : span.end]
    if span_text:
        derived = classify_honorific_in_span(span_text)
        if derived is not None:
            return derived

    if original_text and 0 <= span.start <= span.end <= len(original_text):
        pre_start = max(0, span.start - ROLE_CONTEXT_WINDOW)
        post_end = min(len(original_text), span.end + ROLE_CONTEXT_WINDOW)
        before = original_text[pre_start : span.start]
        after = original_text[span.end : post_end]
        proximity = _classify_by_proximity(before=before, after=after)
        if proximity is not None:
            return proximity

    if context:
        return classify_role_by_cues(context)
    return None


_PERSON_PLACEHOLDER: str = RoleToken.PERSON.value


def upgrade_person_tokens(
    *,
    redacted_text: str,
    original_text: str,
    spans: Sequence[RedactionSpan],
    classifier: RoleClassifier,
) -> str:
    """Walk every ``[PERSON]`` placeholder in ``redacted_text`` and upgrade.

    The walk iterates ``spans`` in document order — same order the
    backend emits them — and replaces ``[PERSON]`` occurrences in
    ``redacted_text`` one-by-one with the classifier's choice (or with
    the original ``[PERSON]`` when the classifier returns ``None``).

    Invariant: the number of ``[PERSON]`` placeholders in ``redacted_text``
    must equal the number of ``PERSON``-type entries in ``spans``. A
    mismatch raises :class:`BackendRedactionError`.
    """
    person_spans = tuple(s for s in spans if s.entity_type == "PERSON")
    placeholder_count = redacted_text.count(_PERSON_PLACEHOLDER)
    if placeholder_count != len(person_spans):
        raise BackendRedactionError(
            "PERSON placeholder count mismatch: redacted_text has "
            f"{placeholder_count} '[PERSON]' tokens but spans has "
            f"{len(person_spans)} PERSON entries"
        )

    if not person_spans:
        return redacted_text

    parts: list[str] = []
    cursor = 0
    span_idx = 0
    placeholder_len = len(_PERSON_PLACEHOLDER)

    while cursor < len(redacted_text):
        idx = redacted_text.find(_PERSON_PLACEHOLDER, cursor)
        if idx == -1:
            parts.append(redacted_text[cursor:])
            break
        parts.append(redacted_text[cursor:idx])

        span = person_spans[span_idx]
        ctx = extract_context(original_text=original_text, span=span)
        role = classifier(original_text=original_text, context=ctx, span=span)
        parts.append(role.value if role is not None else _PERSON_PLACEHOLDER)

        cursor = idx + placeholder_len
        span_idx += 1

    return "".join(parts)
