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


def default_role_classifier(
    *,
    original_text: str,
    context: str,
    span: RedactionSpan,
) -> RoleToken | None:
    """The wrapper's built-in :class:`RoleClassifier` implementation.

    Checks the span's own ``original_text`` first (many honorifics —
    ``"Dr."``, ``"นพ."`` — are inside the redacted name span itself,
    not in the surrounding context). Falls back to the caller-supplied
    ``context``.
    """
    span_text = span.original_text or original_text[span.start : span.end]
    if span_text:
        derived = classify_role_by_cues(span_text)
        if derived is not None:
            return derived
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
