"""Role-token mapping for the deid_redactor post-processing wrapper.

PRD §8: "Emits role-preserving tokens [ATTENDING] / [NURSE] / [PATIENT] /
[FAMILY] (post-processing wrapper, not a fork of the redactor)." The
underlying ``thai-medical-deid`` backend tags every PHI person span with
the generic ``[PERSON]`` token; the wrapper inspects the original text
plus surrounding context for each span and upgrades the placeholder to a
role-specific token.

The mapping is a small lexicon + window-context rule, not an LLM:

* :data:`ATTENDING_CUES` — physician titles / honorifics (``"Dr."``,
  ``"MD"``, Thai ``"นพ."``, ``"พญ."``, ``"อาจารย์หมอ"``).
* :data:`NURSE_CUES` — nursing role markers (``"Nurse"``, ``"RN"``, Thai
  ``"พยาบาล"``).
* :data:`PATIENT_CUES` — patient-of-record markers (``"Patient"``, Thai
  ``"ผู้ป่วย"``, ``"คนไข้"``).
* :data:`FAMILY_CUES` — family-relation markers (``"Mother"``,
  ``"Father"``, ``"Spouse"``, Thai ``"แม่"``, ``"พ่อ"``, ``"สามี"``,
  ``"ภรรยา"``, ``"ลูก"``, ``"ญาติ"``).

The classifier looks at the ``ROLE_CONTEXT_WINDOW`` characters of
original text immediately before AND after the span. Cues are searched
in that window; first matching family wins (priority order:
ATTENDING > NURSE > PATIENT > FAMILY). No cue → returns ``None`` and the
wrapper preserves the generic ``[PERSON]`` token.

Determinism: the lexicon is a module-level constant, the search is a
case-insensitive substring scan. Pure function; no I/O; deterministic
output for the bundle-hash stability AC.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.deid_redactor.models import RedactionSpan, RoleClassifier, RoleToken


ROLE_CONTEXT_WINDOW: int = 40
"""Half-window (in characters) of original-text context around a span.

The classifier reads ``[span.start - ROLE_CONTEXT_WINDOW, span.end +
ROLE_CONTEXT_WINDOW]`` from the original text and searches it for role
cues. 40 chars is a deliberate sweet spot: large enough to catch
"the attending physician Dr. ___ stated that..." (cue precedes the name
by ~15 chars), small enough not to leak the role of the SOAP block's
prior speaker onto the current one (typical block boundary is ~80 chars).
"""


ATTENDING_CUES: tuple[str, ...] = (
    "dr.",
    "dr ",
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

Lowercase (the matcher lowercases the context window). ``"dr "`` with a
trailing space is paired with ``"dr."`` so we accept both ``"Dr. Smith"``
and ``"Dr Smith"`` without also matching ``"drainage"`` mid-word.
"""


NURSE_CUES: tuple[str, ...] = (
    "nurse",
    "rn ",
    "rn.",
    "พยาบาล",
)


PATIENT_CUES: tuple[str, ...] = (
    "patient",
    "pt.",
    "pt ",
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
    "son ",
    "son.",
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


# Priority order — the classifier walks these in sequence and returns the
# FIRST family whose cue is present. ATTENDING beats NURSE beats PATIENT
# beats FAMILY because the upstream lexicons may overlap (a SOAP note
# saying "the nurse told the attending physician Dr. ___" should classify
# the redacted [PERSON] as ATTENDING, not NURSE — the closer cue wins
# only when at equal priority).
_ROLE_PRIORITY: tuple[tuple[RoleToken, tuple[str, ...]], ...] = (
    (RoleToken.ATTENDING, ATTENDING_CUES),
    (RoleToken.NURSE, NURSE_CUES),
    (RoleToken.PATIENT, PATIENT_CUES),
    (RoleToken.FAMILY, FAMILY_CUES),
)


def extract_context(
    *,
    original_text: str,
    span: RedactionSpan,
    window: int = ROLE_CONTEXT_WINDOW,
) -> str:
    """Slice the original-text context around ``span`` (±``window`` chars).

    Returns a single string ``before + " " + after`` so the cue search
    is a single ``str.find`` instead of two. The two halves are joined
    by a single space so a cue that straddles the span boundary (e.g.
    ``"Dr. " + "[PERSON]" + " MD"``) is not glued into a false match.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #17")


def classify_role_by_cues(context: str) -> RoleToken | None:
    """Return the role implied by cue presence in ``context``, or ``None``.

    Walks :data:`_ROLE_PRIORITY` in order; the first role whose cue list
    has any substring present in ``context`` (lowercased, NFC-normalized
    by the wrapper before this call) wins. ``None`` means no cue → leave
    the span as :attr:`RoleToken.PERSON`.

    Determinism: search order is fixed (lexicon order within a family).
    """
    raise NotImplementedError("RED-phase scaffold; see issue #17")


def default_role_classifier(
    *,
    original_text: str,
    context: str,
    span: RedactionSpan,
) -> RoleToken | None:
    """The wrapper's built-in :class:`RoleClassifier` implementation.

    Composes :func:`extract_context` + :func:`classify_role_by_cues`.
    Callers can plug in a custom classifier via the
    :func:`bba.deid_redactor.redactor.redact_bundle` parameter — useful
    for evaluation harnesses that want to A/B a richer classifier
    without forking the wrapper.

    The classifier is called ONLY for spans whose
    ``entity_type == "PERSON"``. Spans of other types (``"DATE"``,
    ``"LOCATION"``, ...) bypass classification and are replaced with
    their type-matching :class:`RoleToken`.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #17")


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
    mismatch is a backend contract violation; the wrapper catches it
    and raises :class:`bba.deid_redactor.exceptions.BackendRedactionError`.

    Other-type tokens (``[DATE]``, ``[LOCATION]``, ...) are NOT touched
    by this function — the wrapper handles them separately.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #17")
