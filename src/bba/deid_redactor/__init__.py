"""bba.deid_redactor — thai-medical-deid wrapper with role tokens.

See issue #17 for acceptance criteria. PRD §8 (Implementation Decisions)
defines the wrapper's contract: a pinned-version post-processing layer on
top of ``macsermkiat/thai-medical-deid`` that emits role-preserving tokens
(``[ATTENDING]`` / ``[NURSE]`` / ``[PATIENT]`` / ``[FAMILY]``), shifts
literal dates to days from a caller-supplied anchor date, and flags
semantic degradation when too many PERSON-class tokens cluster in a small
window. No demographic field reaches the prompt, so the wrapper takes none.

The wrapper is a *pure function* (no I/O, zero deps on Anthropic or HF
transformers): the underlying redactor is plugged in via the
:class:`bba.deid_redactor.RedactorBackend` Protocol. Production wires this
to the vendored ``thai-medical-deid`` package (``TRANSFORMERS_OFFLINE=1``
+ ``HF_HUB_OFFLINE=1`` per PRD §"Stack"); tests use deterministic stubs.
"""

from bba.deid_redactor.phi_patterns import PHI_REGEXES
from bba.deid_redactor.canonical import (
    build_envelope,
    canonical_serialize,
    compute_redaction_hash,
)
from bba.deid_redactor.date_shift import (
    DATE_PATTERNS,
    DateMatch,
    format_offset,
    parse_dates,
    shift_date_spans_in_text,
    shift_dates_in_text,
)
from bba.deid_redactor.exceptions import (
    BackendRedactionError,
    DateShiftError,
    DeidRedactorError,
    HashMismatchError,
)
from bba.deid_redactor.models import (
    BackendRedactionResult,
    NeedsReviewReason,
    NoteInput,
    PERSON_CLASS_TOKENS,
    RedactedNote,
    RedactionRequest,
    RedactionResult,
    RedactionSpan,
    RedactorBackend,
    RedactorVersion,
    RoleClassifier,
    RoleToken,
    SEMANTIC_PERSON_THRESHOLD,
    SEMANTIC_WINDOW_CHARS,
)
from bba.deid_redactor.redactor import redact_bundle
from bba.deid_redactor.roles import (
    ATTENDING_CUES,
    FAMILY_CUES,
    NURSE_CUES,
    PATIENT_CUES,
    ROLE_CONTEXT_WINDOW,
    classify_honorific_in_span,
    classify_role_by_cues,
    default_role_classifier,
    extract_context,
    upgrade_person_tokens,
)
from bba.deid_redactor.semantic import detect_semantic_degradation

__all__ = [
    "ATTENDING_CUES",
    "BackendRedactionError",
    "BackendRedactionResult",
    "DATE_PATTERNS",
    "DateMatch",
    "DateShiftError",
    "DeidRedactorError",
    "FAMILY_CUES",
    "HashMismatchError",
    "NURSE_CUES",
    "NeedsReviewReason",
    "NoteInput",
    "PATIENT_CUES",
    "PHI_REGEXES",
    "PERSON_CLASS_TOKENS",
    "ROLE_CONTEXT_WINDOW",
    "RedactedNote",
    "RedactionRequest",
    "RedactionResult",
    "RedactionSpan",
    "RedactorBackend",
    "RedactorVersion",
    "RoleClassifier",
    "RoleToken",
    "SEMANTIC_PERSON_THRESHOLD",
    "SEMANTIC_WINDOW_CHARS",
    "build_envelope",
    "canonical_serialize",
    "classify_honorific_in_span",
    "classify_role_by_cues",
    "compute_redaction_hash",
    "default_role_classifier",
    "detect_semantic_degradation",
    "extract_context",
    "format_offset",
    "parse_dates",
    "redact_bundle",
    "shift_date_spans_in_text",
    "shift_dates_in_text",
    "upgrade_person_tokens",
]
