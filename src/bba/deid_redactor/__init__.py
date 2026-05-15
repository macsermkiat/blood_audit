"""bba.deid_redactor — thai-medical-deid wrapper with role tokens + k-anonymity.

See issue #17 for acceptance criteria. PRD §8 (Implementation Decisions)
defines the wrapper's contract: a pinned-version post-processing layer on
top of ``macsermkiat/thai-medical-deid`` that emits role-preserving tokens
(``[ATTENDING]`` / ``[NURSE]`` / ``[PATIENT]`` / ``[FAMILY]``), enforces
k-anonymity (k ≥ 5 on the ``{ward, ICD-3char, age-band, sex,
admission-month}`` tuple), shifts literal dates to Δ-days-from-admission,
caps age at 89, and flags semantic degradation when too many PERSON-class
tokens cluster in a small window.

The wrapper is a *pure function* (no I/O, zero deps on Anthropic or HF
transformers): the underlying redactor is plugged in via the
:class:`bba.deid_redactor.RedactorBackend` Protocol. Production wires this
to the vendored ``thai-medical-deid`` package (``TRANSFORMERS_OFFLINE=1``
+ ``HF_HUB_OFFLINE=1`` per PRD §"Stack"); tests use deterministic stubs.
"""

from bba.deid_redactor.age import apply_age_cap
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
    shift_dates_in_text,
)
from bba.deid_redactor.exceptions import (
    BackendRedactionError,
    DateShiftError,
    DeidRedactorError,
    HashMismatchError,
)
from bba.deid_redactor.k_anonymity import (
    compute_k_groups,
    k_anonymity_passed,
)
from bba.deid_redactor.models import (
    AGE_CAP,
    BackendRedactionResult,
    K_ANONYMITY_MIN,
    KAnonymityGate,
    NeedsReviewReason,
    NoteInput,
    PERSON_CLASS_TOKENS,
    QuasiIdentifiers,
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
    classify_role_by_cues,
    default_role_classifier,
    extract_context,
    upgrade_person_tokens,
)
from bba.deid_redactor.semantic import detect_semantic_degradation

__all__ = [
    "AGE_CAP",
    "ATTENDING_CUES",
    "BackendRedactionError",
    "BackendRedactionResult",
    "DATE_PATTERNS",
    "DateMatch",
    "DateShiftError",
    "DeidRedactorError",
    "FAMILY_CUES",
    "HashMismatchError",
    "K_ANONYMITY_MIN",
    "KAnonymityGate",
    "NURSE_CUES",
    "NeedsReviewReason",
    "NoteInput",
    "PATIENT_CUES",
    "PERSON_CLASS_TOKENS",
    "QuasiIdentifiers",
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
    "apply_age_cap",
    "build_envelope",
    "canonical_serialize",
    "classify_role_by_cues",
    "compute_k_groups",
    "compute_redaction_hash",
    "default_role_classifier",
    "detect_semantic_degradation",
    "extract_context",
    "format_offset",
    "k_anonymity_passed",
    "parse_dates",
    "redact_bundle",
    "shift_dates_in_text",
    "upgrade_person_tokens",
]
