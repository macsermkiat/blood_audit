"""Pydantic v2 models, enums, and Protocols for :mod:`bba.deid_redactor`.

The module surface mirrors :mod:`bba.quote_grounder`: a Protocol boundary
isolates the wrapper from the heavy ``thai-medical-deid`` dependency so the
post-processing logic (role-token mapping, date-shift, age cap, k-anonymity
gate, semantic-degradation flag, canonical hashing) is unit-testable in
isolation. The real backend is plugged in at the audit-pipeline boundary
(#24); tests supply a stub that returns fixed
:class:`BackendRedactionResult` values.

All public models are frozen — the audit chain (PRD §"Reproducibility")
depends on the redacted bundle being immutable from the moment it crosses
the redactor boundary to the moment its hash lands on
:class:`bba.audit_store.AuditRow.evidence_bundle_hash`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Protocol, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


# =============================================================================
# Role-token vocabulary
# =============================================================================


class RoleToken(StrEnum):
    """The vocabulary of post-processed PHI tokens emitted by the wrapper.

    Issue #17 AC: "Emits role-preserving tokens [ATTENDING] / [NURSE] /
    [PATIENT] / [FAMILY] (post-processing wrapper, not a fork of the
    redactor)." :attr:`PERSON` is the backend's pre-classification token —
    when the role classifier returns ``None`` for a span (no contextual
    cue), the original :attr:`PERSON` is preserved so the audit row can
    still distinguish "redacted but unclassified" from "unredacted".

    :attr:`DATE` / :attr:`LOCATION` / :attr:`HOSPITAL` / :attr:`ID` /
    :attr:`PHONE` are pass-through tokens — they are not re-classified by
    the wrapper, but appear in the token set so the semantic-degradation
    flag (which counts ``[PERSON]``-class tokens) can exclude them.
    """

    PERSON = "[PERSON]"
    ATTENDING = "[ATTENDING]"
    NURSE = "[NURSE]"
    PATIENT = "[PATIENT]"
    FAMILY = "[FAMILY]"
    DATE = "[DATE]"
    LOCATION = "[LOCATION]"
    HOSPITAL = "[HOSPITAL]"
    ID = "[ID]"
    PHONE = "[PHONE]"


PERSON_CLASS_TOKENS: frozenset[RoleToken] = frozenset(
    {
        RoleToken.PERSON,
        RoleToken.ATTENDING,
        RoleToken.NURSE,
        RoleToken.PATIENT,
        RoleToken.FAMILY,
    }
)
"""Tokens counted by the semantic-degradation detector.

PRD §8: "if redacted note has > 4 ``[PERSON]``-class tokens within 50 chars
→ NEEDS_REVIEW". The "class" is the role-bearing token family —
non-personal tokens (``[DATE]``, ``[LOCATION]``, etc.) do not contribute to
the density because their redaction does not erode semantic content.
"""


# =============================================================================
# Backend boundary (Protocol)
# =============================================================================


class RedactionSpan(BaseModel):
    """One PHI span identified by the backend.

    ``start`` / ``end`` are half-open offsets into the ORIGINAL (pre-
    redaction) text. ``entity_type`` is the backend's coarse label
    (e.g. ``"PERSON"``, ``"DATE"``, ``"LOCATION"``); the wrapper maps this
    to a :class:`RoleToken` via the role classifier and falls back to the
    type-matching token when classification yields ``None``.

    ``original_text`` is the original substring the backend redacted; the
    role classifier inspects it (alongside the surrounding ±N chars of
    context) to decide the role. Backends that do not surface the original
    text MUST still set this field — pass an empty string if unavailable
    and the wrapper falls back to context-only classification.
    """

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    entity_type: str
    original_text: str = ""


class BackendRedactionResult(BaseModel):
    """The return shape the wrapper expects from a :class:`RedactorBackend`.

    ``text`` is the redacted text already containing token placeholders
    (e.g. ``"[PERSON]"``, ``"[DATE]"``). ``spans`` is the list of
    redactions in document order (one per token in ``text``); the wrapper
    walks them in order to upgrade ``[PERSON]`` tokens to role-specific
    tokens.

    ``len(spans)`` must equal the number of placeholder tokens in
    ``text``; backends that violate this invariant cause the wrapper to
    raise :class:`bba.deid_redactor.exceptions.BackendRedactionError`.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    spans: tuple[RedactionSpan, ...] = ()


class RedactorBackend(Protocol):
    """Protocol the wrapper consumes for the underlying redactor.

    Implementations: production wraps ``thai-medical-deid`` (pinned in
    pyproject.toml with model SHA + gazetteer version); tests supply a
    deterministic stub returning a fixed :class:`BackendRedactionResult`.
    The wrapper NEVER imports ``thai-medical-deid`` directly — keeps the
    post-processing logic deps-free and testable without the heavy HF
    model load.
    """

    def redact(self, text: str) -> BackendRedactionResult:  # pragma: no cover - protocol
        ...


class RoleClassifier(Protocol):
    """Protocol for the role-inference step.

    Called once per :class:`RedactionSpan` with ``entity_type == "PERSON"``.
    Returns the upgraded :class:`RoleToken` (one of ATTENDING / NURSE /
    PATIENT / FAMILY) when context supports a role assignment, or ``None``
    to leave the span as :attr:`RoleToken.PERSON`.

    Implementations must be PURE: same ``original_text`` + same ``context``
    yields the same role. The wrapper relies on determinism for bundle-hash
    stability (PRD §"Output schema" — ``evidence_bundle_hash`` reproducible).
    """

    def __call__(
        self,
        *,
        original_text: str,
        context: str,
        span: RedactionSpan,
    ) -> RoleToken | None:  # pragma: no cover - protocol
        ...


class KAnonymityGate(Protocol):
    """Protocol the wrapper calls to look up the QI group size.

    Returns the cohort size (number of records sharing the same
    :class:`QuasiIdentifiers` tuple) for this record. The wrapper compares
    against :data:`K_ANONYMITY_MIN` (5) and routes to NEEDS_REVIEW when
    below threshold. The caller — typically the audit-pipeline orchestrator
    (#24) — precomputes the population's QI groups (via
    :func:`bba.deid_redactor.k_anonymity.compute_k_groups`) and adapts it
    to this callable.
    """

    def __call__(self, qi: QuasiIdentifiers) -> int:  # pragma: no cover - protocol
        ...


# =============================================================================
# Quasi-identifier + version metadata
# =============================================================================


_SEX_LITERAL = {"M", "F", "U"}
"""Allowed sex codes. ``U`` (unknown) is required because HOSxP exports
include records where sex is missing or unspecified — rejecting them at
the redactor boundary would silently drop audit-eligible orders."""


def _ensure_admission_month(value: str) -> str:
    """Validate the admission-month format ``YYYY-MM``.

    A free-form string here would silently break k-anonymity grouping:
    ``"2026-05"`` and ``"2026-5"`` would be two distinct groups, halving k.
    """
    parts = value.split("-")
    if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 2:
        raise ValueError(
            f"admission_month must be 'YYYY-MM' (got {value!r}); "
            "deviation breaks k-anonymity group equality"
        )
    if not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(
            f"admission_month must be 'YYYY-MM' digits-only (got {value!r})"
        )
    month_int = int(parts[1])
    if month_int < 1 or month_int > 12:
        raise ValueError(
            f"admission_month month must be 01-12 (got {value!r})"
        )
    return value


AdmissionMonth = Annotated[str, AfterValidator(_ensure_admission_month)]


def _ensure_age_band(value: str) -> str:
    """Validate the age-band format ``LO-HI`` (e.g. ``"60-69"``).

    Mirrors :func:`_ensure_admission_month` — free-form strings would
    fragment k-anonymity groups silently.
    """
    parts = value.split("-")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(
            f"age_band must be 'LO-HI' integers (got {value!r}); "
            "deviation breaks k-anonymity group equality"
        )
    lo, hi = int(parts[0]), int(parts[1])
    if lo < 0 or hi < lo:
        raise ValueError(
            f"age_band must have lo <= hi and lo >= 0 (got {value!r})"
        )
    return value


AgeBand = Annotated[str, AfterValidator(_ensure_age_band)]


class QuasiIdentifiers(BaseModel):
    """The five quasi-identifier fields that define a k-anonymity equivalence class.

    PRD §8: "k-anonymity gate (k ≥ 5 on ``{ward, ICD-3char, age-band, sex,
    admission-month}``)". The model is hashable (frozen + only-primitive
    fields) so it can be used as a ``Mapping`` key in
    :func:`bba.deid_redactor.k_anonymity.compute_k_groups`.

    Field formats are validated so equivalent-but-formatted-differently
    inputs do not silently split a group (e.g. ``"2026-05"`` vs
    ``"2026-5"`` would otherwise halve the effective k).
    """

    model_config = ConfigDict(frozen=True)

    ward: str = Field(min_length=1)
    icd_3char: str = Field(min_length=3, max_length=3)
    age_band: AgeBand
    sex: str
    admission_month: AdmissionMonth

    def __hash__(self) -> int:
        return hash(
            (
                self.ward,
                self.icd_3char,
                self.age_band,
                self.sex,
                self.admission_month,
            )
        )


class RedactorVersion(BaseModel):
    """Version metadata stamped on every redaction result.

    PRD §"Output schema" requires ``redactor_version`` and
    ``redactor_model_sha`` on every audit row so the redacted bundle hash
    can be reproduced six months later. ``gazetteer_version`` covers the
    PyThaiNLP gazetteer (PRD §"Stack" — pinned for offline determinism).
    """

    model_config = ConfigDict(frozen=True)

    version: str = Field(min_length=1)
    model_sha: str = Field(min_length=1)
    gazetteer_version: str = Field(min_length=1)


# =============================================================================
# Input + output containers
# =============================================================================


def _ensure_utc_datetime(dt: datetime) -> datetime:
    """Reject naive datetimes; preserve aware ones.

    Mirrors :mod:`bba.evidence_bundle_builder.models`. Same rationale: a
    naive admission datetime would break the date-shift math when
    compared to tz-aware in-text dates.
    """
    if dt.tzinfo is None:
        raise ValueError(
            "datetime must be tz-aware (see CONTEXT.md 'tz-aware UTC')"
        )
    return dt


UTCDatetime = Annotated[datetime, AfterValidator(_ensure_utc_datetime)]


class NoteInput(BaseModel):
    """One note to redact.

    ``note_id`` is the stable reference (e.g. an evidence-bundle ``E1``
    identifier from :mod:`bba.evidence_bundle_builder`) preserved on the
    output so the audit pipeline can join redacted notes back to the
    bundle's original-source slot.
    """

    model_config = ConfigDict(frozen=True)

    note_id: str = Field(min_length=1)
    text: str


class RedactionRequest(BaseModel):
    """Top-level input to :func:`bba.deid_redactor.redact_bundle`.

    ``admission_date`` anchors the date-shift transform. ``patient_age``
    is the raw years-at-admission; the wrapper applies the age cap (PRD
    §8 — age cap 89) and surfaces the capped value on the result.
    """

    model_config = ConfigDict(frozen=True)

    notes: tuple[NoteInput, ...]
    quasi_identifiers: QuasiIdentifiers
    admission_date: date
    patient_age_years: int = Field(ge=0)
    redactor_version: RedactorVersion


class NeedsReviewReason(StrEnum):
    """Mutually-exclusive routing tags surfaced on the redaction result.

    The audit pipeline (#24) reads :attr:`RedactionResult.needs_review_reasons`
    to set the row-level ``review_reason`` field (PRD §"Output schema").
    Multiple reasons may fire simultaneously — the routing decision is
    OR-of-reasons.
    """

    K_ANONYMITY_FAIL = "k_anonymity_below_5"
    SEMANTIC_DEGRADATION = "person_density_above_threshold"


class RedactedNote(BaseModel):
    """One note after redaction + role mapping + date shift.

    ``redacted_text`` is the final form sent to the LLM: PHI replaced
    with role-bearing tokens, dates rewritten to ``Day N`` offsets from
    admission. ``semantic_degraded`` is the per-note degradation flag
    (PRD §8 — >4 PERSON-class tokens within 50 chars); the result-level
    flag is the OR across all notes.
    """

    model_config = ConfigDict(frozen=True)

    note_id: str
    redacted_text: str
    semantic_degraded: bool


class RedactionResult(BaseModel):
    """Top-level output of :func:`bba.deid_redactor.redact_bundle`.

    Carries the redacted notes and every metadata field the audit row
    needs (PRD §"Output schema"): the redactor version stamps, the
    post-cap age, the k-anonymity decision, the NEEDS_REVIEW routing,
    and the canonical-bytes hash for bundle-hash stability.

    The ``redaction_hash`` is computed by
    :func:`bba.deid_redactor.canonical.compute_redaction_hash` over a
    canonical-JSON envelope of the result. Same input + same redactor
    version → same canonical bytes → same hash (issue #17 AC: bundle-hash
    stability). The model validator recomputes the hash at construction
    and rejects mismatches — mirrors the
    :class:`bba.evidence_bundle_builder.EvidenceBundle` audit-chain
    invariant.
    """

    model_config = ConfigDict(frozen=True)

    notes: tuple[RedactedNote, ...]
    redactor_version: RedactorVersion
    redacted_age: int = Field(ge=0)
    age_capped: bool
    k_anonymity_size: int = Field(ge=0)
    k_anonymity_passed: bool
    route_to_needs_review: bool
    needs_review_reasons: tuple[NeedsReviewReason, ...]
    redaction_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def _hash_must_match_envelope(self) -> Self:
        # Lazy import to avoid a circular import: canonical depends on
        # nothing in models at import time, but importing canonical at
        # module load would create a top-level cycle if canonical ever
        # learned about models (mirrors the evidence_bundle_builder
        # pattern).
        from bba.deid_redactor.canonical import (
            build_envelope,
            compute_redaction_hash,
        )

        if not all(c in "0123456789abcdef" for c in self.redaction_hash):
            raise ValueError(
                f"redaction_hash must be lowercase hex (got {self.redaction_hash!r})"
            )

        envelope = build_envelope(
            notes=[
                {
                    "note_id": n.note_id,
                    "redacted_text": n.redacted_text,
                    "semantic_degraded": n.semantic_degraded,
                }
                for n in self.notes
            ],
            redactor_version={
                "version": self.redactor_version.version,
                "model_sha": self.redactor_version.model_sha,
                "gazetteer_version": self.redactor_version.gazetteer_version,
            },
            redacted_age=self.redacted_age,
            age_capped=self.age_capped,
            k_anonymity_size=self.k_anonymity_size,
            k_anonymity_passed=self.k_anonymity_passed,
            route_to_needs_review=self.route_to_needs_review,
            needs_review_reasons=[r.value for r in self.needs_review_reasons],
        )
        expected = compute_redaction_hash(envelope)
        if self.redaction_hash != expected:
            raise ValueError(
                f"redaction_hash ({self.redaction_hash}) does not match "
                f"sha256(canonical envelope) ({expected}); construct via "
                "redact_bundle() to maintain the audit-chain invariant"
            )
        return self


# =============================================================================
# Module-level constants
# =============================================================================


AGE_CAP: int = 89
"""HIPAA-derived age cap. PRD §8: "age cap at 89" — ages above 89 are
collapsed to 89 to defeat re-identification of the elderly tail."""


K_ANONYMITY_MIN: int = 5
"""Minimum cohort size below which a record routes to NEEDS_REVIEW.

PRD §8: "k-anonymity gate (k ≥ 5 on ...)". The threshold is part of the
contract — changing it changes every routed row, so it is exposed as a
module constant rather than a function default."""


SEMANTIC_WINDOW_CHARS: int = 50
"""Sliding-window width (in NFC characters) over which the
semantic-degradation detector counts PERSON-class tokens. PRD §8."""


SEMANTIC_PERSON_THRESHOLD: int = 4
"""Strict-greater-than threshold: a redacted note with *more than 4*
PERSON-class tokens inside any 50-char window is flagged as semantically
degraded. PRD §8 ("> 4 ``[PERSON]``-class tokens within 50 chars")."""


__all__: Sequence[str] = (
    "AGE_CAP",
    "AdmissionMonth",
    "AgeBand",
    "BackendRedactionResult",
    "K_ANONYMITY_MIN",
    "KAnonymityGate",
    "NeedsReviewReason",
    "NoteInput",
    "PERSON_CLASS_TOKENS",
    "QuasiIdentifiers",
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
    "UTCDatetime",
)
