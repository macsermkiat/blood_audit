"""Pydantic v2 models for the evidence_bundle_builder module.

All models are immutable (``frozen=True``). Field validators reject naive
datetimes at construction so a stray local-time leak cannot reach the window
filter in :func:`bba.evidence_bundle_builder.builder.build_evidence_bundle`
(see CONTEXT.md "tz-aware UTC" — the project-wide tz contract).

The public surface intentionally mirrors the existing module conventions:

* Inputs are pre-filtered by patient/encounter (the caller does ``HN`` /
  ``AN`` filtering before constructing :class:`EvidenceInputs`); this module's
  job is windowing + ranking + canonical serialization, not ID matching.
* Output containers use ``tuple`` (not ``list``) so the public output cannot
  be mutated even though Pydantic ``frozen=True`` only protects the field
  binding (see ``IngestResult`` / ``HbLookupResult`` for the same pattern).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)


# =============================================================================
# Source enumeration
# =============================================================================

EvidenceSource = Literal[
    "Diagnosis",
    "IPDADMPROGRESS",
    "IPDNRFOCUSDT",
    "MED",
    "Lab",
    "Vitals",
]
"""The six per-source families a bundle can contain. The literal order matches
the canonical emission order used to assign stable evidence IDs (E1, E2, ...)
in :func:`bba.evidence_bundle_builder.builder.build_evidence_bundle`.
"""


HbSource = Literal["HEMATOLOGY", "POCT"]
"""LABEXM source preference for Hb history items, mirroring
:mod:`bba.hb_lookup` (HEMATOLOGY 290095 > POCT 500001 per PRD §3)."""


VitalsNoteSource = Literal["IPDADMPROGRESS", "IPDNRFOCUSDT", "LLM_extracted"]
"""Where the vitals values originated, mirroring
:class:`bba.vitals_extractor.SourceProvenance`. ``LLM_extracted`` is preserved
in the bundle so the auditor can see when regex fell back to the LLM
(per PRD §4)."""


SOAPSection = Literal["ASSESSMENT", "PLAN", "OBJECTIVE", "SUBJECTIVE"]
"""SOAP section labels in priority-preservation order.

Issue #16 AC: "ASSESSMENT + PLAN first, OBJECTIVE next, SUBJECTIVE last".
Drop-on-truncation order is the reverse: SUBJECTIVE first, then OBJECTIVE,
then PLAN, then ASSESSMENT (last to drop because it carries the clinician's
diagnosis-time interpretation — losing it changes the audit's meaning)."""


# =============================================================================
# Helpers — frozen-JSON containers (mirrors audit_store.models)
# =============================================================================


def _deep_freeze(value: Any) -> Any:
    """Recursively convert ``value`` into a structurally-immutable form.

    Mirrors :func:`bba.audit_store.models._deep_freeze`. The bundle payload
    travels with downstream modules (deid_redactor, prompt_builder, llm_client)
    and a mutable nested dict could be edited in-flight, breaking the
    bundle-hash contract: same input → same hash. Freezing at the model
    boundary makes that violation impossible.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, str | bytes):
        return value
    if isinstance(value, Sequence):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    """Inverse of :func:`_deep_freeze`: rebuild plain ``dict`` / ``list``.

    Used by field serializers so JSON output is plain Python containers.
    The canonical-JSON serializer needs plain ``dict`` because ``json.dumps``
    cannot natively serialize :class:`MappingProxyType`."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {k: _deep_thaw(v) for k, v in value.items()}
    if isinstance(value, str | bytes):
        return value
    if isinstance(value, Sequence):
        return [_deep_thaw(item) for item in value]
    return value


def _freeze_dict(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _deep_freeze(value)


FrozenJsonDict = Annotated[dict[str, Any], AfterValidator(_freeze_dict)]
"""A deeply-immutable JSON-shaped mapping used for :class:`EvidenceItem`
payloads. Mirrors :data:`bba.audit_store.models.FrozenJsonDict`."""


def _ensure_utc(dt: datetime) -> datetime:
    """Reject naive datetimes; normalize aware non-UTC datetimes to UTC.

    The project-wide tz contract (CONTEXT.md "tz-aware UTC") is asserted at
    the model boundary: if a naive datetime slipped past the ingest layer
    into a downstream caller's evidence inputs, window filtering would later
    raise an opaque ``TypeError: can't compare offset-naive and offset-aware
    datetimes``. Failing loud at construction names the offending field
    instead.
    """
    if dt.tzinfo is None:
        raise ValueError(
            "datetime must be tz-aware; naive datetimes are forbidden in the "
            "evidence_bundle_builder (see CONTEXT.md 'tz-aware UTC')"
        )
    return dt.astimezone(UTC)


UTCDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]
"""A ``datetime`` constrained to tz-aware UTC at validation time."""


# =============================================================================
# Anchor + per-source input records
# =============================================================================


class OrderAnchor(BaseModel):
    """The order context: when, for which encounter, and which products.

    ``order_datetime`` is the point all per-source windows are computed
    relative to. ``products`` is a tuple (not a list) so the public input is
    immutable. ``hn_hash`` / ``an_hash`` are present for echoing back into
    the bundle's JSON so the audit chain is reconstructible — the bundle
    builder does NOT use them to filter records (the caller pre-filters,
    mirroring :mod:`bba.hb_lookup`).
    """

    model_config = ConfigDict(frozen=True)

    order_datetime: UTCDatetime
    hn_hash: str
    an_hash: str
    products: tuple[str, ...]


class DiagnosisRecord(BaseModel):
    """One ICD-10 diagnosis for the order's encounter (AN-scoped).

    No timestamp: per PRD §7 diagnoses are AN-scoped, not time-windowed —
    the full ICD-10 list for the encounter is included regardless of when
    each code was added. ``description`` is optional human-readable context
    for the LLM (HOSxP exports raw codes only)."""

    model_config = ConfigDict(frozen=True)

    icd10: str
    description: str | None = None


class ProgressNote(BaseModel):
    """One IPDADMPROGRESS row.

    Per the v1 ingest schema (:mod:`bba.ingest.schemas`) the ``OBJECTIVE``
    column actually holds the full SOAP note text — the column name is a
    HOSxP misnomer. ``text`` is that full SOAP note; section parsing
    (Subjective / Objective / Assessment / Plan) happens inside the bundle
    builder via :func:`bba.evidence_bundle_builder.ranking.parse_soap_sections`.
    """

    model_config = ConfigDict(frozen=True)

    timestamp: AwareDatetime
    text: str


class FocusNote(BaseModel):
    """One IPDNRFOCUSDT (nursing focus) row.

    Per PRD §7 these are time-window-ranked with a 5-before / 5-after split
    around the order anchor. ``text`` is the concatenated FOCUS / ACTION /
    RESPONSE field (the v1 schema only declares a single ``FOCUS`` column,
    but the bundle builder treats it as opaque free text)."""

    model_config = ConfigDict(frozen=True)

    timestamp: AwareDatetime
    text: str


class MedRecord(BaseModel):
    """One MED row (drug administered)."""

    model_config = ConfigDict(frozen=True)

    timestamp: AwareDatetime
    drug: str


class HbRecord(BaseModel):
    """One Hb result for the patient (PRD §7 Hb history: -7 d window).

    ``value_g_dl`` is constrained to the analytic-validity window [2, 25]
    g/dL for parity with :class:`bba.hb_lookup.HbObservation` — out-of-range
    values are transcription errors, not real measurements, and including
    them in the bundle would mislead the LLM."""

    model_config = ConfigDict(frozen=True)

    timestamp: AwareDatetime
    value_g_dl: float = Field(ge=2.0, le=25.0)
    source: HbSource


class VitalsRecord(BaseModel):
    """A pre-extracted vital-signs snapshot for the bundle.

    Fed by :mod:`bba.vitals_extractor`. ``source`` echoes
    :class:`bba.vitals_extractor.SourceProvenance` minus ``NONE_IN_WINDOW``
    (the bundle builder simply omits the vitals item in that case)."""

    model_config = ConfigDict(frozen=True)

    timestamp: AwareDatetime
    source: VitalsNoteSource
    sbp: int | None = None
    dbp: int | None = None
    hr: int | None = None
    rr: int | None = None
    bt: float | None = None


class EvidenceInputs(BaseModel):
    """The full set of per-source inputs for a single bundle build.

    Caller responsibility (mirrors :mod:`bba.hb_lookup`): pre-filter every
    sequence to the relevant patient (HN-scoped) or encounter (AN-scoped
    for ``diagnoses``). The bundle builder applies the per-source time
    windows itself but does not look at HN/AN."""

    model_config = ConfigDict(frozen=True)

    anchor: OrderAnchor
    diagnoses: tuple[DiagnosisRecord, ...] = ()
    progress_notes: tuple[ProgressNote, ...] = ()
    focus_notes: tuple[FocusNote, ...] = ()
    meds: tuple[MedRecord, ...] = ()
    hb_history: tuple[HbRecord, ...] = ()
    vitals: tuple[VitalsRecord, ...] = ()


# =============================================================================
# Output: EvidenceItem + EvidenceBundle
# =============================================================================


class EvidenceItem(BaseModel):
    """One item in the bundle.

    The ``id`` is the stable evidence reference (E1, E2, ..., EN) that the
    LLM will cite in :mod:`bba.quote_grounder`. IDs are assigned by the
    builder in canonical emission order so they are reproducible across
    re-runs of the same input.

    ``timestamp_utc`` is ``None`` for AN-scoped diagnoses (no time anchor).
    ``payload`` is the source-specific shape; using a frozen dict instead
    of a typed union keeps the bundle JSON open to phase-2 source families
    without an enum churn here."""

    model_config = ConfigDict(frozen=True)

    id: str
    source: EvidenceSource
    timestamp_utc: datetime | None
    payload: FrozenJsonDict

    @field_validator("id")
    @classmethod
    def _id_must_be_E_prefixed(cls, v: str) -> str:
        # Stable-IDs AC: the LLM and quote_grounder both pattern-match on the
        # E-prefix; admitting an arbitrary string here would make a malformed
        # ID a runtime surprise downstream instead of a construction error.
        if not v.startswith("E"):
            raise ValueError(f"evidence id must start with 'E' (got {v!r})")
        if not v[1:].isdigit():
            raise ValueError(
                f"evidence id must be 'E' + digits (got {v!r}); "
                "fractional or alpha suffixes break stable referencing"
            )
        return v

    @field_serializer("payload")
    def _serialize_payload(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return _deep_thaw(value)


class EvidenceBundle(BaseModel):
    """The final structured evidence bundle.

    ``canonical_json`` is the byte-stable JSON that ``bundle_hash`` is the
    SHA-256 of. Both fields are emitted by
    :func:`bba.evidence_bundle_builder.builder.build_evidence_bundle`; the
    hash is computed over the canonical JSON itself, NOT over a re-emitted
    Pydantic dump (which would risk subtle key-ordering or whitespace drift
    between the hashed bytes and the bytes the LLM actually receives).

    The bundle is the input to :mod:`bba.deid_redactor` (#17) and the
    persisted ``evidence_bundle_hash`` lands on every :class:`AuditRow` for
    reproducibility."""

    model_config = ConfigDict(frozen=True)

    items: tuple[EvidenceItem, ...]
    canonical_json: str
    bundle_hash: str

    @field_validator("bundle_hash")
    @classmethod
    def _hash_must_be_sha256_hex(cls, v: str) -> str:
        # Round 1 invariant: anything other than 64 lowercase hex chars is
        # a misuse — either a wrong digest size (sha1 is 40, sha512 is 128)
        # or a non-hex prefix from a typo. Reject at construction instead of
        # silently storing it on AuditRow.evidence_bundle_hash.
        if len(v) != 64:
            raise ValueError(
                f"bundle_hash must be a 64-char sha256 hex (got {len(v)} chars)"
            )
        if not all(c in "0123456789abcdef" for c in v):
            raise ValueError(
                f"bundle_hash must be lowercase hex (got {v!r})"
            )
        return v


__all__: Sequence[str] = (
    "DiagnosisRecord",
    "EvidenceBundle",
    "EvidenceInputs",
    "EvidenceItem",
    "EvidenceSource",
    "FocusNote",
    "FrozenJsonDict",
    "HbRecord",
    "HbSource",
    "MedRecord",
    "OrderAnchor",
    "ProgressNote",
    "SOAPSection",
    "UTCDatetime",
    "VitalsNoteSource",
    "VitalsRecord",
)
