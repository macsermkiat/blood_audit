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

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, cast

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from bba.vitals_extractor.bounds import (
    BT_MAX,
    BT_MIN,
    DBP_MAX,
    DBP_MIN,
    HR_MAX,
    HR_MIN,
    RR_MAX,
    RR_MIN,
    SBP_MAX,
    SBP_MIN,
)
from bba.vitals_extractor.models import PeriopSummary


# =============================================================================
# Source enumeration
# =============================================================================

EvidenceSource = Literal[
    "Hemodynamic",
    "Periop",
    "Diagnosis",
    "IPDADMPROGRESS",
    "IPDNRFOCUSDT",
    "Med",
    "Lab",
    "Vitals",
]
"""The per-source families a bundle can contain. The literal order matches the
canonical emission order used to assign stable evidence IDs (E1, E2, ...) in
:func:`bba.evidence_bundle_builder.builder.build_evidence_bundle`.

``Hemodynamic`` (issue #76) is FIRST: it is a single pinned, fact-only summary
(MAP nadir + vasopressor mentions) synthesized from the in-window narrative,
emitted as E1, and exempt from char-cap truncation so the evidence starved in
Case 2 / REQNO 68012352 always reaches the LLM. It carries no appropriateness
language and never gates the deterministic classifier — it is supporting
evidence only.

``Periop`` (Case 107 / REQNO 68074627) is SECOND, the same pinned, fact-only,
truncation-exempt shape: surgical context, EBL (mL), and intra-op transfusion
recovered from the free-text narrative. Case 107's LLM returned
INSUFFICIENT_EVIDENCE because the structured procedure rows were empty and it
trusted that absence over a post-op nursing note already in the bundle — a model
attention miss. Pinning the signal high makes it un-skippable. Like Hemodynamic
it carries no appropriateness language and never gates the classifier (whose
procedure bypass keys on structured timing, not on this scan)."""


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
    # _deep_freeze returns Any (its recursion fans out across many branches);
    # the cast asserts what the function name guarantees: the top-level call
    # on a Mapping always returns a Mapping.
    return cast("Mapping[str, Any]", _deep_freeze(value))


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

    ``hb_anchor`` is the Hb-lookup anchor when it differs from
    ``order_datetime``. The shared resolver (:func:`bba.hb_lookup.
    resolve_hb_with_fallback`) can anchor the Hb on a post-order draw (a lab
    drawn minutes after REQTIME — see ``docs/handoff-hb-anchor-unification``).
    When that happens the triggering Hb is *after* ``order_datetime``, so the
    default ``h.timestamp <= order_datetime`` Hb window would drop the very
    value that routed the case to the LLM. Set ``hb_anchor`` to that draw's
    timestamp so the bundle's Hb upper bound includes it; leave ``None`` for
    the order-time path (the common case) to keep the original window.

    ``window_anchor`` is the point every per-source window (progress, focus,
    meds, Hb, vitals) is centered on. It defaults to ``order_datetime``. Blood
    reserved for elective surgery is crossmatched days before it is
    issued/transfused; for those orders the caller sets ``window_anchor`` to
    the transfusion datetime (see :func:`bba.hb_lookup.resolve_evidence_anchor`)
    so the bundle captures the op-day evidence instead of the reservation-day
    window. ``order_datetime`` stays the reservation REQTIME for audit identity,
    so ``window_anchor`` is windowing-only and is not echoed into the hashed
    bundle envelope.
    """

    model_config = ConfigDict(frozen=True)

    order_datetime: UTCDatetime
    hn_hash: str
    an_hash: str
    products: tuple[str, ...]
    hb_anchor: UTCDatetime | None = None
    window_anchor: UTCDatetime | None = None


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
    # ``item_no`` mirrors :class:`bba.hb_lookup.HbObservation.item_no`: the
    # Lab table's row identifier. Higher values are later inserts /
    # corrections. The deterministic classifier breaks same-(source,
    # timestamp) ties by max ``item_no`` (the corrected result); the
    # bundle uses the same key in ``_hb_sort_key`` so what the LLM sees
    # matches what the classifier used.
    item_no: int


class VitalsRecord(BaseModel):
    """A pre-extracted vital-signs snapshot for the bundle.

    Fed by :mod:`bba.vitals_extractor`. ``source`` echoes
    :class:`bba.vitals_extractor.SourceProvenance` minus ``NONE_IN_WINDOW``
    (the bundle builder simply omits the vitals item in that case)."""

    model_config = ConfigDict(frozen=True)

    timestamp: AwareDatetime
    source: VitalsNoteSource
    # Field constraints mirror :mod:`bba.vitals_extractor.bounds`. The
    # extractor already rejects out-of-range values at extraction; the
    # bundle re-applies the same gate so a buggy caller cannot persist
    # clinically impossible vitals (sbp=-1, hr=999) as canonical evidence.
    # HbRecord follows the same upstream-bounds-mirroring pattern.
    sbp: int | None = Field(default=None, ge=SBP_MIN, le=SBP_MAX)
    dbp: int | None = Field(default=None, ge=DBP_MIN, le=DBP_MAX)
    hr: int | None = Field(default=None, ge=HR_MIN, le=HR_MAX)
    rr: int | None = Field(default=None, ge=RR_MIN, le=RR_MAX)
    # ``allow_inf_nan=False`` rejects NaN / +/-Infinity at construction so a
    # buggy upstream extractor cannot leak a non-finite float into the
    # bundle, where the canonical-JSON serializer would otherwise raise mid-
    # pipeline (still safe, but with a less-targeted error).
    bt: float | None = Field(default=None, allow_inf_nan=False, ge=BT_MIN, le=BT_MAX)


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
    # UTCDatetime | None — the field name promises UTC and the bundle's
    # canonical-JSON contract requires every persisted timestamp to be
    # tz-aware. Without the validator, a public caller could construct an
    # item with ``datetime(2026, 5, 15, 12, 0)`` and the canonical
    # serializer would emit it without an offset, breaking replay
    # comparability across time zones (CONTEXT.md "tz-aware UTC").
    timestamp_utc: UTCDatetime | None
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
        # _deep_thaw returns Any (recursive across mapping/sequence/scalar);
        # the cast asserts what the call shape guarantees: a Mapping in →
        # a dict out.
        return cast("dict[str, Any]", _deep_thaw(value))


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

    # Deterministic peri-op signal scanned from the SAME shipped note set
    # that becomes the bundle's items (Case 107). This is a convenience
    # return handle for downstream deterministic guardrails (e.g. the
    # replay contradiction check) — it is NOT serialized into
    # ``canonical_json`` and therefore does NOT participate in
    # ``bundle_hash``. A bundle reconstructed from stored bytes carries the
    # default (None); the guardrail then simply has no signal to act on.
    periop_summary: PeriopSummary | None = None

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
            raise ValueError(f"bundle_hash must be lowercase hex (got {v!r})")
        return v

    @model_validator(mode="after")
    def _hash_must_match_canonical_json(self) -> Self:
        # Audit-chain invariant: bundle_hash, canonical_json, and items
        # must all describe the same bundle. Round 9 closed the
        # hash-vs-canonical_json gap; round 10 extends the check to
        # canonical_json structure and items consistency, since a
        # downstream rebuilder could otherwise pair a self-consistent
        # hash with bytes that are not a valid evidence bundle envelope.
        # Layered checks (cheapest first):
        #   1. hash matches canonical_json bytes
        #   2. canonical_json parses as JSON
        #   3. canonical_json IS canonical (re-emit equals input)
        #   4. parsed envelope shape has the expected "items" array
        #   5. parsed item IDs match self.items IDs (in order)
        expected_hash = hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()
        if self.bundle_hash != expected_hash:
            raise ValueError(
                f"bundle_hash ({self.bundle_hash}) does not match "
                f"sha256(canonical_json) ({expected_hash}); construct via "
                "build_evidence_bundle() to maintain the audit-chain invariant"
            )

        try:
            parsed = json.loads(self.canonical_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"canonical_json is not valid JSON: {exc.msg}") from exc

        # Lazy import to avoid any future cycle if canonical.py grows a
        # models import; keeps the validator self-contained.
        from bba.evidence_bundle_builder.canonical import canonical_serialize

        if canonical_serialize(parsed) != self.canonical_json:
            raise ValueError(
                "canonical_json is not in canonical form (sorted keys / "
                "NFC strings / 2-space indent / no trailing newline); "
                "construct via build_evidence_bundle()"
            )

        if not isinstance(parsed, dict):
            raise ValueError("canonical_json must be a JSON object (envelope shape)")
        # Envelope shape lock: builder always emits exactly {anchor, items}.
        # Anything more or less is an upstream bug — extras would be silent
        # data leakage; missing keys would mean the bundle is missing its
        # decision-context anchor and the audit chain becomes ambiguous.
        envelope_keys = set(parsed.keys())
        expected_keys = {"anchor", "items"}
        if envelope_keys != expected_keys:
            extras = envelope_keys - expected_keys
            missing = expected_keys - envelope_keys
            raise ValueError(
                f"canonical_json envelope must have exactly keys "
                f"{sorted(expected_keys)} (extras={sorted(extras)}, "
                f"missing={sorted(missing)})"
            )
        if not isinstance(parsed["anchor"], dict):
            raise ValueError("canonical_json 'anchor' must be a JSON object")
        # Anchor shape lock: builder always emits exactly these four
        # fields. An empty {} or extra-key anchor is silent loss of
        # decision context (an_hash, hn_hash for replay; order_datetime
        # for re-windowing; products for re-classification).
        anchor_obj = parsed["anchor"]
        anchor_keys = set(anchor_obj.keys())
        expected_anchor_keys = {
            "order_datetime",
            "hn_hash",
            "an_hash",
            "products",
        }
        if anchor_keys != expected_anchor_keys:
            extras = anchor_keys - expected_anchor_keys
            missing = expected_anchor_keys - anchor_keys
            raise ValueError(
                f"canonical_json 'anchor' must have exactly keys "
                f"{sorted(expected_anchor_keys)} "
                f"(extras={sorted(extras)}, missing={sorted(missing)})"
            )
        if not isinstance(anchor_obj["order_datetime"], str):
            raise ValueError("'anchor.order_datetime' must be a string")
        # Semantic check: order_datetime must be a real, parseable, tz-aware
        # UTC ISO-8601 string. Otherwise the anchor carries an
        # unreconstructable decision time and the audit-chain replay
        # invariant collapses (re-windowing per-source needs the original
        # anchor moment). Mirrors the project's tz-aware-throughout contract
        # (CONTEXT.md "tz-aware UTC").
        try:
            parsed_dt = datetime.fromisoformat(anchor_obj["order_datetime"])
        except ValueError as exc:
            raise ValueError(
                f"'anchor.order_datetime' must be ISO 8601: {exc}"
            ) from exc
        if parsed_dt.tzinfo is None:
            raise ValueError(
                "'anchor.order_datetime' must be tz-aware "
                "(naive datetimes forbidden by project tz contract)"
            )
        if parsed_dt.utcoffset() != timedelta(0):
            raise ValueError(
                "'anchor.order_datetime' must be UTC "
                f"(got offset {parsed_dt.utcoffset()})"
            )
        if not isinstance(anchor_obj["hn_hash"], str):
            raise ValueError("'anchor.hn_hash' must be a string")
        if not isinstance(anchor_obj["an_hash"], str):
            raise ValueError("'anchor.an_hash' must be a string")
        if not isinstance(anchor_obj["products"], list) or not all(
            isinstance(p, str) for p in anchor_obj["products"]
        ):
            raise ValueError("'anchor.products' must be an array of strings")
        parsed_items = parsed["items"]
        if not isinstance(parsed_items, list):
            raise ValueError("canonical_json 'items' must be an array")

        if len(parsed_items) != len(self.items):
            raise ValueError(
                f"canonical_json items count ({len(parsed_items)}) does not "
                f"match EvidenceBundle.items count ({len(self.items)})"
            )

        # Full structural comparison: re-canonicalize self.items into the
        # same form parsed_items has, then compare. Catches any payload /
        # source / timestamp drift between the two halves of the audit
        # chain, not just ID renames.
        expected_items_canonical = canonical_serialize(
            [
                {
                    "id": it.id,
                    "source": it.source,
                    "timestamp_utc": it.timestamp_utc,
                    "payload": dict(it.payload),
                }
                for it in self.items
            ]
        )
        parsed_items_canonical = canonical_serialize(parsed_items)
        if expected_items_canonical != parsed_items_canonical:
            raise ValueError(
                "canonical_json items disagree with EvidenceBundle.items "
                "(payload, source, timestamp, or ID mismatch); construct "
                "via build_evidence_bundle() to maintain the audit-chain "
                "invariant"
            )

        return self


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
