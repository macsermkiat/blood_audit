"""Canonical-JSON + hash machinery for :class:`RedactionResult`.

Mirrors :mod:`bba.evidence_bundle_builder.canonical`: sorted keys, NFC
strings (recursively), 2-space indent, no trailing newline. The hash is
``sha256(canonical_json.encode("utf-8")).hexdigest()`` and underwrites
the issue #17 AC bundle-hash stability: same input + same redactor
version → byte-identical canonical JSON → same hash.

Why a separate canonical layer instead of reusing the evidence bundle's
canonical_serialize? The redacted-bundle hash and the pre-redaction
:class:`bba.evidence_bundle_builder.EvidenceBundle.bundle_hash` are two
distinct fields on :class:`bba.audit_store.AuditRow`: the pre-redaction
hash anchors the evidence-bundle audit chain (#16), the post-redaction
hash anchors the redactor audit chain (#17 — "Bundle-hash stability:
same input bundle + same redactor version → same redacted output → same
hash"). Keeping the two serializers structurally separate prevents a
refactor on one from silently changing the other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _nfc_recursive(value: Any) -> Any:
    """NFC-normalize every string reachable from ``value``.

    Mirrors :func:`bba.evidence_bundle_builder.canonical._nfc_recursive`.
    NFC on both keys and values — without it, a note containing Thai NFD
    characters would hash differently on two runs whose only difference
    is the locale-specific normalization of the source CSV.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #17")


def canonical_serialize(value: Any) -> str:
    """Serialize ``value`` to canonical JSON.

    Output contract:

    * UTF-8 encoded
    * NFC-normalized strings at every nesting level
    * Sorted keys at every mapping level
    * 2-space indent
    * No trailing newline
    * Rejects non-finite floats (``NaN``, ``±Inf``) — not valid JSON
      per RFC 7159 and would silently break the bundle-hash invariant
      if a buggy upstream allowed one to leak through.

    Pure function. Same input → byte-identical output across Python
    runs, OSes, and process invocations.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #17")


def compute_redaction_hash(envelope: Mapping[str, Any]) -> str:
    """SHA-256 of :func:`canonical_serialize`'s UTF-8 bytes.

    ``envelope`` is the canonical view of the
    :class:`bba.deid_redactor.models.RedactionResult` — see
    :func:`build_envelope` for its exact shape.

    Returns 64-char lowercase hex. The
    :class:`bba.deid_redactor.models.RedactionResult` model validator
    asserts the recorded hash matches this function's output, mirroring
    the :class:`bba.evidence_bundle_builder.EvidenceBundle` invariant.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #17")


def build_envelope(
    *,
    notes: Sequence[Mapping[str, Any]],
    redactor_version: Mapping[str, str],
    redacted_age: int,
    age_capped: bool,
    k_anonymity_size: int,
    k_anonymity_passed: bool,
    route_to_needs_review: bool,
    needs_review_reasons: Sequence[str],
) -> Mapping[str, Any]:
    """Assemble the canonical envelope hashed for bundle-hash stability.

    Envelope shape (locked — extras would change the hash silently):

    ``{
        "notes": [{"note_id", "redacted_text", "semantic_degraded"}, ...],
        "redactor_version": {"version", "model_sha", "gazetteer_version"},
        "redacted_age": int,
        "age_capped": bool,
        "k_anonymity_size": int,
        "k_anonymity_passed": bool,
        "route_to_needs_review": bool,
        "needs_review_reasons": ["..."],
    }``

    ``notes`` order MUST match :attr:`RedactionResult.notes` order so the
    hash is order-sensitive — preserves the "same input → same output"
    contract end-to-end even when an upstream caller reorders notes
    between runs.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #17")
