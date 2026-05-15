"""Canonical-JSON + hash machinery for :class:`RedactionResult`.

Mirrors :mod:`bba.evidence_bundle_builder.canonical`: sorted keys, NFC
strings (recursively), 2-space indent, no trailing newline. The hash is
``sha256(canonical_json.encode("utf-8")).hexdigest()`` and underwrites
the issue #17 AC bundle-hash stability: same input + same redactor
version → byte-identical canonical JSON → same hash.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


def _nfc_recursive(value: Any) -> Any:
    """NFC-normalize every string reachable from ``value``.

    NFC on both keys and values — without it, a note containing Thai NFD
    characters would hash differently on two runs whose only difference
    is the locale-specific normalization of the source CSV.
    """
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        return {
            (_nfc_recursive(k) if isinstance(k, str) else k): _nfc_recursive(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_nfc_recursive(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"non-finite float ({value!r}) is not valid JSON per RFC 7159"
            )
    return value


def canonical_serialize(value: Any) -> str:
    """Serialize ``value`` to canonical JSON.

    Output contract:

    * UTF-8 encoded
    * NFC-normalized strings at every nesting level
    * Sorted keys at every mapping level
    * 2-space indent
    * No trailing newline
    * Rejects non-finite floats (``NaN``, ``±Inf``)
    """
    normalized = _nfc_recursive(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ": "),
    )


def compute_redaction_hash(envelope: Mapping[str, Any]) -> str:
    """SHA-256 of :func:`canonical_serialize`'s UTF-8 bytes.

    Returns 64-char lowercase hex.
    """
    canonical = canonical_serialize(envelope)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    """Assemble the canonical envelope hashed for bundle-hash stability."""
    return {
        "notes": [dict(n) for n in notes],
        "redactor_version": dict(redactor_version),
        "redacted_age": int(redacted_age),
        "age_capped": bool(age_capped),
        "k_anonymity_size": int(k_anonymity_size),
        "k_anonymity_passed": bool(k_anonymity_passed),
        "route_to_needs_review": bool(route_to_needs_review),
        "needs_review_reasons": list(needs_review_reasons),
    }
