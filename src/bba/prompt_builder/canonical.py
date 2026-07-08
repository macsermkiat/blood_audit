"""Canonical-JSON + hash machinery for :class:`PromptBuildResult`.

Mirrors :mod:`bba.evidence_bundle_builder.canonical` and
:mod:`bba.deid_redactor.canonical`: sorted keys, NFC strings (recursively),
2-space indent, no trailing newline. The hash is
``sha256(canonical_json.encode("utf-8")).hexdigest()`` and underwrites
issue #21's prompt-cache marker correctness verification: same input ->
byte-identical canonical envelope -> same hash.

Three pieces compose the contract:

* :func:`canonical_serialize` — value -> canonical-JSON string.
* :func:`compute_prompt_hash` — envelope -> 64-char lowercase hex.
* :func:`build_envelope` — keyword-only assembly of the eight fields
  that participate in the audit-chain replay. Adding a field to
  :class:`PromptBuildResult` without adding it here would silently
  weaken the hash, so the result-model validator constructs the
  envelope through this helper.
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

    NFC on both keys and values — without it, Thai NFD-vs-NFC drift in
    source CSVs would change the canonical bytes on two runs of the same
    input. Mirrors :mod:`bba.deid_redactor.canonical`.
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

    Output contract: UTF-8 encoded, NFC-normalized strings at every
    nesting level, sorted keys at every mapping level, 2-space indent,
    no trailing newline, rejects non-finite floats. Mirrors the existing
    module pattern so the audit chain can hash any module's envelope
    with the same algorithm.
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


def compute_prompt_hash(envelope: Mapping[str, Any]) -> str:
    """SHA-256 of :func:`canonical_serialize`'s UTF-8 bytes (64 lowercase hex)."""
    canonical = canonical_serialize(envelope)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_envelope(
    *,
    blocks: Sequence[Mapping[str, Any]],
    task_mode: str,
    cohort_threshold: float | None,
    injection_matches: Sequence[Mapping[str, Any]],
    route_to_needs_review: bool,
    needs_review_reasons: Sequence[str],
) -> Mapping[str, Any]:
    """Assemble the canonical envelope hashed for prompt-hash stability.

    ``cohort_threshold`` is ``None`` for ``PLATELET_REVIEW`` (platelet
    transfusion has no Hb cohort) and a validated float for RBC modes.
    The envelope serialises ``None`` as JSON ``null`` so the platelet
    hash is distinct from any RBC hash regardless of content.

    ``injection_matches`` is a sequence of full match records, one dict
    per :class:`InjectionMatch` with keys ``category``, ``pattern_id``,
    ``evidence_id``, ``span_text``, ``start``, ``end``. The full record
    participates in the hash so a downstream caller cannot swap a match's
    ``evidence_id`` / ``span_text`` / offsets and retain a self-consistent
    ``prompt_hash`` (codex review #21 round 3 P2 — reviewer-visible
    injection evidence must be byte-stable through the audit chain).

    Every field that participates in audit-chain replay appears in the
    envelope; adding a field to :class:`PromptBuildResult` without adding
    it here would silently weaken the hash.
    """
    return {
        "blocks": [dict(b) for b in blocks],
        "task_mode": str(task_mode),
        "cohort_threshold": float(cohort_threshold)
        if cohort_threshold is not None
        else None,
        "injection_matches": [dict(m) for m in injection_matches],
        "route_to_needs_review": bool(route_to_needs_review),
        "needs_review_reasons": list(needs_review_reasons),
    }
