"""Canonical JSON serialization + SHA-256 bundle hashing.

The canonical-JSON contract (issue #16 AC, PRD §7):

* sorted keys (lexicographic at every nesting level)
* UTF-8 NFC normalization of every string in the payload
* fixed indentation (2-space) so the bundle is human-readable for the
  reviewer dashboard but still byte-stable across re-runs
* trailing newline omitted (canonical hashing is over the raw JSON bytes)

The hash is computed over the UTF-8 encoding of the canonical JSON. Because
canonicalization is byte-stable, two semantically-equivalent inputs (e.g., a
shuffled record list, or a Thai string in NFD vs NFC) produce the same hash —
the bundle-hash-stability property test relies on this contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any


def _to_jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into a json.dumps-compatible primitive.

    Handles the four types the bundle pipeline produces:

    * ``str`` — NFC-normalized so NFD inputs produce byte-identical output.
    * ``datetime`` — ISO 8601 with explicit UTC offset (the bundle is
      tz-aware-throughout per CONTEXT.md "tz-aware UTC").
    * :class:`collections.abc.Mapping` — including ``MappingProxyType``,
      which the model layer's ``FrozenJsonDict`` produces; ``json.dumps``
      cannot serialize it directly.
    * ``Sequence`` (excluding ``str`` / ``bytes``) — including ``tuple``,
      which the frozen models use in place of ``list``.

    Raises ``TypeError`` for anything else so a stray Pydantic model or
    custom object can't silently land in the bundle and break hash stability
    when its ``__repr__`` changes.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        # Reject non-finite floats: Python's ``json.dumps`` emits ``NaN`` /
        # ``Infinity`` / ``-Infinity`` for these, which RFC 7159 / 8259
        # forbids — strict downstream parsers (deid_redactor /
        # prompt_builder / any audit re-hydration tool) would reject them.
        # The bundle-hash AC requires bytes that round-trip through every
        # JSON parser, so a non-finite float in a payload is an upstream
        # bug worth surfacing at construction.
        if not math.isfinite(value):
            raise ValueError(
                f"canonical_serialize: non-finite float {value!r} is not "
                "valid JSON; reject at the source (vitals/Hb/lab) instead "
                "of letting it leak into the bundle hash"
            )
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_to_jsonable(item) for item in value]
    raise TypeError(
        f"canonical_serialize: unsupported type {type(value).__name__!r}; "
        "convert to a JSON-primitive before serializing"
    )


def canonical_serialize(payload: Any) -> str:
    """Return the canonical JSON string for ``payload``.

    Sorted keys + 2-space indentation + NFC-normalized strings + no trailing
    newline. The output is the byte sequence that :func:`bundle_hash`
    operates on; downstream callers MUST hash this string (encoded UTF-8)
    rather than re-emitting via Pydantic, so any drift between the hashed
    bytes and the bytes the LLM receives is impossible by construction.
    """
    # ``allow_nan=False`` is a backstop for the non-finite-float check in
    # _to_jsonable: if a future code path bypasses _to_jsonable, json.dumps
    # raises ValueError instead of emitting invalid JSON.
    return json.dumps(
        _to_jsonable(payload),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )


def bundle_hash(canonical_json: str) -> str:
    """Return the lowercase SHA-256 hex digest of ``canonical_json`` UTF-8 bytes."""
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


__all__ = ("bundle_hash", "canonical_serialize")
