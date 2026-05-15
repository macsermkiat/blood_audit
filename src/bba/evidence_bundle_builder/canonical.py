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

from typing import Any


def canonical_serialize(payload: Any) -> str:
    """Return the canonical JSON string for ``payload``.

    The returned string is the byte sequence the bundle hash is computed
    over (after UTF-8 encoding). Implementation MUST:

    * sort object keys lexicographically at every nesting level
    * NFC-normalize every string value (recursively, including nested dicts /
      lists / tuples) — without this, a Thai diagnosis string in NFD vs NFC
      would produce two different hashes for the "same" bundle
    * use 2-space indentation and ``", "`` / ``": "`` separators (i.e., the
      ``json.dumps`` defaults — fixed at construction so a future caller
      cannot accidentally pass ``separators=(",", ":")`` and break stability)
    * NOT emit a trailing newline (the hash is over the raw JSON, not a file)

    Raises ``TypeError`` if ``payload`` contains a non-JSON-encodable type.
    """
    raise NotImplementedError("canonical_serialize: not implemented in RED phase")


def bundle_hash(canonical_json: str) -> str:
    """Return the lowercase SHA-256 hex digest of ``canonical_json`` UTF-8 bytes.

    Always operates on the *output* of :func:`canonical_serialize` — never
    on a Pydantic ``model_dump_json()`` or ad-hoc ``json.dumps``. Hashing
    the canonical bytes (not a re-emit) is what makes
    :data:`EvidenceBundle.bundle_hash` reproducible across pipeline stages.
    """
    raise NotImplementedError("bundle_hash: not implemented in RED phase")


__all__ = ("bundle_hash", "canonical_serialize")
