"""Stable per-row identity for the audit_orders canonical table.

``audit_id`` is the immutable identity of a single audited order. It must be
stable across re-runs given the same ``(hn, reqno)`` so downstream stages
(#9 quote-grounding, #19 audit_store idempotency, #24 audit_pipeline retries)
can re-derive the same row without mutating prior outputs.

The audit_id is a sha256-derived hex digest of the canonical
``"<hn>:<reqno>"`` payload, truncated to 32 hex chars. The truncation is
safe because ``(hn, reqno)`` is already unique by construction (PRD §2's
"one row per (HN, REQNO)"); the hash is for stable filesystem-safe naming,
not for collision resistance against an adversary.

Filesystem-safety of the digest is enforced at the downstream
:class:`bba.audit_store.models.SafeId` validator, but the format here
already satisfies the ``[A-Za-z0-9._-]+`` allow-list (lowercase hex chars
only).
"""

from __future__ import annotations


def build_audit_id(hn: str, reqno: str) -> str:
    """Return the stable audit_id for ``(hn, reqno)``.

    Same input → same output, byte-for-byte, forever. Different ``hn`` or
    different ``reqno`` → a different audit_id. The function is the single
    source of truth for the canonical identity formula; tests assert
    determinism and pairwise disjointness, not the exact algorithm.
    """
    raise NotImplementedError


__all__ = ("build_audit_id",)
