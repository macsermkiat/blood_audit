"""Errors raised by :mod:`bba.audit_store`.

PRD §10 makes one invariant load-bearing: ``audit_results`` is the *commit
marker*. An ``audit_results`` row without a matching ``llm_calls`` row is a bug
— it means the deterministic / LLM step that produced the classification was
not persisted, so the row is no longer reproducible. The opposite (``llm_calls``
without a matching ``audit_results``) is a crash mid-write and is handled by
``reconcile``; it is *not* an error condition.
"""

from __future__ import annotations


class TransactionalOrderingError(Exception):
    """Raised by :meth:`AuditStore.validate_invariants` when the commit-marker
    invariant is violated: at least one ``audit_results`` row has no matching
    ``llm_calls`` row.

    The error message names the offending ``audit_id``\\s so the operator can
    quarantine them without re-running with extra logging.
    """
