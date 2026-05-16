"""Idempotent application of an Anthropic batch result set (issue #24).

User constraint #7 — replay idempotency property test:

    Apply the same Anthropic batch result set twice to bba.audit_store
    via the pipeline. Second application must be a no-op (zero new
    rows, zero updates).

This module isolates the "results in → audit_store rows out" function
so the property test can call it directly without re-running the full
LLM pipeline. The audit_store's own idempotency contract
(``WriteResult.skipped_idempotent``) is the load-bearing primitive;
this layer composes it with the verifier + winning-attempt rule and
returns a per-call summary so the test can assert "zero new rows" on
the second pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from bba.audit_pipeline.models import PipelineRunResult
from bba.audit_store import AuditStore
from bba.llm_client.models import RawBatchResponse


def apply_batch_results(
    response: RawBatchResponse,
    *,
    audit_store: AuditStore,
    run_id: str,
) -> PipelineRunResult:
    """Apply a single :class:`RawBatchResponse` to the audit_store.

    For each ``BatchSubmissionResult`` in ``response.results``:

    1. Parse the structured-output payload (fail-closed; parse failures
       route to ``NEEDS_REVIEW``).
    2. Run the quote_grounder verifier on every Tier-1 citation.
    3. Pick the winning attempt per user constraint #6: "last
       verifier-passed attempt wins; if none passes, classification =
       NEEDS_REVIEW with hallucination_suspect flag".
    4. Write the resulting :class:`bba.audit_store.AuditRow` +
       :class:`bba.audit_store.LlmCall` rows through
       :meth:`AuditStore.write` — which is itself idempotent on
       ``(audit_id, run_id, code_version)`` (PRD §10).

    Returns a :class:`PipelineRunResult` whose ``audit_ids_persisted``
    tuple lets the property test count "rows written on this call".
    A second call with the same ``response`` MUST return an empty
    ``audit_ids_persisted`` (zero new rows, zero updates).

    The implementation lives in GREEN (issue #24).
    """
    _ = (response, audit_store, run_id)
    raise NotImplementedError("RED-phase scaffold; see issue #24")


def select_winning_attempt(
    calls: Sequence[object],
) -> object | None:
    """Pick the winning attempt per user constraint #6.

    Winning attempt = the one whose ``verifier_pass=True`` AND has the
    latest ``attempt_id`` (last verifier-passed wins). Returns ``None``
    when no attempt passed verifier — caller routes that to
    ``NEEDS_REVIEW`` with ``hallucination_suspect=True``.

    Callers pass a sequence of mapping-shaped records (test fixtures
    use plain ``dict`` with ``attempt_id`` + ``verifier_pass`` keys;
    production wires the verified-call tuple emitted by
    :mod:`bba.quote_grounder`). The lookup is duck-typed so both paths
    share the same primitive.
    """
    passing = [c for c in calls if _verifier_passed(c)]
    if not passing:
        return None
    return max(passing, key=_attempt_key)


def _verifier_passed(call: object) -> bool:
    """Return True iff ``call``'s ``verifier_pass`` field is truthy.

    Supports both ``Mapping`` (test fixtures) and attribute-bearing
    record types (production verified-call tuples) so the winning-
    attempt rule has one implementation across both call sites.
    """
    if isinstance(call, Mapping):
        return bool(call.get("verifier_pass"))
    return bool(getattr(call, "verifier_pass", False))


def _attempt_key(call: object) -> int:
    """Extract ``attempt_id`` as the comparison key.

    Same duck-typed lookup as :func:`_verifier_passed`. Raises
    ``TypeError`` if neither shape provides the field — the caller is
    handing us malformed data and should not silently fall back to 0.
    """
    if isinstance(call, Mapping):
        attempt = call.get("attempt_id")
    else:
        attempt = getattr(call, "attempt_id", None)
    if attempt is None:
        raise TypeError(
            f"call {call!r} is missing 'attempt_id'; "
            "winning-attempt rule needs a deterministic ordering key"
        )
    return int(attempt)


__all__ = ["apply_batch_results", "select_winning_attempt"]
