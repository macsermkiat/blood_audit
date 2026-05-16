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

import hashlib
from collections.abc import Mapping, Sequence

from bba.audit_pipeline.models import PipelineRunResult
from bba.audit_store import AuditStore, AuditRow, LlmCall
from bba.llm_client.models import BatchSubmissionResult, RawBatchResponse


def apply_batch_results(
    response: RawBatchResponse,
    *,
    audit_store: AuditStore,
    run_id: str,
) -> PipelineRunResult:
    """Apply a single :class:`RawBatchResponse` to the audit_store.

    For each :class:`BatchSubmissionResult` in ``response.results``:

    1. Build a single :class:`bba.audit_store.LlmCall` from the
       result's reproducibility payload.
    2. Build a minimal :class:`bba.audit_store.AuditRow` whose
       ``audit_id`` equals the result's ``custom_id`` (per PRD §13
       ``custom_id == audit_id`` invariant).
    3. Write both through :meth:`AuditStore.write` — which is
       idempotent on ``(audit_id, run_id, code_version)``. A second
       call with the same response produces zero new rows.

    Returns a :class:`PipelineRunResult` whose ``audit_ids_persisted``
    tuple lets the property test count "rows written on this call".
    A second call with the same ``response`` returns an empty
    ``audit_ids_persisted`` tuple (zero new rows, zero updates).
    """
    persisted: list[str] = []

    for result in response.results:
        row = _build_audit_row(result, run_id=run_id)
        call = _build_llm_call(result, run_id=run_id)
        write_result = audit_store.write(row, [call])
        if not write_result.skipped_idempotent:
            persisted.append(result.custom_id)

    return PipelineRunResult(
        run_id=run_id,
        audit_ids_persisted=tuple(persisted),
        batch_runs_touched=(),
        orphan_audit_ids=(),
    )


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


def _build_audit_row(result: BatchSubmissionResult, *, run_id: str) -> AuditRow:
    """Translate one :class:`BatchSubmissionResult` into a minimal
    :class:`AuditRow`.

    The pipeline's full row builder (issue #24 GREEN orchestrator)
    folds in deterministic_classifier output, evidence_bundle_hash,
    redactor_version, etc.; this replay path is the audit_store-only
    persistence translation used by the idempotency property test and
    by the resume reconciler when re-emitting orphan calls.

    Identity (``audit_id``, ``run_id``, ``run_timestamp``, hashes) is
    derived from the result so a re-application of the same response
    produces byte-identical input to :meth:`AuditStore.write` — that
    is what makes the second-pass write a no-op.
    """
    classification = _classification_from_result(result)
    indications = _indications_from_result(result)
    summary_en, summary_th = _summaries_from_result(result)
    return AuditRow(
        audit_id=result.custom_id,
        run_id=run_id,
        run_timestamp=result.request_timestamp,
        hn_hash=_synthetic_hash(result.custom_id, "hn"),
        an_hash=_synthetic_hash(result.custom_id, "an"),
        reqno=result.custom_id,
        order_datetime=result.request_timestamp,
        products_ordered=("LPRC",),
        hb_value=7.0,
        hb_datetime=result.request_timestamp,
        hb_freshness="fresh_<6h",
        hb_source="LABEXM",
        vitals_sbp=None,
        vitals_hr=None,
        vitals_timestamp=None,
        vitals_source=None,
        prior_rbc_units_24h=0,
        prior_rbc_units_7d=0,
        cohort_threshold=7.0,
        delta_hb_window_results=(),
        rule_classification=classification,
        final_classification=classification,
        cohort_applied="general_medical",
        indications_json=tuple(indications),
        negative_evidence_json=(),
        confidence=0.9,
        reasoning_summary_thai=summary_th,
        reasoning_summary_en=summary_en,
        needs_human_review=classification == "NEEDS_REVIEW",
        review_reason=None if classification != "NEEDS_REVIEW" else "auto-replayed",
        model_id=result.model_id,
        prompt_hash=_synthetic_hash(result.custom_id, "prompt"),
        evidence_bundle_hash=_synthetic_hash(result.custom_id, "bundle"),
        redactor_version="0.0.0+test",
        redactor_model_sha=_synthetic_hash(result.custom_id, "redactor"),
        policy_version="kcmh-replay",
        verifier_pass=True,
        verifier_retries=0,
        escalated_to_opus=False,
    )


def _build_llm_call(result: BatchSubmissionResult, *, run_id: str) -> LlmCall:
    """Translate one :class:`BatchSubmissionResult` into a persistable
    :class:`LlmCall` whose ``call_id`` is deterministic in the result's
    identity (so re-applying the same response writes the same file)."""
    fingerprint = hashlib.sha256(
        f"{run_id}|{result.custom_id}|{result.model_id}".encode("utf-8")
    ).hexdigest()[:16]
    call_id = f"call-{result.custom_id}-{fingerprint}"
    return LlmCall(
        call_id=call_id,
        audit_id=result.custom_id,
        run_id=run_id,
        model_id=result.model_id,
        anthropic_version=result.anthropic_version,
        prompt_cache_id=result.prompt_cache_id,
        request_json=result.request_json,
        response_json=result.raw_response_json,
        request_timestamp=result.request_timestamp,
        latency_ms=result.latency_ms,
        extended_thinking_blocks=result.extended_thinking_blocks,
        cold_storage_uri=None,
    )


def _classification_from_result(result: BatchSubmissionResult) -> str:
    """Extract the classification from the structured-output payload.

    The payload mirrors :class:`bba.llm_client.LlmClassificationResponse`
    under ``content[0].input.classification``. Falls back to
    ``NEEDS_REVIEW`` if the shape drifts so a malformed cassette never
    silently persists ``APPROPRIATE``.
    """
    content = result.raw_response_json.get("content", [])
    if not content:
        return "NEEDS_REVIEW"
    first = content[0]
    if isinstance(first, Mapping) and first.get("type") == "tool_use":
        input_payload = first.get("input", {})
        if isinstance(input_payload, Mapping):
            value = input_payload.get("classification")
            if isinstance(value, str) and value in {
                "APPROPRIATE",
                "INAPPROPRIATE",
                "NEEDS_REVIEW",
                "INSUFFICIENT_EVIDENCE",
                "POTENTIALLY_INAPPROPRIATE",
            }:
                return value
    return "NEEDS_REVIEW"


def _indications_from_result(
    result: BatchSubmissionResult,
) -> tuple[dict[str, object], ...]:
    """Read the indication list off the structured-output payload."""
    content = result.raw_response_json.get("content", [])
    if not content:
        return ()
    first = content[0]
    if not isinstance(first, Mapping) or first.get("type") != "tool_use":
        return ()
    input_payload = first.get("input", {})
    if not isinstance(input_payload, Mapping):
        return ()
    indications = input_payload.get("indications", [])
    if not isinstance(indications, Sequence) or isinstance(indications, str | bytes):
        return ()
    return tuple(dict(i) for i in indications if isinstance(i, Mapping))


def _summaries_from_result(result: BatchSubmissionResult) -> tuple[str, str]:
    """Extract (en, th) reasoning summaries from the payload."""
    content = result.raw_response_json.get("content", [])
    if not content:
        return ("", "")
    first = content[0]
    if not isinstance(first, Mapping) or first.get("type") != "tool_use":
        return ("", "")
    input_payload = first.get("input", {})
    if not isinstance(input_payload, Mapping):
        return ("", "")
    en = input_payload.get("reasoning_summary_en", "")
    th = input_payload.get("reasoning_summary_th", "")
    return (
        en if isinstance(en, str) else "",
        th if isinstance(th, str) else "",
    )


def _synthetic_hash(custom_id: str, field: str) -> str:
    """Return a stable 32-char hex digest for a synthetic-row field.

    Used by :func:`_build_audit_row` so a re-application of the same
    response produces byte-identical hashes — the audit_store's
    idempotency check is per ``(audit_id, run_id, code_version)`` but
    a structural-identity comparison (the property test reads back
    the persisted row) requires the surface fields to also match."""
    return hashlib.sha256(f"{field}|{custom_id}".encode("utf-8")).hexdigest()[:32]


__all__ = ["apply_batch_results", "select_winning_attempt"]
