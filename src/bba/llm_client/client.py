"""Top-level orchestration: submit → assert custom_ids → escalate → persist.

PRD §13 + issue #22 acceptance criteria.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence

from bba.audit_store.models import LlmCall
from bba.llm_client.custom_id import assert_custom_ids_match
from bba.llm_client.disagreement import detect_disagreement
from bba.llm_client.escalation import run_with_escalation
from bba.llm_client.exceptions import BatchSubmissionError
from bba.llm_client.models import (
    AnthropicTransport,
    BatchSubmissionRequest,
    BatchSubmissionResult,
    Classification,
    DisagreementVerdict,
    EscalationLog,
    LlmClassificationResponse,
    LlmClientConfig,
    LlmClientResult,
    ParseOutcome,
    RawBatchResponse,
)


def submit_batch(
    requests: Sequence[BatchSubmissionRequest],
    transport: AnthropicTransport,
    config: LlmClientConfig,
    *,
    model_id: str | None = None,
) -> RawBatchResponse:
    """Single-model batch submission with pre-flight + custom_id checks."""
    _preflight(requests)
    target_model = model_id if model_id is not None else config.sonnet_model_id
    response = transport.submit_batch(
        model=target_model,
        requests=list(requests),
        prompt_cache_enabled=config.prompt_cache_enabled,
    )
    assert_custom_ids_match(list(requests), list(response.results))
    return response


def process_batch(
    requests: Sequence[BatchSubmissionRequest],
    transport: AnthropicTransport,
    config: LlmClientConfig,
) -> tuple[LlmClientResult, ...]:
    """Run the retry + escalation + disagreement pipeline."""
    _preflight(requests)
    results: list[LlmClientResult] = []
    for request in requests:
        results.append(_process_one(request, transport, config))
    return tuple(results)


def _preflight(requests: Sequence[BatchSubmissionRequest]) -> None:
    if not requests:
        raise BatchSubmissionError(
            "submit_batch requires at least one request (PRD §13: empty "
            "batches are a contract violation, not a no-op)"
        )
    counts = Counter(r.audit_id for r in requests)
    duplicates = sorted(aid for aid, n in counts.items() if n > 1)
    if duplicates:
        raise BatchSubmissionError(
            f"duplicate audit_id(s) in submission: {duplicates} "
            "(custom_id ↔ audit_id mapping must be 1:1 per batch)"
        )


def _process_one(
    request: BatchSubmissionRequest,
    transport: AnthropicTransport,
    config: LlmClientConfig,
) -> LlmClientResult:
    escalation, raw_calls, parse_outcomes = run_with_escalation(
        request, transport, config
    )
    persisted_calls = tuple(
        _call_from_result(call, attempt_index=i, run_id=request.run_id)
        for i, call in enumerate(raw_calls)
    )

    sonnet_response = _last_sonnet_response(parse_outcomes, escalation)
    opus_response = _opus_response(parse_outcomes, escalation)

    if escalation.escalated_to_opus and opus_response is not None:
        # Opus parsed successfully → its answer is recorded.
        disagreement = detect_disagreement(sonnet_response, opus_response)
        final, needs_review, review_reason = _resolve_with_opus(
            opus_response, disagreement
        )
        return LlmClientResult(
            audit_id=request.audit_id,
            run_id=request.run_id,
            final_classification=final,
            response=opus_response,
            parse_failure=False,
            needs_review=needs_review,
            review_reason=review_reason,
            escalation=escalation,
            disagreement=disagreement,
            persisted_calls=persisted_calls,
        )

    if escalation.escalated_to_opus and opus_response is None:
        # Every attempt — including Opus — failed to parse. Fail closed.
        return LlmClientResult(
            audit_id=request.audit_id,
            run_id=request.run_id,
            final_classification="NEEDS_REVIEW",
            response=None,
            parse_failure=True,
            needs_review=True,
            review_reason="parse_failure",
            escalation=escalation,
            disagreement=None,
            persisted_calls=persisted_calls,
        )

    # No Opus call: Sonnet eventually succeeded.
    if sonnet_response is not None:
        return LlmClientResult(
            audit_id=request.audit_id,
            run_id=request.run_id,
            final_classification=sonnet_response.classification,
            response=sonnet_response,
            parse_failure=False,
            needs_review=False,
            review_reason=None,
            escalation=escalation,
            disagreement=None,
            persisted_calls=persisted_calls,
        )

    # Defensive: should be unreachable because escalation runs Opus when
    # Sonnet fails to max_sonnet_attempts. Keeps the type checker honest.
    return LlmClientResult(
        audit_id=request.audit_id,
        run_id=request.run_id,
        final_classification="NEEDS_REVIEW",
        response=None,
        parse_failure=True,
        needs_review=True,
        review_reason="parse_failure",
        escalation=escalation,
        disagreement=None,
        persisted_calls=persisted_calls,
    )


def _last_sonnet_response(
    parse_outcomes: Sequence[ParseOutcome],
    escalation: EscalationLog,
) -> LlmClassificationResponse | None:
    """Return the last successful Sonnet parse, or None if every Sonnet
    attempt failed."""
    # The first ``sonnet_attempts`` outcomes are Sonnet's; anything
    # after belongs to Opus.
    sonnet_slice = parse_outcomes[: escalation.sonnet_attempts]
    for outcome in reversed(sonnet_slice):
        if not outcome.parse_failure:
            return outcome.parsed
    return None


def _opus_response(
    parse_outcomes: Sequence[ParseOutcome],
    escalation: EscalationLog,
) -> LlmClassificationResponse | None:
    if not escalation.escalated_to_opus:
        return None
    opus_slice = parse_outcomes[escalation.sonnet_attempts :]
    if not opus_slice:
        return None
    return opus_slice[-1].parsed


def _resolve_with_opus(
    opus_response: LlmClassificationResponse,
    disagreement: DisagreementVerdict,
) -> tuple[Classification, bool, str | None]:
    """Pick the final classification when Opus parsed successfully.

    Opus is the recorded classification. If Sonnet and Opus disagree,
    the row routes to NEEDS_REVIEW (the disagreement reason is
    persisted) but the Opus answer is still recorded as
    ``final_classification`` so the audit row carries the deeper
    model's verdict.
    """
    if disagreement.routed_to_needs_review:
        return "NEEDS_REVIEW", True, "disagreement"
    return opus_response.classification, False, None


def _call_from_result(
    result: BatchSubmissionResult,
    *,
    attempt_index: int,
    run_id: str,
) -> LlmCall:
    """Translate one :class:`BatchSubmissionResult` into a persistable
    :class:`LlmCall`.

    The ``call_id`` is derived deterministically from
    ``(custom_id, model_id, attempt_index, request_timestamp)`` so a
    re-run with the same inputs produces the same ID — required by
    the audit_store's append-only contract (a filename collision would
    overwrite the prior call).
    """
    fingerprint = hashlib.sha256(
        f"{result.custom_id}|{result.model_id}|{attempt_index}|"
        f"{result.request_timestamp.isoformat()}".encode("utf-8")
    ).hexdigest()[:16]
    call_id = f"call-{result.custom_id}-{attempt_index}-{fingerprint}"
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
