"""Top-level orchestration: submit → assert custom_ids → escalate → persist.

PRD §13 + issue #22 acceptance criteria.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from typing import Any

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
from bba.llm_client.parser import parse_structured_response


_BBA_HEADERS_KEY: str = "__bba_response_headers__"
"""Namespaced key under which response_headers are folded into the
persisted response_json. The double-underscore prefix is unlikely to
collide with any Anthropic field; the writer raises
:class:`BatchSubmissionError` if it ever does."""


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
    persisted_calls = list(
        _call_from_result(call, attempt_index=i, run_id=request.run_id)
        for i, call in enumerate(raw_calls)
    )

    sonnet_response = _last_sonnet_response(parse_outcomes, escalation)
    opus_response = _opus_response(parse_outcomes, escalation)

    # Optional cross-check: when configured, ALWAYS run Opus once
    # Sonnet has produced a parseable classification — even if no
    # escalation was triggered. Disagreement then routes to NEEDS_REVIEW
    # without losing the Opus call's record. PRD §13 keeps this
    # explicit and opt-in because shadow-Opus on every row is a ~5x
    # cost regression.
    cross_check_disagreement: DisagreementVerdict | None = None
    cross_check_opus_parse_failure = False
    if (
        config.cross_check_with_opus
        and not escalation.escalated_to_opus
        and sonnet_response is not None
    ):
        opus_call = _run_opus_cross_check(request, transport, config)
        opus_outcome = parse_structured_response(opus_call)
        persisted_calls.append(
            _call_from_result(
                opus_call,
                attempt_index=len(persisted_calls),
                run_id=request.run_id,
            )
        )
        if not opus_outcome.parse_failure:
            opus_response = opus_outcome.parsed
            cross_check_disagreement = detect_disagreement(
                sonnet_response, opus_response
            )
        else:
            # Opus cross-check failed to parse: the quality gate
            # degraded but the row's chain of custody is intact. Route
            # to NEEDS_REVIEW with a distinct review reason so the
            # reviewer knows WHY they're seeing this row (not a model
            # disagreement, but a cross-check parse failure that
            # invalidated the comparison).
            cross_check_opus_parse_failure = True

    persisted_tuple = tuple(persisted_calls)

    if escalation.escalated_to_opus and opus_response is not None:
        # Escalation path: Opus parsed successfully → its answer is recorded.
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
            persisted_calls=persisted_tuple,
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
            persisted_calls=persisted_tuple,
        )

    # Sonnet eventually succeeded (with or without cross-check).
    if sonnet_response is not None:
        if cross_check_disagreement is not None and cross_check_disagreement.routed_to_needs_review:
            return LlmClientResult(
                audit_id=request.audit_id,
                run_id=request.run_id,
                final_classification="NEEDS_REVIEW",
                response=sonnet_response,
                parse_failure=False,
                needs_review=True,
                review_reason="disagreement",
                escalation=escalation,
                disagreement=cross_check_disagreement,
                persisted_calls=persisted_tuple,
            )
        if cross_check_opus_parse_failure:
            return LlmClientResult(
                audit_id=request.audit_id,
                run_id=request.run_id,
                final_classification="NEEDS_REVIEW",
                response=sonnet_response,
                parse_failure=False,
                needs_review=True,
                review_reason="opus_cross_check_parse_failure",
                escalation=escalation,
                disagreement=None,
                persisted_calls=persisted_tuple,
            )
        return LlmClientResult(
            audit_id=request.audit_id,
            run_id=request.run_id,
            final_classification=sonnet_response.classification,
            response=sonnet_response,
            parse_failure=False,
            needs_review=False,
            review_reason=None,
            escalation=escalation,
            disagreement=cross_check_disagreement,
            persisted_calls=persisted_tuple,
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
        persisted_calls=persisted_tuple,
    )


def _run_opus_cross_check(
    request: BatchSubmissionRequest,
    transport: AnthropicTransport,
    config: LlmClientConfig,
) -> BatchSubmissionResult:
    """Single Opus call for cross-check (config.cross_check_with_opus=True).

    Mirrors :func:`bba.llm_client.escalation.escalate_to_opus` but is
    invoked from the success path rather than the failure path. The
    custom_id assertion still holds; the result is folded into the
    persisted calls so the audit row carries the full reproducibility
    chain.
    """
    response = transport.submit_batch(
        model=config.opus_model_id,
        requests=[request],
        prompt_cache_enabled=config.prompt_cache_enabled,
    )
    mapping = assert_custom_ids_match([request], list(response.results))
    return mapping[request.audit_id]


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

    On agreement (or no Sonnet response to compare against), Opus's
    classification is recorded as the final answer. On disagreement
    the row routes to ``NEEDS_REVIEW`` with the ``disagreement``
    reason — the verbatim Opus and Sonnet labels remain on
    :class:`DisagreementVerdict` so reviewers can see both verdicts
    without re-reading the raw responses.
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

    The ``call_id`` is derived deterministically from STABLE inputs only
    — ``(run_id, audit_id, model_id, attempt_index, canonical
    request_json hash)``. Including ``request_timestamp`` would make a
    re-run with the same inputs produce a different ``call_id`` each
    time, breaking the audit_store's append-only idempotency contract
    (re-running with the same ``(audit_id, run_id, code_version)``
    must be a no-op, not append a duplicate row).

    ``response_headers`` lands inside :attr:`LlmCall.response_json`
    under the ``__bba_response_headers__`` key (double-underscore
    namespaced so a future Anthropic response field is unlikely to
    collide; if Anthropic ever does ship that exact key, we raise
    :class:`BatchSubmissionError` rather than silently overwrite it).
    The audit_store schema does not have a dedicated headers column,
    but the PRD reproducibility requirement ("Persist the full
    Anthropic Batch API request and response per audit_id, including
    ``anthropic-version`` header") is satisfied by folding the headers
    into the envelope before persistence.
    """
    canonical_request = json.dumps(
        result.request_json, sort_keys=True, ensure_ascii=False, default=str
    )
    request_hash = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()[:16]
    fingerprint = hashlib.sha256(
        f"{run_id}|{result.custom_id}|{result.model_id}|{attempt_index}|"
        f"{request_hash}".encode("utf-8")
    ).hexdigest()[:16]
    call_id = f"call-{result.custom_id}-{attempt_index}-{fingerprint}"
    # Shallow-copy the top-level frozen mapping so the dict spread below
    # works on regular `dict` semantics. The LlmCall field validator
    # deep-freezes the nested structure again on construction. The
    # double-underscore-namespaced key avoids any future collision
    # with an Anthropic-supplied top-level field; the explicit
    # collision check raises loudly if Anthropic ever does ship that
    # key so we don't silently overwrite real response data.
    response_envelope: dict[str, Any] = dict(result.raw_response_json)
    if _BBA_HEADERS_KEY in response_envelope:
        raise BatchSubmissionError(
            f"Anthropic response contains reserved key "
            f"{_BBA_HEADERS_KEY!r}; this is a contract drift — refusing "
            "to overwrite the vendor field with response_headers"
        )
    response_envelope[_BBA_HEADERS_KEY] = dict(result.response_headers)
    return LlmCall(
        call_id=call_id,
        audit_id=result.custom_id,
        run_id=run_id,
        model_id=result.model_id,
        anthropic_version=result.anthropic_version,
        prompt_cache_id=result.prompt_cache_id,
        request_json=result.request_json,
        response_json=response_envelope,
        request_timestamp=result.request_timestamp,
        latency_ms=result.latency_ms,
        extended_thinking_blocks=result.extended_thinking_blocks,
        cold_storage_uri=None,
    )
