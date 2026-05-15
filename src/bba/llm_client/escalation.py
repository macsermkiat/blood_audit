"""Retry → Sonnet → Opus 4.7 escalation orchestration.

PRD §13: "Retry → Sonnet (≤ 2x) → escalate to Opus 4.7". The escalation
policy is purely a function of :class:`ParseOutcome`:

* Parseable Sonnet result on attempt 1 → no retry, no Opus call.
* Parse failure on attempt 1 → second Sonnet attempt.
* Parse failure on attempts 1 + 2 → invoke Opus.
* Parse failure on Opus → row routes to ``NEEDS_REVIEW`` with the
  ``parse_failure`` reason; no further escalation (Opus is the top of
  the model ladder for Phase 1).

This module is the *policy* layer — it never makes HTTP calls itself.
The transport boundary is :class:`AnthropicTransport`, injected by the
caller (production: SDK wrapper; tests: cassette).
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.llm_client.custom_id import assert_custom_ids_match
from bba.llm_client.models import (
    AnthropicTransport,
    BatchSubmissionRequest,
    BatchSubmissionResult,
    EscalationLog,
    LlmClientConfig,
    ParseOutcome,
)
from bba.llm_client.parser import parse_structured_response


def should_escalate(
    parse_outcomes: Sequence[ParseOutcome],
    config: LlmClientConfig,
) -> bool:
    """Return True iff every Sonnet attempt failed and Opus must run."""
    if len(parse_outcomes) < config.max_sonnet_attempts:
        return False
    return all(o.parse_failure for o in parse_outcomes)


def escalate_to_opus(
    request: BatchSubmissionRequest,
    transport: AnthropicTransport,
    config: LlmClientConfig,
) -> BatchSubmissionResult:
    """Invoke Opus on a single audit_id after Sonnet exhaustion."""
    response = transport.submit_batch(
        model=config.opus_model_id,
        requests=[request],
        prompt_cache_enabled=config.prompt_cache_enabled,
    )
    mapping = assert_custom_ids_match([request], list(response.results))
    return mapping[request.audit_id]


def run_with_escalation(
    request: BatchSubmissionRequest,
    transport: AnthropicTransport,
    config: LlmClientConfig,
) -> tuple[EscalationLog, tuple[BatchSubmissionResult, ...], tuple[ParseOutcome, ...]]:
    """Run the retry + escalation loop for ONE audit_id.

    Returns ``(escalation_log, persisted_results, parse_outcomes)``.
    ``persisted_results`` carries every Anthropic call in order
    (Sonnet attempts, then optional Opus). ``parse_outcomes`` is the
    parallel parse-outcome tuple — Sonnet attempts only (Opus's
    parse outcome is on :attr:`EscalationLog.opus_parse_failure`).
    """
    from bba.llm_client.models import ParseFailureReason

    sonnet_results: list[BatchSubmissionResult] = []
    sonnet_outcomes: list[ParseOutcome] = []
    sonnet_failures: list[ParseFailureReason] = []

    for _ in range(config.max_sonnet_attempts):
        result = _submit_one(request, transport, config.sonnet_model_id, config)
        outcome = parse_structured_response(result)
        sonnet_results.append(result)
        sonnet_outcomes.append(outcome)
        if outcome.parse_failure and outcome.parse_failure_reason is not None:
            sonnet_failures.append(outcome.parse_failure_reason)
        if not outcome.parse_failure:
            return (
                EscalationLog(
                    audit_id=request.audit_id,
                    sonnet_attempts=len(sonnet_results),
                    sonnet_parse_failures=tuple(sonnet_failures),
                    escalated_to_opus=False,
                ),
                tuple(sonnet_results),
                tuple(sonnet_outcomes),
            )

    # Every Sonnet attempt failed → escalate to Opus.
    opus_result = escalate_to_opus(request, transport, config)
    opus_outcome = parse_structured_response(opus_result)
    return (
        EscalationLog(
            audit_id=request.audit_id,
            sonnet_attempts=len(sonnet_results),
            sonnet_parse_failures=tuple(sonnet_failures),
            escalated_to_opus=True,
            opus_parse_failure=opus_outcome.parse_failure_reason
            if opus_outcome.parse_failure
            else None,
        ),
        tuple(sonnet_results) + (opus_result,),
        tuple(sonnet_outcomes) + (opus_outcome,),
    )


def _submit_one(
    request: BatchSubmissionRequest,
    transport: AnthropicTransport,
    model_id: str,
    config: LlmClientConfig,
) -> BatchSubmissionResult:
    response = transport.submit_batch(
        model=model_id,
        requests=[request],
        prompt_cache_enabled=config.prompt_cache_enabled,
    )
    mapping = assert_custom_ids_match([request], list(response.results))
    return mapping[request.audit_id]
