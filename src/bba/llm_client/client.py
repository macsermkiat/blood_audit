"""Top-level orchestration: submit -> assert custom_ids -> escalate -> persist.

PRD §13 + issue #22 acceptance criteria. The orchestration sequence:

1. :func:`submit_batch` — one HTTP boundary call via the injected
   :class:`AnthropicTransport`. Returns the raw batch response.
2. :func:`bba.llm_client.custom_id.assert_custom_ids_match` — enforces
   the PRD §13 invariant before any per-row work begins. A mismatch
   aborts the whole batch (raises :class:`CustomIdMismatchError`).
3. Per-row: :func:`bba.llm_client.escalation.run_with_escalation`
   handles retry + Opus escalation; returns the persisted call list
   and the escalation log.
4. Per-row: :func:`bba.llm_client.disagreement.detect_disagreement`
   compares Sonnet vs Opus classifications when both succeed.
5. Assemble :class:`LlmClientResult` per row; route to NEEDS_REVIEW
   on parse failure or disagreement.

The function is pure-control-flow on top of the transport boundary —
no I/O of its own, no hidden state. Persistence to ``llm_calls`` is
the caller's responsibility (:meth:`bba.audit_store.AuditStore.write`).
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.llm_client.models import (
    AnthropicTransport,
    BatchSubmissionRequest,
    LlmClientConfig,
    LlmClientResult,
    RawBatchResponse,
)


def submit_batch(
    requests: Sequence[BatchSubmissionRequest],
    transport: AnthropicTransport,
    config: LlmClientConfig,
    *,
    model_id: str | None = None,
) -> RawBatchResponse:
    """Single-model batch submission.

    ``model_id`` defaults to :attr:`LlmClientConfig.sonnet_model_id`
    so the common case (first pass) reads cleanly; Opus escalation
    passes :attr:`LlmClientConfig.opus_model_id` explicitly.

    Raises :class:`BatchSubmissionError` for any pre-flight failure
    (empty requests, duplicate custom_ids, oversized payload).
    """
    raise NotImplementedError("RED-phase scaffold; see issue #22")


def process_batch(
    requests: Sequence[BatchSubmissionRequest],
    transport: AnthropicTransport,
    config: LlmClientConfig,
) -> tuple[LlmClientResult, ...]:
    """Run the full retry + escalation + disagreement pipeline.

    For each :class:`BatchSubmissionRequest`:

    1. Run with Sonnet (up to :attr:`LlmClientConfig.max_sonnet_attempts`).
    2. On total failure, escalate to Opus.
    3. If Opus parses but disagrees with the last successful Sonnet
       attempt, route to ``NEEDS_REVIEW`` with the ``disagreement``
       reason (Opus answer is the recorded classification).
    4. Persist every Anthropic invocation as :class:`LlmCall` on the
       result's ``persisted_calls`` tuple.

    The function never raises on a routing decision: parse failure,
    disagreement, and grounding failure (handled by the caller) all
    travel through the :class:`LlmClientResult`. Only contract
    violations raise (:class:`CustomIdMismatchError`,
    :class:`BatchSubmissionError`).
    """
    raise NotImplementedError("RED-phase scaffold; see issue #22")
