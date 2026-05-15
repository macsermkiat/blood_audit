"""Retry → Sonnet → Opus 4.7 escalation orchestration.

PRD §13: "Retry → Sonnet (≤ 2x) → escalate to Opus 4.7". The
escalation policy is purely a function of :class:`ParseOutcome`:

* Parseable Sonnet result on attempt 1 → no retry, no Opus call.
* Parse failure on attempt 1 → second Sonnet attempt.
* Parse failure on attempts 1 + 2 → invoke Opus.
* Parse failure on Opus → row routes to ``NEEDS_REVIEW`` with the
  ``parse_failure`` reason; no further escalation (Opus is the top
  of the model ladder for Phase 1).

This module is the *policy* layer — it never makes HTTP calls itself.
The transport boundary is :class:`AnthropicTransport`, injected by
the caller (production: SDK wrapper; tests: cassette).
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.llm_client.models import (
    AnthropicTransport,
    BatchSubmissionRequest,
    BatchSubmissionResult,
    EscalationLog,
    LlmClientConfig,
    ParseOutcome,
)


def should_escalate(
    parse_outcomes: Sequence[ParseOutcome],
    config: LlmClientConfig,
) -> bool:
    """Return True iff every Sonnet attempt failed and Opus must run.

    ``parse_outcomes`` is the list of per-attempt :class:`ParseOutcome`
    for ONE audit_id (ordered by attempt). The decision is structural:
    True iff the list has reached :attr:`LlmClientConfig.max_sonnet_attempts`
    AND every outcome has ``parse_failure=True``.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #22")


def escalate_to_opus(
    request: BatchSubmissionRequest,
    transport: AnthropicTransport,
    config: LlmClientConfig,
) -> BatchSubmissionResult:
    """Invoke Opus on a single audit_id after Sonnet exhaustion.

    The Opus call is structurally identical to a Sonnet submission
    except for the ``model`` field. The custom_id ↔ audit_id assertion
    still holds; the caller composes a single-element batch from
    ``request`` and submits.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #22")


def run_with_escalation(
    request: BatchSubmissionRequest,
    transport: AnthropicTransport,
    config: LlmClientConfig,
) -> tuple[EscalationLog, tuple[BatchSubmissionResult, ...]]:
    """Run the full retry + escalation loop for ONE audit_id.

    Returns:
        ``(escalation_log, persisted_results)`` — ``persisted_results``
        is the tuple of every Anthropic invocation in order (Sonnet
        attempts, then optional Opus). Every result is persisted to
        :class:`bba.audit_store.LlmCall` by the caller.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #22")
