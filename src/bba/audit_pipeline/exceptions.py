"""Typed exceptions for :mod:`bba.audit_pipeline`.

The orchestration layer fails LOUD: every recoverable boundary surfaces
its own exception class so callers can route on type rather than parsing
error strings. PRD §15 ("row-level checkpointing means a crash never
silently drops work") motivates the dedicated exception hierarchy.
"""

from __future__ import annotations


class AuditPipelineError(Exception):
    """Base for every typed failure raised by :mod:`bba.audit_pipeline`."""


class BatchStateTransitionError(AuditPipelineError):
    """A caller attempted a state-machine move outside
    :data:`bba.audit_pipeline.state_machine.VALID_TRANSITIONS`.

    Example: jumping ``PENDING -> COMPLETE`` (skipping ``SUBMITTED``).
    The state machine is the only safe gate that keeps the
    ``batch_runs`` row coherent under crash + resume; an unchecked
    transition would let a resume-on-startup poll an Anthropic
    ``batch_id`` that was never submitted.
    """


class ResumeReconciliationError(AuditPipelineError):
    """The startup reconciler found a state that is unrecoverable without
    operator action.

    Surfaces with the offending ``audit_id`` / ``batch_id`` set so
    operators can quarantine the row in ``batch_runs`` and re-emit
    manually rather than the pipeline silently retrying forever.
    """


class LiveAnthropicApiError(AuditPipelineError):
    """A test invoked the pipeline with the *live*
    :class:`bba.llm_client.AnthropicBatchTransport`.

    The audit pipeline must NEVER call the live Anthropic Batch API in
    unit or integration tests (PRD §"Cost guard during ralph-loop
    iteration"; user constraint #10). Tests opt in by injecting a
    :class:`bba.llm_client.CassetteTransport`; the
    :func:`bba.audit_pipeline.cost_guard.assert_test_safe_transport`
    helper raises this error when a live transport is detected so the
    cassette-replay setup never silently regresses.
    """


__all__ = [
    "AuditPipelineError",
    "BatchStateTransitionError",
    "LiveAnthropicApiError",
    "ResumeReconciliationError",
]
