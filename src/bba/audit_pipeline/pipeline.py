"""Top-level orchestration: per-audit-row pipeline + batch runner.

This is the composition layer. It NEVER re-implements any of the
upstream modules; it imports + calls them in the canonical order
(user constraint #1):

    audit_orders → deterministic_classifier → evidence_bundle_builder
    → deid_redactor (BEFORE LLM) → prompt_builder → llm_client
    → quote_grounder → confidence_calibrator → audit_store

Row-level checkpointing (issue #24 AC ②):

* :func:`run_pipeline` creates a :class:`BatchRun` in ``PENDING`` state
  BEFORE submitting to Anthropic, transitions to ``SUBMITTED`` once
  Anthropic returns a ``batch_id``, then advances through ``PARTIAL``
  → ``COMPLETE`` as results land. A crash at any point leaves the
  ``batch_runs`` row in a recoverable state for the resume reconciler.
* :func:`process_audit_order` is the per-row entry point used both by
  :func:`run_pipeline` and by the resume reconciler when re-emitting
  orphan llm_calls (user constraint #5).
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.audit_pipeline.models import (
    AuditPipelineConfig,
    PipelineRunResult,
)
from bba.audit_pipeline.store import BatchRunStore
from bba.audit_orders import AuditOrder
from bba.audit_store import AuditStore
from bba.llm_client.models import AnthropicTransport, LlmClientConfig


def run_pipeline(
    orders: Sequence[AuditOrder],
    *,
    transport: AnthropicTransport,
    audit_store: AuditStore,
    batch_run_store: BatchRunStore,
    llm_config: LlmClientConfig,
    pipeline_config: AuditPipelineConfig,
    run_id: str,
) -> PipelineRunResult:
    """Run the full pipeline over ``orders``, checkpointing through
    ``batch_run_store``.

    The implementation will:

    1. Partition ``orders`` into batches of size
       ``pipeline_config.max_batch_size``.
    2. For each batch, create a :class:`BatchRun` row in ``PENDING``,
       build prompts via :mod:`bba.prompt_builder` (deterministic
       layer first to skip rows whose Hb-tier rule produces a final
       answer), redact via :mod:`bba.deid_redactor`, submit via
       :mod:`bba.llm_client`, transition to ``SUBMITTED``.
    3. As Anthropic returns results, verify via
       :mod:`bba.quote_grounder`, pick the winning attempt
       (:func:`bba.audit_pipeline.replay.select_winning_attempt`),
       persist via :mod:`bba.audit_store`, transition to ``PARTIAL``
       or ``COMPLETE``.
    4. Return a :class:`PipelineRunResult` listing every ``audit_id``
       whose row landed and every ``batch_id`` whose state changed.

    The implementation lives in GREEN (issue #24).
    """
    _ = (
        orders,
        transport,
        audit_store,
        batch_run_store,
        llm_config,
        pipeline_config,
        run_id,
    )
    raise NotImplementedError("RED-phase scaffold; see issue #24")


def process_audit_order(
    order: AuditOrder,
    *,
    transport: AnthropicTransport,
    audit_store: AuditStore,
    batch_run_store: BatchRunStore,
    llm_config: LlmClientConfig,
    pipeline_config: AuditPipelineConfig,
    run_id: str,
) -> PipelineRunResult:
    """Run the pipeline for a single :class:`AuditOrder`.

    Convenience wrapper around :func:`run_pipeline` with a length-1
    sequence; used by the resume reconciler when re-emitting an
    orphan ``llm_calls`` row through the verifier.

    Implementation lives in GREEN (issue #24).
    """
    _ = (
        order,
        transport,
        audit_store,
        batch_run_store,
        llm_config,
        pipeline_config,
        run_id,
    )
    raise NotImplementedError("RED-phase scaffold; see issue #24")


__all__ = ["process_audit_order", "run_pipeline"]
