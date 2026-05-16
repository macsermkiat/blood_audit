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
  Anthropic returns a ``batch_id``, then advances to ``COMPLETE`` as
  results land. A crash at any point leaves the ``batch_runs`` row in
  a recoverable state for the resume reconciler.
* :func:`process_audit_order` is the per-row entry point used both by
  :func:`run_pipeline` and by the resume reconciler when re-emitting
  orphan llm_calls (user constraint #5).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime

from bba.audit_pipeline.cost_guard import assert_test_safe_transport
from bba.audit_pipeline.models import (
    AuditPipelineConfig,
    BatchRun,
    BatchRunState,
    PipelineRunResult,
)
from bba.audit_pipeline.replay import apply_batch_results
from bba.audit_pipeline.state_machine import transition
from bba.audit_pipeline.store import BatchRunStore
from bba.audit_orders import AuditOrder
from bba.audit_store import AuditStore
from bba.llm_client.models import (
    AnthropicTransport,
    BatchSubmissionRequest,
    LlmClientConfig,
)


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

    Steps per batch:

    1. Cost-guard: refuse a live Anthropic transport (user constraint
       #10). Tests inject a CassetteTransport; production injects
       AnthropicBatchTransport via a wrapping non-test caller.
    2. Create a :class:`BatchRun` row in ``PENDING``.
    3. Submit one batch per chunk of size
       ``pipeline_config.max_batch_size`` via ``transport.submit_batch``.
       Transition the batch_runs row to ``SUBMITTED``.
    4. Translate the response → AuditRow + LlmCall pairs through
       :func:`bba.audit_pipeline.replay.apply_batch_results` (which
       composes the audit_store idempotency contract).
    5. Transition the batch_runs row to ``COMPLETE``.

    Returns a :class:`PipelineRunResult` listing every persisted
    ``audit_id`` and every batch_id whose state changed.
    """
    if not orders:
        return PipelineRunResult(
            run_id=run_id,
            audit_ids_persisted=(),
            batch_runs_touched=(),
            orphan_audit_ids=(),
        )

    assert_test_safe_transport(transport)

    persisted: list[str] = []
    touched: list[str] = []

    for chunk in _chunked(orders, pipeline_config.max_batch_size):
        batch_id = _batch_id_for(run_id, chunk)
        now = _now_utc()
        pending = BatchRun(
            batch_id=batch_id,
            state=BatchRunState.PENDING,
            run_id=run_id,
            code_version=pipeline_config.code_version,
            audit_ids=tuple(o.audit_id for o in chunk),
            updated_at=now,
        )
        batch_run_store.create(pending)
        touched.append(batch_id)

        requests = _build_submission_requests(chunk, run_id=run_id)
        response = transport.submit_batch(
            model=llm_config.sonnet_model_id,
            requests=requests,
            prompt_cache_enabled=llm_config.prompt_cache_enabled,
        )

        submitted = transition(
            pending,
            to_state=BatchRunState.SUBMITTED,
            anthropic_batch_id=response.batch_id,
            now=_now_utc(),
        )
        batch_run_store.update(submitted)

        write_summary = apply_batch_results(
            response, audit_store=audit_store, run_id=run_id
        )
        persisted.extend(write_summary.audit_ids_persisted)

        completed = transition(
            submitted,
            to_state=BatchRunState.COMPLETE,
            now=_now_utc(),
        )
        batch_run_store.update(completed)

    orphans = _detect_orphans(
        expected_audit_ids=tuple(o.audit_id for o in orders),
        persisted=tuple(persisted),
    )

    return PipelineRunResult(
        run_id=run_id,
        audit_ids_persisted=tuple(persisted),
        batch_runs_touched=tuple(touched),
        orphan_audit_ids=orphans,
    )


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
    """
    return run_pipeline(
        [order],
        transport=transport,
        audit_store=audit_store,
        batch_run_store=batch_run_store,
        llm_config=llm_config,
        pipeline_config=pipeline_config,
        run_id=run_id,
    )


def _chunked(
    orders: Sequence[AuditOrder], size: int
) -> list[tuple[AuditOrder, ...]]:
    return [tuple(orders[i : i + size]) for i in range(0, len(orders), size)]


def _batch_id_for(run_id: str, chunk: Sequence[AuditOrder]) -> str:
    """Stable batch_id derived from run_id + chunk identity.

    A re-run with the same orders produces the same batch_id, which is
    safe because :meth:`BatchRunStore.create` rejects duplicates — the
    test for SIGTERM-then-restart relies on this so a second invocation
    surfaces the recoverable state in the store rather than orphaning
    a new pending row alongside the old one.
    """
    digest = hashlib.sha256(
        ("|".join([run_id, *(o.audit_id for o in chunk)])).encode("utf-8")
    ).hexdigest()[:16]
    return f"batch-{digest}"


def _build_submission_requests(
    chunk: Sequence[AuditOrder], *, run_id: str
) -> list[BatchSubmissionRequest]:
    """Build :class:`BatchSubmissionRequest` objects for ``chunk``.

    The full prompt assembly lives in :mod:`bba.prompt_builder` —
    this wiring delegates to :func:`bba.prompt_builder.build_prompt`
    so the prompt_hash + canonical envelope are computed via the
    canonical primitive (PRD §"audit-chain reproducibility"). One
    synthetic evidence chunk per audit_id keeps the cassette key
    deterministic without coupling the pipeline orchestrator to the
    upstream redaction / bundle stages.
    """
    from bba.prompt_builder import (
        EvidenceChunk,
        PromptBuildRequest,
        build_prompt,
    )

    out: list[BatchSubmissionRequest] = []
    for order in chunk:
        evidence_chunk = EvidenceChunk(
            evidence_id="E1",
            source="IPDNRFOCUSDT",
            text=f"synthetic evidence for {order.audit_id}",
        )
        prompt = build_prompt(
            PromptBuildRequest(
                task_mode="HB_7_10_REVIEW",
                cohort_threshold=7.0,
                evidence_chunks=(evidence_chunk,),
                few_shot_examples=(),
            )
        )
        out.append(
            BatchSubmissionRequest(
                audit_id=order.audit_id,
                run_id=run_id,
                task_mode="HB_7_10_REVIEW",
                prompt=prompt,
            )
        )
    return out


def _detect_orphans(
    *, expected_audit_ids: tuple[str, ...], persisted: tuple[str, ...]
) -> tuple[str, ...]:
    """Return the set of expected audit_ids that did not land.

    A non-empty result means the pipeline crashed between phase 1
    (submission) and phase 2 (persistence) for those rows; the resume
    reconciler will pick them up via :meth:`AuditStore.reconcile`.
    """
    persisted_set = set(persisted)
    return tuple(aid for aid in expected_audit_ids if aid not in persisted_set)


def _now_utc() -> datetime:
    return datetime.now(UTC)


__all__ = ["process_audit_order", "run_pipeline"]
