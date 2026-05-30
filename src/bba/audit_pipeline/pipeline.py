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
  / ``COMPLETE`` as results land. A crash at any point leaves the
  ``batch_runs`` row in a recoverable state for the resume reconciler.
* :func:`process_audit_order` is the per-row entry point used both by
  :func:`run_pipeline` and by the resume reconciler when re-emitting
  orphan llm_calls (user constraint #5).

The pipeline takes a sequence of :class:`PipelineRowContext`,
NOT raw :class:`AuditOrder` rows. The caller is responsible for
assembling the context from upstream modules — that boundary prevents
the orchestrator from silently fabricating clinical data (Codex review
HIGH #5).
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
    PipelineRowContext,
    PipelineRunResult,
)
from bba.audit_pipeline.replay import (
    Verifier,
    apply_batch_results,
    default_verifier,
)
from bba.audit_pipeline.state_machine import transition
from bba.audit_pipeline.store import BatchRunStore
from bba.audit_store import AuditRow, AuditStore, LlmCall
from bba.deterministic_classifier import (
    BypassReason,
    ClassifierInputs,
    ClassifierResult,
    classify,
)
from bba.llm_client.models import (
    AnthropicTransport,
    BatchSubmissionRequest,
    LlmClientConfig,
)


# Deterministic classifier results that produce a final answer without
# the LLM stage. POTENTIALLY_INAPPROPRIATE and NEEDS_REVIEW route to the
# LLM for a positive-evidence call (PRD §6); APPROPRIATE and
# INSUFFICIENT_EVIDENCE are final at the deterministic layer.
_DETERMINISTIC_FINAL_CLASSIFICATIONS = frozenset(
    {"APPROPRIATE", "INSUFFICIENT_EVIDENCE", "INAPPROPRIATE"}
)


def run_pipeline(
    contexts: Sequence[PipelineRowContext],
    *,
    transport: AnthropicTransport,
    audit_store: AuditStore,
    batch_run_store: BatchRunStore,
    llm_config: LlmClientConfig,
    pipeline_config: AuditPipelineConfig,
    run_id: str,
    verifier: Verifier = default_verifier,
) -> PipelineRunResult:
    """Run the full pipeline over ``contexts``, checkpointing through
    ``batch_run_store``.

    Steps per batch:

    1. Cost-guard: refuse a live Anthropic transport (user constraint
       #10). Tests inject a CassetteTransport; production injects
       AnthropicBatchTransport via a wrapping non-test caller.
    2. Partition contexts into deterministic-final vs LLM-required
       buckets via :attr:`PipelineRowContext.classifier_result`.
    3. For deterministic-final rows: build the AuditRow directly from
       the classifier result + context, write via audit_store. No
       Anthropic call.
    4. For LLM-required rows: create a :class:`BatchRun` row in
       ``PENDING``, submit one batch per chunk via
       ``transport.submit_batch``, transition to ``SUBMITTED``.
    5. Translate the response → AuditRow + LlmCall pairs through
       :func:`apply_batch_results` (verifier + winning-attempt rule
       + audit_store idempotency).
    6. Transition the batch_runs row to ``COMPLETE``.

    Returns a :class:`PipelineRunResult` listing every persisted
    ``audit_id`` and every batch_id whose state changed.
    """
    if not contexts:
        return PipelineRunResult(
            run_id=run_id,
            audit_ids_persisted=(),
            batch_runs_touched=(),
            orphan_audit_ids=(),
        )

    assert_test_safe_transport(transport)

    persisted: list[str] = []
    touched: list[str] = []

    # Call the deterministic classifier ON EACH context (user
    # constraint #1: the pipeline composes the deterministic engine —
    # callers no longer pre-compute it). Partition into the two
    # buckets based on the classifier's verdict.
    classified: list[tuple[PipelineRowContext, ClassifierResult]] = []
    deterministic_final: list[tuple[PipelineRowContext, ClassifierResult]] = []
    llm_required: list[tuple[PipelineRowContext, ClassifierResult]] = []
    for ctx in contexts:
        result = classify(_classifier_inputs_for(ctx))
        classified.append((ctx, result))
        if result.classification in _DETERMINISTIC_FINAL_CLASSIFICATIONS:
            deterministic_final.append((ctx, result))
        else:
            llm_required.append((ctx, result))

    for ctx, result in deterministic_final:
        if _persist_deterministic_row(
            ctx,
            classifier_result=result,
            audit_store=audit_store,
            run_id=run_id,
        ):
            persisted.append(ctx.order.audit_id)

    llm_contexts_only = [ctx for ctx, _result in llm_required]
    llm_results_by_id: dict[str, ClassifierResult] = {
        ctx.order.audit_id: result for ctx, result in llm_required
    }

    for chunk_contexts in _chunked(llm_contexts_only, pipeline_config.max_batch_size):
        chunk_results = {
            ctx.order.audit_id: llm_results_by_id[ctx.order.audit_id]
            for ctx in chunk_contexts
        }
        batch_id = _batch_id_for(run_id, chunk_contexts)
        now = _now_utc()
        pending = BatchRun(
            batch_id=batch_id,
            state=BatchRunState.PENDING,
            run_id=run_id,
            code_version=pipeline_config.code_version,
            audit_ids=tuple(c.order.audit_id for c in chunk_contexts),
            updated_at=now,
        )
        batch_run_store.create(pending)
        touched.append(batch_id)

        requests = _build_submission_requests(chunk_contexts, run_id=run_id)

        # Split-phase submission (PR #54 codex P1 fix): create the
        # remote batch and persist the anthropic_batch_id BEFORE
        # polling. A SIGTERM during the polling window now leaves a
        # SUBMITTED row whose batch_id the resume reconciler can
        # poll, instead of a stranded PENDING row.
        anthropic_batch_id = transport.submit_batch_only(
            model=llm_config.sonnet_model_id,
            requests=requests,
            prompt_cache_enabled=llm_config.prompt_cache_enabled,
        )

        submitted = transition(
            pending,
            to_state=BatchRunState.SUBMITTED,
            anthropic_batch_id=anthropic_batch_id,
            now=_now_utc(),
        )
        batch_run_store.update(submitted)

        response = transport.fetch_batch_results(
            anthropic_batch_id,
            model=llm_config.sonnet_model_id,
            requests=requests,
            prompt_cache_enabled=llm_config.prompt_cache_enabled,
        )

        context_map = {ctx.order.audit_id: ctx for ctx in chunk_contexts}
        write_summary = apply_batch_results(
            response,
            audit_store=audit_store,
            run_id=run_id,
            contexts=context_map,
            classifier_results=chunk_results,
            verifier=verifier,
        )
        persisted.extend(write_summary.audit_ids_persisted)

        completed = transition(
            submitted,
            to_state=BatchRunState.COMPLETE,
            now=_now_utc(),
        )
        batch_run_store.update(completed)

    orphans = _detect_orphans(
        expected_audit_ids=tuple(c.order.audit_id for c in contexts),
        persisted=tuple(persisted),
    )

    return PipelineRunResult(
        run_id=run_id,
        audit_ids_persisted=tuple(persisted),
        batch_runs_touched=tuple(touched),
        orphan_audit_ids=orphans,
    )


def process_audit_order(
    context: PipelineRowContext,
    *,
    transport: AnthropicTransport,
    audit_store: AuditStore,
    batch_run_store: BatchRunStore,
    llm_config: LlmClientConfig,
    pipeline_config: AuditPipelineConfig,
    run_id: str,
    verifier: Verifier = default_verifier,
) -> PipelineRunResult:
    """Run the pipeline for a single :class:`PipelineRowContext`.

    Convenience wrapper around :func:`run_pipeline` with a length-1
    sequence; used by the resume reconciler when re-emitting an
    orphan ``llm_calls`` row through the verifier.
    """
    return run_pipeline(
        [context],
        transport=transport,
        audit_store=audit_store,
        batch_run_store=batch_run_store,
        llm_config=llm_config,
        pipeline_config=pipeline_config,
        run_id=run_id,
        verifier=verifier,
    )


def _persist_deterministic_row(
    context: PipelineRowContext,
    *,
    classifier_result: ClassifierResult,
    audit_store: AuditStore,
    run_id: str,
) -> bool:
    """Write the deterministic-final row directly without an LLM call.

    Returns True iff a new row was committed (False on idempotent
    skip — same ``(audit_id, run_id, code_version)`` already on disk).
    """
    from bba.audit_pipeline.replay import _audit_row_for_needs_review

    row: AuditRow
    classifier = classifier_result
    if classifier.classification == "NEEDS_REVIEW":
        # POTENTIALLY_INAPPROPRIATE and friends that route through the
        # LLM never land here — only legitimate deterministic-final
        # NEEDS_REVIEW (rare; today only the hemodilution-flagged path).
        row = _audit_row_for_needs_review(
            run_id=run_id,
            context=context,
            classifier_result=classifier,
            review_reason=classifier.bypass_reason.value
            if classifier.bypass_reason
            else "deterministic_review",
            verifier_pass=True,
            verifier_retries=0,
            model_id="deterministic",
            reasoning_en=f"deterministic: {classifier.rationale}",
            reasoning_th="",
            indications=(),
            negative_evidence=(),
            confidence=1.0,
            escalated=False,
        )
    else:
        row = _deterministic_audit_row(
            context=context,
            classifier_result=classifier,
            run_id=run_id,
        )

    # Deterministic-final rows have no Anthropic call; we still need
    # at least one llm_call to satisfy the audit_store transactional
    # invariant. Build a "deterministic" marker call that records the
    # rule rationale instead of an Anthropic payload.
    marker_call = _deterministic_marker_call(
        context=context, classifier_result=classifier, run_id=run_id
    )
    write_result = audit_store.write(row, [marker_call])
    return not write_result.skipped_idempotent


def _deterministic_audit_row(
    *,
    context: PipelineRowContext,
    classifier_result: ClassifierResult,
    run_id: str,
) -> AuditRow:
    """AuditRow for the deterministic-final path.

    Mirrors the LLM path's AuditRow construction but uses
    ``rule_classification`` as the final answer and a placeholder
    LLM payload (no Anthropic call was made).
    """
    from bba.audit_store import AuditRow

    # INSUFFICIENT_EVIDENCE legitimately has a missing Hb; APPROPRIATE
    # with an Hb-independent bypass (MTP or peri-procedural) also has a
    # legitimately missing Hb — both use 0.0 + freshness=="missing" as
    # the explicit "no Hb" sentinel (PRD §"Output schema"). All other
    # non-INSUFFICIENT classifications must supply real Hb — fail loud.
    _HB_INDEPENDENT_BYPASSES = frozenset(
        {BypassReason.MTP, BypassReason.PERI_PROCEDURAL_6H}
    )
    classifier = classifier_result
    if context.hb_result.value_g_dl is None and not (
        classifier.classification == "INSUFFICIENT_EVIDENCE"
        or classifier.bypass_reason in _HB_INDEPENDENT_BYPASSES
    ):
        raise ValueError(
            f"audit_id={context.order.audit_id!r}: classifier "
            f"emitted {classifier.classification!r} (bypass={classifier.bypass_reason!r}) "
            "but hb_result.value_g_dl is None; the pipeline must not fabricate a numeric Hb"
        )
    return AuditRow(
        audit_id=context.order.audit_id,
        run_id=run_id,
        run_timestamp=context.order.order_datetime,
        hn_hash=context.hn_hash,
        an_hash=context.an_hash,
        reqno=context.order.reqno,
        order_datetime=context.order.order_datetime,
        products_ordered=tuple(context.order.products_ordered),
        hb_value=context.hb_result.value_g_dl
        if context.hb_result.value_g_dl is not None
        else 0.0,
        hb_datetime=context.hb_result.datetime_utc
        if context.hb_result.datetime_utc is not None
        else context.order.order_datetime,
        hb_freshness=context.hb_result.freshness,
        hb_source=str(context.hb_result.source)
        if context.hb_result.source
        else "missing",
        vitals_sbp=context.vitals_result.vitals.sbp,
        vitals_hr=context.vitals_result.vitals.hr,
        vitals_timestamp=context.vitals_result.note_timestamp,
        vitals_source=context.vitals_result.source.value,
        prior_rbc_units_24h=context.prior_rbc_units_24h,
        prior_rbc_units_7d=context.prior_rbc_units_7d,
        cohort_threshold=context.cohort_assignment.threshold
        if context.cohort_assignment.threshold is not None
        else classifier.cohort_threshold or 0.0,
        delta_hb_window_results=tuple(
            {
                "window_hours": w.window_hours,
                "threshold_g_dl": w.threshold_g_dl,
                "triggered": w.triggered,
                "drop_g_dl": w.drop_g_dl,
            }
            for w in context.hb_result.delta_hb_windows
        ),
        rule_classification=classifier.classification,
        final_classification=classifier.classification,
        cohort_applied=context.cohort_assignment.label.value,
        indications_json=(),
        negative_evidence_json=(),
        confidence=1.0,
        reasoning_summary_thai="",
        reasoning_summary_en=f"deterministic: {classifier.rationale}",
        needs_human_review=False,
        review_reason=None,
        model_id="deterministic",
        prompt_hash=context.prompt_hash,
        evidence_bundle_hash=context.evidence_bundle_hash,
        redactor_version=context.redactor_version,
        redactor_model_sha=context.redactor_model_sha,
        policy_version=context.policy_version,
        verifier_pass=True,
        verifier_retries=0,
        escalated_to_opus=False,
    )


def _deterministic_marker_call(
    *,
    context: PipelineRowContext,
    classifier_result: ClassifierResult,
    run_id: str,
) -> LlmCall:
    """A placeholder :class:`LlmCall` for deterministic-final rows.

    The audit_store's transactional-ordering invariant rejects an
    ``audit_results`` row without a paired ``llm_calls`` row. For the
    deterministic-final path there is no Anthropic call, so we emit
    a stamped marker that records the rule rationale instead.
    """
    from bba.audit_store import LlmCall

    classifier = classifier_result
    fingerprint = hashlib.sha256(
        f"{run_id}|{context.order.audit_id}|deterministic".encode("utf-8")
    ).hexdigest()[:16]
    return LlmCall(
        call_id=f"call-{context.order.audit_id}-det-{fingerprint}",
        audit_id=context.order.audit_id,
        run_id=run_id,
        model_id="deterministic",
        anthropic_version="n/a",
        prompt_cache_id=None,
        request_json={
            "rationale": classifier.rationale,
            "bypass_reason": classifier.bypass_reason.value
            if classifier.bypass_reason
            else "none",
        },
        response_json={
            "classification": classifier.classification,
            "rationale": classifier.rationale,
        },
        request_timestamp=context.order.order_datetime,
        latency_ms=0,
        extended_thinking_blocks=None,
        cold_storage_uri=None,
    )


def _chunked(
    contexts: Sequence[PipelineRowContext], size: int
) -> list[tuple[PipelineRowContext, ...]]:
    return [tuple(contexts[i : i + size]) for i in range(0, len(contexts), size)]


def _batch_id_for(run_id: str, chunk: Sequence[PipelineRowContext]) -> str:
    """Stable batch_id derived from run_id + chunk identity.

    A re-run with the same contexts produces the same batch_id, which
    is safe because :meth:`BatchRunStore.create` rejects duplicates —
    the test for SIGTERM-then-restart relies on this so a second
    invocation surfaces the recoverable state in the store rather
    than orphaning a new pending row alongside the old one.
    """
    digest = hashlib.sha256(
        ("|".join([run_id, *(c.order.audit_id for c in chunk)])).encode("utf-8")
    ).hexdigest()[:16]
    return f"batch-{digest}"


def _build_submission_requests(
    chunk: Sequence[PipelineRowContext], *, run_id: str
) -> list[BatchSubmissionRequest]:
    """Build :class:`BatchSubmissionRequest` objects for ``chunk``.

    Each context MUST carry redacted evidence chunks assembled by the
    caller (via :mod:`bba.evidence_bundle_builder` +
    :mod:`bba.deid_redactor`). The prompt builder is invoked here to
    produce the canonical-envelope :class:`PromptBuildResult`; the
    caller never re-implements that stage.

    Raises :class:`ValueError` for any context with empty
    ``evidence_chunks`` — the LLM path requires redacted evidence, and
    silently inserting a placeholder would let a row route to Anthropic
    with no real content to ground citations against (Codex review
    CRITICAL #2: "fabricates fallback evidence when evidence_chunks is
    empty").
    """
    from bba.prompt_builder import PromptBuildRequest, build_prompt

    out: list[BatchSubmissionRequest] = []
    for context in chunk:
        if not context.evidence_chunks:
            raise ValueError(
                f"audit_id={context.order.audit_id!r}: LLM-required row "
                "has empty evidence_chunks; caller must supply redacted "
                "evidence (bba.evidence_bundle_builder + "
                "bba.deid_redactor) before routing through the pipeline"
            )
        chunks = context.evidence_chunks
        prompt = build_prompt(
            PromptBuildRequest(
                task_mode="HB_7_10_REVIEW",
                cohort_threshold=context.cohort_assignment.threshold
                if context.cohort_assignment.threshold is not None
                else 7.0,
                evidence_chunks=chunks,
                few_shot_examples=(),
            )
        )
        out.append(
            BatchSubmissionRequest(
                audit_id=context.order.audit_id,
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


def _classifier_inputs_for(context: PipelineRowContext) -> ClassifierInputs:
    """Compose the :class:`bba.deterministic_classifier.ClassifierInputs`
    from the upstream-derived data on ``context``.

    The pipeline calls :func:`bba.deterministic_classifier.classify`
    on each context — never delegates that to the caller — so the
    composition chain order (user constraint #1) is structural:
    ``deterministic_classifier`` always runs BEFORE the LLM path
    decision.
    """
    return ClassifierInputs(
        audit_id=context.order.audit_id,
        hb_result=context.hb_result,
        cohort_assignment=context.cohort_assignment,
        order_datetime=context.order.order_datetime,
        procedure_proximity_hours=context.procedure_proximity_hours,
        upcoming_procedure_hours=context.upcoming_procedure_hours,
        crystalloid_liters_prior_4h=context.crystalloid_liters_prior_4h,
        enable_missing_hb_positive_evidence=context.enable_missing_hb_positive_evidence,
    )


def _now_utc() -> datetime:
    return datetime.now(UTC)


__all__ = ["process_audit_order", "run_pipeline"]
