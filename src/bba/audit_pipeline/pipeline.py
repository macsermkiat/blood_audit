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
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

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
    _platelet_source_repr,
    apply_batch_results,
    default_verifier,
)
from bba.audit_pipeline.state_machine import transition
from bba.audit_pipeline.store import BatchRunStore
from bba.audit_store import AuditRow, AuditStore, LlmCall
from bba.cohort_detector import CohortLabel
from bba.deterministic_classifier import (
    BypassReason,
    ClassifierInputs,
    ClassifierResult,
    HB_GT_10_THRESHOLD,
    classify,
    periop_envelope,
)
from bba.llm_client.models import (
    AnthropicTransport,
    BatchSubmissionRequest,
    LlmClientConfig,
)
from bba import feature_flags
from bba.platelet_classifier import (
    PlateletClassifierInputs,
    PlateletClassifierResult,
    classify_platelet,
)

if TYPE_CHECKING:
    from bba.prompt_builder import TaskMode


# Deterministic classifier results that produce a final answer without
# the LLM stage. POTENTIALLY_INAPPROPRIATE and NEEDS_REVIEW route to the
# LLM for a positive-evidence call (PRD §6); APPROPRIATE and
# INSUFFICIENT_EVIDENCE are final at the deterministic layer.
_DETERMINISTIC_FINAL_CLASSIFICATIONS = frozenset(
    {
        "APPROPRIATE",
        "INSUFFICIENT_EVIDENCE",
        "INAPPROPRIATE",
        "RETURNED_NOT_TRANSFUSED",
        "PERIOP_TRANSFUSION_EXEMPT",
    }
)

# Deterministic returns-ledger terminals produced by the RBC classifier
# (:func:`bba.deterministic_classifier.classify`) that must also short-circuit
# the platelet gate — returns are component-agnostic, so an all-returned or
# peri-op-exempt platelet order skips the platelet LLM leg exactly like the RBC
# path. Kept separate from :data:`_DETERMINISTIC_FINAL_CLASSIFICATIONS` so only
# these two returns terminals (never APPROPRIATE / INAPPROPRIATE) divert away
# from ``classify_platelet``.
_RETURNS_TERMINAL_CLASSIFICATIONS = frozenset(
    {"RETURNED_NOT_TRANSFUSED", "PERIOP_TRANSFUSION_EXEMPT"}
)

_RESERVE_AHEAD_RATIONALES = frozenset({"preop_defer_llm", "preop_defer_llm_declared"})

# Returns-terminal classification -> its structured bypass-reason value, for the
# deterministic-platelet marker call's request_json. A platelet result carries
# no BypassReason of its own, so the reason is recovered from the classification.
_RETURNS_TERMINAL_BYPASS: dict[str, str] = {
    "RETURNED_NOT_TRANSFUSED": BypassReason.RETURNED_NOT_TRANSFUSED.value,
    "PERIOP_TRANSFUSION_EXEMPT": BypassReason.PERIOP_TRANSFUSION_EXEMPT.value,
}

# Platelet gate verdicts that route onward to the platelet LLM leg (Stage C2).
# INSUFFICIENT_EVIDENCE is deterministic-final (persisted above); everything
# with a count (POTENTIALLY_INAPPROPRIATE / NEEDS_REVIEW) defers to the LLM.
_PLATELET_LLM_CLASSIFICATIONS = frozenset({"POTENTIALLY_INAPPROPRIATE", "NEEDS_REVIEW"})


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

    # Separate MTP-suppressed platelet rows first. MTP-suppressed platelet units
    # are co-ordered inside an active massive-transfusion-protocol window and are
    # intentionally not audit rows — they neither persist nor become orphans.
    active_contexts: list[PipelineRowContext] = []
    for ctx in contexts:
        if ctx.component == "platelet" and ctx.platelet_mtp_suppressed:
            continue  # suppress: no AuditRow emitted
        active_contexts.append(ctx)

    # Platelet bucket (Phase 2 dispatch).
    platelet_classified: list[tuple[PipelineRowContext, PlateletClassifierResult]] = []

    # Call the deterministic classifier ON EACH context (user
    # constraint #1: the pipeline composes the deterministic engine —
    # callers no longer pre-compute it). Partition into the two
    # buckets based on the classifier's verdict.
    classified: list[tuple[PipelineRowContext, ClassifierResult]] = []
    deterministic_final: list[tuple[PipelineRowContext, ClassifierResult]] = []
    llm_required: list[tuple[PipelineRowContext, ClassifierResult]] = []
    for ctx in active_contexts:
        if ctx.component == "platelet":
            returns_result = classify(_classifier_inputs_for(ctx))
            if returns_result.classification in _RETURNS_TERMINAL_CLASSIFICATIONS:
                platelet_classified.append(
                    (
                        ctx,
                        PlateletClassifierResult(
                            classification=returns_result.classification,
                            review_ceiling=None,
                            rationale=returns_result.rationale,
                        ),
                    )
                )
            else:
                plt_inputs = PlateletClassifierInputs(
                    audit_id=ctx.order.audit_id,
                    platelet_count=ctx.platelet_result.value_k_ul
                    if ctx.platelet_result is not None
                    else None,
                    enable_missing_platelet_defer=pipeline_config.enable_missing_platelet_defer,
                )
                platelet_classified.append((ctx, classify_platelet(plt_inputs)))
        else:
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

    # Persist deterministic-final platelet rows.
    # POTENTIALLY_INAPPROPRIATE and NEEDS_REVIEW route onward (Stage C wires LLM).
    for ctx, plt_result in platelet_classified:
        if plt_result.classification in _DETERMINISTIC_FINAL_CLASSIFICATIONS:
            if _persist_deterministic_platelet_row(
                ctx,
                classifier_result=plt_result,
                audit_store=audit_store,
                run_id=run_id,
            ):
                persisted.append(ctx.order.audit_id)
        # else: POTENTIALLY_INAPPROPRIATE / NEEDS_REVIEW → Stage C; orphan in Stage B

    llm_contexts_only = [ctx for ctx, _result in llm_required]
    llm_results_by_id: dict[str, ClassifierResult] = {
        ctx.order.audit_id: result for ctx, result in llm_required
    }

    for chunk_contexts in _chunked(llm_contexts_only, pipeline_config.max_batch_size):
        chunk_results = {
            ctx.order.audit_id: llm_results_by_id[ctx.order.audit_id]
            for ctx in chunk_contexts
        }
        _run_batch_chunk(
            chunk_contexts,
            run_id=run_id,
            classifier_results=chunk_results,
            transport=transport,
            audit_store=audit_store,
            batch_run_store=batch_run_store,
            llm_config=llm_config,
            pipeline_config=pipeline_config,
            verifier=verifier,
            enable_missing_platelet_defer=False,
            persisted=persisted,
            touched=touched,
        )

    # Platelet LLM leg (Stage C2, feature-flagged). Runs as SEPARATE batches
    # AFTER the RBC loop so the RBC batch composition (chunking + batch_id) is
    # byte-identical whether or not the platelet leg is enabled. With the flag
    # off, POTENTIALLY_INAPPROPRIATE / NEEDS_REVIEW platelet rows simply orphan
    # (Stage B behaviour); the deterministic-final INSUFFICIENT_EVIDENCE rows
    # already persisted above.
    if feature_flags.PLATELET_LLM_ENABLED:
        platelet_llm_contexts = [
            ctx
            for ctx, plt_result in platelet_classified
            if plt_result.classification in _PLATELET_LLM_CLASSIFICATIONS
        ]
        for chunk_contexts in _chunked(
            platelet_llm_contexts, pipeline_config.max_batch_size
        ):
            _run_batch_chunk(
                chunk_contexts,
                run_id=run_id,
                classifier_results={},
                transport=transport,
                audit_store=audit_store,
                batch_run_store=batch_run_store,
                llm_config=llm_config,
                pipeline_config=pipeline_config,
                verifier=verifier,
                enable_missing_platelet_defer=pipeline_config.enable_missing_platelet_defer,
                persisted=persisted,
                touched=touched,
            )

    orphans = _detect_orphans(
        expected_audit_ids=tuple(c.order.audit_id for c in active_contexts),
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


def _run_batch_chunk(
    chunk_contexts: Sequence[PipelineRowContext],
    *,
    run_id: str,
    classifier_results: dict[str, ClassifierResult],
    transport: AnthropicTransport,
    audit_store: AuditStore,
    batch_run_store: BatchRunStore,
    llm_config: LlmClientConfig,
    pipeline_config: AuditPipelineConfig,
    verifier: Verifier,
    enable_missing_platelet_defer: bool,
    persisted: list[str],
    touched: list[str],
) -> None:
    """Submit one chunk as a Batch run and apply its results.

    Shared by the RBC and platelet LLM legs so the crash-recovery checkpoint
    state machine (create PENDING -> submit -> SUBMITTED -> fetch -> apply ->
    COMPLETE) has ONE implementation. ``persisted`` and ``touched`` are the
    run-level accumulators, mutated in the SAME order the original inline RBC
    loop did: ``touched`` records the batch_id immediately after the PENDING row
    is created — before any fallible Anthropic step — so a SIGTERM during the
    polling window leaves a recoverable SUBMITTED (or PENDING) checkpoint the
    resume reconciler can poll, never a stranded row.

    ``enable_missing_platelet_defer`` is forwarded to
    :func:`apply_batch_results`; it is inert for RBC rows (their
    ``rule_classification`` never comes from the platelet gate) so the RBC leg
    passes ``False`` and stays byte-identical."""
    # Codex P1 (security): build submission requests FIRST so injection-flagged
    # contexts are routed to NEEDS_REVIEW before a BatchRun is created.  A
    # BatchRun only covers the clean (non-injection) contexts.
    injection_sink: list[PipelineRowContext] = []
    requests = _build_submission_requests(
        chunk_contexts,
        run_id=run_id,
        classifier_results=classifier_results,
        injection_sink=injection_sink,
    )

    for inj_ctx in injection_sink:
        _persist_injection_flagged_row(
            inj_ctx,
            classifier_result=classifier_results.get(inj_ctx.order.audit_id),
            audit_store=audit_store,
            run_id=run_id,
            enable_missing_platelet_defer=enable_missing_platelet_defer,
        )
        persisted.append(inj_ctx.order.audit_id)

    if not requests:
        # All contexts in this chunk were injection-flagged; no Anthropic batch
        # is needed.  The rows were persisted above as NEEDS_REVIEW.
        return

    # Remaining contexts are clean: compute the batch from clean contexts only.
    clean_audit_ids = frozenset(r.audit_id for r in requests)
    clean_contexts = [c for c in chunk_contexts if c.order.audit_id in clean_audit_ids]

    batch_id = _batch_id_for(run_id, clean_contexts)
    now = _now_utc()
    pending = BatchRun(
        batch_id=batch_id,
        state=BatchRunState.PENDING,
        run_id=run_id,
        code_version=pipeline_config.code_version,
        audit_ids=tuple(c.order.audit_id for c in clean_contexts),
        updated_at=now,
    )
    batch_run_store.create(pending)
    touched.append(batch_id)

    # Split-phase submission (PR #54 codex P1 fix): create the remote batch and
    # persist the anthropic_batch_id BEFORE polling. A SIGTERM during the polling
    # window now leaves a SUBMITTED row whose batch_id the resume reconciler can
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

    context_map = {ctx.order.audit_id: ctx for ctx in clean_contexts}
    write_summary = apply_batch_results(
        response,
        audit_store=audit_store,
        run_id=run_id,
        contexts=context_map,
        classifier_results=classifier_results,
        verifier=verifier,
        enable_missing_platelet_defer=enable_missing_platelet_defer,
    )
    persisted.extend(write_summary.audit_ids_persisted)

    completed = transition(
        submitted,
        to_state=BatchRunState.COMPLETE,
        now=_now_utc(),
    )
    batch_run_store.update(completed)


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

    # Defensive guard: platelet rows must use _deterministic_platelet_audit_row.
    # The run_pipeline dispatch prevents platelet rows from reaching here, but
    # this guard ensures a platelet context never trips the Hb=None raise below.
    if context.component == "platelet":
        plt_inputs = PlateletClassifierInputs(
            audit_id=context.order.audit_id,
            platelet_count=context.platelet_result.value_k_ul
            if context.platelet_result is not None
            else None,
        )
        return _deterministic_platelet_audit_row(
            context=context,
            classifier_result=classify_platelet(plt_inputs),
            run_id=run_id,
        )

    # INSUFFICIENT_EVIDENCE legitimately has a missing Hb; APPROPRIATE
    # with an Hb-independent bypass (MTP, peri-procedural, or hard peri-op
    # evidence) also has a legitimately missing Hb — both use 0.0 +
    # freshness=="missing" as the explicit "no Hb" sentinel (PRD §"Output
    # schema"). All other non-INSUFFICIENT classifications must supply real
    # Hb — fail loud.
    _HB_INDEPENDENT_BYPASSES = frozenset(
        {
            BypassReason.MTP,
            BypassReason.PERI_PROCEDURAL_6H,
            BypassReason.PERIOP_EVIDENCE,
            BypassReason.RETURNED_NOT_TRANSFUSED,
            BypassReason.PERIOP_TRANSFUSION_EXEMPT,
        }
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


def _deterministic_platelet_audit_row(
    *,
    context: PipelineRowContext,
    classifier_result: PlateletClassifierResult,
    run_id: str,
) -> AuditRow:
    """AuditRow for a deterministic-final platelet order.

    Sets component="platelet" and populates platelet_* fields.
    The Hb-shaped required fields hold their missing-sentinels.
    """
    plt = context.platelet_result
    return AuditRow(
        audit_id=context.order.audit_id,
        run_id=run_id,
        run_timestamp=context.order.order_datetime,
        hn_hash=context.hn_hash,
        an_hash=context.an_hash,
        reqno=context.order.reqno,
        order_datetime=context.order.order_datetime,
        products_ordered=tuple(context.order.products_ordered),
        hb_value=0.0,
        hb_datetime=context.order.order_datetime,
        hb_freshness="missing",
        hb_source="missing",
        vitals_sbp=None,
        vitals_hr=None,
        vitals_timestamp=None,
        vitals_source=None,
        prior_rbc_units_24h=context.prior_rbc_units_24h,
        prior_rbc_units_7d=context.prior_rbc_units_7d,
        cohort_threshold=0.0,
        delta_hb_window_results=(),
        rule_classification=classifier_result.classification,
        final_classification=classifier_result.classification,
        cohort_applied=CohortLabel.UNKNOWN.value,
        indications_json=(),
        negative_evidence_json=(),
        confidence=1.0,
        reasoning_summary_thai="",
        reasoning_summary_en=f"deterministic: {classifier_result.rationale}",
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
        component="platelet",
        platelet_value=plt.value_k_ul if plt is not None else None,
        platelet_datetime=plt.datetime_utc if plt is not None else None,
        platelet_freshness=plt.freshness if plt is not None else None,
        platelet_source=_platelet_source_repr(plt.source if plt is not None else None),
        platelet_review_ceiling=classifier_result.review_ceiling,
    )


def _platelet_marker_call(
    *,
    context: PipelineRowContext,
    classifier_result: PlateletClassifierResult,
    run_id: str,
) -> LlmCall:
    """A placeholder LlmCall for deterministic-final platelet rows."""
    fingerprint = hashlib.sha256(
        f"{run_id}|{context.order.audit_id}|deterministic-plt".encode("utf-8")
    ).hexdigest()[:16]
    return LlmCall(
        call_id=f"call-{context.order.audit_id}-det-plt-{fingerprint}",
        audit_id=context.order.audit_id,
        run_id=run_id,
        model_id="deterministic",
        anthropic_version="n/a",
        prompt_cache_id=None,
        request_json={
            "rationale": classifier_result.rationale,
            "bypass_reason": _RETURNS_TERMINAL_BYPASS.get(
                classifier_result.classification, "none"
            ),
            "component": "platelet",
        },
        response_json={
            "classification": classifier_result.classification,
            "rationale": classifier_result.rationale,
        },
        request_timestamp=context.order.order_datetime,
        latency_ms=0,
        extended_thinking_blocks=None,
        cold_storage_uri=None,
    )


def _persist_deterministic_platelet_row(
    context: PipelineRowContext,
    *,
    classifier_result: PlateletClassifierResult,
    audit_store: AuditStore,
    run_id: str,
) -> bool:
    """Persist a deterministic-final platelet row (INSUFFICIENT_EVIDENCE only).

    Returns True iff a new row was committed.
    """
    row = _deterministic_platelet_audit_row(
        context=context,
        classifier_result=classifier_result,
        run_id=run_id,
    )
    marker_call = _platelet_marker_call(
        context=context, classifier_result=classifier_result, run_id=run_id
    )
    write_result = audit_store.write(row, [marker_call])
    return not write_result.skipped_idempotent


def _persist_injection_flagged_row(
    context: PipelineRowContext,
    *,
    classifier_result: ClassifierResult | None,
    audit_store: AuditStore,
    run_id: str,
    enable_missing_platelet_defer: bool = False,
) -> None:
    """Persist a NEEDS_REVIEW row for an injection-flagged context.

    The prompt builder's injection scanner raised a flag; the context
    is routed to NEEDS_REVIEW without an Anthropic call so the
    poisoned evidence chunk never reaches the model (Codex P1 security).
    A synthetic marker LlmCall satisfies the audit_store transactional
    invariant (every audit_results row must have a paired llm_calls row).

    Only ``final_classification`` (NEEDS_REVIEW) and ``review_reason`` (the
    injection constant) are forced; ``rule_classification`` must stay the TRUE
    deterministic verdict so the audit trail is not skewed (Codex round-5 P2).
    For an RBC context the verdict is the ``classifier_result`` the batch path
    already computed (POTENTIALLY_INAPPROPRIATE / NEEDS_REVIEW — deterministic-
    final RBC rows never reach the batch path). For a PLATELET context the RBC
    classifier is absent (the platelet leg submits with ``classifier_results={}``)
    and would be wrong anyway; :func:`_audit_row_for_needs_review` re-derives the
    platelet ``rule_classification`` from the platelet gate, so
    ``enable_missing_platelet_defer`` is threaded through for that gate rather
    than running the RBC ``classify()`` on a platelet sentinel Hb. The fallback
    sentinel is only reached when no real verdict is available; its
    ``classification`` is ignored for platelet (gate overrides) and its
    ``cohort_threshold=None`` reproduces the prior 0.0 fallback for the field.
    """
    from bba.audit_pipeline.replay import _audit_row_for_needs_review

    deterministic_result = classifier_result or ClassifierResult(
        classification="INSUFFICIENT_EVIDENCE",
        rationale="injection_detected",
        cohort_threshold=None,
        bypass_reason=BypassReason.NONE,
    )
    row = _audit_row_for_needs_review(
        run_id=run_id,
        context=context,
        classifier_result=deterministic_result,
        review_reason="injection_detected",
        verifier_pass=False,
        verifier_retries=0,
        model_id="injection-filter",
        reasoning_en="Prompt injection detected in evidence chunks; not submitted to LLM.",
        reasoning_th="",
        indications=(),
        negative_evidence=(),
        confidence=0.0,
        escalated=False,
        enable_missing_platelet_defer=enable_missing_platelet_defer,
    )
    fingerprint = hashlib.sha256(
        f"{run_id}|{context.order.audit_id}|injection-filtered".encode("utf-8")
    ).hexdigest()[:16]
    marker_call = LlmCall(
        call_id=f"call-{context.order.audit_id}-inj-{fingerprint}",
        audit_id=context.order.audit_id,
        run_id=run_id,
        model_id="injection-filter",
        anthropic_version="n/a",
        prompt_cache_id=None,
        request_json={
            "injection_detected": True,
            "audit_id": context.order.audit_id,
        },
        response_json={
            "classification": "NEEDS_REVIEW",
            "review_reason": "injection_detected",
        },
        request_timestamp=context.order.order_datetime,
        latency_ms=0,
        extended_thinking_blocks=None,
        cold_storage_uri=None,
    )
    audit_store.write(row, [marker_call])


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


def rbc_task_mode(
    hb_value_g_dl: float | None, *, reserve_ahead: bool = False
) -> TaskMode:
    """Select the RBC task mode from reserve-ahead status and measured Hb.

    ``reserve_ahead=True`` selects ``RESERVE_AHEAD_REVIEW``. Otherwise the
    existing Hb-keyed behavior is unchanged:

    ``HB_GT_10_OVERRIDE`` at or above ``HB_GT_10_THRESHOLD`` (10.0),
    matching the deterministic engine exactly: its ``hb_ge_10`` branch
    assigns POTENTIALLY_INAPPROPRIATE for Hb >= 10.0 while the
    ``hb_7_to_10`` gray zone is strictly below 10.0
    (:mod:`bba.deterministic_classifier.classifier`). A missing Hb
    (None) cannot establish Hb >= 10 and stays on the gray-zone
    prompt. The callers derive ``reserve_ahead`` from the default-off feature
    flag AND the deterministic ``preop_defer_llm`` rationale; the router does
    not duplicate that policy predicate.

    Single source of truth for the production pipeline, the resume
    rebuild path (:mod:`bba.audit_pipeline.resume`) and the pilot
    runner (scripts/pilot/run_llm_leg.py) — three-way drift across
    those call sites is what caused the #93 dispatch bug.
    """
    if reserve_ahead:
        return "RESERVE_AHEAD_REVIEW"
    if hb_value_g_dl is not None and hb_value_g_dl >= HB_GT_10_THRESHOLD:
        return "HB_GT_10_OVERRIDE"
    return "HB_7_10_REVIEW"


def _build_submission_requests(
    chunk: Sequence[PipelineRowContext],
    *,
    run_id: str,
    classifier_results: Mapping[str, ClassifierResult] | None = None,
    injection_sink: list[PipelineRowContext] | None = None,
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

    ``injection_sink``: when provided, contexts whose prompt builder
    raises the injection flag (``route_to_needs_review=True``) are
    appended here instead of to the submission list.  The caller is
    responsible for persisting them as NEEDS_REVIEW without a batch
    submission (Codex P1 security fix).
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
        task_mode: TaskMode
        if context.component == "platelet":
            # Platelet review has no Hb cohort threshold: cohort_threshold=None
            # is required (PromptBuildRequest validator) and build_prompt selects
            # the platelet system prompt for PLATELET_REVIEW (Stage C2).
            task_mode = "PLATELET_REVIEW"
            prompt = build_prompt(
                PromptBuildRequest(
                    task_mode="PLATELET_REVIEW",
                    cohort_threshold=None,
                    evidence_chunks=chunks,
                    few_shot_examples=(),
                )
            )
        else:
            classifier_result = (
                classifier_results.get(context.order.audit_id)
                if classifier_results is not None
                else None
            )
            reserve_ahead = (
                feature_flags.RESERVE_AHEAD_ROUTER_ENABLED
                and classifier_result is not None
                and classifier_result.rationale in _RESERVE_AHEAD_RATIONALES
            )
            task_mode = rbc_task_mode(
                context.hb_result.value_g_dl, reserve_ahead=reserve_ahead
            )
            prompt = build_prompt(
                PromptBuildRequest(
                    task_mode=task_mode,
                    cohort_threshold=context.cohort_assignment.threshold
                    if context.cohort_assignment.threshold is not None
                    else 7.0,
                    evidence_chunks=chunks,
                    few_shot_examples=(),
                )
            )

        # Codex P1 (security): prompt builder flagged injection in the evidence
        # chunks.  Route this context to NEEDS_REVIEW without submitting to
        # Anthropic — a poisoned chunk must never reach the model.
        if prompt.route_to_needs_review:
            if injection_sink is not None:
                injection_sink.append(context)
            continue

        out.append(
            BatchSubmissionRequest(
                audit_id=context.order.audit_id,
                run_id=run_id,
                task_mode=task_mode,
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
    periop = context.periop_summary
    return ClassifierInputs(
        audit_id=context.order.audit_id,
        hb_result=context.hb_result,
        cohort_assignment=context.cohort_assignment,
        order_datetime=context.order.order_datetime,
        procedure_proximity_hours=context.procedure_proximity_hours,
        upcoming_procedure_hours=context.upcoming_procedure_hours,
        crystalloid_liters_prior_4h=context.crystalloid_liters_prior_4h,
        enable_missing_hb_positive_evidence=context.enable_missing_hb_positive_evidence,
        periop_blood_loss_ml=periop.blood_loss_ml if periop else None,
        periop_intraop_transfusion=periop.intraop_transfusion if periop else False,
        periop_surgical_context=periop.surgical_context if periop else False,
        returns_disposition=(
            context.returns_summary.disposition
            if feature_flags.RETURNS_LEDGER_ENABLED
            and context.returns_summary is not None
            else "inconclusive"
        ),
        returns_periop_context=(
            periop_envelope(
                surgical_context=periop.surgical_context if periop else False,
                intraop_transfusion=periop.intraop_transfusion if periop else False,
                procedure_proximity_hours=context.procedure_proximity_hours,
                upcoming_procedure_hours=context.upcoming_procedure_hours,
            )
            if feature_flags.RETURNS_LEDGER_ENABLED
            and context.returns_summary is not None
            else False
        ),
        declared_use=(
            context.declared_use if feature_flags.DECLARED_USETYPE_ENABLED else None
        ),
    )


def _now_utc() -> datetime:
    return datetime.now(UTC)


__all__ = ["process_audit_order", "run_pipeline"]
