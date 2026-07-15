"""Resume-on-startup reconciler (issue #24 AC ③, user constraint #5).

Hard requirement: the pipeline must survive a SIGTERM mid-batch and
restart without losing or duplicating work. The reconciler handles
three distinct failure-window classes:

1. **PENDING with no anthropic_batch_id** — local create landed but
   submission never happened. Without a batch_id there is nothing to
   poll; transition the row to FAILED with an operator-visible
   error_message. Re-submission is the operator's call (a re-run of
   :func:`bba.audit_pipeline.run_pipeline` with the same ``run_id``
   is idempotent via audit_store's commit-marker).

2. **SUBMITTED / PARTIAL with cached llm_calls** — the orphan-call
   case (PRD §10): phase 1 of audit_store.write succeeded but phase
   2 didn't. Each orphan is re-emitted through
   :func:`bba.audit_pipeline.replay.apply_batch_results` using the
   cached :class:`LlmCall.response_json` payload.

3. **SUBMITTED / PARTIAL with NO cached llm_calls** — the
   "Anthropic-polling crash" case (codex PR #54 P1 finding): the
   pipeline persisted the anthropic_batch_id but died before any
   results were retrieved. The reconciler calls
   :meth:`AnthropicTransport.fetch_batch_results` on the cached
   batch_id to pull the results down, then routes them through
   :func:`apply_batch_results`. Without this step, repeated resumes
   would keep declaring the row failed even though Anthropic has
   completed the batch.

The reconciler is idempotent — a second run produces zero additional
writes (audit_store's :class:`WriteResult.skipped_idempotent`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from bba import feature_flags
from bba.audit_pipeline.models import (
    AuditPipelineConfig,
    BatchRun,
    BatchRunState,
    PipelineRowContext,
    ResumeReport,
)
from bba.audit_pipeline.pipeline import _classifier_inputs_for, rbc_task_mode
from bba.audit_pipeline.state_machine import is_terminal, transition
from bba.audit_pipeline.store import BatchRunStore
from bba.audit_store import AuditStore, LlmCall
from bba.deterministic_classifier import ClassifierResult, classify
from bba.llm_client.models import (
    AnthropicTransport,
    BatchSubmissionRequest,
    BatchSubmissionResult,
    LlmClientConfig,
    RawBatchResponse,
)


_RESERVE_AHEAD_RATIONALES = frozenset({"preop_defer_llm", "preop_defer_llm_declared"})


def resume_on_startup(
    *,
    batch_run_store: BatchRunStore,
    audit_store: AuditStore,
    contexts: Mapping[str, PipelineRowContext] | None = None,
    transport: AnthropicTransport | None = None,
    llm_config: LlmClientConfig | None = None,
    pipeline_config: AuditPipelineConfig | None = None,
) -> ResumeReport:
    """Reconcile every non-terminal ``batch_runs`` row and return a report.

    ``contexts`` maps ``audit_id`` to a :class:`PipelineRowContext` so
    the reconciler can rebuild :class:`bba.audit_store.AuditRow` rows
    from cached LLM responses without re-querying upstream tables.
    Missing contexts surface as ``failed_audit_ids`` (the orphan is
    real but the reconciler has no upstream snapshot to render it).

    ``transport`` is the production :class:`AnthropicTransport` (or a
    cassette in tests). When provided, the reconciler polls
    SUBMITTED/PARTIAL batches that have no cached llm_calls yet —
    this is the failure-window codex flagged on PR #54 (SIGTERM
    during Anthropic polling left the row stranded indefinitely).
    When ``transport`` is ``None``, the no-cached-calls case falls
    through to ``failed_audit_ids`` for operator action.

    ``llm_config`` is consulted for the model id used to rebuild the
    submission set that :meth:`fetch_batch_results` needs. Default
    behaviour (None) skips Anthropic polling entirely.

    Returns a typed report:

    * ``polled_batch_ids`` — every batch_run touched by the resume.
    * ``completed_audit_ids`` — audit_ids whose audit_results row is
      now committed.
    * ``reemitted_audit_ids`` — orphan calls whose audit_row was
      written by this resume pass.
    * ``failed_audit_ids`` — audit_ids that cannot be reconciled
      without operator action.
    """
    contexts = contexts or {}

    polled: list[str] = []
    completed: list[str] = []
    reemitted: list[str] = []
    failed: list[str] = []

    for run in batch_run_store.list_all():
        if is_terminal(run.state):
            continue

        polled.append(run.batch_id)

        if run.state is BatchRunState.PENDING:
            failed.extend(run.audit_ids)
            failed_run = transition(
                run,
                to_state=BatchRunState.FAILED,
                error_message=(
                    "PENDING batch had no Anthropic submission; "
                    "resume found a local create with no batch_id to poll"
                ),
                now=_now_utc(),
            )
            batch_run_store.update(failed_run)
            continue

        run_completed, run_reemitted, run_failed = _reconcile_submitted_or_partial(
            run,
            batch_run_store=batch_run_store,
            audit_store=audit_store,
            contexts=contexts,
            transport=transport,
            llm_config=llm_config,
            pipeline_config=pipeline_config,
        )
        completed.extend(run_completed)
        reemitted.extend(run_reemitted)
        failed.extend(run_failed)

    return ResumeReport(
        polled_batch_ids=tuple(polled),
        completed_audit_ids=tuple(completed),
        reemitted_audit_ids=tuple(reemitted),
        failed_audit_ids=tuple(failed),
    )


def _reconcile_submitted_or_partial(
    run: BatchRun,
    *,
    batch_run_store: BatchRunStore,
    audit_store: AuditStore,
    contexts: Mapping[str, PipelineRowContext],
    transport: AnthropicTransport | None,
    llm_config: LlmClientConfig | None,
    pipeline_config: AuditPipelineConfig | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Reconcile one SUBMITTED/PARTIAL row.

    Returns ``(completed, reemitted, failed)`` lists for the row. The
    caller folds these into the top-level report.
    """
    cached_calls = _cached_calls_by_audit_id(audit_store, run_id=run.run_id)
    persisted_audit_ids = {
        row.audit_id for row in audit_store.read_audit_results(run_id=run.run_id)
    }

    audit_ids_without_audit_row = [
        aid for aid in run.audit_ids if aid not in persisted_audit_ids
    ]
    audit_ids_missing_cached_calls = [
        aid for aid in audit_ids_without_audit_row if not cached_calls.get(aid)
    ]

    # Codex PR #54 P1: poll Anthropic for batches whose batch_id is
    # known but whose results never landed. Without this step the
    # SIGTERM-during-polling crash window stays stranded.
    if (
        audit_ids_missing_cached_calls
        and transport is not None
        and llm_config is not None
        and run.anthropic_batch_id is not None
    ):
        try:
            response = _fetch_response_for_batch(
                run,
                contexts=contexts,
                transport=transport,
                llm_config=llm_config,
                audit_ids=audit_ids_missing_cached_calls,
            )
        except Exception:
            # Anthropic-side failure (timeout, custom_id drift, SDK
            # error) — leave the row in its current state; the next
            # resume retries. Don't escalate to FAILED here because
            # the cause may be transient (network blip, Anthropic
            # rate limit).
            response = None
        if response is not None:
            from bba.audit_pipeline.replay import apply_batch_results

            polled_contexts = {
                aid: contexts[aid]
                for aid in audit_ids_missing_cached_calls
                if aid in contexts
            }
            if polled_contexts:
                _defer = (
                    pipeline_config.enable_missing_platelet_defer
                    if pipeline_config is not None
                    else False
                )
                apply_batch_results(
                    response,
                    audit_store=audit_store,
                    run_id=run.run_id,
                    contexts=polled_contexts,
                    enable_missing_platelet_defer=_defer,
                )
                # Refresh cached state — the call(s) and audit row(s)
                # are now persisted.
                cached_calls = _cached_calls_by_audit_id(audit_store, run_id=run.run_id)
                persisted_audit_ids = {
                    row.audit_id
                    for row in audit_store.read_audit_results(run_id=run.run_id)
                }

    run_completed: list[str] = []
    run_reemitted: list[str] = []
    run_failed: list[str] = []

    for audit_id in run.audit_ids:
        if audit_id in persisted_audit_ids:
            run_completed.append(audit_id)
            continue
        calls = cached_calls.get(audit_id, ())
        if calls:
            if audit_id not in contexts:
                run_failed.append(audit_id)
                continue
            _re_emit_audit_row(
                audit_id=audit_id,
                calls=calls,
                context=contexts[audit_id],
                audit_store=audit_store,
                run_id=run.run_id,
                pipeline_config=pipeline_config,
            )
            run_reemitted.append(audit_id)
            run_completed.append(audit_id)
        else:
            run_failed.append(audit_id)

    if not run_failed and len(set(run.audit_ids)) == len(
        {aid for aid in run.audit_ids if aid in persisted_audit_ids}
        | set(run_reemitted)
    ):
        completed_run = transition(run, to_state=BatchRunState.COMPLETE, now=_now_utc())
        batch_run_store.update(completed_run)
    elif run.state is BatchRunState.SUBMITTED:
        partial_run = transition(run, to_state=BatchRunState.PARTIAL, now=_now_utc())
        batch_run_store.update(partial_run)

    return run_completed, run_reemitted, run_failed


def _fetch_response_for_batch(
    run: BatchRun,
    *,
    contexts: Mapping[str, PipelineRowContext],
    transport: AnthropicTransport,
    llm_config: LlmClientConfig,
    audit_ids: Sequence[str],
) -> RawBatchResponse | None:
    """Rebuild the submission set + call :meth:`fetch_batch_results`.

    Returns ``None`` when the reconciler can't synthesise the
    submission set (missing contexts for every audit_id in
    ``audit_ids``). The caller then leaves the batch_run in its
    current state for the next resume to retry.
    """
    requests = _rebuild_submission_requests(
        run=run, contexts=contexts, audit_ids=audit_ids
    )
    if not requests:
        return None
    assert run.anthropic_batch_id is not None
    return transport.fetch_batch_results(
        run.anthropic_batch_id,
        model=llm_config.sonnet_model_id,
        requests=requests,
        prompt_cache_enabled=llm_config.prompt_cache_enabled,
    )


def _rebuild_submission_requests(
    *,
    run: BatchRun,
    contexts: Mapping[str, PipelineRowContext],
    audit_ids: Sequence[str],
) -> list[BatchSubmissionRequest]:
    """Reconstruct the :class:`BatchSubmissionRequest` list for an
    in-flight batch by re-running :mod:`bba.prompt_builder` against
    the supplied contexts.

    The prompt builder is a pure function over canonical inputs so
    the rebuild produces the same prompt_hash + envelope_hash bytes
    the submission used — :func:`apply_batch_results` then accepts
    the response without contract drift.

    Branches on ``context.component``:

    * ``"platelet"`` → ``PLATELET_REVIEW`` with ``cohort_threshold=None``,
      mirroring :func:`bba.audit_pipeline.pipeline._build_submission_requests`.
      Without this branch a resumed platelet batch would be rebuilt as an RBC
      request (wrong prompt, wrong tool schema, wrong threshold).

    * anything else → Hb-keyed selection: Hb >= 10.0 uses
      ``HB_GT_10_OVERRIDE``; otherwise ``HB_7_10_REVIEW``, via
      :func:`bba.audit_pipeline.pipeline.rbc_task_mode`, mirroring the live
      submission path.

    The byte-identity invariant holds only when the resuming process runs
    the same code version that submitted the batch. Resuming an in-flight
    batch across a prompt- or dispatch-changing upgrade would pair the old
    responses with new request bytes; cross-version recovery uses a fresh
    run (new ``run_id``/``code_version``) instead.
    """
    from bba.prompt_builder import PromptBuildRequest, TaskMode, build_prompt

    requests: list[BatchSubmissionRequest] = []
    for audit_id in audit_ids:
        ctx = contexts.get(audit_id)
        if ctx is None or not ctx.evidence_chunks:
            continue
        task_mode: TaskMode
        if ctx.component == "platelet":
            # Platelet resume: cohort_threshold=None, platelet system prompt.
            # Mirrors pipeline._build_submission_requests's platelet branch so
            # the rebuilt prompt_hash + envelope_hash stay byte-identical.
            task_mode = "PLATELET_REVIEW"
            prompt = build_prompt(
                PromptBuildRequest(
                    task_mode="PLATELET_REVIEW",
                    cohort_threshold=None,
                    evidence_chunks=ctx.evidence_chunks,
                    few_shot_examples=(),
                )
            )
        else:
            # RBC / default: mirrors live submission via the shared selector.
            # Compose classifier inputs through the SAME production composer the
            # live pipeline uses (``_classifier_inputs_for``) with NO returns
            # override (spec #119, ticket #124). #122/#123 forced
            # returns_disposition="inconclusive" + returns_periop_context=False
            # here; #124 removes that force so resume re-derives returns routing
            # identically to run/replay — no divergence across run / resume /
            # legs. A returned/exempt order is terminal at submission time and is
            # therefore never in ``audit_ids`` here, so it is never re-submitted;
            # for any order that WAS submitted (non-terminal), the real returns
            # disposition yields the same classifier rationale as the forced
            # "inconclusive" did, so the rebuilt prompt stays byte-identical.
            # Flag-off, ``_classifier_inputs_for`` already yields inconclusive /
            # False, so today's task-mode selection is unchanged.
            reserve_ahead = False
            if feature_flags.RESERVE_AHEAD_ROUTER_ENABLED:
                reserve_ahead = (
                    classify(_classifier_inputs_for(ctx)).rationale
                    in _RESERVE_AHEAD_RATIONALES
                )
            task_mode = rbc_task_mode(
                ctx.hb_result.value_g_dl, reserve_ahead=reserve_ahead
            )
            threshold = (
                ctx.cohort_assignment.threshold
                if ctx.cohort_assignment.threshold is not None
                else 7.0
            )
            prompt = build_prompt(
                PromptBuildRequest(
                    task_mode=task_mode,
                    cohort_threshold=threshold,
                    evidence_chunks=ctx.evidence_chunks,
                    few_shot_examples=(),
                )
            )
        requests.append(
            BatchSubmissionRequest(
                audit_id=audit_id,
                run_id=run.run_id,
                task_mode=task_mode,
                prompt=prompt,
            )
        )
    return requests


def _cached_calls_by_audit_id(
    audit_store: AuditStore, *, run_id: str
) -> dict[str, tuple[LlmCall, ...]]:
    """Group every cached ``llm_calls`` row for ``run_id`` by audit_id."""
    grouped: dict[str, list[LlmCall]] = {}
    for call in audit_store.read_llm_calls(run_id=run_id):
        grouped.setdefault(call.audit_id, []).append(call)
    return {aid: tuple(calls) for aid, calls in grouped.items()}


def _re_emit_audit_row(
    *,
    audit_id: str,
    calls: tuple[LlmCall, ...],
    context: PipelineRowContext,
    audit_store: AuditStore,
    run_id: str,
    pipeline_config: AuditPipelineConfig | None = None,
) -> None:
    """Rebuild an audit_row from cached call responses + context.

    Synthesises a :class:`RawBatchResponse` from the cached
    ``llm_calls`` rows and feeds it back through
    :func:`bba.audit_pipeline.replay.apply_batch_results` — the same
    code path that the original submission used. The audit_store's
    idempotency contract guarantees re-running this for an already-
    persisted audit_id is a no-op.

    ``pipeline_config`` threads ``enable_missing_platelet_defer`` so that
    a missing-count platelet order recomputes the same verdict as the
    original submission rather than defaulting to INSUFFICIENT_EVIDENCE
    (Codex P2 resume drift fix).
    """
    from bba.audit_pipeline.replay import apply_batch_results

    results = tuple(_result_from_cached_call(call) for call in calls)
    synthetic_response = RawBatchResponse(
        batch_id=f"resume-{audit_id}",
        results=results,
    )
    _defer = (
        pipeline_config.enable_missing_platelet_defer
        if pipeline_config is not None
        else False
    )
    apply_batch_results(
        synthetic_response,
        audit_store=audit_store,
        run_id=run_id,
        contexts={audit_id: context},
        enable_missing_platelet_defer=_defer,
    )


def _result_from_cached_call(call: LlmCall) -> BatchSubmissionResult:
    """Reconstruct a :class:`BatchSubmissionResult` from a cached
    :class:`LlmCall`.

    The audit_store persists the full Anthropic response payload on
    every call (PRD §10), so the re-emit path has byte-exact fidelity
    without re-contacting Anthropic. The reconstructed result drops
    the ``__bba_response_headers__`` envelope key that the writer
    folded in — the cassette/raw schema doesn't carry it back out.
    """
    raw_response = {
        k: v for k, v in call.response_json.items() if k != "__bba_response_headers__"
    }
    headers = call.response_json.get("__bba_response_headers__", {})
    if not isinstance(headers, dict):
        headers = {}
    return BatchSubmissionResult(
        custom_id=call.audit_id,
        model_id=call.model_id,
        raw_response_json=raw_response,
        request_json=dict(call.request_json),
        response_headers=headers,
        request_timestamp=call.request_timestamp,
        latency_ms=call.latency_ms,
        anthropic_version=call.anthropic_version,
        prompt_cache_id=call.prompt_cache_id,
        extended_thinking_blocks=call.extended_thinking_blocks,
    )


def _now_utc() -> datetime:
    return datetime.now(UTC)


# hashlib + ClassifierResult held for forward-compat reconciler
# extensions; both are imported but not directly used in the current
# implementation (lint exemption acknowledged).
_ = (hashlib, ClassifierResult)


__all__ = ["resume_on_startup"]
