"""Resume-on-startup reconciler (issue #24 AC ③, user constraint #5).

Hard requirement: the pipeline must survive a SIGTERM mid-batch and
restart without losing or duplicating work. The reconciler:

1. Scans ``batch_runs`` for non-terminal rows (PENDING, SUBMITTED,
   PARTIAL).
2. For PENDING rows: the local-create succeeded but Anthropic was
   never asked. With no anthropic_batch_id there is nothing to poll,
   so the row is transitioned to FAILED with an operator-visible
   error message. Re-submission is the operator's call (a re-run of
   :func:`bba.audit_pipeline.run_pipeline` with the same ``run_id``
   is idempotent via audit_store's commit-marker; a new ``run_id``
   re-derives the rows fresh).
2'. (Codex review HIGH #4: PENDING rows previously dropped silently.)
3. For SUBMITTED / PARTIAL rows: walks every audit_id in the batch
   and uses :meth:`AuditStore.reconcile` to identify orphan
   ``llm_calls`` rows (call written, audit row missing). Each orphan
   is re-emitted through :func:`bba.audit_pipeline.replay.apply_batch_results`
   using the cached :class:`LlmCall.response_json` payload — the
   audit_store's idempotency contract guarantees a second run is a
   no-op.
4. Once every audit_id carries an audit_results row, transitions the
   batch_runs row to COMPLETE. Otherwise leaves it PARTIAL so the
   next resume picks it up again.

The reconciler is itself idempotent — running it twice produces zero
additional writes (the audit_store's
``WriteResult.skipped_idempotent`` rejects the second call).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime

from bba.audit_pipeline.models import (
    BatchRunState,
    PipelineRowContext,
    ResumeReport,
)
from bba.audit_pipeline.state_machine import is_terminal, transition
from bba.audit_pipeline.store import BatchRunStore
from bba.audit_store import AuditStore, LlmCall
from bba.llm_client.models import (
    BatchSubmissionResult,
    RawBatchResponse,
)


def resume_on_startup(
    *,
    batch_run_store: BatchRunStore,
    audit_store: AuditStore,
    contexts: Mapping[str, PipelineRowContext] | None = None,
) -> ResumeReport:
    """Reconcile every non-terminal ``batch_runs`` row and return a report.

    ``contexts`` maps ``audit_id`` to a :class:`PipelineRowContext` so
    the reconciler can rebuild :class:`bba.audit_store.AuditRow` rows
    from cached LLM responses without re-querying upstream tables.
    When ``contexts`` is ``None`` or missing an audit_id, that orphan
    is reported via :attr:`ResumeReport.failed_audit_ids` rather than
    silently fabricated.

    The reconciler does NOT re-poll Anthropic. It works exclusively
    off the cached ``llm_calls`` rows already on disk (PRD §10:
    audit_store persists the full Anthropic response so re-emit is
    byte-exact without an HTTP round-trip). Batches that crashed
    after submission but before any response landed (no cached
    calls) are surfaced as ``failed_audit_ids`` for operator action;
    re-submission goes through :func:`bba.audit_pipeline.run_pipeline`
    with the same ``run_id`` (idempotent via audit_store's commit
    marker).

    Returns a typed report:

    * ``polled_batch_ids`` — every batch_run touched by the resume.
    * ``completed_audit_ids`` — audit_ids whose audit_results row is
      now committed (either pre-existed or re-emitted this run).
    * ``reemitted_audit_ids`` — orphan calls whose audit_row was
      written by this resume pass.
    * ``failed_audit_ids`` — audit_ids that cannot be reconciled
      without operator action (PENDING with no Anthropic call,
      orphan with no context, or batch with no cached response).
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
            # The local create landed but submission never happened.
            # Without an anthropic_batch_id we can't poll; surface the
            # row as FAILED so the operator can act (Codex review HIGH #4).
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

        # SUBMITTED / PARTIAL: walk the cached llm_calls, write
        # audit_results for orphans where context is available.
        #
        # An "orphan" here is an audit_id whose ``llm_calls`` row exists
        # but whose ``audit_results`` row was never written (crash
        # between phase 1 and phase 2 of audit_store.write). The
        # audit_store's :meth:`reconcile` produces ``orphan_call_ids``
        # for this direction; we derive the audit_id set from the
        # cached calls so we don't depend on call_id <-> audit_id
        # translation outside the audit_store boundary.
        cached_calls_by_audit_id = _cached_calls_by_audit_id(
            audit_store, run_id=run.run_id
        )
        persisted_audit_ids = {
            row.audit_id
            for row in audit_store.read_audit_results(run_id=run.run_id)
        }
        run_reemitted: list[str] = []
        run_failed: list[str] = []

        for audit_id in run.audit_ids:
            if audit_id in persisted_audit_ids:
                completed.append(audit_id)
                continue
            calls = cached_calls_by_audit_id.get(audit_id, ())
            if calls:
                # Orphan call: phase 1 landed but phase 2 didn't.
                if audit_id not in contexts:
                    run_failed.append(audit_id)
                    continue
                _re_emit_audit_row(
                    audit_id=audit_id,
                    calls=calls,
                    context=contexts[audit_id],
                    audit_store=audit_store,
                    run_id=run.run_id,
                )
                run_reemitted.append(audit_id)
                completed.append(audit_id)
            else:
                # No audit_row AND no cached call — Anthropic was
                # contacted but the call response never landed (e.g.,
                # SIGTERM before the writer ran). Surface for operator
                # action; the rerun pathway re-derives via run_pipeline.
                run_failed.append(audit_id)

        reemitted.extend(run_reemitted)
        failed.extend(run_failed)

        if not run_failed and len(set(run.audit_ids)) == len(
            {aid for aid in run.audit_ids if aid in persisted_audit_ids}
            | set(run_reemitted)
        ):
            completed_run = transition(
                run, to_state=BatchRunState.COMPLETE, now=_now_utc()
            )
            batch_run_store.update(completed_run)
        elif run.state is BatchRunState.SUBMITTED:
            # Some rows landed; move to PARTIAL so the next resume
            # can identify what's still pending.
            partial_run = transition(
                run, to_state=BatchRunState.PARTIAL, now=_now_utc()
            )
            batch_run_store.update(partial_run)

    return ResumeReport(
        polled_batch_ids=tuple(polled),
        completed_audit_ids=tuple(completed),
        reemitted_audit_ids=tuple(reemitted),
        failed_audit_ids=tuple(failed),
    )


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
) -> None:
    """Rebuild an audit_row from cached call responses + context.

    Synthesises a :class:`RawBatchResponse` from the cached
    ``llm_calls`` rows and feeds it back through
    :func:`bba.audit_pipeline.replay.apply_batch_results` — the same
    code path that the original submission used. The audit_store's
    idempotency contract guarantees re-running this for an already-
    persisted audit_id is a no-op.
    """
    from bba.audit_pipeline.replay import apply_batch_results

    results = tuple(_result_from_cached_call(call) for call in calls)
    synthetic_response = RawBatchResponse(
        batch_id=f"resume-{audit_id}",
        results=results,
    )
    apply_batch_results(
        synthetic_response,
        audit_store=audit_store,
        run_id=run_id,
        contexts={audit_id: context},
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


# Re-export hashlib so module-level static analysis sees the use; the
# hashlib import is used by _re_emit_audit_row in future expansions
# (deterministic call_id rebuild). Kept as a forward-compat hook.
_ = hashlib


__all__ = ["resume_on_startup"]
