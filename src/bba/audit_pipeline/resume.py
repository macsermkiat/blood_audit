"""Resume-on-startup reconciler (issue #24 AC ③, user constraint #5).

Hard requirement: the pipeline must survive a SIGTERM mid-batch and
restart without losing or duplicating work. The reconciler is the
single load-bearing primitive that delivers this:

1. Scan ``batch_runs`` for rows in ``SUBMITTED`` or ``PARTIAL`` state.
2. Poll Anthropic via :class:`bba.llm_client.AnthropicTransport` for
   each row's ``anthropic_batch_id``.
3. Reconcile against :mod:`bba.audit_store`: for any ``audit_id`` whose
   ``llm_calls`` row exists but ``audit_results`` row does NOT, re-emit
   through the verifier + winning-attempt rule and persist.
4. Transition the ``batch_runs`` row to ``COMPLETE`` once every
   ``audit_id`` carries an ``audit_results`` row.

The reconciler is idempotent on its own — running it twice in a row
must produce zero additional writes (the audit_store's
``WriteResult.skipped_idempotent`` rejects the second call).
"""

from __future__ import annotations

from bba.audit_pipeline.models import ResumeReport
from bba.audit_pipeline.store import BatchRunStore
from bba.audit_store import AuditStore
from bba.llm_client.models import AnthropicTransport, LlmClientConfig


def resume_on_startup(
    *,
    batch_run_store: BatchRunStore,
    audit_store: AuditStore,
    transport: AnthropicTransport,
    llm_config: LlmClientConfig,
) -> ResumeReport:
    """Reconcile every non-terminal ``batch_runs`` row and return a report.

    Walks every row in :data:`BatchRunState.SUBMITTED` and
    :data:`BatchRunState.PARTIAL`. For each:

    * Poll Anthropic via ``transport`` for the batch's status.
    * For every ``audit_id`` returned, check
      :meth:`AuditStore.reconcile`: any orphan call (call recorded,
      audit row missing) is re-emitted through the verifier +
      winning-attempt rule.
    * Once every ``audit_id`` in the batch carries an ``audit_results``
      row, transition the ``batch_runs`` row to ``COMPLETE`` via
      :func:`bba.audit_pipeline.state_machine.transition`.

    Idempotency: a second call against the same Postgres state must
    produce zero new writes. The audit_store's own idempotency
    contract (write-skipped-idempotent on duplicate ``(audit_id,
    run_id, code_version)``) is the load-bearing primitive — this
    function never bypasses it.

    The implementation lives in GREEN (issue #24).
    """
    _ = (batch_run_store, audit_store, transport, llm_config)
    raise NotImplementedError("RED-phase scaffold; see issue #24")


__all__ = ["resume_on_startup"]
