"""bba.audit_pipeline — per-audit-row orchestration + batch_runs checkpoint.

See issue #24 for acceptance criteria. PRD §15 (Implementation Decisions)
defines the contract:

* Pure orchestration glue. Composes (never re-implements):
  :mod:`bba.ingest`, :mod:`bba.audit_orders`, :mod:`bba.hb_lookup`,
  :mod:`bba.vitals_extractor`, :mod:`bba.cohort_detector`,
  :mod:`bba.deterministic_classifier`, :mod:`bba.evidence_bundle_builder`,
  :mod:`bba.deid_redactor`, :mod:`bba.prompt_builder`,
  :mod:`bba.llm_client`, :mod:`bba.quote_grounder`,
  :mod:`bba.confidence_calibrator`, :mod:`bba.audit_store`.

* Row-level checkpointing via the ``batch_runs`` Postgres table
  (chosen to match :mod:`bba.review_actions` — user constraint #4).
  Five-state machine: ``pending -> submitted -> partial -> complete``
  with ``failed`` as a sink (user constraint #4).

* Resume-on-startup is a HARD requirement (user constraint #5):
  the pipeline must survive a SIGTERM mid-batch and restart without
  losing or duplicating work.

* Winning-attempt rule (user constraint #6): last verifier-passed
  attempt wins; if no attempt passed verifier, classification =
  ``NEEDS_REVIEW`` with ``hallucination_suspect`` flag. The rule lives
  in :mod:`bba.audit_store` already; this module just orchestrates.

* Cost guard (user constraint #10): the live Anthropic API is never
  called from unit / integration tests. The
  :func:`bba.audit_pipeline.cost_guard.assert_test_safe_transport`
  helper detects and rejects the live transport.
"""

from bba.audit_pipeline.cost_guard import assert_test_safe_transport
from bba.audit_pipeline.exceptions import (
    AuditPipelineError,
    BatchStateTransitionError,
    LiveAnthropicApiError,
    ResumeReconciliationError,
)
from bba.audit_pipeline.models import (
    AuditPipelineConfig,
    BatchRun,
    BatchRunState,
    PipelineRunResult,
    ResumeReport,
)
from bba.audit_pipeline.pipeline import process_audit_order, run_pipeline
from bba.audit_pipeline.replay import apply_batch_results, select_winning_attempt
from bba.audit_pipeline.resume import resume_on_startup
from bba.audit_pipeline.state_machine import (
    VALID_TRANSITIONS,
    is_terminal,
    transition,
)
from bba.audit_pipeline.store import BatchRunStore, InMemoryBatchRunStore


__all__ = [
    "VALID_TRANSITIONS",
    "AuditPipelineConfig",
    "AuditPipelineError",
    "BatchRun",
    "BatchRunState",
    "BatchRunStore",
    "BatchStateTransitionError",
    "InMemoryBatchRunStore",
    "LiveAnthropicApiError",
    "PipelineRunResult",
    "ResumeReconciliationError",
    "ResumeReport",
    "apply_batch_results",
    "assert_test_safe_transport",
    "is_terminal",
    "process_audit_order",
    "resume_on_startup",
    "run_pipeline",
    "select_winning_attempt",
    "transition",
]
