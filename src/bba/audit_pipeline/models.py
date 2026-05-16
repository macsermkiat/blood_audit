"""Frozen pydantic models for the audit_pipeline contract.

The pipeline is pure orchestration glue (issue #24, user constraint #1):
it composes deterministic_classifier, evidence_bundle_builder,
deid_redactor, prompt_builder, llm_client, quote_grounder,
confidence_calibrator, and audit_store. The persistable surface is
therefore narrow:

* :class:`BatchRunState` — the five-state machine
  (``pending -> submitted -> partial -> complete`` with ``failed``
  as a sink). Mirrors user constraint #4.
* :class:`BatchRun` — one row in the ``batch_runs`` table. Identifies
  one Anthropic Batch API submission plus the ``audit_id`` rows it
  carries. The state field is what survives a crash; the resume-on-
  startup reconciler scans on it.
* :class:`AuditPipelineConfig` — operator-supplied wiring (DB URL,
  ``code_version``, max batch size). The deeper module configurations
  (audit_store, llm_client, redactor) are passed alongside, not
  embedded, so a per-call refactor in any of them does not force a
  pipeline config bump.
* :class:`PipelineRunResult` — what :func:`run_pipeline` returns after
  one invocation completes (or after resume).
* :class:`ResumeReport` — what :func:`resume_on_startup` returns after
  polling Anthropic + reconciling against the audit_store.

The frozen / SafeId / UTCDatetime annotations follow
:mod:`bba.audit_store.models` exactly so the cross-module replay
invariant (same inputs → same bytes → same hash) holds across the
pipeline → store boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bba.audit_store.models import SafeId, UTCDatetime


class BatchRunState(StrEnum):
    """The five-state machine for a ``batch_runs`` row.

    Transitions (validated by :func:`bba.audit_pipeline.state_machine.transition`):

    * ``PENDING`` — batch envelope created locally; not yet submitted
      to Anthropic. The pipeline is permitted to crash here; resume
      will pick the row up and submit.
    * ``SUBMITTED`` — Anthropic returned a ``batch_id``. Resume polls
      this state on startup.
    * ``PARTIAL`` — some results have been returned + persisted to
      ``audit_store``; some are still pending at Anthropic. The poller
      stays in this state until every ``audit_id`` carries an
      ``audit_results`` row.
    * ``COMPLETE`` — every ``audit_id`` has been persisted. Terminal.
    * ``FAILED`` — unrecoverable error (e.g., Anthropic rejected the
      batch shape, or every ``audit_id`` exhausted retries). Terminal;
      requires operator action.

    Terminal states are sinks: a resume that encounters a ``COMPLETE``
    or ``FAILED`` row leaves it untouched.
    """

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    COMPLETE = "complete"
    FAILED = "failed"


class BatchRun(BaseModel):
    """One row in the ``batch_runs`` Postgres table.

    Identity is ``batch_id`` (locally generated, stable across re-runs).
    ``anthropic_batch_id`` is set only after the SUBMITTED transition
    so a crash between local-create and Anthropic-submit is recoverable
    (the row stays PENDING and resume retries).

    ``audit_ids`` is the full set of audit IDs the batch carries; the
    resume reconciler uses this to identify orphan ``llm_calls`` rows
    whose corresponding ``audit_results`` row was never written.

    Frozen so a concurrent reviewer dashboard cannot silently observe
    a half-mutated state mid-transition; transitions produce a NEW
    :class:`BatchRun` via :func:`bba.audit_pipeline.state_machine.transition`.
    """

    model_config = ConfigDict(frozen=True)

    batch_id: SafeId
    state: BatchRunState
    run_id: SafeId
    code_version: str = Field(min_length=1)
    audit_ids: tuple[SafeId, ...] = Field(min_length=1)
    anthropic_batch_id: str | None = None
    submitted_at: UTCDatetime | None = None
    updated_at: UTCDatetime
    error_message: str | None = None

    @model_validator(mode="after")
    def _state_consistency(self) -> Self:
        if self.state is BatchRunState.PENDING and self.anthropic_batch_id is not None:
            raise ValueError(
                "BatchRun in PENDING must not carry anthropic_batch_id "
                "(set only on SUBMITTED transition)"
            )
        if (
            self.state in {BatchRunState.SUBMITTED, BatchRunState.PARTIAL, BatchRunState.COMPLETE}
            and self.anthropic_batch_id is None
        ):
            raise ValueError(
                f"BatchRun in {self.state.value} requires anthropic_batch_id "
                "(set at SUBMITTED transition; preserved through terminal states)"
            )
        if self.state is BatchRunState.FAILED and self.error_message is None:
            raise ValueError(
                "BatchRun in FAILED requires error_message (operator must see "
                "the failure reason without re-deriving from logs)"
            )
        return self


class AuditPipelineConfig(BaseModel):
    """Operator-supplied configuration for the pipeline orchestrator.

    The pipeline does not embed sub-module configs (audit_store,
    llm_client, redactor) — those are passed alongside. This config
    is the surface that operators tune directly: DB connection, code
    version stamping, batch sizing.

    ``code_version`` participates in audit_store idempotency exactly
    as in :class:`bba.audit_store.AuditStoreConfig`: a bump invalidates
    the cached completion marker so the same ``(audit_id, run_id)`` is
    re-derived.
    """

    model_config = ConfigDict(frozen=True)

    db_url: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    max_batch_size: int = Field(default=100, ge=1, le=10_000)
    poll_interval_seconds: float = Field(default=30.0, gt=0.0)


class PipelineRunResult(BaseModel):
    """Outcome of one :func:`bba.audit_pipeline.run_pipeline` invocation.

    ``audit_ids_persisted`` is the set whose ``audit_results`` row hit
    disk during this run. ``batch_runs_touched`` is the set of
    ``batch_id``\\s whose state changed. ``orphan_audit_ids`` is the
    set whose ``llm_calls`` row exists but ``audit_results`` was not
    written within this run — typically empty unless the run was
    interrupted; the resume reconciler will pick them up on next boot.
    """

    model_config = ConfigDict(frozen=True)

    run_id: SafeId
    audit_ids_persisted: tuple[SafeId, ...]
    batch_runs_touched: tuple[SafeId, ...]
    orphan_audit_ids: tuple[SafeId, ...]


class ResumeReport(BaseModel):
    """Result of :func:`bba.audit_pipeline.resume_on_startup`.

    ``polled_batch_ids`` are the ``batch_runs`` rows whose Anthropic
    submission was polled (SUBMITTED + PARTIAL states). ``completed_audit_ids``
    are the rows whose ``audit_results`` were written during the resume
    pass. ``reemitted_audit_ids`` are the orphans whose cached LLM
    response was re-run through the verifier and persisted as a final
    classification.
    """

    model_config = ConfigDict(frozen=True)

    polled_batch_ids: tuple[SafeId, ...]
    completed_audit_ids: tuple[SafeId, ...]
    reemitted_audit_ids: tuple[SafeId, ...]
    failed_audit_ids: tuple[SafeId, ...]


__all__: Sequence[str] = (
    "AuditPipelineConfig",
    "BatchRun",
    "BatchRunState",
    "PipelineRunResult",
    "ResumeReport",
)
