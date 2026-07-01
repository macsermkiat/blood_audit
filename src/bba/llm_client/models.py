"""Pydantic v2 models, enums, and type aliases for :mod:`bba.llm_client`.

All public models are frozen. The module surface mirrors the convention
established by :mod:`bba.prompt_builder` and :mod:`bba.audit_store`:

* Allow-set-pinned model IDs in :data:`ALLOWED_MODELS` — the client may
  only invoke the two IDs in that set; swapping models requires a code
  change (PRD §13). The IDs are bare aliases (Claude Sonnet 5 / Opus 4.8
  ship without dated snapshots), so this guards model *swaps*, not
  Anthropic point releases under the same alias.
* Frozen Pydantic models for every persistable record so the audit
  chain's replay invariant (same inputs → same bytes → same hash) holds
  across the boundary into :mod:`bba.audit_store`.
* The Anthropic HTTP boundary is a :class:`typing.Protocol`
  (:class:`AnthropicTransport`) so tests inject recorded cassettes
  (PRD §"contract tests against the Anthropic SDK using
  betamax-style cassettes for offline replay").
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import (
    Annotated,
    Final,
    Literal,
    Protocol,
    Self,
    runtime_checkable,
)

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from bba.audit_store.models import (
    FrozenJsonDict,
    FrozenJsonList,
    LlmCall,
    SafeId,
    UTCDatetime,
)
from bba.prompt_builder.models import PromptBuildResult, TaskMode


# =============================================================================
# Snapshot-pinned model IDs (PRD §13)
# =============================================================================


SONNET_MODEL_ID: Final[str] = "claude-sonnet-5"
"""Primary model ID — Claude Sonnet 5.

Claude Sonnet 5 ships as a bare alias with no dated snapshot; the alias
is the canonical ID (per Anthropic's Sonnet 5 model docs). The pin is
the allow-set below, not a date suffix — bumping the model still forces
an explicit edit to this constant + a golden-set re-run, but Anthropic
point releases under the same alias are no longer detected here."""


OPUS_MODEL_ID: Final[str] = "claude-opus-4-8"
"""Escalation model ID — Claude Opus 4.8, a bare alias like
:data:`SONNET_MODEL_ID`. Opus is the escalation target after Sonnet
retries are exhausted (PRD §13)."""


ALLOWED_MODELS: Final[frozenset[str]] = frozenset({SONNET_MODEL_ID, OPUS_MODEL_ID})
"""The two model IDs the client is permitted to invoke.

A free-form model_id would let a refactor silently swap the model out
from under the audit chain. The validator on :class:`LlmClientConfig`
enforces membership in this set — that allow-list is the pin.
"""


MAX_SONNET_ATTEMPTS: Final[int] = 2
"""Sonnet retry budget before escalation (PRD §13: "retry → Sonnet
(≤ 2x) → escalate to Opus 4.7")."""


ANTHROPIC_BETA_HEADER: Final[str] = "message-batches-2024-09-24"
"""Anthropic Batch API beta header. Pinned so a contract drift in the
beta channel breaks loudly at startup rather than silently in
production."""


Model = Literal[
    "claude-sonnet-5",
    "claude-opus-4-8",
]
"""Type-level alias for the two allowed model IDs."""


def _validate_model_id(value: str) -> str:
    """Reject any model ID outside :data:`ALLOWED_MODELS`."""
    if value not in ALLOWED_MODELS:
        raise ValueError(
            f"model_id {value!r} not in {sorted(ALLOWED_MODELS)} "
            "(PRD §13: only the pinned allow-set is permitted)"
        )
    return value


PinnedModel = Annotated[str, AfterValidator(_validate_model_id)]


# =============================================================================
# Classification + parse-failure vocabulary
# =============================================================================


Classification = Literal[
    "APPROPRIATE",
    "INAPPROPRIATE",
    "NEEDS_REVIEW",
    "INSUFFICIENT_EVIDENCE",
]
"""The four audit labels. Mirrors :data:`bba.audit_store.Classification`
so persistence has zero translation cost."""


_ALLOWED_CLASSIFICATIONS: Final[frozenset[str]] = frozenset(
    {"APPROPRIATE", "INAPPROPRIATE", "NEEDS_REVIEW", "INSUFFICIENT_EVIDENCE"}
)


class ParseFailureReason(StrEnum):
    """Mutually-exclusive parse-failure category.

    Surfaced on :class:`ParseOutcome.parse_failure_reason` so reviewers
    can monitor whether the LLM is drifting from the structured-output
    shape (``malformed_json``), schema (``schema_mismatch``), or
    classification vocabulary (``classification_out_of_set``).
    """

    MALFORMED_JSON = "malformed_json"
    SCHEMA_MISMATCH = "schema_mismatch"
    CLASSIFICATION_OUT_OF_SET = "classification_out_of_set"
    EMPTY_RESPONSE = "empty_response"
    TOOL_USE_MISSING = "tool_use_missing"


# =============================================================================
# Structured-output (tool-use) input + LLM response models
# =============================================================================


class IndicationCitation(BaseModel):
    """One indication + verbatim quote returned by the LLM.

    ``code`` is a stable taxonomy slot (e.g. ``"B1.active_bleeding"``)
    so reviewer aggregation can group by indication family. ``quote`` is
    the verbatim substring from the evidence; the post-LLM quote
    grounder (:mod:`bba.quote_grounder`) verifies it against the
    redacted bundle. ``source_id`` is the evidence chunk's stable ID
    (``E1``, ``E2``, ...). ``confidence`` is the LLM's self-reported
    confidence in [0, 1].
    """

    model_config = ConfigDict(frozen=True)

    code: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    source_id: str = Field(min_length=1, pattern=r"^E\d+$")
    confidence: float = Field(ge=0.0, le=1.0)


class LlmClassificationResponse(BaseModel):
    """Parsed structured-output response from one Anthropic call.

    The structured-output (tool-use) JSON shape is fixed (PRD §13:
    "structured-output (tool-use) JSON shape with fail-closed parsing").
    Drift detection happens in :func:`parse_structured_response`: any
    deviation from this shape → :class:`ParseFailureReason.SCHEMA_MISMATCH`
    and the row routes to ``NEEDS_REVIEW``.
    """

    model_config = ConfigDict(frozen=True)

    classification: Classification
    indications: tuple[IndicationCitation, ...]
    negative_evidence: tuple[str, ...]
    reasoning_summary_en: str
    reasoning_summary_th: str


class StructuredToolInput(BaseModel):
    """The Anthropic tool-use ``input_schema`` mirror.

    Exposed as a model so the client builds the tool block once at
    config time, never per-row. Mutability would let a downstream caller
    nudge the schema between submissions inside one batch — frozen
    blocks that path entirely.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(min_length=1)
    tool_description: str = Field(min_length=1)


class ParseOutcome(BaseModel):
    """Result of :func:`parse_structured_response`.

    Fail-closed contract: ``parsed is None`` iff ``parse_failure is True``
    iff ``parse_failure_reason is not None``. A caller that sees
    ``parse_failure=True`` MUST route the row to ``NEEDS_REVIEW`` with
    the ``parse_failure`` review reason (PRD §13).
    """

    model_config = ConfigDict(frozen=True)

    parsed: LlmClassificationResponse | None
    parse_failure: bool
    parse_failure_reason: ParseFailureReason | None
    raw_text: str = Field(default="")

    @model_validator(mode="after")
    def _failure_fields_consistent(self) -> Self:
        if self.parse_failure:
            if self.parsed is not None:
                raise ValueError(
                    "ParseOutcome.parsed must be None when "
                    "parse_failure=True (fail-closed contract)"
                )
            if self.parse_failure_reason is None:
                raise ValueError(
                    "ParseOutcome.parse_failure_reason is required when "
                    "parse_failure=True"
                )
        else:
            if self.parsed is None:
                raise ValueError(
                    "ParseOutcome.parsed must be set when parse_failure=False"
                )
            if self.parse_failure_reason is not None:
                raise ValueError(
                    "ParseOutcome.parse_failure_reason must be None when "
                    "parse_failure=False"
                )
        return self


# =============================================================================
# Batch submission / response shapes
# =============================================================================


class BatchSubmissionRequest(BaseModel):
    """One row in a Batch API submission.

    ``audit_id`` doubles as the Anthropic Batch API ``custom_id`` — the
    PRD §13 invariant ("``custom_id == audit_id`` assertion on every
    result, never positional zip") collapses two identifiers into one.
    ``prompt`` carries the assembled prompt blocks + cache markers from
    :mod:`bba.prompt_builder`.
    """

    model_config = ConfigDict(frozen=True)

    audit_id: SafeId
    run_id: SafeId
    task_mode: TaskMode
    prompt: PromptBuildResult


class BatchSubmissionResult(BaseModel):
    """One row in the parsed Batch API response.

    ``custom_id`` MUST equal the matching :class:`BatchSubmissionRequest.audit_id`
    — :func:`bba.llm_client.custom_id.assert_custom_ids_match` enforces
    this. ``raw_response_json`` is the full Anthropic response payload
    persisted to ``llm_calls`` for reproducibility (PRD §"Persist the
    full Anthropic Batch API request and response per audit_id").

    The nested JSON payloads are deeply frozen via
    :data:`bba.audit_store.models.FrozenJsonDict` /
    :data:`bba.audit_store.models.FrozenJsonList`: a downstream caller
    cannot patch ``raw_response_json["content"]`` after construction.
    Mutability here would let a refactor silently rewrite the audit
    chain between parse and persist.
    """

    model_config = ConfigDict(frozen=True)

    custom_id: SafeId
    model_id: PinnedModel
    raw_response_json: FrozenJsonDict
    request_json: FrozenJsonDict
    response_headers: FrozenJsonDict
    request_timestamp: UTCDatetime
    latency_ms: int = Field(ge=0)
    anthropic_version: str = Field(min_length=1)
    prompt_cache_id: str | None = None
    extended_thinking_blocks: FrozenJsonList | None = None


class RawBatchResponse(BaseModel):
    """The raw shape returned by :meth:`AnthropicTransport.submit_batch`.

    Wrapping the bare-dict response in a model gives the parser a fixed
    surface to validate against, and lets cassettes lock in a fixture
    shape independent of Anthropic SDK version drift.
    """

    model_config = ConfigDict(frozen=True)

    batch_id: str = Field(min_length=1)
    results: tuple[BatchSubmissionResult, ...]


# =============================================================================
# Escalation + disagreement + final result
# =============================================================================


class EscalationLog(BaseModel):
    """Per-audit-row record of the retry → escalation path.

    ``sonnet_attempts`` counts every Sonnet call (1 or 2 per
    :data:`MAX_SONNET_ATTEMPTS`). ``escalated_to_opus`` is True iff
    every Sonnet attempt failed and Opus was invoked.
    ``parse_failure_reasons`` records the per-attempt parse failure
    so the reviewer can see whether Sonnet keeps making the same
    structural mistake.
    """

    model_config = ConfigDict(frozen=True)

    audit_id: SafeId
    sonnet_attempts: int = Field(ge=0, le=MAX_SONNET_ATTEMPTS)
    sonnet_parse_failures: tuple[ParseFailureReason, ...]
    escalated_to_opus: bool
    opus_parse_failure: ParseFailureReason | None = None

    @model_validator(mode="after")
    def _escalation_consistent(self) -> Self:
        # Opus is only invoked when every Sonnet attempt failed.
        if self.escalated_to_opus and self.sonnet_attempts < MAX_SONNET_ATTEMPTS:
            raise ValueError(
                f"escalated_to_opus=True but sonnet_attempts="
                f"{self.sonnet_attempts} < MAX_SONNET_ATTEMPTS="
                f"{MAX_SONNET_ATTEMPTS}; escalation must exhaust the "
                "Sonnet budget first (PRD §13)"
            )
        if (
            self.escalated_to_opus
            and len(self.sonnet_parse_failures) != self.sonnet_attempts
        ):
            raise ValueError(
                "sonnet_parse_failures length must equal sonnet_attempts "
                "when escalated_to_opus=True (one failure per attempt)"
            )
        if not self.escalated_to_opus and self.opus_parse_failure is not None:
            raise ValueError(
                "opus_parse_failure must be None when escalated_to_opus=False"
            )
        return self


class DisagreementVerdict(BaseModel):
    """Sonnet/Opus classification-disagreement record.

    When both Sonnet and Opus return parseable responses with different
    :data:`Classification` values, the row routes to ``NEEDS_REVIEW``
    with the ``disagreement`` reason (PRD §13). ``agreed=True`` means
    both produced the same classification; the Opus answer is then
    accepted as the final answer (Opus has the deeper reasoning budget).
    """

    model_config = ConfigDict(frozen=True)

    sonnet_classification: Classification | None
    opus_classification: Classification | None
    agreed: bool
    routed_to_needs_review: bool

    @model_validator(mode="after")
    def _routing_matches_state(self) -> Self:
        # Only meaningful when both classifications exist (otherwise
        # there is no disagreement to detect — escalation handled the
        # parse failure separately).
        if (
            self.sonnet_classification is not None
            and self.opus_classification is not None
        ):
            should_agree = self.sonnet_classification == self.opus_classification
            if should_agree != self.agreed:
                raise ValueError(
                    f"agreed ({self.agreed}) must equal "
                    f"(sonnet == opus) ({should_agree})"
                )
            if self.routed_to_needs_review == self.agreed:
                raise ValueError(
                    "routed_to_needs_review must be True iff agreed is False "
                    "(disagreement routes to NEEDS_REVIEW)"
                )
        return self


class LlmClientResult(BaseModel):
    """Top-level per-audit-row outcome from :func:`process_batch`.

    The :attr:`persisted_calls` tuple is what the caller hands to
    :meth:`bba.audit_store.AuditStore.write` — one :class:`LlmCall`
    per Anthropic invocation (Sonnet attempts + optional Opus). The
    transactional-ordering contract holds because all calls share the
    same ``(audit_id, run_id)`` pair as the audit row they back.
    """

    model_config = ConfigDict(frozen=True)

    audit_id: SafeId
    run_id: SafeId
    final_classification: Classification
    response: LlmClassificationResponse | None
    parse_failure: bool
    needs_review: bool
    review_reason: str | None
    escalation: EscalationLog
    disagreement: DisagreementVerdict | None
    persisted_calls: tuple[LlmCall, ...]

    @model_validator(mode="after")
    def _review_state_consistent(self) -> Self:
        if self.parse_failure and self.response is not None:
            raise ValueError(
                "response must be None when parse_failure=True (fail-closed)"
            )
        if self.parse_failure and self.final_classification != "NEEDS_REVIEW":
            raise ValueError(
                "parse_failure=True forces final_classification=NEEDS_REVIEW "
                "(PRD §13 fail-closed)"
            )
        if self.parse_failure and not self.needs_review:
            raise ValueError("parse_failure=True forces needs_review=True")
        if self.needs_review and self.review_reason is None:
            raise ValueError("needs_review=True requires a review_reason")
        # Every persisted call must share the result's (audit_id, run_id).
        # The audit_store contract rejects mismatches at write time; we
        # reject them at the model boundary so the bug surfaces earlier.
        for call in self.persisted_calls:
            if call.audit_id != self.audit_id:
                raise ValueError(
                    f"persisted_calls[*].audit_id ({call.audit_id!r}) must "
                    f"match LlmClientResult.audit_id ({self.audit_id!r})"
                )
            if call.run_id != self.run_id:
                raise ValueError(
                    f"persisted_calls[*].run_id ({call.run_id!r}) must "
                    f"match LlmClientResult.run_id ({self.run_id!r})"
                )
        return self


# =============================================================================
# Configuration
# =============================================================================


class LlmClientConfig(BaseModel):
    """Operator-supplied configuration for :func:`process_batch`.

    ``max_sonnet_attempts`` is bounded by :data:`MAX_SONNET_ATTEMPTS`
    (the PRD §13 cap). ``prompt_cache_enabled`` defaults to ``True`` —
    disabling is a cost regression and exists only for the integration
    test that verifies cache-marker translation.
    """

    model_config = ConfigDict(frozen=True)

    sonnet_model_id: PinnedModel = SONNET_MODEL_ID
    opus_model_id: PinnedModel = OPUS_MODEL_ID
    max_sonnet_attempts: int = Field(
        default=MAX_SONNET_ATTEMPTS, ge=2, le=MAX_SONNET_ATTEMPTS
    )
    prompt_cache_enabled: bool = True
    cross_check_with_opus: bool = False
    code_version: str = Field(min_length=1)

    @field_validator("sonnet_model_id")
    @classmethod
    def _sonnet_is_sonnet(cls, v: str) -> str:
        if "sonnet" not in v:
            raise ValueError(f"sonnet_model_id must contain 'sonnet' (got {v!r})")
        return v

    @field_validator("opus_model_id")
    @classmethod
    def _opus_is_opus(cls, v: str) -> str:
        if "opus" not in v:
            raise ValueError(f"opus_model_id must contain 'opus' (got {v!r})")
        return v


# =============================================================================
# Transport boundary (Anthropic HTTP -> Protocol)
# =============================================================================


@runtime_checkable
class AnthropicTransport(Protocol):
    """The HTTP boundary, exposed as a Protocol for test injection.

    The production implementation wraps the official ``anthropic`` SDK;
    the test implementation (:class:`bba.llm_client.cassette.CassetteTransport`)
    replays a recorded JSON cassette so unit tests run offline.

    The Protocol exposes three operations:

    * :meth:`submit_batch_only` — create the remote batch and return
      the ``batch_id`` immediately (no polling). Persisting the
      ``batch_id`` BEFORE waiting for results is what makes the
      checkpoint table recoverable across SIGTERM during the polling
      window (PRD §15 row-level checkpointing).
    * :meth:`fetch_batch_results` — given a previously-submitted
      ``batch_id``, poll until completion and return the parsed
      results. Idempotent: callers MAY invoke this multiple times for
      the same ``batch_id``; the audit_store's own commit-marker
      handles double-application.
    * :meth:`submit_batch` — convenience wrapper that calls both
      operations in sequence. Preserved for backward compatibility
      with callers that do not need split-phase checkpointing.
    """

    def submit_batch_only(
        self,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> str:
        """Create the remote batch and return its ``batch_id`` immediately.

        Must NOT poll. The caller is expected to persist ``batch_id``
        before invoking :meth:`fetch_batch_results` so a crash during
        polling leaves a recoverable checkpoint row."""
        ...

    def fetch_batch_results(
        self,
        batch_id: str,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> RawBatchResponse:
        """Poll ``batch_id`` until completion and return the parsed results.

        ``requests`` carries the original submission set so the result
        envelope can be reconstructed (each result's ``request_json``
        echoes the originating ``BatchSubmissionRequest``). Implementations
        MAY raise :class:`AnthropicAPIError` on timeout or non-recoverable
        error; the caller (resume reconciler) catches and surfaces the
        row as failed for operator action."""
        ...

    def submit_batch(
        self,
        *,
        model: str,
        requests: Sequence[BatchSubmissionRequest],
        prompt_cache_enabled: bool,
    ) -> RawBatchResponse:
        """Convenience: :meth:`submit_batch_only` then :meth:`fetch_batch_results`.

        Preserved for callers that don't need split-phase checkpointing;
        the audit_pipeline orchestrator uses the split methods directly.
        Implementations MUST set ``custom_id`` on each result to the
        submission's ``audit_id`` and must surface the response headers
        (specifically ``anthropic-version`` and ``prompt_cache_id``).
        """
        ...


class CassetteInteraction(BaseModel):
    """One recorded request/response pair inside a cassette.

    The cassette is keyed on ``(model, sorted_custom_ids_tuple)`` so a
    replay matches regardless of submission ordering — the Batch API
    itself does not promise ordering. The interaction's response is
    the full :class:`RawBatchResponse` payload.
    """

    model_config = ConfigDict(frozen=True)

    model: PinnedModel
    custom_ids: tuple[str, ...]
    response: RawBatchResponse


__all__: Sequence[str] = (
    "ALLOWED_MODELS",
    "ANTHROPIC_BETA_HEADER",
    "AnthropicTransport",
    "BatchSubmissionRequest",
    "BatchSubmissionResult",
    "CassetteInteraction",
    "Classification",
    "DisagreementVerdict",
    "EscalationLog",
    "IndicationCitation",
    "LlmClassificationResponse",
    "LlmClientConfig",
    "LlmClientResult",
    "MAX_SONNET_ATTEMPTS",
    "Model",
    "OPUS_MODEL_ID",
    "ParseFailureReason",
    "ParseOutcome",
    "PinnedModel",
    "RawBatchResponse",
    "SONNET_MODEL_ID",
    "StructuredToolInput",
)
