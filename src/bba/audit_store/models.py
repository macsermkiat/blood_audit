"""Frozen pydantic models for the audit-store contract.

Two persisted records:

* :class:`AuditRow` — one row per audited RBC order. Fields follow PRD §Output
  schema (per audit row, persisted immutably) verbatim; bumping the schema
  means a new column in Parquet and a new ``run_id``-namespace upstream.
* :class:`LlmCall` — one row per Anthropic API call associated with an
  ``audit_id``. ``extended_thinking_blocks`` is the cold-storage candidate
  after 90 days (PRD §10).

All datetimes are tz-aware UTC at persistence time — enforced at write by the
store, declared here as ``datetime`` (Pydantic does not have a tz-aware-only
type). The ban on naive timestamps is structural and tested in ``bba.ingest``;
the audit_store re-applies it at the write boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, field_serializer


Classification = Literal[
    "APPROPRIATE",
    "INAPPROPRIATE",
    "NEEDS_REVIEW",
    "INSUFFICIENT_EVIDENCE",
]


def _ensure_utc(dt: datetime) -> datetime:
    """Reject naive datetimes; normalize aware non-UTC datetimes to UTC.

    The store-level invariant (PRD §"Tz-aware throughout", CONTEXT.md
    "tz-aware UTC") is asserted at the model boundary so a naive timestamp
    cannot leak past construction. Comparisons like
    ``request_timestamp < older_than`` in cold-storage migration assume both
    sides are aware; admitting naive values would later raise an opaque
    ``TypeError: can't compare offset-naive and offset-aware datetimes``.
    """
    if dt.tzinfo is None:
        raise ValueError(
            "datetime must be tz-aware; naive datetimes are forbidden in the "
            "audit_store (see CONTEXT.md 'tz-aware UTC')"
        )
    return dt.astimezone(UTC)


UTCDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]
"""A ``datetime`` constrained to tz-aware UTC at validation time.

Use on every persisted timestamp. Aware non-UTC inputs are converted to UTC.
"""


def _freeze_dict_items(
    items: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Return a tuple of read-only ``MappingProxyType`` views over copies of
    each input dict.

    Each input dict is shallow-copied (to detach from the caller's reference,
    so a later mutation on the caller's side cannot leak into the model) and
    then wrapped in :class:`types.MappingProxyType`, which raises ``TypeError``
    on ``__setitem__``. The audit-record immutability promise (``frozen=True``
    + tuple containers) extends to the nested JSON contents.
    """
    return tuple(MappingProxyType(dict(item)) for item in items)


FrozenJsonList = Annotated[
    tuple[dict[str, Any], ...],
    AfterValidator(_freeze_dict_items),
]
"""A ``tuple`` of immutable, read-only JSON-shaped mappings.

Use for ``indications_json``-style fields where the schema is dynamic but the
contents must not be mutated after model construction. Pair with the model's
``_serialize_frozen_json_list`` field serializer so JSON output is plain
``list[dict]`` regardless of in-memory representation.
"""


class AuditRow(BaseModel):
    """One audited RBC order. Persisted immutably to ``audit_results.parquet``.

    Field groups match PRD §Output schema:

    * Identity — stable across re-runs given the same inputs + code.
    * Anchor + inputs — what the order looked like at decision time.
    * Pipeline outputs — what the deterministic + LLM stack concluded.
    * Reproducibility metadata — every byte needed to re-derive the call.

    The nested-dict fields (``indications_json``, ``negative_evidence_json``,
    ``delta_hb_window_results``) use :data:`FrozenJsonList`, which wraps each
    dict in :class:`types.MappingProxyType` so post-construction mutation
    raises ``TypeError``. The model's frozen contract therefore extends
    through the JSON contents, not just the outer container.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    audit_id: str
    run_id: str
    run_timestamp: UTCDatetime
    hn_hash: str
    an_hash: str
    reqno: str

    # Anchor + inputs
    order_datetime: UTCDatetime
    products_ordered: tuple[str, ...]
    hb_value: float
    hb_datetime: UTCDatetime
    hb_freshness: str
    hb_source: str
    vitals_sbp: float | None
    vitals_hr: float | None
    vitals_timestamp: UTCDatetime | None
    vitals_source: str | None
    prior_rbc_units_24h: int
    prior_rbc_units_7d: int
    cohort_threshold: float
    delta_hb_window_results: FrozenJsonList

    # Pipeline outputs
    rule_classification: Classification
    final_classification: Classification
    cohort_applied: str
    indications_json: FrozenJsonList
    negative_evidence_json: FrozenJsonList
    confidence: float
    reasoning_summary_thai: str
    reasoning_summary_en: str
    needs_human_review: bool
    review_reason: str | None

    # Reproducibility metadata
    model_id: str
    prompt_hash: str
    evidence_bundle_hash: str
    redactor_version: str
    redactor_model_sha: str
    policy_version: str
    verifier_pass: bool
    verifier_retries: int
    escalated_to_opus: bool

    @field_serializer(
        "indications_json", "negative_evidence_json", "delta_hb_window_results"
    )
    def _serialize_frozen_json_list(
        self, value: tuple[Mapping[str, Any], ...]
    ) -> list[dict[str, Any]]:
        """Unwrap :class:`MappingProxyType` items back to plain ``dict`` for
        JSON output, so payloads serialize to ``[{...}, {...}]`` regardless
        of in-memory representation.
        """
        return [dict(item) for item in value]


class LlmCall(BaseModel):
    """One Anthropic API call. Multiple per ``audit_id`` is normal (retry
    + Sonnet→Opus escalation; PRD §13).

    ``extended_thinking_blocks`` is the bulky field. PRD §10 moves these to
    cold storage after 90 days; after migration the field is ``None`` here and
    the blob lives at ``cold_storage_uri``.
    """

    model_config = ConfigDict(frozen=True)

    call_id: str
    audit_id: str
    run_id: str
    model_id: str
    anthropic_version: str
    prompt_cache_id: str | None
    request_json: dict[str, Any]
    response_json: dict[str, Any]
    request_timestamp: UTCDatetime
    latency_ms: int
    extended_thinking_blocks: tuple[dict[str, Any], ...] | None
    cold_storage_uri: str | None


class AuditStoreConfig(BaseModel):
    """Filesystem-rooted store configuration.

    The root directory holds two Parquet datasets and a cold-storage subdir::

        <root>/audit_results/         <- AuditRow parquet partitions
        <root>/llm_calls/             <- LlmCall parquet partitions
        <root>/cold_storage/          <- migrated extended_thinking_blocks
        <root>/_snapshots/            <- daily DuckDB snapshot view files

    ``code_version`` participates in idempotency: a code-version bump invalidates
    the cached completion marker so a re-run is forced.
    """

    model_config = ConfigDict(frozen=True)

    root_dir: Path
    code_version: str


class WriteResult(BaseModel):
    """Outcome of a single :meth:`AuditStore.write` call.

    ``skipped_idempotent=True`` means a prior write with the same
    ``(audit_id, run_id)`` is already on disk and this call no-op'd. The store
    NEVER mutates an existing row — re-deriving an answer means a new ``run_id``.
    """

    model_config = ConfigDict(frozen=True)

    audit_id: str
    run_id: str
    llm_calls_written: int
    skipped_idempotent: bool


class ReconciliationReport(BaseModel):
    """Result of :meth:`AuditStore.reconcile` over a run.

    An orphan ``llm_calls`` row (no matching ``audit_results``) is the expected
    failure mode for a crash *after* phase 1 but *before* phase 2. The reconciler
    catalogues these so the operator can re-emit (re-derive the classification
    from the cached response) or quarantine.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    orphan_call_ids: tuple[str, ...]
    orphan_audit_ids: tuple[str, ...]


class ColdStorageReport(BaseModel):
    """Result of :func:`migrate_cold_storage`.

    ``moved_call_ids`` are the ``LlmCall.call_id``\\s whose
    ``extended_thinking_blocks`` were spilled to ``cold_storage_uri``. The
    in-line field is replaced with ``None`` post-migration; the URI points to
    the file in ``<root>/cold_storage/`` (S3 in production, local in tests).
    """

    model_config = ConfigDict(frozen=True)

    moved_call_ids: tuple[str, ...]
    bytes_moved: int


__all__: Sequence[str] = (
    "AuditRow",
    "AuditStoreConfig",
    "Classification",
    "ColdStorageReport",
    "LlmCall",
    "ReconciliationReport",
    "WriteResult",
)
