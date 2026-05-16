"""Idempotent application of an Anthropic batch result set (issue #24).

User constraint #7 — replay idempotency property test:

    Apply the same Anthropic batch result set twice to bba.audit_store
    via the pipeline. Second application must be a no-op (zero new
    rows, zero updates).

User constraint #6 — winning-attempt rule (applied here):

    Multiple llm_calls per audit_id is normal (retry, escalation).
    Winning attempt = the latest verifier_pass=True. If none passes,
    classification = NEEDS_REVIEW with hallucination_suspect flag.

The audit_store's own idempotency contract
(``WriteResult.skipped_idempotent``) is the load-bearing primitive;
this layer composes it with the verifier + winning-attempt rule and
returns a per-call summary so the test can assert "zero new rows" on
the second pass.

No silent fabrication: every persisted row's clinical + reproducibility
fields come from a caller-supplied :class:`PipelineRowContext`.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

from bba.audit_pipeline.models import PipelineRowContext, PipelineRunResult
from bba.audit_store import AuditRow, AuditStore, LlmCall
from bba.audit_store.models import Classification
from bba.deterministic_classifier import ClassifierResult
from bba.llm_client.models import BatchSubmissionResult, RawBatchResponse


Verifier = Callable[[BatchSubmissionResult, PipelineRowContext], bool]
"""Verifier signature: returns True iff every Tier-1 citation grounds.

Production wires :func:`bba.quote_grounder.verify_citations`; tests
inject deterministic stubs (always-True for happy path, always-False
for the adversarial-grounder case).
"""


def default_verifier(
    result: BatchSubmissionResult, context: PipelineRowContext
) -> bool:
    """Phase-1 placeholder verifier: every attempt grounds.

    Replaced by :mod:`bba.quote_grounder` integration in the next
    ticket. Until then, callers that want to exercise the
    hallucination-suspect branch inject a stub returning ``False``.
    """
    _ = (result, context)
    return True


def apply_batch_results(
    response: RawBatchResponse,
    *,
    audit_store: AuditStore,
    run_id: str,
    contexts: Mapping[str, PipelineRowContext],
    classifier_results: Mapping[str, ClassifierResult] | None = None,
    verifier: Verifier = default_verifier,
) -> PipelineRunResult:
    """Apply a single :class:`RawBatchResponse` to the audit_store.

    For each ``audit_id`` (grouped from ``response.results`` by
    ``custom_id``):

    1. Verify each attempt via ``verifier`` (Phase-1 default: pass).
    2. Pick the winning attempt via the last-verifier-passed rule
       (user constraint #6). If no attempt passes verifier, surface
       the row as ``NEEDS_REVIEW`` with the ``hallucination_suspect``
       review reason.
    3. Build :class:`AuditRow` from the winning result + caller-
       supplied :class:`PipelineRowContext` (no hardcoded clinical
       data — Codex review HIGH #5).
    4. Build one :class:`LlmCall` per persisted attempt.
    5. Write through :meth:`AuditStore.write` — idempotent on
       ``(audit_id, run_id, code_version)``.

    A second call with the same ``response`` + ``contexts`` returns
    an empty ``audit_ids_persisted`` (zero new rows).

    Raises ``KeyError`` when any ``custom_id`` has no matching
    context. The orchestrator fails loud rather than fabricating
    clinical data.
    """
    persisted: list[str] = []

    # When the caller supplies classifier_results explicitly we use
    # them; otherwise compose ClassifierInputs from each context and
    # call the deterministic engine ourselves. The replay path
    # (resume reconciler + property test) hands the classifier_results
    # in pre-computed; the LLM-bound call site in run_pipeline does
    # the same so we run classify() at most once per audit_id.
    resolved_classifier_results: dict[str, ClassifierResult] = (
        dict(classifier_results) if classifier_results is not None else {}
    )

    by_audit_id: dict[str, list[BatchSubmissionResult]] = defaultdict(list)
    for result in response.results:
        by_audit_id[result.custom_id].append(result)

    for audit_id, attempts in by_audit_id.items():
        if audit_id not in contexts:
            raise KeyError(
                f"apply_batch_results: no PipelineRowContext for "
                f"audit_id={audit_id!r}; caller must supply one per result "
                "to avoid silent fabrication of clinical data"
            )
        context = contexts[audit_id]
        classifier = resolved_classifier_results.get(audit_id)
        if classifier is None:
            classifier = _classify_from_context(context)
            resolved_classifier_results[audit_id] = classifier
        attempt_records = tuple(
            _AttemptRecord(
                attempt_id=i,
                result=attempt,
                verifier_pass=verifier(attempt, context),
            )
            for i, attempt in enumerate(attempts)
        )
        winner = select_winning_attempt(attempt_records)
        row = _build_audit_row(
            attempts=attempt_records,
            winner=winner,  # type: ignore[arg-type]
            context=context,
            classifier_result=classifier,
            run_id=run_id,
        )
        calls = [
            _build_llm_call(record.result, attempt_index=record.attempt_id, run_id=run_id)
            for record in attempt_records
        ]
        write_result = audit_store.write(row, calls)
        if not write_result.skipped_idempotent:
            persisted.append(audit_id)

    return PipelineRunResult(
        run_id=run_id,
        audit_ids_persisted=tuple(persisted),
        batch_runs_touched=(),
        orphan_audit_ids=(),
    )


class _AttemptRecord(NamedTuple):
    """In-pipeline record shape consumed by :func:`select_winning_attempt`.

    Wraps a single :class:`BatchSubmissionResult` with the verifier's
    verdict and a stable ``attempt_id`` (the submission-order index).
    The orchestrator emits attempts in order, so the latest index is
    the latest try (escalation attempts come last per PRD §13).
    """

    attempt_id: int
    result: BatchSubmissionResult
    verifier_pass: bool


def select_winning_attempt(
    calls: Sequence[object],
) -> object | None:
    """Pick the winning attempt per user constraint #6.

    Winning attempt = the one whose ``verifier_pass=True`` AND has
    the latest ``attempt_id``. Returns ``None`` when no attempt
    passed verifier — caller routes that to ``NEEDS_REVIEW`` with
    ``hallucination_suspect=True``.

    This is the CANONICAL primitive — :func:`apply_batch_results`
    calls it directly on :class:`_AttemptRecord` tuples emitted by
    the pipeline (Codex review MEDIUM #10: the function was previously
    only exposed and never wired). Callers may also pass mapping-shaped
    records (``{"attempt_id": int, "verifier_pass": bool, ...}``);
    the lookup is duck-typed so the same primitive serves both call
    sites.
    """
    passing = [c for c in calls if _verifier_passed(c)]
    if not passing:
        return None
    return max(passing, key=_attempt_key)


def _classify_from_context(context: "PipelineRowContext") -> ClassifierResult:
    """Compose ClassifierInputs and run the deterministic engine.

    Mirrors :func:`bba.audit_pipeline.pipeline._classifier_inputs_for`
    so the resume / property paths get the same classifier result
    the main pipeline does."""
    from bba.deterministic_classifier import ClassifierInputs, classify

    return classify(
        ClassifierInputs(
            audit_id=context.order.audit_id,
            hb_result=context.hb_result,
            cohort_assignment=context.cohort_assignment,
            order_datetime=context.order.order_datetime,
            procedure_proximity_hours=context.procedure_proximity_hours,
            crystalloid_liters_prior_4h=context.crystalloid_liters_prior_4h,
        )
    )


def _verifier_passed(call: object) -> bool:
    """Return True iff ``call``'s ``verifier_pass`` field is truthy.

    Supports both ``Mapping`` (test fixtures) and attribute-bearing
    record types (production verified-call tuples) so the winning-
    attempt rule has one implementation across both call sites.
    """
    if isinstance(call, Mapping):
        return bool(call.get("verifier_pass"))
    return bool(getattr(call, "verifier_pass", False))


def _attempt_key(call: object) -> int:
    """Extract ``attempt_id`` as the comparison key.

    Raises ``TypeError`` if neither shape provides the field — the
    caller is handing us malformed data and should not silently fall
    back to 0.
    """
    if isinstance(call, Mapping):
        attempt = call.get("attempt_id")
    else:
        attempt = getattr(call, "attempt_id", None)
    if attempt is None:
        raise TypeError(
            f"call {call!r} is missing 'attempt_id'; "
            "winning-attempt rule needs a deterministic ordering key"
        )
    return int(attempt)


def _build_audit_row(
    *,
    attempts: Sequence[_AttemptRecord],
    winner: _AttemptRecord | None,
    context: PipelineRowContext,
    classifier_result: ClassifierResult,
    run_id: str,
) -> AuditRow:
    """Translate the winning :class:`BatchSubmissionResult` + caller
    context + deterministic classifier result into a persistable
    :class:`AuditRow`.

    Every clinical / reproducibility field comes from ``context`` or
    ``classifier_result`` so a re-application of the same response
    produces byte-identical bytes (the audit_store idempotency contract
    relies on this). There is NO hardcoded clinical data (Codex review
    HIGH #5).
    """
    rule_classification = classifier_result.classification

    if winner is None:
        # No attempt passed verifier → hallucination-suspect path
        # (user constraint #6). The final classification is forced
        # to NEEDS_REVIEW and the review_reason carries the typed
        # slug so operators can quarantine the row.
        last_result = attempts[-1].result if attempts else None
        return _audit_row_for_needs_review(
            run_id=run_id,
            context=context,
            classifier_result=classifier_result,
            review_reason="hallucination_suspect",
            verifier_pass=False,
            verifier_retries=max(len(attempts) - 1, 0),
            model_id=last_result.model_id if last_result else "unknown",
            reasoning_en="",
            reasoning_th="",
            indications=(),
            negative_evidence=(),
            confidence=0.0,
            escalated=False,
        )

    winning_result = winner.result
    parsed = _classification_from_result(winning_result)
    final_classification = parsed.classification
    review_reason = parsed.parse_failure_reason
    summary_en, summary_th = _summaries_from_result(winning_result)
    indications = _indications_from_result(winning_result)
    negative_evidence = _negative_evidence_from_result(winning_result)
    confidence = _confidence_from_attempts(indications)
    escalated = any("opus" in record.result.model_id for record in attempts)
    return AuditRow(
        audit_id=context.order.audit_id,
        run_id=run_id,
        run_timestamp=winning_result.request_timestamp,
        hn_hash=context.hn_hash,
        an_hash=context.an_hash,
        reqno=context.order.reqno,
        order_datetime=context.order.order_datetime,
        products_ordered=tuple(context.order.products_ordered),
        hb_value=context.hb_result.value_g_dl if context.hb_result.value_g_dl is not None else 0.0,
        hb_datetime=context.hb_result.datetime_utc
        if context.hb_result.datetime_utc is not None
        else context.order.order_datetime,
        hb_freshness=context.hb_result.freshness,
        hb_source=str(context.hb_result.source) if context.hb_result.source else "missing",
        vitals_sbp=context.vitals_result.vitals.sbp,
        vitals_hr=context.vitals_result.vitals.hr,
        vitals_timestamp=context.vitals_result.note_timestamp,
        vitals_source=context.vitals_result.source.value,
        prior_rbc_units_24h=context.prior_rbc_units_24h,
        prior_rbc_units_7d=context.prior_rbc_units_7d,
        cohort_threshold=context.cohort_assignment.threshold
        if context.cohort_assignment.threshold is not None
        else classifier_result.cohort_threshold or 0.0,
        delta_hb_window_results=tuple(
            {
                "window_hours": w.window_hours,
                "threshold_g_dl": w.threshold_g_dl,
                "triggered": w.triggered,
                "drop_g_dl": w.drop_g_dl,
            }
            for w in context.hb_result.delta_hb_windows
        ),
        rule_classification=rule_classification,
        final_classification=final_classification,
        cohort_applied=context.cohort_assignment.label.value,
        indications_json=tuple(indications),
        negative_evidence_json=tuple({"text": ne} for ne in negative_evidence),
        confidence=confidence,
        reasoning_summary_thai=summary_th,
        reasoning_summary_en=summary_en,
        needs_human_review=final_classification == "NEEDS_REVIEW",
        review_reason=review_reason,
        model_id=winning_result.model_id,
        prompt_hash=context.prompt_hash,
        evidence_bundle_hash=context.evidence_bundle_hash,
        redactor_version=context.redactor_version,
        redactor_model_sha=context.redactor_model_sha,
        policy_version=context.policy_version,
        verifier_pass=True,
        verifier_retries=max(len(attempts) - 1, 0),
        escalated_to_opus=escalated,
    )


def _audit_row_for_needs_review(
    *,
    run_id: str,
    context: PipelineRowContext,
    classifier_result: ClassifierResult,
    review_reason: str,
    verifier_pass: bool,
    verifier_retries: int,
    model_id: str,
    reasoning_en: str,
    reasoning_th: str,
    indications: tuple[dict[str, object], ...],
    negative_evidence: tuple[str, ...],
    confidence: float,
    escalated: bool,
) -> AuditRow:
    """Construct a NEEDS_REVIEW AuditRow with a typed review_reason.

    Used by the hallucination-suspect branch (verifier rejected every
    attempt). The clinical fields still come from ``context`` so the
    row is fully reproducible — the only "missing" data is the LLM
    answer, which is exactly what NEEDS_REVIEW signals.
    """
    return AuditRow(
        audit_id=context.order.audit_id,
        run_id=run_id,
        run_timestamp=context.order.order_datetime,
        hn_hash=context.hn_hash,
        an_hash=context.an_hash,
        reqno=context.order.reqno,
        order_datetime=context.order.order_datetime,
        products_ordered=tuple(context.order.products_ordered),
        hb_value=context.hb_result.value_g_dl if context.hb_result.value_g_dl is not None else 0.0,
        hb_datetime=context.hb_result.datetime_utc
        if context.hb_result.datetime_utc is not None
        else context.order.order_datetime,
        hb_freshness=context.hb_result.freshness,
        hb_source=str(context.hb_result.source) if context.hb_result.source else "missing",
        vitals_sbp=context.vitals_result.vitals.sbp,
        vitals_hr=context.vitals_result.vitals.hr,
        vitals_timestamp=context.vitals_result.note_timestamp,
        vitals_source=context.vitals_result.source.value,
        prior_rbc_units_24h=context.prior_rbc_units_24h,
        prior_rbc_units_7d=context.prior_rbc_units_7d,
        cohort_threshold=context.cohort_assignment.threshold
        if context.cohort_assignment.threshold is not None
        else classifier_result.cohort_threshold or 0.0,
        delta_hb_window_results=tuple(
            {
                "window_hours": w.window_hours,
                "threshold_g_dl": w.threshold_g_dl,
                "triggered": w.triggered,
                "drop_g_dl": w.drop_g_dl,
            }
            for w in context.hb_result.delta_hb_windows
        ),
        rule_classification=classifier_result.classification,
        final_classification="NEEDS_REVIEW",
        cohort_applied=context.cohort_assignment.label.value,
        indications_json=indications,
        negative_evidence_json=tuple({"text": ne} for ne in negative_evidence),
        confidence=confidence,
        reasoning_summary_thai=reasoning_th,
        reasoning_summary_en=reasoning_en,
        needs_human_review=True,
        review_reason=review_reason,
        model_id=model_id,
        prompt_hash=context.prompt_hash,
        evidence_bundle_hash=context.evidence_bundle_hash,
        redactor_version=context.redactor_version,
        redactor_model_sha=context.redactor_model_sha,
        policy_version=context.policy_version,
        verifier_pass=verifier_pass,
        verifier_retries=verifier_retries,
        escalated_to_opus=escalated,
    )


def _build_llm_call(
    result: BatchSubmissionResult, *, attempt_index: int, run_id: str
) -> LlmCall:
    """Translate one :class:`BatchSubmissionResult` into a persistable
    :class:`LlmCall` whose ``call_id`` is deterministic in the result's
    identity (so re-applying the same response writes the same file)."""
    fingerprint = hashlib.sha256(
        f"{run_id}|{result.custom_id}|{result.model_id}|{attempt_index}".encode("utf-8")
    ).hexdigest()[:16]
    call_id = f"call-{result.custom_id}-{attempt_index}-{fingerprint}"
    return LlmCall(
        call_id=call_id,
        audit_id=result.custom_id,
        run_id=run_id,
        model_id=result.model_id,
        anthropic_version=result.anthropic_version,
        prompt_cache_id=result.prompt_cache_id,
        request_json=result.request_json,
        response_json=result.raw_response_json,
        request_timestamp=result.request_timestamp,
        latency_ms=result.latency_ms,
        extended_thinking_blocks=result.extended_thinking_blocks,
        cold_storage_uri=None,
    )


_VALID_CLASSIFICATIONS: frozenset[Classification] = frozenset(
    {
        "APPROPRIATE",
        "INAPPROPRIATE",
        "NEEDS_REVIEW",
        "INSUFFICIENT_EVIDENCE",
        "POTENTIALLY_INAPPROPRIATE",
    }
)


class _ParsedClassification:
    """Outcome of :func:`_classification_from_result`.

    ``parse_failure_reason`` is ``None`` when the structured-output
    payload matches the contract; otherwise it carries a typed slug
    that lands on the persisted ``AuditRow.review_reason`` field so
    operators can distinguish clinical NEEDS_REVIEW from schema drift /
    API breakage.
    """

    __slots__ = ("classification", "parse_failure_reason")

    def __init__(
        self,
        classification: Classification,
        parse_failure_reason: str | None,
    ) -> None:
        self.classification = classification
        self.parse_failure_reason = parse_failure_reason


def _classification_from_result(result: BatchSubmissionResult) -> _ParsedClassification:
    """Extract the classification from the structured-output payload.

    The payload mirrors :class:`bba.llm_client.LlmClassificationResponse`
    under ``content[0].input.classification``. On any shape drift the
    function returns ``NEEDS_REVIEW`` *with a typed parse-failure
    reason* (Codex review MEDIUM #7).
    """
    content = result.raw_response_json.get("content", [])
    if not content:
        return _ParsedClassification("NEEDS_REVIEW", "empty_response")
    first = content[0]
    if not (isinstance(first, Mapping) and first.get("type") == "tool_use"):
        return _ParsedClassification("NEEDS_REVIEW", "tool_use_missing")
    input_payload = first.get("input", {})
    if not isinstance(input_payload, Mapping):
        return _ParsedClassification("NEEDS_REVIEW", "schema_mismatch")
    value = input_payload.get("classification")
    if not isinstance(value, str):
        return _ParsedClassification("NEEDS_REVIEW", "schema_mismatch")
    if value not in _VALID_CLASSIFICATIONS:
        return _ParsedClassification("NEEDS_REVIEW", "classification_out_of_set")
    return _ParsedClassification(value, None)


def _indications_from_result(
    result: BatchSubmissionResult,
) -> tuple[dict[str, object], ...]:
    """Read the indication list off the structured-output payload."""
    content = result.raw_response_json.get("content", [])
    if not content:
        return ()
    first = content[0]
    if not isinstance(first, Mapping) or first.get("type") != "tool_use":
        return ()
    input_payload = first.get("input", {})
    if not isinstance(input_payload, Mapping):
        return ()
    indications = input_payload.get("indications", [])
    if not isinstance(indications, Sequence) or isinstance(indications, str | bytes):
        return ()
    return tuple(dict(i) for i in indications if isinstance(i, Mapping))


def _negative_evidence_from_result(
    result: BatchSubmissionResult,
) -> tuple[str, ...]:
    """Read the negative_evidence list off the structured-output payload."""
    content = result.raw_response_json.get("content", [])
    if not content:
        return ()
    first = content[0]
    if not isinstance(first, Mapping) or first.get("type") != "tool_use":
        return ()
    input_payload = first.get("input", {})
    if not isinstance(input_payload, Mapping):
        return ()
    ne = input_payload.get("negative_evidence", [])
    if not isinstance(ne, Sequence) or isinstance(ne, str | bytes):
        return ()
    return tuple(item for item in ne if isinstance(item, str))


def _summaries_from_result(result: BatchSubmissionResult) -> tuple[str, str]:
    """Extract (en, th) reasoning summaries from the payload."""
    content = result.raw_response_json.get("content", [])
    if not content:
        return ("", "")
    first = content[0]
    if not isinstance(first, Mapping) or first.get("type") != "tool_use":
        return ("", "")
    input_payload = first.get("input", {})
    if not isinstance(input_payload, Mapping):
        return ("", "")
    en = input_payload.get("reasoning_summary_en", "")
    th = input_payload.get("reasoning_summary_th", "")
    return (
        en if isinstance(en, str) else "",
        th if isinstance(th, str) else "",
    )


def _confidence_from_attempts(
    indications: tuple[dict[str, object], ...],
) -> float:
    """Pool indication confidences for the persisted ``confidence`` field.

    Uses the minimum indication confidence as the row-level value
    (conservative — one weak citation drags the row down). Returns
    0.0 when no indications are present so the field is non-null.
    """
    if not indications:
        return 0.0
    values: list[float] = []
    for ind in indications:
        raw = ind.get("confidence")
        if isinstance(raw, (int, float)):
            values.append(float(raw))
    if not values:
        return 0.0
    return min(values)


__all__ = [
    "Verifier",
    "apply_batch_results",
    "default_verifier",
    "select_winning_attempt",
]
