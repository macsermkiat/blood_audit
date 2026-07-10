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
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Final, NamedTuple

from bba.audit_pipeline.bleeding import (
    LLM_OVERCLEAR_MIN_BLEED_CONFIDENCE,
    qualified_bleeding_exempt,
)
from bba.audit_pipeline.models import PipelineRowContext, PipelineRunResult
from bba.audit_store import AuditRow, AuditStore, LlmCall
from bba.audit_store.models import Classification
from bba.cohort_detector import CohortLabel
from bba.deterministic_classifier import (
    PERIOP_MIN_EBL_ML,
    UNIVERSAL_LOW_HB_APPROPRIATE_THRESHOLD,
    ClassifierResult,
)
from bba.llm_client.models import BatchSubmissionResult, RawBatchResponse
from bba.llm_client.parser import parse_platelet_structured_response
from bba.platelet_classifier import (
    PlateletClassifierInputs,
    PlateletClassifierResult,
    classify_platelet,
)
from bba.platelet_guardrail import (
    PLATELET_OVERCLEAR_REVIEW_REASON,
    platelet_overclear_suspect,
)
from bba.platelet_guardrail.models import PlateletHardSignals
from bba.quote_grounder.layers import (
    contiguous_match,
    find_cited_source,
    nfc_normalize,
)
from bba.quote_grounder.models import EvidenceSource
from bba.vitals_extractor import PeriopSummary


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


# =============================================================================
# Peri-op contradiction guardrail (Case 107)
#
# WHY: the verifier only checks that Tier-1 citations *ground* — it cannot
# catch the model trusting an empty structured-procedure list over surgical
# detail buried in free-text nursing notes. On Case 107 the bundle
# deterministically documented a 1500 ml-blood-loss ORIF inside the
# peri-transfusion window, yet the LLM returned INSUFFICIENT_EVIDENCE
# ("no operative procedure documented"). Part 1 surfaces those facts to the
# model; this guardrail is the deterministic backstop for when the prompt
# signal alone still is not enough.
#
# A "hard" peri-op signal disagreeing with a non-committal / negative LLM
# verdict is a CONTRADICTION, not a hallucination: the citations grounded,
# the model just weighed them wrong. We therefore force the row to human
# review with a DISTINCT review_reason so operators can tell it apart from
# the verifier-rejected (hallucination_suspect) path, and we preserve the
# LLM's reasoning/indications so the reviewer sees the conflict in full.
# =============================================================================

PERIOP_GUARDRAIL_MIN_EBL_ML = PERIOP_MIN_EBL_ML
"""Estimated-blood-loss floor (mL) that counts as a hard peri-op signal.

Alias of :data:`bba.deterministic_classifier.PERIOP_MIN_EBL_ML` — the single
source of truth shared with the classifier's missing-Hb auto-approve bar so
the two thresholds cannot drift. Sub-500 mL losses are routine and do not, on
their own, contradict an "insufficient evidence" verdict."""

PERIOP_CONTRADICTION_REVIEW_REASON = "periop_signal_contradiction"
"""Typed review_reason stamped on rows escalated by this guardrail.

Distinct from ``hallucination_suspect`` (verifier rejected every attempt)
so a reviewer dashboard can triage the two failure modes separately."""

_PERIOP_CONTRADICTION_CLASSES: frozenset[Classification] = frozenset(
    {"INSUFFICIENT_EVIDENCE", "POTENTIALLY_INAPPROPRIATE"}
)
"""LLM verdicts that a hard peri-op signal contradicts.

APPROPRIATE / INAPPROPRIATE are committed verdicts the model reached *with*
the evidence in view; only the non-committal ("insufficient") and the
soft-negative ("potentially inappropriate") verdicts are overridden, since
those are exactly the shapes Case 107 produced when the model discounted a
documented surgery."""


def _has_hard_periop_signal(summary: PeriopSummary | None) -> bool:
    """True iff ``summary`` carries a deterministically-extracted peri-op
    signal strong enough to contradict an "insufficient evidence" verdict.

    Any one of documented surgery (op-time), estimated blood loss at or
    above :data:`PERIOP_GUARDRAIL_MIN_EBL_ML`, or an intra-op transfusion
    qualifies — these are the three signals scan_periop recovers from the
    shipped notes, and each alone is enough to warrant a human look when the
    model said the evidence was insufficient.
    """
    if summary is None:
        return False
    if summary.surgical_context or summary.intraop_transfusion:
        return True
    return (
        summary.blood_loss_ml is not None
        and summary.blood_loss_ml >= PERIOP_GUARDRAIL_MIN_EBL_ML
    )


def periop_contradiction(
    classification: Classification, context: PipelineRowContext
) -> bool:
    """True iff a hard peri-op signal in ``context`` contradicts the LLM's
    ``classification`` and the row must be escalated to human review.

    Deterministic and side-effect-free so the override in
    :func:`_build_audit_row` is trivially testable and a re-application of
    the same response reaches the same verdict (replay invariant)."""
    if classification not in _PERIOP_CONTRADICTION_CLASSES:
        return False
    return _has_hard_periop_signal(context.periop_summary)


# =============================================================================
# LLM over-clear guardrail — B1 (Cases 47 / 100)
#
# WHY: the peri-op guardrail above only catches the LLM UNDER-calling
# (INSUFFICIENT_EVIDENCE / POTENTIALLY_INAPPROPRIATE) against a hard signal.
# The 300-case pilot review showed the dominant dangerous failure is the
# OPPOSITE: the LLM returns APPROPRIATE on a gray-zone / missing-Hb order the
# deterministic leg deliberately WITHHELD (NEEDS_REVIEW / INSUFFICIENT_EVIDENCE),
# resting on soft or misread indications — a stale-history epistaxis (Case 47,
# 68062324) or a specialist "keep Hb > 9" target misread as breached at Hb 9.4
# (Case 100, 68069089).
#
# This is the symmetric arm: when the LLM upgrades a withholding deterministic
# verdict to APPROPRIATE and no accepted exemption justifies it, spec #89 / T5
# requires the guardrail to ASSERT INAPPROPRIATE with a distinct review_reason.
# ``needs_human_review`` then auto-clears because it derives from the final
# classification. The prompt teaches the same clear-cut policy; this guardrail
# is the deterministic net for when the model clears anyway.
#
# "Hard signal" is deterministic-only (B1): a genuinely low Hb (< 7.0), a hard
# peri-op signal, an MTP cohort, or hemodynamic instability. Bleeding /
# symptomatic anaemia prose remains untrusted except for the one committee-
# accepted exemption: an ACTIVE_BLEEDING indication with its own confidence
# >= 0.8 and either volume strictly > 300 mL or a life-threatening marker (see
# bleeding.py), and whose quote GROUNDS in the row's own evidence bundle
# (:func:`_grounded_indications` — the batch path's verifier is still the
# Phase-1 pass-through, so quote existence must be checked here, not assumed).
# Small or merely qualitative bleeds do not clear the guardrail.
# =============================================================================

LLM_OVERCLEAR_REVIEW_REASON = "llm_overclear_suspect"
"""Review-reason slug for an over-clear that could not be safely asserted.

Pre-#94 every over-clear floor stamped this. Post-#94 it survives on two
narrow paths: (1) an over-clear whose tool payload is missing a
schema-required list field (:func:`_rbc_payload_well_formed`) — drift may
have dropped cited evidence, so the row floors to human review instead of
asserting ``INAPPROPRIATE`` on evidence-absence we cannot distinguish from
loss; (2) an over-clear citing a grounded, high-confidence ACS indication
(:func:`_grounded_acs_indication`) — a prompt-defined hard code the
structured system cannot verify, so neither asserting nor auto-clearing is
safe. Historical pre-#94 rows also carry this value.
"""

LLM_OVERCLEAR_ASSERT_REASON = "llm_overclear_asserted_inappropriate"
"""Reason slug for a clear-cut over-clear assertion.

WHY: spec #89 requires the committee's clear-cut ``INAPPROPRIATE`` call rather
than deferral; ``needs_human_review`` auto-clears from the final classification.
"""

LLM_NATIVE_REVIEW_ASSERT_REASON = "llm_native_review_asserted_inappropriate"
"""Reason slug for converting a well-formed native LLM hedge.

WHY: the prompt no longer allows ``NEEDS_REVIEW``. A hedge with reasoning, no
structured hard signal, and no qualified bleed becomes the clear-cut verdict.
Parse failures already have a non-None reason and are never converted, while
the non-empty-reasoning gate leaves unexplained verdicts to the empty-reasoning
net. Platelet rows are excluded by the component gate.
"""

EMPTY_REASONING_REVIEW_REASON = "empty_reasoning"
"""Review-reason slug for a verdict with no reasoning in either language.

WHY: pilot run 2026-07-06 contained 9 rows whose reasoning summaries
were completely empty — one classified APPROPRIATE with
needs_human_review=False, i.e. an unexplained automatic clear. A
verdict the committee cannot audit is floored to NEEDS_REVIEW.
"""

_LLM_ASSERT_REASONS: frozenset[str] = frozenset(
    {LLM_OVERCLEAR_ASSERT_REASON, LLM_NATIVE_REVIEW_ASSERT_REASON}
)

LLM_OVERCLEAR_UNSTABLE_SBP: float = 90.0
"""Systolic BP (mmHg) strictly below which the patient is hemodynamically
unstable — a hard signal that exempts an LLM APPROPRIATE from the over-clear
guardrail (transfusing an unstable patient in the gray zone is defensible)."""

LLM_OVERCLEAR_UNSTABLE_HR: float = 120.0
"""Heart rate (bpm) strictly above which the patient is tachycardic /
hemodynamically stressed — the second hard hemodynamic exemption signal."""

_LLM_OVERCLEAR_DET_VERDICTS: frozenset[Classification] = frozenset(
    {"NEEDS_REVIEW", "INSUFFICIENT_EVIDENCE", "POTENTIALLY_INAPPROPRIATE"}
)
"""Deterministic verdicts that withheld a clear.

An LLM ``APPROPRIATE`` upgrading any of these is an over-clear candidate. Per
spec #89, the high-Hb (Hb >= 10) ``POTENTIALLY_INAPPROPRIATE`` verdict is also
withheld: #93 dispatches the HB_GT_10 override prompt, but a soft clear there
must still be asserted ``INAPPROPRIATE``. The hard-signal exemption set is
unchanged.
"""


def _has_hard_hemodynamic_signal(context: PipelineRowContext) -> bool:
    """True iff the ±6 h vitals show hypotension (SBP < 90) or tachycardia
    (HR > 120) — the hemodynamic-instability arm of the hard-signal set."""
    vitals = context.vitals_result.vitals
    if vitals.sbp is not None and vitals.sbp < LLM_OVERCLEAR_UNSTABLE_SBP:
        return True
    return vitals.hr is not None and vitals.hr > LLM_OVERCLEAR_UNSTABLE_HR


def _has_structured_hard_signal(context: PipelineRowContext) -> bool:
    """True iff ``context`` carries a deterministic hard signal that justifies
    an LLM APPROPRIATE in the gray zone (B1 exemption set).

    Any one of: a genuinely low Hb (< 7.0 g/dL), an MTP cohort, a hard peri-op
    signal, or hemodynamic instability. Deliberately structured-only — soft
    prose indications are not trusted here (they are what over-cleared the
    motivating cases)."""
    hb = context.hb_result.value_g_dl
    if hb is not None and hb < UNIVERSAL_LOW_HB_APPROPRIATE_THRESHOLD:
        return True
    if context.cohort_assignment.label == CohortLabel.MTP:
        return True
    if _has_hard_periop_signal(context.periop_summary):
        return True
    return _has_hard_hemodynamic_signal(context)


def llm_overclear_suspect(
    final_classification: Classification,
    rule_classification: Classification,
    context: PipelineRowContext,
) -> bool:
    """True iff the LLM over-cleared a withholding deterministic verdict.

    Fires only when the LLM returned ``APPROPRIATE``, the deterministic leg
    withheld the clear (``NEEDS_REVIEW``, ``INSUFFICIENT_EVIDENCE``, or the
    high-Hb ``POTENTIALLY_INAPPROPRIATE`` verdict), and no structured hard
    signal (:func:`_has_structured_hard_signal`) justifies the clear. The
    qualified-bleeding prose exemption is deliberately composed at the call
    site so this pure predicate's signature and structured-signal scope stay
    unchanged. Deterministic and side-effect-free so the override in
    :func:`_build_audit_row` is trivially testable and replay-stable."""
    if final_classification != "APPROPRIATE":
        return False
    if rule_classification not in _LLM_OVERCLEAR_DET_VERDICTS:
        return False
    return not _has_structured_hard_signal(context)


def _indication_element_well_formed(item: object) -> bool:
    """True iff one ``indications[]`` element matches the tool item schema.

    Mirrors ``transport._TOOL_INPUT_SCHEMA``'s item contract: an object with
    string ``code`` / ``quote`` / ``source_id`` and numeric ``confidence``
    (bool is not a number). Shape only — value-level failures (out-of-range
    confidence, a non-qualifying bleed) are semantics, judged downstream by
    :func:`bba.audit_pipeline.bleeding.qualified_bleeding_exempt`.
    """
    if not isinstance(item, Mapping):
        return False
    if not all(
        isinstance(item.get(key), str) for key in ("code", "quote", "source_id")
    ):
        return False
    confidence = item.get("confidence")
    return not isinstance(confidence, bool) and isinstance(confidence, (int, float))


def _rbc_payload_well_formed(result: BatchSubmissionResult) -> bool:
    """True iff the payload carries schema-shaped ``indications`` and
    ``negative_evidence`` — required list fields with schema-shaped elements.

    WHY: the batch path's shallow extraction (:func:`_indications_from_result`)
    cannot distinguish "the model cited nothing" from "citations lost to
    schema drift": a missing key reads back as ``()``, a non-mapping element
    is silently dropped, and a garbled element survives extraction only to be
    skipped by every defensive downstream reader. The RBC tool schema
    (``transport._TOOL_INPUT_SCHEMA``) pins both fields and their element
    shapes, so any deviation means the #94 assert branches must not treat the
    resulting evidence-absence as "no genuine indication". Shape only —
    in-shape value failures stay semantic (bleeding.py). Scoped to the assert
    branches; row-level parse-failure classification is unchanged.
    """
    content = result.raw_response_json.get("content", [])
    if not content:
        return False
    first = content[0]
    if not isinstance(first, Mapping) or first.get("type") != "tool_use":
        return False
    input_payload = first.get("input", {})
    if not isinstance(input_payload, Mapping):
        return False
    indications = input_payload.get("indications")
    if not isinstance(indications, Sequence) or isinstance(indications, str | bytes):
        return False
    if not all(_indication_element_well_formed(item) for item in indications):
        return False
    negative_evidence = input_payload.get("negative_evidence")
    if not isinstance(negative_evidence, Sequence) or isinstance(
        negative_evidence, str | bytes
    ):
        return False
    return all(isinstance(item, str) for item in negative_evidence)


def _grounded_indications(
    indications: tuple[dict[str, object], ...],
    context: PipelineRowContext,
) -> tuple[dict[str, object], ...]:
    """Filter ``indications`` to those whose quote grounds in the row's own
    evidence bundle (quote_grounder Layers 2+3: the cited source exists and
    the quote is a word-boundary contiguous match inside it, NFC-normalized).

    WHY: the batch path's verifier is still the Phase-1 pass-through
    (:func:`default_verifier`), so a winning attempt does NOT guarantee its
    quotes exist in the notes. The qualified-bleeding exemption turns prose
    directly into a final APPROPRIATE with no human review (#94); feeding it
    an unverified quote would let a hallucinated bleed clear a withheld
    order. The committee's prose-trust decision (spec #89 #2) covers the
    model's *characterization* of documented text, not the text's existence
    — existence is verified here.

    Layers 4 (uniqueness) and 5 (min length 25) are deliberately NOT
    applied: genuine bleed quotes are often short ("EBL 400 mL") and a
    re-charted bleed strengthens rather than weakens the evidence; the
    semantic bar (> 300 mL strictly, or a life-threatening marker, at
    confidence >= 0.8) is enforced by
    :func:`bba.audit_pipeline.bleeding.qualified_bleeding_exempt` on the
    filtered list. Fail-closed: a missing/non-string quote or source_id, an
    unknown cited id, or an empty bundle grounds nothing.
    """
    sources = tuple(
        EvidenceSource(source_id=chunk.evidence_id, text=nfc_normalize(chunk.text))
        for chunk in context.evidence_chunks
    )
    grounded: list[dict[str, object]] = []
    for indication in indications:
        quote = indication.get("quote")
        cited_id = indication.get("source_id")
        if not isinstance(quote, str) or not isinstance(cited_id, str):
            continue
        source = find_cited_source(cited_id, sources)
        if source is None:
            continue
        if contiguous_match(nfc_normalize(quote), source.text):
            grounded.append(indication)
    return tuple(grounded)


# The prompt's fixed HARD indication code for acute coronary syndrome /
# active myocardial ischemia (prompt_builder._RBC_INDICATION_VOCABULARY).
_ACS_HARD_CODE = "ACS"


def _grounded_acs_indication(
    grounded_indications: tuple[dict[str, object], ...],
) -> bool:
    """True iff a grounded indication cites the ACS hard code at or above the
    shared prose-trust confidence bar.

    WHY (Codex PR #97 P1): ACS is the one code in the prompt's HARD
    vocabulary with neither a structured extractor
    (:func:`_has_structured_hard_signal` covers low Hb / MTP / peri-op /
    hemodynamic instability) nor a prose exemption path (qualified bleeding
    covers ACTIVE_BLEEDING only). Without this check the guardrail would
    assert ``INAPPROPRIATE`` against a prompt-compliant, grounded ACS clear —
    an unreviewed committee verdict contradicting the prompt's own contract.
    The caller floors such rows to ``NEEDS_REVIEW`` instead; it never
    auto-clears them, because extending prose auto-clear trust beyond
    qualified bleeding is a committee decision (spec #89 accepted bleeding
    only). Confidence is read defensively, same rules as
    :func:`bba.audit_pipeline.bleeding.qualified_bleeding_exempt`:
    non-numeric, bool, NaN, or out-of-[0,1] never counts.
    """
    for indication in grounded_indications:
        code = indication.get("code")
        if not isinstance(code, str) or code.strip().upper() != _ACS_HARD_CODE:
            continue
        confidence = indication.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            continue
        if not (0.0 <= confidence <= 1.0):
            continue
        if confidence >= LLM_OVERCLEAR_MIN_BLEED_CONFIDENCE:
            return True
    return False


# =============================================================================
# Platelet leg helpers (Stage C2)
# =============================================================================


def _platelet_source_repr(source: object | None) -> str | None:
    """Serialize a platelet lab source to its persisted string form.

    ``.value``-aware: a future :class:`enum.StrEnum` source serializes to its
    ``.value``, while today's plain-string ``PlateletSource`` Literal passes
    through unchanged. Returns ``None`` for a missing source."""
    if source is None:
        return None
    if hasattr(source, "value"):
        return str(source.value)
    return str(source)


def _platelet_gate_result(
    context: PipelineRowContext,
    *,
    enable_missing_platelet_defer: bool,
) -> PlateletClassifierResult | None:
    """Run the deterministic platelet gate for a PLATELET row; ``None`` for RBC.

    Stage B MED-1: a PLATELET row's RBC ``classifier_result`` was computed off
    the inert Hb / cohort sentinels (:meth:`PipelineRowContext.for_platelet`) and
    is WRONG for a platelet order. The platelet gate (:func:`classify_platelet`)
    reads the count instead, and its result supplies BOTH the row's
    ``rule_classification`` and its ``platelet_review_ceiling`` so the two are
    computed from one call and cannot drift."""
    if context.component != "platelet":
        return None
    count = (
        context.platelet_result.value_k_ul
        if context.platelet_result is not None
        else None
    )
    return classify_platelet(
        PlateletClassifierInputs(
            audit_id=context.order.audit_id,
            platelet_count=count,
            enable_missing_platelet_defer=enable_missing_platelet_defer,
        )
    )


def _platelet_overclear_floor(
    final_classification: Classification,
    rule_classification: Classification,
    platelet_hard_signals: PlateletHardSignals | None,
) -> bool:
    """True iff the platelet over-clear guardrail must floor this row to review.

    Accepts the already-parsed :class:`PlateletHardSignals` from the primary
    platelet parse (``_build_audit_row`` calls ``parse_platelet_structured_response``
    once and passes the result here, avoiding a second parse).  A ``None``
    signals object (parse failure) leaves the guardrail maximally protective —
    all signals False, so any APPROPRIATE on a withheld deterministic verdict
    floors to human review."""
    hard_signals = platelet_hard_signals or PlateletHardSignals()
    return platelet_overclear_suspect(
        final_classification, rule_classification, hard_signals
    )


def apply_batch_results(
    response: RawBatchResponse,
    *,
    audit_store: AuditStore,
    run_id: str,
    contexts: Mapping[str, PipelineRowContext],
    classifier_results: Mapping[str, ClassifierResult] | None = None,
    verifier: Verifier = default_verifier,
    enable_missing_platelet_defer: bool = False,
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
            enable_missing_platelet_defer=enable_missing_platelet_defer,
        )
        calls = [
            _build_llm_call(
                record.result, attempt_index=record.attempt_id, run_id=run_id
            )
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

    periop = context.periop_summary
    return classify(
        ClassifierInputs(
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
    enable_missing_platelet_defer: bool = False,
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
    platelet_gate = _platelet_gate_result(
        context, enable_missing_platelet_defer=enable_missing_platelet_defer
    )
    rule_classification: Classification = (
        platelet_gate.classification
        if platelet_gate is not None
        else classifier_result.classification
    )

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
            enable_missing_platelet_defer=enable_missing_platelet_defer,
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
    # For platelet rows, parse via parse_platelet_structured_response which
    # enforces the three hard-signal booleans.  A schema mismatch (missing or
    # malformed bools) fails closed to NEEDS_REVIEW regardless of the returned
    # classification — matching the RBC parse-failure→NEEDS_REVIEW contract but
    # applied end-to-end for every platelet verdict, not only APPROPRIATE.
    # The parsed signals are re-used by the over-clear guardrail below so the
    # response is parsed exactly once.
    if context.component == "platelet":
        _plt_outcome = parse_platelet_structured_response(winning_result)
        if _plt_outcome.parse_failure:
            final_classification: Classification = "NEEDS_REVIEW"
            review_reason: str | None = (
                _plt_outcome.parse_failure_reason.value
                if _plt_outcome.parse_failure_reason is not None
                else "schema_mismatch"
            )
        else:
            assert (
                _plt_outcome.parsed is not None
            )  # guaranteed by ParseOutcome contract
            final_classification = _plt_outcome.parsed.classification
            review_reason = None
        _plt_signals = _plt_outcome.platelet_hard_signals
    else:
        parsed = _classification_from_result(winning_result)
        final_classification = parsed.classification
        review_reason = parsed.parse_failure_reason
        _plt_signals = None
    summary_en, summary_th = _summaries_from_result(winning_result)
    indications = _indications_from_result(winning_result)
    negative_evidence = _negative_evidence_from_result(winning_result)
    confidence = _confidence_from_attempts(indications)
    escalated = any("opus" in record.result.model_id for record in attempts)
    # Peri-op contradiction guardrail (Case 107): a hard deterministic
    # peri-op signal overrides a non-committal / soft-negative LLM verdict.
    # We rewrite the classification + review_reason but keep verifier_pass,
    # the reasoning summaries, and the indications so the human reviewer sees
    # exactly what the model concluded and why it is being second-guessed.
    # The peri-op + RBC over-clear guardrails read Hb / cohort / periop_summary
    # off ``context``; on a PLATELET row those are inert sentinels
    # (:meth:`PipelineRowContext.for_platelet`), so both are gated OFF platelet
    # rows (Stage B MED-2). The platelet leg has its own over-clear guardrail
    # below.
    if context.component != "platelet" and periop_contradiction(
        final_classification, context
    ):
        final_classification = "NEEDS_REVIEW"
        review_reason = PERIOP_CONTRADICTION_REVIEW_REASON
    # B1 over-clear guardrail (Cases 47 / 100): assert INAPPROPRIATE when an
    # LLM clears a withheld deterministic verdict without a structured hard
    # signal or the committee-approved qualified-major-bleeding exemption.
    # The exemption only ever sees indications whose quote grounds in the
    # row's evidence bundle — a fabricated bleed quote must never auto-clear.
    # A payload missing a schema-required list field may have LOST cited
    # evidence, so it floors to human review instead of asserting. A grounded
    # high-confidence ACS citation also floors (never asserts, never
    # auto-clears): ACS is in the prompt's HARD vocabulary but has no
    # structured extractor and no prose exemption path. Checked after peri-op
    # so that earlier winner remains authoritative; the LLM reasoning and
    # indications are preserved for auditability.
    elif context.component != "platelet" and llm_overclear_suspect(
        final_classification, rule_classification, context
    ):
        if not _rbc_payload_well_formed(winning_result):
            final_classification = "NEEDS_REVIEW"
            review_reason = LLM_OVERCLEAR_REVIEW_REASON
        else:
            _grounded = _grounded_indications(indications, context)
            if not qualified_bleeding_exempt(_grounded):
                if _grounded_acs_indication(_grounded):
                    # ACS is a prompt-defined hard indication the structured
                    # system cannot see — floor to a human, never assert
                    # against it (and never auto-clear on it either).
                    final_classification = "NEEDS_REVIEW"
                    review_reason = LLM_OVERCLEAR_REVIEW_REASON
                else:
                    final_classification = "INAPPROPRIATE"
                    review_reason = LLM_OVERCLEAR_ASSERT_REASON
            # else: a grounded qualified major bleed keeps the clear
            # APPROPRIATE.
    # WHY: the prompt no longer permits a native NEEDS_REVIEW hedge. Convert a
    # well-formed, explained hedge to the committee's clear-cut verdict only
    # when neither a structured hard signal nor a qualified (grounded) bleed
    # makes it a genuine human case; a drifted payload or a parse/schema
    # failure (non-None reason) stays a human case.
    elif (
        context.component != "platelet"
        and final_classification == "NEEDS_REVIEW"
        and review_reason is None
        and (summary_en.strip() or summary_th.strip())
        and _rbc_payload_well_formed(winning_result)
        and not _has_structured_hard_signal(context)
        and not qualified_bleeding_exempt(_grounded_indications(indications, context))
        # A grounded high-confidence ACS citation makes the hedge a genuine
        # human case (same rationale as the over-clear ACS floor above).
        and not _grounded_acs_indication(_grounded_indications(indications, context))
    ):
        final_classification = "INAPPROPRIATE"
        review_reason = LLM_NATIVE_REVIEW_ASSERT_REASON
    # Platelet over-clear guardrail (Stage C2, "ADD hard signals" ruling): an
    # LLM APPROPRIATE on any withheld platelet verdict with NO grounded platelet
    # hard signal floors to review.  Keyed on context.component only — NOT on
    # PLATELET_LLM_ENABLED — so the guardrail stays active during crash-recovery
    # replay/resume even when the flag is toggled off after the batch was
    # submitted. The flag's sole job is gating SUBMISSION (whether platelet rows
    # enter the LLM leg at all); once a row reaches this persist path it must
    # always be protected. Uses the already-parsed _plt_signals from the primary
    # platelet parse above (no second parse of winning_result).
    elif context.component == "platelet" and _platelet_overclear_floor(
        final_classification, rule_classification, _plt_signals
    ):
        final_classification = "NEEDS_REVIEW"
        review_reason = PLATELET_OVERCLEAR_REVIEW_REASON
    # Empty-reasoning guardrail: a verdict with no reasoning in either
    # language cannot be audited by the committee. Separate `if` (not
    # elif) so it composes with the guardrails above. Preserve genuine review
    # provenance, but overwrite assertion provenance when this net changes the
    # final verdict: invariant — an assertion slug implies INAPPROPRIATE.
    if not summary_en.strip() and not summary_th.strip():
        final_classification = "NEEDS_REVIEW"
        if review_reason is None or review_reason in _LLM_ASSERT_REASONS:
            review_reason = EMPTY_REASONING_REVIEW_REASON
    return AuditRow(
        audit_id=context.order.audit_id,
        run_id=run_id,
        run_timestamp=winning_result.request_timestamp,
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
        component=context.component,
        platelet_value=context.platelet_result.value_k_ul
        if context.platelet_result is not None
        else None,
        platelet_datetime=context.platelet_result.datetime_utc
        if context.platelet_result is not None
        else None,
        platelet_freshness=context.platelet_result.freshness
        if context.platelet_result is not None
        else None,
        platelet_source=_platelet_source_repr(
            context.platelet_result.source
            if context.platelet_result is not None
            else None
        ),
        platelet_review_ceiling=(
            platelet_gate.review_ceiling if platelet_gate is not None else None
        ),
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
    enable_missing_platelet_defer: bool = False,
) -> AuditRow:
    """Construct a NEEDS_REVIEW AuditRow with a typed review_reason.

    Used by the hallucination-suspect branch (verifier rejected every
    attempt). The clinical fields still come from ``context`` so the
    row is fully reproducible — the only "missing" data is the LLM
    answer, which is exactly what NEEDS_REVIEW signals.
    """
    platelet_gate = _platelet_gate_result(
        context, enable_missing_platelet_defer=enable_missing_platelet_defer
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
        rule_classification=(
            platelet_gate.classification
            if platelet_gate is not None
            else classifier_result.classification
        ),
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
        component=context.component,
        platelet_value=context.platelet_result.value_k_ul
        if context.platelet_result is not None
        else None,
        platelet_datetime=context.platelet_result.datetime_utc
        if context.platelet_result is not None
        else None,
        platelet_freshness=context.platelet_result.freshness
        if context.platelet_result is not None
        else None,
        platelet_source=_platelet_source_repr(
            context.platelet_result.source
            if context.platelet_result is not None
            else None
        ),
        platelet_review_ceiling=(
            platelet_gate.review_ceiling if platelet_gate is not None else None
        ),
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


# =============================================================================
# Structured-output tag-leak salvage (pilot run 2026-07-06)
#
# WHY: on 131/165 pilot rows, claude-sonnet-5 serialized BOTH reasoning
# summaries into the ``reasoning_summary_en`` tool field — separated by
# fragments of its internal tool-call tag syntax — and returned an empty
# ``reasoning_summary_th``. Observed separators, verbatim:
#
#   ...EN text...</reasoning_summary_en>
#   <reasoning_summary_th">...TH text...</reasoning_summary_th>          (96x)
#   <reasoning_summary_th>...                                            (28x)
#   <parameter name="reasoning_summary_th">...</parameter>/</invoke>     (rest)
#   <reasoning_summary_th name="reasoning_summary_th">...
#
# The split is fully deterministic, so it is code, not an LLM judgment.
# Salvage only fires when the th field came back empty — a row that
# parsed cleanly passes through byte-identical.
# =============================================================================

_LEAK_EN_CLOSE_RE: Final[re.Pattern[str]] = re.compile(r"</reasoning_summary_en>")
_LEAK_TH_OPEN_RE: Final[re.Pattern[str]] = re.compile(
    r'<(?:parameter\s+name="reasoning_summary_th"'
    r'|reasoning_summary_th(?:\s+name="reasoning_summary_th")?"?)>'
)
_LEAK_TH_CLOSE_RE: Final[re.Pattern[str]] = re.compile(
    r'</(?:parameter|invoke|reasoning_summary_th"?)>'
)


def split_leaked_summaries(en: str, th: str) -> tuple[str, str]:
    """Recover (en, th) from a tag-leaked ``reasoning_summary_en`` blob.

    No-op unless ``th`` is empty AND ``en`` contains the leaked
    ``</reasoning_summary_en>`` separator. An unterminated Thai block
    (no closing tag) is recovered to end-of-string.
    """
    if th.strip():
        return (en, th)
    en_close = _LEAK_EN_CLOSE_RE.search(en)
    if en_close is None:
        return (en, th)
    clean_en = en[: en_close.start()].strip()
    rest = en[en_close.end() :]
    th_open = _LEAK_TH_OPEN_RE.search(rest)
    if th_open is None:
        return (clean_en, th)
    tail = rest[th_open.end() :]
    th_close = _LEAK_TH_CLOSE_RE.search(tail)
    clean_th = (tail[: th_close.start()] if th_close else tail).strip()
    return (clean_en, clean_th)


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
    return split_leaked_summaries(
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
    "EMPTY_REASONING_REVIEW_REASON",
    "LLM_NATIVE_REVIEW_ASSERT_REASON",
    "LLM_OVERCLEAR_ASSERT_REASON",
    "LLM_OVERCLEAR_REVIEW_REASON",
    "LLM_OVERCLEAR_UNSTABLE_HR",
    "LLM_OVERCLEAR_UNSTABLE_SBP",
    "PERIOP_CONTRADICTION_REVIEW_REASON",
    "PERIOP_GUARDRAIL_MIN_EBL_ML",
    "Verifier",
    "apply_batch_results",
    "default_verifier",
    "llm_overclear_suspect",
    "periop_contradiction",
    "select_winning_attempt",
    "split_leaked_summaries",
]
