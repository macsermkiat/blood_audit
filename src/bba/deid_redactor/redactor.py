"""Top-level :func:`redact_bundle` orchestration.

Composes the per-step transforms into one pure function:

1. For each :class:`NoteInput`:
   a. Call ``backend.redact(note.text)`` to get the redacted text + spans.
   b. Run the :class:`RoleClassifier` over PERSON spans; upgrade ``[PERSON]``
      placeholders in the redacted text to role tokens.
   c. Shift literal in-text dates to ``Day N`` offsets from
      ``request.anchor_date``.
   d. Detect semantic degradation (>4 PERSON-class tokens in any 50-char
      window).
2. Aggregate semantic-degradation across notes to compute
   :attr:`route_to_needs_review`.
3. Build the canonical envelope, compute the hash, return the frozen
   :class:`RedactionResult`.

The function NEVER raises on a quality-gate failure (semantic
degradation): it routes to NEEDS_REVIEW via the result. It DOES raise on
backend errors (caught and re-raised as :class:`BackendRedactionError`).
"""

from __future__ import annotations

from datetime import date

from bba.deid_redactor.canonical import build_envelope, compute_redaction_hash
from bba.deid_redactor.date_shift import shift_date_spans_in_text, shift_dates_in_text
from bba.deid_redactor.exceptions import BackendRedactionError
from bba.deid_redactor.models import (
    NeedsReviewReason,
    NoteInput,
    RedactedNote,
    RedactionRequest,
    RedactionResult,
    RedactorBackend,
    RoleClassifier,
)
from bba.deid_redactor.roles import default_role_classifier, upgrade_person_tokens
from bba.deid_redactor.semantic import detect_semantic_degradation


def redact_bundle(
    request: RedactionRequest,
    *,
    backend: RedactorBackend,
    role_classifier: RoleClassifier | None = None,
) -> RedactionResult:
    """Apply the full deid_redactor pipeline to ``request``."""
    classifier: RoleClassifier = (
        role_classifier if role_classifier is not None else default_role_classifier
    )

    redacted_notes: list[RedactedNote] = []
    any_semantic_degraded = False

    for note in request.notes:
        redacted_text = _redact_one_note(
            note=note,
            backend=backend,
            classifier=classifier,
            anchor_date=request.anchor_date,
        )
        degraded = detect_semantic_degradation(redacted_text)
        any_semantic_degraded = any_semantic_degraded or degraded
        redacted_notes.append(
            RedactedNote(
                note_id=note.note_id,
                redacted_text=redacted_text,
                semantic_degraded=degraded,
            )
        )

    reasons: list[NeedsReviewReason] = []
    if any_semantic_degraded:
        reasons.append(NeedsReviewReason.SEMANTIC_DEGRADATION)

    route = bool(reasons)

    envelope = build_envelope(
        notes=[
            {
                "note_id": n.note_id,
                "redacted_text": n.redacted_text,
                "semantic_degraded": n.semantic_degraded,
            }
            for n in redacted_notes
        ],
        redactor_version={
            "version": request.redactor_version.version,
            "model_sha": request.redactor_version.model_sha,
            "gazetteer_version": request.redactor_version.gazetteer_version,
        },
        route_to_needs_review=route,
        needs_review_reasons=[r.value for r in reasons],
    )
    redaction_hash = compute_redaction_hash(envelope)

    return RedactionResult(
        notes=tuple(redacted_notes),
        redactor_version=request.redactor_version,
        route_to_needs_review=route,
        needs_review_reasons=tuple(reasons),
        redaction_hash=redaction_hash,
    )


def _redact_one_note(
    *,
    note: NoteInput,
    backend: RedactorBackend,
    classifier: RoleClassifier,
    anchor_date: date,
) -> str:
    """Run the per-note redaction sub-pipeline.

    Catches any backend exception and re-raises as
    :class:`BackendRedactionError` so the audit pipeline can branch on
    one exception type regardless of which backend is plugged in.
    """
    try:
        backend_result = backend.redact(note.text)
    except BackendRedactionError:
        raise
    except Exception as exc:
        raise BackendRedactionError(
            f"backend.redact raised on note_id={note.note_id!r}: {exc}"
        ) from exc

    role_upgraded = upgrade_person_tokens(
        redacted_text=backend_result.text,
        original_text=note.text,
        spans=backend_result.spans,
        classifier=classifier,
    )

    # Order: convert backend-tagged DATE placeholders first (one per
    # DATE span, in document order), then catch any remaining literal
    # dates the backend missed via regex over the final text.
    date_span_shifted = shift_date_spans_in_text(
        role_upgraded,
        spans=backend_result.spans,
        anchor_date=anchor_date,
    )
    return shift_dates_in_text(date_span_shifted, anchor_date=anchor_date)
