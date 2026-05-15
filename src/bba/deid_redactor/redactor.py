"""Top-level :func:`redact_bundle` orchestration.

Composes the per-step transforms into one pure function:

1. Apply the age cap on ``request.patient_age_years``.
2. Call the :class:`KAnonymityGate` once with the request's QI tuple;
   record the group size and the pass/fail decision.
3. For each :class:`NoteInput`:
   a. Call ``backend.redact(note.text)`` to get the redacted text + spans.
   b. Run the :class:`RoleClassifier` over PERSON spans; upgrade ``[PERSON]``
      placeholders in the redacted text to role tokens.
   c. Replace non-PERSON placeholders with their type-matching
      :class:`RoleToken` (``[DATE]``, ``[LOCATION]``, ...).
   d. Shift literal in-text dates to ``Day N`` offsets.
   e. Detect semantic degradation (>4 PERSON-class tokens in any 50-char
      window).
4. Aggregate semantic-degradation across notes; OR it with the k-anonymity
   decision to compute :attr:`route_to_needs_review`.
5. Build the canonical envelope, compute the hash, return the frozen
   :class:`RedactionResult`.

The function is PURE in the literal sense — given the same inputs
(including same backend behavior, same classifier behavior, same gate
return value), the output bytes are identical. This is what unlocks the
issue #17 AC "Bundle-hash stability".

The function NEVER raises on a quality-gate failure (k-anonymity below
threshold, semantic degradation): those route to NEEDS_REVIEW via the
result. It DOES raise on backend errors (caught and re-raised as
:class:`BackendRedactionError`) and on contract violations (span count
mismatch, naive admission datetime, etc.).
"""

from __future__ import annotations

from bba.deid_redactor.models import (
    RedactionRequest,
    RedactionResult,
    RedactorBackend,
    RoleClassifier,
    KAnonymityGate,
)


def redact_bundle(
    request: RedactionRequest,
    *,
    backend: RedactorBackend,
    k_gate: KAnonymityGate,
    role_classifier: RoleClassifier | None = None,
) -> RedactionResult:
    """Apply the full deid_redactor pipeline to ``request``.

    Arguments:

    * ``request`` — :class:`RedactionRequest` with notes, QI tuple,
      admission date, raw age, and redactor-version metadata.
    * ``backend`` — :class:`RedactorBackend` Protocol; supplies the
      pre-role redaction.
    * ``k_gate`` — :class:`KAnonymityGate` Protocol; supplies the group
      size for the request's QI tuple.
    * ``role_classifier`` — optional :class:`RoleClassifier`; defaults
      to :func:`bba.deid_redactor.roles.default_role_classifier` (cue-
      based, deterministic).

    Returns a frozen :class:`RedactionResult` whose ``redaction_hash`` is
    the SHA-256 over the canonical envelope of all of the above.

    Raises:

    * :class:`bba.deid_redactor.exceptions.BackendRedactionError` — the
      backend raised, or returned an invalid result (span count mismatch).
    * :class:`bba.deid_redactor.exceptions.DateShiftError` — admission
      date or in-text dates failed strict validation.

    Pure function: no I/O, no global state, no logging.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #17")
