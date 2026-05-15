"""Custom exceptions for the :mod:`bba.deid_redactor` module.

Two raise paths exist in the wrapper:

* :class:`DeidRedactorError` — base for any failure in the post-processing
  wrapper itself (date parse, role classification, hash mismatch).
* :class:`BackendRedactionError` — the underlying ``thai-medical-deid`` backend
  raised. We wrap it so the audit pipeline can branch on
  ``isinstance(exc, DeidRedactorError)`` without importing the backend's
  exception types (the backend is plugged in via the
  :class:`bba.deid_redactor.models.RedactorBackend` Protocol).

NEEDS_REVIEW routing (k-anonymity fail / semantic degradation) is NOT an
exception — it is a value on :class:`bba.deid_redactor.models.RedactionResult`.
Quality-gate failures must travel through the audit chain so reviewers can
audit them, not raise mid-pipeline and lose the bundle.
"""

from __future__ import annotations


class DeidRedactorError(Exception):
    """Base class for failures inside the deid_redactor post-processing wrapper."""


class BackendRedactionError(DeidRedactorError):
    """The ``thai-medical-deid`` backend raised while redacting a note.

    The wrapper re-raises as this type so the audit pipeline can catch a
    single exception class regardless of which backend implementation is
    plugged in via the :class:`bba.deid_redactor.models.RedactorBackend`
    Protocol. The original exception is chained via ``__cause__``.
    """


class DateShiftError(DeidRedactorError):
    """A date inside the redacted text could not be shifted.

    Raised when the date-shift step encounters a parseable date that is
    before the admission date by more than the configured slack (a date
    parser bug — the audit pipeline expects clinical dates to be at or after
    admission), or when admission_date is naive (violating the project's
    tz-aware-UTC contract — see :mod:`bba.ingest`).
    """


class HashMismatchError(DeidRedactorError):
    """The recomputed :class:`RedactionResult.redaction_hash` does not match.

    Asserted by the result model validator: same canonical-JSON bytes →
    same hash. A mismatch means a downstream caller forged a hash or the
    canonical-JSON serializer drifted under a refactor — either way, the
    audit-chain replay invariant is broken (mirrors the
    :mod:`bba.evidence_bundle_builder` ``bundle_hash`` contract).
    """
