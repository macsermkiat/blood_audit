"""bba.evidence_bundle_builder — note ranking + canonical JSON evidence bundle.

See issue #16 for acceptance criteria. Implementation Decisions §7 in the PRD
defines the per-source windows (Diagnosis AN-scoped; IPDADMPROGRESS +/- 24 h
cap 8; IPDNRFOCUSDT +/- 24 h cap 10 in a 5-before / 5-after split; MED -72 h /
+24 h; Lab Hb -7 d; vitals +/- 6 h), the SOAP section priority on truncation
(ASSESSMENT + PLAN first, OBJECTIVE next, SUBJECTIVE last), the stable
evidence-ID assignment (E1, E2, ...), the canonical-JSON serialization, and
the SHA-256 bundle-hash contract.

This module feeds :mod:`bba.deid_redactor` (#17), which receives the bundle
and returns a redacted variant whose ``bundle_hash`` round-trips through
:mod:`bba.audit_store.AuditRow.evidence_bundle_hash`.
"""

from bba.evidence_bundle_builder.builder import (
    CAP_FOCUS_AFTER,
    CAP_FOCUS_BEFORE,
    CAP_PROGRESS,
    DEFAULT_CHAR_CAP,
    WINDOW_FOCUS,
    WINDOW_HB_BEFORE,
    WINDOW_MED_AFTER,
    WINDOW_MED_BEFORE,
    WINDOW_PROGRESS,
    WINDOW_VITALS,
    build_evidence_bundle,
)
from bba.evidence_bundle_builder.canonical import bundle_hash, canonical_serialize
from bba.evidence_bundle_builder.models import (
    DiagnosisRecord,
    EvidenceBundle,
    EvidenceInputs,
    EvidenceItem,
    EvidenceSource,
    FocusNote,
    HbRecord,
    HbSource,
    MedRecord,
    OrderAnchor,
    ProgressNote,
    SOAPSection,
    VitalsNoteSource,
    VitalsRecord,
)
from bba.evidence_bundle_builder.ranking import (
    SECTION_PRIORITY,
    parse_soap_sections,
    split_focus_notes_5_5,
    truncate_to_char_cap,
)

__all__ = [
    "CAP_FOCUS_AFTER",
    "CAP_FOCUS_BEFORE",
    "CAP_PROGRESS",
    "DEFAULT_CHAR_CAP",
    "DiagnosisRecord",
    "EvidenceBundle",
    "EvidenceInputs",
    "EvidenceItem",
    "EvidenceSource",
    "FocusNote",
    "HbRecord",
    "HbSource",
    "MedRecord",
    "OrderAnchor",
    "ProgressNote",
    "SECTION_PRIORITY",
    "SOAPSection",
    "VitalsNoteSource",
    "VitalsRecord",
    "WINDOW_FOCUS",
    "WINDOW_HB_BEFORE",
    "WINDOW_MED_AFTER",
    "WINDOW_MED_BEFORE",
    "WINDOW_PROGRESS",
    "WINDOW_VITALS",
    "build_evidence_bundle",
    "bundle_hash",
    "canonical_serialize",
    "parse_soap_sections",
    "split_focus_notes_5_5",
    "truncate_to_char_cap",
]
