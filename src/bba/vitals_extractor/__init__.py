"""bba.vitals_extractor — vital-sign extraction from HOSxP free-text columns.

See issue #6 for acceptance criteria. Implementation Decisions §4 in the PRD
defines the regex-first / LLM-fallback contract, the ±6 h window rule, the
source-provenance tags, and the sanity bounds.

Inputs are :class:`VitalsNote` records derived from the ingested
``IPDADMPROGRESS`` (``OBJECTIVE`` column — cleaner SOAP) and ``IPDNRFOCUSDT``
(``FOCUS``/``ACTION``/``RESPONSE`` — fresher but noisier) tables; the pipeline
returns a single :class:`VitalsResult` per order anchor with the chosen note's
:class:`VitalSigns`, the source provenance, and any quality flags.
"""

from bba.vitals_extractor.bounds import (
    BT_MAX,
    BT_MIN,
    DBP_MAX,
    DBP_MIN,
    HR_MAX,
    HR_MIN,
    MAP_MAX,
    MAP_MIN,
    RR_MAX,
    RR_MIN,
    SBP_MAX,
    SBP_MIN,
    is_bt_valid,
    is_dbp_valid,
    is_hr_valid,
    is_map_valid,
    is_rr_valid,
    is_sbp_valid,
)
from bba.vitals_extractor.extractor import extract_vitals_from_text
from bba.vitals_extractor.hemodynamic import scan_hemodynamics
from bba.vitals_extractor.models import (
    HemodynamicSummary,
    LLMFallback,
    PeriopFinding,
    PeriopSummary,
    SourceProvenance,
    VasopressorMention,
    VitalSigns,
    VitalsFlag,
    VitalsNote,
    VitalsResult,
)
from bba.vitals_extractor.periop import scan_periop
from bba.vitals_extractor.pipeline import extract_vitals

__all__ = [
    "BT_MAX",
    "BT_MIN",
    "DBP_MAX",
    "DBP_MIN",
    "HR_MAX",
    "HR_MIN",
    "MAP_MAX",
    "MAP_MIN",
    "HemodynamicSummary",
    "LLMFallback",
    "PeriopFinding",
    "PeriopSummary",
    "RR_MAX",
    "RR_MIN",
    "SBP_MAX",
    "SBP_MIN",
    "SourceProvenance",
    "VasopressorMention",
    "VitalSigns",
    "VitalsFlag",
    "VitalsNote",
    "VitalsResult",
    "extract_vitals",
    "extract_vitals_from_text",
    "is_bt_valid",
    "is_dbp_valid",
    "is_hr_valid",
    "is_map_valid",
    "is_rr_valid",
    "is_sbp_valid",
    "scan_hemodynamics",
    "scan_periop",
]
