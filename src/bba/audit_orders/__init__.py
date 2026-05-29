"""bba.audit_orders — per-(HN, REQNO) RBC filter with hard-exclusions.

See issue #4 for acceptance criteria. Implementation Decisions §2 in the PRD
defines the inclusion gates (RBC products LPRC / LDPRC / SDR, BDVSTST in
{4, 5}, CANCELDATE IS NULL, AN-scoped, REQTYPE = 'P'), the hard-exclusion
ICD-10 set (hemoglobinopathy D55/D56/D57/D58), and the anchor-datetime
resolution (REQDATE + REQTIME with BDVSTDATE + BDVSTTIME fallback, flagged
via ``anchor_imputed``). Age-based pediatric exclusion is upstream of this
module (IT pre-filter ``age > 15``); see ``docs/ingest-mapping.md``.

This module is the canonical input for #5 (hb_lookup), #6 (vitals_extractor),
#7 (cohort_detector), and ultimately #8 (deterministic_classifier) — every
downstream stage consumes :class:`AuditOrder` rows (one per included
``(HN, REQNO)``).
"""

from bba.audit_orders.anchor import AnchorResolution, resolve_anchor
from bba.audit_orders.exceptions import AuditOrdersError, UnrecoverableAnchorError
from bba.audit_orders.identity import build_audit_id
from bba.audit_orders.models import (
    AuditOrder,
    AuditOrdersConfig,
    BloodOrderInput,
    ExcludedRecord,
    ExclusionReason,
    FilterResult,
    RBCProduct,
)
from bba.audit_orders.pipeline import build_audit_orders
from bba.audit_orders.rules import (
    ELIGIBLE_STATUS,
    HEMOGLOBINOPATHY_PREFIXES,
    RBC_PRODUCTS,
    check_an_scoped,
    check_cancelled,
    check_hemoglobinopathy,
    check_rbc_product,
    check_request_type,
    check_status,
    is_rbc_product,
    rbc_products_in,
)

__all__ = [
    "AnchorResolution",
    "AuditOrder",
    "AuditOrdersConfig",
    "AuditOrdersError",
    "BloodOrderInput",
    "ELIGIBLE_STATUS",
    "ExcludedRecord",
    "ExclusionReason",
    "FilterResult",
    "HEMOGLOBINOPATHY_PREFIXES",
    "RBCProduct",
    "RBC_PRODUCTS",
    "UnrecoverableAnchorError",
    "build_audit_id",
    "build_audit_orders",
    "check_an_scoped",
    "check_cancelled",
    "check_hemoglobinopathy",
    "check_rbc_product",
    "check_request_type",
    "check_status",
    "is_rbc_product",
    "rbc_products_in",
    "resolve_anchor",
]
