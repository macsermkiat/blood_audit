"""bba.audit_orders — per-(HN, REQNO) RBC filter with hard-exclusions.

See issue #4 for acceptance criteria. Implementation Decisions §2 in the PRD
defines the inclusion gates (RBC products LPRC / LDPRC / SDR, BDVSTST in
{4, 5}, CANCELDATE IS NULL, AN-scoped, REQTYPE = 'P'), the hard-exclusion
ICD-10 sets (hemoglobinopathy D55/D56/D57/D58, AIHA D59.x, TMA), and the
anchor-datetime resolution (REQDATE + REQTIME with BDVSTDATE + BDVSTTIME
fallback, flagged via ``anchor_imputed``). Age-based pediatric exclusion
is upstream of this module (IT pre-filter ``age > 15``); see
``docs/ingest-mapping.md``.

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
    AIHA_PREFIX,
    ELIGIBLE_STATUS,
    HEMOGLOBINOPATHY_PREFIXES,
    OBSTETRIC_PREFIX,
    RBC_PRODUCTS,
    TMA_PREFIXES,
    check_aiha,
    check_an_scoped,
    check_cancelled,
    check_hemoglobinopathy,
    check_obstetric,
    check_rbc_product,
    check_request_type,
    check_status,
    check_tma,
    is_rbc_product,
    rbc_products_in,
)

__all__ = [
    "AIHA_PREFIX",
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
    "OBSTETRIC_PREFIX",
    "RBCProduct",
    "RBC_PRODUCTS",
    "TMA_PREFIXES",
    "UnrecoverableAnchorError",
    "build_audit_id",
    "build_audit_orders",
    "check_aiha",
    "check_an_scoped",
    "check_cancelled",
    "check_hemoglobinopathy",
    "check_obstetric",
    "check_rbc_product",
    "check_request_type",
    "check_status",
    "check_tma",
    "is_rbc_product",
    "rbc_products_in",
    "resolve_anchor",
]
