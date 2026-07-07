"""bba.audit_store — append-only Parquet + transactional ordering.

See issue #19 for acceptance criteria. Implementation Decisions §10 in the PRD
defines the split-storage contract, transactional-ordering invariant, run-level
idempotency, daily snapshot views, and 90-day cold-storage policy for Opus
extended-thinking blocks.

This module is the persistence dependency for #20 (eval_harness), #24
(audit_pipeline), #25 (review_actions), #26 (dashboard), #28 (report_generator).
"""

from bba.audit_store.cold_storage import migrate_cold_storage
from bba.audit_store.exceptions import TransactionalOrderingError
from bba.audit_store.models import (
    AuditRow,
    AuditStoreConfig,
    Classification,
    ColdStorageReport,
    Component,
    LlmCall,
    ReconciliationReport,
    WriteResult,
)
from bba.audit_store.snapshot import SnapshotView
from bba.audit_store.store import AuditStore

__all__ = [
    "AuditRow",
    "AuditStore",
    "AuditStoreConfig",
    "Classification",
    "ColdStorageReport",
    "Component",
    "LlmCall",
    "ReconciliationReport",
    "SnapshotView",
    "TransactionalOrderingError",
    "WriteResult",
    "migrate_cold_storage",
]
