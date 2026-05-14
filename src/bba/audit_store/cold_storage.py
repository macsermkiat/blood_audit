"""Cold-storage migration policy for Opus extended-thinking blocks.

PRD §10: Opus extended-thinking blocks are bulky and rarely re-read after the
classification is committed. After 90 days they move to cold storage:

* the ``LlmCall.extended_thinking_blocks`` field is rewritten to ``None``;
* the original blob is spilled to ``<root>/cold_storage/<call_id>.json`` (a
  local file in tests; S3 in production via a future adapter);
* the ``LlmCall.cold_storage_uri`` field on the rewritten row points to it.

The migration is content-preserving: the original bytes are recoverable from
``cold_storage_uri`` and the new row is still byte-identical to the old except
for the two affected fields.
"""

from __future__ import annotations

from datetime import datetime

from bba.audit_store.models import ColdStorageReport
from bba.audit_store.store import AuditStore


def migrate_cold_storage(store: AuditStore, older_than: datetime) -> ColdStorageReport:
    """Spill ``extended_thinking_blocks`` to cold storage for every ``LlmCall``
    with ``request_timestamp < older_than``.

    Returns the call_ids touched and the total bytes moved. The migration is
    idempotent: calling twice with the same cutoff is a no-op on the second run.
    """
    raise NotImplementedError
