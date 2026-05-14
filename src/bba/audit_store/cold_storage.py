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

import json
from datetime import datetime

from bba.audit_store.models import ColdStorageReport, _deep_thaw
from bba.audit_store.store import AuditStore, _hash_id


def migrate_cold_storage(store: AuditStore, older_than: datetime) -> ColdStorageReport:
    """Spill ``extended_thinking_blocks`` to cold storage for every ``LlmCall``
    with ``request_timestamp < older_than``.

    Returns the call_ids touched and the total bytes moved. The migration is
    idempotent: a call already migrated (``extended_thinking_blocks is None``)
    is skipped on the second run.
    """
    store.cold_storage_dir.mkdir(parents=True, exist_ok=True)

    moved: list[str] = []
    bytes_moved = 0

    # Iterate via _iter_call_records so each call comes paired with the slug
    # it was originally persisted under. Rewriting at the original slug keeps
    # the call file at its existing path (no orphaned old file, no leaked
    # extra file under the migrator's slug — see Codex P1 round 4).
    for call, original_slug in store._iter_call_records():
        if call.request_timestamp >= older_than:
            continue
        if call.extended_thinking_blocks is None:
            continue  # already migrated — idempotency invariant

        # Blob filename uses the same hashed-id + slug scheme as the call
        # parquet so call_id reuse across code_versions cannot collide on
        # the blob path (Codex P2 round 5) and underscore-containing
        # call_ids cannot collide via the `_` separator (Codex P1 round 8).
        cold_path = (
            store.cold_storage_dir / f"{_hash_id(call.call_id)}_{original_slug}.json"
        )
        # _deep_thaw recursively unwraps MappingProxyType + frozen tuples
        # so json.dumps sees plain dict/list. The field is frozen for
        # in-memory immutability (PRD §"persisted immutably") but the
        # cold blob is a passive byte record.
        blob = json.dumps(
            _deep_thaw(call.extended_thinking_blocks), ensure_ascii=False
        ).encode("utf-8")
        cold_path.write_bytes(blob)
        bytes_moved += len(blob)

        migrated = call.model_copy(
            update={
                "extended_thinking_blocks": None,
                "cold_storage_uri": str(cold_path),
            }
        )
        store._persist_call_record(migrated, code_version_slug=original_slug)
        moved.append(call.call_id)

    return ColdStorageReport(
        moved_call_ids=tuple(moved),
        bytes_moved=bytes_moved,
    )
