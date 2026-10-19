"""Atomic compare-and-append for immutable, evidence-bound review decisions."""

from __future__ import annotations

import json

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from bba.review_actions.exceptions import ReviewConflictError
from bba.review_actions.models import ReviewAction, ReviewActionInput


_COLUMNS = (
    "action_id, audit_id, reviewer_id, action, override_reason, note, created_at, "
    "run_id, code_version, row_hash, expected_revision, submission_id, "
    "override_classification"
)


def append_decision(pool: ConnectionPool, action: ReviewActionInput) -> ReviewAction:
    """Serialize contenders per row and submission, then compare the revision."""
    identity = (action.audit_id, action.run_id, action.code_version, action.row_hash)
    identity_key = json.dumps(identity, separators=(",", ":"))
    with pool.connection() as conn:
        with conn.transaction():
            with conn.cursor(row_factory=dict_row) as cur:
                # A submission can be retried against a different row by mistake.
                # Lock it first so unique-key races yield a domain conflict.
                for key in (f"review-submission:{action.submission_id}", identity_key):
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (key,)
                    )
                cur.execute(
                    f"SELECT {_COLUMNS} FROM review_actions WHERE submission_id = %s",
                    (action.submission_id,),
                )
                prior = cur.fetchone()
                if prior is not None:
                    persisted = ReviewAction.model_validate(prior)
                    if all(
                        getattr(persisted, key) == value
                        for key, value in action.model_dump().items()
                    ):
                        return persisted
                    raise ReviewConflictError(
                        "submission ID already belongs to a different decision"
                    )
                cur.execute(
                    "SELECT action_id FROM review_actions WHERE audit_id = %s AND run_id = %s "
                    "AND code_version = %s AND row_hash = %s ORDER BY action_id DESC LIMIT 1",
                    identity,
                )
                latest = cur.fetchone()
                revision = latest["action_id"] if latest else 0
                if revision != action.expected_revision:
                    raise ReviewConflictError(
                        "another reviewer changed this decision; reload the case"
                    )
                cur.execute(
                    "INSERT INTO review_actions (audit_id, reviewer_id, action, override_reason, note, "
                    "run_id, code_version, row_hash, expected_revision, submission_id, override_classification) "
                    f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING {_COLUMNS}",
                    (
                        action.audit_id,
                        action.reviewer_id,
                        action.action,
                        action.override_reason,
                        action.note,
                        action.run_id,
                        action.code_version,
                        action.row_hash,
                        action.expected_revision,
                        action.submission_id,
                        action.override_classification,
                    ),
                )
                row = cur.fetchone()
                assert row is not None
                return ReviewAction.model_validate(row)
