"""Bind append-only reviewer decisions to exact audit evidence and revisions."""

from collections.abc import Sequence

from alembic import op


revision: str = "d2e3f4a5b6c7"
down_revision: str | Sequence[str] | None = "a1c2e3f4b5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE review_actions
          ADD COLUMN run_id TEXT,
          ADD COLUMN code_version TEXT,
          ADD COLUMN row_hash TEXT,
          ADD COLUMN expected_revision BIGINT,
          ADD COLUMN submission_id TEXT,
          ADD COLUMN override_classification TEXT,
          ADD CONSTRAINT review_actions_bound_identity CHECK (
            (run_id IS NULL AND code_version IS NULL AND row_hash IS NULL
             AND expected_revision IS NULL AND submission_id IS NULL)
            OR
            (run_id IS NOT NULL AND code_version IS NOT NULL AND row_hash IS NOT NULL AND expected_revision IS NOT NULL AND row_hash ~ '^[0-9a-f]{64}$'
             AND expected_revision >= 0 AND submission_id IS NOT NULL)
          ),
          ADD CONSTRAINT review_actions_bound_override CHECK (
            run_id IS NULL OR
            (action = 'override' AND override_classification IS NOT NULL
             AND length(trim(override_reason)) > 0) OR
            (action <> 'override' AND override_classification IS NULL)
          );
    """)
    classifications = "'APPROPRIATE','INAPPROPRIATE','NEEDS_REVIEW','INSUFFICIENT_EVIDENCE','POTENTIALLY_INAPPROPRIATE','PREOP_RESERVATION_UNCONFIRMED','PREOP_OVER_RESERVATION','RETURNED_NOT_TRANSFUSED','PERIOP_TRANSFUSION_EXEMPT'"
    op.execute(
        f"ALTER TABLE review_actions ADD CONSTRAINT review_actions_classification CHECK (override_classification IN ({classifications}));"
    )
    op.execute(
        "CREATE UNIQUE INDEX review_actions_submission_id_idx ON review_actions (submission_id) WHERE submission_id IS NOT NULL;"
    )
    op.execute(
        "CREATE UNIQUE INDEX review_actions_row_revision_idx ON review_actions (audit_id, run_id, code_version, row_hash, expected_revision) WHERE run_id IS NOT NULL;"
    )
    op.execute(
        "CREATE INDEX review_actions_row_history_idx ON review_actions (audit_id, run_id, code_version, row_hash, action_id);"
    )


def downgrade() -> None:
    raise RuntimeError(
        "review decisions are append-only; archive before schema rollback"
    )
