"""audit_pipeline batch_runs schema

Revision ID: a1c2e3f4b5d6
Revises: cfb3b5460004
Create Date: 2026-05-16 14:30:00.000000

Issue #24 — row-level checkpointing table for the audit pipeline.

Schema mirrors the :class:`bba.audit_pipeline.BatchRun` pydantic model.
Unlike ``review_actions`` / ``phi_access_log`` (append-only with trigger
guards), ``batch_runs`` is *mutable* state by design — the pipeline
transitions rows through PENDING → SUBMITTED → PARTIAL → COMPLETE | FAILED
via :func:`bba.audit_pipeline.state_machine.transition`. The state machine
is enforced application-side; the DB only stores the latest state.

The migration ALSO installs the (audit_id, run_id, code_version)
uniqueness invariant that the audit_store relies on for idempotency —
two batch_runs rows for the same triple are a contract violation
(would let two concurrent pipelines double-write audit_results).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "a1c2e3f4b5d6"
down_revision: str | Sequence[str] | None = "cfb3b5460004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_STATE_VALUES = "('pending', 'submitted', 'partial', 'complete', 'failed')"
"""Allowed values for batch_runs.state; mirrored in
:class:`bba.audit_pipeline.BatchRunState`. A bump here requires a bump
there and a state-machine table update."""


def upgrade() -> None:
    """Create the batch_runs table + the audit_pipeline_app role."""
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_roles WHERE rolname = 'audit_pipeline_app'
            ) THEN
                CREATE ROLE audit_pipeline_app;
            END IF;
        END$$;
        """
    )

    op.execute(
        f"""
        CREATE TABLE batch_runs (
            batch_id TEXT PRIMARY KEY,
            state TEXT NOT NULL CHECK (state IN {_STATE_VALUES}),
            run_id TEXT NOT NULL,
            code_version TEXT NOT NULL,
            audit_ids TEXT[] NOT NULL,
            anthropic_batch_id TEXT,
            submitted_at TIMESTAMPTZ,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            error_message TEXT,
            CONSTRAINT batch_runs_audit_ids_nonempty
                CHECK (cardinality(audit_ids) > 0),
            CONSTRAINT batch_runs_pending_no_anthropic
                CHECK (state <> 'pending' OR anthropic_batch_id IS NULL),
            CONSTRAINT batch_runs_post_pending_has_anthropic
                CHECK (
                    state = 'pending'
                    OR state = 'failed'
                    OR anthropic_batch_id IS NOT NULL
                ),
            CONSTRAINT batch_runs_failed_has_error
                CHECK (state <> 'failed' OR error_message IS NOT NULL)
        );
        """
    )

    # Index for resume-on-startup's list_by_state scan over the
    # non-terminal states (PENDING, SUBMITTED, PARTIAL). Without this
    # index the scan is a sequential read of every settled batch_run.
    op.execute(
        """
        CREATE INDEX batch_runs_state_idx
            ON batch_runs (state)
            WHERE state IN ('pending', 'submitted', 'partial');
        """
    )

    op.execute(
        """
        GRANT SELECT, INSERT, UPDATE ON batch_runs TO audit_pipeline_app;
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS batch_runs;")
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_roles WHERE rolname = 'audit_pipeline_app'
            ) THEN
                DROP ROLE audit_pipeline_app;
            END IF;
        END$$;
        """
    )
