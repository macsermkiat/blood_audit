"""initial review_actions and phi_access_log schema

Revision ID: cfb3b5460004
Revises:
Create Date: 2026-05-15 12:38:31.312782

PRD §16 — two append-only tables for reviewer decisions and PHI-access events.

Defense in depth against silent mutation of the audit trail:

1. The application role is GRANTed only ``INSERT`` and ``SELECT``; ``UPDATE``
   and ``DELETE`` are REVOKEd. A misconfigured client running as this role
   simply cannot mutate.
2. A trigger guard on both tables raises an exception for any ``UPDATE``,
   ``DELETE``, or ``TRUNCATE`` regardless of the role's grants. The trigger
   is the load-bearing defense — REVOKE alone is bypassable by a superuser
   misconfiguration.

The application role name (``review_actions_app``) is created here so the
revocation surface is precise; the GREEN-phase store connects as this role.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op


revision: str = "cfb3b5460004"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_ACTION_KIND_VALUES = "('agree', 'override', 'escalate', 'use_as_few_shot_candidate')"
"""Allowed values for review_actions.action; mirrored in
``bba.review_actions.models.ACTION_KINDS``. A bump here requires a bump there."""


def upgrade() -> None:
    """Create the two append-only tables, install trigger guards, and revoke
    UPDATE/DELETE from the application role.

    Steps run in this order so a failure mid-migration leaves the DB at the
    last good state (alembic's transactional-DDL default on Postgres).
    """
    # ----- Application role ----------------------------------------------
    # CREATE ROLE is idempotent via the DO block so re-running the migration
    # against a partially-rolled-back DB doesn't trip on "role exists".
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_roles WHERE rolname = 'review_actions_app'
            ) THEN
                CREATE ROLE review_actions_app;
            END IF;
        END$$;
        """
    )

    # ----- review_actions table -------------------------------------------
    op.execute(
        f"""
        CREATE TABLE review_actions (
            action_id BIGSERIAL PRIMARY KEY,
            audit_id TEXT NOT NULL,
            reviewer_id TEXT NOT NULL,
            action TEXT NOT NULL CHECK (action IN {_ACTION_KIND_VALUES}),
            override_reason TEXT,
            note TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            CONSTRAINT review_actions_override_requires_reason
                CHECK (
                    (action = 'override' AND override_reason IS NOT NULL)
                    OR (action <> 'override' AND override_reason IS NULL)
                )
        );
        """
    )

    op.execute(
        "CREATE INDEX review_actions_audit_id_idx "
        "ON review_actions (audit_id, created_at, action_id);"
    )
    op.execute(
        "CREATE INDEX review_actions_reviewer_id_idx "
        "ON review_actions (reviewer_id, created_at, action_id);"
    )

    # ----- phi_access_log table -------------------------------------------
    op.execute(
        """
        CREATE TABLE phi_access_log (
            access_id BIGSERIAL PRIMARY KEY,
            reviewer_id TEXT NOT NULL,
            audit_id TEXT NOT NULL,
            hn_hash TEXT NOT NULL,
            an_hash TEXT NOT NULL,
            break_glass_justification TEXT,
            accessed_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )

    op.execute(
        "CREATE INDEX phi_access_log_reviewer_id_idx "
        "ON phi_access_log (reviewer_id, accessed_at, access_id);"
    )
    op.execute(
        "CREATE INDEX phi_access_log_audit_id_idx "
        "ON phi_access_log (audit_id, accessed_at, access_id);"
    )

    # ----- Trigger guard function -----------------------------------------
    # Raises with SQLSTATE 'P0001' so the Python store detects the violation
    # by SQLSTATE rather than by parsing the message string. The user-facing
    # message is deliberately terse: TG_OP / TG_TABLE_NAME are written to
    # the Postgres server log by the standard RAISE EXCEPTION machinery, so
    # an operator debugging a violation still has the full context — but a
    # bad actor probing the schema does not get free schema disclosure
    # from the client-side error.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION review_actions_block_mutation()
        RETURNS TRIGGER
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RAISE EXCEPTION 'append_only_violation'
                USING ERRCODE = 'P0001';
        END;
        $$;
        """
    )

    # Attach the guard to BOTH tables for UPDATE, DELETE, TRUNCATE.
    for table in ("review_actions", "phi_access_log"):
        op.execute(
            f"""
            CREATE TRIGGER {table}_block_update
            BEFORE UPDATE ON {table}
            FOR EACH ROW
            EXECUTE FUNCTION review_actions_block_mutation();
            """
        )
        op.execute(
            f"""
            CREATE TRIGGER {table}_block_delete
            BEFORE DELETE ON {table}
            FOR EACH ROW
            EXECUTE FUNCTION review_actions_block_mutation();
            """
        )
        op.execute(
            f"""
            CREATE TRIGGER {table}_block_truncate
            BEFORE TRUNCATE ON {table}
            FOR EACH STATEMENT
            EXECUTE FUNCTION review_actions_block_mutation();
            """
        )

    # ----- Grant minimal privileges to application role -------------------
    op.execute(
        "GRANT INSERT, SELECT ON review_actions TO review_actions_app;"
    )
    op.execute(
        "GRANT INSERT, SELECT ON phi_access_log TO review_actions_app;"
    )
    op.execute(
        "GRANT USAGE, SELECT ON SEQUENCE review_actions_action_id_seq "
        "TO review_actions_app;"
    )
    op.execute(
        "GRANT USAGE, SELECT ON SEQUENCE phi_access_log_access_id_seq "
        "TO review_actions_app;"
    )

    # ----- Explicit revocation of UPDATE/DELETE ---------------------------
    # No-op for the role since INSERT/SELECT were the only grants, but
    # explicit so a future migration that re-grants ALL privileges does not
    # silently re-enable mutation without removing the trigger guard.
    op.execute(
        "REVOKE UPDATE, DELETE, TRUNCATE "
        "ON review_actions FROM review_actions_app;"
    )
    op.execute(
        "REVOKE UPDATE, DELETE, TRUNCATE "
        "ON phi_access_log FROM review_actions_app;"
    )
    op.execute(
        "REVOKE UPDATE, DELETE, TRUNCATE ON review_actions FROM PUBLIC;"
    )
    op.execute(
        "REVOKE UPDATE, DELETE, TRUNCATE ON phi_access_log FROM PUBLIC;"
    )

    # ----- Grant SELECT on alembic_version --------------------------------
    # The store's startup integrity check reads alembic_version to compare
    # the live DB's revision against the on-disk head. Without this grant,
    # the app role gets a permission-denied. SELECT-only — only the migrator
    # (running as a privileged role) updates the version row.
    op.execute(
        "GRANT SELECT ON alembic_version TO review_actions_app;"
    )


def downgrade() -> None:
    """Reverse :func:`upgrade` for development; production never downgrades.

    Drops triggers, function, indexes, tables, and the application role in
    dependency order. The role drop is guarded by an existence check so a
    re-run on a partially-rolled-back DB doesn't trip.
    """
    for table in ("review_actions", "phi_access_log"):
        op.execute(f"DROP TRIGGER IF EXISTS {table}_block_update ON {table};")
        op.execute(f"DROP TRIGGER IF EXISTS {table}_block_delete ON {table};")
        op.execute(
            f"DROP TRIGGER IF EXISTS {table}_block_truncate ON {table};"
        )

    op.execute("DROP FUNCTION IF EXISTS review_actions_block_mutation();")

    op.execute("DROP TABLE IF EXISTS phi_access_log;")
    op.execute("DROP TABLE IF EXISTS review_actions;")

    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_roles WHERE rolname = 'review_actions_app'
            ) THEN
                DROP ROLE review_actions_app;
            END IF;
        END$$;
        """
    )
