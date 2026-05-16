"""RED-phase failing tests for issue #25 (bba.review_actions).

Each ``class`` maps to one acceptance criterion in the issue body.

* AC ① "Implementation in ``src/bba/review_actions/``"
    → :class:`TestModulePublicSurface`
* AC ② "Append-only invariant enforced at DB level
    (revoke UPDATE/DELETE; trigger guard)"
    → :class:`TestAppendOnlyInvariantDBLevel`,
      :class:`TestAppendOnlyApiLevel`
* AC ③ "PHI-access log completeness"
    → :class:`TestPhiAccessLogCompleteness`,
      :class:`TestPhiAccessLogContent`
* AC ④ "Concurrent-write test: 5 simulated reviewers, no lock contention"
    → :class:`TestConcurrentWrites`
* AC ⑤ "Schema migration scripts in ``migrations/``"
    → :class:`TestMigrationScripts`
* AC ⑥ "Coverage ≥ 70%; ruff + mypy clean"
    → verified by the build (not a behavioral test)

Tests assert contracts (the WHY), not implementation choices. In this RED
scaffold, behavioral tests fail with :class:`NotImplementedError`; the model
construction tests in :class:`TestModelContracts` pass because the frozen
Pydantic models are part of the interface itself (regression guards on the
declared contract, not behavioral assertions).
"""

from __future__ import annotations

import concurrent.futures
import re
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest
from pydantic import ValidationError

from bba.review_actions import (
    ACTION_KINDS,
    AppendOnlyViolationError,
    MIGRATIONS_ROOT_DEFAULT,
    MigrationStateError,
    PhiAccessInput,
    PhiAccessLog,
    ReviewAction,
    ReviewActionInput,
    ReviewActionsConfig,
    ReviewActionsStore,
    apply_migrations,
    current_revision,
    head_revision,
)


# =============================================================================
# Fixtures
#
# The Postgres-required fixture spins up a real container via
# testcontainers-python. RED-phase store methods raise NotImplementedError
# before any DB call, so most tests fail without ever needing the container.
# DB-level tests (trigger-guards, concurrent-writes) genuinely require
# Postgres; they fail in RED on NotImplementedError too, and will exercise
# the live container in GREEN.
# =============================================================================


REPO_ROOT = Path(__file__).resolve().parents[2]
"""Absolute path to the repo root (``tests/unit/test_review_actions.py`` is
two ``parents`` up from the file). Tests that look for ``migrations/`` use
this to be independent of pytest's invocation cwd."""


@pytest.fixture
def config(tmp_path: Path) -> ReviewActionsConfig:
    """A ReviewActionsConfig pointing at a placeholder DSN.

    RED-phase store methods raise NotImplementedError before touching the DB,
    so the DSN does not need to be real. GREEN-phase will replace this with
    :func:`postgres_config` for tests that genuinely connect.
    """
    return ReviewActionsConfig(
        dsn=f"postgresql://test:test@localhost:5432/test_{tmp_path.name}",
    )


@pytest.fixture(scope="session")
def _postgres_container() -> Iterator[object]:
    """Session-scoped Postgres container via testcontainers-python.

    Imported lazily so test-collection doesn't fail when testcontainers is
    not yet installed (GREEN-phase will pin it as a dev dep). RED tests that
    consume this fixture will ERROR with ImportError until then — which is
    the desired failure mode.
    """
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as pg:
        yield pg


_TEST_ROLES_TO_DROP: tuple[str, ...] = (
    "review_actions_app",
    "writer_role",
    "owner_role",
    "table_owner",
    "shady_member",
    "noinherit_member",
)
"""Roles tests create that must be dropped between runs.

Schema-CASCADE drop removes tables and grants attached to them but leaves
the roles themselves; without explicit cleanup, a second test that tries
to ``CREATE ROLE`` with the same name would fail."""


def _reset_database(config: ReviewActionsConfig) -> None:
    """Drop and recreate the ``public`` schema and all test-created roles.

    The session-scoped Postgres container is shared; without a reset, prior
    tests' rows / migration head / custom roles leak into the next test.
    ``DROP SCHEMA ... CASCADE`` removes the migration's tables, sequences,
    triggers, and the alembic_version table in one shot; an explicit role
    sweep handles roles (which live outside any schema).
    """
    with psycopg.connect(config.dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
            cur.execute("CREATE SCHEMA public")
            cur.execute("GRANT ALL ON SCHEMA public TO PUBLIC")
            for role in _TEST_ROLES_TO_DROP:
                # REASSIGN OWNED + DROP OWNED handle objects still owned by
                # the role outside the public schema; both are idempotent.
                cur.execute(
                    f"DO $$ BEGIN IF EXISTS ("
                    f"SELECT 1 FROM pg_roles WHERE rolname = '{role}'"
                    f") THEN "
                    f" EXECUTE 'REASSIGN OWNED BY {role} TO CURRENT_USER'; "
                    f" EXECUTE 'DROP OWNED BY {role}'; "
                    f" EXECUTE 'DROP ROLE {role}'; "
                    f"END IF; END $$;"
                )


def _swap_dsn_credentials(dsn: str, *, user: str, password: str) -> str:
    """Return a copy of ``dsn`` with the ``user:password`` portion replaced.

    Used to derive an unprivileged-role DSN from the superuser DSN that
    testcontainers emits.
    """
    parsed = urlparse(dsn)
    host = parsed.hostname or "localhost"
    port_part = f":{parsed.port}" if parsed.port else ""
    new_netloc = f"{user}:{password}@{host}{port_part}"
    return urlunparse(parsed._replace(netloc=new_netloc))


@pytest.fixture
def postgres_config(_postgres_container: object) -> ReviewActionsConfig:
    """Superuser-role ReviewActionsConfig pointing at a freshly-reset DB.

    The reset runs per-test so each test sees a clean schema (no rows from
    prior tests, no alembic_version row, no tables). The reset is fast
    (sub-100ms) vs spinning up a new container per test (several seconds).

    This config connects as the container's default user, which is a
    superuser. Most data-mutating tests use :func:`app_config` instead,
    which exercises the production privilege model.
    """
    dsn = _postgres_container.get_connection_url()  # type: ignore[attr-defined]
    config = ReviewActionsConfig(dsn=dsn)
    _reset_database(config)
    return config


@pytest.fixture
def app_config(
    postgres_config: ReviewActionsConfig,
) -> ReviewActionsConfig:
    """Unprivileged-role ReviewActionsConfig against a migrated DB.

    Migrates the freshly-reset DB using the superuser config, sets a
    password on the ``review_actions_app`` role, and returns a config that
    connects as that role. This mirrors the production deployment model:
    migrations run as a privileged operator role, the app process connects
    as a least-privilege INSERT/SELECT-only role.
    """
    apply_migrations(postgres_config)
    with psycopg.connect(postgres_config.dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "ALTER ROLE review_actions_app WITH LOGIN PASSWORD 'app_test_pw'"
            )
    app_dsn = _swap_dsn_credentials(
        postgres_config.dsn, user="review_actions_app", password="app_test_pw"
    )
    return ReviewActionsConfig(dsn=app_dsn)


@pytest.fixture
def migrated_store(app_config: ReviewActionsConfig) -> Iterator[ReviewActionsStore]:
    """A live store against a fully-migrated Postgres database, connecting
    as the unprivileged ``review_actions_app`` role.

    Used by tests that exercise the live append-only invariant / concurrent
    writes / read path / context-manager API.
    """
    with ReviewActionsStore(
        app_config, migrations_root=REPO_ROOT / "migrations"
    ) as store:
        yield store


# =============================================================================
# Builders
# =============================================================================


def _action_input(
    *,
    audit_id: str = "audit-001",
    reviewer_id: str = "reviewer-a",
    action: str = "agree",
    override_reason: str | None = None,
    note: str | None = None,
) -> ReviewActionInput:
    kwargs: dict[str, object] = {
        "audit_id": audit_id,
        "reviewer_id": reviewer_id,
        "action": action,
    }
    if override_reason is not None:
        kwargs["override_reason"] = override_reason
    if note is not None:
        kwargs["note"] = note
    return ReviewActionInput.model_validate(kwargs)


def _phi_access_input(
    *,
    reviewer_id: str = "reviewer-a",
    audit_id: str = "audit-001",
    hn_hash: str = "hn-sha256-aaa",
    an_hash: str = "an-sha256-bbb",
    break_glass_justification: str | None = None,
) -> PhiAccessInput:
    kwargs: dict[str, object] = {
        "reviewer_id": reviewer_id,
        "audit_id": audit_id,
        "hn_hash": hn_hash,
        "an_hash": an_hash,
    }
    if break_glass_justification is not None:
        kwargs["break_glass_justification"] = break_glass_justification
    return PhiAccessInput.model_validate(kwargs)


# =============================================================================
# AC ① — Implementation in ``src/bba/review_actions/`` (public surface)
#
# WHY: downstream consumers (#26 dashboard) import a stable public surface.
# A regression that drops or renames a name is a breaking change for them.
# =============================================================================


class TestModulePublicSurface:
    """The module exposes the expected names from its package root."""

    def test_models_are_importable(self) -> None:
        import bba.review_actions as m

        assert m.ReviewAction is not None
        assert m.ReviewActionInput is not None
        assert m.PhiAccessLog is not None
        assert m.PhiAccessInput is not None
        assert m.ReviewActionsConfig is not None

    def test_store_is_importable(self) -> None:
        from bba.review_actions import ReviewActionsStore

        assert ReviewActionsStore.__module__ == "bba.review_actions.store"

    def test_exceptions_are_importable(self) -> None:
        from bba.review_actions import (
            AppendOnlyViolationError,
            MigrationStateError,
            OverrideRequiresReasonError,
        )

        # Each exception must subclass Exception (catchable as such).
        assert issubclass(AppendOnlyViolationError, Exception)
        assert issubclass(MigrationStateError, Exception)
        assert issubclass(OverrideRequiresReasonError, ValueError)

    def test_action_kinds_match_prd_section_16(self) -> None:
        """The four action kinds from PRD §16 are present and complete."""
        assert set(ACTION_KINDS) == {
            "agree",
            "override",
            "escalate",
            "use_as_few_shot_candidate",
        }


# =============================================================================
# Model contracts (regression guards on the declared interface)
#
# These tests pass in RED because they assert structural properties of the
# scaffolded models — frozen-ness, tz-aware enforcement, override-reason
# requirement. They guard against regressions to the public contract.
# =============================================================================


class TestModelContracts:
    """Pydantic-model invariants that the interface itself promises."""

    def test_review_action_input_is_frozen(self) -> None:
        action = _action_input()
        with pytest.raises(ValidationError):
            action.action = "override"  # type: ignore[misc]

    def test_review_action_is_frozen(self) -> None:
        ra = ReviewAction(
            action_id=1,
            audit_id="audit-001",
            reviewer_id="reviewer-a",
            action="agree",
            override_reason=None,
            note=None,
            created_at=datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
        )
        with pytest.raises(ValidationError):
            ra.action_id = 99  # type: ignore[misc]

    def test_phi_access_log_is_frozen(self) -> None:
        log = PhiAccessLog(
            access_id=1,
            reviewer_id="reviewer-a",
            audit_id="audit-001",
            hn_hash="x",
            an_hash="y",
            break_glass_justification=None,
            accessed_at=datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
        )
        with pytest.raises(ValidationError):
            log.access_id = 99  # type: ignore[misc]

    def test_action_kind_rejects_unknown_value(self) -> None:
        with pytest.raises(ValidationError):
            _action_input(action="bogus-kind")

    def test_override_requires_reason(self) -> None:
        """``action='override'`` without ``override_reason`` is rejected."""
        with pytest.raises(ValidationError):
            _action_input(action="override")

    def test_override_accepts_reason(self) -> None:
        """``action='override'`` WITH ``override_reason`` constructs cleanly."""
        action = _action_input(
            action="override",
            override_reason="Hb threshold met but symptomatic anemia overrules",
        )
        assert action.override_reason is not None
        assert action.action == "override"

    def test_override_reason_only_valid_for_override(self) -> None:
        """Setting ``override_reason`` with ``action != 'override'`` is rejected.

        WHY: a stale ``override_reason`` on an ``agree`` row would mislead the
        dashboard's rationale renderer. Force the constraint at the input
        boundary so the store never has to second-guess.
        """
        with pytest.raises(ValidationError):
            _action_input(action="agree", override_reason="stale text from earlier")

    def test_override_reason_rejects_empty_string(self) -> None:
        """Empty / whitespace-only ``override_reason`` is rejected even when
        ``action='override'`` — an empty reason defeats the audit trail."""
        with pytest.raises(ValidationError):
            _action_input(action="override", override_reason="")
        with pytest.raises(ValidationError):
            _action_input(action="override", override_reason="   ")

    def test_naive_datetime_rejected_on_review_action(self) -> None:
        with pytest.raises(ValidationError):
            ReviewAction(
                action_id=1,
                audit_id="audit-001",
                reviewer_id="reviewer-a",
                action="agree",
                override_reason=None,
                note=None,
                created_at=datetime(2026, 5, 1, 12, 0, 0),  # naive
            )

    def test_naive_datetime_rejected_on_phi_access_log(self) -> None:
        with pytest.raises(ValidationError):
            PhiAccessLog(
                access_id=1,
                reviewer_id="reviewer-a",
                audit_id="audit-001",
                hn_hash="x",
                an_hash="y",
                break_glass_justification=None,
                accessed_at=datetime(2026, 5, 1, 12, 0, 0),  # naive
            )

    def test_non_utc_datetime_normalized_to_utc(self) -> None:
        from datetime import timedelta, timezone

        bangkok = timezone(timedelta(hours=7))
        ra = ReviewAction(
            action_id=1,
            audit_id="audit-001",
            reviewer_id="reviewer-a",
            action="agree",
            override_reason=None,
            note=None,
            created_at=datetime(2026, 5, 1, 19, 0, 0, tzinfo=bangkok),
        )
        assert (
            ra.created_at.utcoffset()
            == datetime(2026, 5, 1, 0, 0, 0, tzinfo=UTC).utcoffset()
        )
        # 19:00 +07 == 12:00 UTC
        assert ra.created_at.hour == 12

    def test_safe_id_rejects_path_traversal(self) -> None:
        with pytest.raises(ValidationError):
            _action_input(audit_id="../etc/passwd")
        with pytest.raises(ValidationError):
            _action_input(audit_id="..")
        with pytest.raises(ValidationError):
            _action_input(reviewer_id=".")

    def test_safe_id_rejects_special_chars(self) -> None:
        with pytest.raises(ValidationError):
            _action_input(reviewer_id="alice;DROP TABLE--")
        with pytest.raises(ValidationError):
            _action_input(audit_id="has space")

    def test_safe_id_rejects_empty(self) -> None:
        with pytest.raises(ValidationError):
            _action_input(audit_id="")


# =============================================================================
# AC ② — Append-only invariant
#
# WHY: a reviewer who can rewrite an earlier decision can launder the audit
# trail. The DB enforces this with REVOKE + trigger guard (defense in depth);
# the Python layer translates the SQLSTATE so callers see a typed exception.
# =============================================================================


class TestAppendOnlyApiLevel:
    """The public API offers no mutate or delete methods.

    The DB-level invariant is the load-bearing defense; the Python layer is
    a regression guard that the API surface itself never grew an
    ``update_action`` / ``delete_action`` method.
    """

    def test_no_update_method_on_store(self) -> None:
        assert not hasattr(ReviewActionsStore, "update_action")
        assert not hasattr(ReviewActionsStore, "update_phi_access")

    def test_no_delete_method_on_store(self) -> None:
        assert not hasattr(ReviewActionsStore, "delete_action")
        assert not hasattr(ReviewActionsStore, "delete_phi_access")

    def test_review_action_input_has_no_action_id(self) -> None:
        """Callers cannot pass a server-side ``action_id`` — that would let a
        bad actor "claim" a specific ID and effectively overwrite a row by
        racing inserts. The PK is DB-assigned (bigserial)."""
        assert "action_id" not in ReviewActionInput.model_fields

    def test_phi_access_input_has_no_access_id(self) -> None:
        assert "access_id" not in PhiAccessInput.model_fields


class TestAppendOnlyInvariantDBLevel:
    """The live DB rejects UPDATE/DELETE on the append-only tables.

    These tests genuinely require Postgres. They fail in RED at
    ``NotImplementedError`` (store/migrator unimplemented) and at GREEN
    exercise the trigger guards installed by the migration.
    """

    def test_update_review_actions_revoked(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        """Direct UPDATE on ``review_actions`` raises AppendOnlyViolationError.

        WHY: REVOKE UPDATE on the role used by the app must be in place,
        AND the trigger guard must raise if the privilege is ever
        re-granted (defense in depth).
        """
        written = migrated_store.record_action(_action_input())

        # The store does not expose a low-level UPDATE; reach into the pool
        # for a privileged connection to attempt the violation. Implementation
        # detail: the migrator returns a "raw" handle for tests; the public
        # surface never offers it.
        with pytest.raises(AppendOnlyViolationError):
            _raw_update_action(migrated_store, written.action_id, "override")

    def test_delete_review_actions_revoked(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        """Direct DELETE on ``review_actions`` raises AppendOnlyViolationError."""
        written = migrated_store.record_action(_action_input())

        with pytest.raises(AppendOnlyViolationError):
            _raw_delete_action(migrated_store, written.action_id)

    def test_update_phi_access_log_revoked(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        """Direct UPDATE on ``phi_access_log`` raises AppendOnlyViolationError."""
        written = migrated_store.record_phi_access(_phi_access_input())

        with pytest.raises(AppendOnlyViolationError):
            _raw_update_phi_access(migrated_store, written.access_id, "new-hn-hash")

    def test_delete_phi_access_log_revoked(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        """Direct DELETE on ``phi_access_log`` raises AppendOnlyViolationError."""
        written = migrated_store.record_phi_access(_phi_access_input())

        with pytest.raises(AppendOnlyViolationError):
            _raw_delete_phi_access(migrated_store, written.access_id)

    def test_truncate_review_actions_revoked(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        """TRUNCATE is a deletion under a different name; equally forbidden."""
        migrated_store.record_action(_action_input())

        with pytest.raises(AppendOnlyViolationError):
            _raw_truncate_table(migrated_store, "review_actions")


# Test helpers — raw-SQL bypass of the store's safe API to verify the
# trigger guards actually fire. Each opens a fresh psycopg connection
# (bypassing the store's pool) so it can issue UPDATE/DELETE/TRUNCATE
# directly. The store's translation layer catches Postgres P0001 from the
# trigger and re-raises :class:`AppendOnlyViolationError`; here we
# replicate that translation at the call site so the tests can be
# written in terms of the typed exception.


def _execute_raw(
    store: ReviewActionsStore, statement: str, params: object = None
) -> None:
    """Open a fresh connection (bypassing the store's pool) and execute
    ``statement``. Translates both the REVOKE-style block
    (``InsufficientPrivilege`` / SQLSTATE 42501) and the trigger-style block
    (``RaiseException`` / SQLSTATE P0001 + ``append_only_violation``
    message) into :class:`AppendOnlyViolationError` so tests can express the
    invariant in domain terms regardless of which defense actually fired.
    """
    try:
        with psycopg.connect(store.config.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(statement, params or ())
    except psycopg.errors.InsufficientPrivilege as exc:
        raise AppendOnlyViolationError(str(exc)) from exc
    except psycopg.errors.RaiseException as exc:
        message = str(exc.diag.message_primary) if exc.diag else str(exc)
        if exc.sqlstate == "P0001" and "append_only_violation" in message:
            raise AppendOnlyViolationError(message) from exc
        raise


def _raw_update_action(
    store: ReviewActionsStore, action_id: int, new_action: str
) -> None:
    _execute_raw(
        store,
        "UPDATE review_actions SET action = %s WHERE action_id = %s",
        (new_action, action_id),
    )


def _raw_delete_action(store: ReviewActionsStore, action_id: int) -> None:
    _execute_raw(
        store,
        "DELETE FROM review_actions WHERE action_id = %s",
        (action_id,),
    )


def _raw_update_phi_access(
    store: ReviewActionsStore, access_id: int, new_hn_hash: str
) -> None:
    _execute_raw(
        store,
        "UPDATE phi_access_log SET hn_hash = %s WHERE access_id = %s",
        (new_hn_hash, access_id),
    )


def _raw_delete_phi_access(store: ReviewActionsStore, access_id: int) -> None:
    _execute_raw(
        store,
        "DELETE FROM phi_access_log WHERE access_id = %s",
        (access_id,),
    )


def _raw_truncate_table(store: ReviewActionsStore, table_name: str) -> None:
    # Table name is from the test code, never from user input.
    assert table_name in {"review_actions", "phi_access_log"}
    _execute_raw(store, f"TRUNCATE TABLE {table_name}")


# =============================================================================
# AC ③ — PHI-access log completeness
#
# WHY: PRD §17 (deid_redactor) keeps PHI out of the LLM prompt; the
# dashboard is the one place un-redacted text is shown to a human. If the
# dashboard can render un-redacted text without writing a log row, the
# regulator-facing audit trail has gaps.
# =============================================================================


class TestPhiAccessLogCompleteness:
    """Every PHI access produces exactly one log row."""

    def test_record_phi_access_writes_a_row(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        access = _phi_access_input()
        migrated_store.record_phi_access(access)

        rows = migrated_store.list_phi_access(reviewer_id=access.reviewer_id)
        assert len(rows) == 1

    def test_record_phi_access_returns_db_assigned_fields(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        access = _phi_access_input()
        log = migrated_store.record_phi_access(access)

        assert log.access_id > 0
        assert log.accessed_at.tzinfo is not None
        assert log.reviewer_id == access.reviewer_id
        assert log.hn_hash == access.hn_hash

    def test_break_glass_justification_persisted(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        access = _phi_access_input(
            break_glass_justification="Code blue, attending requested chart"
        )
        log = migrated_store.record_phi_access(access)

        assert log.break_glass_justification == access.break_glass_justification

    def test_verify_phi_access_completeness_returns_true_after_write(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        access = _phi_access_input(reviewer_id="reviewer-a", audit_id="audit-1")
        migrated_store.record_phi_access(access)

        assert (
            migrated_store.verify_phi_access_completeness(
                reviewer_id="reviewer-a", audit_id="audit-1"
            )
            is True
        )

    def test_verify_phi_access_completeness_false_when_never_accessed(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        # No record_phi_access call yet.
        assert (
            migrated_store.verify_phi_access_completeness(
                reviewer_id="reviewer-never", audit_id="audit-never"
            )
            is False
        )

    def test_multiple_reviewers_same_audit_row_each_log_independently(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        migrated_store.record_phi_access(
            _phi_access_input(reviewer_id="alice", audit_id="audit-1")
        )
        migrated_store.record_phi_access(
            _phi_access_input(reviewer_id="bob", audit_id="audit-1")
        )

        alice_rows = migrated_store.list_phi_access(reviewer_id="alice")
        bob_rows = migrated_store.list_phi_access(reviewer_id="bob")

        assert len(alice_rows) == 1
        assert len(bob_rows) == 1


class TestPhiAccessLogContent:
    """Log content faithfully records the access event."""

    def test_log_carries_hashed_identifiers_not_plaintext(self) -> None:
        """The model rejects nothing about hash format here — that's the
        deid_redactor's contract — but the field NAMES end in ``_hash`` so
        a future contributor cannot accidentally land plaintext HN/AN."""
        assert "hn_hash" in PhiAccessLog.model_fields
        assert "an_hash" in PhiAccessLog.model_fields
        assert "hn" not in PhiAccessLog.model_fields
        assert "an" not in PhiAccessLog.model_fields

    def test_log_has_break_glass_justification_field(self) -> None:
        assert "break_glass_justification" in PhiAccessLog.model_fields

    def test_break_glass_justification_rejects_empty_string(self) -> None:
        with pytest.raises(ValidationError):
            _phi_access_input(break_glass_justification="")
        with pytest.raises(ValidationError):
            _phi_access_input(break_glass_justification="   ")


# =============================================================================
# AC ④ — Concurrent-write test: 5 simulated reviewers, no lock contention
#
# WHY: the entire reason this module uses Postgres instead of SQLite is
# multi-reviewer concurrency. If five reviewers can simultaneously
# acknowledge five different audit rows and one of them blocks the others,
# the choice of Postgres is wasted.
# =============================================================================


class TestConcurrentWrites:
    """Five simultaneous reviewers race on writes; all complete without
    locking each other out."""

    def test_five_reviewers_concurrent_record_action(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        """Five threads, each writing one ReviewAction for a different audit.
        All five must complete; no two ``action_id`` values collide."""
        barrier = threading.Barrier(5)
        results: list[ReviewAction] = []
        lock = threading.Lock()

        def reviewer_task(idx: int) -> None:
            barrier.wait()  # all five fire writes at the same instant
            ra = migrated_store.record_action(
                _action_input(
                    audit_id=f"audit-{idx:02d}",
                    reviewer_id=f"reviewer-{idx:02d}",
                )
            )
            with lock:
                results.append(ra)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(reviewer_task, range(5)))

        assert len(results) == 5
        assert len({r.action_id for r in results}) == 5

    def test_five_reviewers_concurrent_completes_under_timeout(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        """Concurrent writes must NOT serialize behind a global lock — the
        five-reviewer task must complete well under five times a single-
        reviewer write. The 2.0s ceiling is a generous upper bound that
        still catches a SQLite-style table-lock regression."""
        import time

        barrier = threading.Barrier(5)

        def reviewer_task(idx: int) -> None:
            barrier.wait()
            migrated_store.record_phi_access(
                _phi_access_input(
                    reviewer_id=f"reviewer-{idx:02d}",
                    audit_id=f"audit-{idx:02d}",
                )
            )

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(reviewer_task, range(5)))
        elapsed = time.perf_counter() - t0

        assert elapsed < 2.0, (
            f"5 concurrent writes took {elapsed:.2f}s — likely serialized "
            f"behind a table lock (multi-reviewer load would degrade badly)"
        )

    def test_concurrent_writes_all_persist(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        """After the race, every issued action must be readable back. A
        write that was silently dropped (e.g., serialization conflict
        unhandled) would corrupt the audit trail."""

        def reviewer_task(idx: int) -> None:
            migrated_store.record_action(
                _action_input(
                    audit_id=f"audit-{idx:02d}",
                    reviewer_id=f"reviewer-{idx:02d}",
                )
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(reviewer_task, range(5)))

        all_rows = migrated_store.list_actions()
        assert len({r.audit_id for r in all_rows}) >= 5


# =============================================================================
# AC ⑤ — Schema migration scripts in ``migrations/``
#
# WHY: the schema lives in version-controlled files so the DBA / SRE can
# review changes, replay them on staging, and roll forward / back. A schema
# defined only in Python code (programmatically applied) cannot be reviewed
# the same way.
# =============================================================================


class TestMigrationScripts:
    """Alembic-or-equivalent migration files exist at the expected path."""

    def test_migrations_root_exists(self) -> None:
        assert (REPO_ROOT / "migrations").is_dir(), (
            "AC §⑤: migrations/ directory must exist at repo root"
        )

    def test_alembic_ini_exists(self) -> None:
        # Alembic's standard config file. If the migrator switches to an
        # equivalent tool (yoyo, dbmate), this test gets reworded — the AC
        # says "alembic OR equivalent".
        assert (REPO_ROOT / "migrations" / "alembic.ini").exists() or (
            REPO_ROOT / "alembic.ini"
        ).exists(), "AC §⑤: alembic.ini must exist"

    def test_versions_directory_exists(self) -> None:
        assert (REPO_ROOT / "migrations" / "versions").is_dir(), (
            "AC §⑤: migrations/versions/ must exist for alembic revision files"
        )

    def test_at_least_one_revision_exists(self) -> None:
        versions = REPO_ROOT / "migrations" / "versions"
        revision_files = list(versions.glob("*.py")) if versions.exists() else []
        assert len(revision_files) >= 1, (
            "AC §⑤: at least one migration revision must exist (the "
            "schema-creation initial migration)"
        )

    def test_initial_migration_creates_review_actions_table(self) -> None:
        """The initial migration's UP body references the ``review_actions``
        table — read as text to avoid executing the migration in this test.
        """
        versions = REPO_ROOT / "migrations" / "versions"
        if not versions.exists():
            pytest.fail("migrations/versions/ does not exist")
        revision_files = list(versions.glob("*.py"))
        assert revision_files, "no migration files found"

        all_text = "\n".join(p.read_text() for p in revision_files)
        assert "review_actions" in all_text, (
            "initial migration must create the review_actions table"
        )
        assert "phi_access_log" in all_text, (
            "initial migration must create the phi_access_log table"
        )

    def test_initial_migration_revokes_update_delete(self) -> None:
        """The initial migration includes ``REVOKE UPDATE, DELETE`` against
        both tables. The trigger guard is verified separately in
        :class:`TestAppendOnlyInvariantDBLevel` against the live DB."""
        versions = REPO_ROOT / "migrations" / "versions"
        if not versions.exists():
            pytest.fail("migrations/versions/ does not exist")
        all_text = "\n".join(p.read_text() for p in versions.glob("*.py"))

        # Match REVOKE on the two protected tables; alembic op.execute()
        # bodies use raw SQL strings.
        assert re.search(
            r"REVOKE\s+(?:UPDATE|DELETE).*review_actions", all_text, re.IGNORECASE
        ), "initial migration must REVOKE UPDATE/DELETE on review_actions"
        assert re.search(
            r"REVOKE\s+(?:UPDATE|DELETE).*phi_access_log",
            all_text,
            re.IGNORECASE,
        ), "initial migration must REVOKE UPDATE/DELETE on phi_access_log"

    def test_initial_migration_installs_trigger_guards(self) -> None:
        """A REVOKE alone is bypassable by a superuser misconfiguration; the
        trigger guard is the second line of defense (AC §② explicitly says
        ``trigger guard``)."""
        versions = REPO_ROOT / "migrations" / "versions"
        if not versions.exists():
            pytest.fail("migrations/versions/ does not exist")
        all_text = "\n".join(p.read_text() for p in versions.glob("*.py"))

        assert re.search(
            r"CREATE\s+(?:OR\s+REPLACE\s+)?TRIGGER", all_text, re.IGNORECASE
        ), "initial migration must install at least one trigger guard"

    def test_migrations_root_default_points_at_migrations(self) -> None:
        assert MIGRATIONS_ROOT_DEFAULT.name == "migrations"


class TestMigrationFunctions:
    """The Python wrapper exposes apply / current / head for callers."""

    def test_head_revision_returns_string(self) -> None:
        rev = head_revision(migrations_root=REPO_ROOT / "migrations")
        assert isinstance(rev, str)
        assert rev  # non-empty

    def test_current_revision_against_unmigrated_db_is_none(
        self, postgres_config: ReviewActionsConfig
    ) -> None:
        """A fresh container has no alembic version table → ``None``."""
        assert current_revision(postgres_config) is None

    def test_apply_migrations_advances_revision(
        self, postgres_config: ReviewActionsConfig
    ) -> None:
        assert current_revision(postgres_config) is None

        apply_migrations(postgres_config)

        assert current_revision(postgres_config) == head_revision(
            migrations_root=REPO_ROOT / "migrations"
        )

    def test_apply_migrations_is_idempotent(
        self, postgres_config: ReviewActionsConfig
    ) -> None:
        """Calling apply twice is a no-op the second time (alembic standard
        behavior). Guards against bootstrap scripts that call apply on every
        start."""
        apply_migrations(postgres_config)
        rev1 = current_revision(postgres_config)
        apply_migrations(postgres_config)
        rev2 = current_revision(postgres_config)
        assert rev1 == rev2


class TestStoreRequiresMigrations:
    """A store constructed against an unmigrated DB rejects writes loudly."""

    def test_record_action_against_unmigrated_db_raises(
        self, postgres_config: ReviewActionsConfig
    ) -> None:
        """Skipping ``apply_migrations`` → write raises MigrationStateError
        instead of "relation does not exist" (more actionable for ops).

        Uses ``require_unprivileged_role=False`` because the unprivileged
        role is created BY the migration; on an unmigrated DB the only
        available role is the superuser, and we want the alembic head check
        to fire (not the privilege check).
        """
        with ReviewActionsStore(
            postgres_config,
            migrations_root=REPO_ROOT / "migrations",
            require_unprivileged_role=False,
        ) as store:
            with pytest.raises(MigrationStateError):
                store.record_action(_action_input())


# =============================================================================
# Listing / filtering — exercise the read path that the dashboard consumes.
# =============================================================================


class TestListActions:
    """``list_actions`` filters and orders correctly."""

    def test_list_actions_filters_by_audit_id(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        migrated_store.record_action(_action_input(audit_id="a1"))
        migrated_store.record_action(_action_input(audit_id="a2"))

        only_a1 = migrated_store.list_actions(audit_id="a1")
        assert {r.audit_id for r in only_a1} == {"a1"}

    def test_list_actions_filters_by_reviewer_id(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        migrated_store.record_action(_action_input(audit_id="a1", reviewer_id="alice"))
        migrated_store.record_action(_action_input(audit_id="a2", reviewer_id="bob"))

        only_alice = migrated_store.list_actions(reviewer_id="alice")
        assert {r.reviewer_id for r in only_alice} == {"alice"}

    def test_list_actions_orders_chronologically(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        """The dashboard renders the reviewer-decision timeline; ordering
        must be ``(created_at, action_id)`` ascending."""
        first = migrated_store.record_action(
            _action_input(audit_id="a1", reviewer_id="alice")
        )
        second = migrated_store.record_action(
            _action_input(
                audit_id="a1",
                reviewer_id="alice",
                action="override",
                override_reason="changed my mind after rounds",
            )
        )

        rows = migrated_store.list_actions(audit_id="a1")
        assert rows[0].action_id == first.action_id
        assert rows[1].action_id == second.action_id


class TestAccessPhiContextManager:
    """``access_phi`` writes the log row BEFORE the body runs (AC §③).

    The wrapper exists because the AC says "every dashboard access of
    un-redacted text writes a row" — a plain ``record_phi_access`` puts the
    discipline on the dashboard. The context manager makes the structural
    pattern "wrap the un-redacted-text retrieval" the obvious one.
    """

    def test_access_phi_writes_log_at_enter(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        """Entering the context writes the row; the log row is visible to
        ``verify_phi_access_completeness`` before the body executes."""

        observed_completeness: list[bool] = []
        with migrated_store.access_phi(
            _phi_access_input(reviewer_id="alice", audit_id="audit-1")
        ):
            observed_completeness.append(
                migrated_store.verify_phi_access_completeness(
                    reviewer_id="alice", audit_id="audit-1"
                )
            )
        assert observed_completeness == [True]

    def test_access_phi_yields_persisted_log(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        with migrated_store.access_phi(_phi_access_input()) as log:
            assert isinstance(log, PhiAccessLog)
            assert log.access_id > 0

    def test_access_phi_log_survives_body_exception(
        self, migrated_store: ReviewActionsStore
    ) -> None:
        """An access attempt that crashed mid-render is still an access for
        audit purposes; the row stays committed."""
        with pytest.raises(RuntimeError, match="render crashed"):
            with migrated_store.access_phi(
                _phi_access_input(reviewer_id="alice", audit_id="audit-1")
            ):
                raise RuntimeError("render crashed")

        # Open a new store handle (the previous one may have been damaged
        # by the raise) and verify the row is there.
        rows = migrated_store.list_phi_access(reviewer_id="alice")
        assert len(rows) == 1


class TestStorePrivilegeCheck:
    """The store refuses to operate as a role that can bypass triggers."""

    def test_superuser_role_is_rejected(
        self, postgres_config: ReviewActionsConfig
    ) -> None:
        """Connecting as the container's superuser → MigrationStateError.

        Superusers can ``ALTER TABLE ... DISABLE TRIGGER`` or
        ``SET session_replication_role = 'replica'`` to bypass the trigger
        guard. The startup check refuses this configuration.
        """
        apply_migrations(postgres_config)
        with ReviewActionsStore(
            postgres_config, migrations_root=REPO_ROOT / "migrations"
        ) as store:
            with pytest.raises(MigrationStateError, match="superuser"):
                store.record_action(_action_input())

    def test_owner_role_is_rejected(self, postgres_config: ReviewActionsConfig) -> None:
        """The owner of a protected table can DISABLE TRIGGER on it. Reject
        any DSN that connects as the table owner.

        The testcontainers default user is both superuser AND owner; the
        superuser branch fires first, so this test creates a non-superuser
        owner explicitly.
        """
        apply_migrations(postgres_config)
        # Create a non-superuser role and transfer table ownership to it.
        with psycopg.connect(postgres_config.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE ROLE owner_role WITH LOGIN PASSWORD 'owner_pw'")
                cur.execute("ALTER TABLE review_actions OWNER TO owner_role")
                cur.execute("ALTER TABLE phi_access_log OWNER TO owner_role")
        owner_dsn = _swap_dsn_credentials(
            postgres_config.dsn, user="owner_role", password="owner_pw"
        )
        with ReviewActionsStore(
            ReviewActionsConfig(dsn=owner_dsn),
            migrations_root=REPO_ROOT / "migrations",
        ) as store:
            with pytest.raises(MigrationStateError, match="owns"):
                store.record_action(_action_input())

    def test_privilege_check_can_be_explicitly_disabled(
        self, postgres_config: ReviewActionsConfig
    ) -> None:
        """``require_unprivileged_role=False`` allows the superuser DSN —
        used in tests that exercise the trigger-side defense directly."""
        apply_migrations(postgres_config)
        with ReviewActionsStore(
            postgres_config,
            migrations_root=REPO_ROOT / "migrations",
            require_unprivileged_role=False,
        ) as store:
            written = store.record_action(_action_input())
            assert written.action_id > 0


class TestSchemaDriftDetection:
    """The store rejects a drifted schema (missing/disabled/swapped trigger,
    drifted alembic head, multi-head graph)."""

    def test_missing_trigger_is_rejected(
        self,
        app_config: ReviewActionsConfig,
        postgres_config: ReviewActionsConfig,
    ) -> None:
        """Drop one of the required triggers; the store refuses to operate.

        The drop runs as superuser (since the app role cannot ALTER TABLE);
        this simulates an operator who manually removed a guard between
        migrations.
        """
        with psycopg.connect(postgres_config.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DROP TRIGGER review_actions_block_update ON review_actions"
                )
        with ReviewActionsStore(
            app_config, migrations_root=REPO_ROOT / "migrations"
        ) as store:
            with pytest.raises(MigrationStateError, match="trigger guards missing"):
                store.record_action(_action_input())

    def test_disabled_trigger_is_rejected(
        self,
        app_config: ReviewActionsConfig,
        postgres_config: ReviewActionsConfig,
    ) -> None:
        """``ALTER TABLE ... DISABLE TRIGGER`` keeps the row in
        ``pg_trigger`` but neuters the guard. The integrity check must
        examine ``tgenabled``, not just the row's existence."""
        with psycopg.connect(postgres_config.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "ALTER TABLE review_actions "
                    "DISABLE TRIGGER review_actions_block_update"
                )
        with ReviewActionsStore(
            app_config, migrations_root=REPO_ROOT / "migrations"
        ) as store:
            with pytest.raises(MigrationStateError, match="not enabled"):
                store.record_action(_action_input())

    def test_trigger_bound_to_wrong_function_is_rejected(
        self,
        app_config: ReviewActionsConfig,
        postgres_config: ReviewActionsConfig,
    ) -> None:
        """An attacker who swaps the trigger function for a no-op silently
        disables the guard while leaving ``pg_trigger`` populated. The
        integrity check verifies the function name on each trigger row."""
        with psycopg.connect(postgres_config.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE OR REPLACE FUNCTION noop_guard() "
                    "RETURNS TRIGGER LANGUAGE plpgsql AS "
                    "$$ BEGIN RETURN NULL; END; $$"
                )
                cur.execute(
                    "DROP TRIGGER review_actions_block_update ON review_actions"
                )
                cur.execute(
                    "CREATE TRIGGER review_actions_block_update "
                    "BEFORE UPDATE ON review_actions "
                    "FOR EACH ROW EXECUTE FUNCTION noop_guard()"
                )
        with ReviewActionsStore(
            app_config, migrations_root=REPO_ROOT / "migrations"
        ) as store:
            with pytest.raises(MigrationStateError, match="noop_guard"):
                store.record_action(_action_input())

    def test_alembic_revision_drift_is_rejected(
        self,
        app_config: ReviewActionsConfig,
        postgres_config: ReviewActionsConfig,
    ) -> None:
        """If the live DB's alembic version doesn't match the on-disk head,
        the store refuses. Forge the drift by writing a bogus revision id."""
        with psycopg.connect(postgres_config.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("UPDATE alembic_version SET version_num = 'forged_drift'")
        with ReviewActionsStore(
            app_config, migrations_root=REPO_ROOT / "migrations"
        ) as store:
            with pytest.raises(MigrationStateError, match="revision drift"):
                store.record_action(_action_input())

    def test_multi_head_alembic_state_is_rejected(
        self,
        app_config: ReviewActionsConfig,
        postgres_config: ReviewActionsConfig,
    ) -> None:
        """Forge a multi-head DB state (an unmerged revision graph) by
        inserting a second row into ``alembic_version``. The migrator's
        current_revision must refuse to collapse to a scalar."""
        with psycopg.connect(postgres_config.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO alembic_version (version_num) "
                    "VALUES ('phantom_head_for_test')"
                )
        with ReviewActionsStore(
            app_config, migrations_root=REPO_ROOT / "migrations"
        ) as store:
            with pytest.raises(MigrationStateError, match="multiple"):
                store.record_action(_action_input())


class TestTriggerActuallyFires:
    """The trigger guard fires (P0001) even when the role has UPDATE granted.

    The append-only DB-level tests verify the COMBINATION of REVOKE +
    trigger blocks mutations. They pass even if the trigger is broken,
    because REVOKE alone produces InsufficientPrivilege. This class isolates
    the trigger contribution by granting UPDATE/DELETE on a fresh role and
    asserting the trigger still raises ``append_only_violation``.
    """

    @pytest.fixture
    def writer_role_dsn(
        self,
        postgres_config: ReviewActionsConfig,
        app_config: ReviewActionsConfig,
    ) -> str:
        """A role with UPDATE/DELETE granted; not the owner, not a superuser.

        The ``app_config`` fixture has already migrated the DB (via the
        ``app_config`` -> ``postgres_config`` chain). Here we create a
        privileged-write role and return a DSN connecting as it.
        """
        with psycopg.connect(postgres_config.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE ROLE writer_role WITH LOGIN PASSWORD 'writer_pw'")
                cur.execute(
                    "GRANT INSERT, SELECT, UPDATE, DELETE "
                    "ON review_actions TO writer_role"
                )
                cur.execute(
                    "GRANT INSERT, SELECT, UPDATE, DELETE "
                    "ON phi_access_log TO writer_role"
                )
        return _swap_dsn_credentials(
            postgres_config.dsn, user="writer_role", password="writer_pw"
        )

    def test_update_with_granted_privilege_still_blocked_by_trigger(
        self,
        migrated_store: ReviewActionsStore,
        writer_role_dsn: str,
    ) -> None:
        """A role WITH UPDATE granted still cannot mutate — the trigger
        raises P0001 ``append_only_violation``. Verifies the trigger guard
        is doing real work, not just shadow of the REVOKE."""
        written = migrated_store.record_action(_action_input())

        try:
            with psycopg.connect(writer_role_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE review_actions SET note = %s WHERE action_id = %s",
                        ("tampered", written.action_id),
                    )
        except psycopg.errors.RaiseException as exc:
            assert exc.sqlstate == "P0001"
            assert "append_only_violation" in str(exc.diag.message_primary)
        else:
            pytest.fail("UPDATE succeeded — trigger did not fire")

    def test_delete_with_granted_privilege_still_blocked_by_trigger(
        self,
        migrated_store: ReviewActionsStore,
        writer_role_dsn: str,
    ) -> None:
        """Same as the UPDATE test, for DELETE."""
        written = migrated_store.record_action(_action_input())

        try:
            with psycopg.connect(writer_role_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM review_actions WHERE action_id = %s",
                        (written.action_id,),
                    )
        except psycopg.errors.RaiseException as exc:
            assert exc.sqlstate == "P0001"
        else:
            pytest.fail("DELETE succeeded — trigger did not fire")


class TestOwnerMembershipBypassRejected:
    """A role with MEMBER access to the table owner can SET ROLE owner
    and DISABLE TRIGGER. The store's privilege check uses
    ``pg_has_role(..., 'MEMBER')`` (not ``'USAGE'``) so both inheriting
    AND non-inheriting members are rejected."""

    def test_inheriting_member_of_owner_role_is_rejected(
        self,
        postgres_config: ReviewActionsConfig,
    ) -> None:
        """Standard (inheriting) member of the owner role → rejected."""
        apply_migrations(postgres_config)
        with psycopg.connect(postgres_config.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE ROLE table_owner")
                cur.execute("ALTER TABLE review_actions OWNER TO table_owner")
                cur.execute("ALTER TABLE phi_access_log OWNER TO table_owner")
                cur.execute("GRANT SELECT ON alembic_version TO table_owner")
                cur.execute("CREATE ROLE shady_member WITH LOGIN PASSWORD 'pw'")
                cur.execute("GRANT table_owner TO shady_member")
                cur.execute("GRANT SELECT ON alembic_version TO shady_member")
        member_dsn = _swap_dsn_credentials(
            postgres_config.dsn, user="shady_member", password="pw"
        )
        with ReviewActionsStore(
            ReviewActionsConfig(dsn=member_dsn),
            migrations_root=REPO_ROOT / "migrations",
        ) as store:
            with pytest.raises(MigrationStateError, match="MEMBER access"):
                store.record_action(_action_input())

    def test_noinherit_member_of_owner_role_is_rejected(
        self,
        postgres_config: ReviewActionsConfig,
    ) -> None:
        """``NOINHERIT`` member can still ``SET ROLE`` to the owner; the
        old ``USAGE``-mode check would have missed this path."""
        apply_migrations(postgres_config)
        with psycopg.connect(postgres_config.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE ROLE table_owner")
                cur.execute("ALTER TABLE review_actions OWNER TO table_owner")
                cur.execute("ALTER TABLE phi_access_log OWNER TO table_owner")
                cur.execute("GRANT SELECT ON alembic_version TO table_owner")
                cur.execute(
                    "CREATE ROLE noinherit_member WITH LOGIN NOINHERIT PASSWORD 'pw'"
                )
                cur.execute("GRANT table_owner TO noinherit_member")
                cur.execute("GRANT SELECT ON alembic_version TO noinherit_member")
        member_dsn = _swap_dsn_credentials(
            postgres_config.dsn,
            user="noinherit_member",
            password="pw",
        )
        with ReviewActionsStore(
            ReviewActionsConfig(dsn=member_dsn),
            migrations_root=REPO_ROOT / "migrations",
        ) as store:
            with pytest.raises(MigrationStateError, match="MEMBER access"):
                store.record_action(_action_input())


class TestTriggerShapeDriftDetection:
    """Beyond name + table, the integrity check verifies the trigger event,
    timing, schema, and the function's schema."""

    def test_wrong_event_trigger_is_rejected(
        self,
        app_config: ReviewActionsConfig,
        postgres_config: ReviewActionsConfig,
    ) -> None:
        """Drop the BEFORE UPDATE trigger and recreate it as BEFORE
        INSERT — same name, same function, but wrong event. The integrity
        check must catch the ``tgtype`` drift."""
        with psycopg.connect(postgres_config.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DROP TRIGGER review_actions_block_update ON review_actions"
                )
                cur.execute(
                    "CREATE TRIGGER review_actions_block_update "
                    "BEFORE INSERT ON review_actions "
                    "FOR EACH ROW EXECUTE FUNCTION "
                    "review_actions_block_mutation()"
                )
        with ReviewActionsStore(
            app_config, migrations_root=REPO_ROOT / "migrations"
        ) as store:
            with pytest.raises(MigrationStateError, match="tgtype"):
                store.record_action(_action_input())

    def test_trigger_in_wrong_schema_is_rejected(
        self,
        app_config: ReviewActionsConfig,
        postgres_config: ReviewActionsConfig,
    ) -> None:
        """An attacker moves the protected table (and its trigger) to a
        non-public schema, then re-grants on a same-named decoy table in
        public. The integrity check filters by ``relnamespace = 'public'``,
        so the public-schema table now has zero matching triggers and
        fails the check."""
        with psycopg.connect(postgres_config.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE SCHEMA shadow_schema")
                cur.execute("ALTER TABLE review_actions SET SCHEMA shadow_schema")
                cur.execute(
                    "CREATE TABLE review_actions ("
                    "  action_id BIGSERIAL PRIMARY KEY, "
                    "  audit_id TEXT, reviewer_id TEXT, action TEXT, "
                    "  override_reason TEXT, note TEXT, "
                    "  created_at TIMESTAMPTZ"
                    ")"
                )
                cur.execute(
                    "GRANT INSERT, SELECT ON review_actions TO review_actions_app"
                )
                cur.execute(
                    "GRANT USAGE, SELECT ON SEQUENCE "
                    "review_actions_action_id_seq "
                    "TO review_actions_app"
                )
        with ReviewActionsStore(
            app_config, migrations_root=REPO_ROOT / "migrations"
        ) as store:
            with pytest.raises(MigrationStateError):
                store.record_action(_action_input())


class TestAppendOnlyTranslationContract:
    """The store translates a trigger's ``RAISE EXCEPTION
    'append_only_violation'`` into :class:`AppendOnlyViolationError`
    when it fires during a store INSERT.

    In production today the store only INSERTs and the migration's
    triggers fire only on UPDATE/DELETE/TRUNCATE, so this translation
    is dead in normal flow. The contract still has to hold: a future
    migration may add a defense-in-depth BEFORE INSERT guard, or an
    operator may attach an external trigger that uses the same
    SQLSTATE + message. Either case must surface the typed exception
    so callers don't have to know about psycopg internals.

    PR #37 codex P2 — the matcher token had drifted out of sync with
    the trigger message after the round-2 info-disclosure fix; this
    test regression-guards that synchronization.
    """

    def test_insert_blocked_by_trigger_raises_append_only_error(
        self,
        app_config: ReviewActionsConfig,
        postgres_config: ReviewActionsConfig,
    ) -> None:
        """Install a temporary BEFORE INSERT trigger reusing the
        production ``review_actions_block_mutation`` function. A store
        INSERT now hits the trigger; the resulting psycopg
        ``RaiseException`` (SQLSTATE P0001 + 'append_only_violation')
        must be translated to :class:`AppendOnlyViolationError`."""
        with psycopg.connect(postgres_config.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE TRIGGER test_block_insert_for_translation "
                    "BEFORE INSERT ON review_actions "
                    "FOR EACH ROW EXECUTE FUNCTION "
                    "review_actions_block_mutation()"
                )

        with ReviewActionsStore(
            app_config, migrations_root=REPO_ROOT / "migrations"
        ) as store:
            with pytest.raises(AppendOnlyViolationError):
                store.record_action(_action_input())


class TestSecurityLogging:
    """Security-critical integrity failures emit structured log records
    so operators can alert on repeated bypass attempts."""

    def test_superuser_rejection_logs_security_event(
        self,
        postgres_config: ReviewActionsConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The ``superuser_role_rejected`` event lands on the
        ``bba.review_actions.security`` logger with structured fields."""
        apply_migrations(postgres_config)
        caplog.set_level("WARNING", logger="bba.review_actions.security")
        with ReviewActionsStore(
            postgres_config, migrations_root=REPO_ROOT / "migrations"
        ) as store:
            with pytest.raises(MigrationStateError):
                store.record_action(_action_input())

        security_records = [
            r for r in caplog.records if r.name == "bba.review_actions.security"
        ]
        assert len(security_records) == 1
        record = security_records[0]
        assert (
            getattr(record, "review_actions_security_event", None)
            == "superuser_role_rejected"
        )
        # The current_user field is on the record's extra dict.
        assert hasattr(record, "current_user")

    def test_trigger_drift_logs_security_event(
        self,
        app_config: ReviewActionsConfig,
        postgres_config: ReviewActionsConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with psycopg.connect(postgres_config.dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DROP TRIGGER review_actions_block_update ON review_actions"
                )

        caplog.set_level("WARNING", logger="bba.review_actions.security")
        with ReviewActionsStore(
            app_config, migrations_root=REPO_ROOT / "migrations"
        ) as store:
            with pytest.raises(MigrationStateError):
                store.record_action(_action_input())

        security_records = [
            r for r in caplog.records if r.name == "bba.review_actions.security"
        ]
        assert any(
            getattr(r, "review_actions_security_event", None)
            == "required_triggers_missing"
            for r in security_records
        )


class TestStoreLifecycle:
    """Store can be used as a context manager."""

    def test_context_manager_closes_on_exit(
        self, app_config: ReviewActionsConfig
    ) -> None:
        with ReviewActionsStore(
            app_config, migrations_root=REPO_ROOT / "migrations"
        ) as store:
            assert isinstance(store, ReviewActionsStore)
        # The post-exit state is implementation-private; the contract is
        # that ``close()`` was called. Re-entering the context after close
        # is not part of the contract.

    def test_close_is_idempotent(self, app_config: ReviewActionsConfig) -> None:
        store = ReviewActionsStore(app_config, migrations_root=REPO_ROOT / "migrations")
        store.close()
        store.close()  # must not raise


# =============================================================================
# End of test file.
#
# Test count budget: ~50 tests across 12 classes. Each test states a single
# WHY and exercises a single contract. The Postgres-required tests are
# clustered so the live container is spun up only once per pytest session.
# =============================================================================
