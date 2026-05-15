"""Frozen pydantic models for the review_actions contract.

PRD §16 Implementation Decisions — Postgres-backed mutable state for reviewer
decisions and PHI-access audit. Two append-only tables, both immutable from
the application's POV (DB-level UPDATE/DELETE revoked + trigger guard).

The Python-layer models declared here mirror the persisted row shapes:

* :class:`ReviewActionInput` / :class:`ReviewAction` — reviewer-issued
  decisions on a classified audit row: ``agree``, ``override(reason)``,
  ``escalate``, ``use_as_few_shot_candidate``.
* :class:`PhiAccessInput` / :class:`PhiAccessLog` — every dashboard access of
  un-redacted text writes a row carrying
  ``(reviewer_id, accessed_at, hn_hash, an_hash, break_glass_justification)``.

All datetimes are tz-aware UTC at persistence time — enforced at the model
boundary so a naive timestamp cannot leak past construction (mirrors
``bba.audit_store`` and ``bba.ingest`` invariants).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator


ActionKind = Literal[
    "agree",
    "override",
    "escalate",
    "use_as_few_shot_candidate",
]
"""The four reviewer-decision kinds enumerated in PRD §16.

The literal-union form (vs ``enum.Enum``) is symmetric with
``bba.audit_store.Classification`` — the audit pipeline already speaks string
literals and Pydantic v2 round-trips them losslessly to JSON / Postgres ``text``.
"""


ACTION_KINDS: tuple[str, ...] = (
    "agree",
    "override",
    "escalate",
    "use_as_few_shot_candidate",
)
"""Runtime tuple of valid ``ActionKind`` values for introspection / tests.

Kept in lock-step with :data:`ActionKind`. A new kind requires updating BOTH
this tuple and the ``Literal`` declaration (the type checker catches
mismatches at use-sites).
"""


def _ensure_utc(dt: datetime) -> datetime:
    """Reject naive datetimes; normalize aware non-UTC to UTC.

    The store-level invariant (PRD §"Tz-aware throughout", CONTEXT.md
    "tz-aware UTC") is asserted at the model boundary so a naive timestamp
    cannot leak past construction. A naive ``accessed_at`` on a PHI-access row
    would compare incorrectly against the audit row's UTC ``order_datetime``
    in dashboard / report-generator queries.
    """
    if dt.tzinfo is None:
        raise ValueError(
            "datetime must be tz-aware; naive datetimes are forbidden in "
            "review_actions (see CONTEXT.md 'tz-aware UTC')"
        )
    return dt.astimezone(UTC)


UTCDatetime = Annotated[datetime, AfterValidator(_ensure_utc)]
"""A ``datetime`` constrained to tz-aware UTC at validation time."""


_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_safe_id(value: str) -> str:
    """Reject identifiers that would be unsafe to embed in identifier
    contexts (path components, log lines, error messages echoing the value).

    Same allow-list as ``bba.audit_store``: non-empty, ``[A-Za-z0-9._-]+``,
    not ``.``/``..``. The defense lives at the model boundary so downstream
    consumers (SQL query composers, file paths) can treat IDs as opaque safe
    strings. SQL injection is independently prevented at the DB layer via
    parameterized queries; this validator's job is to keep the *log surface*
    safe — a reviewer_id containing a newline would corrupt audit-log lines.
    """
    if not value:
        raise ValueError("identifier must not be empty")
    if not _SAFE_ID_PATTERN.match(value):
        raise ValueError(
            f"identifier must match [A-Za-z0-9._-]+ (got {value!r})"
        )
    if value in {".", ".."}:
        raise ValueError(
            f"identifier must not be a path-traversal segment (got {value!r})"
        )
    return value


SafeId = Annotated[str, AfterValidator(_validate_safe_id)]
"""A ``str`` constrained to a filesystem / log-safe identifier shape."""


def _validate_nonempty(value: str) -> str:
    """Reject empty / whitespace-only strings on required-text fields.

    ``override_reason`` and ``break_glass_justification`` carry the reviewer's
    rationale into the audit trail; an empty string would defeat the entire
    point of requiring them. Strips before checking to catch e.g. ``"   "``.
    """
    if not value or not value.strip():
        raise ValueError("text must not be empty or whitespace-only")
    return value


NonEmptyStr = Annotated[str, AfterValidator(_validate_nonempty)]


class ReviewActionInput(BaseModel):
    """Input to :meth:`ReviewActionsStore.record_action`.

    No ``action_id`` / ``created_at`` — those are DB-assigned at INSERT time.
    The four :data:`ActionKind` values mirror PRD §16:

    * ``agree`` — reviewer confirms the pipeline's classification.
    * ``override`` — reviewer dissents; ``override_reason`` is REQUIRED so the
      dashboard's reviewer-rationale view never surfaces a bare dissent.
    * ``escalate`` — reviewer flags for senior review.
    * ``use_as_few_shot_candidate`` — reviewer marks the row as exemplar for
      future few-shot prompts (audit-pipeline #24 reads this back).

    Field-level enforcement (``override_reason`` non-empty when set) lives
    here at the input boundary so the store never has to re-validate.
    """

    model_config = ConfigDict(frozen=True)

    audit_id: SafeId
    reviewer_id: SafeId
    action: ActionKind
    override_reason: NonEmptyStr | None = None
    note: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _override_requires_reason(self) -> ReviewActionInput:
        """``action == 'override'`` MUST carry a non-empty ``override_reason``.

        Bare dissent (``override`` with no reason) breaks the reviewer-rationale
        audit trail the dashboard renders for senior-review handoff.
        """
        if self.action == "override" and self.override_reason is None:
            raise ValueError(
                "ReviewActionInput.override_reason is required when "
                "action='override'; a bare dissent is not auditable"
            )
        if self.action != "override" and self.override_reason is not None:
            raise ValueError(
                f"ReviewActionInput.override_reason is only valid when "
                f"action='override' (got action={self.action!r})"
            )
        return self


class ReviewAction(BaseModel):
    """A persisted reviewer decision.

    Append-only — once written, immutable. ``action_id`` is the DB-assigned
    bigserial PK; ``created_at`` is the server-side ``now()`` at INSERT (so
    multiple processes cannot disagree about wall-clock).
    """

    model_config = ConfigDict(frozen=True)

    action_id: int
    audit_id: SafeId
    reviewer_id: SafeId
    action: ActionKind
    override_reason: str | None
    note: str | None
    created_at: UTCDatetime


class PhiAccessInput(BaseModel):
    """Input to :meth:`ReviewActionsStore.record_phi_access`.

    Every dashboard read of un-redacted text MUST construct one of these and
    write a row BEFORE surfacing the text. Break-glass workflows (clinical
    emergency override of the standard redaction policy) carry
    ``break_glass_justification`` so post-hoc audit can trace the override.

    ``hn_hash`` / ``an_hash`` are the SHA-256 hashes computed at ingest time;
    the un-hashed values never enter this module — PRD §17 ``deid_redactor``
    is the single source of truth for PHI identifiers.
    """

    model_config = ConfigDict(frozen=True)

    reviewer_id: SafeId
    audit_id: SafeId
    hn_hash: str
    an_hash: str
    break_glass_justification: NonEmptyStr | None = None


class PhiAccessLog(BaseModel):
    """A persisted PHI-access event. Append-only (same DB-level guards as
    :class:`ReviewAction`)."""

    model_config = ConfigDict(frozen=True)

    access_id: int
    reviewer_id: SafeId
    audit_id: SafeId
    hn_hash: str
    an_hash: str
    break_glass_justification: str | None
    accessed_at: UTCDatetime


def _normalize_dsn(dsn: str) -> str:
    """Normalize a Postgres DSN to libpq form (``postgresql://...``).

    Accepts both libpq URIs (``postgresql://...``, ``postgres://...``) and
    SQLAlchemy dialect-prefixed forms (``postgresql+psycopg2://...``,
    ``postgresql+psycopg://...``). The dialect prefix is stripped because
    psycopg's connection pool wants a libpq URI; the migrator re-adds
    ``+psycopg`` when handing off to SQLAlchemy / alembic.

    Why this matters: ``testcontainers.postgres.PostgresContainer`` returns
    a SQLAlchemy URL by default. Centralizing the normalization here means
    callers (the test fixture, env-var resolution, dashboard bootstrap) all
    pass through one canonical form.
    """
    if dsn.startswith(("postgresql+psycopg2://", "postgresql+psycopg://")):
        return "postgresql://" + dsn.split("://", 1)[1]
    if dsn.startswith(("postgresql://", "postgres://")):
        return dsn
    raise ValueError(
        f"unsupported DSN scheme: {dsn!r}; expected postgresql:// or "
        f"postgresql+psycopg://"
    )


LibpqDsn = Annotated[str, AfterValidator(_normalize_dsn)]
"""A Postgres DSN normalized to libpq form."""


class ReviewActionsConfig(BaseModel):
    """Postgres connection configuration.

    ``dsn`` is stored in libpq form (``postgresql://user:pass@host:port/db``)
    regardless of the input shape — see :func:`_normalize_dsn`. Secrets
    MUST NOT be hardcoded; the caller resolves credentials from env vars or
    a secret manager and constructs the config object.

    ``app_name`` is reported back via Postgres ``application_name`` so the
    DBA can correlate connections to the audit pipeline.
    """

    model_config = ConfigDict(frozen=True)

    dsn: LibpqDsn
    app_name: str = "bba.review_actions"

    @property
    def sqlalchemy_dsn(self) -> str:
        """The DSN as a SQLAlchemy URL using the psycopg 3 driver.

        Alembic's env.py uses ``engine_from_config(prefix='sqlalchemy.')``
        which expects a dialect-prefixed URL. The store consumes the plain
        libpq form (``self.dsn``); the migrator consumes this property.
        """
        return self.dsn.replace("postgresql://", "postgresql+psycopg://", 1)


__all__: Sequence[str] = (
    "ACTION_KINDS",
    "ActionKind",
    "LibpqDsn",
    "NonEmptyStr",
    "PhiAccessInput",
    "PhiAccessLog",
    "ReviewAction",
    "ReviewActionInput",
    "ReviewActionsConfig",
    "SafeId",
    "UTCDatetime",
)
