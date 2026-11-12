"""Bind reviewer forms and decisions to immutable audit-row evidence."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from typing import Literal, get_args
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bba.audit_store import AuditRow, Classification
from bba.dashboard.auth import csrf_token, check_csrf
from bba.dashboard.exceptions import AuditNotFoundError
from bba.dashboard.models import DashboardConfig, RouteContext
from bba.review_actions import ReviewAction, ReviewActionInput, ReviewConflictError


class DecisionForm(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal["agree", "override", "escalate"]
    run_id: str
    code_version: str
    row_hash: str
    expected_revision: int = Field(ge=0)
    submission_id: str
    override_classification: Classification | None = None
    override_reason: str | None = None
    note: str | None = None


def row_hash(row: AuditRow) -> str:
    payload = json.dumps(
        row.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def selected_version(config: DashboardConfig) -> str:
    return config.code_version or config.audit_store.config.code_version


def history(config: DashboardConfig, row: AuditRow) -> tuple[ReviewAction, ...]:
    version = selected_version(config)
    digest = row_hash(row)
    return tuple(
        sorted(
            (
                action
                for action in config.review_actions_store.list_actions(
                    audit_id=row.audit_id
                )
                if (action.run_id, action.code_version, action.row_hash)
                == (row.run_id, version, digest)
            ),
            key=lambda action: action.action_id,
        )
    )


def unresolved(config: DashboardConfig, row: AuditRow) -> bool:
    decisions = history(config, row)
    latest_decision = next(
        (
            item
            for item in reversed(decisions)
            if item.action != "use_as_few_shot_candidate"
        ),
        None,
    )
    return latest_decision is None or latest_decision.action == "escalate"


def case_context(
    request: Request, context: RouteContext, config: DashboardConfig, row: AuditRow
) -> dict[str, object]:
    decisions = history(config, row)
    return {
        "history": decisions,
        "expected_revision": decisions[-1].action_id if decisions else 0,
        "submission_id": str(uuid4()),
        "csrf_token": csrf_token(request, context),
        "classifications": get_args(Classification),
        "code_version": selected_version(config),
    }


def record_decision(
    config: DashboardConfig, context: RouteContext, row: AuditRow, form: DecisionForm
) -> ReviewAction:
    if (form.run_id, form.code_version, form.row_hash) != (
        row.run_id,
        selected_version(config),
        row_hash(row),
    ):
        raise HTTPException(
            status_code=409, detail="audit row changed; reload before deciding"
        )
    # The live store check refuses a form for a row no longer identical to its snapshot.
    current = config.audit_store.read_audit_results(
        run_id=row.run_id, code_version=selected_version(config)
    )
    if not any(
        item.audit_id == row.audit_id and row_hash(item) == form.row_hash
        for item in current
    ):
        raise HTTPException(
            status_code=409, detail="audit evidence changed; reload before deciding"
        )
    try:
        action = ReviewActionInput(
            audit_id=row.audit_id,
            reviewer_id=context.reviewer.reviewer_id,
            **form.model_dump(),
        )
        return config.review_actions_store.record_action(action)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail="override requires a classification and non-empty reason",
        ) from exc
    except ReviewConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


async def submit_decision(
    request: Request,
    audit_id: str,
    context: RouteContext,
    config: DashboardConfig,
    find_row: Callable[[str, str | None], AuditRow],
) -> Response:
    fields: dict[str, object] = dict(await request.form())
    check_csrf(request, context, str(fields.pop("csrf_token", "")))
    for name in ("override_classification", "override_reason", "note"):
        if fields.get(name) == "":
            fields[name] = None
    try:
        form = DecisionForm.model_validate(fields)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail="invalid decision fields") from exc
    try:
        row = await run_in_threadpool(find_row, audit_id, form.run_id)
    except AuditNotFoundError as exc:
        raise HTTPException(
            status_code=409, detail="audit row is no longer available"
        ) from exc
    await run_in_threadpool(record_decision, config, context, row, form)
    return RedirectResponse(f"/case/{audit_id}?run_id={row.run_id}", status_code=303)
