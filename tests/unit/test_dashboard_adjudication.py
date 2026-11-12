"""Reviewer decisions must stay bound to the evidence a reviewer saw."""

from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError

from test_dashboard import _audit_row, _llm_call, _InMemoryReviewActionsStore
from bba.audit_store import AuditStore, AuditStoreConfig
from bba.dashboard import DashboardConfig, Reviewer, create_app
from bba.review_actions import ReviewAction, ReviewActionInput


class Actions(_InMemoryReviewActionsStore):
    def __init__(self):
        super().__init__()
        self.actions = []

    def list_actions(self, *, audit_id=None, reviewer_id=None):
        return tuple(
            a
            for a in self.actions
            if (audit_id is None or a.audit_id == audit_id)
            and (reviewer_id is None or a.reviewer_id == reviewer_id)
        )

    def record_action(self, action):
        from bba.review_actions import ReviewConflictError

        matching = [
            a
            for a in self.actions
            if (a.audit_id, a.run_id, a.code_version, a.row_hash)
            == (action.audit_id, action.run_id, action.code_version, action.row_hash)
        ]
        for prior in self.actions:
            if prior.submission_id == action.submission_id:
                if all(
                    getattr(prior, key) == value
                    for key, value in action.model_dump().items()
                ):
                    return prior
                raise ReviewConflictError("submission reused")
        if (matching[-1].action_id if matching else 0) != action.expected_revision:
            raise ReviewConflictError("stale decision")
        result = ReviewAction(
            action_id=len(self.actions) + 1,
            created_at=datetime.now(UTC),
            **action.model_dump(),
        )
        self.actions.append(result)
        return result


class Fields(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.values = {}
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "input" and attrs.get("type") == "hidden":
            self.values[attrs["name"]] = attrs["value"]


@pytest.fixture
def dashboard(tmp_path):
    store = AuditStore(AuditStoreConfig(root_dir=tmp_path / "audit", code_version="v1"))
    actions = Actions()
    reviewer = Reviewer(
        reviewer_id="trusted-reviewer", name="Reviewer", role="reviewer"
    )
    config = DashboardConfig(
        audit_store=store,
        review_actions_store=actions,
        snapshot_dir=tmp_path / "snap",
        default_reviewer=reviewer,
    )
    return store, actions, config


def write(store, row):
    store.write(row, (_llm_call(audit_id=row.audit_id, run_id=row.run_id),))


def test_queue_covers_review_flags_and_both_components(dashboard):
    store, _, config = dashboard
    for idx, classification in enumerate(
        (
            "NEEDS_REVIEW",
            "INSUFFICIENT_EVIDENCE",
            "POTENTIALLY_INAPPROPRIATE",
            "APPROPRIATE",
        )
    ):
        row = _audit_row(audit_id=f"audit-{idx}", final_classification=classification)
        write(store, row)
    write(
        store,
        _audit_row(audit_id="platelet").model_copy(
            update={
                "component": "platelet",
                "hb_freshness": "missing",
                "platelet_value": 15.0,
                "platelet_freshness": "fresh",
            }
        ),
    )
    response = TestClient(create_app(config)).get("/queue")
    assert response.status_code == 200
    for audit_id in ("audit-0", "audit-1", "audit-2", "audit-3", "platelet"):
        assert f"/case/{audit_id}" in response.text
    assert "Platelet" in response.text and "Red cell" in response.text


def test_review_journey_citations_history_and_resolution(dashboard):
    store, actions, config = dashboard
    row = _audit_row().model_copy(
        update={
            "indications_json": (
                {"quote": "synthetic indication", "source_timeline_index": 3},
            )
        }
    )
    write(store, row)
    client = TestClient(create_app(config))
    page = client.get("/case/audit-001")
    assert "สรุปเหตุผล" in page.text and "synthetic indication" in page.text
    assert "Asia/Bangkok" in page.text
    form = Fields(page.text).values
    response = client.post(
        "/case/audit-001/decision",
        data={
            **form,
            "action": "override",
            "override_classification": "APPROPRIATE",
            "override_reason": "Evidence reviewed",
            "reviewer_id": "forged",
        },
    )
    assert response.status_code == 200
    assert actions.actions[0].reviewer_id == "trusted-reviewer"
    assert actions.actions[0].override_classification == "APPROPRIATE"
    assert "Evidence reviewed" in client.get("/case/audit-001").text
    assert "/case/audit-001" not in client.get("/queue").text
    assert (
        client.post(
            "/case/audit-001/decision",
            data={
                **form,
                "action": "override",
                "override_classification": "APPROPRIATE",
                "override_reason": "Evidence reviewed",
            },
        ).status_code
        == 200
    )
    assert len(actions.actions) == 1
    assert (
        client.post(
            "/case/audit-001/decision",
            data={**form, "submission_id": str(uuid4()), "action": "agree"},
        ).status_code
        == 409
    )


def test_escalation_remains_in_queue_and_stale_row_is_rejected(dashboard):
    store, _, config = dashboard
    write(store, _audit_row())
    client = TestClient(create_app(config))
    form = Fields(client.get("/case/audit-001").text).values
    assert (
        client.post(
            "/case/audit-001/decision",
            data={**form, "action": "escalate", "note": "Senior review"},
        ).status_code
        == 200
    )
    assert "/case/audit-001" in client.get("/queue").text
    assert (
        client.post(
            "/case/audit-001/decision",
            data={**form, "row_hash": "0" * 64, "action": "agree"},
        ).status_code
        == 409
    )


def test_csrf_and_security_headers(dashboard):
    store, actions, config = dashboard
    write(store, _audit_row())
    client = TestClient(create_app(config))
    page = client.get("/case/audit-001")
    form = Fields(page.text).values
    assert (
        client.post(
            "/case/audit-001/decision",
            data={**form, "csrf_token": "bad", "action": "agree"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/case/audit-001/decision",
            data={**form, "action": "agree"},
            headers={"Origin": "https://evil.example"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/case/audit-001/break-glass", data={"justification": "urgent"}
        ).status_code
        == 403
    )
    assert not actions.actions
    assert page.headers["cache-control"] == "no-store, private"
    assert page.headers["referrer-policy"] == "same-origin"
    assert "default-src 'none'" in page.headers["content-security-policy"]
    assert "cdn.tailwindcss.com" not in page.text and "unpkg.com" not in page.text
    assert client.get("/static/dashboard.css").status_code == 200


def test_signed_authentication_and_anonymous_fail_closed(dashboard):
    from bba.dashboard.auth import sign_session

    store, _, config = dashboard
    write(store, _audit_row())
    secret = SecretStr("synthetic-secret-for-tests-at-least-32-bytes")
    config = config.model_copy(
        update={
            "default_reviewer": None,
            "auth_secret": secret,
            "public_origin": "https://review.example",
        }
    )
    client = TestClient(create_app(config), base_url="https://review.example")
    assert client.get("/queue", headers={"X-Reviewer-Id": "admin"}).status_code == 401
    token = sign_session(
        Reviewer(reviewer_id="trusted", name="Trusted", role="reviewer"),
        secret,
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    client.cookies.set("bba_session", token)
    assert client.get("/queue").status_code == 200
    client.cookies.set("bba_session", token + "x")
    assert client.get("/queue").status_code == 401


def test_version_selection_never_double_counts_or_closes_new_row(dashboard):
    store, actions, config = dashboard
    row = _audit_row()
    write(store, row)
    client = TestClient(create_app(config))
    form = Fields(client.get("/case/audit-001").text).values
    assert (
        client.post(
            "/case/audit-001/decision", data={**form, "action": "agree"}
        ).status_code
        == 200
    )
    store2 = AuditStore(
        AuditStoreConfig(root_dir=store.config.root_dir, code_version="v2")
    )
    write(store2, row.model_copy(update={"reasoning_summary_en": "new evidence"}))
    config2 = config.model_copy(update={"audit_store": store2})
    client2 = TestClient(create_app(config2))
    # Snapshot rotation is explicit; materialize a fresh view for the new version.
    for path in store.snapshots_dir.glob("*.parquet"):
        path.unlink()
    assert "/case/audit-001" in client2.get("/queue").text
    page2 = client2.get("/case/audit-001")
    assert "new evidence" in page2.text
    assert (
        client2.post(
            "/case/audit-001/decision",
            data={
                **form,
                "csrf_token": Fields(page2.text).values["csrf_token"],
                "action": "agree",
            },
        ).status_code
        == 409
    )


def test_bound_override_requires_classification():
    with pytest.raises(ValidationError):
        ReviewActionInput(
            audit_id="a",
            reviewer_id="r",
            action="override",
            override_reason="reason",
            run_id="run",
            code_version="v1",
            row_hash="a" * 64,
            expected_revision=0,
            submission_id="s",
        )
