from pathlib import Path

from tempfile import TemporaryDirectory
from testcontainers.postgres import PostgresContainer
import psycopg
import uvicorn
from test_dashboard import _audit_row, _llm_call
from bba.audit_store import AuditStore, AuditStoreConfig
from bba.review_actions import ReviewActionsConfig, ReviewActionsStore, apply_migrations
from bba.dashboard import DashboardConfig, Reviewer, create_app
from urllib.parse import urlparse, urlunparse

with (
    TemporaryDirectory(prefix="bba-ui-") as raw,
    PostgresContainer("postgres:16-alpine") as pg,
):
    dsn = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")
    apply_migrations(ReviewActionsConfig(dsn=dsn))
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "ALTER ROLE review_actions_app WITH LOGIN PASSWORD 'synthetic-ui-password'"
        )
    parsed = urlparse(dsn)
    app_dsn = urlunparse(
        parsed._replace(
            netloc=f"review_actions_app:synthetic-ui-password@{parsed.hostname}:{parsed.port}"
        )
    )
    actions = ReviewActionsStore(ReviewActionsConfig(dsn=app_dsn))
    store = AuditStore(
        AuditStoreConfig(root_dir=Path(raw) / "audit", code_version="ui-v1")
    )
    for aid in ("audit-rbc", "audit-plt"):
        row = _audit_row(audit_id=aid).model_copy(
            update={
                "indications_json": (
                    {
                        "quote": "Synthetic evidence supports senior review",
                        "source_id": "E1",
                    },
                )
            }
        )
        if aid.endswith("plt"):
            row = row.model_copy(
                update={
                    "component": "platelet",
                    "platelet_value": 15.0,
                    "platelet_freshness": "fresh",
                    "hb_freshness": "missing",
                }
            )
        store.write(row, (_llm_call(audit_id=aid),))
    app = create_app(
        DashboardConfig(
            audit_store=store,
            review_actions_store=actions,
            snapshot_dir=Path(raw) / "snap",
            code_version="ui-v1",
            run_id="run-001",
            default_reviewer=Reviewer(
                reviewer_id="synthetic-reviewer",
                name="Synthetic Reviewer",
                role="reviewer",
            ),
        )
    )
    print("SYNTHETIC_UI_READY", flush=True)
    try:
        uvicorn.run(app, host="127.0.0.1", port=8766, log_level="warning")
    finally:
        actions.close()
