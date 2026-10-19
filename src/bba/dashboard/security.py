"""Local assets, CSRF key, and response protection for reviewer pages."""

from __future__ import annotations

import secrets
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

from bba.dashboard.auth import _secret_bytes
from bba.dashboard.models import DashboardConfig


def configure_security(app: FastAPI, config: DashboardConfig) -> None:
    app.state.csrf_secret = (
        _secret_bytes(config.auth_secret)
        if config.auth_secret
        else secrets.token_bytes(32)
    )
    app.mount(
        "/static",
        StaticFiles(directory=str(Path(__file__).parent / "static")),
        name="static",
    )

    @app.middleware("http")
    async def _security_headers(
        request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        response.headers.update(
            {
                "Cache-Control": "no-store, private",
                "Content-Security-Policy": "default-src 'none'; style-src 'self'; script-src 'self'; img-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'",
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "Referrer-Policy": "same-origin",
                "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
            }
        )
        if config.public_origin:
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.get("/", include_in_schema=False)
    def _root_route() -> RedirectResponse:
        return RedirectResponse("/queue")
