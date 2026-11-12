"""Signed sessions from a trusted identity adapter and session-bound CSRF tokens."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from pydantic import SecretStr, ValidationError

from bba.dashboard.models import Reviewer, RouteContext


_SESSION_COOKIE = "bba_session"


def _secret_bytes(secret: SecretStr) -> bytes:
    value = secret.get_secret_value().encode()
    if len(value) < 32:
        raise ValueError("authentication secret must contain at least 32 bytes")
    return value


def _sign(payload: dict[str, Any], secret: bytes) -> str:
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).rstrip(b"=")
    signature = hmac.new(secret, encoded, hashlib.sha256).hexdigest()
    return f"{encoded.decode()}.{signature}"


def _verify(token: str, secret: bytes, *, purpose: str) -> dict[str, Any]:
    try:
        if len(token) > 16384:
            raise ValueError("oversized credential")
        encoded, signature = token.split(".")
        expected = hmac.new(secret, encoded.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise ValueError("invalid signature")
        payload = json.loads(
            base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        )
        if not isinstance(payload, dict) or payload.get("purpose") != purpose:
            raise ValueError("invalid credential purpose")
        if (
            not isinstance(payload.get("expires"), (int, float))
            or payload["expires"] <= datetime.now(UTC).timestamp()
        ):
            raise ValueError("expired credential")
        return payload
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError("invalid or expired credential") from exc


def sign_session(reviewer: Reviewer, secret: SecretStr, *, expires_at: datetime) -> str:
    """Issue only after trusted authentication, using a dedicated shared secret.

    The identity adapter sets this as Secure, HttpOnly, SameSite=Strict with
    Path=/ and a matching expiry. This helper is never exposed as an endpoint.
    """
    if expires_at.tzinfo is None:
        raise ValueError("session expiry must be timezone-aware")
    lifetime = expires_at.timestamp() - datetime.now(UTC).timestamp()
    if not 0 < lifetime <= 3600:
        raise ValueError("session lifetime must be between zero and one hour")
    return _sign(
        {
            "purpose": "session",
            "expires": expires_at.timestamp(),
            "reviewer": reviewer.model_dump(mode="json"),
            "nonce": secrets.token_hex(24),
        },
        _secret_bytes(secret),
    )


def authenticated_context(request: Request, secret: SecretStr) -> RouteContext:
    try:
        payload = _verify(
            request.cookies.get(_SESSION_COOKIE, ""),
            _secret_bytes(secret),
            purpose="session",
        )
        reviewer = Reviewer.model_validate(payload["reviewer"])
    except (ValueError, KeyError, ValidationError) as exc:
        raise HTTPException(
            status_code=401, detail="valid reviewer session required"
        ) from exc
    return RouteContext(reviewer=reviewer)


def _session_binding(request: Request, context: RouteContext) -> str:
    session = request.cookies.get(_SESSION_COOKIE, "")
    return hashlib.sha256(
        f"{context.reviewer.reviewer_id}:{session}".encode()
    ).hexdigest()


def csrf_token(request: Request, context: RouteContext) -> str:
    return _sign(
        {
            "purpose": "csrf",
            "expires": datetime.now(UTC).timestamp() + 3600,
            "session": _session_binding(request, context),
        },
        request.app.state.csrf_secret,
    )


def check_csrf(request: Request, context: RouteContext, token: str) -> None:
    config = request.app.state.dashboard_config
    expected_origin = config.public_origin or str(request.base_url).rstrip("/")
    origin = request.headers.get("origin")
    if origin is not None and origin != expected_origin:
        raise HTTPException(
            status_code=403, detail="cross-origin form submission refused"
        )
    if request.headers.get("sec-fetch-site") == "cross-site":
        raise HTTPException(
            status_code=403, detail="cross-site form submission refused"
        )
    try:
        payload = _verify(token, request.app.state.csrf_secret, purpose="csrf")
        if not hmac.compare_digest(
            payload.get("session", ""), _session_binding(request, context)
        ):
            raise ValueError("session mismatch")
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=403, detail="invalid or expired CSRF token; reload the form"
        ) from exc


def validate_origin(origin: str) -> str:
    parsed = urlsplit(origin)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "public_origin must be an HTTPS origin without path or credentials"
        )
    return origin
