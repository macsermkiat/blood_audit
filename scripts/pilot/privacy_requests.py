"""Hand-off files between the pilot leg and the privacy issuer.

The pilot writes a request file holding the exact prompt requests it will
submit (raw note text, so mode 0600). The issuer reads it and returns a
``{audit_id: PrivacyArtifact}`` map. Both files are written here so the mode is
set the same way every time.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from bba.prompt_builder import PromptBuildRequest

PREPARE_ENV = "BBA_PILOT_PRIVACY_PREPARE"
ARTIFACTS_ENV = "BBA_PRIVACY_ARTIFACTS"
HOSPITAL_TZ = ZoneInfo("Asia/Bangkok")


class PrivacyPrepareError(ValueError):
    """Prepare mode was configured in a way that cannot work."""


class PrivacyCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    anchor_date: date
    prompt_request: PromptBuildRequest


class PrivacyRequests(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    cases: dict[str, PrivacyCase]


def prepare_path(env: Mapping[str, str]) -> Path | None:
    """Return the request-file path when prepare mode is on, else None."""
    raw = env.get(PREPARE_ENV)
    if not raw:
        return None
    if env.get(ARTIFACTS_ENV):
        # Artifacts replace the raw chunks the issuer needs to redact.
        raise PrivacyPrepareError(
            f"{PREPARE_ENV} needs the raw evidence chunks; unset {ARTIFACTS_ENV}"
        )
    return Path(raw)


def bangkok_anchor(order_datetime: datetime) -> date:
    """Day 0 for date shifting: the order's calendar date in hospital time."""
    if order_datetime.tzinfo is None:
        # astimezone would silently assume the host zone and shift Day N.
        raise PrivacyPrepareError("order_datetime must be timezone-aware")
    return order_datetime.astimezone(HOSPITAL_TZ).date()


def write_private(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` readable only by the owner, even if it existed."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(text)


def write_requests(path: Path, requests: PrivacyRequests) -> None:
    write_private(path, requests.model_dump_json(indent=2))


def load_requests(path: Path) -> PrivacyRequests:
    return PrivacyRequests.model_validate_json(path.read_text(encoding="utf-8"))
