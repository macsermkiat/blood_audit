"""Validated operator settings. Configuration files contain data only."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator
from typing_extensions import Self


class ApplicationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_version: str = Field(default="repository-policy", min_length=1)
    transport: Literal["local", "cassette", "live"] = "local"
    cassette: Path | None = None
    allow_live_api: StrictBool = False
    enable_missing_hb_positive_evidence: bool = False
    enable_missing_platelet_defer: bool = False

    @model_validator(mode="after")
    def _transport_settings(self) -> Self:
        if self.transport == "cassette" and self.cassette is None:
            raise ValueError("cassette transport requires a cassette path")
        if self.transport != "cassette" and self.cassette is not None:
            raise ValueError("cassette path is only valid for cassette transport")
        if self.transport == "live" and not self.allow_live_api:
            raise ValueError("live transport requires explicit allow_live_api=true")
        if self.allow_live_api and self.transport != "live":
            raise ValueError("allow_live_api requires live transport")
        return self


def load_config(path: Path | None) -> ApplicationConfig:
    return (
        ApplicationConfig()
        if path is None
        else ApplicationConfig.model_validate_json(path.read_text(encoding="utf-8"))
    )
