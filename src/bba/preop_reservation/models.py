"""Frozen value models for MSBOS pre-op reservation judgments."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MsbosToken = Literal["none", "G/M", "T/S"]
ReservationReason = Literal[
    "over_none",
    "over_gm_excess",
    "within_recommendation",
    "type_and_screen_deferred",
    "unresolved_code",
    "ambiguous_code",
    "no_planned_op",
    "ambiguous_planned_op",
]


class MsbosRow(BaseModel):
    """One resolved MSBOS recommendation for an ICD-9 procedure code."""

    model_config = ConfigDict(frozen=True)

    msbos: MsbosToken
    recommended_units: int = Field(ge=0)


class ReservationDecision(BaseModel):
    """Durable per-order snapshot of the MSBOS reservation judgment."""

    model_config = ConfigDict(frozen=True)

    resolved_icd9: str = ""
    msbos: str = ""
    recommended_units: int = 0
    reserved_units: int = Field(default=0, ge=0)
    is_over: bool = False
    reason: ReservationReason
    reference_hash: str


__all__ = [
    "MsbosRow",
    "MsbosToken",
    "ReservationDecision",
    "ReservationReason",
]
