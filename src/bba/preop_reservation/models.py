"""Frozen value models for MSBOS pre-op reservation judgments."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MsbosToken = Literal["none", "G/M", "T/S"]
ReservationReason = Literal[
    "over_none",
    "over_gm_excess",
    "over_type_and_screen_crossmatched",
    "within_recommendation",
    "type_and_screen_screen_only",
    "unresolved_code",
    "ambiguous_code",
    "operation_unresolved",
    "no_planned_op",
    "ambiguous_planned_op",
]


class MsbosRow(BaseModel):
    """One resolved MSBOS recommendation for an ICD-9 procedure code."""

    model_config = ConfigDict(frozen=True)

    msbos: MsbosToken
    recommended_units: int = Field(ge=0)


class CandidateOperation(BaseModel):
    """One raw MSBOS row for a code, carrying its operation name for note disambiguation."""

    model_config = ConfigDict(frozen=True)

    operation: str
    msbos: MsbosToken
    recommended_units: int = Field(ge=0)


class PlannedOpProvenance(BaseModel):
    """Picker-v2 provenance for one planned-op selection (spec #196, T2/T3).

    Attached to a reservation decision ONLY when the picker-v2 seam produced
    the pick; ``None`` (the default) means the legacy picker ran and every
    pre-picker-v2 code path is byte-identical. ``gate`` records the verdict
    gate's ruling for a bridge-sourced pick: ``bridge_disagreement`` (the
    First-Choice and human-selected codes differ and either resolves in
    MSBOS) or ``bridge_over_unconfirmed`` (an over that lacks the
    score>=threshold + human-agreement confirmation required for a hard
    verdict); ``""`` means no gate applies (exact-ICD9 pick, confirmed
    bridge pick, or an ambiguous pick whose gate is suppressed).
    """

    model_config = ConfigDict(frozen=True)

    source_code: str = ""
    source: str = ""
    bridge_icd9: str = ""
    bridge_score: float | None = None
    human_index: str = ""
    human_agreed: bool | None = None
    human_icd9: str = ""
    pick_status: str = ""
    candidate_count: int = 0
    tie_count: int = 0
    bridge_hash: str = ""
    gate: Literal["", "bridge_disagreement", "bridge_over_unconfirmed"] = ""


class ReservationDecision(BaseModel):
    """Frozen per-order, in-run snapshot of the MSBOS reservation judgment."""

    model_config = ConfigDict(frozen=True)

    resolved_icd9: str = ""
    msbos: str = ""
    recommended_units: int = 0
    reserved_units: int = Field(default=0, ge=0)
    is_over: bool = False
    reason: ReservationReason
    reference_hash: str
    note_resolved: bool = False
    planned_op: PlannedOpProvenance | None = None


__all__ = [
    "CandidateOperation",
    "MsbosRow",
    "MsbosToken",
    "PlannedOpProvenance",
    "ReservationDecision",
    "ReservationReason",
]
