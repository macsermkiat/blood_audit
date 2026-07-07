"""Pydantic v2 models + lab config for the platelet_lookup module.

Scope note: this module currently supplies the platelet lab CONFIG
(:data:`PLATELET_LABEXM`, unit, validity range) and the validated
:class:`PlateletObservation`. The recent-value SELECTION engine (source
preference, freshness tiers, trend) is shared with :mod:`bba.hb_lookup` and is
extracted into a component-parameterised core in a follow-up (docs plan §5.3
stage 2); it is intentionally NOT duplicated here.

All models are immutable (``frozen=True``); constructors enforce invariants so
any instance is valid by construction, mirroring :class:`bba.hb_lookup.HbObservation`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from bba.platelet_lookup.parse import MAX_PLATELET, MIN_PLATELET

# LABEXM 290078 = "Platelets Counts", LABGRP 29 HEMATOLOGY, unit ×10³/µL,
# reference range 150-450 (verified against the Lab dictionary 2026-07-08).
# Unlike Hb (which has a POCT fallback, LABEXM 500001), the platelet count is
# HEMATOLOGY-only in this dataset.
PLATELET_LABEXM: str = "290078"
PLATELET_UNIT: str = "x10*3 /uL"

PlateletSource = Literal["HEMATOLOGY"]
"""The lab source group for platelet counts. Single-valued today (290078 is
HEMATOLOGY-only); a ``Literal`` so an unexpected source fails loud rather than
being silently accepted."""


class PlateletObservation(BaseModel):
    """One validated platelet-count result from the Lab table.

    The numeric range [:data:`MIN_PLATELET`, :data:`MAX_PLATELET`] ×10³/µL is
    the analytic-validity window — anything outside is a transcription or unit
    error. ``item_no`` is the Lab row identifier and is the tie-breaker when
    two observations share an exact datetime (same contract as
    :class:`bba.hb_lookup.HbObservation.item_no`).
    """

    model_config = ConfigDict(frozen=True)

    value_k_ul: float = Field(ge=MIN_PLATELET, le=MAX_PLATELET)
    datetime_utc: datetime
    source: PlateletSource
    item_no: int

    @field_validator("datetime_utc")
    @classmethod
    def _datetime_must_be_utc(cls, v: datetime) -> datetime:
        # Strict-loud tz contract (mirrors HbObservation): the persisted
        # timestamp is UTC, not merely tz-aware. A Bangkok-aware datetime here
        # would silently leak a local time into the classifier.
        if v.tzinfo is None or v.utcoffset() != timedelta(0):
            raise ValueError("datetime_utc must be tz-aware UTC")
        return v


__all__: Sequence[str] = (
    "PLATELET_LABEXM",
    "PLATELET_UNIT",
    "PlateletObservation",
    "PlateletSource",
)
