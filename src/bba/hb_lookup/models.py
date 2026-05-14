"""Pydantic v2 models for the hb_lookup module.

All models are immutable (``frozen=True``). Constructors enforce invariants —
``HbObservation`` rejects values outside [2, 25] g/dL and naive datetimes — so
any instance is, by construction, valid input or output for ``lookup_hb``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# LABEXM 290095 (HEMATOLOGY) is preferred over 500001 (POCT) per PRD §3.
HbSource = Literal["HEMATOLOGY", "POCT"]

# Freshness tier boundaries (relative to order anchor):
#   <24h         → fresh
#   [24h, 72h)   → stale_24_72h
#   [72h, 7d)    → stale_3_7d
#   >=7d / none  → missing
HbFreshness = Literal["fresh", "stale_24_72h", "stale_3_7d", "missing"]


class HbObservation(BaseModel):
    """One validated Hb result from the Lab table.

    Numeric range [2, 25] g/dL is the analytic-validity window per PRD §3 —
    anything outside is a transcription or unit error rather than a real
    measurement. ``item_no`` is the Lab table's row identifier and acts as
    the tie-breaker when two observations share an exact datetime.
    """

    model_config = ConfigDict(frozen=True)

    value_g_dl: float = Field(ge=2.0, le=25.0)
    datetime_utc: datetime
    source: HbSource
    item_no: int

    @field_validator("datetime_utc")
    @classmethod
    def _datetime_must_be_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("datetime_utc must be tz-aware (UTC)")
        return v


class DeltaHbWindow(BaseModel):
    """Per-window delta-Hb threshold result.

    ``triggered`` is ``True`` iff a prior Hb was found in the window and the
    drop from that prior to the current observation is at or above
    ``threshold_g_dl``. The "prior" used to compute ``drop_g_dl`` is the
    *highest* Hb in the window — a bigger drop is the more conservative
    bleed signal, and ``prior_value_g_dl`` / ``prior_datetime_utc`` carry
    that reference observation so a reviewer can audit which pair fired
    the threshold.
    """

    model_config = ConfigDict(frozen=True)

    window_hours: int
    threshold_g_dl: float
    prior_value_g_dl: float | None
    prior_datetime_utc: datetime | None
    drop_g_dl: float | None
    triggered: bool


class HbLookupResult(BaseModel):
    """Result of looking up the most-recent Hb before an order anchor.

    Invariants:

    * If ``freshness == "missing"``: ``value_g_dl``, ``datetime_utc``, and
      ``source`` are all ``None``; ``delta_hb_bypass`` is ``False``.
    * ``delta_hb_windows`` always has length 3 — one entry per (6h, 12h, 24h)
      in that order. A tuple, not a list, so the public contract is
      genuinely immutable.
    * ``delta_hb_bypass`` is ``True`` iff at least one window is triggered.
    * ``needs_review_single_low_hb`` is ``True`` only when the most-recent Hb
      is < 8 g/dL and no prior observation exists in the 24h-window before it
      — i.e., we have a worrying value but no trend to interpret it against.
    """

    model_config = ConfigDict(frozen=True)

    value_g_dl: float | None
    datetime_utc: datetime | None
    source: HbSource | None
    freshness: HbFreshness
    delta_hb_bypass: bool
    delta_hb_windows: tuple[DeltaHbWindow, ...]
    needs_review_single_low_hb: bool
