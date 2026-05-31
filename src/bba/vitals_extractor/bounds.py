"""Sanity bounds for extracted vital signs (issue #6, PRD §4).

Bounds are inclusive on both ends. A regex hit that lies outside its bound
is discarded by the pipeline and the surviving :class:`VitalsResult` gains
the :class:`VitalsFlag.DATA_ERROR` flag.

The bounds are intentionally wider than physiologic norm so the auditor
sees genuine clinical outliers; they exist to filter OCR noise and the
occasional misparse (e.g. ``BT 384`` from a missing decimal), not to second-
guess medical judgement.
"""

from __future__ import annotations

SBP_MIN = 60
SBP_MAX = 220
DBP_MIN = 30
DBP_MAX = 150
HR_MIN = 30
HR_MAX = 200
RR_MIN = 5
RR_MAX = 50
BT_MIN = 30.0
BT_MAX = 43.0
# MAP (mean arterial pressure). Bounds bracket the physiologic span derivable
# from the SBP/DBP bounds above (~(SBP + 2*DBP)/3) with headroom, so a real
# septic-shock nadir in the 40s-50s survives while an OCR misparse like
# "MAP 8" or "MAP 250" is dropped before it can masquerade as the window nadir.
MAP_MIN = 30
MAP_MAX = 180


def is_sbp_valid(value: int) -> bool:
    """Return True iff ``value`` is within the inclusive SBP sanity bounds."""
    return SBP_MIN <= value <= SBP_MAX


def is_dbp_valid(value: int) -> bool:
    """Return True iff ``value`` is within the inclusive DBP sanity bounds."""
    return DBP_MIN <= value <= DBP_MAX


def is_hr_valid(value: int) -> bool:
    """Return True iff ``value`` is within the inclusive HR sanity bounds."""
    return HR_MIN <= value <= HR_MAX


def is_rr_valid(value: int) -> bool:
    """Return True iff ``value`` is within the inclusive RR sanity bounds."""
    return RR_MIN <= value <= RR_MAX


def is_bt_valid(value: float) -> bool:
    """Return True iff ``value`` is within the inclusive BT sanity bounds (deg C)."""
    return BT_MIN <= value <= BT_MAX


def is_map_valid(value: int) -> bool:
    """Return True iff ``value`` is within the inclusive MAP sanity bounds (mmHg)."""
    return MAP_MIN <= value <= MAP_MAX
