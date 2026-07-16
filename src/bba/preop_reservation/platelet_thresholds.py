"""Clinician-signed platelet reservation thresholds and procedure categories.

The KCMH Transfusion Committee signed off the pre-op platelet reservation
thresholds, the transfusion->reservation reinterpretation, and the
procedure->category mapping (worksheet T4 / issue #166). The values below are
the SIGNED decisions — no longer SEED. Each category applies a single count
cutoff (no gray band, per signed Open question A-i): a reservation is
APPROPRIATE when the pre-op count is at/below the cutoff and OVER when it is
strictly above. The feature stays behind the default-OFF
``MSBOS_RESERVATION_ENABLED`` flag; enabling it is a separate deliberate
go-live step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum


class PlateletCategory(StrEnum):
    """Signed reservation categories, each with a single count cutoff."""

    MAJOR_NON_NEURAXIAL = "major_non_neuraxial"
    CARDIAC_CPB = "cardiac_cpb"
    NEURAXIAL = "neuraxial"


# Clinician-signed (KCMH Transfusion Committee worksheet, T4/#166), Section A1
# "Major non-neuraxial surgery": reserving is over when count > 80,000/uL.
MAJOR_NON_NEURAXIAL_OVER_ABOVE_PER_UL = 80_000
# Clinician-signed (worksheet T4/#166), Section B-i: the merged cardiothoracic
# group is APPROPRIATE at count <= 100,000/uL and over above it (the committee
# declined to apply "any reserved unit = over" to the un-splittable group).
CARDIAC_CPB_OVER_ABOVE_PER_UL = 100_000
# Clinician-signed (worksheet T4/#166), Section D: neuraxial reservation is
# APPROPRIATE at count <= 100,000/uL and over above it.
NEURAXIAL_OVER_ABOVE_PER_UL = 100_000

# Signed thresholds transcribed for citation/reference only — these categories
# are transfusion-threshold buckets, not surgical procedure_groups, so they are
# NOT reachable through the procedure->category mapping (Section A-ii marked
# "in scope", but the vendored reference exposes no LP/CVC/consumptive group to
# key them to). Kept as documented constants so a later data update can wire them.
MAJOR_NON_NEURAXIAL_HIGH_RISK_OVER_ABOVE_PER_UL = 100_000  # A2 high-bleeding-risk
LP_OVER_ABOVE_PER_UL = 80_000  # A3 lumbar puncture (surgical)
CVC_OVER_ABOVE_PER_UL = 50_000  # A4 central venous catheter
CONSUMPTIVE_OVER_ABOVE_PER_UL = 10_000  # A5 consumptive thrombocytopenia / DIC


#: Signed count cutoff per category: reserving is over when count > cutoff.
CATEGORY_OVER_ABOVE_PER_UL: Mapping[PlateletCategory, int] = {
    PlateletCategory.MAJOR_NON_NEURAXIAL: MAJOR_NON_NEURAXIAL_OVER_ABOVE_PER_UL,
    PlateletCategory.CARDIAC_CPB: CARDIAC_CPB_OVER_ABOVE_PER_UL,
    PlateletCategory.NEURAXIAL: NEURAXIAL_OVER_ABOVE_PER_UL,
}


# Clinician-signed procedure_group -> platelet-category mapping (worksheet
# Section C, all 28 groups confirmed; Tumor/TR/Pediatric signed to MNS).
PROCEDURE_GROUP_TO_CATEGORY: Mapping[str, PlateletCategory] = {
    "Arthroplasty": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "C Spine": PlateletCategory.NEURAXIAL,
    "C/S": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Enucleation": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Facial plastic": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Foot": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Head-Neck": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Hysterectomy": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Hysteroscopy": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Larynx": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Metastasis spine": PlateletCategory.NEURAXIAL,
    "Myomectomy": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Oto": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Pediatric": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Rhino": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Salpingectomy": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Sleep": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Sport": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "TL spine": PlateletCategory.NEURAXIAL,
    "TR": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Trauma": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "Tumor": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "ศัลยกรรมตกแต่ง": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "ศัลยกรรมทั่วไป": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "ศัลยกรรมทางเดินปัสสาวะ": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "ศัลยกรรมระบบประสาท": PlateletCategory.NEURAXIAL,
    "ศัลยกรรมลำไส้ใหญ่และทวารหนัก": PlateletCategory.MAJOR_NON_NEURAXIAL,
    "ศัลยกรรมหัวใจและทรวงอก": PlateletCategory.CARDIAC_CPB,
}


def category_for_groups(groups: Sequence[str]) -> frozenset[PlateletCategory]:
    """Return the distinct known platelet categories for procedure groups."""
    return frozenset(
        PROCEDURE_GROUP_TO_CATEGORY[group]
        for group in groups
        if group in PROCEDURE_GROUP_TO_CATEGORY
    )


__all__ = [
    "CARDIAC_CPB_OVER_ABOVE_PER_UL",
    "CATEGORY_OVER_ABOVE_PER_UL",
    "CONSUMPTIVE_OVER_ABOVE_PER_UL",
    "CVC_OVER_ABOVE_PER_UL",
    "LP_OVER_ABOVE_PER_UL",
    "MAJOR_NON_NEURAXIAL_HIGH_RISK_OVER_ABOVE_PER_UL",
    "MAJOR_NON_NEURAXIAL_OVER_ABOVE_PER_UL",
    "NEURAXIAL_OVER_ABOVE_PER_UL",
    "PROCEDURE_GROUP_TO_CATEGORY",
    "PlateletCategory",
    "category_for_groups",
]
