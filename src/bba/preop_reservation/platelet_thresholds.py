"""Vendored SEED thresholds and procedure categories for platelet reservation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum


class PlateletCategory(StrEnum):
    """Reservation categories represented by the surgical reference."""

    MAJOR_NON_NEURAXIAL = "major_non_neuraxial"
    CARDIAC_CPB = "cardiac_cpb"
    NEURAXIAL = "neuraxial"
    UNCATEGORISED = "uncategorised"


class SeedStatus(StrEnum):
    """Whether a category can produce a numeric reservation judgment."""

    RESOLVED = "resolved"
    UNRESOLVED_ROUTE_REVIEW = "unresolved_route_review"


# SEED — pending clinician sign-off (worksheet T4/#166), DRAFT §4 major non-neuraxial transfuse-if-count-below clause.
MNS_THRESHOLD_PER_UL = 80_000
# SEED — pending clinician sign-off (worksheet T4/#166), DRAFT §4 high-bleeding-risk ceiling clause.
MNS_HIGH_RISK_CEILING_PER_UL = 100_000
# SEED — pending clinician sign-off (worksheet T4/#166), DRAFT §4 LP surgical transfuse-if-count-below clause; citation only.
LP_THRESHOLD_PER_UL = 80_000
# SEED — pending clinician sign-off (worksheet T4/#166), DRAFT §4 CVC transfuse-if-count-below clause; citation only.
CVC_THRESHOLD_PER_UL = 50_000
# SEED — pending clinician sign-off (worksheet T4/#166), DRAFT §4 consumptive/DIC no-bleed transfuse-if-count-below clause; citation only.
CONSUMPTIVE_THRESHOLD_PER_UL = 10_000


PROCEDURE_GROUP_TO_CATEGORY: Mapping[str, PlateletCategory] = {
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Arthroplasty.
    "Arthroplasty": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row C Spine.
    "C Spine": PlateletCategory.NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row C/S.
    "C/S": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Enucleation.
    "Enucleation": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Facial plastic.
    "Facial plastic": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Foot.
    "Foot": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Head-Neck.
    "Head-Neck": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Hysterectomy.
    "Hysterectomy": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Hysteroscopy.
    "Hysteroscopy": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Larynx.
    "Larynx": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Metastasis spine.
    "Metastasis spine": PlateletCategory.NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Myomectomy.
    "Myomectomy": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Oto.
    "Oto": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Pediatric.
    "Pediatric": PlateletCategory.UNCATEGORISED,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Rhino.
    "Rhino": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Salpingectomy.
    "Salpingectomy": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Sleep.
    "Sleep": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Sport.
    "Sport": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row TL spine.
    "TL spine": PlateletCategory.NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row TR.
    "TR": PlateletCategory.UNCATEGORISED,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Trauma.
    "Trauma": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row Tumor.
    "Tumor": PlateletCategory.UNCATEGORISED,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row ศัลยกรรมตกแต่ง.
    "ศัลยกรรมตกแต่ง": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row ศัลยกรรมทั่วไป.
    "ศัลยกรรมทั่วไป": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row ศัลยกรรมทางเดินปัสสาวะ.
    "ศัลยกรรมทางเดินปัสสาวะ": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row ศัลยกรรมระบบประสาท.
    "ศัลยกรรมระบบประสาท": PlateletCategory.NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row ศัลยกรรมลำไส้ใหญ่และทวารหนัก.
    "ศัลยกรรมลำไส้ใหญ่และทวารหนัก": PlateletCategory.MAJOR_NON_NEURAXIAL,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C row ศัลยกรรมหัวใจและทรวงอก.
    "ศัลยกรรมหัวใจและทรวงอก": PlateletCategory.CARDIAC_CPB,
}


CATEGORY_SEED_STATUS: Mapping[PlateletCategory, SeedStatus] = {
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT §4 major non-neuraxial route is numerically resolved.
    PlateletCategory.MAJOR_NON_NEURAXIAL: SeedStatus.RESOLVED,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT Open B-i merged cardiac/thoracic route remains unresolved.
    PlateletCategory.CARDIAC_CPB: SeedStatus.UNRESOLVED_ROUTE_REVIEW,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT Open D supplies no neuraxial reservation rule.
    PlateletCategory.NEURAXIAL: SeedStatus.UNRESOLVED_ROUTE_REVIEW,
    # SEED — pending clinician sign-off (worksheet T4/#166), DRAFT worksheet Section C leaves uncategorised groups unresolved.
    PlateletCategory.UNCATEGORISED: SeedStatus.UNRESOLVED_ROUTE_REVIEW,
}


def category_for_groups(groups: Sequence[str]) -> frozenset[PlateletCategory]:
    """Return the distinct known platelet categories for procedure groups."""
    return frozenset(
        PROCEDURE_GROUP_TO_CATEGORY[group]
        for group in groups
        if group in PROCEDURE_GROUP_TO_CATEGORY
    )


__all__ = [
    "CATEGORY_SEED_STATUS",
    "CONSUMPTIVE_THRESHOLD_PER_UL",
    "CVC_THRESHOLD_PER_UL",
    "LP_THRESHOLD_PER_UL",
    "MNS_HIGH_RISK_CEILING_PER_UL",
    "MNS_THRESHOLD_PER_UL",
    "PROCEDURE_GROUP_TO_CATEGORY",
    "PlateletCategory",
    "SeedStatus",
    "category_for_groups",
]
