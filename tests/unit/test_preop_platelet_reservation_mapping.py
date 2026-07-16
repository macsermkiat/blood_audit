"""Vendored platelet procedure-category SEED tests for ticket #166."""

from __future__ import annotations

import csv
from importlib import resources

from bba.preop_reservation import (
    CATEGORY_SEED_STATUS,
    PROCEDURE_GROUP_TO_CATEGORY,
    PlateletCategory,
    SeedStatus,
    category_for_groups,
)
from bba.preop_reservation.reference import MSBOS_REFERENCE_FILENAME


def test_mapping_exactly_covers_all_vendored_procedure_groups() -> None:
    reference_path = resources.files("bba.preop_reservation").joinpath(
        "data", MSBOS_REFERENCE_FILENAME
    )
    with reference_path.open("r", encoding="utf-8-sig", newline="") as handle:
        csv_groups = {
            row["procedure_group"].strip()
            for row in csv.DictReader(handle)
            if row["procedure_group"].strip()
        }

    assert set(PROCEDURE_GROUP_TO_CATEGORY) == csv_groups
    assert len(csv_groups) == 28


def test_category_seed_statuses_keep_unsigned_routes_in_review() -> None:
    assert CATEGORY_SEED_STATUS == {
        PlateletCategory.MAJOR_NON_NEURAXIAL: SeedStatus.RESOLVED,
        PlateletCategory.CARDIAC_CPB: SeedStatus.UNRESOLVED_ROUTE_REVIEW,
        PlateletCategory.NEURAXIAL: SeedStatus.UNRESOLVED_ROUTE_REVIEW,
        PlateletCategory.UNCATEGORISED: SeedStatus.UNRESOLVED_ROUTE_REVIEW,
    }


def test_category_for_groups_deduplicates_categories_and_ignores_unknowns() -> None:
    assert category_for_groups(()) == frozenset()
    assert category_for_groups(("not-vendored",)) == frozenset()
    assert category_for_groups(("Head-Neck", "Head-Neck", "Larynx")) == frozenset(
        {PlateletCategory.MAJOR_NON_NEURAXIAL}
    )
    assert category_for_groups(("Head-Neck", "C Spine")) == frozenset(
        {PlateletCategory.MAJOR_NON_NEURAXIAL, PlateletCategory.NEURAXIAL}
    )
