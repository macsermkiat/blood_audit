"""Vendored platelet procedure-category tests for ticket #166 (clinician-signed)."""

from __future__ import annotations

import csv
from importlib import resources

from bba.preop_reservation import (
    CATEGORY_OVER_ABOVE_PER_UL,
    PROCEDURE_GROUP_TO_CATEGORY,
    PlateletCategory,
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


def test_tumor_tr_pediatric_are_signed_to_major_non_neuraxial() -> None:
    # Signed Section C: the three previously-uncategorised groups are MNS.
    for group in ("Tumor", "TR", "Pediatric"):
        assert (
            PROCEDURE_GROUP_TO_CATEGORY[group] is PlateletCategory.MAJOR_NON_NEURAXIAL
        )


def test_every_category_has_a_signed_cutoff() -> None:
    # Every category the mapping can produce must have a numeric cutoff so the
    # evaluator never reaches an unresolved category.
    for category in set(PROCEDURE_GROUP_TO_CATEGORY.values()):
        assert category in CATEGORY_OVER_ABOVE_PER_UL
    assert CATEGORY_OVER_ABOVE_PER_UL == {
        PlateletCategory.MAJOR_NON_NEURAXIAL: 80_000,
        PlateletCategory.CARDIAC_CPB: 100_000,
        PlateletCategory.NEURAXIAL: 100_000,
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
