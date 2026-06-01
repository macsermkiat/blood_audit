"""RED-phase failing tests for the clinical-salience ranker (issue #76).

The ranker is a PURE category-to-bucket mapping (issue #76 User Story 13):
given a MED drug string, return a static salience bucket used ONLY as a
truncation sort key. It exists because Case 2 / REQNO 68012352 shipped the LLM
a medication list "reduced to saline / irrigation / omeprazole" — the
char-cap dropped items in arrival order, so a vasopressor could be shed while
maintenance fluids survived.

Lower bucket value = emitted earlier = survives tail-drop longer. The binding
assertion (Testing Decision #2) is the Case 2 shape: a norepinephrine order
must outrank a saline flush, so the pressor survives cap pressure while the
flush is dropped.

BINDING GUARDRAIL under test: salience is an ORDERING signal only. It is never
a clinical verdict and must never gate or weight a transfusion decision. The
ambiguity guardrail mirrored from the hemodynamic scan also holds here — a
clinical abbreviation like ``NAD`` (no acute distress) or ``Na`` (sodium) must
never be promoted to the critical (vasopressor) bucket.

These are pure-function tests with no bundle scaffolding (the module-level
import doubles as the public-API surface check).
"""

from __future__ import annotations

import pytest

from bba.evidence_bundle_builder.salience import (
    SalienceBucket,
    med_salience,
)


# =============================================================================
# AC: Buckets are ordered so that "lower = survives truncation longer".
# =============================================================================
class TestBucketOrdering:
    def test_critical_outranks_routine_outranks_maintenance(self) -> None:
        # The whole point of the ranker is this strict order: emission sorts
        # ascending, tail-drop removes the LAST item, so CRITICAL must compare
        # less than ROUTINE which must compare less than MAINTENANCE.
        assert SalienceBucket.CRITICAL < SalienceBucket.ROUTINE
        assert SalienceBucket.ROUTINE < SalienceBucket.MAINTENANCE

    def test_buckets_are_integer_sort_keys(self) -> None:
        # Used directly as a sort key, so the values must be plain ints.
        assert int(SalienceBucket.CRITICAL) == 0
        assert int(SalienceBucket.ROUTINE) == 1
        assert int(SalienceBucket.MAINTENANCE) == 2


# =============================================================================
# AC: Vasopressors / inotropes are CRITICAL (issue US3, US8).
# =============================================================================
class TestVasopressorsAreCritical:
    @pytest.mark.parametrize(
        "drug",
        [
            "Norepinephrine 4mg/250mL",
            "norepinephrine",
            "Levophed",
            "Noradrenaline 8mg",
            "Epinephrine 1mg",
            "Adrenaline",
            "Dopamine 200mg/250mL",
            "Dobutamine",
            "Vasopressin 20 units",
            "Phenylephrine 10mg",
            "Milrinone",
        ],
    )
    def test_pressor_or_inotrope_is_critical(self, drug: str) -> None:
        assert med_salience(drug) is SalienceBucket.CRITICAL


# =============================================================================
# AC: Blood products are CRITICAL (issue US8 — survive ahead of crystalloids).
# =============================================================================
class TestBloodProductsAreCritical:
    @pytest.mark.parametrize(
        "drug",
        [
            "LPRC 1 unit",
            "PRC",
            "Packed red cells",
            "FFP 2 units",
            "Fresh frozen plasma",
            "Cryoprecipitate",
            "Platelet concentrate",
            "Platelets",
        ],
    )
    def test_blood_product_is_critical(self, drug: str) -> None:
        assert med_salience(drug) is SalienceBucket.CRITICAL


# =============================================================================
# AC: Crystalloids / saline / flushes / irrigation are MAINTENANCE (drop first).
# =============================================================================
class TestMaintenanceFluids:
    @pytest.mark.parametrize(
        "drug",
        [
            "0.9% NSS 1000mL",
            "Normal saline flush",
            "0.9% NaCl",
            "NSS flush",
            "RLS 1000mL",
            "Lactated Ringer's",
            "Acetar",
            "Plasma-Lyte",
            "5% Dextrose",
            "D5W",
            "Sterile water for irrigation",
            "Irrigation fluid",
        ],
    )
    def test_maintenance_fluid_is_maintenance(self, drug: str) -> None:
        assert med_salience(drug) is SalienceBucket.MAINTENANCE


# =============================================================================
# AC: Everything else is ROUTINE (the default bucket).
# =============================================================================
class TestRoutineDefault:
    @pytest.mark.parametrize(
        "drug",
        [
            "Omeprazole 40mg IV",
            "Furosemide 20mg IV",
            "Paracetamol 500mg",
            "Ceftriaxone 2g",
            "Insulin aspart",
            "Some unrecognised compound",
        ],
    )
    def test_unclassified_drug_is_routine(self, drug: str) -> None:
        assert med_salience(drug) is SalienceBucket.ROUTINE


# =============================================================================
# AC (BINDING): the Case 2 shape — norepinephrine survives, saline flush drops.
# =============================================================================
class TestCase2Shape:
    def test_norepinephrine_outranks_saline_flush(self) -> None:
        # The exact Case 2 inversion the ranker exists to prevent: under cap
        # pressure the med list collapsed to saline/irrigation/omeprazole and
        # the pressor was shed. A lower bucket survives tail-drop, so the
        # pressor MUST compare strictly less than the flush.
        assert med_salience("Norepinephrine 4mg/250mL D5W") < med_salience(
            "0.9% NSS flush"
        )

    def test_pressor_diluted_in_saline_is_still_critical(self) -> None:
        # A pressor charted as an infusion in saline must not be demoted to
        # maintenance by the saline carrier — the drug is the signal.
        assert med_salience("Norepinephrine in 0.9% NSS") is SalienceBucket.CRITICAL


# =============================================================================
# AC (BINDING clinical-safety): ambiguous abbreviations are never CRITICAL.
# =============================================================================
class TestAmbiguityGuardrail:
    @pytest.mark.parametrize("drug", ["NAD", "Na", "N/A", "NaCl 0.45%"])
    def test_ambiguous_token_is_not_critical(self, drug: str) -> None:
        # NAD = no acute distress, Na = sodium, N/A = not applicable. Promoting
        # any to the vasopressor bucket would fabricate pressor support exactly
        # as it would in the hemodynamic scan. None may be CRITICAL.
        assert med_salience(drug) is not SalienceBucket.CRITICAL
