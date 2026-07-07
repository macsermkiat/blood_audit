"""Contract tests for :mod:`bba.component_map` (Phase 2 intake gating prereq).

The auditor routes a BDTYPE product code to a component auditor (or excludes
it). Two failure modes matter clinically and are each pinned here:

* A platelet order mis-classified as red_cell (or vice versa) would blend
  into the wrong stats / be judged against the wrong policy. So the
  platelet↔red-cell boundary is asserted exactly, from real dictionary names.
* A dictionary drift that desyncs the fast code map from the NAME classifier
  would silently mis-route. :class:`TestCodeMapAgreesWithNameClassifier`
  fails CI if the two ever disagree on a known code.

Product NAME strings below are verbatim from the KCMH BDTYPE dictionary
(product-type descriptions, not PHI), verified 2026-07-08 against the raw
issued-units feed.
"""

from __future__ import annotations

import pytest

from bba.component_map import (
    BDTYPE_FAMILY,
    PLATELET_PRODUCTS,
    ComponentFamily,
    classify_component,
    component_of_code,
    is_platelet_product,
)

# (BDTYPE code, dictionary NAME, expected family) — every code observed in the
# raw BDVSTDT issued-units feed on 2026-07-08.
_ISSUED_CODES: tuple[tuple[str, str, ComponentFamily], ...] = (
    ("LPRC", "Leukocyte Poor PRC", ComponentFamily.RED_CELL),
    ("LDPRC", "Leukodepleted PRC", ComponentFamily.RED_CELL),
    ("LDPRCI", "Leukocyte depleted PRC + Irradiate", ComponentFamily.RED_CELL),
    ("LPRCI", "Leukocyte poor PRC + Irradiate", ComponentFamily.RED_CELL),
    ("LPRCF", "Leukocyte poor PRC  Filter", ComponentFamily.RED_CELL),
    ("SDRF", "Single donor red cells Filter unit 1", ComponentFamily.RED_CELL),
    (
        "SDRFI",
        "Single donor red cells Filter + Irradiate unit 1",
        ComponentFamily.RED_CELL,
    ),
    ("PRCF", "Packed Red Cell(PRC) Filter", ComponentFamily.RED_CELL),
    ("LDPRC4", "Leukodepleted Packed Red Cells (NAT)", ComponentFamily.RED_CELL),
    ("LDPPC", "Leukodepleted Pool Platelet Concentrates", ComponentFamily.PLATELET),
    (
        "LDPPCI",
        "Leukocyte depleted platelet conc + Irradiate",
        ComponentFamily.PLATELET,
    ),
    (
        "SDPFI",
        "Single donor platelet conc. Filter + Irradiate",
        ComponentFamily.PLATELET,
    ),
    ("SDPF", "Single donor platelet conc. Filter", ComponentFamily.PLATELET),
    (
        "SDPPI",
        "Single donor platelet with PI Psoralen-Treated",
        ComponentFamily.PLATELET,
    ),
    ("LPPC", "Pooled Leukocyte Poor Platelet Concentrates", ComponentFamily.PLATELET),
    ("PC", "Platelets Concentrates", ComponentFamily.PLATELET),
    ("LDPC", "Leukocyte Depleted Platelet Concentrates", ComponentFamily.PLATELET),
    ("FFP", "Fresh Frozen Plasma", ComponentFamily.FFP),
    ("SDFFP", "Secured Plasma Apheresis HRIG", ComponentFamily.FFP),
    ("CRP", "Cryo Removed Plasma", ComponentFamily.FFP),
    ("CPP", "Cryoprecipitate", ComponentFamily.CRYO),
    ("HTFDC", "Heat Treated Freeze Dried Cryoprecipitate", ComponentFamily.CRYO),
    ("ATX", "Autologous blood (Whole blood)", ComponentFamily.WHOLE_BLOOD),
)


class TestClassifyComponentByName:
    """``classify_component`` resolves every real product NAME to its family."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [(name, fam) for _code, name, fam in _ISSUED_CODES],
        ids=[code for code, _name, _fam in _ISSUED_CODES],
    )
    def test_real_dictionary_names_classify_correctly(
        self, name: str, expected: ComponentFamily
    ) -> None:
        assert classify_component(name) == expected

    def test_case_insensitive(self) -> None:
        # HOSxP exports are not case-normalised; a lower/upper mix must not
        # change the family.
        assert classify_component("LEUKODEPLETED POOL PLATELET CONCENTRATES") == (
            ComponentFamily.PLATELET
        )

    def test_cryo_removed_plasma_is_ffp_not_cryo(self) -> None:
        # Regression: "Cryo Removed Plasma" is the cryo-DEPLETED leftover
        # (a plasma product), not cryoprecipitate. A naive `"cryo" in name`
        # check mis-files it as CRYO.
        assert classify_component("Cryo Removed Plasma") == ComponentFamily.FFP
        assert classify_component("Cryoprecipitate") == ComponentFamily.CRYO
        # A bare / abbreviated cryo name (no full "cryoprecipitate") still
        # resolves to the cryo family, not UNKNOWN.
        assert classify_component("Cryo ppt pooled") == ComponentFamily.CRYO

    def test_unknown_name_is_not_guessed(self) -> None:
        # An unrecognised / empty name must fall back to UNKNOWN, never a
        # real family — the intake gate excludes-and-flags rather than
        # silently admitting a product to the wrong auditor.
        assert classify_component("") == ComponentFamily.UNKNOWN
        assert classify_component("Granulocyte Concentrate") == ComponentFamily.UNKNOWN


class TestCodeMapAgreesWithNameClassifier:
    """The fast code map and the NAME classifier must never disagree.

    This is the drift guard: the auditor keys off codes for speed, but the
    NAME classifier is the source of truth. If a future dictionary update
    changes a product's meaning, this test fails rather than the auditor
    silently mis-routing.
    """

    @pytest.mark.parametrize(
        ("code", "name", "expected"),
        _ISSUED_CODES,
        ids=[code for code, _name, _fam in _ISSUED_CODES],
    )
    def test_code_map_matches_name_and_expected(
        self, code: str, name: str, expected: ComponentFamily
    ) -> None:
        assert component_of_code(code) == expected
        assert classify_component(name) == expected
        assert BDTYPE_FAMILY[code] == expected

    def test_unknown_code_returns_unknown(self) -> None:
        assert component_of_code("NOTACODE") == ComponentFamily.UNKNOWN

    def test_map_is_immutable(self) -> None:
        # The shared lookup table must not be mutable at runtime (a caller
        # mutating it would silently change routing for everyone).
        with pytest.raises(TypeError):
            BDTYPE_FAMILY["LPRC"] = ComponentFamily.PLATELET  # type: ignore[index]


class TestPlateletAllowList:
    """``PLATELET_PRODUCTS`` is exactly the platelet codes and nothing else."""

    def test_allow_list_is_the_platelet_family(self) -> None:
        expected = {
            code
            for code, _name, fam in _ISSUED_CODES
            if fam is ComponentFamily.PLATELET
        }
        assert set(PLATELET_PRODUCTS) == expected

    def test_includes_irradiated_variants(self) -> None:
        # Divergence from RBC_PRODUCTS (which excludes irradiated variants):
        # an irradiated platelet is still a platelet transfusion to audit.
        assert is_platelet_product("LDPPCI")
        assert is_platelet_product("SDPFI")

    def test_rejects_red_cell_and_plasma(self) -> None:
        assert not is_platelet_product("LPRC")
        assert not is_platelet_product("FFP")
        assert not is_platelet_product("CPP")
