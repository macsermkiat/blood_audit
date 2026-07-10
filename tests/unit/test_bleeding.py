"""Unit tests for :mod:`bba.audit_pipeline.bleeding` (ticket #90 / spec #89).

The bleeding module is the *only* place the RBC over-clear guardrail is
allowed to trust the LLM's free-text prose: an active-bleeding indication
keeps a gray-zone order ``APPROPRIATE`` instead of being asserted
``INAPPROPRIATE``. Because that trust can flip a clinical verdict, every
test states WHY the behaviour matters (which clinical intent it protects),
not merely what the code does — a 2x2 cm gauze bleed must never clear an
order, a documented 1100 mL loss must.

Deep-module contract (issue #89 "Testing Decisions"): this pure module
carries the >=80% unit-coverage bar on its own.
"""

from __future__ import annotations

import pytest

from bba.audit_pipeline.bleeding import (
    _MAX_SCAN_CHARS,
    LLM_OVERCLEAR_MIN_BLEED_CONFIDENCE,
    LLM_OVERCLEAR_MIN_BLEED_ML,
    has_life_threatening_marker,
    parse_max_volume_ml,
    qualified_bleeding_exempt,
)


def _active_bleed(
    *,
    quote: str,
    confidence: float,
    code: str = "ACTIVE_BLEEDING_GI",
) -> dict[str, object]:
    """A schema-valid active-bleeding indication dict.

    Mirrors the tool-use ``indications[]`` shape
    (``code`` / ``quote`` / ``source_id`` / ``confidence``) that
    ``_indications_from_result`` hands the guardrail.
    """
    return {
        "code": code,
        "quote": quote,
        "source_id": "note_1",
        "confidence": confidence,
    }


class TestParseMaxVolumeMl:
    """``parse_max_volume_ml`` reads a millilitre figure from prose.

    WHY: the 300 mL bleed threshold can only bite if the parser reads the
    documented volume out of English and Thai notes, scales litres, and
    tolerates comma thousands — otherwise a genuine major bleed is missed
    and a legitimate transfusion is mislabelled INAPPROPRIATE.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("EBL 500 ml", 500.0),
            ("drain output 300 mL", 300.0),
            ("suction 250 cc", 250.0),
            ("blood loss 12.5 mL", 12.5),
        ],
    )
    def test_english_units_scale_to_ml(self, text: str, expected: float) -> None:
        assert parse_max_volume_ml(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("massive transfusion 1.5 L", 1500.0),
            ("estimated 2 liters lost", 2000.0),
            ("1 liter drained", 1000.0),
        ],
    )
    def test_litres_scale_times_1000(self, text: str, expected: float) -> None:
        assert parse_max_volume_ml(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("EBL 1,100 mL intra-op", 1100.0),
            ("chest tube 1,050 cc", 1050.0),
        ],
    )
    def test_comma_thousands_are_stripped(self, text: str, expected: float) -> None:
        assert parse_max_volume_ml(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("เสียเลือด 1100 มล.", 1100.0),
            ("เสียเลือด 800 มล", 800.0),
            ("ระบายออก 250 ซีซี", 250.0),
        ],
    )
    def test_thai_units(self, text: str, expected: float) -> None:
        assert parse_max_volume_ml(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("gauze,500 mL", 500.0),  # comma delimiter, no trailing space
            ("output.1200 mL", 1200.0),  # period delimiter, no trailing space
            ("loss,1,100 mL", 1100.0),  # delimiter comma + grouping comma
        ],
    )
    def test_punctuation_delimited_volumes_still_parse(
        self, text: str, expected: float
    ) -> None:
        # A comma/period used as a plain delimiter (preceded by a non-digit)
        # must not block a genuine volume — the number-boundary lookbehinds
        # only reject mid-number continuations.
        assert parse_max_volume_ml(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1,100 mL", 1100.0),  # grouping comma is not a new number
            ("12.5 mL", 12.5),  # decimal point is not a new number
        ],
    )
    def test_grouping_and_decimals_are_not_split(
        self, text: str, expected: float
    ) -> None:
        assert parse_max_volume_ml(text) == expected

    def test_long_number_is_read_whole_not_restarted_midway(self) -> None:
        # A number longer than any bound must be read as ONE value, never
        # restarted mid-digits into a spurious sub-number. Here a sub-microlitre
        # volume must stay tiny, not be misread as a qualifying "350".
        assert parse_max_volume_ml("0.0000000000000350 mL") < 1.0

    def test_returns_the_maximum_of_several_matches(self) -> None:
        # A quote can mention a small dressing loss and a large drain output;
        # the guardrail must judge the worst documented bleed, not the first.
        assert parse_max_volume_ml("gauze 20 mL; then drain 450 mL") == 450.0

    def test_maximum_compares_across_units(self) -> None:
        assert parse_max_volume_ml("0.5 L and 200 mL") == 500.0

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "ecchymosis over the left flank",
            "Hb 7.0 g/dL, no active bleeding",
            "2x2 cm gauze staining",
        ],
    )
    def test_no_volume_returns_none(self, text: str) -> None:
        assert parse_max_volume_ml(text) is None

    def test_hb_units_do_not_false_match_as_volume(self) -> None:
        # "9.4 g/dL" must not be read as a 9.4 mL bleed.
        assert parse_max_volume_ml("Hb 9.4 g/dL") is None

    def test_scan_length_is_bounded(self) -> None:
        # Replay-critical path: an unbounded quote length must not stall the
        # guardrail. A volume within the cap is found even behind a huge
        # benign suffix; a volume only present past the cap is not scanned.
        within = "drain 450 mL" + "x" * (_MAX_SCAN_CHARS * 2)
        assert parse_max_volume_ml(within) == 450.0
        beyond = "x" * (_MAX_SCAN_CHARS + 50) + " 900 mL"
        assert parse_max_volume_ml(beyond) is None


class TestHasLifeThreateningMarker:
    """``has_life_threatening_marker`` is a conservative keyword scan.

    WHY: an un-quantified bleed can still clear an order, but only when the
    prose explicitly names a life-threatening / uncontrolled event. The scan
    must fire on those and stay silent on benign bleeding language, because
    it is deliberate high-trust prose gated only by the confidence floor.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "life-threatening hemorrhage in OR",
            "life threatening bleed",
            "uncontrolled bleeding from the stump",
            "exsanguinating pelvic injury",
            "ongoing exsanguination",
            "hemorrhagic shock on arrival",
            "haemorrhagic shock",
            "active hemorrhage noted",
            "active haemorrhage",
            "UNCONTROLLED HEMORRHAGE",
        ],
    )
    def test_flags_explicit_markers(self, text: str) -> None:
        assert has_life_threatening_marker(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "เลือดออกไม่หยุด",  # uncontrolled / unstoppable bleeding
            "ช็อกจากการเสียเลือด",  # hemorrhagic shock
        ],
    )
    def test_flags_thai_marker(self, text: str) -> None:
        # Thai prose is a first-class input; both module markers must fire.
        assert has_life_threatening_marker(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "bleeding precaution in place",
            "minor oozing at the site",
            "ecchymosis noted",
            "2x2 cm gauze staining",
            "blood-tinged drain fluid",
        ],
    )
    def test_benign_language_does_not_flag(self, text: str) -> None:
        assert has_life_threatening_marker(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "no active hemorrhage",
            "denies uncontrolled bleeding",
            "without active haemorrhage on exam",
            "no evidence of hemorrhagic shock",
            "not exsanguinating",
            "ไม่มีเลือดออกไม่หยุด",  # "no unstoppable bleeding"
            "ไม่พบ hemorrhagic shock",  # "hemorrhagic shock not found"
        ],
    )
    def test_negated_markers_do_not_flag(self, text: str) -> None:
        # Codex PR #97 P2: a negated marker documents the ABSENCE of the
        # emergency. A raw substring hit on "no active hemorrhage" would let a
        # mislabeled high-confidence quote exempt an over-clear — the exact
        # negated-evidence scenario the guardrail exists to catch.
        assert has_life_threatening_marker(text) is False

    def test_negation_does_not_leak_across_clause_boundary(self) -> None:
        # The lookback is clause-bounded: an unrelated negation earlier in the
        # sentence must not suppress a genuine marker after a boundary.
        assert (
            has_life_threatening_marker(
                "no fever today; active hemorrhage from varices"
            )
            is True
        )

    @pytest.mark.parametrize(
        "text",
        [
            "active hemorrhage is not present",
            "uncontrolled bleeding denied",
            "hemorrhagic shock ruled out",
            "active haemorrhage: absent",
            "exsanguination no longer suspected",
        ],
    )
    def test_post_marker_negations_do_not_flag(self, text: str) -> None:
        # Codex PR #97 round 2: the emergency can be negated AFTER the marker
        # in the same clause; a pre-marker-only screen would still exempt.
        assert has_life_threatening_marker(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "uncontrolled pain despite morphine",
            "uncontrolled agitation, RASS +3",
            "uncontrolled DM on insulin",
            "life-threatening arrhythmia",
            "life threatening sepsis",
        ],
    )
    def test_non_bleeding_uncontrolled_context_does_not_flag(self, text: str) -> None:
        # Codex PR #97 round 4: "uncontrolled" / "life-threatening" are
        # generic clinical intensifiers. Without bleeding context in the same
        # clause, a mislabeled ACTIVE_BLEEDING quote about pain/agitation/DM
        # must not exempt an over-clear.
        assert has_life_threatening_marker(text) is False

    def test_bleeding_context_before_generic_marker_flags(self) -> None:
        assert has_life_threatening_marker("bleeding uncontrolled despite packing")

    def test_post_marker_negation_does_not_leak_across_boundary(self) -> None:
        # A negation in the NEXT clause is about something else; the marker
        # itself stands.
        assert (
            has_life_threatening_marker("active hemorrhage ongoing; wound not infected")
            is True
        )


class TestQualifiedBleedingExempt:
    """The exemption predicate: does a genuine major bleed keep the clear?

    WHY: this predicate is the single gate that lets the guardrail leave an
    over-clear as APPROPRIATE. Its precise behaviour (>300 mL strict, own
    confidence >=0.8, right code family, life-threatening path) is the locked
    clinical policy from spec #89.
    """

    def test_large_quantified_bleed_high_confidence_exempts(self) -> None:
        assert qualified_bleeding_exempt(
            [_active_bleed(quote="EBL 1100 mL intra-op", confidence=0.85)]
        )

    def test_life_threatening_marker_without_number_exempts(self) -> None:
        # No mL figure, but explicit uncontrolled hemorrhage + confidence >=0.8.
        assert qualified_bleeding_exempt(
            [_active_bleed(quote="uncontrolled hemorrhage", confidence=0.85)]
        )

    def test_exactly_300_ml_does_not_exempt(self) -> None:
        # Strict boundary: 300 mL is not > 300 mL.
        assert not qualified_bleeding_exempt(
            [_active_bleed(quote="EBL 300 mL", confidence=0.9)]
        )

    def test_just_over_300_ml_exempts(self) -> None:
        assert qualified_bleeding_exempt(
            [_active_bleed(quote="EBL 301 mL", confidence=0.9)]
        )

    def test_low_confidence_large_bleed_does_not_exempt(self) -> None:
        # Confidence 0.7 < 0.8: model is not sure enough to trust the prose.
        assert not qualified_bleeding_exempt(
            [_active_bleed(quote="EBL 1100 mL", confidence=0.7)]
        )

    def test_small_qualitative_bleed_does_not_exempt(self) -> None:
        assert not qualified_bleeding_exempt(
            [_active_bleed(quote="2x2 cm gauze staining", confidence=0.9)]
        )

    @pytest.mark.parametrize(
        "code",
        [
            "ACTIVE_BLEEDING_RISK",
            "ACTIVE_BLEEDING_HISTORY",
            "ACTIVE_BLEEDING_NOT_ACTIVE",
        ],
    )
    def test_negation_qualified_codes_do_not_exempt(self, code: str) -> None:
        # A *risk of* / *history of* / *not active* bleed is not an active bleed,
        # even at 1100 mL and high confidence.
        assert not qualified_bleeding_exempt(
            [_active_bleed(quote="EBL 1100 mL", confidence=0.95, code=code)]
        )

    def test_brisk_family_code_is_not_negation_qualified(self) -> None:
        # Codex PR #97 round 3: negation qualifiers must match whole
        # _-delimited tokens — BRISK contains RISK as a substring, but a
        # brisk >300 mL bleed is a genuine major active bleed and must exempt.
        assert qualified_bleeding_exempt(
            [
                _active_bleed(
                    quote="brisk bleeding, EBL 400 mL",
                    confidence=0.9,
                    code="ACTIVE_BLEEDING_BRISK",
                )
            ]
        )

    @pytest.mark.parametrize(
        "code",
        [
            "GI_BLEED_HISTORY_NOT_ACTIVE",
            "D62",
            "SUB_THRESHOLD_HB",
            "SYMPTOMATIC_ANEMIA",
        ],
    )
    def test_non_active_bleeding_family_does_not_exempt(self, code: str) -> None:
        assert not qualified_bleeding_exempt(
            [_active_bleed(quote="EBL 1100 mL", confidence=0.95, code=code)]
        )

    def test_tiny_decimal_volume_does_not_false_qualify(self) -> None:
        # A vanishingly small documented volume must NOT exempt: the parser
        # must not restart mid-number and read "350" out of the decimal tail.
        assert not qualified_bleeding_exempt(
            [_active_bleed(quote="0.0000000000000350 mL", confidence=0.95)]
        )

    def test_thai_quantified_bleed_exempts(self) -> None:
        assert qualified_bleeding_exempt(
            [_active_bleed(quote="เสียเลือด 1100 มล.", confidence=0.85)]
        )

    def test_one_qualifying_indication_among_many_exempts(self) -> None:
        # OR across indications: a single genuine major bleed is enough.
        assert qualified_bleeding_exempt(
            [
                _active_bleed(quote="ecchymosis", confidence=0.9),
                _active_bleed(quote="drain 900 mL", confidence=0.85),
                {"code": "SUB_THRESHOLD_HB", "quote": "Hb 7.4", "confidence": 0.9},
            ]
        )

    def test_empty_indications_does_not_exempt(self) -> None:
        assert not qualified_bleeding_exempt([])

    @pytest.mark.parametrize(
        "indication",
        [
            {},  # no keys at all
            {"quote": "EBL 1100 mL", "confidence": 0.9},  # missing code
            {"code": "ACTIVE_BLEEDING_GI", "confidence": 0.9},  # missing quote
            {"code": "ACTIVE_BLEEDING_GI", "quote": "EBL 1100 mL"},  # missing conf
            {"code": 123, "quote": "EBL 1100 mL", "confidence": 0.9},  # non-str code
            {
                "code": "ACTIVE_BLEEDING_GI",
                "quote": None,
                "confidence": 0.9,
            },  # bad quote
            {"code": "ACTIVE_BLEEDING_GI", "quote": "EBL 1100 mL", "confidence": "hi"},
            {"code": "ACTIVE_BLEEDING_GI", "quote": "EBL 1100 mL", "confidence": True},
            # Out-of-range confidence is schema-invalid (tool schema means
            # [0,1]); it must be treated as malformed, never as "very sure".
            {"code": "ACTIVE_BLEEDING_GI", "quote": "EBL 1100 mL", "confidence": 2.0},
            {"code": "ACTIVE_BLEEDING_GI", "quote": "EBL 1100 mL", "confidence": -0.1},
            {
                "code": "ACTIVE_BLEEDING_GI",
                "quote": "EBL 1100 mL",
                "confidence": float("nan"),
            },
            {
                "code": "ACTIVE_BLEEDING_GI",
                "quote": "EBL 1100 mL",
                "confidence": float("inf"),
            },
        ],
    )
    def test_malformed_indication_never_raises_and_never_exempts(
        self, indication: dict[str, object]
    ) -> None:
        # A malformed indication must never accidentally exempt an over-clear.
        assert qualified_bleeding_exempt([indication]) is False

    @pytest.mark.parametrize("junk", ["not-a-dict", None, 42, ["nested"]])
    def test_non_mapping_element_never_raises(self, junk: object) -> None:
        # RBC replay does not fully schema-validate; a non-mapping element in
        # the list must be skipped, not crash the guardrail. A qualifying real
        # bleed alongside it still exempts (the junk is simply ignored).
        assert qualified_bleeding_exempt([junk]) is False  # type: ignore[list-item]
        assert qualified_bleeding_exempt(
            [junk, _active_bleed(quote="drain 900 mL", confidence=0.9)]  # type: ignore[list-item]
        )


def test_policy_constants_match_locked_decisions() -> None:
    # Spec #89 locked decision 2: >300 mL and confidence >=0.8.
    assert LLM_OVERCLEAR_MIN_BLEED_ML == 300.0
    assert LLM_OVERCLEAR_MIN_BLEED_CONFIDENCE == 0.8
