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

from datetime import date

import pytest

from bba.audit_pipeline.bleeding import (
    _MAX_SCAN_CHARS,
    LLM_OVERCLEAR_MAX_BLEED_AGE_DAYS,
    LLM_OVERCLEAR_MIN_BLEED_CONFIDENCE,
    LLM_OVERCLEAR_MIN_BLEED_ML,
    bleeding_quote_is_stale,
    has_life_threatening_marker,
    parse_max_volume_ml,
    qualified_bleeding_exempt,
    quote_negates_bleeding,
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
            "ruled out active hemorrhage",  # PR #97 round 6: pre-marker
            "r/o active haemorrhage",  # exclusion phrases
            "absence of active hemorrhage",
            "history of active hemorrhage",
            "risk of uncontrolled bleeding",
            "r/o: active hemorrhage",  # PR #97 round 7: colon-formatted
            "rule out: active hemorrhage",  # exclusions stay in the clause
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

    def test_label_colon_prefix_does_not_suppress_marker(self) -> None:
        # A plain section label before a colon carries no negator; the marker
        # must still flag (pins that the round-7 colon change only preserves
        # negators, it does not suppress label-prefixed markers).
        assert has_life_threatening_marker("Assessment: active hemorrhage ongoing")

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


# Case 68080335's order day: 23/12/2568 BE == 2025-12-23 CE.
_ORDER_DATE = date(2025, 12, 23)


class TestStaleDatedVolumeGate:
    """The temporal gate: an old episode's figures must not clear this order.

    WHY (owner ruling, case 68080335): the notes charted a 400 mL index
    bleed dated ``Hx.1/12/68`` — 22 days before the order. Bleeding was
    still ongoing but unquantified, so the >300 mL bar was met only by a
    stale figure; auto-clearing on it defeats the restrictive policy. The
    gate blanks spans governed by a date anchor older than
    ``LLM_OVERCLEAR_MAX_BLEED_AGE_DAYS`` — and ONLY those spans, because a
    false stale-read withholds the exemption and falsely asserts a genuine
    transfusion INAPPROPRIATE (the doctor-facing error direction).
    """

    _STALE_QUOTE = "R/O LGIB Hx.1/12/68: ถ่ายอุจจาระเป็นเลือด 400 ml"

    def test_stale_index_bleed_volume_does_not_exempt(self) -> None:
        # The real case-68080335 charting: 400 mL dated 22 days pre-order.
        assert not qualified_bleeding_exempt(
            [_active_bleed(quote=self._STALE_QUOTE, confidence=0.85)],
            order_date=_ORDER_DATE,
        )

    def test_without_order_date_stale_volume_still_exempts(self) -> None:
        # Back-compat contract: the gate exists only when the caller supplies
        # the order moment — which is why the replay call sites must pass it.
        assert qualified_bleeding_exempt(
            [_active_bleed(quote=self._STALE_QUOTE, confidence=0.85)]
        )

    def test_current_episode_dated_volume_exempts(self) -> None:
        # A volume charted the day before the order is this episode's bleed.
        assert qualified_bleeding_exempt(
            [_active_bleed(quote="22/12/68: ถ่ายอุจจาระเป็นเลือด 400 ml", confidence=0.85)],
            order_date=_ORDER_DATE,
        )

    def test_undated_volume_is_treated_as_current(self) -> None:
        # No anchor -> no masking: an undated genuine bleed must keep clearing.
        assert qualified_bleeding_exempt(
            [_active_bleed(quote="EBL 400 mL intra-op", confidence=0.85)],
            order_date=_ORDER_DATE,
        )

    def test_text_before_the_first_anchor_is_kept(self) -> None:
        # A stale anchor governs only what FOLLOWS it; a current-episode
        # volume ahead of it must survive.
        assert qualified_bleeding_exempt(
            [
                _active_bleed(
                    quote="EBL 400 mL now; Hx.1/12/68: melena", confidence=0.85
                )
            ],
            order_date=_ORDER_DATE,
        )

    def test_stale_governance_stops_at_newline(self) -> None:
        # Thai focus notes are line-oriented ("date: content"); the next line
        # is a new statement and its current volume must survive.
        assert qualified_bleeding_exempt(
            [
                _active_bleed(
                    quote="Hx.1/12/68: melena\nวันนี้ EBL 400 mL", confidence=0.85
                )
            ],
            order_date=_ORDER_DATE,
        )

    def test_stale_governance_stops_at_the_next_anchor(self) -> None:
        # A later current-dated span is un-governed by the stale anchor.
        assert qualified_bleeding_exempt(
            [
                _active_bleed(
                    quote="1/12/68: ถ่ายเป็นเลือด 400 ml, 22/12/68: EBL 350 ml",
                    confidence=0.85,
                )
            ],
            order_date=_ORDER_DATE,
        )

    def test_stale_life_threatening_marker_does_not_exempt(self) -> None:
        # The marker path is gated identically: a shock charted three weeks
        # ago is not this order's emergency.
        assert not qualified_bleeding_exempt(
            [_active_bleed(quote="1/12/68: hemorrhagic shock", confidence=0.9)],
            order_date=_ORDER_DATE,
        )
        assert qualified_bleeding_exempt(
            [_active_bleed(quote="22/12/68: hemorrhagic shock", confidence=0.9)],
            order_date=_ORDER_DATE,
        )

    def test_four_digit_buddhist_year_is_recognised(self) -> None:
        assert not qualified_bleeding_exempt(
            [_active_bleed(quote="1/12/2568: ถ่ายเป็นเลือด 400 ml", confidence=0.85)],
            order_date=_ORDER_DATE,
        )

    def test_unparseable_date_token_governs_nothing(self) -> None:
        # An impossible day/month must never mask a genuine current volume.
        assert qualified_bleeding_exempt(
            [_active_bleed(quote="40/13/68: EBL 400 ml", confidence=0.85)],
            order_date=_ORDER_DATE,
        )

    def test_future_dated_anchor_governs_nothing(self) -> None:
        # A planned-procedure date after the order is not a stale anchor.
        assert qualified_bleeding_exempt(
            [_active_bleed(quote="25/12/68: EBL 400 ml", confidence=0.85)],
            order_date=_ORDER_DATE,
        )

    def test_age_boundary_seven_days_is_current_eight_is_stale(self) -> None:
        # Stale is strictly OLDER than the horizon: exactly 7 days still
        # plausibly belongs to the ordering episode.
        assert qualified_bleeding_exempt(
            [_active_bleed(quote="16/12/68: EBL 400 ml", confidence=0.85)],
            order_date=_ORDER_DATE,
        )
        assert not qualified_bleeding_exempt(
            [_active_bleed(quote="15/12/68: EBL 400 ml", confidence=0.85)],
            order_date=_ORDER_DATE,
        )


class TestMelenaDoesNotQualify:
    """Melena is digested/old blood, not active hemorrhage (owner ruling).

    WHY (case 68037502): the model cited ACTIVE_BLEEDING for melena at a
    gray-zone Hb with no shock. Melena's charted "volume" is stool, not blood
    loss, so it must NOT satisfy the >300 mL volume path of the standalone
    exemption. It still clears on a documented life-threatening / shock marker
    (the shock pathway); a qualifying low Hb is a separate exemption upstream.
    """

    @pytest.mark.parametrize(
        "quote",
        [
            "ถ่ายเป็น Melena 400 ml",
            "melena approx 500 mL overnight",
            "coffee-ground emesis 400 mL",
            "tarry black stool 600 ml",
            "ถ่ายดำ 400 ml",
        ],
    )
    def test_melena_volume_does_not_exempt(self, quote: str) -> None:
        assert not qualified_bleeding_exempt(
            [_active_bleed(quote=quote, confidence=0.9)]
        )

    def test_melena_with_shock_marker_still_exempts(self) -> None:
        # "With shock we care": a documented hemorrhagic-shock marker clears
        # even a melena quote (the shock pathway, not the volume path).
        assert qualified_bleeding_exempt(
            [_active_bleed(quote="melena with hemorrhagic shock", confidence=0.9)]
        )

    def test_fresh_blood_volume_still_exempts(self) -> None:
        # The disqualifier is melena-specific: fresh/active blood with a
        # qualifying volume is unaffected.
        assert qualified_bleeding_exempt(
            [_active_bleed(quote="fresh PR bleeding 400 mL", confidence=0.9)]
        )

    def test_melena_and_fresh_hematochezia_volume_disqualified(self) -> None:
        # Conservative simplification: a quote naming melena is disqualified
        # from the volume path even when a fresh term co-occurs — the volume
        # cannot be safely attributed to active loss. Documented as a known
        # limitation; the shock/Hb pathways still apply.
        assert not qualified_bleeding_exempt(
            [_active_bleed(quote="melena and hematochezia 500 mL", confidence=0.9)]
        )

    @pytest.mark.parametrize(
        "quote",
        [
            "denies melena; fresh PR bleeding 500 mL",
            "no melena; fresh PR bleeding 500 mL",
            "ไม่มีถ่ายดำ fresh PR bleeding 500 mL",
        ],
    )
    def test_negated_melena_does_not_suppress_fresh_volume(self, quote: str) -> None:
        # Codex PR #103: the disqualifier is negation-aware — an explicit
        # melena denial must not strip the volume path from a co-documented
        # current fresh bleed in the same quote.
        assert qualified_bleeding_exempt([_active_bleed(quote=quote, confidence=0.9)])

    def test_comma_denial_list_melena_stays_disqualified(self) -> None:
        # The marker screen's comma boundary is kept: a comma-distributed
        # denial ("no hematemesis, melena") still reads melena as present
        # here, which only withholds the volume exemption — fail-closed for
        # the auto-clear surface, same direction as the marker screens.
        assert not qualified_bleeding_exempt(
            [_active_bleed(quote="no hematemesis, melena; EBL 500 mL", confidence=0.9)]
        )

    @pytest.mark.parametrize(
        "quote",
        [
            "melena not controlled 500 mL",
            "melena not resolved 500 mL",
            "ถ่ายดำไม่หยุด 500 ml",
        ],
    )
    def test_still_active_melena_stays_disqualified(self, quote: str) -> None:
        # Codex PR #103 round 2: a post-side double negative means the
        # melena is ONGOING, not absent — the still-active rescue keeps the
        # volume disqualifier engaged, or the "not" would falsely reopen the
        # >300 mL path on a stool figure.
        assert not qualified_bleeding_exempt(
            [_active_bleed(quote=quote, confidence=0.9)]
        )


class TestBleedingQuoteIsStale:
    """``bleeding_quote_is_stale`` gates the hemodynamic-floor accompaniment.

    WHY (case 68080335, PR #100 Codex): the major-bleed exemption is not the
    only place a grounded ACTIVE_BLEEDING citation is trusted — the
    hemodynamic floor treats one as the "possible hemorrhagic picture" that
    routes a bare-hypotension over-clear to human review instead of asserting
    INAPPROPRIATE. A purely stale-dated bleed must be recognised as stale
    HERE too, or the stale-date gate leaks: hypotension + an old bleed would
    floor to NEEDS_REVIEW when the ruling wants INAPPROPRIATE. The predicate
    fails OPEN toward "current" so a genuine ongoing bleed still floors.
    """

    def test_purely_stale_dated_bleed_is_stale(self) -> None:
        # Codex's exact scenario: a bleed dated 22 days before the order.
        assert bleeding_quote_is_stale("1/12/68: active bleeding 400 ml", _ORDER_DATE)

    def test_history_label_prefix_before_stale_date_is_stale(self) -> None:
        # The REAL case-68080335 charting: a "Hx." / "R/O LGIB Hx." label
        # precedes the date, so forward-governance masking leaves the label
        # behind (PR #100 Codex round 2). A bare history label carries no
        # bleed term and no volume, so it must still read as stale — else the
        # hemodynamic floor accepts a 22-day-old bleed as current.
        assert bleeding_quote_is_stale(
            "Hx.1/12/68: active bleeding 400 ml", _ORDER_DATE
        )
        assert bleeding_quote_is_stale(
            "R/O LGIB Hx.1/12/68: ถ่ายอุจจาระเป็นเลือด 400 ml", _ORDER_DATE
        )

    def test_current_bleed_before_stale_history_date_is_not_stale(self) -> None:
        # A documented current volume ahead of a trailing stale history date
        # is real current evidence and survives masking.
        assert not bleeding_quote_is_stale(
            "EBL 250 mL now, prior Hx 1/12/68 bleed", _ORDER_DATE
        )

    def test_current_denial_plus_stale_bleed_is_stale(self) -> None:
        # PR #100 Codex round 3: the caller's up-front quote_negates_bleeding
        # runs on the UNMASKED quote, where the stale non-negated "active
        # bleeding 400 ml" defeats the denial screen. After masking, the only
        # surviving bleed term ("no active bleeding today") is NEGATED, so it
        # is a documented absence, not current evidence — the citation is
        # stale and bare hypotension must assert, not floor.
        assert bleeding_quote_is_stale(
            "no active bleeding today; Hx.1/12/68: active bleeding 400 ml",
            _ORDER_DATE,
        )

    def test_current_nonnegated_bleed_after_unrelated_denial_is_not_stale(
        self,
    ) -> None:
        # The negation re-screen must not over-fire: a denial of a DIFFERENT
        # symptom leaves the current bleed term non-negated and live.
        assert not bleeding_quote_is_stale(
            "denies chest pain; active bleeding per rectum now; Hx.1/12/68: 400 ml",
            _ORDER_DATE,
        )

    def test_current_dated_bleed_is_not_stale(self) -> None:
        assert not bleeding_quote_is_stale(
            "22/12/68: active bleeding 400 ml", _ORDER_DATE
        )

    def test_undated_bleed_is_not_stale(self) -> None:
        # No anchor -> nothing to mask -> current. An "EBL 450 mL" shorthand
        # with no date must still count as accompaniment (fail-open).
        assert not bleeding_quote_is_stale("EBL 450 mL, actively bleeding", _ORDER_DATE)

    def test_mixed_current_and_stale_is_not_stale(self) -> None:
        # Any surviving current text keeps the citation live (conservative:
        # errs toward flooring to human review, never toward asserting).
        assert not bleeding_quote_is_stale(
            "active bleeding now; 1/12/68: 400 ml", _ORDER_DATE
        )

    def test_seven_day_boundary_is_current_eight_is_stale(self) -> None:
        assert not bleeding_quote_is_stale("16/12/68: bleeding", _ORDER_DATE)
        assert bleeding_quote_is_stale("15/12/68: bleeding", _ORDER_DATE)

    def test_empty_and_undated_quotes_are_not_stale(self) -> None:
        # Fail-open contract: with no stale span removed, the predicate leaves
        # the caller's prior behaviour intact (the floor already screens
        # negation and non-bleed codes upstream).
        assert not bleeding_quote_is_stale("", _ORDER_DATE)
        assert not bleeding_quote_is_stale("active bleeding, no date here", _ORDER_DATE)


class TestQuoteNegatesBleeding:
    """``quote_negates_bleeding`` recognises absence-of-bleed prose.

    WHY (Codex PR #99 round 2 / owner ruling #98): qualifier (1) floors a
    bare-hypotension over-clear to a human only when a grounded
    ACTIVE_BLEEDING citation is genuine bleeding EVIDENCE. A citation whose
    grounded quote documents the ABSENCE of bleeding ("no active
    hemorrhage") is a mislabeled citation — trusting it would route bare
    hypotension to review, against the ruling that it stays INAPPROPRIATE.
    A false hit only withholds the floor (the assert stands); it can never
    auto-clear.
    """

    @pytest.mark.parametrize(
        "quote",
        [
            "no active hemorrhage seen",
            "denies bleeding or melena",
            "bleeding ruled out on scope",
            "r/o GI bleed",
            "without further blood loss overnight",
            "ไม่มีเลือดออก",  # "no bleeding"
            # Bleed synonyms (Codex PR #99 round 4): a miscoded negated
            # synonym must not slip past the screen just because the
            # generic terms miss it.
            "denies melena",
            "no hematemesis overnight",
            "negative for hematochezia",
            "ไม่มีถ่ายดำ",  # "no melena (black stool)"
            # Comma-separated denial lists (Codex PR #99 round 5): the
            # denial verb distributes across the list, so the comma must
            # not shield later items from the negator.
            "denies bleeding, melena",
            "no hematemesis, hematochezia, or melena",
            "bleeding, melena denied",
            # A contrastive AFTER the term still lets a trailing negator
            # void it: this bleed is documented as over.
            "melena noted but now resolved",
            "bleeding resolved after packing",
            "bleeding no longer present",
            # The rescue binds only its OWN negator (Codex PR #99 round
            # 9): an unrelated "not controlled" about pain must not
            # cancel the bleed denial in the same window.
            "bleeding denied, pain not controlled",
            # Label-value denials (Codex PR #99 round 8): checklist-style
            # notes deny with a bare value after a separator.
            "GI bleeding: no",
            "active bleeding - no",
            "bleeding: none",
            "melena? negative",
            "เลือดออก: ไม่มี",  # "bleeding: none"
        ],
    )
    def test_negated_bleed_prose_flags(self, quote: str) -> None:
        assert quote_negates_bleeding(quote) is True

    @pytest.mark.parametrize(
        "quote",
        [
            # Post-side double negatives (Codex PR #99 round 7): the
            # negator binds the CONTROL verb, not the bleeding — these
            # bleeds are ongoing and must stay visible to the floor.
            "bleeding not controlled after pressure",
            "bleeding not yet controlled",
            "bleeding no longer controlled",
            "เลือดออกไม่หยุด",  # "bleeding does not stop"
            # Adverb-split double negatives (round 8): still ongoing.
            "bleeding not yet resolved",
            "bleeding not completely resolved",
            "bleeding not fully controlled",
            # A label-value shape whose value is itself a rescued double
            # negative stays visible too.
            "bleeding: not controlled",
        ],
    )
    def test_still_active_double_negative_does_not_flag(self, quote: str) -> None:
        assert quote_negates_bleeding(quote) is False

    @pytest.mark.parametrize(
        "quote",
        [
            # Contrastive connectors (Codex PR #99 round 6): the denial
            # binds only up to "but" — what follows is documented ACTIVE
            # bleeding, so the quote stays usable as accompaniment.
            "denies hematemesis but melena ongoing",
            "no hematemesis, hematochezia but melena ongoing",
            "ไม่มีอาเจียนเป็นเลือด แต่ถ่ายดำ",  # "no hematemesis but melena"
        ],
    )
    def test_contrastive_bleed_clause_does_not_flag(self, quote: str) -> None:
        assert quote_negates_bleeding(quote) is False

    @pytest.mark.parametrize(
        "quote",
        [
            "intramuscular hematoma with active bleeding at Lt. thigh",
            "hemorrhagic shock, resuscitation ongoing",
            "เลือดออกมาก",  # "bleeding heavily"
            "melena x 3 this morning, ongoing",
        ],
    )
    def test_genuine_bleed_prose_does_not_flag(self, quote: str) -> None:
        assert quote_negates_bleeding(quote) is False

    def test_one_genuine_mention_outweighs_a_negated_one(self) -> None:
        # A note often clears one site while another still bleeds; the
        # genuine mention keeps the quote usable as evidence.
        assert (
            quote_negates_bleeding(
                "no bleeding from drain; active bleeding at Lt. thigh"
            )
            is False
        )

    def test_negation_does_not_leak_across_clause_boundary(self) -> None:
        # "no" bound to an earlier clause must not void a standing bleed.
        assert (
            quote_negates_bleeding("no fever today; bleeding from the wound continues")
            is False
        )

    def test_quote_without_bleeding_terms_does_not_flag(self) -> None:
        # Nothing to negate: the caller's code-level trust decides alone.
        assert quote_negates_bleeding("NIBP 79/54 (MAP 63) mmHg, on Levophed") is False


def test_policy_constants_match_locked_decisions() -> None:
    # Spec #89 locked decision 2: >300 mL and confidence >=0.8.
    assert LLM_OVERCLEAR_MIN_BLEED_ML == 300.0
    assert LLM_OVERCLEAR_MIN_BLEED_CONFIDENCE == 0.8
    # Case 68080335 owner ruling: a dated bleed figure older than a week no
    # longer quantifies the CURRENT episode.
    assert LLM_OVERCLEAR_MAX_BLEED_AGE_DAYS == 7
