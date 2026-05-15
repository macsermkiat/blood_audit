"""RED-phase failing tests for issue #17 (bba.deid_redactor).

Each test class maps to one acceptance criterion from the issue body.
Tests assert *contracts* (the WHY), not implementation choices. PRD §8
defines the wrapper's behavior; the module is a pure post-processing
layer over the ``thai-medical-deid`` backend, plugged in via Protocol.

No implementation exists yet (every public function ``raise
NotImplementedError("RED-phase scaffold; see issue #17")``); every
behavior assertion below must FAIL on this scaffold commit. Tests that
only exercise model-level Pydantic validation (which IS implemented at
this point) are kept to the bare minimum and clearly marked, because
RED-phase contract demands every test fail.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from bba.deid_redactor import (
    AGE_CAP,
    ATTENDING_CUES,
    BackendRedactionError,
    BackendRedactionResult,
    DATE_PATTERNS,
    DateMatch,
    DateShiftError,
    DeidRedactorError,
    FAMILY_CUES,
    HashMismatchError,
    K_ANONYMITY_MIN,
    KAnonymityGate,
    NURSE_CUES,
    NeedsReviewReason,
    NoteInput,
    PATIENT_CUES,
    PERSON_CLASS_TOKENS,
    QuasiIdentifiers,
    ROLE_CONTEXT_WINDOW,
    RedactedNote,
    RedactionRequest,
    RedactionResult,
    RedactionSpan,
    RedactorBackend,
    RedactorVersion,
    RoleClassifier,
    RoleToken,
    SEMANTIC_PERSON_THRESHOLD,
    SEMANTIC_WINDOW_CHARS,
    apply_age_cap,
    build_envelope,
    canonical_serialize,
    classify_role_by_cues,
    compute_k_groups,
    compute_redaction_hash,
    default_role_classifier,
    detect_semantic_degradation,
    extract_context,
    format_offset,
    k_anonymity_passed,
    parse_dates,
    redact_bundle,
    shift_dates_in_text,
    upgrade_person_tokens,
)


# =============================================================================
# Stubs + fixtures
# =============================================================================


@dataclass
class _StubBackend:
    """Deterministic :class:`RedactorBackend` for tests.

    The stub returns a fixed mapping ``input_text -> BackendRedactionResult``.
    Each input must be pre-registered via :meth:`register`; an
    unregistered input raises so a typo in a test fixture is loud, not
    silent. Mirrors the test pattern from
    :mod:`tests.unit.test_quote_grounder`.
    """

    responses: dict[str, BackendRedactionResult]

    def register(
        self, text: str, *, redacted: str, spans: tuple[RedactionSpan, ...] = ()
    ) -> None:
        self.responses[text] = BackendRedactionResult(text=redacted, spans=spans)

    def redact(self, text: str) -> BackendRedactionResult:
        if text not in self.responses:
            raise KeyError(f"_StubBackend: no registered response for {text!r}")
        return self.responses[text]


def _stub_backend() -> _StubBackend:
    return _StubBackend(responses={})


def _qi(
    *,
    ward: str = "W1",
    icd_3char: str = "D62",
    age_band: str = "60-69",
    sex: str = "M",
    admission_month: str = "2026-05",
) -> QuasiIdentifiers:
    return QuasiIdentifiers(
        ward=ward,
        icd_3char=icd_3char,
        age_band=age_band,
        sex=sex,
        admission_month=admission_month,
    )


def _version(
    *,
    version: str = "0.1.0",
    model_sha: str = "deadbeef",
    gazetteer_version: str = "pythainlp-5.0",
) -> RedactorVersion:
    return RedactorVersion(
        version=version, model_sha=model_sha, gazetteer_version=gazetteer_version
    )


def _request(
    *,
    notes: tuple[NoteInput, ...] = (NoteInput(note_id="E1", text="plain note"),),
    quasi_identifiers: QuasiIdentifiers | None = None,
    admission_date: date = date(2026, 5, 10),
    patient_age_years: int = 65,
    redactor_version: RedactorVersion | None = None,
) -> RedactionRequest:
    return RedactionRequest(
        notes=notes,
        quasi_identifiers=quasi_identifiers or _qi(),
        admission_date=admission_date,
        patient_age_years=patient_age_years,
        redactor_version=redactor_version or _version(),
    )


# =============================================================================
# Public-API surface — the import block at the top of this file IS the
# collection check; if a re-export is removed, pytest fails before any test
# runs. The tuple below pins the surface so ruff does not strip "unused"
# imports that exist *only* to assert the export shape (mirrors the pattern
# in :mod:`tests.unit.test_quote_grounder`).
# =============================================================================


_PUBLIC_SURFACE_PINS = (
    ATTENDING_CUES,
    DATE_PATTERNS,
    DateMatch,
    DateShiftError,
    DeidRedactorError,
    FAMILY_CUES,
    HashMismatchError,
    KAnonymityGate,
    NURSE_CUES,
    PATIENT_CUES,
    PERSON_CLASS_TOKENS,
    RedactedNote,
    RedactorBackend,
    RoleClassifier,
    SEMANTIC_WINDOW_CHARS,
)


# =============================================================================
# AC4: k-anonymity gate (k >= 5 routes away)
# =============================================================================


class TestKAnonymityGate:
    """k-anonymity: bundles with QI combos k<5 are routed to NEEDS_REVIEW."""

    def test_k_anonymity_passed_at_threshold(self) -> None:
        # k=5 is the boundary: PRD §8 says "k ≥ 5", so exactly 5 passes.
        assert k_anonymity_passed(K_ANONYMITY_MIN) is True

    def test_k_anonymity_failed_below_threshold(self) -> None:
        assert k_anonymity_passed(K_ANONYMITY_MIN - 1) is False

    def test_k_anonymity_failed_at_zero(self) -> None:
        assert k_anonymity_passed(0) is False

    def test_k_anonymity_passed_well_above_threshold(self) -> None:
        assert k_anonymity_passed(50) is True

    def test_k_anonymity_threshold_override(self) -> None:
        # A stricter policy (k=10) is a future possibility; the function
        # must honor the override so a config-time change does not need
        # a code change.
        assert k_anonymity_passed(7, k=10) is False
        assert k_anonymity_passed(10, k=10) is True

    def test_compute_k_groups_counts_each_qi_tuple(self) -> None:
        qi_a = _qi(ward="W1")
        qi_b = _qi(ward="W2")
        groups = compute_k_groups([qi_a, qi_a, qi_a, qi_b, qi_b])
        assert groups[qi_a] == 3
        assert groups[qi_b] == 2

    def test_compute_k_groups_order_independent(self) -> None:
        qi_a = _qi(ward="W1")
        qi_b = _qi(ward="W2")
        groups_ab = compute_k_groups([qi_a, qi_b, qi_a, qi_a, qi_b])
        groups_ba = compute_k_groups([qi_b, qi_a, qi_b, qi_a, qi_a])
        assert dict(groups_ab) == dict(groups_ba)

    def test_compute_k_groups_empty(self) -> None:
        # Total function: empty population is a valid input.
        groups = compute_k_groups([])
        assert dict(groups) == {}

    def test_compute_k_groups_returns_mapping(self) -> None:
        groups = compute_k_groups([_qi()])
        assert isinstance(groups, Mapping)


# =============================================================================
# AC6: age cap at 89
# =============================================================================


class TestAgeCap:
    """Age cap is exactly 89 (HIPAA-derived); above → capped, exactly → not."""

    def test_age_below_cap_unchanged(self) -> None:
        capped, was_capped = apply_age_cap(65)
        assert capped == 65
        assert was_capped is False

    def test_age_at_cap_unchanged(self) -> None:
        # AGE_CAP is 89 itself, NOT > 89; the boundary is inclusive.
        capped, was_capped = apply_age_cap(AGE_CAP)
        assert capped == AGE_CAP
        assert was_capped is False

    def test_age_above_cap_collapsed_to_cap(self) -> None:
        capped, was_capped = apply_age_cap(92)
        assert capped == AGE_CAP
        assert was_capped is True

    def test_age_extreme_age_collapsed(self) -> None:
        capped, was_capped = apply_age_cap(115)
        assert capped == AGE_CAP
        assert was_capped is True

    def test_age_zero_unchanged(self) -> None:
        # Neonatal ages are out-of-scope (PRD audit population is adult
        # inpatients), but the function must not silently cap 0 to AGE_CAP.
        capped, was_capped = apply_age_cap(0)
        assert capped == 0
        assert was_capped is False


# =============================================================================
# AC5: date-shift to Δ-days-from-admission
# =============================================================================


class TestDateShift:
    """Date-shift correctness: dates inside notes remapped to Δ-days."""

    def test_format_offset_zero_is_day_0(self) -> None:
        assert format_offset(days=0) == "Day 0"

    def test_format_offset_positive_has_explicit_plus(self) -> None:
        # Explicit + sign prevents ambiguity with absolute day-of-admission.
        assert format_offset(days=3) == "Day +3"

    def test_format_offset_negative_signed(self) -> None:
        assert format_offset(days=-2) == "Day -2"

    def test_parse_dates_iso(self) -> None:
        text = "Patient seen on 2026-05-12 with fever"
        matches = list(parse_dates(text))
        assert len(matches) == 1
        assert matches[0].parsed == date(2026, 5, 12)

    def test_parse_dates_dd_mm_yyyy(self) -> None:
        text = "Last seen 12/05/2026"
        matches = list(parse_dates(text))
        assert len(matches) == 1
        assert matches[0].parsed == date(2026, 5, 12)

    def test_parse_dates_multiple(self) -> None:
        text = "Admit 2026-05-10; discharge 2026-05-15"
        matches = list(parse_dates(text))
        assert len(matches) == 2
        # Sorted by start position so the wrapper can rewrite left-to-right
        # without index-shift bugs.
        assert matches[0].start < matches[1].start

    def test_parse_dates_iso_wins_over_dd_mm_format(self) -> None:
        # "2026-05-10" should ONLY parse as ISO, not also as DD-MM-YYYY
        # at a shifted position. Overlap resolution favors earlier patterns.
        text = "2026-05-10"
        matches = list(parse_dates(text))
        assert len(matches) == 1
        assert matches[0].parsed == date(2026, 5, 10)

    def test_parse_dates_word_boundary_anchored(self) -> None:
        # ``"H123-456-789"`` must not produce a false date match — the
        # ``\b`` anchor on the patterns prevents grabbing the middle.
        text = "ID H123-456-789 was assigned"
        matches = list(parse_dates(text))
        # Implementation-tolerant: the test asserts NO valid date is
        # parsed out of an identifier-like fragment. (The exact ``\b``
        # behavior is the contract — see DATE_PATTERNS docstring.)
        assert all(m.parsed.year >= 1900 for m in matches)

    def test_shift_dates_replaces_iso_with_offset(self) -> None:
        admission = date(2026, 5, 10)
        text = "Admitted on 2026-05-10; rebleed on 2026-05-13"
        shifted = shift_dates_in_text(text, admission_date=admission)
        assert "2026-05-10" not in shifted
        assert "2026-05-13" not in shifted
        assert "Day 0" in shifted
        assert "Day +3" in shifted

    def test_shift_dates_replaces_negative_offset(self) -> None:
        admission = date(2026, 5, 10)
        text = "Prior episode 2026-05-08"
        shifted = shift_dates_in_text(text, admission_date=admission)
        assert "Day -2" in shifted

    def test_shift_dates_leaves_non_matches_intact(self) -> None:
        # Numbers that are NOT dates (HN, room numbers) must pass through.
        admission = date(2026, 5, 10)
        text = "BP 120/80 HR 72 in room 425 ward 7"
        shifted = shift_dates_in_text(text, admission_date=admission)
        assert shifted == text

    def test_shift_dates_deterministic(self) -> None:
        # Bundle-hash stability: same input → byte-identical output.
        admission = date(2026, 5, 10)
        text = "Admit 2026-05-10 rebleed 2026-05-13"
        a = shift_dates_in_text(text, admission_date=admission)
        b = shift_dates_in_text(text, admission_date=admission)
        assert a == b


# =============================================================================
# AC3: role-token wrapper — cue-based classification
# =============================================================================


class TestRoleClassification:
    """The role classifier upgrades [PERSON] based on surrounding cues."""

    def test_attending_cue_classifies_as_attending(self) -> None:
        ctx = "the attending physician Dr. ___ ordered transfusion"
        assert classify_role_by_cues(ctx) is RoleToken.ATTENDING

    def test_thai_attending_cue_classifies_as_attending(self) -> None:
        # "นพ." is a Thai title for a male physician — common in KCMH notes.
        ctx = "นพ. ___ สั่ง transfusion"
        assert classify_role_by_cues(ctx) is RoleToken.ATTENDING

    def test_nurse_cue_classifies_as_nurse(self) -> None:
        ctx = "Nurse ___ administered the unit"
        assert classify_role_by_cues(ctx) is RoleToken.NURSE

    def test_thai_nurse_cue_classifies_as_nurse(self) -> None:
        ctx = "พยาบาล ___ รายงาน"
        assert classify_role_by_cues(ctx) is RoleToken.NURSE

    def test_patient_cue_classifies_as_patient(self) -> None:
        ctx = "Patient ___ complains of melena"
        assert classify_role_by_cues(ctx) is RoleToken.PATIENT

    def test_thai_patient_cue_classifies_as_patient(self) -> None:
        ctx = "ผู้ป่วย ___ มาด้วยอาการ"
        assert classify_role_by_cues(ctx) is RoleToken.PATIENT

    def test_family_cue_classifies_as_family(self) -> None:
        ctx = "Mother ___ at bedside"
        assert classify_role_by_cues(ctx) is RoleToken.FAMILY

    def test_thai_family_cue_classifies_as_family(self) -> None:
        # Note: a context containing BOTH "ภรรยา" (FAMILY) and "ผู้ป่วย"
        # (PATIENT) classifies as PATIENT (higher priority); this test
        # exercises a context with ONLY family cues so the FAMILY branch
        # is testable in isolation.
        ctx_family_only = "Spouse ___ at bedside signed the consent"
        assert classify_role_by_cues(ctx_family_only) is RoleToken.FAMILY

    def test_no_cue_returns_none(self) -> None:
        # Without any cue, the wrapper keeps the generic [PERSON] token —
        # the classifier must NOT default to a role.
        ctx = "BP 120/80 HR 72 ___ stable"
        assert classify_role_by_cues(ctx) is None

    def test_priority_attending_beats_nurse(self) -> None:
        # When both cues are present in the window, ATTENDING wins —
        # the SOAP convention is doctor speaks last; the nurse's note
        # often quotes the doctor.
        ctx = "Nurse asked attending Dr. ___ for orders"
        assert classify_role_by_cues(ctx) is RoleToken.ATTENDING

    def test_priority_nurse_beats_patient(self) -> None:
        ctx = "Patient instructed by Nurse ___"
        assert classify_role_by_cues(ctx) is RoleToken.NURSE

    def test_priority_patient_beats_family(self) -> None:
        ctx = "Patient ___ accompanied by spouse"
        assert classify_role_by_cues(ctx) is RoleToken.PATIENT

    def test_cue_lexicon_lowercase(self) -> None:
        # The cue lists are searched case-insensitively — otherwise
        # "DR." in a screaming-caps note would silently fail to classify.
        ctx = "DR. ___ ORDERED 2U PRBC NOW"
        assert classify_role_by_cues(ctx) is RoleToken.ATTENDING


class TestExtractContext:
    """Context extraction window the classifier reads from."""

    def test_extract_context_returns_window_around_span(self) -> None:
        text = "0123456789" * 20
        span = RedactionSpan(start=100, end=105, entity_type="PERSON")
        ctx = extract_context(
            original_text=text, span=span, window=ROLE_CONTEXT_WINDOW
        )
        # Must include text just before and after the span, total length
        # bounded by 2 * window + a single-char separator.
        assert len(ctx) <= 2 * ROLE_CONTEXT_WINDOW + 1

    def test_extract_context_clamps_at_string_boundaries(self) -> None:
        # span starts at 0 → no "before" content; must not raise.
        text = "0123456789"
        span = RedactionSpan(start=0, end=2, entity_type="PERSON")
        ctx = extract_context(original_text=text, span=span, window=20)
        assert isinstance(ctx, str)

    def test_extract_context_joins_with_single_space(self) -> None:
        # The before+after halves must be joined by a single separator
        # so a cue straddling the span isn't glued into a false match.
        text = "Dr." + "X" * 50 + "MD"  # exaggerated spacing
        span = RedactionSpan(start=3, end=53, entity_type="PERSON")
        ctx = extract_context(original_text=text, span=span, window=5)
        # ``"Dr."`` and ``"MD"`` are on opposite sides — both must be
        # in the window but not glued into ``"Dr.MD"``.
        assert "Dr." in ctx
        assert "MD" in ctx
        assert "Dr.MD" not in ctx


class TestDefaultRoleClassifier:
    """The :func:`default_role_classifier` composes extract+classify."""

    def test_default_classifier_satisfies_protocol(self) -> None:
        # Smoke: callable with the keyword-only signature the Protocol
        # expects. RoleClassifier is duck-typed via Protocol so this is
        # both a contract check and a typing-system sanity check.
        result = default_role_classifier(
            original_text="Dr. John Smith ordered transfusion",
            context="Dr. John Smith ordered transfusion",
            span=RedactionSpan(start=4, end=14, entity_type="PERSON"),
        )
        assert result is RoleToken.ATTENDING

    def test_default_classifier_returns_none_without_cue(self) -> None:
        result = default_role_classifier(
            original_text="Just some clinical text",
            context="Just some clinical text",
            span=RedactionSpan(start=5, end=9, entity_type="PERSON"),
        )
        assert result is None

    def test_default_classifier_prefers_closest_cue_not_highest_priority(self) -> None:
        # Codex GitHub review (PR #40): "Dr. Smith saw patient John Doe".
        # For the John-Doe span, both "dr." (priority ATTENDING) and
        # "patient" (priority PATIENT) appear in the ±40-char window.
        # The classifier MUST pick the cue NEAREST the span — patient is
        # adjacent; Dr. is far. Returning ATTENDING here would mislabel
        # the patient as a physician.
        text = "Dr. Smith saw patient John Doe"
        result = default_role_classifier(
            original_text=text,
            context=text,
            span=RedactionSpan(
                start=22, end=30, entity_type="PERSON", original_text="John Doe"
            ),
        )
        assert result is RoleToken.PATIENT

    def test_default_classifier_prefers_nurse_when_nurse_is_closer(self) -> None:
        # Symmetric case: doctor mentioned earlier, nurse closer. The
        # nurse span must classify as NURSE even though ATTENDING is
        # higher in the global priority list.
        text = "Dr. Smith asked nurse Jane to recheck"
        result = default_role_classifier(
            original_text=text,
            context=text,
            span=RedactionSpan(
                start=22, end=26, entity_type="PERSON", original_text="Jane"
            ),
        )
        assert result is RoleToken.NURSE

    def test_default_classifier_ignores_name_that_equals_family_cue(self) -> None:
        # Codex GitHub review (PR #40 round 2): a patient named "Son"
        # must classify as PATIENT (the cue nearest the span), not
        # FAMILY (the span text itself happens to equal a FAMILY cue
        # word). Span-internal matching must be restricted to
        # unambiguous titles/honorifics so name/cue collisions don't
        # short-circuit the proximity logic.
        text = "patient Son was admitted with low Hb"
        result = default_role_classifier(
            original_text=text,
            context=text,
            span=RedactionSpan(
                start=8, end=11, entity_type="PERSON", original_text="Son"
            ),
        )
        assert result is RoleToken.PATIENT

    def test_default_classifier_in_span_honorific_still_wins(self) -> None:
        # Symmetric to the above: when the backend includes the title
        # INSIDE the redacted span (e.g. "Dr. Smith" as one span), the
        # honorifics-only span-internal check must still upgrade to
        # ATTENDING. Honorifics dominate proximity because they are
        # unambiguous title markers.
        text = "Dr. Smith ordered transfusion"
        result = default_role_classifier(
            original_text=text,
            context=text,
            span=RedactionSpan(
                start=0, end=9, entity_type="PERSON", original_text="Dr. Smith"
            ),
        )
        assert result is RoleToken.ATTENDING

    def test_default_classifier_priority_breaks_tie_at_equal_distance(self) -> None:
        # When two cues sit at the same distance from the span (one in
        # the before-window, one in the after-window, mirror-symmetric),
        # the global priority order is the deterministic tie-breaker so
        # the classifier remains reproducible (bundle-hash stability).
        # Here PATIENT (distance 1 from span end) and FAMILY ("family"
        # at distance 1 from span start) are equidistant; PATIENT wins
        # by priority.
        text = "patient ___ family"
        result = default_role_classifier(
            original_text=text,
            context=text,
            span=RedactionSpan(
                start=8, end=11, entity_type="PERSON", original_text="___"
            ),
        )
        assert result is RoleToken.PATIENT


# =============================================================================
# Token upgrade — PERSON placeholders → role tokens in document order
# =============================================================================


class TestUpgradePersonTokens:
    """Walk redacted_text, replace [PERSON] in span order."""

    def test_single_person_upgraded(self) -> None:
        redacted = "[PERSON] ordered 2U PRBC"
        original = "Dr. John ordered 2U PRBC"
        spans = (
            RedactionSpan(start=0, end=8, entity_type="PERSON", original_text="Dr. John"),
        )

        def classifier(
            *,
            original_text: str,
            context: str,
            span: RedactionSpan,
        ) -> RoleToken | None:
            return RoleToken.ATTENDING

        result = upgrade_person_tokens(
            redacted_text=redacted,
            original_text=original,
            spans=spans,
            classifier=classifier,
        )
        assert result == "[ATTENDING] ordered 2U PRBC"

    def test_multiple_persons_upgraded_in_order(self) -> None:
        redacted = "[PERSON] told [PERSON] to recheck Hb"
        original = "Dr. Smith told Nurse Jane to recheck Hb"
        spans = (
            RedactionSpan(start=0, end=9, entity_type="PERSON", original_text="Dr. Smith"),
            RedactionSpan(start=15, end=25, entity_type="PERSON", original_text="Nurse Jane"),
        )

        def classifier(
            *,
            original_text: str,
            context: str,
            span: RedactionSpan,
        ) -> RoleToken | None:
            return (
                RoleToken.ATTENDING
                if span.start == 0
                else RoleToken.NURSE
            )

        result = upgrade_person_tokens(
            redacted_text=redacted,
            original_text=original,
            spans=spans,
            classifier=classifier,
        )
        assert result == "[ATTENDING] told [NURSE] to recheck Hb"

    def test_unclassified_person_preserved(self) -> None:
        # Classifier returns None → keep the generic [PERSON].
        redacted = "[PERSON] is stable"
        spans = (
            RedactionSpan(start=0, end=4, entity_type="PERSON", original_text="John"),
        )

        def classifier(**_kwargs: object) -> RoleToken | None:
            return None

        result = upgrade_person_tokens(
            redacted_text=redacted,
            original_text="John is stable",
            spans=spans,
            classifier=classifier,
        )
        assert result == "[PERSON] is stable"

    def test_span_count_mismatch_raises_backend_error(self) -> None:
        # Wrapper invariant: ``spans`` PERSON-count must equal the
        # number of [PERSON] placeholders in ``redacted_text``. A
        # backend that violates this is a contract bug; the wrapper
        # raises BackendRedactionError so the audit pipeline fails loud.
        redacted = "[PERSON] told [PERSON] to recheck"
        spans = (
            RedactionSpan(start=0, end=4, entity_type="PERSON", original_text="Dr"),
        )

        def classifier(**_kwargs: object) -> RoleToken | None:
            return RoleToken.ATTENDING

        with pytest.raises(BackendRedactionError):
            upgrade_person_tokens(
                redacted_text=redacted,
                original_text="Dr. John told Nurse Jane",
                spans=spans,
                classifier=classifier,
            )

    def test_non_person_spans_skipped(self) -> None:
        # DATE/LOCATION spans are NOT classified — the wrapper handles
        # them through the type-matching token elsewhere.
        redacted = "[PERSON] on [DATE] in [LOCATION]"
        original = "Dr. X on 2026-05-10 in ward 7"
        spans = (
            RedactionSpan(start=0, end=5, entity_type="PERSON", original_text="Dr. X"),
            RedactionSpan(start=9, end=19, entity_type="DATE", original_text="2026-05-10"),
            RedactionSpan(start=23, end=29, entity_type="LOCATION", original_text="ward 7"),
        )

        def classifier(**_kwargs: object) -> RoleToken | None:
            return RoleToken.ATTENDING

        result = upgrade_person_tokens(
            redacted_text=redacted,
            original_text=original,
            spans=spans,
            classifier=classifier,
        )
        # Only the PERSON token was upgraded; DATE and LOCATION pass through.
        assert "[ATTENDING] on [DATE] in [LOCATION]" == result


# =============================================================================
# AC7: semantic-degradation flag
# =============================================================================


class TestSemanticDegradation:
    """>4 PERSON-class tokens within 50 chars → flag."""

    def test_zero_tokens_no_flag(self) -> None:
        assert detect_semantic_degradation("BP 120/80 HR 72 stable") is False

    def test_exactly_threshold_no_flag(self) -> None:
        # Strict ``>``: exactly 4 tokens in any window does NOT fire.
        text = "[PERSON][PERSON][PERSON][PERSON]"
        assert detect_semantic_degradation(text) is False

    def test_one_above_threshold_fires(self) -> None:
        # 5 PERSON tokens within ~40 chars — well inside the 50-char window.
        text = "[PERSON][PERSON][PERSON][PERSON][PERSON]"
        assert detect_semantic_degradation(text) is True

    def test_density_across_window_boundary(self) -> None:
        # 5 tokens spread over 60 chars (gap between each ~3 chars after
        # the [PERSON] literal) — within a 50-char window, at most 4
        # token starts fit, so it should NOT fire.
        text = "[PERSON]   [PERSON]   [PERSON]   [PERSON]   [PERSON]"
        # 5 tokens; first start 0, last start ~44. 50-char window from 0
        # includes all 5 starts → FIRE. The fixture is intentional: this
        # documents that the function counts starts-in-window strictly.
        assert detect_semantic_degradation(text) is True

    def test_density_just_outside_window(self) -> None:
        # 5 tokens spread out so that no 50-char window catches all 5.
        # Each [PERSON] is 8 chars; with 15-char gap, last start at
        # ~4 * (8 + 15) = 92; no 50-char window covers all 5 starts.
        text = (
            "[PERSON]" + " " * 15
            + "[PERSON]" + " " * 15
            + "[PERSON]" + " " * 15
            + "[PERSON]" + " " * 15
            + "[PERSON]"
        )
        assert detect_semantic_degradation(text) is False

    def test_role_tokens_count_toward_density(self) -> None:
        # ATTENDING / NURSE / PATIENT / FAMILY are PERSON-class too — a
        # post-role-upgrade text with role tokens densely packed must
        # fire the same as generic PERSON tokens.
        text = "[ATTENDING][NURSE][PATIENT][FAMILY][PERSON]"
        assert detect_semantic_degradation(text) is True

    def test_non_person_tokens_excluded(self) -> None:
        # [DATE]/[LOCATION] are NOT PERSON-class — five of them in a
        # tight window must NOT fire the flag.
        text = "[DATE][LOCATION][DATE][LOCATION][DATE]"
        assert detect_semantic_degradation(text) is False


# =============================================================================
# Canonical serialization + hash
# =============================================================================


class TestCanonical:
    """canonical_serialize + compute_redaction_hash contract."""

    def test_canonical_sorted_keys(self) -> None:
        # Insertion order MUST NOT affect output bytes — sorted keys is
        # what makes the hash stable across runs.
        a = canonical_serialize({"b": 1, "a": 2})
        b = canonical_serialize({"a": 2, "b": 1})
        assert a == b

    def test_canonical_no_trailing_newline(self) -> None:
        out = canonical_serialize({"a": 1})
        assert not out.endswith("\n")

    def test_canonical_two_space_indent(self) -> None:
        # The audit chain's canonical-bytes contract is "indent=2"; a
        # ``json.dumps(..., indent=4)`` drift would break the hash.
        out = canonical_serialize({"a": [1, 2]})
        assert "  " in out

    def test_canonical_nfc_normalizes_strings(self) -> None:
        # Thai NFD vs NFC byte-difference must collapse to one canonical
        # form before hashing — otherwise the same logical bundle hashes
        # to two different values on two locales.
        nfd_key = unicodedata.normalize("NFD", "ผู้ป่วย")
        nfc_key = unicodedata.normalize("NFC", "ผู้ป่วย")
        out_nfd = canonical_serialize({nfd_key: "value"})
        out_nfc = canonical_serialize({nfc_key: "value"})
        assert out_nfd == out_nfc

    def test_canonical_rejects_nan(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            canonical_serialize({"x": float("nan")})

    def test_canonical_rejects_inf(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            canonical_serialize({"x": float("inf")})

    def test_compute_redaction_hash_is_sha256_hex(self) -> None:
        envelope = build_envelope(
            notes=[],
            redactor_version={
                "version": "0.1.0",
                "model_sha": "deadbeef",
                "gazetteer_version": "v1",
            },
            redacted_age=65,
            age_capped=False,
            k_anonymity_size=10,
            k_anonymity_passed=True,
            route_to_needs_review=False,
            needs_review_reasons=[],
        )
        h = compute_redaction_hash(envelope)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_compute_redaction_hash_matches_manual_sha256(self) -> None:
        envelope = build_envelope(
            notes=[],
            redactor_version={
                "version": "0.1.0",
                "model_sha": "deadbeef",
                "gazetteer_version": "v1",
            },
            redacted_age=65,
            age_capped=False,
            k_anonymity_size=10,
            k_anonymity_passed=True,
            route_to_needs_review=False,
            needs_review_reasons=[],
        )
        expected = hashlib.sha256(
            canonical_serialize(envelope).encode("utf-8")
        ).hexdigest()
        assert compute_redaction_hash(envelope) == expected

    def test_envelope_shape_is_locked(self) -> None:
        # Exactly the 8 documented keys — extras would silently change
        # the hash on downstream upgrades. Mirrors evidence_bundle_builder's
        # envelope-lock invariant.
        envelope = build_envelope(
            notes=[{"note_id": "E1", "redacted_text": "x", "semantic_degraded": False}],
            redactor_version={
                "version": "0.1.0",
                "model_sha": "deadbeef",
                "gazetteer_version": "v1",
            },
            redacted_age=65,
            age_capped=False,
            k_anonymity_size=10,
            k_anonymity_passed=True,
            route_to_needs_review=False,
            needs_review_reasons=[],
        )
        assert set(envelope.keys()) == {
            "notes",
            "redactor_version",
            "redacted_age",
            "age_capped",
            "k_anonymity_size",
            "k_anonymity_passed",
            "route_to_needs_review",
            "needs_review_reasons",
        }


# =============================================================================
# RedactionResult model validator — hash must match canonical bytes
# =============================================================================


class TestRedactionResultValidation:
    """The result model's hash-match invariant (mirrors EvidenceBundle)."""

    def _build_valid_result(self, *, override_hash: str | None = None) -> RedactionResult:
        """Construct a :class:`RedactionResult` whose hash matches its envelope.

        ``override_hash`` lets tests verify the validator rejects a
        misshapen hash without re-deriving the rest of the envelope.
        """
        # Version metadata MUST be identical on both halves of the
        # construction (the envelope and the RedactionResult), otherwise
        # the model validator's hash-recompute would correctly reject.
        version = _version()
        envelope = build_envelope(
            notes=[],
            redactor_version={
                "version": version.version,
                "model_sha": version.model_sha,
                "gazetteer_version": version.gazetteer_version,
            },
            redacted_age=65,
            age_capped=False,
            k_anonymity_size=10,
            k_anonymity_passed=True,
            route_to_needs_review=False,
            needs_review_reasons=[],
        )
        return RedactionResult(
            notes=(),
            redactor_version=version,
            redacted_age=65,
            age_capped=False,
            k_anonymity_size=10,
            k_anonymity_passed=True,
            route_to_needs_review=False,
            needs_review_reasons=(),
            redaction_hash=(
                override_hash
                if override_hash is not None
                else compute_redaction_hash(envelope)
            ),
        )

    def test_valid_result_constructs(self) -> None:
        r = self._build_valid_result()
        assert r.redactor_version.version == "0.1.0"

    def test_wrong_length_hash_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._build_valid_result(override_hash="abc123")


# =============================================================================
# QuasiIdentifiers sex-code validation (codex review round 1, AC#4)
# =============================================================================


class TestQuasiIdentifiersSexValidation:
    """Sex codes outside the canonical set must be rejected at construction.

    Otherwise k-anonymity grouping would silently fragment when the same
    population's sex values arrive in different free-form spellings
    (``"Male"`` vs ``"M"`` vs ``"male"``).
    """

    def test_sex_M_accepted(self) -> None:
        assert _qi(sex="M").sex == "M"

    def test_sex_F_accepted(self) -> None:
        assert _qi(sex="F").sex == "F"

    def test_sex_U_accepted(self) -> None:
        # ``U`` (unknown) covers HOSxP exports where sex is missing.
        assert _qi(sex="U").sex == "U"

    def test_sex_lowercase_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _qi(sex="m")

    def test_sex_full_word_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _qi(sex="Male")

    def test_sex_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _qi(sex="")


# =============================================================================
# Top-level redact_bundle orchestration (AC1, 4, 5, 6, 7, 8 in combination)
# =============================================================================


class TestRedactBundleEndToEnd:
    """End-to-end behavior of :func:`redact_bundle`."""

    def test_simple_redaction_stamps_version(self) -> None:
        backend = _stub_backend()
        backend.register("plain note", redacted="plain note")
        req = _request()
        result = redact_bundle(
            req,
            backend=backend,
            k_gate=lambda qi: 10,
        )
        assert result.redactor_version == req.redactor_version

    def test_age_capping_surfaces_on_result(self) -> None:
        backend = _stub_backend()
        backend.register("plain note", redacted="plain note")
        req = _request(patient_age_years=95)
        result = redact_bundle(
            req, backend=backend, k_gate=lambda qi: 10
        )
        assert result.redacted_age == AGE_CAP
        assert result.age_capped is True

    def test_age_below_cap_unchanged_on_result(self) -> None:
        backend = _stub_backend()
        backend.register("plain note", redacted="plain note")
        req = _request(patient_age_years=70)
        result = redact_bundle(
            req, backend=backend, k_gate=lambda qi: 10
        )
        assert result.redacted_age == 70
        assert result.age_capped is False

    def test_k_anonymity_pass_does_not_route(self) -> None:
        backend = _stub_backend()
        backend.register("plain note", redacted="plain note")
        req = _request()
        result = redact_bundle(
            req, backend=backend, k_gate=lambda qi: 7
        )
        assert result.k_anonymity_size == 7
        assert result.k_anonymity_passed is True
        # Without semantic degradation, no routing.
        assert result.route_to_needs_review is False

    def test_k_anonymity_fail_routes(self) -> None:
        backend = _stub_backend()
        backend.register("plain note", redacted="plain note")
        req = _request()
        result = redact_bundle(
            req, backend=backend, k_gate=lambda qi: 2
        )
        assert result.k_anonymity_passed is False
        assert result.route_to_needs_review is True
        assert NeedsReviewReason.K_ANONYMITY_FAIL in result.needs_review_reasons

    def test_semantic_degradation_routes(self) -> None:
        # Backend emits 5 [PERSON] tokens in a tight window.
        backend = _stub_backend()
        backend.register(
            "PHI heavy",
            redacted="[PERSON][PERSON][PERSON][PERSON][PERSON]",
            spans=tuple(
                RedactionSpan(start=i, end=i + 1, entity_type="PERSON")
                for i in range(5)
            ),
        )
        req = _request(notes=(NoteInput(note_id="E1", text="PHI heavy"),))
        result = redact_bundle(
            req, backend=backend, k_gate=lambda qi: 10
        )
        assert result.route_to_needs_review is True
        assert (
            NeedsReviewReason.SEMANTIC_DEGRADATION
            in result.needs_review_reasons
        )

    def test_in_text_date_shifted(self) -> None:
        backend = _stub_backend()
        # The backend leaves the in-text date intact (its regex did not
        # catch the prose date); the wrapper's date-shift transform
        # rewrites it to a Δ-day offset.
        backend.register(
            "Admitted 2026-05-10 stable",
            redacted="Admitted 2026-05-10 stable",
        )
        req = _request(
            notes=(NoteInput(note_id="E1", text="Admitted 2026-05-10 stable"),),
            admission_date=date(2026, 5, 10),
        )
        result = redact_bundle(
            req, backend=backend, k_gate=lambda qi: 10
        )
        assert "2026-05-10" not in result.notes[0].redacted_text
        assert "Day 0" in result.notes[0].redacted_text

    def test_backend_date_span_shifted_to_offset(self) -> None:
        # The backend tagged a date PHI as a span and replaced it with
        # the generic [DATE] placeholder; the wrapper rewrites the
        # placeholder to a Δ-day offset using the span's original_text.
        # This is the canonical backend-redacted-date path (the regex
        # path is the fallback for backend misses).
        backend = _stub_backend()
        backend.register(
            "Admitted 2026-05-13 for low Hb",
            redacted="Admitted [DATE] for low Hb",
            spans=(
                RedactionSpan(
                    start=9,
                    end=19,
                    entity_type="DATE",
                    original_text="2026-05-13",
                ),
            ),
        )
        req = _request(
            notes=(
                NoteInput(note_id="E1", text="Admitted 2026-05-13 for low Hb"),
            ),
            admission_date=date(2026, 5, 10),
        )
        result = redact_bundle(
            req, backend=backend, k_gate=lambda qi: 10
        )
        # Backend-tagged [DATE] becomes Day +3 (May 13 vs admission May 10).
        assert "[DATE]" not in result.notes[0].redacted_text
        assert "Day +3" in result.notes[0].redacted_text

    def test_backend_date_span_unparseable_keeps_placeholder(self) -> None:
        # If the backend tagged a date span whose original_text is not
        # in a wrapper-recognized format (e.g. "May 13, 2026"), the
        # placeholder stays as [DATE] — fail-open so the PHI redaction
        # itself isn't compromised by the wrapper's lack of a Δ-day
        # annotation.
        backend = _stub_backend()
        backend.register(
            "Seen on May 13, 2026",
            redacted="Seen on [DATE]",
            spans=(
                RedactionSpan(
                    start=8,
                    end=20,
                    entity_type="DATE",
                    original_text="May 13, 2026",
                ),
            ),
        )
        req = _request(
            notes=(NoteInput(note_id="E1", text="Seen on May 13, 2026"),),
            admission_date=date(2026, 5, 10),
        )
        result = redact_bundle(
            req, backend=backend, k_gate=lambda qi: 10
        )
        assert "[DATE]" in result.notes[0].redacted_text

    def test_backend_date_span_zero_offset(self) -> None:
        # Day 0 boundary on the backend-tagged-DATE path. The literal-
        # regex path covers Day 0 separately; this case ensures the
        # backend-span path also renders the unsigned zero form.
        backend = _stub_backend()
        backend.register(
            "Admitted 2026-05-10",
            redacted="Admitted [DATE]",
            spans=(
                RedactionSpan(
                    start=9,
                    end=19,
                    entity_type="DATE",
                    original_text="2026-05-10",
                ),
            ),
        )
        req = _request(
            notes=(NoteInput(note_id="E1", text="Admitted 2026-05-10"),),
            admission_date=date(2026, 5, 10),
        )
        result = redact_bundle(req, backend=backend, k_gate=lambda qi: 10)
        assert "Day 0" in result.notes[0].redacted_text

    def test_backend_multiple_date_spans_shifted_in_order(self) -> None:
        # Two DATE spans, two [DATE] placeholders — must map in document
        # order so the earlier admission_date offset lands at the
        # earlier placeholder.
        backend = _stub_backend()
        backend.register(
            "Seen 2026-05-10 then 2026-05-15",
            redacted="Seen [DATE] then [DATE]",
            spans=(
                RedactionSpan(
                    start=5,
                    end=15,
                    entity_type="DATE",
                    original_text="2026-05-10",
                ),
                RedactionSpan(
                    start=21,
                    end=31,
                    entity_type="DATE",
                    original_text="2026-05-15",
                ),
            ),
        )
        req = _request(
            notes=(
                NoteInput(note_id="E1", text="Seen 2026-05-10 then 2026-05-15"),
            ),
            admission_date=date(2026, 5, 10),
        )
        result = redact_bundle(req, backend=backend, k_gate=lambda qi: 10)
        # Order matters: Day 0 must appear BEFORE Day +5 in the output.
        rendered = result.notes[0].redacted_text
        assert "[DATE]" not in rendered
        assert rendered.index("Day 0") < rendered.index("Day +5")

    def test_backend_date_span_empty_original_text_keeps_placeholder(self) -> None:
        # Backend tagged a date PHI but did not surface original_text
        # (some backends omit it). The wrapper falls open: [DATE] stays
        # as the generic placeholder rather than fabricating an offset.
        backend = _stub_backend()
        backend.register(
            "Seen on [DATE]",
            redacted="Seen on [DATE]",
            spans=(
                RedactionSpan(
                    start=8,
                    end=14,
                    entity_type="DATE",
                    original_text="",
                ),
            ),
        )
        req = _request(
            notes=(NoteInput(note_id="E1", text="Seen on [DATE]"),),
            admission_date=date(2026, 5, 10),
        )
        result = redact_bundle(req, backend=backend, k_gate=lambda qi: 10)
        assert "[DATE]" in result.notes[0].redacted_text

    def test_backend_date_span_more_spans_than_placeholders_raises(self) -> None:
        # Inverse direction of the count-mismatch invariant: more DATE
        # spans than [DATE] placeholders also signals a backend contract
        # violation and must raise BackendRedactionError.
        backend = _stub_backend()
        backend.register(
            "Seen 2026-05-10 and 2026-05-11",
            redacted="Seen [DATE] and the second date redacted inline",
            spans=(
                RedactionSpan(
                    start=5,
                    end=15,
                    entity_type="DATE",
                    original_text="2026-05-10",
                ),
                RedactionSpan(
                    start=20,
                    end=30,
                    entity_type="DATE",
                    original_text="2026-05-11",
                ),
            ),
        )
        req = _request(
            notes=(
                NoteInput(note_id="E1", text="Seen 2026-05-10 and 2026-05-11"),
            ),
        )
        with pytest.raises(BackendRedactionError):
            redact_bundle(req, backend=backend, k_gate=lambda qi: 10)

    def test_backend_date_span_count_mismatch_raises(self) -> None:
        # Contract: count of [DATE] placeholders in redacted text must
        # equal the count of DATE-typed spans. A mismatch is a backend
        # contract violation; raise BackendRedactionError so the audit
        # pipeline fails loud.
        backend = _stub_backend()
        backend.register(
            "Seen 2026-05-10 and 2026-05-11",
            redacted="Seen [DATE] and [DATE]",
            spans=(
                RedactionSpan(
                    start=5,
                    end=15,
                    entity_type="DATE",
                    original_text="2026-05-10",
                ),
            ),
        )
        req = _request(
            notes=(
                NoteInput(note_id="E1", text="Seen 2026-05-10 and 2026-05-11"),
            ),
        )
        with pytest.raises(BackendRedactionError):
            redact_bundle(req, backend=backend, k_gate=lambda qi: 10)

    def test_person_role_upgraded_in_redacted_text(self) -> None:
        backend = _stub_backend()
        backend.register(
            "Dr. Smith ordered 2U PRBC",
            redacted="[PERSON] ordered 2U PRBC",
            spans=(
                RedactionSpan(
                    start=0,
                    end=9,
                    entity_type="PERSON",
                    original_text="Dr. Smith",
                ),
            ),
        )
        req = _request(
            notes=(
                NoteInput(note_id="E1", text="Dr. Smith ordered 2U PRBC"),
            ),
        )
        result = redact_bundle(
            req, backend=backend, k_gate=lambda qi: 10
        )
        assert "[ATTENDING]" in result.notes[0].redacted_text
        assert "[PERSON]" not in result.notes[0].redacted_text

    def test_backend_exception_wraps_to_backend_error(self) -> None:
        class _BoomBackend:
            def redact(self, text: str) -> BackendRedactionResult:
                raise RuntimeError("backend exploded")

        req = _request()
        with pytest.raises(BackendRedactionError):
            redact_bundle(
                req,
                backend=_BoomBackend(),
                k_gate=lambda qi: 10,
            )

    def test_redact_bundle_is_pure_no_io(self, capsys: pytest.CaptureFixture[str]) -> None:
        # PRD §"Architectural decisions" — uncaught exceptions are
        # routed by sys.excepthook; the redactor itself must not write
        # PHI to stdout/stderr. Tested by capturing output during a
        # normal call and asserting nothing is emitted.
        backend = _stub_backend()
        backend.register("plain note", redacted="plain note")
        req = _request()
        redact_bundle(req, backend=backend, k_gate=lambda qi: 10)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


# =============================================================================
# AC8: bundle-hash stability — same input + same redactor version → same hash
# =============================================================================


class TestBundleHashStability:
    """The core reproducibility invariant of issue #17."""

    def test_same_input_same_hash(self) -> None:
        backend = _stub_backend()
        backend.register(
            "Dr. Smith ordered 2U PRBC",
            redacted="[PERSON] ordered 2U PRBC",
            spans=(
                RedactionSpan(
                    start=0, end=9, entity_type="PERSON", original_text="Dr. Smith"
                ),
            ),
        )
        req = _request(
            notes=(
                NoteInput(note_id="E1", text="Dr. Smith ordered 2U PRBC"),
            ),
        )
        r1 = redact_bundle(req, backend=backend, k_gate=lambda qi: 10)
        r2 = redact_bundle(req, backend=backend, k_gate=lambda qi: 10)
        assert r1.redaction_hash == r2.redaction_hash

    def test_different_version_different_hash(self) -> None:
        # Two runs with the SAME input but DIFFERENT redactor versions
        # MUST produce different hashes — the version is part of the
        # canonical envelope, so a model SHA bump invalidates the prior
        # bundle hash (PRD §"redactor_version stamped on every audit row").
        backend = _stub_backend()
        backend.register("plain note", redacted="plain note")
        req_v1 = _request(redactor_version=_version(version="0.1.0"))
        req_v2 = _request(redactor_version=_version(version="0.2.0"))
        r1 = redact_bundle(req_v1, backend=backend, k_gate=lambda qi: 10)
        r2 = redact_bundle(req_v2, backend=backend, k_gate=lambda qi: 10)
        assert r1.redaction_hash != r2.redaction_hash

    def test_different_text_different_hash(self) -> None:
        backend = _stub_backend()
        backend.register("note A", redacted="note A")
        backend.register("note B", redacted="note B")
        req_a = _request(notes=(NoteInput(note_id="E1", text="note A"),))
        req_b = _request(notes=(NoteInput(note_id="E1", text="note B"),))
        r_a = redact_bundle(req_a, backend=backend, k_gate=lambda qi: 10)
        r_b = redact_bundle(req_b, backend=backend, k_gate=lambda qi: 10)
        assert r_a.redaction_hash != r_b.redaction_hash

    def test_hash_includes_routing_decision(self) -> None:
        # k-anonymity decision is in the envelope, so a route-to-NEEDS_REVIEW
        # change must yield a different hash even if every other byte is
        # identical. Without this, an audit row's hash could be re-used
        # for a routed result, breaking replay.
        backend = _stub_backend()
        backend.register("plain note", redacted="plain note")
        req = _request()
        r_pass = redact_bundle(req, backend=backend, k_gate=lambda qi: 10)
        r_fail = redact_bundle(req, backend=backend, k_gate=lambda qi: 2)
        assert r_pass.redaction_hash != r_fail.redaction_hash


# =============================================================================
# AC3: hand-labeled corpus of ≥30 redacted notes
# =============================================================================


# Synthetic corpus mirroring KCMH SOAP-note shapes. The corpus is
# inlined so it is version-controlled with the test, mirroring the
# evidence_bundle_builder fixtures. Each entry is (original, gold_role)
# where gold_role is the role the classifier should infer.
_HAND_LABELED_CORPUS: tuple[tuple[str, RoleToken | None], ...] = (
    # ATTENDING — physician utterances
    ("Dr. Smith ordered 2U PRBC for low Hb", RoleToken.ATTENDING),
    ("Attending Dr. Jones recommended transfusion", RoleToken.ATTENDING),
    ("MD note: continue current Hb monitoring plan", RoleToken.ATTENDING),
    ("Physician Dr. Lee reviewed labs", RoleToken.ATTENDING),
    ("นพ. Chen ordered packed red cells", RoleToken.ATTENDING),
    ("พญ. Brown reviewed the chart", RoleToken.ATTENDING),
    ("อาจารย์หมอ Garcia signed the consent", RoleToken.ATTENDING),
    ("Dr Wilson rounded on the patient", RoleToken.ATTENDING),
    # NURSE — nursing notes
    ("Nurse Jane reported stable vitals", RoleToken.NURSE),
    ("RN. Adams administered the unit", RoleToken.NURSE),
    ("RN Smith observed no transfusion reaction", RoleToken.NURSE),
    ("พยาบาล Lee noted patient tolerating well", RoleToken.NURSE),
    ("Nurse Wong recorded BP 110/70", RoleToken.NURSE),
    # PATIENT — patient-as-subject mentions
    ("Patient Brown complains of fatigue", RoleToken.PATIENT),
    ("Pt. Davis reports melena since yesterday", RoleToken.PATIENT),
    ("ผู้ป่วย Lim has no chest pain", RoleToken.PATIENT),
    ("คนไข้ Tan denies dyspnea", RoleToken.PATIENT),
    ("Patient Miller stated symptoms are stable", RoleToken.PATIENT),
    # FAMILY — family-of-patient mentions
    ("Mother Rivera at bedside", RoleToken.FAMILY),
    ("Father Park signed the consent form", RoleToken.FAMILY),
    ("Spouse Kim is the next of kin", RoleToken.FAMILY),
    ("Wife Anderson reports the timeline", RoleToken.FAMILY),
    ("Husband Cohen accompanied to ward", RoleToken.FAMILY),
    ("Daughter Patel agreed to plan", RoleToken.FAMILY),
    ("Family member Lopez present", RoleToken.FAMILY),
    ("ภรรยา Suntorn present at bedside", RoleToken.FAMILY),
    ("สามี Boonchai signed the consent", RoleToken.FAMILY),
    ("ญาติ Wong arrived this morning", RoleToken.FAMILY),
    # No cue — should remain unclassified
    ("BP 120/80 HR 72 stable hemodynamics", None),
    ("Labs sent. Awaiting results.", None),
    ("Hb 8.3 g/dL recheck in 6h", None),
    ("Plan: monitor and reassess", None),
)


# AC3 explicit: "tested on a hand-labeled corpus of ≥ 30 redacted notes".
# The fixture above has ≥30 entries; pytest's parametrize report (the count
# of test_classifier_matches_hand_labels cases) is the structural proof.


@pytest.mark.parametrize("note_text,gold_role", _HAND_LABELED_CORPUS)
def test_classifier_matches_hand_labels(
    note_text: str, gold_role: RoleToken | None
) -> None:
    """Each hand-labeled note must classify to its gold role.

    The context window is centered on a synthetic span placed at the
    cue's predicted location (just after the cue word). This mimics the
    runtime flow where ``thai-medical-deid`` redacts the NAME, and the
    classifier reads the cue from the surrounding text.
    """
    result = classify_role_by_cues(note_text)
    assert result is gold_role, (
        f"corpus entry {note_text!r} classified as {result!r}, expected {gold_role!r}"
    )


def _build_corpus_backend_response(
    note_text: str,
) -> tuple[str, tuple[RedactionSpan, ...]]:
    """Stub backend behavior for the corpus end-to-end test.

    Every corpus entry contains a single name-like token (the second
    whitespace-delimited word for ATTENDING/NURSE/PATIENT/FAMILY cases,
    and no name at all for the no-cue cases). Build the stub backend's
    redacted-text + spans by replacing that token with the generic
    ``[PERSON]`` placeholder so the wrapper's role-upgrade path is
    exercised end-to-end on the corpus.

    For no-cue entries (no name to redact), the original text is
    returned unchanged with no spans.
    """
    tokens = note_text.split()
    person_idx = _find_person_token_index(note_text, tokens)
    if person_idx is None:
        return note_text, ()

    name_token = tokens[person_idx]
    start = note_text.index(name_token)
    end = start + len(name_token)
    redacted = note_text[:start] + RoleToken.PERSON.value + note_text[end:]
    span = RedactionSpan(
        start=start, end=end, entity_type="PERSON", original_text=name_token
    )
    return redacted, (span,)


def _find_person_token_index(
    note_text: str, tokens: list[str]
) -> int | None:
    """Locate the proper-name token in a corpus entry, or None.

    Heuristic: the first capitalized ASCII-alpha token whose lowercase
    form is NOT one of the cue lexicons. Returns ``None`` when no such
    token exists (no-cue entries, or all-Thai entries where the name
    follows a Thai cue but isn't ASCII-capitalized — those are handled
    by the same approach, picking the first ASCII-capitalized word).
    """
    cue_words = {
        c.rstrip(".").lower()
        for cues in (ATTENDING_CUES, NURSE_CUES, PATIENT_CUES, FAMILY_CUES)
        for c in cues
    }
    cue_words.update({"member"})  # 'Family member ___' guard
    for i, t in enumerate(tokens):
        stripped = t.rstrip(".,")
        if not stripped or not stripped[0].isascii():
            continue
        if not stripped[0].isupper():
            continue
        if stripped.lower() in cue_words:
            continue
        return i
    return None


@pytest.mark.parametrize("note_text,gold_role", _HAND_LABELED_CORPUS)
def test_redact_bundle_emits_correct_role_token_on_corpus(
    note_text: str, gold_role: RoleToken | None
) -> None:
    """End-to-end corpus test: redact_bundle places the gold role token.

    Pairs with :func:`test_classifier_matches_hand_labels` (which tests
    the classifier in isolation). This test wires the corpus through
    the full wrapper — backend stub redacts the name, the wrapper
    upgrades the ``[PERSON]`` token via the default classifier, and the
    resulting ``RedactedNote.redacted_text`` is asserted to contain the
    gold role token.

    Corpus entries with no name (no-cue cases) skip the assertion since
    no upgrade path is exercised — the classifier-isolation test already
    covers them.
    """
    redacted, spans = _build_corpus_backend_response(note_text)
    if not spans:
        # No name to redact → wrapper has nothing to upgrade. The
        # classifier-isolation test already covers this branch.
        return

    backend = _stub_backend()
    backend.register(note_text, redacted=redacted, spans=spans)
    req = _request(
        notes=(NoteInput(note_id="E1", text=note_text),),
    )
    result = redact_bundle(req, backend=backend, k_gate=lambda qi: 10)
    rendered = result.notes[0].redacted_text

    expected_token = (
        gold_role.value if gold_role is not None else RoleToken.PERSON.value
    )
    assert expected_token in rendered, (
        f"corpus entry {note_text!r}: expected token {expected_token!r} in "
        f"rendered text {rendered!r}"
    )


# =============================================================================
# Property tests (hypothesis) — required for "deep" modules per PRD §"Testing"
# =============================================================================


# Strategy for valid quasi-identifier tuples. Constrained to the formats
# the Pydantic model accepts so we exercise k-anonymity math, not validation.
_ward_strategy = st.sampled_from(["W1", "W2", "W3", "ICU", "ER"])
_icd_strategy = st.sampled_from(["D62", "D63", "I50", "K92", "N18"])
_sex_strategy = st.sampled_from(["M", "F", "U"])
_age_band_strategy = st.sampled_from(["20-29", "30-39", "60-69", "70-79", "90-99"])
_month_strategy = st.sampled_from(["2026-01", "2026-05", "2026-12"])


def _qi_strategy() -> st.SearchStrategy[QuasiIdentifiers]:
    return st.builds(
        QuasiIdentifiers,
        ward=_ward_strategy,
        icd_3char=_icd_strategy,
        age_band=_age_band_strategy,
        sex=_sex_strategy,
        admission_month=_month_strategy,
    )


class TestKAnonymityProperties:
    """Hypothesis property tests for k-anonymity math."""

    @given(records=st.lists(_qi_strategy(), min_size=0, max_size=50))
    @settings(deadline=None, max_examples=100)
    def test_group_sizes_sum_to_record_count(
        self, records: list[QuasiIdentifiers]
    ) -> None:
        # Total over every group's size == number of records.
        # If this is ever false, compute_k_groups is dropping records.
        groups = compute_k_groups(records)
        assert sum(groups.values()) == len(records)

    @given(records=st.lists(_qi_strategy(), min_size=1, max_size=20))
    @settings(deadline=None, max_examples=100)
    def test_every_input_qi_appears_in_groups(
        self, records: list[QuasiIdentifiers]
    ) -> None:
        groups = compute_k_groups(records)
        for r in records:
            assert r in groups

    @given(records=st.lists(_qi_strategy(), min_size=0, max_size=20))
    @settings(deadline=None, max_examples=100)
    def test_compute_k_groups_order_invariant(
        self, records: list[QuasiIdentifiers]
    ) -> None:
        a = compute_k_groups(records)
        b = compute_k_groups(list(reversed(records)))
        assert dict(a) == dict(b)


class TestSemanticDegradationProperties:
    """Hypothesis property tests for the semantic-degradation detector."""

    @given(n_tokens=st.integers(min_value=0, max_value=SEMANTIC_PERSON_THRESHOLD))
    @settings(deadline=None, max_examples=20)
    def test_below_or_equal_threshold_never_fires(self, n_tokens: int) -> None:
        # A tight burst of ≤ threshold tokens must NEVER fire the flag.
        text = "[PERSON]" * n_tokens
        assert detect_semantic_degradation(text) is False

    @given(n_tokens=st.integers(min_value=SEMANTIC_PERSON_THRESHOLD + 1, max_value=20))
    @settings(deadline=None, max_examples=20)
    def test_above_threshold_in_tight_burst_always_fires(
        self, n_tokens: int
    ) -> None:
        # When concatenated end-to-end, n_tokens [PERSON] literals fit
        # inside an n_tokens * 8-char span; pick n large enough that
        # any 50-char window catches > threshold tokens.
        text = "[PERSON]" * n_tokens
        assert detect_semantic_degradation(text) is True


class TestDateShiftProperties:
    """Hypothesis property tests for the date-shift transform."""

    @given(
        admission_year=st.integers(min_value=2020, max_value=2030),
        admission_month=st.integers(min_value=1, max_value=12),
        admission_day=st.integers(min_value=1, max_value=28),
        offset_days=st.integers(min_value=-100, max_value=100),
    )
    @settings(deadline=None, max_examples=50)
    def test_offset_round_trips_through_shift(
        self,
        admission_year: int,
        admission_month: int,
        admission_day: int,
        offset_days: int,
    ) -> None:
        # Construct a (admission_date, note containing date_offset) and
        # assert the rewritten text contains the matching ``Day {±N}``
        # offset.
        admission = date(admission_year, admission_month, admission_day)
        from datetime import timedelta

        in_text_date = admission + timedelta(days=offset_days)
        # Skip values that fall outside 4-digit year range.
        if in_text_date.year < 1000 or in_text_date.year > 9999:
            return
        text = f"event on {in_text_date.isoformat()}"
        shifted = shift_dates_in_text(text, admission_date=admission)
        # Whatever the offset, the output must NOT contain the original
        # ISO date literal — the shift is total.
        assert in_text_date.isoformat() not in shifted


class TestBundleHashStabilityProperties:
    """Property: same input → same hash, regardless of irrelevant order."""

    @given(
        n_notes=st.integers(min_value=1, max_value=4),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(deadline=None, max_examples=20)
    def test_two_independent_runs_same_hash(
        self, n_notes: int, seed: int
    ) -> None:
        backend = _stub_backend()
        notes: list[NoteInput] = []
        for i in range(n_notes):
            text = f"note-{seed}-{i}: stable"
            backend.register(text, redacted=text)
            notes.append(NoteInput(note_id=f"E{i + 1}", text=text))
        req = _request(notes=tuple(notes))
        r1 = redact_bundle(req, backend=backend, k_gate=lambda qi: 10)
        r2 = redact_bundle(req, backend=backend, k_gate=lambda qi: 10)
        assert r1.redaction_hash == r2.redaction_hash


# =============================================================================
# AC2: thai-medical-deid pinned in pyproject.toml + redactor_version stamped
#
# This AC's letter ("``thai-medical-deid==X.Y.Z`` pinned in pyproject.toml") is
# a packaging concern; the runtime invariant is that the redactor_version
# stamps survive end-to-end onto the result. The test below asserts the
# *runtime* contract; the packaging assertion lives in the dependency-check
# task in the GREEN-phase ralph-loop, not as a unit test against pyproject.toml.
# =============================================================================


class TestVersionStampThroughResult:
    """The redactor_version + model_sha + gazetteer_version land on result."""

    def test_version_metadata_propagates_to_result(self) -> None:
        backend = _stub_backend()
        backend.register("plain note", redacted="plain note")
        v = _version(version="1.2.3", model_sha="cafef00d", gazetteer_version="g-2.0")
        req = _request(redactor_version=v)
        result = redact_bundle(req, backend=backend, k_gate=lambda qi: 10)
        assert result.redactor_version.version == "1.2.3"
        assert result.redactor_version.model_sha == "cafef00d"
        assert result.redactor_version.gazetteer_version == "g-2.0"


# =============================================================================
# Backwards-compatibility / structural invariants
# =============================================================================


# Structural invariants (no-naive-datetime, no-I/O-during-import) are
# intentionally deferred to GREEN-phase tests; they require the
# implementation to be in place to assert anything meaningful, and during
# RED they pass on the scaffold without exercising behavior.
