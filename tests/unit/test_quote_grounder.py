"""RED-phase failing tests for issue #18 (bba.quote_grounder).

Each ``class`` maps to one acceptance criterion in the issue body. Tests
assert contracts (the WHY), not implementation choices — PRD §"Testing
Decisions" + §"Anti-hallucination is layered, not single-shot.".

No implementation exists yet; every test MUST fail in this scaffold commit.
The module-level imports double as the public-API surface check: a missing
re-export fails collection before any test runs.
"""

from __future__ import annotations

import io
import sys
import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from bba.quote_grounder import (
    MIN_QUOTE_LENGTH,
    Citation,
    ConfusionMatrix,
    EvidenceSource,
    LabTuple,
    NLIEntailmentGate,
    Verdict,
    VerdictReason,
    VerdictSequence,
    VerdictTuple,
    confusion_matrix,
    contiguous_match,
    find_cited_source,
    lab_tuple_match,
    min_length_ok,
    nfc_normalize,
    verify_citation,
    verify_citations,
    within_doc_unique,
)


# =============================================================================
# Test fixtures: synthetic but representative bundle.
#
# Each helper builds a minimal correct EvidenceSource/Citation so the focus
# of each test is the specific layer or adversarial property under test.
# =============================================================================


# A long-enough source text containing a quote ≥ MIN_QUOTE_LENGTH chars.
_SOURCE_E1_TEXT = (
    "Patient has melena since yesterday with Hb 8.3 g/dL on admission. "
    "No active bleeding observed during physical examination today. "
    "Plan: transfuse 1 unit PRBC and recheck Hb in 6 hours."
)
_SOURCE_E2_TEXT = (
    "Cardiology consult note: stable angina, no acute coronary syndrome. "
    "Hb trending down from 10.2 to 8.7 g/dL over 48 hours; cause unclear. "
    "Plan: GI workup to rule out occult bleeding source."
)
_QUOTE_E1_VALID = "Patient has melena since yesterday with Hb 8.3 g/dL"
"""≥ 25 chars, contiguous in _SOURCE_E1_TEXT, occurs exactly once. Canonical
positive fixture: every layer must accept this quote when cited as ``E1``."""


def _src(source_id: str, text: str) -> EvidenceSource:
    return EvidenceSource(source_id=source_id, text=text)


def _cit(quote: str, cited_id: str, *, lab_tuple: LabTuple | None = None) -> Citation:
    return Citation(quote=quote, cited_id=cited_id, lab_tuple=lab_tuple)


def _bundle() -> tuple[EvidenceSource, ...]:
    return (
        _src("E1", _SOURCE_E1_TEXT),
        _src("E2", _SOURCE_E2_TEXT),
    )


# =============================================================================
# Public-API surface (collection check).
#
# The imports at the top of this file ARE the surface check; if any re-export
# is missing, pytest collection fails before any test runs. Constant and
# enum-value assertions are intentionally absent — they pass without exercising
# any unimplemented behavior, which would violate the RED-phase contract.
# =============================================================================


# =============================================================================
# Layer 1: NFC normalization.
#
# Thai medical text appears in both NFC and NFD forms across HOSxP exports
# (issue #18 adversarial fixture). Layer 1 collapses both into NFC before
# anything else compares them. The function must be a pure transformer; the
# substring check that consumes it lives in Layer 2.
# =============================================================================


class TestLayer1NFCNormalization:
    """Layer 1 — Unicode NFC normalization (idempotent, pure)."""

    def test_nfc_idempotent_on_ascii(self) -> None:
        # ASCII is its own NFC form: any change here is a regression.
        assert nfc_normalize("hello") == "hello"

    def test_nfc_normalizes_nfd_to_nfc(self) -> None:
        # Thai 'รัก' encoded in NFD differs byte-for-byte from NFC; after
        # normalization both forms collapse to the canonical NFC.
        nfd = unicodedata.normalize("NFD", "รัก")
        nfc = unicodedata.normalize("NFC", "รัก")
        # If the platform's source-string form is already NFC, NFD encoding
        # still differs; the assertion is about the function's output, not
        # the input shape.
        assert nfc_normalize(nfd) == nfc

    def test_nfc_is_idempotent(self) -> None:
        # f(f(x)) == f(x) — the canonical property of a normalizer.
        once = nfc_normalize("BP 120/70 ผู้ป่วยรู้สึกตัวดี")
        twice = nfc_normalize(once)
        assert once == twice

    def test_nfc_empty_string(self) -> None:
        # Total function: empty input is a valid input, not an error.
        assert nfc_normalize("") == ""


# =============================================================================
# Layer 2: Contiguous substring match.
#
# Defeats concatenated-quote attacks where the LLM glues together two
# non-adjacent spans into a single "quote" that does not appear as a
# contiguous run in the source.
# =============================================================================


class TestLayer2ContiguousMatch:
    """Layer 2 — contiguous-substring containment (post-NFC)."""

    def test_substring_present(self) -> None:
        assert contiguous_match(_QUOTE_E1_VALID, _SOURCE_E1_TEXT) is True

    def test_substring_absent(self) -> None:
        # The quote contains a word that does NOT appear in the source.
        assert (
            contiguous_match(
                "Patient has hematuria since yesterday morning today",
                _SOURCE_E1_TEXT,
            )
            is False
        )

    def test_concatenated_non_adjacent_spans_rejected(self) -> None:
        # Splice "Patient has melena" + "transfuse 1 unit PRBC" (non-adjacent
        # spans in _SOURCE_E1_TEXT) into one quote. Neither word changes; the
        # span-as-a-whole simply does not appear contiguously.
        concatenated = "Patient has melena transfuse 1 unit PRBC and recheck"
        assert contiguous_match(concatenated, _SOURCE_E1_TEXT) is False

    def test_empty_quote_matches_or_rejected_consistently(self) -> None:
        # ``""`` is a contiguous substring of every string under Python's
        # ``in`` operator; the layer must NOT use this as a backdoor pass.
        # The combined verifier rejects on EMPTY_QUOTE before reaching this
        # layer, but the layer's own contract is that it returns ``False`` for
        # an empty quote (defense in depth).
        assert contiguous_match("", _SOURCE_E1_TEXT) is False


# =============================================================================
# Layer 3: Strict cited_id match.
#
# Defeats cross-source attribution: the LLM citing source ``E2`` for a quote
# that only appears in ``E1``. The verifier must compare against the EXACT
# cited source, not "any source in bundle".
# =============================================================================


class TestLayer3StrictIdMatch:
    """Layer 3 — find_cited_source resolves cited_id → source uniquely."""

    def test_finds_matching_source(self) -> None:
        sources = _bundle()
        chosen = find_cited_source("E1", sources)
        assert chosen is not None
        assert chosen.source_id == "E1"

    def test_returns_none_for_unknown_id(self) -> None:
        # The LLM hallucinated a cited_id ("E99") that is not in the bundle.
        # The verifier must not silently fall through to another source.
        assert find_cited_source("E99", _bundle()) is None

    def test_returns_none_for_empty_bundle(self) -> None:
        assert find_cited_source("E1", ()) is None

    def test_case_sensitive_id_match(self) -> None:
        # cited_id is an opaque identifier produced by the bundle builder;
        # case must match exactly so a typo cannot silently route the quote
        # to the wrong source.
        sources = (_src("E1", _SOURCE_E1_TEXT),)
        assert find_cited_source("e1", sources) is None


# =============================================================================
# Layer 4: Within-document uniqueness.
#
# Defeats short-common-phrase attacks: "no bleeding" appearing elsewhere in
# the SAME source, in unrelated context. A quote that occurs ≥ 2 times in
# the cited source is rejected because the LLM cannot disambiguate which
# occurrence it meant — and one of them may be in unrelated context.
# =============================================================================


class TestLayer4WithinDocUniqueness:
    """Layer 4 — quote occurs exactly once in cited source."""

    def test_unique_quote_passes(self) -> None:
        assert within_doc_unique(_QUOTE_E1_VALID, _SOURCE_E1_TEXT) is True

    def test_repeated_quote_rejected(self) -> None:
        # The phrase "no active bleeding" appears twice in the same source.
        # Within-doc uniqueness rejects it even though it is contiguous.
        # Case-sensitive: NFC preserves case, so both occurrences must match
        # the quote's casing exactly to count as a repeat.
        source = (
            "no active bleeding on day 1. Repeat exam: no active bleeding on day 2."
        )
        assert within_doc_unique("no active bleeding", source) is False

    def test_absent_quote_rejected(self) -> None:
        # Zero occurrences is also a failure of uniqueness ("exactly one").
        assert (
            within_doc_unique("hematuria since morning", _SOURCE_E1_TEXT) is False
        )


# =============================================================================
# Layer 5: Minimum length.
#
# Defeats trivial-substring attacks (single word, short phrase). Below 25
# NFC chars, even a contiguous unique match is suspect — co-occurrence
# probability in unrelated content is too high.
# =============================================================================


class TestLayer5MinimumLength:
    """Layer 5 — NFC-length ≥ MIN_QUOTE_LENGTH (default 25)."""

    def test_exactly_at_threshold_passes(self) -> None:
        # Boundary: a 25-char quote is accepted (≥, not >).
        quote = "a" * MIN_QUOTE_LENGTH
        assert min_length_ok(quote) is True

    def test_one_below_threshold_rejected(self) -> None:
        quote = "a" * (MIN_QUOTE_LENGTH - 1)
        assert min_length_ok(quote) is False

    def test_empty_quote_rejected(self) -> None:
        assert min_length_ok("") is False

    def test_threshold_parameterized(self) -> None:
        # The eval harness sweeps the threshold during calibration. Verify
        # the kwarg actually overrides the default.
        assert min_length_ok("short", minimum=5) is True
        assert min_length_ok("short", minimum=6) is False


# =============================================================================
# Layer 6: Numeric-tuple grounding for lab citations.
#
# Defeats numeric-paraphrase attacks: the LLM writes "Hgb=8.3" while the
# source says "Hb 8.3 g/dL". Without tuple grounding, the quote fails
# Layer 2 (no contiguous match) even though it is semantically correct.
# The tuple is the structured fallback: same (analyte, value, unit) triple
# in both? Then accept.
# =============================================================================


class TestLayer6LabTupleMatch:
    """Layer 6 — (analyte, value, unit) tuple equivalence after aliasing."""

    def test_exact_match_passes(self) -> None:
        # Source contains "Hb 8.3 g/dL"; tuple references same triple.
        tup = LabTuple(analyte="Hb", value=8.3, unit="g/dL")
        assert lab_tuple_match(tup, "Patient Hb 8.3 g/dL on admission.") is True

    def test_analyte_alias_hgb_equals_hb(self) -> None:
        # The canonical adversarial fixture from the issue body:
        # "Hgb=8.3 vs Hb 8.3 g/dL". The grounder must collapse "Hgb" and
        # "Hb" to one analyte key so semantically-identical phrasings match.
        tup = LabTuple(analyte="Hgb", value=8.3, unit="g/dL")
        assert lab_tuple_match(tup, "Patient Hb 8.3 g/dL on admission.") is True

    def test_unit_case_normalization(self) -> None:
        # "g/dl" vs "g/dL" — same unit, different case. Reject would be a
        # paraphrase failure on a structured field.
        tup = LabTuple(analyte="Hb", value=8.3, unit="g/dl")
        assert lab_tuple_match(tup, "Hb 8.3 g/dL noted.") is True

    def test_value_mismatch_rejected(self) -> None:
        # Same analyte and unit, different value — this IS the
        # hallucination case the layer must catch.
        tup = LabTuple(analyte="Hb", value=7.0, unit="g/dL")
        assert lab_tuple_match(tup, "Patient Hb 8.3 g/dL on admission.") is False

    def test_analyte_mismatch_rejected(self) -> None:
        # WBC tuple cited against an Hb-only source — wrong analyte even if
        # a similar number happens to appear nearby.
        tup = LabTuple(analyte="WBC", value=8.3, unit="K/uL")
        assert lab_tuple_match(tup, "Patient Hb 8.3 g/dL on admission.") is False

    def test_unit_mismatch_rejected(self) -> None:
        # Same analyte and value, incompatible unit.
        tup = LabTuple(analyte="Hb", value=8.3, unit="mmol/L")
        assert lab_tuple_match(tup, "Patient Hb 8.3 g/dL on admission.") is False


# =============================================================================
# Layer 7: Optional medical-NLI semantic-entailment gate.
#
# PRD §9: "if unavailable, omit and accept the false-positive risk". The
# gate is a Protocol callable supplied by the caller; the grounder never
# imports a model. Tests verify the boundary contract, not any model.
# =============================================================================


class TestLayer7NLIEntailmentGate:
    """Layer 7 — optional NLI gate boundary contract."""

    def test_gate_invoked_with_source_and_quote(self) -> None:
        # Capture invocation to assert the verifier calls the gate as
        # gate(premise=source_text, hypothesis=quote) — the documented
        # contract in NLIEntailmentGate.__call__.
        calls: list[tuple[str, str]] = []

        def gate(*, premise: str, hypothesis: str) -> bool:
            calls.append((premise, hypothesis))
            return True

        sources = _bundle()
        cit = _cit(_QUOTE_E1_VALID, "E1")
        verdict = verify_citation(cit, sources, nli_gate=gate)
        # Gate runs on the cited source's text (E1), not E2.
        assert calls == [(_SOURCE_E1_TEXT, _QUOTE_E1_VALID)]
        assert verdict.passed is True

    def test_gate_rejection_propagates(self) -> None:
        # Gate returns False → verdict.passed=False with reason NLI_NOT_ENTAILED.
        def gate(*, premise: str, hypothesis: str) -> bool:
            return False

        verdict = verify_citation(_cit(_QUOTE_E1_VALID, "E1"), _bundle(), nli_gate=gate)
        assert verdict.passed is False
        assert verdict.reason == VerdictReason.NLI_NOT_ENTAILED

    def test_gate_omitted_means_layer_7_skipped(self) -> None:
        # PRD §9 explicitly allows omitting Layer 7. With nli_gate=None and
        # all other layers passing, the verdict is PASS.
        verdict = verify_citation(_cit(_QUOTE_E1_VALID, "E1"), _bundle())
        assert verdict.passed is True
        assert verdict.reason == VerdictReason.PASS


# =============================================================================
# Combined-pipeline contract: verify_citation runs all applicable layers and
# short-circuits on the first failed layer. The order of layers is part of
# the contract because the reason maps 1:1 to the reviewer-dashboard label.
# =============================================================================


class TestVerifyCitationCombined:
    """verify_citation integrates all six layers in canonical order."""

    def test_happy_path_passes(self) -> None:
        verdict = verify_citation(_cit(_QUOTE_E1_VALID, "E1"), _bundle())
        assert verdict.passed is True
        assert verdict.reason == VerdictReason.PASS
        # Verdict echoes the input citation for round-trip persistence.
        assert verdict.citation.quote == _QUOTE_E1_VALID
        assert verdict.citation.cited_id == "E1"

    def test_empty_quote_rejected_first(self) -> None:
        # Empty quote is the trivial-guard layer; runs BEFORE id lookup.
        verdict = verify_citation(_cit("", "E1"), _bundle())
        assert verdict.passed is False
        assert verdict.reason == VerdictReason.EMPTY_QUOTE

    def test_unknown_cited_id_rejected(self) -> None:
        verdict = verify_citation(_cit(_QUOTE_E1_VALID, "E99"), _bundle())
        assert verdict.passed is False
        assert verdict.reason == VerdictReason.CITED_ID_NOT_FOUND

    def test_too_short_quote_rejected(self) -> None:
        verdict = verify_citation(_cit("Patient has melena", "E1"), _bundle())
        assert verdict.passed is False
        assert verdict.reason == VerdictReason.TOO_SHORT

    def test_non_contiguous_rejected(self) -> None:
        # Long enough, cited id exists, but quote is not in source.
        verdict = verify_citation(
            _cit("XX " * 20 + "definitely-not-in-source", "E1"), _bundle()
        )
        assert verdict.passed is False
        assert verdict.reason == VerdictReason.NO_CONTIGUOUS_MATCH

    def test_non_unique_rejected(self) -> None:
        # A source where the quote occurs twice → NOT_UNIQUE.
        repeated = "no active bleeding observed yesterday. " * 2
        sources = (_src("E1", repeated + " followed by closing text."),)
        # Pick a substring that meets length AND appears twice.
        quote = "no active bleeding observed"
        # Pad to ≥ 25 chars; the repeating phrase is already 27.
        assert len(quote) >= MIN_QUOTE_LENGTH
        verdict = verify_citation(_cit(quote, "E1"), sources)
        assert verdict.passed is False
        assert verdict.reason == VerdictReason.NOT_UNIQUE

    def test_lab_tuple_mismatch_rejected(self) -> None:
        # Quote passes layers 1-5, but the structured tuple disagrees.
        tup = LabTuple(analyte="Hb", value=7.0, unit="g/dL")  # source says 8.3
        verdict = verify_citation(
            _cit(_QUOTE_E1_VALID, "E1", lab_tuple=tup), _bundle()
        )
        assert verdict.passed is False
        assert verdict.reason == VerdictReason.LAB_TUPLE_MISMATCH

    def test_layer_order_id_before_length(self) -> None:
        # When BOTH cited_id is unknown AND quote is too short, the reason
        # is CITED_ID_NOT_FOUND (Layer 3 ranks before Layer 5). The order is
        # part of the contract — reviewer-dashboard labels depend on it.
        verdict = verify_citation(_cit("short", "E99"), _bundle())
        assert verdict.reason == VerdictReason.CITED_ID_NOT_FOUND

    def test_layer_order_length_before_substring(self) -> None:
        # Below threshold AND not in any source → TOO_SHORT (Layer 5
        # short-circuits before the Layer 2 substring scan).
        verdict = verify_citation(_cit("nonexistent", "E1"), _bundle())
        assert verdict.reason == VerdictReason.TOO_SHORT


# =============================================================================
# Batch convenience: verify_citations preserves order, returns a tuple.
# =============================================================================


class TestVerifyCitationsBatch:
    """verify_citations preserves order and returns an immutable tuple."""

    def test_preserves_input_order(self) -> None:
        citations = [
            _cit("", "E1"),  # EMPTY_QUOTE
            _cit(_QUOTE_E1_VALID, "E1"),  # PASS
            _cit("short", "E99"),  # CITED_ID_NOT_FOUND
        ]
        verdicts = verify_citations(citations, _bundle())
        assert len(verdicts) == 3
        assert verdicts[0].reason == VerdictReason.EMPTY_QUOTE
        assert verdicts[1].reason == VerdictReason.PASS
        assert verdicts[2].reason == VerdictReason.CITED_ID_NOT_FOUND

    def test_returns_tuple_not_list(self) -> None:
        # Immutability contract — caller can never mutate the verdicts in
        # place between verification and persistence to audit_results.parquet.
        verdicts = verify_citations([_cit(_QUOTE_E1_VALID, "E1")], _bundle())
        assert isinstance(verdicts, tuple)

    def test_empty_citation_list_returns_empty_tuple(self) -> None:
        verdicts = verify_citations([], _bundle())
        assert verdicts == ()


# =============================================================================
# Adversarial fixture set (issue #18 AC §3).
#
# These are the specific attack shapes the verifier MUST catch. Each is a
# named fixture so a regression has a single test that fails with a
# self-describing name.
# =============================================================================


class TestAdversarialNFCvsNFD:
    """Same Thai quote in NFC vs NFD — must match either way after Layer 1."""

    def test_nfd_quote_against_nfc_source(self) -> None:
        # Build a long-enough Thai quote; encode source NFC, quote NFD.
        thai = "ผู้ป่วยมาด้วยอาการถ่ายดำมาตั้งแต่เมื่อวานนี้"
        assert len(thai) >= MIN_QUOTE_LENGTH
        source_nfc = unicodedata.normalize("NFC", thai)
        quote_nfd = unicodedata.normalize("NFD", thai)
        sources = (_src("E1", source_nfc),)
        verdict = verify_citation(_cit(quote_nfd, "E1"), sources)
        assert verdict.passed is True
        assert verdict.reason == VerdictReason.PASS

    def test_nfc_quote_against_nfd_source(self) -> None:
        # The reverse direction — Layer 1 must apply to BOTH sides.
        thai = "ผู้ป่วยมาด้วยอาการถ่ายดำมาตั้งแต่เมื่อวานนี้"
        source_nfd = unicodedata.normalize("NFD", thai)
        quote_nfc = unicodedata.normalize("NFC", thai)
        verdict = verify_citation(
            _cit(quote_nfc, "E1"), (_src("E1", source_nfd),)
        )
        assert verdict.passed is True


class TestAdversarialNumericParaphrase:
    """Hgb=8.3 vs Hb 8.3 g/dL — Layer 6 tuple grounding catches paraphrase."""

    def test_hgb_alias_for_hb(self) -> None:
        # Citation claims structured (Hgb, 8.3, g/dL); source phrases as
        # "Hb 8.3 g/dL". The quote itself need not appear in source — the
        # tuple alone is the grounding signal for a structured lab claim.
        source_text = (
            "Lab results: Hb 8.3 g/dL, WBC 9.2 K/uL, Platelets 230 K/uL. "
            "Clinically stable. Plan transfuse 1 unit PRBC."
        )
        # Use a quote that DOES appear contiguously and uniquely (so layers
        # 1-5 pass cleanly) and attach a paraphrased lab tuple on top.
        quote = "Lab results: Hb 8.3 g/dL, WBC 9.2 K/uL"
        assert len(quote) >= MIN_QUOTE_LENGTH
        tup = LabTuple(analyte="Hgb", value=8.3, unit="g/dL")
        verdict = verify_citation(
            _cit(quote, "E1", lab_tuple=tup), (_src("E1", source_text),)
        )
        assert verdict.passed is True

    def test_paraphrased_value_mismatch_caught(self) -> None:
        # The adversarial case: surface phrasing differs AND the numeric
        # value differs. The tuple layer must reject — paraphrase is on
        # tokens, not numbers.
        source_text = "Lab results: Hb 8.3 g/dL stable across draws."
        # Build a long-enough contiguous quote.
        quote = "Lab results: Hb 8.3 g/dL stable across"
        tup = LabTuple(analyte="Hgb", value=7.0, unit="g/dL")  # value mismatch
        verdict = verify_citation(
            _cit(quote, "E1", lab_tuple=tup), (_src("E1", source_text),)
        )
        assert verdict.passed is False
        assert verdict.reason == VerdictReason.LAB_TUPLE_MISMATCH


class TestAdversarialCrossSourceAttribution:
    """Quote appears in source E2, but the LLM cites it as E1."""

    def test_cross_source_attribution_rejected(self) -> None:
        # The quote is a verbatim contiguous unique substring of E2.
        # The LLM mis-cites it as E1. Layer 3 must catch this even though
        # Layer 2 against E2 would pass.
        quote = "Cardiology consult note: stable angina, no acute coronary"
        assert len(quote) >= MIN_QUOTE_LENGTH
        assert quote in _SOURCE_E2_TEXT  # sanity: present in E2
        assert quote not in _SOURCE_E1_TEXT  # not in E1
        verdict = verify_citation(_cit(quote, "E1"), _bundle())
        assert verdict.passed is False
        assert verdict.reason == VerdictReason.NO_CONTIGUOUS_MATCH


class TestAdversarialConcatenatedQuotes:
    """Quote splices two non-adjacent spans from one source."""

    def test_concatenated_quote_rejected(self) -> None:
        # Take two real fragments from E1 and join them — the seam is the
        # bug; no contiguous run in the source matches.
        quote = (
            "Patient has melena since yesterday and transfuse 1 unit PRBC"
        )
        # Each substring appears; the concatenation does not.
        assert "Patient has melena since yesterday" in _SOURCE_E1_TEXT
        assert "transfuse 1 unit PRBC" in _SOURCE_E1_TEXT
        assert quote not in _SOURCE_E1_TEXT
        verdict = verify_citation(_cit(quote, "E1"), _bundle())
        assert verdict.passed is False
        assert verdict.reason == VerdictReason.NO_CONTIGUOUS_MATCH


class TestAdversarialShortCommonPhrases:
    """``no bleeding`` cited in unrelated-context occurrence — Layer 5 catches
    the trivial-substring attack first; Layer 4 catches it if it sneaks past
    Layer 5 in some longer form."""

    def test_short_common_phrase_rejected_for_length(self) -> None:
        # "no bleeding" is 11 chars — fails Layer 5 trivially.
        verdict = verify_citation(_cit("no bleeding", "E1"), _bundle())
        assert verdict.passed is False
        assert verdict.reason == VerdictReason.TOO_SHORT

    def test_long_enough_but_repeated_phrase_rejected(self) -> None:
        # The phrase ≥ 25 chars but occurs twice in the source — the
        # disambiguation problem the LLM cannot resolve.
        repeated = (
            "no significant active bleeding noted today by exam. "
            "Repeat assessment 6 hours later: no significant active "
            "bleeding noted today by exam. Plan: discharge."
        )
        sources = (_src("E1", repeated),)
        verdict = verify_citation(
            _cit("no significant active bleeding noted today by exam", "E1"),
            sources,
        )
        assert verdict.passed is False
        assert verdict.reason == VerdictReason.NOT_UNIQUE


# =============================================================================
# Verifier-as-classifier: confusion-matrix output (issue #18 AC §4).
#
# The 200-row hand-labeled set lives in the eval harness; the grounder
# only supplies the math. The matrix is exact-count; bad-shape input is a
# loud failure, not a silent truncation.
# =============================================================================


class TestConfusionMatrix:
    """confusion_matrix: 2x2 counts from (verdicts, gold_labels)."""

    def _verdict(self, passed: bool) -> Verdict:
        return Verdict(
            passed=passed,
            reason=VerdictReason.PASS if passed else VerdictReason.NO_CONTIGUOUS_MATCH,
            citation=_cit(_QUOTE_E1_VALID, "E1"),
        )

    def test_balanced_input(self) -> None:
        # 4 verdicts: PP=TP, PN=FN, NP=FP, NN=TN. One each.
        verdicts = [
            self._verdict(True),  # TP (gold T, pred T)
            self._verdict(False),  # FN (gold T, pred F)
            self._verdict(True),  # FP (gold F, pred T)
            self._verdict(False),  # TN (gold F, pred F)
        ]
        gold = [True, True, False, False]
        cm = confusion_matrix(verdicts, gold)
        assert cm == ConfusionMatrix(
            true_positive=1, true_negative=1, false_positive=1, false_negative=1
        )

    def test_all_correct(self) -> None:
        verdicts = [self._verdict(True), self._verdict(False)]
        gold = [True, False]
        cm = confusion_matrix(verdicts, gold)
        assert cm.true_positive == 1
        assert cm.true_negative == 1
        assert cm.false_positive == 0
        assert cm.false_negative == 0

    def test_length_mismatch_raises(self) -> None:
        # Silently truncating to ``min(len(a), len(b))`` would corrupt the
        # metric the acceptance target is graded against. Loud failure.
        with pytest.raises(ValueError):
            confusion_matrix([self._verdict(True)], [True, False])

    def test_empty_inputs_zero_matrix(self) -> None:
        cm = confusion_matrix([], [])
        assert cm == ConfusionMatrix(
            true_positive=0, true_negative=0, false_positive=0, false_negative=0
        )


# =============================================================================
# Pure-function contract (issue #18 AC §5).
#
# No I/O. No global mutation. No dependency on the Anthropic SDK. These
# tests document the contract structurally — a regression that adds a print
# or imports anthropic would trip a named test, not just a code review.
# =============================================================================


class TestPureFunctionContract:
    """No I/O, no Anthropic SDK dep, deterministic."""

    def test_no_stdout_or_stderr_writes(self) -> None:
        # Redirect both streams; any byte produced by the verifier is a
        # contract violation (logging or print() leaked into the pure
        # function).
        buf_out, buf_err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = buf_out, buf_err
            verify_citation(_cit(_QUOTE_E1_VALID, "E1"), _bundle())
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        assert buf_out.getvalue() == ""
        assert buf_err.getvalue() == ""

    def test_determinism_same_input_same_verdict(self) -> None:
        # Two runs with the same inputs produce equal verdicts. Pure
        # function: no hidden state.
        sources = _bundle()
        cit = _cit(_QUOTE_E1_VALID, "E1")
        v1 = verify_citation(cit, sources)
        v2 = verify_citation(cit, sources)
        assert v1 == v2


# =============================================================================
# Property tests (issue #18 AC §6).
#
# "Any random insertion/deletion of 1 character in a valid quote → rejection."
# A 1-char edit breaks Layer 2 contiguity (the quote no longer appears as a
# run in the source). Layer 5 may also catch the deletion form; either
# failure is acceptable — the property is "rejection", not a specific reason.
# =============================================================================


class TestPropertyOneCharEditRejection:
    """One-character edits to a valid quote produce a rejection verdict."""

    @given(
        idx=st.integers(min_value=0, max_value=len(_QUOTE_E1_VALID) - 1),
        ch=st.characters(
            min_codepoint=0x21,
            max_codepoint=0x7E,
            blacklist_characters=set(_QUOTE_E1_VALID),
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_insertion_breaks_grounding(self, idx: int, ch: str) -> None:
        # Insert a character that does not appear in the source's contiguous
        # neighborhood — the result cannot be a contiguous substring of the
        # original source.
        mutated = _QUOTE_E1_VALID[:idx] + ch + _QUOTE_E1_VALID[idx:]
        verdict = verify_citation(_cit(mutated, "E1"), _bundle())
        assert verdict.passed is False

    @given(idx=st.integers(min_value=0, max_value=len(_QUOTE_E1_VALID) - 1))
    @settings(max_examples=50, deadline=None)
    def test_deletion_breaks_grounding(self, idx: int) -> None:
        # Deletion: drop one char. The result either fails Layer 2 (no
        # contiguous run) or Layer 5 (below MIN_QUOTE_LENGTH for the boundary
        # case). Either way: rejection.
        mutated = _QUOTE_E1_VALID[:idx] + _QUOTE_E1_VALID[idx + 1 :]
        verdict = verify_citation(_cit(mutated, "E1"), _bundle())
        assert verdict.passed is False


# =============================================================================
# Immutability contract — verifier output is structurally frozen.
#
# Verdicts persist to audit_results.parquet; mutating a verdict between
# verification and write would corrupt the persisted record. Rather than
# testing pydantic's frozen=True against a hand-constructed model (which
# passes trivially without any verifier behavior), this section asserts the
# end-to-end contract: the verifier's outputs cannot be mutated.
# =============================================================================


class TestVerifierOutputImmutability:
    """Verdicts and the tuple returned by verify_citations are immutable."""

    def test_verify_citation_output_is_frozen(self) -> None:
        verdict = verify_citation(_cit(_QUOTE_E1_VALID, "E1"), _bundle())
        with pytest.raises(ValidationError):
            verdict.passed = False  # type: ignore[misc]

    def test_verify_citations_output_tuple_unmutable(self) -> None:
        verdicts = verify_citations(
            [_cit(_QUOTE_E1_VALID, "E1")], _bundle()
        )
        # tuple has no __setitem__ — direct assertion of the immutability shape.
        with pytest.raises(TypeError):
            verdicts[0] = verdicts[0]  # type: ignore[index]


# =============================================================================
# Protocol contract for the NLI gate.
#
# NLIEntailmentGate is structural typing: any callable with the right
# signature is a valid gate. The verifier accepts user-supplied gates
# without requiring inheritance.
# =============================================================================


class TestNLIGateProtocol:
    """Any kw-only ``(premise, hypothesis) -> bool`` callable is a valid gate."""

    def test_function_satisfies_protocol(self) -> None:
        def gate(*, premise: str, hypothesis: str) -> bool:
            return True

        # Verifies the gate is accepted at the verify_citation boundary
        # without raising a type error (structural typing).
        verdict = verify_citation(
            _cit(_QUOTE_E1_VALID, "E1"), _bundle(), nli_gate=gate
        )
        assert verdict.reason == VerdictReason.PASS

    def test_class_with_call_satisfies_protocol(self) -> None:
        class Gate:
            def __call__(self, *, premise: str, hypothesis: str) -> bool:
                return True

        verdict = verify_citation(
            _cit(_QUOTE_E1_VALID, "E1"), _bundle(), nli_gate=Gate()
        )
        assert verdict.reason == VerdictReason.PASS


# =============================================================================
# Type-alias sanity: VerdictTuple / VerdictSequence are usable at boundaries.
# =============================================================================


class TestTypeAliasUsage:
    """VerdictTuple is the verify_citations output type;
    VerdictSequence is the confusion_matrix input type."""

    def test_verify_citations_returns_verdict_tuple_shape(self) -> None:
        result: VerdictTuple = verify_citations(
            [_cit(_QUOTE_E1_VALID, "E1")], _bundle()
        )
        assert isinstance(result, tuple)
        assert all(isinstance(v, Verdict) for v in result)

    def test_confusion_matrix_accepts_verdict_sequence(self) -> None:
        # Pass a list (Sequence[Verdict]) to demonstrate the wider alias.
        seq: VerdictSequence = [
            Verdict(
                passed=True,
                reason=VerdictReason.PASS,
                citation=_cit(_QUOTE_E1_VALID, "E1"),
            )
        ]
        cm = confusion_matrix(seq, [True])
        assert cm.true_positive == 1
