"""Deterministic platelet gate (docs plan §5.1, revised after clinical review).

The single public function :func:`classify_platelet` is the v1 platelet
appropriateness gate. It is intentionally minimal: unlike the RBC
:func:`bba.deterministic_classifier.classify`, it auto-clears NOTHING.

Precedence (top wins):

1. Count missing → mirror the RBC missing-Hb contract:
   * defer flag ON  → ``NEEDS_REVIEW`` (``plt_missing_defer_llm``)
   * defer flag OFF → ``INSUFFICIENT_EVIDENCE`` (``plt_missing``)
2. Count ≥ :data:`PLATELET_REVIEW_CEILING` → ``POTENTIALLY_INAPPROPRIATE``
   (``plt_ge_100``) — routes to LLM/review, NOT terminal. Agrees with the
   ">100k head-bleed: don't give" rule; safe because it is reviewed.
3. Count < :data:`PLATELET_REVIEW_CEILING` (including < 10) → ``NEEDS_REVIEW``
   (``plt_defer_llm``). The LLM reads notes for the indication (prophylaxis /
   LP / CVC / major surgery / active bleeding / dept) AND the exclusion
   populations (dengue-no-bleed, TTP, HIT, ITP, aplastic-no-bleed,
   snakebite-no-bleed), then applies the policy threshold.

3a. (flag ``PLATELET_PROPHYLAXIS_AUTOCLEAR_ENABLED`` / per-row
   ``enable_prophylaxis_autoclear``) FRESH count <
   :data:`PLATELET_PROPHYLAXIS_THRESHOLD` on an admission whose ICD-10 codes
   match :data:`PLATELET_PROPHYLAXIS_INDICATION_PREFIXES` and match NONE of
   :data:`PLATELET_PROPHYLAXIS_EXCLUSION_PREFIXES` → ``APPROPRIATE``
   (``plt_lt_10_heme_prophylaxis``), terminal. This is the medicine policy's
   "chemotherapy / stem-cell transplant, no bleeding: 10,000/µL" row on
   structured codes only; the withhold populations are excluded by code so
   the CR-C1 failure mode (clearing a dengue / TTP / HIT / ITP / aplastic
   patient at a low count) cannot occur. Anything the codes cannot settle
   (stale count, no indication code, any exclusion code) falls through to 3.

Why nothing ELSE is auto-cleared (§8/CR-C1): ``APPROPRIATE`` is a terminal
deterministic-final class — no LLM, no human. But the policies WITHHOLD
platelets at very low counts for the exclusion populations above (for TTP/HIT,
transfusion is actively harmful). In a dengue-endemic dataset a
``plt < 10 → APPROPRIATE`` auto-clear would systematically false-clear
inappropriate transfusions, so it was removed. Unlike Hb < 7 (≈no
contraindication population), a low platelet count has 6+ named exclusions.

Out of this pure gate's scope (handled at dispatch, docs plan §5.3 stage 4):
MTP-window suppression — platelet units co-ordered inside an active
massive-transfusion-protocol window are suppressed as independent audit units
upstream (the MTP co-order is a signal, never a second audit row). That needs
cross-order context this single-order gate does not have.
"""

from __future__ import annotations

from bba.platelet_classifier.models import (
    PlateletClassifierInputs,
    PlateletClassifierResult,
)

PLATELET_PROPHYLAXIS_THRESHOLD: float = 10.0
"""Count (×10³/µL) below which prophylactic platelets are indicated for a
chemotherapy / stem-cell-transplant patient without bleeding (Chula medicine
DRAFT, AABB/ICTMG 2025). A SEED pending hematology sign-off (docs plan §7)."""

PLATELET_PROPHYLAXIS_INDICATION_PREFIXES: tuple[str, ...] = (
    "C8",  # lymphoma
    "C9",  # leukaemia, myeloma, plasma-cell neoplasms
    "D611",  # drug-induced aplastic anaemia = chemotherapy marrow suppression
    "Z511",  # chemotherapy session for neoplasm
    "Z948",  # other transplanted organ / tissue status (stem cells)
    "T860",  # bone-marrow transplant complication
)
"""Dotless ICD-10 prefixes that place an admission in the chemo / HSCT /
treatment-marrow-failure population the 10k prophylaxis row covers. ``D61.1``
is deliberately an INDICATION, not an exclusion: in this dataset it codes
chemotherapy pancytopenia, the exact patient the row protects (hematology
sign-off 2026-09-20, decision 2)."""

PLATELET_PROPHYLAXIS_CONDITIONAL_INDICATION_PREFIXES: tuple[str, ...] = ("D46",)
"""Dotless ICD-10 prefixes that qualify ONLY with chemotherapy evidence on the
same admission: a ``Z51.1`` code or ``has_recent_chemo_med``. MDS on
supportive care alone is chronic marrow failure, where the medicine draft
(after NICE) gives no prophylactic platelets; MDS on treatment falls under the
10k row (hematology sign-off 2026-09-20, decision 3)."""

_CHEMO_SESSION_PREFIX: tuple[str, ...] = ("Z511",)

PLATELET_PROPHYLAXIS_EXCLUSION_PREFIXES: tuple[str, ...] = (
    "D693",  # idiopathic thrombocytopenic purpura
    "M311",  # thrombotic microangiopathy / TTP
    "D593",  # haemolytic-uraemic syndrome (a thrombotic microangiopathy; sign-off decision 5)
    "D758",  # other specified blood disorders (HIT lands here in WHO ICD-10)
    "D610",  # constitutional aplastic anaemia
    "D612",  # aplastic anaemia due to other external agents
    "D613",  # idiopathic aplastic anaemia
    "D618",  # other specified aplastic anaemias
    "D619",  # aplastic anaemia, unspecified
    "A90",  # dengue fever
    "A91",  # dengue haemorrhagic fever
    "A97",  # dengue (ICD-10 2019 recode)
    "T630",  # snake venom
)
"""Dotless ICD-10 prefixes of the populations every Chula draft WITHHOLDS
prophylactic platelets from (or where transfusion is harmful). Any match
blocks the auto-clear and the order defers to the LLM, which reads the notes
for bleeding. Fail-closed: a broad prefix (``D758``) only ever routes to
review, never clears."""

PLATELET_REVIEW_CEILING: float = 100.0
"""Count (×10³/µL) at or above which a platelet order is
``POTENTIALLY_INAPPROPRIATE`` (routes to review). A SEED pending clinician
sign-off (docs plan §7). OPEN (§5.4): confirm 100 vs an 80-100k band for the
high-bleeding-risk surgical rule — surgical indications permit up to 80-100k,
so the ceiling may need to drop to 80. Encoded as a single named constant so
that ruling changes one line."""


def _dotless(code: str) -> str:
    return code.strip().upper().replace(".", "")


def _matches_any(codes: tuple[str, ...], prefixes: tuple[str, ...]) -> bool:
    return any(_dotless(c).startswith(prefixes) for c in codes if c and c.strip())


def qualifies_for_prophylaxis_autoclear(inputs: PlateletClassifierInputs) -> bool:
    """True iff every structured condition of rule 3a holds (see module doc).

    Pure and side-effect free; exposed so the pilot can list the rows the
    flag WOULD clear before it is switched on.
    """
    count = inputs.platelet_count
    codes = inputs.diagnosis_codes
    chemo_evidence = inputs.has_recent_chemo_med or _matches_any(
        codes, _CHEMO_SESSION_PREFIX
    )
    indicated = _matches_any(codes, PLATELET_PROPHYLAXIS_INDICATION_PREFIXES) or (
        chemo_evidence
        and _matches_any(codes, PLATELET_PROPHYLAXIS_CONDITIONAL_INDICATION_PREFIXES)
    )
    return (
        inputs.enable_prophylaxis_autoclear
        and count is not None
        and count < PLATELET_PROPHYLAXIS_THRESHOLD
        and inputs.platelet_freshness == "fresh"
        and indicated
        and not _matches_any(codes, PLATELET_PROPHYLAXIS_EXCLUSION_PREFIXES)
    )


def classify_platelet(
    inputs: PlateletClassifierInputs,
) -> PlateletClassifierResult:
    """Return the deterministic platelet classification for ``inputs``.

    Pure function — same input always yields the same result; no module-global
    state, no mutation, never raises on a well-formed
    :class:`PlateletClassifierInputs`. See the module docstring for precedence.

    Invariant: never returns ``INAPPROPRIATE``, and never returns
    ``APPROPRIATE`` unless ``inputs.enable_prophylaxis_autoclear`` is set AND
    :func:`qualifies_for_prophylaxis_autoclear` holds. With the flag off every
    present-count verdict (``POTENTIALLY_INAPPROPRIATE`` / ``NEEDS_REVIEW``)
    routes onward exactly as in v1.
    """
    count = inputs.platelet_count

    # 1. Missing count — mirror the RBC missing-Hb contract.
    if count is None:
        if inputs.enable_missing_platelet_defer:
            return PlateletClassifierResult(
                classification="NEEDS_REVIEW",
                review_ceiling=None,
                rationale="plt_missing_defer_llm",
            )
        return PlateletClassifierResult(
            classification="INSUFFICIENT_EVIDENCE",
            review_ceiling=None,
            rationale="plt_missing",
        )

    # 2. At/above the review ceiling — reviewed, not auto-condemned.
    if count >= PLATELET_REVIEW_CEILING:
        return PlateletClassifierResult(
            classification="POTENTIALLY_INAPPROPRIATE",
            review_ceiling=PLATELET_REVIEW_CEILING,
            rationale="plt_ge_100",
        )

    # 3a. Flag-gated cohort prophylaxis clear (see module doc). Every condition
    #     is a structured code or a lookup bucket; nothing is read from notes.
    if qualifies_for_prophylaxis_autoclear(inputs):
        return PlateletClassifierResult(
            classification="APPROPRIATE",
            review_ceiling=PLATELET_REVIEW_CEILING,
            rationale="plt_lt_10_heme_prophylaxis",
        )

    # 3. Below the ceiling (incl. < 10) — defer to the LLM. Nothing else is
    #    auto-cleared: the indication and the exclusion populations are
    #    otherwise free-text / context-dependent.
    return PlateletClassifierResult(
        classification="NEEDS_REVIEW",
        review_ceiling=PLATELET_REVIEW_CEILING,
        rationale="plt_defer_llm",
    )


__all__ = (
    "PLATELET_PROPHYLAXIS_CONDITIONAL_INDICATION_PREFIXES",
    "PLATELET_PROPHYLAXIS_EXCLUSION_PREFIXES",
    "PLATELET_PROPHYLAXIS_INDICATION_PREFIXES",
    "PLATELET_PROPHYLAXIS_THRESHOLD",
    "PLATELET_REVIEW_CEILING",
    "classify_platelet",
    "qualifies_for_prophylaxis_autoclear",
)
