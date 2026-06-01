"""Top-level evidence-bundle pipeline (issue #16).

Orchestrates the per-source windowing, ranking, ID assignment, canonical
serialization, and hashing into a single :class:`EvidenceBundle`. The window
constants live here so every threshold the issue body names has exactly one
authoritative declaration in the codebase.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from bba.evidence_bundle_builder.canonical import bundle_hash, canonical_serialize
from bba.evidence_bundle_builder.exceptions import EvidenceBundleTooLargeError
from bba.evidence_bundle_builder.models import (
    DiagnosisRecord,
    EvidenceBundle,
    EvidenceInputs,
    EvidenceItem,
    EvidenceSource,
    FocusNote,
    HbRecord,
    MedRecord,
    OrderAnchor,
    ProgressNote,
    VitalsRecord,
)
from bba.evidence_bundle_builder.ranking import (
    SECTION_PRIORITY,
    parse_soap_sections,
    split_focus_notes_5_5,
)
from bba.evidence_bundle_builder.salience import med_salience
from bba.vitals_extractor.hemodynamic import scan_hemodynamics
from bba.vitals_extractor.models import (
    HemodynamicSummary,
    VasopressorMention,
    VitalsNote,
)


def _nfc(s: str) -> str:
    """NFC-normalize a sort-key string. Without this, tied-timestamp
    records whose text differs only by NFC/NFD encoding would order
    differently before canonical_serialize unifies them, leaking the
    input encoding into the bundle hash."""
    return unicodedata.normalize("NFC", s)


# =============================================================================
# Per-source windows (PRD §7 / issue #16 body)
#
# Every constant here is referenced by a test in
# ``tests/unit/test_evidence_bundle_builder.py``. Changing a value is a
# spec-level change — the windows are not tunable knobs, they encode the
# clinical "what's relevant to a transfusion decision" decision.
# =============================================================================

WINDOW_PROGRESS = timedelta(hours=24)
"""IPDADMPROGRESS window: ``[anchor - 24h, anchor + 24h]``."""

WINDOW_FOCUS = timedelta(hours=24)
"""IPDNRFOCUSDT window: ``[anchor - 24h, anchor + 24h]``, then 5+5 split."""

WINDOW_MED_BEFORE = timedelta(hours=72)
"""MED window backward edge: ``anchor - 72h`` (drug history shaping the order)."""

WINDOW_MED_AFTER = timedelta(hours=24)
"""MED window forward edge: ``anchor + 24h`` (post-order administration)."""

WINDOW_HB_BEFORE = timedelta(days=7)
"""Lab Hb history backward edge: ``anchor - 7d`` (mirrors :mod:`bba.hb_lookup`)."""

WINDOW_VITALS = timedelta(hours=6)
"""Vitals window: ``[anchor - 6h, anchor + 6h]`` (mirrors :mod:`bba.vitals_extractor`)."""

CAP_PROGRESS = 8
"""Maximum IPDADMPROGRESS entries in a bundle (issue #16 body)."""

CAP_FOCUS_BEFORE = 5
"""Maximum IPDNRFOCUSDT entries strictly before the order anchor."""

CAP_FOCUS_AFTER = 5
"""Maximum IPDNRFOCUSDT entries at or after the order anchor."""

DEFAULT_CHAR_CAP = 40000
"""Bundle character-count proxy for the LLM token budget (issue #16 / #76).

A char-count proxy is used instead of a tiktoken-style estimate because the
bundle goes to Anthropic's tokenizer (not OpenAI's) and the published Claude
tokenization is not stable enough to depend on for a hard cap.

Raised from the original 8 K (issue #16) to 40 K under issue #76: the old cap
was a root cause of Case 2 / REQNO 68012352 — narrative suppression plus an
8 K ceiling shed the vasopressor and MAP evidence the auditor needed. The
bundle now ships the prose narrative AND a pinned, truncation-exempt
hemodynamic summary, which needs the headroom. 40 K is PROVISIONAL: it will be
finalized against the pilot worst-case bundle measurement (issue #76 Task #3)
before any clinical use."""

EXEMPT_FROM_DROP: frozenset[EvidenceSource] = frozenset({"Hemodynamic", "Lab"})
"""Sources the whole-item truncation pass must NEVER drop (issue #76).

These are the load-bearing decision signals: ``Lab`` (Hb) is the decision-time
anemia value the deterministic classifier itself keys on, and ``Hemodynamic``
is the pinned MAP/vasopressor summary that was starved in Case 2 / REQNO
68012352. Both are bounded in count (Hb by the 7-day lookback window; the
Hemodynamic summary is a single item), so exempting them cannot make the
bundle grow without limit. If the exempt set plus the anchor envelope alone
still exceeds the cap, :func:`_enforce_char_cap` fails loud rather than
silently shedding a load-bearing signal.

Invariant: ``EXEMPT_FROM_DROP`` and :data:`DROP_PRIORITY` partition the
``EvidenceSource`` literal — disjoint, and together total."""

DROP_PRIORITY: tuple[EvidenceSource, ...] = (
    "IPDADMPROGRESS",
    "IPDNRFOCUSDT",
    "Med",
    "Vitals",
    "Diagnosis",
)
"""Whole-item drop priority for char-cap truncation: lowest-clinical-relevance
first. ``IPDADMPROGRESS`` and ``IPDNRFOCUSDT`` are long supportive narratives
that the LLM can audit without; ``MED`` is contextual; ``Vitals`` is the
bedside state; ``Diagnosis`` is the encounter context (dropped last). ``Lab``
and ``Hemodynamic`` are absent because they are exempt (:data:`EXEMPT_FROM_DROP`).

Independent of the source EMISSION order (which is fixed by the canonical-
ID assignment); without a separate drop priority, a dense MED list would
silently evict every droppable item before any MED item dropped — exactly
inverted from clinical relevance."""


# =============================================================================
# Window filtering
# =============================================================================


def _within(
    *, ts: datetime, anchor: datetime, before: timedelta, after: timedelta
) -> bool:
    """True iff ``anchor - before <= ts <= anchor + after``."""
    delta = ts - anchor
    return -before <= delta <= after


def _filter_progress(
    notes: Sequence[ProgressNote], anchor: datetime
) -> tuple[ProgressNote, ...]:
    return tuple(
        n
        for n in notes
        if _within(
            ts=n.timestamp, anchor=anchor, before=WINDOW_PROGRESS, after=WINDOW_PROGRESS
        )
    )


def _filter_focus(
    notes: Sequence[FocusNote], anchor: datetime
) -> tuple[FocusNote, ...]:
    return tuple(
        n
        for n in notes
        if _within(
            ts=n.timestamp, anchor=anchor, before=WINDOW_FOCUS, after=WINDOW_FOCUS
        )
    )


def _filter_meds(meds: Sequence[MedRecord], anchor: datetime) -> tuple[MedRecord, ...]:
    return tuple(
        m
        for m in meds
        if _within(
            ts=m.timestamp,
            anchor=anchor,
            before=WINDOW_MED_BEFORE,
            after=WINDOW_MED_AFTER,
        )
    )


def _filter_hb(hbs: Sequence[HbRecord], anchor: datetime) -> tuple[HbRecord, ...]:
    # Hb history is pre-order only — post-order labs belong to the response
    # analysis, not the decision evidence. The lower bound is STRICT
    # (``anchor - h.timestamp < WINDOW_HB_BEFORE``) to match
    # :mod:`bba.hb_lookup`'s ``anchor_utc - o.datetime_utc < _LOOKBACK`` —
    # an Hb at exactly 7 d old is invisible to the deterministic classifier,
    # and admitting it to the bundle would let the LLM cite evidence the
    # classifier never saw.
    return tuple(
        h
        for h in hbs
        if anchor - h.timestamp < WINDOW_HB_BEFORE and h.timestamp <= anchor
    )


def _filter_vitals(
    vitals: Sequence[VitalsRecord], anchor: datetime
) -> tuple[VitalsRecord, ...]:
    return tuple(
        v
        for v in vitals
        if _within(
            ts=v.timestamp, anchor=anchor, before=WINDOW_VITALS, after=WINDOW_VITALS
        )
    )


# =============================================================================
# Per-source ranking
# =============================================================================


def _has_any_vital_value(v: VitalsRecord) -> bool:
    """True iff at least one of sbp/dbp/hr/rr/bt is populated.

    A VitalsRecord with every measurement None has nothing for the LLM
    to quote; emitting it would consume bundle budget for an E_N
    citation that points at provenance metadata only — same dead-
    reference shape the round-7 progress-note filter closes."""
    return any(
        getattr(v, field) is not None for field in ("sbp", "dbp", "hr", "rr", "bt")
    )


def _has_quoteable_content(note: ProgressNote) -> bool:
    """True iff parsing the note text yields at least one non-empty SOAP section.

    Used as a pre-cap filter: blank-text or header-only IPDADMPROGRESS rows
    (e.g. ``"S:\\nO:\\nA:\\nP:"``) carry no quoteable content for the LLM.
    Without this filter, eight closer-to-anchor header-only notes would
    consume :data:`CAP_PROGRESS` slots, evict a valid farther note, then
    all be skipped at item-construction time — leaving the bundle with
    zero progress evidence. The check parses each note (an O(n) cost
    roughly equal to what _assign_ids was going to pay anyway), so the
    pre-cap filter is essentially free."""
    sections = parse_soap_sections(note.text)
    return any(sections[k] for k in SECTION_PRIORITY)


def _cap_progress_closest(
    notes: Sequence[ProgressNote], anchor: datetime
) -> tuple[ProgressNote, ...]:
    """Keep the :data:`CAP_PROGRESS` notes closest to ``anchor`` (by abs offset).

    Sort key is TOTAL — ``(abs_offset, timestamp, text)``:

    * ``abs_offset`` is the primary signal (closeness to the order anchor).
    * ``timestamp`` disambiguates -Nh vs +Nh at the same absolute offset.
    * ``text`` is the final tiebreak so two notes charted at the same minute
      with different content drop in a deterministic order. Without the text
      tiebreak, Python's stable-sort would let input order leak through and
      change the cap-set selection across re-runs of the same data."""
    ranked = sorted(
        notes,
        key=lambda n: (
            abs((n.timestamp - anchor).total_seconds()),
            n.timestamp,
            n.text,
        ),
    )
    return tuple(ranked[:CAP_PROGRESS])


# =============================================================================
# Item construction (per source)
# =============================================================================


def _diagnosis_payload(d: DiagnosisRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {"icd10": d.icd10}
    if d.description is not None:
        payload["description"] = d.description
    return payload


def _progress_payload(text: str) -> dict[str, Any]:
    """Parse SOAP sections and emit them in priority order, dropping empties.

    Section ordering matters: the LLM's reading order puts ASSESSMENT and
    PLAN first so the model sees the clinician's interpretation and treatment
    plan before the supporting Objective data and the patient's Subjective
    quote. Empty sections are omitted to keep the bundle tight under the
    char cap."""
    parsed = parse_soap_sections(text)
    sections: list[dict[str, str]] = [
        {"label": label, "text": parsed[label]}
        for label in SECTION_PRIORITY
        if parsed[label]
    ]
    return {"sections": sections}


def _focus_payload(text: str) -> dict[str, Any]:
    return {"text": text}


def _med_payload(m: MedRecord) -> dict[str, Any]:
    return {"drug": m.drug}


def _hb_payload(h: HbRecord) -> dict[str, Any]:
    # ``item_no`` lands in the bundle so the LLM (and any downstream
    # quote_grounder lookup) can resolve a citation back to the exact
    # Lab row, not just the (timestamp, value) pair which can collide
    # for corrected results.
    return {
        "value_g_dl": h.value_g_dl,
        "lab_source": h.source,
        "item_no": h.item_no,
    }


def _vitals_payload(v: VitalsRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {"note_source": v.source}
    for field in ("sbp", "dbp", "hr", "rr", "bt"):
        val = getattr(v, field)
        if val is not None:
            payload[field] = val
    return payload


def _to_utc(ts: datetime) -> datetime:
    return ts.astimezone(UTC)


def _lag_min(ts: datetime, anchor_dt: datetime) -> int:
    """Signed minutes of ``ts`` relative to the order anchor (negative = before).

    The Hemodynamic item carries ``timestamp_utc=None`` (it is a synthesized
    summary, not one charted note), so the nadir/pressor proximity is encoded
    as a relative offset instead of an absolute timestamp. Relative offsets are
    inherently PHI-free, which is why the summary can be pinned even though the
    underlying notes are time-stamped."""
    return int(round((_to_utc(ts) - anchor_dt).total_seconds() / 60.0))


def _vasopressor_entry(v: VasopressorMention, anchor_dt: datetime) -> dict[str, Any]:
    """Fact-only dict for one vasopressor mention: agent, source, lag, dose?."""
    entry: dict[str, Any] = {
        "agent": v.agent,
        "source": v.source,
        "lag_min": _lag_min(v.at, anchor_dt),
    }
    if v.dose is not None:
        entry["dose"] = v.dose
    return entry


def _hemodynamic_payload(
    summary: HemodynamicSummary, anchor_dt: datetime
) -> dict[str, Any]:
    """Render the pinned hemodynamic summary payload (issue #76).

    FACT-ONLY by contract: the MAP nadir (+ its lag and source) and the
    vasopressor mentions, nothing else. There is deliberately no
    'instability' / 'refractory' / 'appropriateness' field — hemodynamic
    status is a supporting factor the LLM weighs, never a standalone verdict,
    and the deterministic classifier has no hemodynamic gate. Caller emits
    this only when ``summary.is_empty`` is False."""
    payload: dict[str, Any] = {}
    if summary.map_nadir is not None:
        payload["map_nadir"] = summary.map_nadir
        payload["map_nadir_source"] = summary.map_nadir_source
        if summary.map_nadir_at is not None:
            payload["map_nadir_lag_min"] = _lag_min(summary.map_nadir_at, anchor_dt)
    if summary.vasopressors:
        payload["vasopressors"] = [
            _vasopressor_entry(v, anchor_dt) for v in summary.vasopressors
        ]
    return payload


def _build_vitals_notes(
    progress: Sequence[ProgressNote], focus: Sequence[FocusNote]
) -> tuple[VitalsNote, ...]:
    """Convert the SHIPPED progress + focus notes into hemodynamic-scan input.

    Built from the post-window / post-cap note set that becomes the bundle's
    narrative items, so the summary is a strict subset of evidence the LLM also
    sees in full (issue #76 corroboration: the summary never asserts a fact
    absent from a shipped note)."""
    notes: list[VitalsNote] = [
        VitalsNote(source="IPDADMPROGRESS", timestamp=_to_utc(p.timestamp), text=p.text)
        for p in progress
    ]
    notes.extend(
        VitalsNote(source="IPDNRFOCUSDT", timestamp=_to_utc(f.timestamp), text=f.text)
        for f in focus
    )
    return tuple(notes)


def _hb_sort_key(h: HbRecord) -> tuple[int, datetime, int, float]:
    """Total sort key for Hb records that mirrors :mod:`bba.hb_lookup`
    classifier semantics so the bundle never disagrees with the
    deterministic Hb selection.

    Composite key, primary to last:

    * ``source rank`` (1 for HEMATOLOGY, 0 for POCT) — PRD §3 source
      preference; HEMATOLOGY emits before POCT regardless of recency.
    * ``timestamp`` — newest-first within source.
    * ``item_no`` — Lab row identifier; higher = later insert /
      correction. Mirrors :func:`bba.hb_lookup._select_current` which
      breaks same-timestamp ties by max ``item_no``. Without this,
      a same-time corrected row could be discarded under cap pressure
      while the stale earlier row survived, putting bundle Hb out of
      sync with classifier Hb.
    * ``value_g_dl`` — final tiebreak so the order is genuinely TOTAL
      across input shuffles."""
    return (
        1 if h.source == "HEMATOLOGY" else 0,
        h.timestamp,
        h.item_no,
        h.value_g_dl,
    )


def _vitals_emission_key(
    v: VitalsRecord, anchor_dt: datetime
) -> tuple[float, datetime, str, str, str, str, str, str]:
    """Closest-to-anchor first emission key for Vitals.

    Tail-drop in :func:`_enforce_char_cap` removes the LAST emitted item;
    closest-first means the FARTHEST vitals snapshot drops first under
    cap pressure, so the bedside state nearest to the order moment
    (decision-time vitals) survives longest.

    Composite tiebreak ``(timestamp, source, str(sbp..bt))`` keeps the
    order TOTAL across input shuffles (Round-2 codex determinism gate)."""
    return (
        abs((v.timestamp - anchor_dt).total_seconds()),
        v.timestamp,
        v.source,
        str(v.sbp),
        str(v.dbp),
        str(v.hr),
        str(v.rr),
        str(v.bt),
    )


def _progress_emission_key(
    p: ProgressNote, anchor_dt: datetime
) -> tuple[float, datetime, str]:
    """Closest-to-anchor first emission key for IPDADMPROGRESS notes.

    Mirrors :func:`_vitals_emission_key`: closest-first emission so
    tail-drop discards the farthest progress note first under cap
    pressure. The cap-selection in :func:`_cap_progress_closest`
    already kept the 8 closest notes; this preserves "closest" through
    the truncation pass too. The text tiebreak is NFC-normalized so
    same-time notes whose only difference is Unicode normalization
    form sort identically — without that, the bundle hash would leak
    the input encoding even though canonical_serialize unifies it."""
    return (
        abs((p.timestamp - anchor_dt).total_seconds()),
        p.timestamp,
        _nfc(p.text),
    )


# =============================================================================
# Section-level truncation (IPDADMPROGRESS only)
# =============================================================================


def _drop_section_from_item(item: EvidenceItem, section: str) -> EvidenceItem:
    """Return a copy of ``item`` with one section removed from its payload.

    Only meaningful for IPDADMPROGRESS items; returns ``item`` unchanged
    otherwise so callers can blindly map this across the whole item list.
    Preserves ``id`` and ``timestamp_utc`` so the stable-IDs invariant
    survives the truncation pass."""
    if item.source != "IPDADMPROGRESS":
        return item
    sections_raw = item.payload.get("sections", [])
    if not isinstance(sections_raw, Sequence):
        return item
    new_sections = [s for s in sections_raw if _section_label(s) != section]
    if len(new_sections) == len(sections_raw):
        return item
    return EvidenceItem(
        id=item.id,
        source=item.source,
        timestamp_utc=item.timestamp_utc,
        payload={"sections": new_sections},
    )


def _section_label(section: Any) -> str | None:
    if isinstance(section, Mapping):
        label = section.get("label")
        return label if isinstance(label, str) else None
    return None


def _drop_one_by_priority(
    items: Sequence[EvidenceItem],
) -> tuple[EvidenceItem, ...]:
    """Drop ONE item, choosing the lowest-priority source first.

    Walks :data:`DROP_PRIORITY` in order; for the first source group that
    has items, removes the LAST occurrence of that source. Sources in
    :data:`EXEMPT_FROM_DROP` (``Lab``, ``Hemodynamic``) are absent from
    ``DROP_PRIORITY`` and therefore never dropped here — the load-bearing
    anemia and hemodynamic signals survive even under extreme cap pressure
    (issue #76).

    Returns the input unchanged if no source in :data:`DROP_PRIORITY`
    has any items. The caller treats that as a terminal state and either
    accepts the over-cap bundle or raises :class:`EvidenceBundleTooLargeError`."""
    items_list = list(items)
    for source_to_drop in DROP_PRIORITY:
        idx = _find_last_of_source(items_list, source_to_drop)
        if idx >= 0:
            return tuple(items_list[:idx] + items_list[idx + 1 :])
    return tuple(items_list)


def _find_last_of_source(items: Sequence[EvidenceItem], source: EvidenceSource) -> int:
    """Return last index where ``items[i].source == source``, or -1."""
    last = -1
    for i, it in enumerate(items):
        if it.source == source:
            last = i
    return last


def _drop_empty_progress_items(
    items: Sequence[EvidenceItem],
) -> tuple[EvidenceItem, ...]:
    """Filter out IPDADMPROGRESS items whose section list became empty.

    A zero-section item gives the LLM an evidence ID with no quoteable
    content (the prompt_builder + quote_grounder would both treat it as a
    dead reference). Pruning preserves the canonical-source-order
    invariant for the surviving items."""
    return tuple(it for it in items if not _is_empty_progress(it))


def _is_empty_progress(item: EvidenceItem) -> bool:
    if item.source != "IPDADMPROGRESS":
        return False
    sections = item.payload.get("sections")
    return not sections


# =============================================================================
# Top-level pipeline
# =============================================================================


def _bundle_envelope(
    anchor: OrderAnchor, items: Sequence[EvidenceItem]
) -> dict[str, Any]:
    """Plain-dict envelope passed to ``canonical_serialize`` + hashed.

    Pulled into a helper so the same JSON shape is produced when measuring
    bundle size during truncation AND when building the final
    :class:`EvidenceBundle` — drift between the two would mean the cap
    enforcement passes a false negative."""
    return {
        "anchor": {
            "order_datetime": _to_utc(anchor.order_datetime),
            "hn_hash": anchor.hn_hash,
            "an_hash": anchor.an_hash,
            "products": list(anchor.products),
        },
        "items": [
            {
                "id": it.id,
                "source": it.source,
                "timestamp_utc": it.timestamp_utc,
                "payload": dict(it.payload),
            }
            for it in items
        ],
    }


def _bundle_size(anchor: OrderAnchor, items: Sequence[EvidenceItem]) -> int:
    return len(canonical_serialize(_bundle_envelope(anchor, items)))


def _enforce_char_cap(
    *,
    anchor: OrderAnchor,
    items: Sequence[EvidenceItem],
    char_cap: int,
) -> tuple[EvidenceItem, ...]:
    """Drop sections (and, last resort, whole items) until the bundle fits.

    Pass 1 — section truncation, in reverse :data:`SECTION_PRIORITY` order
    (SUBJECTIVE → OBJECTIVE → PLAN → ASSESSMENT). After dropping each
    section across every IPDADMPROGRESS item, re-measure; stop the moment
    the bundle fits.

    Pass 2 — whole-item truncation, in reverse order (last item first), until
    the bundle fits or the list is empty. Diagnoses are emitted first by
    :func:`_assign_ids`, so they survive longest under this drop order."""
    current = tuple(items)
    if _bundle_size(anchor, current) <= char_cap:
        return current

    for section in reversed(SECTION_PRIORITY):
        current = tuple(_drop_section_from_item(it, section) for it in current)
        # Prune progress items whose section list is now empty — without
        # this, the bundle would emit an E_N citation pointing at a
        # quoteable-empty payload (the LLM has nothing to reference) and
        # the source-priority order would shield the empty item from the
        # last-resort whole-item drop pass below.
        current = _drop_empty_progress_items(current)
        if _bundle_size(anchor, current) <= char_cap:
            return current

    while current and _bundle_size(anchor, current) > char_cap:
        next_current = _drop_one_by_priority(current)
        if next_current == current:
            # Defensive: nothing more to drop (every source covered by
            # DROP_PRIORITY is exhausted). Fall through to the
            # below structural check, which will raise if the anchor
            # envelope alone is over cap.
            break
        current = next_current

    # Final safety check: if even the anchor envelope alone exceeds char_cap,
    # the cap is structurally unsatisfiable — fail loud rather than ship an
    # over-budget bundle that violates the AC. The caller can catch this
    # and route to a longer-context tier or split the anchor.
    final_size = _bundle_size(anchor, current)
    if final_size > char_cap:
        raise EvidenceBundleTooLargeError(
            f"anchor envelope alone is {final_size} chars; cannot satisfy "
            f"char_cap={char_cap}. Reduce anchor field sizes (hn_hash, "
            "an_hash, products) or raise char_cap."
        )
    return current


def _assign_ids(
    *,
    hemo_summary: HemodynamicSummary,
    diagnoses: Sequence[DiagnosisRecord],
    progress: Sequence[ProgressNote],
    focus: Sequence[FocusNote],
    meds: Sequence[MedRecord],
    hbs: Sequence[HbRecord],
    vitals: Sequence[VitalsRecord],
    anchor_dt: datetime,
) -> tuple[EvidenceItem, ...]:
    """Construct the items list with sequential E1...EN IDs in canonical order.

    Canonical order = source order (Hemodynamic, Diagnosis, IPDADMPROGRESS,
    IPDNRFOCUSDT, MED, Lab, Vitals) → within each source, a deterministic sort
    key. The per-source sort keys are picked so input shuffles map to the same
    kept set AND the same per-source emission order, satisfying the stable-IDs
    AC (E1..EN are byte-stable across re-runs of the same input)."""
    items: list[EvidenceItem] = []
    counter = 0

    def _next_id() -> str:
        nonlocal counter
        counter += 1
        return f"E{counter}"

    # Hemodynamic summary FIRST (issue #76): a single pinned, fact-only item
    # so the LLM reads the MAP nadir / vasopressor evidence before anything
    # else. Emitted only when the scan found something — an empty summary
    # adds no item (and so leaves all downstream IDs unchanged).
    if not hemo_summary.is_empty:
        items.append(
            EvidenceItem(
                id=_next_id(),
                source="Hemodynamic",
                timestamp_utc=None,
                payload=_hemodynamic_payload(hemo_summary, anchor_dt),
            )
        )

    # Sort key is TOTAL — ``description is not None`` (a bool, sortable) is
    # part of the key so a record with ``description=None`` and one with
    # ``description=""`` order deterministically. Without that bit, both
    # collapse to the same ``(icd10, "")`` key and Python's stable sort
    # would let input order leak through, breaking the hash AC despite the
    # records producing different payloads (None omits the field; "" emits
    # ``"description": ""``).
    for d in sorted(
        diagnoses,
        key=lambda x: (
            _nfc(x.icd10),
            x.description is not None,
            _nfc(x.description or ""),
        ),
    ):
        items.append(
            EvidenceItem(
                id=_next_id(),
                source="Diagnosis",
                timestamp_utc=None,
                payload=_diagnosis_payload(d),
            )
        )

    for p in sorted(progress, key=lambda n: _progress_emission_key(n, anchor_dt)):
        payload = _progress_payload(p.text)
        # Skip blank notes (text="") and header-only notes (e.g. "S:\nO:"
        # with no content) at construction. Without this, an empty E_N
        # citation ships under the normal-cap path because
        # _enforce_char_cap returns early when the bundle already fits —
        # the round-5 truncation-path fix only catches the over-cap path.
        if not payload.get("sections"):
            continue
        items.append(
            EvidenceItem(
                id=_next_id(),
                source="IPDADMPROGRESS",
                timestamp_utc=_to_utc(p.timestamp),
                payload=payload,
            )
        )

    # IPDNRFOCUSDT preserves the closest-first-per-side order from
    # split_focus_notes_5_5 — the LLM reads earlier IDs first and the
    # ticket explicitly demands proximity ranking, not chronological order.
    # The helper's output is already deterministic across input shuffles
    # (sorted internally), so the stable-IDs invariant survives.
    for f in focus:
        items.append(
            EvidenceItem(
                id=_next_id(),
                source="IPDNRFOCUSDT",
                timestamp_utc=_to_utc(f.timestamp),
                payload=_focus_payload(f.text),
            )
        )

    # MED: pre-order meds (decision context) BEFORE post-order meds
    # (treatment after the order). Within pre, newest-first so the most
    # decision-relevant survives cap pressure. Within post, closest-to-
    # anchor first so the farthest post-order entry drops first under
    # tail-drop. Without the pre/post split, a +2h administration could
    # outlive a -1h decision-context med — exactly inverted from
    # clinical relevance for an audit task.
    pre_meds = sorted(
        (m for m in meds if m.timestamp <= anchor_dt),
        key=lambda x: (x.timestamp, _nfc(x.drug)),
        reverse=True,
    )
    post_meds = sorted(
        (m for m in meds if m.timestamp > anchor_dt),
        key=lambda x: (x.timestamp, _nfc(x.drug)),
    )
    # Clinical-salience is the PRIMARY emission key (issue #76): CRITICAL
    # drugs (vasopressors, inotropes, blood products) emit before ROUTINE
    # before MAINTENANCE (crystalloids, saline flushes, irrigation), so the
    # tail-drop sheds maintenance fluids before a pressor under cap pressure
    # — the exact Case 2 inversion (REQNO 68012352) this fixes. The sort is
    # STABLE, so within a bucket the pre-before-post / newest-first-pre order
    # above is preserved untouched. Salience is an ORDERING signal only; it
    # never gates or weights the transfusion decision.
    for m in sorted([*pre_meds, *post_meds], key=lambda x: med_salience(x.drug)):
        items.append(
            EvidenceItem(
                id=_next_id(),
                source="Med",
                timestamp_utc=_to_utc(m.timestamp),
                payload=_med_payload(m),
            )
        )

    # Hb history: emit NEWEST-first (descending timestamp). Two reasons:
    #   1. Clinical: the most-recent pre-order Hb is the decision-time
    #      anemia signal; putting it at the head of the Lab section means
    #      the LLM reads it before historical context.
    #   2. Truncation safety: the whole-item tail-drop in
    #      _enforce_char_cap removes the LAST emitted item first. With
    #      ascending order, that drop would discard the most-recent Hb
    #      and retain stale labs — exactly inverted from clinical
    #      relevance. Newest-first inverts the drop priority correctly.
    # The source-rank tiebreak (1 for HEMATOLOGY, 0 for POCT) preserves
    # PRD §3's HEMATOLOGY > POCT preference at the bundle layer too:
    # under reverse=True, higher source-rank emits first; tail-drop in
    # _enforce_char_cap therefore evicts POCT before HEMATOLOGY when the
    # cap forces a same-timestamp/same-value tie to break.
    for h in sorted(hbs, key=_hb_sort_key, reverse=True):
        items.append(
            EvidenceItem(
                id=_next_id(),
                source="Lab",
                timestamp_utc=_to_utc(h.timestamp),
                payload=_hb_payload(h),
            )
        )

    # Vitals: pre-anchor (decision-time state) ALWAYS BEFORE post-anchor
    # (response state). Mirrors :mod:`bba.vitals_extractor` which prefers
    # any pre-anchor note over the closest post-anchor one — the audit
    # needs trigger state, not response. Without the split, a +5min
    # post-order vital could displace a -1h pre-order vital under cap
    # pressure, exactly inverting clinical relevance for an audit.
    pre_vitals = sorted(
        (v for v in vitals if v.timestamp <= anchor_dt),
        key=lambda x: _vitals_emission_key(x, anchor_dt),
    )
    post_vitals = sorted(
        (v for v in vitals if v.timestamp > anchor_dt),
        key=lambda x: _vitals_emission_key(x, anchor_dt),
    )
    for v in [*pre_vitals, *post_vitals]:
        items.append(
            EvidenceItem(
                id=_next_id(),
                source="Vitals",
                timestamp_utc=_to_utc(v.timestamp),
                payload=_vitals_payload(v),
            )
        )

    return tuple(items)


def _renumber(items: Iterable[EvidenceItem]) -> tuple[EvidenceItem, ...]:
    """Re-issue E1..EN sequentially after truncation may have rebuilt items.

    The section-truncation pass rebuilds IPDADMPROGRESS items with the same
    ``id`` they had before, so the IDs survive intact when nothing is dropped.
    But the whole-item truncation pass drops items from the end — the
    remaining IDs are still contiguous because we drop from the tail. This
    helper is a defensive belt-and-suspenders so even out-of-order
    drops would still produce E1..EN sequentially.
    """
    return tuple(
        EvidenceItem(
            id=f"E{i}",
            source=it.source,
            timestamp_utc=it.timestamp_utc,
            payload=dict(it.payload),
        )
        for i, it in enumerate(items, start=1)
    )


def build_evidence_bundle(
    *,
    inputs: EvidenceInputs,
    char_cap: int = DEFAULT_CHAR_CAP,
) -> EvidenceBundle:
    """Build the canonical evidence bundle for one order anchor.

    Pipeline:

    1. Window-filter each source to the windows in this module.
    2. Rank IPDNRFOCUSDT 5-before / 5-after the anchor (closest-first).
    3. Cap IPDADMPROGRESS at :data:`CAP_PROGRESS`; on tie, prefer entries
       closer to the anchor.
    4. Emit items in canonical source order (the literal order of
       :data:`bba.evidence_bundle_builder.models.EvidenceSource`); within
       each source, sort by a deterministic key so the assignment of
       E1..EN is byte-stable across input shuffles.
    5. Truncate to ``char_cap``: section-level first (SUBJECTIVE → OBJECTIVE
       → PLAN → ASSESSMENT), then whole-item drop from the end.
    6. Serialize via :func:`canonical_serialize` and hash via
       :func:`bundle_hash`.

    The returned :class:`EvidenceBundle.bundle_hash` lands on every
    :class:`bba.audit_store.AuditRow.evidence_bundle_hash` and is what
    :mod:`bba.deid_redactor` reads to detect mid-pipeline mutation.
    """
    anchor_dt = _to_utc(inputs.anchor.order_datetime)

    progress = _filter_progress(inputs.progress_notes, anchor_dt)
    # Drop blank / header-only notes BEFORE the cap. Otherwise eight closer
    # empty SOAP shells would consume CAP_PROGRESS slots, exclude a
    # legitimate ninth note, then all be skipped at item construction —
    # leaving the bundle with zero progress evidence.
    progress = tuple(p for p in progress if _has_quoteable_content(p))
    progress = _cap_progress_closest(progress, anchor_dt)

    focus_in_window = _filter_focus(inputs.focus_notes, anchor_dt)
    # Same pre-cap filter as progress: drop notes with no quoteable content
    # so blank/whitespace-only entries cannot consume CAP_FOCUS_BEFORE +
    # CAP_FOCUS_AFTER slots and silently evict valid farther entries.
    focus_in_window = tuple(f for f in focus_in_window if f.text.strip())
    focus = split_focus_notes_5_5(
        notes=focus_in_window,
        anchor=anchor_dt,
        cap_before=CAP_FOCUS_BEFORE,
        cap_after=CAP_FOCUS_AFTER,
    )

    meds = _filter_meds(inputs.meds, anchor_dt)
    hbs = _filter_hb(inputs.hb_history, anchor_dt)
    # Drop all-null vitals (only note_source populated): no measurement to
    # cite, so the LLM would get a dead E_N reference into the bundle.
    vitals_in_window = _filter_vitals(inputs.vitals, anchor_dt)
    vitals = tuple(v for v in vitals_in_window if _has_any_vital_value(v))

    # Scan the SHIPPED narrative (post-window/post-cap progress + focus) for
    # the MAP nadir + vasopressor mentions (issue #76). Computing it from the
    # same note set that becomes the bundle's items keeps the summary a strict
    # subset of evidence the LLM also sees in full.
    hemo_summary = scan_hemodynamics(_build_vitals_notes(progress, focus))

    items = _assign_ids(
        hemo_summary=hemo_summary,
        diagnoses=inputs.diagnoses,
        progress=progress,
        focus=focus,
        meds=meds,
        hbs=hbs,
        vitals=vitals,
        anchor_dt=anchor_dt,
    )

    items = _enforce_char_cap(anchor=inputs.anchor, items=items, char_cap=char_cap)
    items = _renumber(items)

    canonical_json = canonical_serialize(_bundle_envelope(inputs.anchor, items))
    return EvidenceBundle(
        items=items,
        canonical_json=canonical_json,
        bundle_hash=bundle_hash(canonical_json),
    )


__all__ = (
    "CAP_FOCUS_AFTER",
    "CAP_FOCUS_BEFORE",
    "CAP_PROGRESS",
    "DEFAULT_CHAR_CAP",
    "WINDOW_FOCUS",
    "WINDOW_HB_BEFORE",
    "WINDOW_MED_AFTER",
    "WINDOW_MED_BEFORE",
    "WINDOW_PROGRESS",
    "WINDOW_VITALS",
    "build_evidence_bundle",
)
