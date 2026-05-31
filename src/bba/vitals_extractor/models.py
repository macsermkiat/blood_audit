"""Pydantic v2 + dataclass models for the vitals_extractor module.

All public types are immutable. :class:`VitalSigns` is a frozen dataclass (a
pure value container); :class:`VitalsNote` and :class:`VitalsResult` are
Pydantic models with ``frozen=True`` so the pipeline output cannot be mutated
after the fact (mirrors the ingest-module contract for :class:`IngestResult`).

The :data:`LLMFallback` alias is the boundary used by the regex-first / LLM
fallback rule (issue #6): the pipeline invokes the callable only when regex
fails to populate SBP or HR, never as a default extraction path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict


@dataclass(frozen=True, slots=True)
class VitalSigns:
    """A single set of extracted vital signs.

    Any field may be ``None`` when the source text did not contain that
    measurement or when the value failed sanity bounds (in which case the
    pipeline records the :class:`VitalsFlag.DATA_ERROR` flag on the result).

    Units: ``sbp``/``dbp`` mmHg, ``hr`` bpm, ``rr`` breaths/min, ``bt`` deg C.
    """

    sbp: int | None = None
    dbp: int | None = None
    hr: int | None = None
    rr: int | None = None
    bt: float | None = None


class VitalsFlag(StrEnum):
    """Quality flags attached to a :class:`VitalsResult`.

    ``DATA_ERROR`` — at least one regex hit fell outside sanity bounds and
    was discarded; surviving fields remain populated.

    ``POST_ORDER`` — no pre-order note was available in the +/-6 h window,
    so the pipeline fell back to the earliest post-order note; vitals were
    measured AFTER the transfusion request, which the auditor must weigh.
    """

    DATA_ERROR = "vitals_data_error"
    POST_ORDER = "vitals_post_order"


class SourceProvenance(StrEnum):
    """Origin of the vitals in a :class:`VitalsResult`.

    The pipeline records exactly one provenance per result. ``LLM_EXTRACTED``
    means the LLM fallback contributed at least one populated field that
    regex missed (issue #6, regex-vs-llm-structured-text skill).
    """

    IPDADMPROGRESS = "IPDADMPROGRESS"
    IPDNRFOCUSDT = "IPDNRFOCUSDT"
    LLM_EXTRACTED = "LLM_extracted"
    NONE_IN_WINDOW = "none_in_window"


class VitalsNote(BaseModel):
    """One free-text vitals note: source table + tz-aware timestamp + text.

    ``timestamp`` MUST be tz-aware (the ingest layer normalizes every row
    timestamp to UTC via :class:`bba.ingest.row_timestamp.RowTimestamp`).
    Pydantic's :class:`AwareDatetime` validator rejects naive datetimes at
    construction, so a stray naive value never reaches the window filter in
    :func:`bba.vitals_extractor.pipeline.extract_vitals` and cannot silently
    bypass the UTC contract.
    """

    model_config = ConfigDict(frozen=True)

    source: Literal["IPDADMPROGRESS", "IPDNRFOCUSDT"]
    timestamp: AwareDatetime
    text: str


class VitalsResult(BaseModel):
    """Outcome of vital-sign extraction for a single order anchor.

    ``note_timestamp`` is the tz-aware UTC timestamp of the note the vitals
    were extracted from, or ``None`` when ``source == NONE_IN_WINDOW`` (no
    note was available in the +/-6 h window). ``flags`` is a frozenset so
    the public output is genuinely immutable (not just non-reassignable).
    """

    model_config = ConfigDict(frozen=True)

    vitals: VitalSigns
    source: SourceProvenance
    flags: frozenset[VitalsFlag]
    note_timestamp: datetime | None


class VasopressorMention(BaseModel):
    """One vasopressor/inotrope finding from the hemodynamic window scan (issue #76).

    Pure fact, no interpretation: the canonical ``agent`` name, the raw ``dose``
    phrase if one was charted alongside it (else ``None``), and the provenance
    (``at`` = tz-aware UTC note timestamp, ``source`` = origin table). The scan
    deduplicates by agent, so each agent appears at most once per summary.
    """

    model_config = ConfigDict(frozen=True)

    agent: str
    dose: str | None
    at: AwareDatetime
    source: Literal["IPDADMPROGRESS", "IPDNRFOCUSDT"]


class HemodynamicSummary(BaseModel):
    """Fact-only hemodynamic evidence aggregated across an order's note window.

    Produced by :func:`bba.vitals_extractor.hemodynamic.scan_hemodynamics` and
    surfaced to the LLM as a pinned, truncation-exempt evidence item (issue #76,
    Case 2 / REQNO 68012352). It records the worst charted mean arterial pressure
    (the ``map_nadir``, the LOWEST value across the window — not the most-recent)
    with its provenance, plus every distinct vasopressor mention.

    BINDING GUARDRAIL: this model carries facts only. There is deliberately no
    "refractory"/"escalating"/appropriateness field — hemodynamic instability is
    a supporting factor, never a standalone transfusion verdict. Do not add such
    a field; the deterministic classifier has no hemodynamic gate.

    Every field is nullable/empty: ``map_nadir`` is ``None`` when no measured MAP
    was found in the window, and ``vasopressors`` is empty when none were named.
    """

    model_config = ConfigDict(frozen=True)

    map_nadir: int | None = None
    map_nadir_at: datetime | None = None
    map_nadir_source: Literal["IPDADMPROGRESS", "IPDNRFOCUSDT"] | None = None
    vasopressors: tuple[VasopressorMention, ...] = ()

    @property
    def is_empty(self) -> bool:
        """True when the scan found neither a measured MAP nor any vasopressor."""
        return self.map_nadir is None and not self.vasopressors


LLMFallback = Callable[[str], VitalSigns]
"""Callable boundary for the LLM fallback step.

Invoked by :func:`bba.vitals_extractor.pipeline.extract_vitals` ONLY when
regex extraction returned ``None`` for SBP or HR (issue #6 AC). Implementations
should be deterministic for the same input so the audit pipeline remains
reproducible across re-runs.
"""
