"""Tier-1 deterministic operation-name matcher for unresolved MSBOS reservations.

Ships INERT (not exported from the package): the offline validation study in
spec #185 consumes it next, and later pipeline wiring reuses it unchanged.

The matcher re-reads the packaged MSBOS reference CSV itself rather than reusing
``MsbosReference``, because that index deliberately drops the 7 blank-code
(name-only) rows (reference.py:108-110). Name matching needs those rows, so this
module keeps every row with a non-blank operation name and indexes by name.

Rules are fail-closed: a case's event names resolve to at most ONE distinct
recommendation, and any ambiguity is surfaced as a conflict, never guessed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from bba.preop_reservation.evaluate import _decide_from_row
from bba.preop_reservation.models import (
    MsbosRow,
    MsbosToken,
    ReservationDecision,
)
from bba.preop_reservation.note_operation import _normalize
from bba.preop_reservation.reference import (
    MSBOS_REFERENCE_FILENAME,
    MsbosReferenceError,
    parse_recommended_units,
)

_MSBOS_TOKENS = frozenset({"none", "G/M", "T/S"})
_REQUIRED_COLUMNS = frozenset({"operation", "msbos", "recommended_units"})
# An acronym-like parenthesized token: all-caps alphanumeric, 3-6 chars
# (CABG, EVAR, TKA). Applied to the RAW paren content before normalization,
# because the normalizer casefolds and would erase the all-caps signal.
_ACRONYM_RE = re.compile(r"^[A-Z0-9]{3,6}$")
_PAREN_RE = re.compile(r"\(([^()]*)\)")

MatchStatus = Literal["matched", "no_match", "conflicting_recommendations"]


class OperationNameIndex(BaseModel):
    """Immutable, content-addressed index of MSBOS operation names → recommendations.

    ``_needles`` maps each normalized needle string to the SET of canonical
    operation names it points at (collisions are represented, never
    overwritten). ``_full_needles`` is the subset of needles that are full
    operation names (eligible for reverse matching; acronym-only variants are
    not). ``_recommendation_by_operation`` carries each canonical operation's
    resolved recommendation (T/S units already normalized to 0).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    content_hash: str
    _needles: Mapping[str, frozenset[str]]
    _full_needles: frozenset[str]
    _recommendation_by_operation: Mapping[str, MsbosRow]
    _operations_by_full_name: Mapping[str, frozenset[str]]

    def __init__(
        self,
        *,
        content_hash: str,
        needles: Mapping[str, frozenset[str]],
        full_needles: frozenset[str],
        recommendation_by_operation: Mapping[str, MsbosRow],
        operations_by_full_name: Mapping[str, frozenset[str]],
    ) -> None:
        super().__init__(content_hash=content_hash)
        object.__setattr__(self, "_needles", needles)
        object.__setattr__(self, "_full_needles", full_needles)
        object.__setattr__(
            self, "_recommendation_by_operation", recommendation_by_operation
        )
        object.__setattr__(
            self, "_operations_by_full_name", operations_by_full_name
        )

    def recommendation_for(self, operation: str) -> MsbosRow | None:
        """The resolved recommendation for a canonical operation name; None if absent."""
        return self._recommendation_by_operation.get(operation)

    def operations(self) -> frozenset[str]:
        """All canonical operation names in the index."""
        return frozenset(self._recommendation_by_operation)


class NameMatchResult(BaseModel):
    """Frozen outcome of matching a case's event names against the index."""

    model_config = ConfigDict(frozen=True)

    status: MatchStatus
    matched_operations: tuple[str, ...] = ()
    matched_event_name: str = ""
    recommendation: MsbosRow | None = None
    distinct_recommendation_count: int = 0


class VerificationResult(BaseModel):
    """Frozen outcome of verifying an LLM-proposed operation string (Tier-2 helper)."""

    model_config = ConfigDict(frozen=True)

    accepted: bool
    operation: str = ""
    recommendation: MsbosRow | None = None


def _acronym_variants(operation: str) -> tuple[frozenset[str], str | None]:
    """Return acronym-like paren needles and a paren-stripped variant, if any.

    Parenthesized content becomes a needle ONLY when it looks like an acronym
    (all-caps alphanumeric, 3-6 chars). A paren-stripped variant of the full
    name exists ONLY when the parens held such an acronym. Semantic modifiers
    (tumor, redo, major, Elective, Thai text, ...) never become needles.
    """
    acronyms: set[str] = set()
    for raw_inner in _PAREN_RE.findall(operation):
        inner = raw_inner.strip()
        if _ACRONYM_RE.fullmatch(inner):
            needle = _normalize(inner)
            if needle.strip():
                acronyms.add(needle)
    stripped: str | None = None
    if acronyms:
        stripped_raw = _PAREN_RE.sub(" ", operation)
        candidate = _normalize(stripped_raw)
        if candidate.strip():
            stripped = candidate
    return frozenset(acronyms), stripped


def _index_from_rows(
    rows: Iterable[Mapping[str, str]], *, content_hash: str
) -> OperationNameIndex:
    """Build a validated name index from parsed rows (test seam).

    Mirrors reference._reference_from_rows, but KEEPS blank-code rows (indexing
    by operation name, not code) and drops only rows with a blank operation
    name. Construction is row-order-independent.
    """
    needles: dict[str, set[str]] = {}
    full_needles: set[str] = set()
    recommendation_by_operation: dict[str, MsbosRow] = {}
    operations_by_full_name: dict[str, set[str]] = {}

    for row_number, row in enumerate(rows, start=2):
        missing = _REQUIRED_COLUMNS.difference(row)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise MsbosReferenceError(
                f"malformed schedule rejected at row {row_number}: "
                f"missing columns {missing_text}"
            )
        operation = row["operation"].strip()
        if not operation:
            continue
        token = row["msbos"].strip()
        if token not in _MSBOS_TOKENS:
            raise MsbosReferenceError(
                f"malformed schedule rejected at row {row_number}: "
                f"unknown msbos token {token!r}"
            )
        try:
            units = parse_recommended_units(row["recommended_units"])
        except MsbosReferenceError as exc:
            raise MsbosReferenceError(
                f"malformed schedule rejected at row {row_number}: {exc}"
            ) from exc
        if token == "T/S":
            # Committee ruling parity (reference.py:100-107): a Type & Screen
            # recommends screening only, so recommended_units is meaningless and
            # is normalized to 0 BEFORE the distinct-recommendation uniqueness
            # check — a T/S-1 row and a T/S-2 row collapse to one recommendation.
            units = 0
        recommendation = MsbosRow(
            msbos=cast(MsbosToken, token), recommended_units=units
        )
        # The same operation name can legitimately recur; identity is the
        # (msbos, units) recommendation, so re-adding the same name is a no-op.
        # A name recurring with a DIFFERENT recommendation is not overwritten:
        # the fail-closed uniqueness rule at match time surfaces it as conflict.
        recommendation_by_operation.setdefault(operation, recommendation)

        full_needle = _normalize(operation)
        if not full_needle.strip():
            continue
        needles.setdefault(full_needle, set()).add(operation)
        full_needles.add(full_needle)
        operations_by_full_name.setdefault(full_needle, set()).add(operation)

        acronyms, stripped = _acronym_variants(operation)
        for acronym in acronyms:
            needles.setdefault(acronym, set()).add(operation)
        if stripped is not None:
            needles.setdefault(stripped, set()).add(operation)
            # A paren-stripped full name IS a full name for reverse matching.
            full_needles.add(stripped)

    frozen_needles = MappingProxyType(
        {needle: frozenset(names) for needle, names in sorted(needles.items())}
    )
    frozen_recs = MappingProxyType(dict(sorted(recommendation_by_operation.items())))
    frozen_full_names = MappingProxyType(
        {
            full_name: frozenset(names)
            for full_name, names in sorted(operations_by_full_name.items())
        }
    )
    return OperationNameIndex(
        content_hash=content_hash,
        needles=frozen_needles,
        full_needles=frozenset(full_needles),
        recommendation_by_operation=frozen_recs,
        operations_by_full_name=frozen_full_names,
    )


@lru_cache(maxsize=1)
def load_operation_name_index() -> OperationNameIndex:
    """Load and index the packaged MSBOS reference by operation name (once per process).

    Uses the exact byte-reading and content-hash pattern of
    reference.load_msbos_reference, so this index's content_hash equals that
    loader's. Tests drive the ``_index_from_rows`` seam, not this cache.
    """
    reference_path = resources.files("bba.preop_reservation").joinpath(
        "data", MSBOS_REFERENCE_FILENAME
    )
    raw_bytes = reference_path.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    reader = csv.DictReader(io.StringIO(raw_bytes.decode("utf-8-sig")))
    if reader.fieldnames is None:
        raise MsbosReferenceError("malformed schedule rejected: missing header row")
    missing = _REQUIRED_COLUMNS.difference(reader.fieldnames)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise MsbosReferenceError(
            f"malformed schedule rejected: missing columns {missing_text}"
        )
    return _index_from_rows(reader, content_hash=content_hash)


def _word_bounded_in(needle: str, haystack: str) -> bool:
    """True when the (space-padded) needle appears word-bounded in the haystack."""
    return needle in haystack


def _matches_for_event(index: OperationNameIndex, event_name: str) -> frozenset[str]:
    """Canonical operation names matched by a single event name.

    Forward: a needle is word-bounded inside the normalized event name.
    Reverse: the event name is contained in a FULL operation name only (never an
    acronym-only variant), AND only when the event name has >=2 words.
    Then longest-match-wins subphrase disqualification WITHIN this event: a
    matched needle that is a proper sub-phrase of another matched needle for
    this event is disqualified (mirrors note_operation.py:34-41).
    """
    norm_event = _normalize(event_name)
    if not norm_event.strip():
        return frozenset()
    event_word_count = len(norm_event.split())

    hit_needles: set[str] = set()
    # Forward: index needle word-bounded inside the event name.
    for needle in index._needles:
        if _word_bounded_in(needle, norm_event):
            hit_needles.add(needle)
    # Reverse: event name contained in a FULL operation name, event has >=2 words.
    if event_word_count >= 2:
        for full_needle in index._full_needles:
            if _word_bounded_in(norm_event, full_needle):
                hit_needles.add(full_needle)

    # Longest-match-wins: drop a needle that is a proper sub-phrase of another
    # matched needle for THIS event (word-bounded containment).
    surviving = {
        needle
        for needle in hit_needles
        if not any(
            other != needle and _word_bounded_in(needle, other)
            for other in hit_needles
        )
    }

    operations: set[str] = set()
    for needle in surviving:
        operations.update(index._needles[needle])
    return frozenset(operations)


def match_operation_names(
    index: OperationNameIndex, event_names: Sequence[str]
) -> NameMatchResult:
    """Resolve a case's event names to at most ONE MSBOS recommendation.

    Accepts ONLY when exactly one distinct (msbos token, recommended units)
    recommendation survives across all matches (T/S units already zeroed).
    Same-recommendation multi-operation collapse is allowed — ALL matched
    canonical operation names are reported. Any different-recommendation
    collision fails closed with status ``conflicting_recommendations``.
    """
    # Map each matched operation to the event names that produced it, so an
    # accepted match can report a deterministic representative event name.
    events_by_operation: dict[str, set[str]] = {}
    for event_name in event_names:
        for operation in _matches_for_event(index, event_name):
            events_by_operation.setdefault(operation, set()).add(event_name)

    matched_operations = set(events_by_operation)
    if not matched_operations:
        return NameMatchResult(status="no_match")

    distinct_recommendations = {
        index._recommendation_by_operation[operation]
        for operation in matched_operations
    }
    ordered_operations = tuple(sorted(matched_operations))
    distinct_count = len(distinct_recommendations)

    if distinct_count != 1:
        return NameMatchResult(
            status="conflicting_recommendations",
            matched_operations=ordered_operations,
            distinct_recommendation_count=distinct_count,
        )

    # Deterministic representative event: the sorted-first event that produced
    # the sorted-first matched operation.
    representative_event = sorted(events_by_operation[ordered_operations[0]])[0]
    return NameMatchResult(
        status="matched",
        matched_operations=ordered_operations,
        matched_event_name=representative_event,
        recommendation=next(iter(distinct_recommendations)),
        distinct_recommendation_count=1,
    )


def would_be_verdict(
    *,
    row: MsbosRow,
    reserved_units: int,
    reference_hash: str,
) -> ReservationDecision:
    """The would-be reservation verdict for a name-matched row.

    Delegates to evaluate._decide_from_row for zero rule duplication. A name
    match has no ICD-9 code, so ``resolved_icd9`` is empty and
    ``note_resolved`` is False. The result is labeled would-be and is NEVER a
    verdict (annotation-first invariant, spec #185).
    """
    return _decide_from_row(
        row=row,
        reserved_units=reserved_units,
        resolved_icd9="",
        reference_hash=reference_hash,
        note_resolved=False,
    )


def verify_proposed_operation(
    index: OperationNameIndex, proposed_operation: str
) -> VerificationResult:
    """Verify an LLM-proposed operation string against the index (Tier-2 helper).

    Accepts ONLY when the proposal, normalized, EXACTLY equals a reference
    operation's FULL name (the LLM is instructed to copy an exact operation
    name, not an acronym variant) AND the same single-distinct-recommendation
    uniqueness rule holds. Near-miss, non-member, and ambiguous (an exact
    full-name collision spanning conflicting recommendations) proposals are
    rejected. Ships inert; consumed by the Tier-2 ticket.
    """
    needle = _normalize(proposed_operation)
    if not needle.strip():
        return VerificationResult(accepted=False)
    operations = index._operations_by_full_name.get(needle)
    if not operations:
        return VerificationResult(accepted=False)
    distinct_recommendations = {
        index._recommendation_by_operation[operation] for operation in operations
    }
    if len(distinct_recommendations) != 1:
        return VerificationResult(accepted=False)
    return VerificationResult(
        accepted=True,
        operation=sorted(operations)[0],
        recommendation=next(iter(distinct_recommendations)),
    )
