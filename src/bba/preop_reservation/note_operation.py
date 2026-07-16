"""Pure clinical-note disambiguation for conflicting MSBOS operation codes."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence

from bba.preop_reservation.models import CandidateOperation, MsbosRow


def _normalize(text: str) -> str:
    """NFKC + casefold; replace non-alphanumeric with space, collapse, and pad."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    spaced = "".join(ch if ch.isalnum() else " " for ch in folded)
    return f" {' '.join(spaced.split())} "


def resolve_operation_from_notes(
    *, candidates: Sequence[CandidateOperation], note_texts: Sequence[str]
) -> MsbosRow | None:
    """Whole-phrase, word-bounded match against windowed notes.

    Return the single distinct resolved recommendation, or None when zero or
    multiple distinct recommendations match. Input ordering does not matter.
    """
    norm_notes = [_normalize(text) for text in note_texts]
    matched: set[MsbosRow] = set()
    for candidate in candidates:
        needle = _normalize(candidate.operation)
        if needle.strip() == "":
            continue
        if any(needle in note for note in norm_notes):
            matched.add(
                MsbosRow(
                    msbos=candidate.msbos,
                    recommended_units=candidate.recommended_units,
                )
            )
    if len(matched) == 1:
        return next(iter(matched))
    return None
