"""Semantic-degradation flag for the deid_redactor.

PRD §8: "Semantic-degradation flag: if redacted note has > 4 ``[PERSON]``-
class tokens within 50 chars → NEEDS_REVIEW". A high density of person
tokens means the note has lost too much referential content for the LLM
to extract clinically meaningful evidence — better to route the audit row
to a human reviewer than to risk a hallucinated indication.

Implementation contract:

* The unit of measurement is **NFC characters** in the redacted text
  (post-role-mapping). The role-bearing tokens vary in length
  (``"[NURSE]"`` is 7 chars, ``"[ATTENDING]"`` is 11), and the threshold
  is on token-count, NOT character-budget — so the function counts tokens
  whose anchor (the ``[`` start) falls inside any 50-char sliding
  window.
* The threshold is strict: ``> SEMANTIC_PERSON_THRESHOLD`` (i.e., 5 or
  more tokens in any window fires the flag; exactly 4 does not).
* PERSON-class is exactly the set in :data:`PERSON_CLASS_TOKENS`. Other
  tokens (``[DATE]``, ``[LOCATION]``, ...) do not contribute.
"""

from __future__ import annotations

import re

from bba.deid_redactor.models import (
    PERSON_CLASS_TOKENS,
    SEMANTIC_PERSON_THRESHOLD,
    SEMANTIC_WINDOW_CHARS,
)


def _build_token_regex() -> re.Pattern[str]:
    """Build the PERSON-class alternation regex.

    Sort by descending length so longer alternatives (``[ATTENDING]``) try
    before shorter ones (``[PERSON]``) that could be substrings under a
    different vocabulary. With the current vocabulary none of the role
    tokens prefix another, but the sort guards against future additions.
    """
    sorted_tokens = sorted(
        (t.value for t in PERSON_CLASS_TOKENS), key=len, reverse=True
    )
    return re.compile("|".join(re.escape(t) for t in sorted_tokens))


_PERSON_TOKEN_RE: re.Pattern[str] = _build_token_regex()


def detect_semantic_degradation(
    redacted_text: str,
    *,
    window_chars: int = SEMANTIC_WINDOW_CHARS,
    threshold: int = SEMANTIC_PERSON_THRESHOLD,
) -> bool:
    """Return ``True`` iff any sliding ``window_chars`` window of
    ``redacted_text`` contains *strictly more than* ``threshold``
    PERSON-class token starts.

    Pure function. Used by :func:`bba.deid_redactor.redactor.redact_bundle`
    once per note.
    """
    starts = [m.start() for m in _PERSON_TOKEN_RE.finditer(redacted_text)]
    if len(starts) <= threshold:
        return False

    # Sliding window via two pointers: for each "left" anchor, count
    # the number of starts in [starts[i], starts[i] + window_chars).
    j = 0
    for i, left in enumerate(starts):
        right_bound = left + window_chars
        while j < len(starts) and starts[j] < right_bound:
            j += 1
        in_window = j - i
        if in_window > threshold:
            return True
    return False
