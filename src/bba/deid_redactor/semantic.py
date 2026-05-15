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


_PERSON_TOKEN_RE: re.Pattern[str] = re.compile(
    "|".join(re.escape(t.value) for t in PERSON_CLASS_TOKENS)
)
"""Alternation regex matching any PERSON-class token literal.

Compiled at module import so the per-note scan is a single ``finditer``
pass. Iterating :data:`PERSON_CLASS_TOKENS` once at import time also
freezes the search order — important because Python's :class:`re`
alternation is left-to-right; reorder the tokens and a substring-
prefixed token (e.g. ``[PERSON]`` vs a hypothetical ``[PERSONA]``) would
silently change which match wins.
"""


def detect_semantic_degradation(
    redacted_text: str,
    *,
    window_chars: int = SEMANTIC_WINDOW_CHARS,
    threshold: int = SEMANTIC_PERSON_THRESHOLD,
) -> bool:
    """Return ``True`` iff any sliding ``window_chars`` window of
    ``redacted_text`` contains *strictly more than* ``threshold``
    PERSON-class token starts.

    Sliding-window semantics: for every PERSON-class token at character
    position ``p``, count how many other tokens have starts in
    ``[p, p + window_chars)``. If that count, plus the token at ``p``,
    exceeds ``threshold``, the flag fires.

    Boundary cases:

    * Empty / short text → ``False`` (no tokens).
    * Exactly ``threshold`` tokens in a window → ``False`` (strict ``>``).
    * ``threshold + 1`` tokens overlapping a single ``window_chars``
      span → ``True``.

    Pure function. Used by
    :func:`bba.deid_redactor.redactor.redact_bundle` once per note.
    """
    raise NotImplementedError("RED-phase scaffold; see issue #17")
