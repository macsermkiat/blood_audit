"""Source-of-truth PHI regex set, shared by redaction and traceback scrubbing.

``PHI_REGEXES`` is the deliberately coarse, fail-loud net used by:

* :mod:`bba.cli.phi_scrubber` — sweeps every string value in an
  uncaught-exception traceback so PHI cannot leak through frame locals
  / repr() output.
* (Future) :mod:`bba.deid_redactor.redactor` — augments the
  thai-medical-deid backend's output as a defensive second pass.

Living the constant in ``bba.deid_redactor`` keeps the redactor and the
scrubber from drifting apart. Both surfaces import the same tuple so a
new PHI shape added here is caught everywhere it could leak.

The set is intentionally narrower than what the thai-medical-deid model
itself produces — the model is precision-oriented (avoid over-redacting
clinical text), while ``PHI_REGEXES`` is recall-oriented (catch obvious
PHI shapes even when stripped of context). They complement each other.
"""

from __future__ import annotations

import re
from typing import Final


_THAI_HONORIFICS: Final[str] = r"นาย|นาง|นางสาว|เด็กชาย|เด็กหญิง"
"""Five Thai honorifics common in HOSxP free-text notes."""


PHI_REGEXES: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\b\d{7,10}\b"),
    re.compile(
        r"\b(?:Mr|Mrs|Ms|Dr)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
    ),
    re.compile(_THAI_HONORIFICS),
)
"""Tuple of pre-compiled patterns covering the three most common PHI
shapes in this corpus:

1. HN / AN-shaped digit runs (7–10 digits).
2. Western honorific + 1–N capitalised name tokens.
3. Thai honorifics — matching the honorific alone is enough to redact
   the surrounding string; we deliberately do not try to grab the
   trailing given name, which would mis-fire on benign sentences."""
