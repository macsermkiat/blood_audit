"""Regex-first vital-sign extractor from a single free-text note.

Surface contract: :func:`extract_vitals_from_text` returns a :class:`VitalSigns`
in which any field that the regex layer could not confidently populate is
``None``. Values outside their sanity bound (see :mod:`bba.vitals_extractor.bounds`)
are discarded at this layer too — the pipeline records the
:class:`VitalsFlag.DATA_ERROR` flag based on whether any discard happened
versus this layer's clean output.

This module deliberately does NOT call the LLM fallback. The fallback is
orchestrated in :mod:`bba.vitals_extractor.pipeline`, gated on the AC rule
"only invoked when regex returns null SBP or HR" (issue #6).
"""

from __future__ import annotations

from bba.vitals_extractor.models import VitalSigns


def extract_vitals_from_text(text: str) -> VitalSigns:
    """Extract SBP/DBP/HR/RR/BT from ``text`` via regex.

    Recognized patterns (Thai + English; bounds-enforced):

    * BP: ``BP 110/60``, ``BP:118/63``
    * HR: ``PR108``, ``HR 97``
    * RR: ``P 14``, ``RR 20-23`` (range → lower bound)
    * BT: ``BT 38.4``, ``Temp 37``

    Missing or out-of-bound values yield ``None`` for that field. The function
    is total: it never raises on malformed input.
    """
    del text  # signature pinned; body implemented in the GREEN phase
    raise NotImplementedError("extract_vitals_from_text — RED phase (issue #6)")
