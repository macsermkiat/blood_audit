"""Stand-in for the thai-medical-deid module: no torch, no network."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def fake_module(replacements: Mapping[str, str], *, fail_on: str | None = None):
    """Mimic anonymize_batch: normalize Thai digits, then swap listed strings."""

    def anonymize_batch(ner, texts):
        out = []
        for text in texts:
            if fail_on is not None and fail_on in text:
                raise RuntimeError(f"model crashed on {text!r}")
            final = text.translate(THAI_DIGITS)
            for original, token in replacements.items():
                final = final.replace(original, token)
            out.append((final, []))
        return out

    return SimpleNamespace(
        anonymize_batch=anonymize_batch, _THAI_DIGIT_TABLE=THAI_DIGITS
    )
