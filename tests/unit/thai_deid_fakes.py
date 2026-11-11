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


class WindowedNer:
    """Finds each name only inside the first ``window`` characters, as a
    truncating tokenizer would, unless called with ``stride``."""

    def __init__(self, names: tuple[str, ...], window: int) -> None:
        self._names = names
        self._window = window

    def __call__(self, texts, *, batch_size=None, stride=None):
        out = []
        for text in texts:
            reach = len(text) if stride else self._window
            entities = []
            for name in self._names:
                start = text.find(name)
                if start != -1 and start + len(name) <= reach:
                    entities.append({"start": start, "end": start + len(name)})
            out.append(entities)
        return out


def ner_driven_module():
    """Mimic anonymize_batch masking exactly the spans the NER pipeline returns."""

    def anonymize_batch(ner, texts):
        out = []
        for text, entities in zip(texts, ner(texts, batch_size=1)):
            final = text
            for entity in sorted(entities, key=lambda e: -e["start"]):
                final = final[: entity["start"]] + "[PERSON]" + final[entity["end"] :]
            out.append((final, []))
        return out

    return SimpleNamespace(
        anonymize_batch=anonymize_batch, _THAI_DIGIT_TABLE=THAI_DIGITS
    )
