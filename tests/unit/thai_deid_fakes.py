"""Stand-in for the thai-medical-deid module: no torch, no network."""

from __future__ import annotations

import re
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
                    entities.append(
                        {
                            "start": start,
                            "end": start + len(name),
                            "entity_group": "PERSON",
                        }
                    )
            out.append(entities)
        return out


class LengthLimitedNer:
    """Finds ``always`` names in any text and ``short_only`` names only when
    the text it is given is at most ``limit`` characters, whatever the stride.
    Also reports the letters inside any placeholder, as a real model might."""

    def __init__(
        self,
        always: tuple[str, ...],
        short_only: tuple[str, ...],
        limit: int,
        *,
        short_only_ids: tuple[str, ...] = (),
    ) -> None:
        self._always = always
        self._short_only = short_only
        self._short_only_ids = short_only_ids
        self._limit = limit

    def __call__(self, texts, *, batch_size=None, stride=None):
        out = []
        for text in texts:
            short = len(text) <= self._limit
            tagged = [(n, "PERSON") for n in self._always]
            if short:
                tagged += [(n, "PERSON") for n in self._short_only]
                tagged += [(n, "HOSPITAL_IDS") for n in self._short_only_ids]
            entities = [
                {"start": m.start(), "end": m.end(), "entity_group": group}
                for name, group in tagged
                for m in re.finditer(re.escape(name), text)
            ]
            entities += [
                {"start": m.start() + 1, "end": m.end() - 1, "entity_group": "PERSON"}
                for m in re.finditer(r"\[[A-Z_]+\]", text)
            ]
            out.append(entities)
        return out


_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")


def _merge_same_label(entities):
    merged: list[dict] = []
    for entity in sorted(entities, key=lambda e: e["start"]):
        last = merged[-1] if merged else None
        if (
            last is not None
            and last["entity_group"] == entity["entity_group"]
            and entity["start"] <= last["end"] + 1
        ):
            merged[-1] = {**last, "end": max(last["end"], entity["end"])}
        else:
            merged.append(dict(entity))
    return merged


def ner_driven_module():
    """Mimic anonymize_batch masking exactly the spans the NER pipeline returns,
    unioning same-label overlaps as deid's merge_entities does."""

    def anonymize_batch(ner, texts):
        out = []
        for text, entities in zip(texts, ner(texts, batch_size=1)):
            final = text
            protected = [m.span() for m in _PLACEHOLDER.finditer(text)]
            for entity in reversed(_merge_same_label(entities)):
                if any(
                    entity["start"] < hi and lo < entity["end"] for lo, hi in protected
                ):
                    continue
                token = f"[{entity['entity_group']}]"
                final = final[: entity["start"]] + token + final[entity["end"] :]
            out.append((final, []))
        return out

    return SimpleNamespace(
        anonymize_batch=anonymize_batch, _THAI_DIGIT_TABLE=THAI_DIGITS
    )


class WholeTextIdNer:
    """Tags ``code`` as an ID only in a text longer than ``limit`` and ``name``
    as a person only in a text of at most ``limit`` characters, so the whole
    note yields the ID and a window yields the name."""

    def __init__(self, code: str, name: str, limit: int) -> None:
        self._code = code
        self._name = name
        self._limit = limit

    def __call__(self, texts, *, batch_size=None, stride=None):
        out = []
        for text in texts:
            if len(text) > self._limit:
                tagged = [(self._code, "HOSPITAL_IDS")]
            else:
                tagged = [(self._name, "PERSON"), (self._code, "HOSPITAL_IDS")]
            out.append(
                [
                    {"start": m.start(), "end": m.end(), "entity_group": group}
                    for name, group in tagged
                    for m in re.finditer(re.escape(name), text)
                ]
            )
        return out


class ContextShiftNer:
    """Tags ``code`` as an ID always and ``word`` as a person only once the
    text already holds a placeholder, as the real model did on pass-1 output."""

    def __init__(self, code: str, word: str) -> None:
        self._code = code
        self._word = word

    def __call__(self, texts, *, batch_size=None, stride=None):
        out = []
        for text in texts:
            tagged = [(self._code, "HOSPITAL_IDS")]
            if "[" in text:
                tagged.append((self._word, "PERSON"))
            out.append(
                [
                    {"start": m.start(), "end": m.end(), "entity_group": group}
                    for name, group in tagged
                    for m in re.finditer(re.escape(name), text)
                ]
            )
        return out
