"""Interpret the order-time declared use from ``BDVSTDT.USETYPE``.

This leaf module is the single source of truth for USETYPE interpretation so
the ingest, classifier, bundle, and pilot layers cannot drift. Codes ``2``
(``surgery``) and ``3`` (``type_screen``) form the declared surgical group.
Per-order collapse accepts one distinct non-blank detail-line code; conflicting
codes collapse to no signal. Unknown codes, including ``5``, drive no routing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

DeclaredUseLabel = Literal["ward", "surgery", "type_screen", "day_care", "unknown"]

USETYPE_LABELS: Mapping[str, DeclaredUseLabel] = MappingProxyType(
    {
        "1": "ward",
        "2": "surgery",
        "3": "type_screen",
        "4": "day_care",
    }
)

DECLARED_SURGICAL_LABELS: frozenset[DeclaredUseLabel] = frozenset(
    {"surgery", "type_screen"}
)

logger = logging.getLogger(__name__)


def label_for(code: str) -> DeclaredUseLabel:
    """Return the declared-use label for a raw USETYPE code."""
    return USETYPE_LABELS.get(code.strip(), "unknown")


def collapse_usetype(values: Iterable[str]) -> str | None:
    """Collapse detail lines to one code, or ``None`` for blank or mixed values.

    One distinct non-blank value wins. Empty and blank-only inputs return
    ``None``; mixed values log a warning and return ``None``, never raising.
    """
    codes = frozenset(
        value.strip() for value in values if value is not None and value.strip()
    )
    if not codes:
        return None
    if len(codes) == 1:
        return next(iter(codes))
    logger.warning("Mixed USETYPE codes for one order: %s", sorted(codes))
    return None


class DeclaredUse(BaseModel):
    """A raw USETYPE code bound to its declared-use label.

    The label is not free: it MUST equal ``label_for(code)``. Constructing an
    inconsistent pair (e.g. ``DeclaredUse(code="2", label="ward")``) fails loud,
    so a surgical code can never be represented as non-surgical for the
    downstream ``.label``-keyed surgical routing. Prefer :meth:`from_code`, which
    derives the label; the validator is the backstop for direct construction and
    for deserialization.
    """

    code: str
    label: DeclaredUseLabel

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _label_matches_code(self) -> DeclaredUse:
        expected = label_for(self.code)
        if self.label != expected:
            raise ValueError(
                f"DeclaredUse label {self.label!r} does not match code "
                f"{self.code!r} (expected {expected!r})"
            )
        return self

    @classmethod
    def from_code(cls, code: str) -> DeclaredUse:
        """Build a declared use with the label derived from its raw code."""
        return cls(code=code, label=label_for(code))


__all__: Sequence[str] = (
    "DeclaredUse",
    "DeclaredUseLabel",
    "DECLARED_SURGICAL_LABELS",
    "USETYPE_LABELS",
    "collapse_usetype",
    "label_for",
)
