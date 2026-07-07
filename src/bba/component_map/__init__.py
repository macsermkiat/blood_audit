"""Blood-component family classification (Phase 2 intake gating prerequisite).

Public surface: :class:`ComponentFamily`, the NAME classifier
:func:`classify_component`, the verified code→family map
:data:`BDTYPE_FAMILY` / :func:`component_of_code`, and the platelet intake
allow-list :data:`PLATELET_PRODUCTS` / :func:`is_platelet_product`.
"""

from __future__ import annotations

from collections.abc import Sequence

from bba.component_map.families import (
    BDTYPE_FAMILY,
    PLATELET_PRODUCTS,
    ComponentFamily,
    classify_component,
    component_of_code,
    is_platelet_product,
)

__all__: Sequence[str] = (
    "BDTYPE_FAMILY",
    "PLATELET_PRODUCTS",
    "ComponentFamily",
    "classify_component",
    "component_of_code",
    "is_platelet_product",
)
