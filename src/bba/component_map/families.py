"""Blood-component family classification for the audit inclusion gate.

Phase 2 widens the auditor beyond red cells to platelets. The RBC path keys
off :data:`bba.audit_orders.rules.RBC_PRODUCTS` — a small allow-list of BDTYPE
product codes. This module supplies the analogous machinery for every
component family so an intake seam can route a BDTYPE code to the right
auditor (or exclude it) without blending, e.g., a platelet order into RBC
statistics.

Two complementary primitives:

* :func:`classify_component` — a NAME-keyword classifier over the BDTYPE
  dictionary's ``NAME`` column. Robust to new product codes: a bag of
  irradiated / filtered / pooled variants all resolve to the same family
  from their descriptive name. This is the source of truth.

* :data:`PLATELET_PRODUCTS` / :data:`BDTYPE_FAMILY` — a frozen, verified
  code→family map for the fast intake gate (mirrors ``RBC_PRODUCTS``). Every
  entry was cross-checked against the KCMH BDTYPE dictionary's
  ``GRPCAUSELABCBC`` grouping AND the ``classify_component`` output; the
  ``test_component_map`` suite asserts the two never disagree on a known
  code, so a dictionary drift that breaks the agreement fails CI rather than
  silently mis-routing a product.

Verification provenance (2026-07-08, see docs/transfusion-policy-application-plan.md
§5.3 gating prerequisite AR-M8): the dictionary's ``GRPCAUSELABCBC`` column
tags only 13 rows (Platelet / Red cell); the NAME classifier agrees with all
13 and extends coverage to every one of the 22 product codes actually issued
in the raw ``BDVSTDT`` feed (~40k units: red_cell 23867, platelet 8193,
ffp 5350, cryo 2335, whole_blood 4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from types import MappingProxyType


class ComponentFamily(StrEnum):
    """The blood-component family a BDTYPE product belongs to.

    Only :attr:`RED_CELL` and :attr:`PLATELET` have auditors (RED_CELL is
    Phase 1; PLATELET is Phase 2). :attr:`FFP`, :attr:`CRYO`, and
    :attr:`WHOLE_BLOOD` are recognised so the intake gate can EXCLUDE them
    with a precise reason rather than a catch-all; they are out of audit
    scope (docs plan §6). :attr:`UNKNOWN` is the honest fallback for a name
    the classifier does not recognise — never guessed into a real family.
    """

    RED_CELL = "red_cell"
    PLATELET = "platelet"
    FFP = "ffp"
    CRYO = "cryo"
    WHOLE_BLOOD = "whole_blood"
    UNKNOWN = "unknown"


def classify_component(product_name: str) -> ComponentFamily:
    """Classify a BDTYPE product to its :class:`ComponentFamily` by name.

    Case-insensitive keyword match over the dictionary ``NAME`` text. The
    check order encodes clinical precedence for names that mention more than
    one keyword:

    * ``platelet`` wins outright — no other family names it.
    * ``cryoprecipitate`` → :attr:`CRYO`, but ``cryo removed`` /
      ``cryo-poor`` / ``cryosupernatant`` plasma is the cryo-DEPLETED
      leftover and is a plasma (``FFP``) product, not cryoprecipitate — the
      "removed" qualifier is checked before the bare ``cryo`` keyword so
      "Cryo Removed Plasma" is not mis-filed as cryoprecipitate.
    * plasma / ``ffp`` / ``fresh frozen`` → :attr:`FFP`.
    * ``whole blood`` → :attr:`WHOLE_BLOOD` before the red-cell check, so an
      autologous whole-blood unit is not swept up as a red-cell product.
    * ``prc`` / ``red cell`` / ``packed red`` → :attr:`RED_CELL`.

    An empty / unrecognised name returns :attr:`ComponentFamily.UNKNOWN` — the
    caller decides whether that is an exclusion or a hard error.
    """
    name = product_name.casefold()

    if "platelet" in name:
        return ComponentFamily.PLATELET
    if "cryoprecipitate" in name:
        return ComponentFamily.CRYO
    if (
        "cryo removed" in name
        or "cryo-poor" in name
        or "cryo poor" in name
        or "cryosupernatant" in name
    ):
        return ComponentFamily.FFP
    if "cryo" in name:
        return ComponentFamily.CRYO
    if "plasma" in name or "ffp" in name or "fresh frozen" in name:
        return ComponentFamily.FFP
    if "whole blood" in name:
        return ComponentFamily.WHOLE_BLOOD
    if "prc" in name or "red cell" in name or "packed red" in name:
        return ComponentFamily.RED_CELL
    return ComponentFamily.UNKNOWN


# Verified BDTYPE code → family map. Keys are every product code observed in
# the raw BDVSTDT issued-units feed (2026-07-08), each value cross-checked
# against both the GRPCAUSELABCBC dictionary grouping and classify_component()
# over the dictionary NAME. The auditor keys off codes (fast, exact), while
# classify_component() is the drift-detecting source of truth the test suite
# holds this table against.
BDTYPE_FAMILY: Mapping[str, ComponentFamily] = MappingProxyType(
    {
        # Red cells (Phase 1 scope; superset of the audited RBC_PRODUCTS allow-list)
        "LPRC": ComponentFamily.RED_CELL,
        "LDPRC": ComponentFamily.RED_CELL,
        "LDPRCI": ComponentFamily.RED_CELL,
        "LPRCI": ComponentFamily.RED_CELL,
        "LPRCF": ComponentFamily.RED_CELL,
        "SDRF": ComponentFamily.RED_CELL,
        "SDRFI": ComponentFamily.RED_CELL,
        "PRCF": ComponentFamily.RED_CELL,
        "LDPRC4": ComponentFamily.RED_CELL,
        # "SDR" is the canonical RBC allow-list code in
        # bba.audit_orders.rules.RBC_PRODUCTS. It is never issued in the raw
        # feed (the real single-donor-red-cell products are SDRF / SDRFI), so
        # it has no dictionary NAME to cross-check — but it MUST map to
        # RED_CELL so an SDR order that passes the Phase 1 RBC gate is not
        # mis-routed to UNKNOWN by this component map (Codex review, PR #84).
        "SDR": ComponentFamily.RED_CELL,
        # Platelets (Phase 2 scope)
        "LDPPC": ComponentFamily.PLATELET,
        "LDPPCI": ComponentFamily.PLATELET,
        "SDPFI": ComponentFamily.PLATELET,
        "SDPF": ComponentFamily.PLATELET,
        "SDPPI": ComponentFamily.PLATELET,
        "LPPC": ComponentFamily.PLATELET,
        "PC": ComponentFamily.PLATELET,
        "LDPC": ComponentFamily.PLATELET,
        # Plasma / cryo / whole blood (out of audit scope, recognised for exclusion)
        "FFP": ComponentFamily.FFP,
        "SDFFP": ComponentFamily.FFP,
        "CRP": ComponentFamily.FFP,
        "CPP": ComponentFamily.CRYO,
        "HTFDC": ComponentFamily.CRYO,
        "ATX": ComponentFamily.WHOLE_BLOOD,
    }
)


# The platelet intake allow-list — mirrors bba.audit_orders.rules.RBC_PRODUCTS.
# A SEED pending clinician sign-off (docs plan §7): unlike RBC_PRODUCTS, which
# deliberately excludes irradiated / filtered variants, this includes every
# platelet-family code (irradiated or not) because an irradiated leukodepleted
# platelet is still a platelet transfusion whose appropriateness must be
# judged. See project memory project_platelet_component_map.
PLATELET_PRODUCTS: frozenset[str] = frozenset(
    code for code, family in BDTYPE_FAMILY.items() if family is ComponentFamily.PLATELET
)


def component_of_code(bdtype: str) -> ComponentFamily:
    """Return the verified :class:`ComponentFamily` for a BDTYPE code.

    Fast exact lookup against :data:`BDTYPE_FAMILY`. An unrecognised code
    returns :attr:`ComponentFamily.UNKNOWN` — the intake gate treats that as
    "exclude and flag", never as a silent pass. Use :func:`classify_component`
    on the dictionary NAME when a code is genuinely new (not yet in the
    verified table).
    """
    return BDTYPE_FAMILY.get(bdtype, ComponentFamily.UNKNOWN)


def is_platelet_product(bdtype: str) -> bool:
    """True iff ``bdtype`` is an allow-listed platelet product."""
    return bdtype in PLATELET_PRODUCTS


__all__: Sequence[str] = (
    "BDTYPE_FAMILY",
    "PLATELET_PRODUCTS",
    "ComponentFamily",
    "classify_component",
    "component_of_code",
    "is_platelet_product",
)
