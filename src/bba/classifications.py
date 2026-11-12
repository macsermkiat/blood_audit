"""Shared routing classifications that do not enter appropriateness scoring."""

RETURNS_TERMINALS: frozenset[str] = frozenset(
    {"RETURNED_NOT_TRANSFUSED", "PERIOP_TRANSFUSION_EXEMPT"}
)
