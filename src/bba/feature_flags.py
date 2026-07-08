"""Runtime feature flags for bba.

All flags default to OFF (False). Stage C2 wires the gate logic that reads
these flags; defining them here keeps the flag contract in one place and
ensures the RBC audit path is unaffected when a new flag is OFF.

Flags
-----
PLATELET_LLM_ENABLED
    Gates the platelet-LLM leg of the audit pipeline (Stage C2 wiring).
    When False, platelet orders are handled by the deterministic gate only.
    Default: False.
"""

from __future__ import annotations

from collections.abc import Sequence

PLATELET_LLM_ENABLED: bool = False
"""Enable the platelet LLM audit path (default: OFF).

Stage C2 reads this flag to decide whether to route PLATELET_REVIEW orders
through the LLM client. The flag is defined here and defaulted to False so
the RBC path is byte-identical regardless of the platelet feature state.
"""

__all__: Sequence[str] = ("PLATELET_LLM_ENABLED",)
