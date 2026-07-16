"""Runtime feature flags for bba.

Most flags default to OFF (False); ``RETURNS_LEDGER_ENABLED`` (returns-ledger
go-live #138), ``DECLARED_USETYPE_ENABLED`` (declared-usetype go-live
2026-07-15), and ``MSBOS_RESERVATION_ENABLED`` (MSBOS reservation go-live
2026-07-16, #167) are default-ON. Each flag's default is documented on the flag
itself. Defining the contracts here keeps feature state in one place.

Flags
-----
PLATELET_LLM_ENABLED
    Gates the platelet-LLM leg of the audit pipeline (Stage C2 wiring).
    When False, platelet orders are handled by the deterministic gate only.
    Default: False.
RESERVE_AHEAD_ROUTER_ENABLED
    Gates reserve-ahead RBC prompt routing and replay semantics. When False,
    RBC dispatch and replay remain unchanged. Default: False.
RETURNS_LEDGER_ENABLED
    Gates the BDVSTTRANS returns-ledger disposition read path. When False, the
    pilot report reproduces the pre-feature behavior. Default: True.
DECLARED_USETYPE_ENABLED
    Gates the declared surgical-intent signal from BDVSTDT.USETYPE. Default:
    True (go-live 2026-07-15).
MSBOS_RESERVATION_ENABLED
    Gates the MSBOS pre-op reservation-appropriateness arm. Default: True
    (go-live 2026-07-16, #167).
"""

from __future__ import annotations

from collections.abc import Sequence

PLATELET_LLM_ENABLED: bool = False
"""Enable the platelet LLM audit path (default: OFF).

Stage C2 reads this flag to decide whether to route PLATELET_REVIEW orders
through the LLM client. The flag is defined here and defaulted to False so
the RBC path is byte-identical regardless of the platelet feature state.
"""

RESERVE_AHEAD_ROUTER_ENABLED: bool = False
"""Enable reserve-ahead RBC routing and replay semantics (default: OFF).

Issue #108 reads this flag at live dispatch, resume rebuild, pilot dispatch,
and replay. It stays False until the #109 gate makes the end-to-end semantics
safe to enable outside tests.
"""

RETURNS_LEDGER_ENABLED: bool = True
"""Enable the BDVSTTRANS returns-ledger disposition read path (default: ON).

Ticket #120 (spec #119) wired the REQNO-exact returns join, the returns_ledger
summary, and the new report columns behind this flag; the disposition router
(RETURNED_NOT_TRANSFUSED + PERIOP_TRANSFUSION_EXEMPT) landed in #122/#123. The
NARROW go-live decision (over-dispensed reissues excluded from the screen) was
validated by the #125 pre-flight and the deterministic-leg smoke, so the flag is
now enabled by default. Tests that need the pre-feature behavior monkeypatch it
back to False.
"""

DECLARED_USETYPE_ENABLED: bool = True
"""Enable the BDVSTDT.USETYPE declared surgical-intent signal (default: ON).

Spec #147. Default-ON since the declared-usetype go-live (representative
preflight flip matrix + flag-on LLM-leg comparison + clinician sign-off on the
hb_ge_10 bucket, 2026-07-15). ``classify()`` stays pure; the signal only fires
where a caller populates ``declared_use`` (the pilot legs), so contexts that
leave it ``None`` are unaffected. Set ``BBA_PILOT_DECLARED_USETYPE=0`` to force
it off for a pilot run.
"""

MSBOS_RESERVATION_ENABLED: bool = True
"""Enable the MSBOS pre-op reservation-appropriateness arm (default: ON).

Spec MSBOS, tickets #162-#167. When enabled, the pilot leg emits clinical
terminal rows: the RBC over-reservation verdict ``PREOP_OVER_RESERVATION``
(T1-T3, #163-#165) and the platelet reservation verdict/review (T4, #166).

Default-ON since the MSBOS reservation go-live (2026-07-16): the KCMH Transfusion
Committee signed off on the T5 committee pilot report (T5/#167, the go-live gate),
the platelet thresholds are CLINICIAN-SIGNED (T4/#166), and the RBC-path T2
crossmatch wrinkle is resolved (committee ruling: keep the strict >0 crossmatch
proxy, ignore any recommended_units the reference lists for a T/S item). The
library batch pipeline stays inert regardless (it never populates a reservation
snapshot, so the overlays no-op there). Set ``BBA_PILOT_MSBOS_RESERVATION=0`` to
force it off for a pilot run.
"""

__all__: Sequence[str] = (
    "DECLARED_USETYPE_ENABLED",
    "MSBOS_RESERVATION_ENABLED",
    "PLATELET_LLM_ENABLED",
    "RESERVE_AHEAD_ROUTER_ENABLED",
    "RETURNS_LEDGER_ENABLED",
)
