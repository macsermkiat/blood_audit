"""Superseded: per-REQNO go-live preflight for BDVSTDT.USETYPE declared intent.

It compared the per-REQNO deterministic leg with declared use absent and
present before the declared-use go-live (#151). The audit unit is now the issue
event (blood_audit #276), which decides declared use per event
(``declared_use_applies``), so this one-off evidence script is not ported. The
last working version is in git history before the issue-event switch.
"""

from __future__ import annotations

import sys

SUPERSEDED = (
    "preflight_declared_usetype.py is superseded by issue events (blood_audit "
    "#276): run_pipeline.py now writes declared_use_applies per event. The "
    "per-REQNO version is in git history."
)


def main() -> None:
    sys.exit(SUPERSEDED)


if __name__ == "__main__":
    main()
