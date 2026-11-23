"""Superseded: per-REQNO pre-flight report for the returns-ledger screen (#125).

It produced the go-live evidence for ``RETURNS_LEDGER_ENABLED`` on per-REQNO
dispositions. The audit unit is now the issue event (blood_audit #276), with
per-event returns (owner decision 1.3), so this one-off evidence script is not
ported. The last working version is in git history before the issue-event
switch.
"""

from __future__ import annotations

import sys

SUPERSEDED = (
    "preflight_returns_validation.py is superseded by issue events (blood_audit "
    "#276): run_pipeline.py now screens returns per event. The per-REQNO "
    "version is in git history."
)


def main() -> None:
    sys.exit(SUPERSEDED)


if __name__ == "__main__":
    main()
