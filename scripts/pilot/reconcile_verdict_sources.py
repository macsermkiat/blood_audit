"""Superseded: per-REQNO cross-check of pipeline verdicts against the 300-case
human review.

It compared the audit store's verdict per REQNO with the review workbook's
label per REQNO. The audit unit is now the issue event (blood_audit #276): one
REQNO can hold several events with different verdicts, and the plan keeps the
REQNO-level human labels out of scope (they would join through
``parent_audit_id`` if ever reused). The last working version is in git
history before the issue-event switch.
"""

from __future__ import annotations

import sys

SUPERSEDED = (
    "reconcile_verdict_sources.py is superseded by issue events (blood_audit "
    "#276): the review labels are per REQNO, the audit store holds per-event "
    "verdicts. The per-REQNO version is in git history."
)


def main() -> None:
    sys.exit(SUPERSEDED)


if __name__ == "__main__":
    main()
