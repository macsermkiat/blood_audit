"""The per-REQNO preflight and reconcile scripts exit superseded (#276).

They imported the PICK/USE anchor resolvers the issue-event switch retires, so
they now refuse to run instead of producing per-REQNO evidence that no longer
matches what the pipeline judges. The reconcile script compared per-REQNO
human labels with what are now per-event verdicts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"


@pytest.mark.parametrize(
    "script",
    [
        "preflight_declared_usetype.py",
        "preflight_returns_validation.py",
        "reconcile_verdict_sources.py",
    ],
)
def test_preflight_script_exits_superseded(script: str) -> None:
    done = subprocess.run(
        [sys.executable, str(PILOT_DIR / script)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 1
    assert "superseded by issue events" in done.stderr


def test_rank_doctors_human_source_exits_superseded() -> None:
    # The review labels are per REQNO; a per-event ranking cannot use them
    # without a join rule the plan leaves out of scope.
    done = subprocess.run(
        [sys.executable, str(PILOT_DIR / "rank_doctors.py")],
        capture_output=True,
        text=True,
        check=False,
        env={"BBA_VERDICT_SOURCE": "human", "PATH": "/usr/bin:/bin"},
    )

    assert done.returncode == 1
    assert "superseded by issue events" in done.stderr
