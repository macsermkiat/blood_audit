"""The per-REQNO preflight scripts exit with a superseded message (#276).

They imported the PICK/USE anchor resolvers the issue-event switch retires, so
they now refuse to run instead of producing per-REQNO evidence that no longer
matches what the pipeline judges.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"


@pytest.mark.parametrize(
    "script", ["preflight_declared_usetype.py", "preflight_returns_validation.py"]
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
