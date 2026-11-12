import sys
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pilot"
if str(PILOT_DIR) not in sys.path:
    sys.path.insert(0, str(PILOT_DIR))
