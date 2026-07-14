"""Resolve the BDVSTTRANS returns ledger for the pilot production driver.

The pilot scripts (``run_pipeline.py`` deterministic leg + ``run_llm_leg.py``
model leg) are the de-facto production classification driver: they build the
``PipelineRowContext``s, run the classifier, and write the classification
``AuditRow``s that ``bba report`` reads. (The ``bba audit`` CLI's analysis-leg
orchestrator is a not-yet-built seam; until it lands, these scripts are how a
run is produced.) This module is the single seam that points that driver at the
returns ledger.

Source resolution (spec #119 complete-ledger go-live):

* ``$BBA_BDVSTTRANS_CSV`` — when set, read the ledger from this absolute path.
  Point it at the canonical complete export
  (``Bloodbank/data/encrypted/BDVSTTRANS.csv``, 134k rows, pseudonymised HN/AN)
  for a production-scale run, instead of staging a REQNO-scoped slice into the
  bundle. The full file is a ~134k-row per-REQNO index in memory (small); the
  driver still only processes the orders present in its BDVST input, so run
  cost scales with order count, not ledger size.
* otherwise the REQNO-scoped ``BDVSTTRANS.csv`` copied into the pilot bundle by
  ``sample_bundle.py`` (which itself reads ``data/encrypted``). Absent -> ``[]``.

Rows are returned with UPPERCASE keys so ``summarize_returns`` /
``rows_for_admission`` read ``UNITSTAT``/``DNRNO``/``SEQNO``/``BDTYPE``/``AN``.

--- Sanctioned production run (deterministic leg; no paid batch) ---
1. Build a bundle for the target cohort from ``data/encrypted``:
   ``BBA_PILOT_SAMPLE_N=<n> uv run python scripts/pilot/sample_bundle.py``
   (or stage the full BDVST/BDVSTDT for the cohort).
2. Run the deterministic leg against the canonical ledger, fresh run id:
   ``BBA_BDVSTTRANS_CSV=.../data/encrypted/BDVSTTRANS.csv \\
     BBA_PILOT_RUN_ID=<fresh> uv run python scripts/pilot/run_pipeline.py``
   -> RETURNED_NOT_TRANSFUSED / PERIOP_TRANSFUSION_EXEMPT AuditRows land in the
   audit store (excluded from scoring).
3. ``bba report --run-id <fresh>`` reads them (the #138 projector drops the two
   excluded terminals before projection; no crash).
The model (LLM) leg submits a REAL paid Anthropic batch and stays gated on
explicit go-ahead; the deterministic leg + returns terminals need no batch.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

_ENV_OVERRIDE = "BBA_BDVSTTRANS_CSV"


def resolve_bdvsttrans_path(bundle_dir: Path) -> Path | None:
    """Return the ledger path: the ``$BBA_BDVSTTRANS_CSV`` override, else the
    bundle copy if present, else ``None`` (no ledger available)."""
    override = os.environ.get(_ENV_OVERRIDE, "").strip()
    if override:
        return Path(override)
    bundle_copy = bundle_dir / "BDVSTTRANS.csv"
    return bundle_copy if bundle_copy.exists() else None


def load_bdvsttrans_rows(bundle_dir: Path) -> list[dict[str, str]]:
    """Load the returns ledger rows with UPPERCASE keys, or ``[]`` if absent.

    Callers gate the call on ``RETURNS_LEDGER_ENABLED`` so a flag-off run never
    opens the (possibly large) ledger.
    """
    path = resolve_bdvsttrans_path(bundle_dir)
    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return [{k.upper(): v for k, v in row.items()} for row in csv.DictReader(fh)]
