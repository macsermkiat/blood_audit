"""Resolve the BDVSTTRANS returns ledger for the pilot production driver.

The pilot scripts (``run_pipeline.py`` deterministic leg + ``run_llm_leg.py``
model leg) are the de-facto production classification driver: they build the
``PipelineRowContext``s, run the classifier, and write the classification
``AuditRow``s that ``bba report`` reads. (The ``bba audit`` CLI's analysis-leg
orchestrator is a not-yet-built seam; until it lands, these scripts are how a
run is produced.) This module is the single seam that points that driver at the
returns ledger.

Source resolution (spec #119 complete-ledger go-live), in precedence order:

1. ``$BBA_BDVSTTRANS_CSV`` — an explicit override path (e.g. a hand-staged
   slice, or ``data/raw`` for real-identifier joins). Rarely needed.
2. a REQNO-scoped ``BDVSTTRANS.csv`` copied into the pilot bundle by
   ``sample_bundle.py`` — used when a bundle stages its own slice.
3. the canonical complete export
   (``Bloodbank/data/encrypted/BDVSTTRANS.csv``, 134k rows, pseudonymised HN/AN
   matching the bundle's namespace) — the ZERO-CONFIG DEFAULT, so a standard
   production run needs no env var. Same directory ``sample_bundle.py`` reads.
4. none present -> ``[]`` (fail open: the driver falls through to the legacy
   pipeline for every order, no crash).

The full file is a ~134k-row per-REQNO index in memory (small); the driver only
processes the orders present in its BDVST input, so run cost scales with order
count, not ledger size.

Rows are returned with UPPERCASE keys so ``summarize_returns`` /
``rows_for_admission`` read ``UNITSTAT``/``DNRNO``/``SEQNO``/``BDTYPE``/``AN``.

--- Sanctioned production run (deterministic leg; no paid batch) ---
1. Build a bundle for the target cohort from ``data/encrypted``:
   ``BBA_PILOT_SAMPLE_N=<n> uv run python scripts/pilot/sample_bundle.py``
   (or stage the full BDVST/BDVSTDT for the cohort).
2. Run the deterministic leg (reads the canonical ledger by default), fresh id:
   ``BBA_PILOT_RUN_ID=<fresh> uv run python scripts/pilot/run_pipeline.py``
   -> RETURNED_NOT_TRANSFUSED / PERIOP_TRANSFUSION_EXEMPT AuditRows land in the
   audit store (excluded from scoring). Set ``$BBA_BDVSTTRANS_CSV`` only to
   point at a non-default ledger.
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

# Canonical complete export, shipped alongside the other HOSxP tables. This is
# the zero-config default so a production run needs no env var. Mirrors
# sample_bundle.py's ``../Bloodbank/data/encrypted`` source directory.
_CANONICAL_DEFAULT = (
    Path(__file__).resolve().parents[2].parent
    / "Bloodbank"
    / "data"
    / "encrypted"
    / "BDVSTTRANS.csv"
)


def resolve_bdvsttrans_path(bundle_dir: Path) -> Path | None:
    """Return the ledger path by precedence: ``$BBA_BDVSTTRANS_CSV`` override,
    else a bundle-staged copy, else the canonical ``data/encrypted`` export,
    else ``None`` (no ledger available)."""
    override = os.environ.get(_ENV_OVERRIDE, "").strip()
    if override:
        return Path(override)
    bundle_copy = bundle_dir / "BDVSTTRANS.csv"
    if bundle_copy.exists():
        return bundle_copy
    if _CANONICAL_DEFAULT.exists():
        return _CANONICAL_DEFAULT
    return None


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
