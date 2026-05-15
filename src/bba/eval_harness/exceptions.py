"""Errors raised by :mod:`bba.eval_harness`.

The harness is the *graded layer* in the audit pipeline: it consumes audit
rows produced upstream and produces population-level metrics with their
uncertainty. Three error shapes appear at the module boundary:

* :class:`ShapeMismatchError` — two parallel sequences (predictions/labels,
  observations/cluster ids, etc.) disagree in length. Silently truncating to
  ``min(len(a), len(b))`` would corrupt the metric the publication report
  cites, so the harness fails loud at the input boundary.

* :class:`EmptyInputError` — a metric was requested on an empty population.
  Returning a zero point estimate would be indistinguishable from a real
  zero-prevalence finding; raising surfaces the absence of evidence.

* :class:`InsufficientStratumError` — a stratum's drawn sample is smaller
  than the requested target AND smaller than the population, which means the
  sampler hit an arithmetic bound (e.g., the enrichment target exceeds the
  available positives). Distinct from "population smaller than target", which
  is a routine truncation, not an error.
"""

from __future__ import annotations


class EvalHarnessError(Exception):
    """Base class for every public exception raised by the eval harness."""


class ShapeMismatchError(EvalHarnessError, ValueError):
    """Two parallel sequences disagree in length at a metric boundary."""


class EmptyInputError(EvalHarnessError, ValueError):
    """A metric was requested over an empty population."""


class InsufficientStratumError(EvalHarnessError, ValueError):
    """A stratum cannot meet its enrichment target from the population."""
