"""Verdict sources — the swappable input to the scorecard aggregation.

A :data:`VerdictSource` yields ``REQNO`` → 4-value
:data:`~bba.report_generator.models.Classification`. Two implementations
are planned; only the first ships in this build:

* **Now** — :func:`human_label_verdict_source`: the 300-case human
  review workbook (Sheet1 col J). Correct but thin per doctor.
* **Next build** — a pipeline-verdict source over the full ~40k-order
  cohort, gated on the peri-op classifier fix landing (today's
  deterministic leg over-clears peri-op orders, which would credit
  false "appropriate" to surgical doctors). Only this adapter swaps;
  resolvers, aggregation, ranking, and outputs are unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType

import openpyxl

from bba.report_generator.models import Classification


VerdictSource = Callable[[], Mapping[str, Classification]]
"""Zero-argument callable yielding ``reqno -> Classification``."""


HUMAN_LABEL_TO_CLASSIFICATION: Mapping[str, Classification] = MappingProxyType(
    {
        "สมเหตุสมผล": "APPROPRIATE",
        "ไม่สมเหตุสมผล": "INAPPROPRIATE",
        "ไม่สามารถสรุปได้": "NEEDS_REVIEW",
    }
)
"""The three Thai review labels and their classification mapping.

``ไม่สามารถสรุปได้`` ("cannot conclude") lands on ``NEEDS_REVIEW`` so it
collapses into the Unresolved bucket — the reviewers explicitly declined
to call these appropriate *or* inappropriate (81/88 were surgery
reservations), so neither confident bucket may claim them.
"""


def _normalize_reqno(value: object) -> str:
    """Canonicalize a worksheet REQNO cell to the BDVST string form.

    Excel types the CaseNumber column as float (``68049423.0``); BDVST
    keys are plain digit strings (``"68049423"``). A non-integral float
    is a corrupted key and fails loud.
    """
    if isinstance(value, bool):  # bool is an int subtype; never a REQNO
        raise ValueError(f"REQNO cell has non-numeric value {value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"REQNO cell {value!r} is not an integral number; refusing "
                "to guess a truncation"
            )
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    return str(value).strip()


def human_label_verdict_source(
    xlsx_path: Path,
    *,
    sheet_name: str = "Sheet1",
    first_data_row: int = 3,
    reqno_column: int = 1,
    verdict_column: int = 10,
) -> VerdictSource:
    """Return a :data:`VerdictSource` over the human-review workbook.

    Workbook shape (as reviewed 2026-07): two header rows; data from row
    3; ``CaseNumber`` (the REQNO) in column A; the human verdict
    ``ความสมเหตุสมผล`` in column J. The keyword parameters exist so a
    future revision of the workbook does not require code changes.

    Fail-loud contract: an unknown label, a labelled row without a
    verdict, and a duplicated REQNO all raise :class:`ValueError` naming
    the offending REQNO — a silent skip would redistribute the 162/32/106
    totals without any warning. Rows with an empty REQNO cell are
    skipped (the workbook's trailing tally block has no CaseNumber).
    """

    def read() -> Mapping[str, Classification]:
        workbook = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
        try:
            worksheet = workbook[sheet_name]
            verdicts: dict[str, Classification] = {}
            for row in worksheet.iter_rows(min_row=first_data_row, values_only=True):
                raw_reqno = row[reqno_column - 1] if len(row) >= reqno_column else None
                if raw_reqno is None or str(raw_reqno).strip() == "":
                    continue
                reqno = _normalize_reqno(raw_reqno)
                raw_verdict = (
                    row[verdict_column - 1] if len(row) >= verdict_column else None
                )
                if raw_verdict is None or str(raw_verdict).strip() == "":
                    raise ValueError(
                        f"{xlsx_path} row for REQNO {reqno} has no verdict in "
                        f"column {verdict_column}; every reviewed case must "
                        "carry one of the three labels"
                    )
                label = str(raw_verdict).strip()
                classification = HUMAN_LABEL_TO_CLASSIFICATION.get(label)
                if classification is None:
                    raise ValueError(
                        f"{xlsx_path} REQNO {reqno} carries unknown verdict "
                        f"label {label!r}; expected one of "
                        f"{sorted(HUMAN_LABEL_TO_CLASSIFICATION)}"
                    )
                if reqno in verdicts:
                    raise ValueError(
                        f"{xlsx_path} contains REQNO {reqno} more than once; "
                        "duplicate labels would double-count the order"
                    )
                verdicts[reqno] = classification
            return verdicts
        finally:
            workbook.close()

    return read
