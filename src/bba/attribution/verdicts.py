"""Verdict sources — the swappable input to the scorecard aggregation.

A :data:`VerdictSource` yields ``REQNO`` → 4-value
:data:`~bba.report_generator.models.Classification`. Two implementations
ship:

* :func:`human_label_verdict_source` — the 300-case human review
  workbook (Sheet1 col J). Correct but thin per doctor.
* :func:`pipeline_verdict_source` — the application's own per-order
  verdicts (``bba.audit_store`` ``AuditRow.final_classification``) over
  the full ~40k-order cohort. Far more orders per doctor. Only this
  adapter differs; resolvers, aggregation, ranking, and outputs are
  unchanged.

Which source the pilot *ranks* on is a separate decision. Promoting
:func:`pipeline_verdict_source` to the default is gated on validating it
against a real audit-store run and reconciling it against the human
labels (:func:`reconcile_verdict_sources`): the peri-op classifier fix
that would otherwise over-clear surgical orders landed in #80/#81, but
the reconciliation is what confirms the pipeline is no longer crediting
false "appropriate" to surgical doctors before those verdicts drive a
public ranking.

This module keeps to :class:`~bba.report_generator.models.Classification`
and never imports the storage layer — :func:`pipeline_verdict_source`
takes the rows via the :class:`SupportsVerdict` protocol so the caller
owns the ``bba.audit_store`` read (and its pyarrow/duckdb dependency).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple, Protocol

import openpyxl

from bba.report_generator.models import Classification


VerdictSource = Callable[[], Mapping[str, str]]
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


class SupportsVerdict(Protocol):
    """An audited order carrying its ``reqno`` and the pipeline's
    ``final_classification`` — satisfied structurally by
    :class:`bba.audit_store.models.AuditRow`.

    Typing the input as a protocol (rather than importing ``AuditRow``)
    keeps this module free of the storage layer's pyarrow/duckdb
    dependency, mirroring :class:`bba.attribution.resolvers.SupportsReqno`.
    The members are read-only ``@property`` so a frozen model — whose
    fields the pydantic mypy plugin treats as read-only — still satisfies
    the protocol.

    ``final_classification`` is typed ``str`` (not
    :data:`~bba.report_generator.models.Classification`) because the
    audit store's own ``Classification`` adds store-only values —
    ``POTENTIALLY_INAPPROPRIATE`` and
    ``PREOP_RESERVATION_UNCONFIRMED`` — and narrowing those onto the report
    generator's four values is the ``projector``'s job below.
    """

    @property
    def reqno(self) -> str: ...

    @property
    def final_classification(self) -> str: ...


VerdictProjector = Callable[[str], str]
"""Project an audit-store verdict for attribution aggregation."""


_REPORT_CLASSIFICATIONS: frozenset[str] = frozenset(
    {"APPROPRIATE", "INAPPROPRIATE", "NEEDS_REVIEW", "INSUFFICIENT_EVIDENCE"}
)
"""The four values the ranking layer can bucket. Store-only values are not
members and must be projected explicitly."""


def strict_verdict_projector(value: str) -> str:
    """Identity on scorable report values and the explicit excluded values.

    The excluded values (``RETURNED_NOT_TRANSFUSED``,
    ``PERIOP_TRANSFUSION_EXEMPT``) pass through so the ranking layer can hold
    them in their own non-scorable counters. Fail loud on other store-only
    values, including ``POTENTIALLY_INAPPROPRIATE`` and
    ``PREOP_RESERVATION_UNCONFIRMED``.

    The default so no build silently buckets a store-only value: mapping it is
    a clinical decision (mirrors
    :func:`bba.report_generator.builder.default_classification_projector`).
    Callers that want it pooled into Unresolved pass
    :func:`needs_review_verdict_projector`.
    """
    if value in _REPORT_CLASSIFICATIONS:
        return value
    if value in ("RETURNED_NOT_TRANSFUSED", "PERIOP_TRANSFUSION_EXEMPT"):
        return value
    raise ValueError(
        f"final_classification {value!r} is not one of the four report "
        f"classifications {sorted(_REPORT_CLASSIFICATIONS)}; if this is the "
        "audit-store-only 'POTENTIALLY_INAPPROPRIATE' or "
        "'PREOP_RESERVATION_UNCONFIRMED', pass "
        "needs_review_verdict_projector to pool it into Unresolved"
    )


def needs_review_verdict_projector(value: str) -> str:
    """Like :func:`strict_verdict_projector` but maps the audit-store-only
    store-only verdicts onto ``NEEDS_REVIEW``, which the ranking buckets into
    Unresolved, consistent with :data:`bba.verification.models._BUCKET_OF`.
    """
    if value in (
        "POTENTIALLY_INAPPROPRIATE",
        "PREOP_RESERVATION_UNCONFIRMED",
    ):
        return "NEEDS_REVIEW"
    return strict_verdict_projector(value)


def pipeline_verdict_source(
    rows: Iterable[SupportsVerdict],
    *,
    projector: VerdictProjector = strict_verdict_projector,
) -> VerdictSource:
    """Return a :data:`VerdictSource` over the application's own per-order
    verdicts — the full-cohort counterpart to
    :func:`human_label_verdict_source`.

    ``rows`` are audited orders (typically
    ``AuditStore.read_audit_results(run_id=...)`` scoped to one run so each
    ``reqno`` appears once). They are materialized once here, so the
    returned source is re-callable even if a generator is passed.

    ``projector`` narrows each order's audit-store
    ``final_classification`` onto the ranking layer's four values. The
    default fails loud on store-only values; pass
    :func:`needs_review_verdict_projector` to pool them into Unresolved.

    Fail-loud contract (mirrors
    :func:`bba.attribution.order_doctor_map.load_reqno_to_doctor`): the
    same ``reqno`` carrying two *different* projected verdicts raises
    :class:`ValueError` naming the reqno — that only happens when the caller
    mixed rows across runs/code-versions, which would silently mis-bucket
    the order. An identical repeat is tolerated (idempotent re-read). An
    empty ``reqno`` raises rather than collapsing distinct orders onto a
    blank key.
    """
    materialized = tuple(rows)

    # Values are str, not Classification: the projector passes the two excluded
    # values (RETURNED_NOT_TRANSFUSED, PERIOP_TRANSFUSION_EXEMPT) through, and
    # those are not members of report_generator's four-value Classification.
    # This matches the VerdictSource contract (Mapping[str, str]); the ranking
    # layer buckets the four scorable values and holds the excluded ones apart.
    def read() -> Mapping[str, str]:
        verdicts: dict[str, str] = {}
        for row in materialized:
            reqno = row.reqno.strip()
            if reqno == "":
                raise ValueError(
                    "audit row has an empty REQNO; refusing to key a verdict "
                    "on a blank order id"
                )
            classification = projector(row.final_classification)
            existing = verdicts.get(reqno)
            if existing is not None and existing != classification:
                raise ValueError(
                    f"REQNO {reqno} carries conflicting verdicts "
                    f"{existing!r} and {classification!r}; scope the read to a "
                    "single run_id so each order resolves to one classification"
                )
            verdicts[reqno] = classification
        return verdicts

    return read


class VerdictReconciliation(NamedTuple):
    """Overlap between two verdict mappings — the pre-swap cross-check.

    ``pipeline_over_clears`` is the safety-critical field: orders the human
    reviewer called INAPPROPRIATE that the pipeline called APPROPRIATE. A
    non-trivial count is the peri-op over-clear signal (the pipeline
    crediting a surgical order as fine) and is the reason
    :func:`pipeline_verdict_source` is validated before it becomes the
    ranking default.
    """

    overlap: int
    agree: int
    disagree: int
    pipeline_over_clears: int
    human_only: int
    pipeline_only: int


def reconcile_verdict_sources(
    pipeline: Mapping[str, Classification],
    human: Mapping[str, Classification],
) -> VerdictReconciliation:
    """Compare two verdict mappings over the REQNOs they share.

    Agreement is measured on the intersection only; ``human_only`` and
    ``pipeline_only`` report the non-overlapping keys so a shrunken overlap
    (e.g. the human 300 barely landing in the ranked cohort) is visible
    rather than silently inflating the agreement rate.
    """
    shared = pipeline.keys() & human.keys()
    agree = sum(1 for reqno in shared if pipeline[reqno] == human[reqno])
    over_clears = sum(
        1
        for reqno in shared
        if human[reqno] == "INAPPROPRIATE" and pipeline[reqno] == "APPROPRIATE"
    )
    return VerdictReconciliation(
        overlap=len(shared),
        agree=agree,
        disagree=len(shared) - agree,
        pipeline_over_clears=over_clears,
        human_only=len(human.keys() - pipeline.keys()),
        pipeline_only=len(pipeline.keys() - human.keys()),
    )
