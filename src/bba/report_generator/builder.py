"""Build :class:`ReportInputs` from a run's audit-store rows (issue #28).

The CLI command ``bba report --run-id R`` needs to feed
:func:`generate_monthly_report` a fully-resolved :class:`ReportInputs`.
That requires a projection from :class:`~bba.audit_store.AuditRow`
(which intentionally omits ``ward_id`` / ``physician_id``) into
:class:`~bba.report_generator.MonthlyReportRow`. This module owns that
projection so the CLI stays thin glue and the audit-store schema stays
narrow.

The ward / physician join is supplied as injected resolvers, mirroring
:mod:`bba.dashboard.models`'s established
``WardAttributionResolver`` / ``PhysicianAttributionResolver`` shape.
Production wiring sources these from the HOSxP ingest store; tests
inject lambdas.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path

from bba.audit_store import AuditRow, AuditStore
from bba.audit_store.models import Classification as AuditClassification
from bba.report_generator.aggregate import REPORT_TZ
from bba.report_generator.exceptions import (
    EmptyInputError,
    FooterStampError,
    ReportGenerationError,
)
from bba.report_generator.models import (
    Classification as ReportClassification,
    MonthlyReportRow,
    ReportFooter,
    ReportInputs,
)


WardAttributionResolver = Callable[[AuditRow], str]
"""Resolve ward attribution for an :class:`AuditRow`. Mirrors
:data:`bba.dashboard.models.WardAttributionResolver`. The audit-store
schema deliberately omits ``ward_id`` (computed at view time)."""


PhysicianAttributionResolver = Callable[[AuditRow], str]
"""Resolve attending-physician attribution for an :class:`AuditRow`.
Mirrors :data:`bba.dashboard.models.PhysicianAttributionResolver`."""


IndicationCodesExtractor = Callable[[AuditRow], tuple[str, ...]]
"""Extract the indication-code tuple for one :class:`AuditRow`.

The audit-store stores indications as opaque JSON (``indications_json``);
the report-generator section schemas use a flat ``tuple[str, ...]`` of
codes. The mapping is application-defined, so it is injected with a
sensible default (:func:`default_indication_codes_extractor`)."""


ClassificationProjector = Callable[[AuditClassification], ReportClassification]
"""Project the audit-store's :class:`Classification` literal onto
the report-generator's 4-value literal. The default
(:func:`default_classification_projector`) is identity on the four shared
values and raises on audit-store-only values — callers that want a non-default
treatment (e.g., bucket as ``NEEDS_REVIEW``) inject their own."""


class MissingResolverError(ReportGenerationError):
    """Raised when the production wiring for a resolver is not yet plumbed.

    Distinct from :class:`bba.cli.exceptions.CliError` so the report-
    generator stays decoupled from the CLI layer; the CLI catches this
    and re-raises as a ``CliError`` naming the seam."""


class MixedRunMetadataError(ReportGenerationError):
    """Raised when audit rows for a single ``run_id`` disagree on the
    reproducibility footer (``policy_version``, ``model_id``, etc.) or
    span multiple local-time months.

    Both conditions indicate the run is not a single committee-readable
    bucket — refuse rather than silently emit a report with mixed
    metadata."""


def default_indication_codes_extractor(row: AuditRow) -> tuple[str, ...]:
    """Extract ``item["code"]`` from each ``indications_json`` entry.

    Defensive against the dynamic-schema field: items missing a ``code``
    key, or whose ``code`` is not a non-empty ``str``, are silently
    elided. Callers that need stricter semantics (e.g., raise on a
    missing code) inject their own extractor."""
    codes: list[str] = []
    for item in row.indications_json:
        candidate = item.get("code")
        if isinstance(candidate, str) and candidate:
            codes.append(candidate)
    return tuple(codes)


_EXCLUDED_NONSCORABLE: frozenset[str] = frozenset(
    {"RETURNED_NOT_TRANSFUSED", "PERIOP_TRANSFUSION_EXEMPT"}
)
"""Returns-ledger terminal classifications excluded from the report before
projection (spec #119). Non-scorable — mirrors their exclusion from attribution
rates; the default projector fails loud on them so they must be dropped upstream.
"""


def default_classification_projector(
    value: AuditClassification,
) -> ReportClassification:
    """Identity on the four shared classifications; raise on the audit-
    store-only classifications.

    Failing loud is the right default because mapping
    ``POTENTIALLY_INAPPROPRIATE`` is a clinical decision (the PRD does
    not promise a 1:1 onto the four reportable labels); the caller must
    inject a remap if the run produces this label."""
    if value == "POTENTIALLY_INAPPROPRIATE":
        raise MissingResolverError(
            "audit_store row carries final_classification="
            "'POTENTIALLY_INAPPROPRIATE'; inject a classification_projector "
            "to map this onto the report-generator's 4-value Classification "
            "(APPROPRIATE / INAPPROPRIATE / NEEDS_REVIEW / "
            "INSUFFICIENT_EVIDENCE)"
        )
    if value == "PREOP_RESERVATION_UNCONFIRMED":
        raise MissingResolverError(
            "audit_store row carries final_classification="
            "'PREOP_RESERVATION_UNCONFIRMED'; inject a projector to pool it "
            "into Unresolved"
        )
    if value == "RETURNED_NOT_TRANSFUSED":
        raise MissingResolverError(
            "audit_store row carries final_classification="
            "'RETURNED_NOT_TRANSFUSED'; inject an explicit projector for this "
            "excluded, non-scorable terminal classification"
        )
    if value == "PERIOP_TRANSFUSION_EXEMPT":
        raise MissingResolverError(
            "audit_store row carries final_classification="
            "'PERIOP_TRANSFUSION_EXEMPT'; inject an explicit projector for this "
            "excluded, non-scorable terminal classification"
        )
    # The remaining four audit-store classifications are exactly the
    # report-generator's literal members; the narrowed return is safe.
    return value


def build_report_inputs(
    *,
    run_id: str,
    audit_store: AuditStore,
    output_dir: Path,
    ward_resolver: WardAttributionResolver,
    physician_resolver: PhysicianAttributionResolver,
    indication_codes_extractor: IndicationCodesExtractor = (
        default_indication_codes_extractor
    ),
    classification_projector: ClassificationProjector = (
        default_classification_projector
    ),
    physician_ids_for_own_view: Sequence[str] | None = None,
) -> ReportInputs:
    """Read all committed :class:`AuditRow`\\s for ``run_id`` and project
    them into a :class:`ReportInputs` ready for
    :func:`generate_monthly_report`.

    The month is inferred from the rows themselves: every row's
    ``order_datetime`` must fall inside the same Asia/Bangkok local
    month, otherwise :class:`MixedRunMetadataError` is raised. The
    reproducibility footer is reconstructed by asserting all rows agree
    on every footer field (same invariant — fail loud on disagreement).

    ``physician_ids_for_own_view`` defaults to the set of distinct
    ``physician_id``\\s observed in the run (sorted, deduplicated). Pass
    an explicit tuple to restrict (e.g., committee-only run).

    The read uses :meth:`AuditStore.read_run_records` so each row's
    ``code_version_slug`` is visible. The builder does **not** filter
    by the running binary's ``code_version`` —
    :func:`bba.cli.identity.compute_run_id` already namespaces
    ``run_id`` by version in the normal flow, so historical runs remain
    readable after an upgrade. The rarer ``--run-id``-override
    re-commit case (which can land rows from multiple code_versions
    at the same ``run_id``) is caught directly by inspecting the slug
    set: if more than one distinct slug is present,
    :class:`MixedRunMetadataError` is raised. This catches both the
    **overlapping** case (same ``audit_id`` committed twice) AND the
    **disjoint** case (each version committed a non-overlapping audit
    set), which a duplicate-id heuristic would miss (Codex P1 review
    on PR #71).
    """
    records = audit_store.read_run_records(run_id=run_id)
    if not records:
        raise EmptyInputError(
            f"audit_store has no committed AuditRow for run_id={run_id!r}; "
            "the run either has not completed or its commit markers are "
            "missing — investigate before shipping an empty report"
        )
    _assert_single_code_version(records, run_id)
    # Filter to red_cell rows before projection. Platelet rows (component=
    # "platelet") use different clinical thresholds and must not dilute RBC
    # report statistics. The filter happens at the source so every downstream
    # aggregate function operates on a homogeneous red_cell set.
    rows = tuple(row for row, _slug in records if row.component == "red_cell")
    if not rows:
        raise EmptyInputError(
            f"audit_store has rows for run_id={run_id!r} but none are "
            "component='red_cell'; the RBC report cannot be generated from "
            "a platelet-only run"
        )

    # Validate run-level integrity over the FULL red_cell set BEFORE dropping
    # any rows. The reproducibility footer and single-local-month invariants
    # must hold across every committed red_cell AuditRow, including the returns
    # terminals: a terminal row carrying a divergent policy_version / model_id /
    # prompt_hash or a different local month is the same mixed-run corruption
    # these checks exist to fail loud on, and dropping it first would mask it.
    footer = _reconstruct_footer(rows)
    month = _infer_month(rows)

    # Drop the non-scorable returns terminals for projection / scoring only.
    # With the returns-ledger router enabled (spec #119 go-live) a run's
    # audit_store can carry RETURNED_NOT_TRANSFUSED / PERIOP_TRANSFUSION_EXEMPT
    # final_classifications; these are non-events / OR-exempt, excluded from the
    # report exactly as they are excluded from attribution rates, and the
    # default projector fails loud on them.
    scorable_rows = tuple(
        row for row in rows if row.final_classification not in _EXCLUDED_NONSCORABLE
    )
    if not scorable_rows:
        raise EmptyInputError(
            f"audit_store red_cell rows for run_id={run_id!r} are all "
            "non-scorable returns terminals (RETURNED_NOT_TRANSFUSED / "
            "PERIOP_TRANSFUSION_EXEMPT); there is no scorable order to report"
        )

    monthly_rows = tuple(
        _project_row(
            row,
            ward_resolver=ward_resolver,
            physician_resolver=physician_resolver,
            indication_codes_extractor=indication_codes_extractor,
            classification_projector=classification_projector,
        )
        for row in scorable_rows
    )
    physician_ids = _resolve_physician_ids_for_own_view(
        monthly_rows, physician_ids_for_own_view
    )

    return ReportInputs(
        month=month,
        rows=monthly_rows,
        footer=footer,
        output_dir=output_dir,
        physician_ids_for_own_view=physician_ids,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_FOOTER_EQUALITY_FIELDS: tuple[str, ...] = (
    "policy_version",
    "model_id",
    "redactor_version",
    "redactor_model_sha",
    "prompt_hash",
)
"""Footer fields that MUST agree across every audit row in the run.

These are the truly run-level reproducibility stamps: the deterministic
classifier's policy version, the LLM model ID, the redactor version /
model SHA, and the system-prompt hash. :class:`PipelineRowContext`
documents them as "pinned per run".

:attr:`AuditRow.evidence_bundle_hash` is deliberately omitted: it is a
**per-row** value (each row's evidence bundle has its own SHA — see
:class:`bba.evidence_bundle_builder.builder.EvidenceBundle`), so
asserting cross-row equality would fail every multi-row run. The
:class:`ReportFooter` field of the same name is derived via
:func:`_aggregate_evidence_bundle_hash` instead.
"""


def _assert_single_code_version(
    records: Sequence[tuple[AuditRow, str]], run_id: str
) -> None:
    """Raise :class:`MixedRunMetadataError` if ``records`` span more
    than one ``code_version_slug``.

    Reading via :meth:`AuditStore.read_run_records` gives the builder
    direct visibility into each row's committed version. A single
    ``run_id`` mapping to multiple slugs means the audit_store holds a
    cross-version mix — either an **overlapping** set (the same
    ``audit_id`` committed under two versions) or a **disjoint** set
    (each version committed different audit_ids under a shared
    ``--run-id`` override). Both shapes corrupt the report:
    overlapping ones inflate per-order counts, disjoint ones merge
    semantically-incompatible runs into one aggregate. Inspecting the
    slug set directly catches both shapes regardless of whether any
    footer field drifted, which a duplicate-id heuristic alone cannot
    do (Codex P1 review on PR #71)."""
    distinct_slugs = sorted({slug for _row, slug in records})
    if len(distinct_slugs) > 1:
        raise MixedRunMetadataError(
            f"audit_store returned AuditRows from multiple code_version "
            f"slugs for run_id={run_id!r}: {distinct_slugs}; this means "
            "the same run_id was committed under more than one "
            "code_version (a --run-id override re-run). Refusing to "
            "render a report that would merge incompatible runs. "
            "Re-derive the run with a version-namespaced run_id (let "
            "compute_run_id pick it) or scope the read to one version "
            "via the audit_store directly."
        )


def _reconstruct_footer(rows: Sequence[AuditRow]) -> ReportFooter:
    """Build the :class:`ReportFooter` from the run's audit rows.

    The five :data:`_FOOTER_EQUALITY_FIELDS` are required to be identical
    across every row (a single run produces a single reproducibility
    stamp); disagreement raises :class:`MixedRunMetadataError`.

    The footer's ``evidence_bundle_hash`` field is derived from the
    per-row hashes via :func:`_aggregate_evidence_bundle_hash` rather
    than equality-checked, because each audit row carries its own
    bundle-specific SHA by design (Codex review on PR #71).
    """
    pivot = rows[0]
    expected = {field: getattr(pivot, field) for field in _FOOTER_EQUALITY_FIELDS}
    for row in rows[1:]:
        for field, value in expected.items():
            actual = getattr(row, field)
            if actual != value:
                raise MixedRunMetadataError(
                    f"audit rows for run disagree on {field}: "
                    f"expected {value!r} (from audit_id={pivot.audit_id!r}) "
                    f"got {actual!r} (from audit_id={row.audit_id!r}); "
                    "a single run must produce a single reproducibility "
                    "footer — investigate the audit_store contents"
                )
    expected["evidence_bundle_hash"] = _aggregate_evidence_bundle_hash(rows)
    try:
        return ReportFooter(**expected)
    except ValueError as exc:  # pragma: no cover - defensive
        raise FooterStampError(
            f"unable to reconstruct ReportFooter from run rows: {exc}"
        ) from exc


def _aggregate_evidence_bundle_hash(rows: Sequence[AuditRow]) -> str:
    """Return a deterministic per-run digest of the rows' bundle hashes.

    Each :class:`AuditRow` carries its own ``evidence_bundle_hash`` (the
    SHA of that row's specific evidence bundle), so there is no single
    "schema hash" available from the audit_store today. To still stamp
    the footer with a reproducible per-run identifier, we hash the
    sorted set of ``(audit_id, evidence_bundle_hash)`` pairs — sorted so
    the aggregate is order-independent, paired with ``audit_id`` so it
    is deterministically derivable from the audit_store's contents.

    A future refactor that lands a true canonical-schema hash (e.g.,
    via :mod:`bba.evidence_bundle_builder`) should replace this with the
    schema-level value; the call site is single-sourced here."""
    hasher = hashlib.sha256()
    for audit_id, bundle_hash in sorted(
        (row.audit_id, row.evidence_bundle_hash) for row in rows
    ):
        hasher.update(audit_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(bundle_hash.encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _project_row(
    row: AuditRow,
    *,
    ward_resolver: WardAttributionResolver,
    physician_resolver: PhysicianAttributionResolver,
    indication_codes_extractor: IndicationCodesExtractor,
    classification_projector: ClassificationProjector,
) -> MonthlyReportRow:
    """Project one :class:`AuditRow` into one :class:`MonthlyReportRow`."""
    return MonthlyReportRow(
        audit_id=row.audit_id,
        an_hash=row.an_hash,
        hn_hash=row.hn_hash,
        order_datetime=row.order_datetime,
        ward_id=ward_resolver(row),
        physician_id=physician_resolver(row),
        final_classification=classification_projector(row.final_classification),
        cohort_applied=row.cohort_applied,
        indication_codes=indication_codes_extractor(row),
        needs_human_review=row.needs_human_review,
    )


def _infer_month(rows: Sequence[AuditRow]) -> date:
    """Return the Asia/Bangkok local-month-first-of-month shared by every
    row. Raise :class:`MixedRunMetadataError` on any disagreement.

    Runs over the full red_cell audit set (including any non-scorable returns
    terminals) so a divergent-month terminal row is caught, not silently
    dropped, before scorable rows are projected.

    Using the same :data:`REPORT_TZ` as
    :func:`bba.report_generator.aggregate.filter_rows_for_month` keeps
    the bucketing rule single-sourced: a row that survives this check
    is guaranteed to land in the filter's local-month window."""
    months = {_local_month(row.order_datetime) for row in rows}
    if len(months) > 1:
        sorted_months = sorted(months)
        raise MixedRunMetadataError(
            f"audit rows span multiple Asia/Bangkok local months: "
            f"{[m.isoformat() for m in sorted_months]}; the monthly "
            "report bucket is a single business month — split the run "
            "or pass an explicit --month flag (not yet implemented)"
        )
    return next(iter(months))


def _local_month(order_datetime: datetime) -> date:
    """Return first-of-month (Asia/Bangkok local) for ``order_datetime``.

    The audit-store enforces UTC; this conversion makes the month
    boundary match the one used downstream by
    :func:`filter_rows_for_month`."""
    local = order_datetime.astimezone(REPORT_TZ)
    return date(local.year, local.month, 1)


def _resolve_physician_ids_for_own_view(
    rows: Sequence[MonthlyReportRow],
    override: Sequence[str] | None,
) -> tuple[str, ...]:
    """Default to every distinct physician in the run; honour explicit
    overrides verbatim (including the empty tuple)."""
    if override is not None:
        return tuple(override)
    return tuple(sorted({row.physician_id for row in rows}))


__all__ = [
    "ClassificationProjector",
    "IndicationCodesExtractor",
    "MissingResolverError",
    "MixedRunMetadataError",
    "PhysicianAttributionResolver",
    "WardAttributionResolver",
    "build_report_inputs",
    "default_classification_projector",
    "default_indication_codes_extractor",
]
