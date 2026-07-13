"""Regression tests for :mod:`bba.report_generator.builder`.

The builder projects :class:`bba.audit_store.AuditRow`\\s into a
:class:`bba.report_generator.ReportInputs` ready for
:func:`generate_monthly_report`. These tests lock the cross-row
invariants the builder enforces (and the ones it deliberately does
*not*).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from bba.audit_store import AuditRow
from bba.report_generator import (
    MissingResolverError,
    MixedRunMetadataError,
    build_report_inputs,
    default_classification_projector,
)


def test_default_projector_rejects_preop_reservation_unconfirmed() -> None:
    with pytest.raises(MissingResolverError, match="pool it into Unresolved"):
        default_classification_projector("PREOP_RESERVATION_UNCONFIRMED")


def test_default_projector_rejects_returned_not_transfused() -> None:
    with pytest.raises(MissingResolverError, match="excluded, non-scorable"):
        default_classification_projector("RETURNED_NOT_TRANSFUSED")


def test_default_projector_rejects_periop_transfusion_exempt() -> None:
    with pytest.raises(MissingResolverError, match="excluded, non-scorable"):
        default_classification_projector("PERIOP_TRANSFUSION_EXEMPT")


def _row(
    *,
    audit_id: str = "audit-001",
    evidence_bundle_hash: str = "bundle-sha256-aaa",
    order_datetime: datetime | None = None,
    **overrides: object,
) -> AuditRow:
    """Minimal valid :class:`AuditRow` builder for builder-level tests.

    Mirrors the shape of ``tests/unit/test_audit_store.py::_row`` so the
    cross-test convention stays consistent; defaults are deliberately
    bland so each test reads as "the field under test"."""
    base: dict[str, object] = {
        "audit_id": audit_id,
        "run_id": "run-aaa",
        "run_timestamp": datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC),
        "hn_hash": "hn-sha256-aaa",
        "an_hash": "an-sha256-bbb",
        "reqno": "REQ-12345",
        "order_datetime": order_datetime or datetime(2026, 5, 1, 8, 30, 0, tzinfo=UTC),
        "products_ordered": ("LPRC",),
        "hb_value": 6.8,
        "hb_datetime": datetime(2026, 5, 1, 7, 0, 0, tzinfo=UTC),
        "hb_freshness": "fresh_<6h",
        "hb_source": "LABEXM",
        "vitals_sbp": 95.0,
        "vitals_hr": 110.0,
        "vitals_timestamp": datetime(2026, 5, 1, 8, 0, 0, tzinfo=UTC),
        "vitals_source": "IPDADMPROGRESS",
        "prior_rbc_units_24h": 0,
        "prior_rbc_units_7d": 2,
        "cohort_threshold": 7.0,
        "delta_hb_window_results": (),
        "rule_classification": "APPROPRIATE",
        "final_classification": "APPROPRIATE",
        "cohort_applied": "general_medical",
        "indications_json": (),
        "negative_evidence_json": (),
        "confidence": 0.91,
        "reasoning_summary_thai": "",
        "reasoning_summary_en": "",
        "needs_human_review": False,
        "review_reason": None,
        "model_id": "claude-sonnet-4-6-20260201",
        "prompt_hash": "prompt-sha256-ccc",
        "evidence_bundle_hash": evidence_bundle_hash,
        "redactor_version": "0.4.1",
        "redactor_model_sha": "redactor-sha256-eee",
        "policy_version": "kcmh-pr17.2-2024",
        "verifier_pass": True,
        "verifier_retries": 0,
        "escalated_to_opus": False,
    }
    base.update(overrides)
    return AuditRow.model_validate(base)


def _store_returning(
    *rows: AuditRow,
    code_version_slug: str = "slug-v0-1-0",
    code_version: str = "v0.1.0+test",
) -> MagicMock:
    """Return a MagicMock that quacks like :class:`AuditStore` for the
    builder's two calls.

    The builder reads via :meth:`AuditStore.read_run_records`, which
    returns ``(AuditRow, code_version_slug)`` tuples — every row in
    the helper output gets paired with the supplied ``code_version_slug``
    (single-version, the normal case). For multi-version test setups
    use :func:`_store_with_slugged_records` directly."""
    store = MagicMock(name="audit_store")
    store.config.code_version = code_version
    store.read_run_records.return_value = tuple(
        (row, code_version_slug) for row in rows
    )
    return store


def _store_with_slugged_records(
    *records: tuple[AuditRow, str],
) -> MagicMock:
    """Return a MagicMock whose ``read_run_records`` returns the
    supplied ``(row, slug)`` pairs verbatim. Use for multi-version
    fixtures where different rows carry different slugs."""
    store = MagicMock(name="audit_store")
    store.config.code_version = "v0.1.0+test"
    store.read_run_records.return_value = tuple(records)
    return store


class TestEvidenceBundleHashIsPerRow:
    """The audit-store schema documents ``evidence_bundle_hash`` as a
    **per-row** SHA (each row's evidence bundle has its own hash). The
    report-generator's :class:`ReportFooter` field of the same name is
    documented as the schema-level hash. The builder must therefore not
    equality-check ``evidence_bundle_hash`` across rows — doing so would
    fail every multi-row run.

    Codex flagged this on PR #71; the regression here ensures a future
    refactor cannot silently re-add the field to the equality set."""

    def test_rows_with_different_bundle_hashes_do_not_raise(
        self, tmp_path: Path
    ) -> None:
        """Two rows in the same run carrying different bundle hashes
        is the *normal* case (each audit_id has its own evidence). The
        builder must succeed."""
        rows = (
            _row(audit_id="a1", evidence_bundle_hash="bundle-sha-aaa"),
            _row(audit_id="a2", evidence_bundle_hash="bundle-sha-bbb"),
        )
        inputs = build_report_inputs(
            run_id="run-aaa",
            audit_store=_store_returning(*rows),
            output_dir=tmp_path,
            ward_resolver=lambda _r: "ward-1",
            physician_resolver=lambda _r: "phys-1",
        )
        assert len(inputs.rows) == 2

    def test_footer_bundle_hash_is_deterministic_per_run(self, tmp_path: Path) -> None:
        """The footer's ``evidence_bundle_hash`` is a stable per-run
        digest (sha256 over the sorted ``(audit_id, bundle_hash)``
        pairs), so two invocations on the same audit_store yield the
        same footer value — reproducibility is preserved even though
        the field is no longer equality-checked."""
        rows = (
            _row(audit_id="a1", evidence_bundle_hash="bundle-sha-aaa"),
            _row(audit_id="a2", evidence_bundle_hash="bundle-sha-bbb"),
        )
        first = build_report_inputs(
            run_id="run-aaa",
            audit_store=_store_returning(*rows),
            output_dir=tmp_path,
            ward_resolver=lambda _r: "ward-1",
            physician_resolver=lambda _r: "phys-1",
        )
        second = build_report_inputs(
            run_id="run-aaa",
            audit_store=_store_returning(*rows),
            output_dir=tmp_path,
            ward_resolver=lambda _r: "ward-1",
            physician_resolver=lambda _r: "phys-1",
        )
        assert first.footer.evidence_bundle_hash != ""
        assert first.footer.evidence_bundle_hash == second.footer.evidence_bundle_hash

    def test_footer_bundle_hash_is_order_independent(self, tmp_path: Path) -> None:
        """Re-ordering the rows must not change the footer's bundle-
        hash digest; the builder sorts before hashing for exactly this
        property (the audit_store's read order is implementation-
        defined)."""
        row_a = _row(audit_id="a1", evidence_bundle_hash="bundle-sha-aaa")
        row_b = _row(audit_id="a2", evidence_bundle_hash="bundle-sha-bbb")
        forward = build_report_inputs(
            run_id="run-aaa",
            audit_store=_store_returning(row_a, row_b),
            output_dir=tmp_path,
            ward_resolver=lambda _r: "ward-1",
            physician_resolver=lambda _r: "phys-1",
        )
        reversed_ = build_report_inputs(
            run_id="run-aaa",
            audit_store=_store_returning(row_b, row_a),
            output_dir=tmp_path,
            ward_resolver=lambda _r: "ward-1",
            physician_resolver=lambda _r: "phys-1",
        )
        assert (
            forward.footer.evidence_bundle_hash == reversed_.footer.evidence_bundle_hash
        )


class TestRunLevelFieldsAreEqualityChecked:
    """The five truly run-level fields (``policy_version``, ``model_id``,
    ``redactor_version``, ``redactor_model_sha``, ``prompt_hash``) MUST
    agree across every row — :class:`PipelineRowContext` documents them
    as "pinned per run". Disagreement signals a corrupt run (e.g., two
    different runs accidentally sharing a ``run_id``) and must fail
    loud."""

    @pytest.mark.parametrize(
        "field",
        [
            "policy_version",
            "model_id",
            "redactor_version",
            "redactor_model_sha",
            "prompt_hash",
        ],
    )
    def test_mismatch_raises_mixed_run_metadata_error(
        self, tmp_path: Path, field: str
    ) -> None:
        rows = (
            _row(audit_id="a1"),
            _row(audit_id="a2", **{field: "different-value"}),
        )
        with pytest.raises(MixedRunMetadataError) as excinfo:
            build_report_inputs(
                run_id="run-aaa",
                audit_store=_store_returning(*rows),
                output_dir=tmp_path,
                ward_resolver=lambda _r: "ward-1",
                physician_resolver=lambda _r: "phys-1",
            )
        assert field in str(excinfo.value)


class TestReadUsesReadRunRecords:
    """The builder reads via :meth:`AuditStore.read_run_records` so it
    has direct visibility into each row's committed ``code_version_slug``.
    This is the API surface that lets :func:`_assert_single_code_version`
    catch a multi-version run regardless of audit_id overlap (Codex P1
    review on PR #71). Historical-run readability is preserved because
    the read does not filter on the running binary's ``code_version``."""

    def test_read_run_records_called_with_run_id_only(self, tmp_path: Path) -> None:
        """The read must take just ``run_id`` — no ``code_version``
        kwarg — so a historical run committed under an earlier
        package version remains readable after an upgrade."""
        store = _store_returning(_row(audit_id="a1"))
        build_report_inputs(
            run_id="run-aaa",
            audit_store=store,
            output_dir=tmp_path,
            ward_resolver=lambda _r: "ward-1",
            physician_resolver=lambda _r: "phys-1",
        )
        store.read_run_records.assert_called_once_with(run_id="run-aaa")
        # Explicit negative assertion preserved from the previous test
        # iteration: scoping to the binary's code_version was the
        # earlier wrong fix that hid historical runs after upgrade.
        _args, kwargs = store.read_run_records.call_args
        assert "code_version" not in kwargs, (
            "filtering by the binary's code_version would hide historical "
            "runs after a package upgrade — read_run_records must be called "
            "with run_id only"
        )


class TestRejectMultiVersionRunRows:
    """The audit_store layer permits the same ``run_id`` to carry rows
    from more than one ``code_version_slug`` when a user re-commits
    under ``--run-id``. The builder must refuse this shape outright —
    both the **overlapping** case (same ``audit_id`` committed twice,
    inflating per-order counts) and the **disjoint** case (different
    audit_id sets per version, silently merging incompatible runs)
    (Codex P1 review on PR #71)."""

    def test_overlapping_audit_ids_across_two_slugs_raises(
        self, tmp_path: Path
    ) -> None:
        """Same ``audit_id`` committed under two slugs (the
        ``--run-id`` override re-commit) — the classic
        double-counted aggregator case."""
        row_v1 = _row(audit_id="a1")
        row_v2 = _row(audit_id="a1")
        with pytest.raises(MixedRunMetadataError) as excinfo:
            build_report_inputs(
                run_id="run-aaa",
                audit_store=_store_with_slugged_records(
                    (row_v1, "slug-v1"),
                    (row_v2, "slug-v2"),
                ),
                output_dir=tmp_path,
                ward_resolver=lambda _r: "ward-1",
                physician_resolver=lambda _r: "phys-1",
            )
        msg = str(excinfo.value)
        assert "slug-v1" in msg
        assert "slug-v2" in msg

    def test_disjoint_audit_ids_across_two_slugs_raises(self, tmp_path: Path) -> None:
        """**Disjoint** audit_id sets across two slugs (the
        ``--run-id`` override re-commit with changed input scope).
        A duplicate-id heuristic would miss this — no audit_id
        repeats — but the slug set still reveals the cross-version
        membership and the builder must refuse the merge."""
        row_v1_a = _row(audit_id="a1")
        row_v1_b = _row(audit_id="a2")
        row_v2_c = _row(audit_id="a3")
        with pytest.raises(MixedRunMetadataError) as excinfo:
            build_report_inputs(
                run_id="run-aaa",
                audit_store=_store_with_slugged_records(
                    (row_v1_a, "slug-v1"),
                    (row_v1_b, "slug-v1"),
                    (row_v2_c, "slug-v2"),
                ),
                output_dir=tmp_path,
                ward_resolver=lambda _r: "ward-1",
                physician_resolver=lambda _r: "phys-1",
            )
        msg = str(excinfo.value)
        assert "slug-v1" in msg
        assert "slug-v2" in msg

    def test_single_slug_multi_row_run_does_not_raise(self, tmp_path: Path) -> None:
        """Sanity check: the multi-version detector must not
        false-positive on a normal multi-row run from a single
        ``code_version``."""
        rows = (_row(audit_id="a1"), _row(audit_id="a2"), _row(audit_id="a3"))
        inputs = build_report_inputs(
            run_id="run-aaa",
            audit_store=_store_returning(*rows),
            output_dir=tmp_path,
            ward_resolver=lambda _r: "ward-1",
            physician_resolver=lambda _r: "phys-1",
        )
        assert len(inputs.rows) == 3


# =============================================================================
# A2 — Platelet rows excluded from RBC report inputs
# =============================================================================


class TestPlateletRowsExcludedFromRbcReport:
    """Platelet AuditRows must not appear in RBC build_report_inputs output.

    WHY: platelet appropriateness is evaluated against platelet-count thresholds,
    not Hb thresholds. Blending platelet rows into RBC report inputs would dilute
    the RBC inappropriate_rate and produce clinically meaningless monthly committee
    reports. The filter at build_report_inputs (before projection) is the single
    choke point that keeps the RBC and platelet pipelines separate in reporting.
    """

    def test_platelet_rows_not_projected_into_report(self, tmp_path: Path) -> None:
        """A platelet row in the store must not appear in the projected report rows."""
        rbc_row = _row(audit_id="rbc-001")
        platelet_row = _row(audit_id="plt-001", component="platelet")
        inputs = build_report_inputs(
            run_id="run-aaa",
            audit_store=_store_returning(rbc_row, platelet_row),
            output_dir=tmp_path,
            ward_resolver=lambda _r: "ward-1",
            physician_resolver=lambda _r: "phys-1",
        )
        assert len(inputs.rows) == 1
        assert inputs.rows[0].audit_id == "rbc-001"

    def test_rbc_row_count_unchanged_when_platelet_rows_added(
        self, tmp_path: Path
    ) -> None:
        """A mixed store (RBC + platelet) produces the same projected row count as
        an RBC-only store. Platelet rows contribute exactly zero to RBC totals."""
        rbc_only = (_row(audit_id="rbc-001"), _row(audit_id="rbc-002"))
        mixed = (
            _row(audit_id="rbc-001"),
            _row(audit_id="rbc-002"),
            _row(audit_id="plt-001", component="platelet"),
            _row(audit_id="plt-002", component="platelet"),
        )
        rbc_inputs = build_report_inputs(
            run_id="run-aaa",
            audit_store=_store_returning(*rbc_only),
            output_dir=tmp_path / "rbc",
            ward_resolver=lambda _r: "ward-1",
            physician_resolver=lambda _r: "phys-1",
        )
        mixed_inputs = build_report_inputs(
            run_id="run-aaa",
            audit_store=_store_returning(*mixed),
            output_dir=tmp_path / "mixed",
            ward_resolver=lambda _r: "ward-1",
            physician_resolver=lambda _r: "phys-1",
        )
        assert len(mixed_inputs.rows) == len(rbc_inputs.rows) == 2
        assert {r.audit_id for r in mixed_inputs.rows} == {
            r.audit_id for r in rbc_inputs.rows
        }
