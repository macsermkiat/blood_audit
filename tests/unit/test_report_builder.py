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
from bba.report_generator import MixedRunMetadataError, build_report_inputs


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


def _store_returning(*rows: AuditRow, code_version: str = "v0.1.0+test") -> MagicMock:
    """Return a MagicMock that quacks like :class:`AuditStore` for the
    builder's only call (``read_audit_results(run_id=...)``).

    ``config.code_version`` is set explicitly even though the builder no
    longer reads it (see :class:`TestReadDoesNotFilterByBinaryCodeVersion`):
    keeping the field configured documents the audit_store's contract
    surface and lets a future reviewer search-and-find the rationale."""
    store = MagicMock(name="audit_store")
    store.config.code_version = code_version
    store.read_audit_results.return_value = rows
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


class TestReadDoesNotFilterByBinaryCodeVersion:
    """The builder must NOT scope the audit_store read to the running
    binary's ``code_version``. :func:`compute_run_id` already includes
    code_version in its hash (so a ``run_id`` normally pins to one
    committed version by construction), and pinning the read to the
    *current* binary's version would make a run committed under an
    older version invisible after an upgrade (Codex P1 review on PR #71).

    The real safety net for the rare cross-version-mix case is
    :func:`_reconstruct_footer`'s equality check on the five "pinned per
    run" fields, exercised by :class:`TestRunLevelFieldsAreEqualityChecked`."""

    def test_read_is_called_without_code_version(self, tmp_path: Path) -> None:
        """The read must omit ``code_version`` so a historical run
        committed under an earlier package version remains readable
        after an upgrade."""
        store = _store_returning(_row(audit_id="a1"), code_version="v0.2.0+current")
        build_report_inputs(
            run_id="run-aaa",
            audit_store=store,
            output_dir=tmp_path,
            ward_resolver=lambda _r: "ward-1",
            physician_resolver=lambda _r: "phys-1",
        )
        store.read_audit_results.assert_called_once_with(run_id="run-aaa")
        _args, kwargs = store.read_audit_results.call_args
        assert "code_version" not in kwargs, (
            "filtering by the binary's code_version would hide historical "
            "runs after a package upgrade"
        )
