"""RED-phase failing tests for issue #4 (bba.audit_orders).

Each ``class`` maps to one acceptance criterion in the issue body. Tests assert
contracts (the WHY) — see PRD §"Testing Decisions". No implementation exists
yet; every test MUST fail in this scaffold commit (NotImplementedError on the
behavioral predicates, or assertion failure when the pipeline can't even
produce a partition).

The acceptance-criterion → test-class map:

* AC ① "Implementation in ``src/bba/audit_orders/``"
  → import surface verified at module top; collection failure means the
    public API is mis-scaffolded.
* AC ② "Golden-fixture tests per excluded subgroup"
  → :class:`TestOpdNoAnExclusion`, :class:`TestInterHospitalExclusion`,
    :class:`TestHemoglobinopathyExclusion` (one test per code),
    :class:`TestRefusedStatusExclusion`, :class:`TestCancelledExclusion`,
    :class:`TestNonRbcProductExclusion`.
* AC ③ "Anchor-imputation flag emitted when REQDATE/REQTIME is null"
  → :class:`TestAnchorPrimary`, :class:`TestAnchorImputed`,
    :class:`TestAnchorUnrecoverable`.
* AC ④ "Output schema matches PRD §Output schema identity + anchor fields"
  → :class:`TestOutputSchemaIdentityAndAnchor`,
    :class:`TestAuditIdDeterminism`.
* AC ⑤ "Coverage >= 80%" — covered by the suite as a whole, plus property
  tests below.

Property + adversarial tests:

* :class:`TestPartitionInvariant` — every input lands in exactly one bucket
  (the foundational anti-silent-drop contract).
* :class:`TestAuditIdDeterminism` — same ``(hn, reqno)`` → same audit_id
  across runs; different pairs → different audit_ids (hypothesis).
* :class:`TestAdversarialIcdMatching` — D55.999 still counts as D55;
  dotless ``"D550"`` is canonicalized to D55.0 and excluded; case
  sensitivity is preserved (Round 1 B1 hard-exclude must not be evadable
  by code formatting drift).
* :class:`TestAdversarialAnchorTimes` — the strict time parser's
  ``parse_warning`` path disqualifies a pair from being the anchor; the
  fallback must take over, not silently emit a sentinel time.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bba.audit_orders import (
    AnchorResolution,
    AuditOrder,
    AuditOrdersConfig,
    BloodOrderInput,
    ExcludedRecord,
    FilterResult,
    UnrecoverableAnchorError,
    build_audit_id,
    build_audit_orders,
    check_an_scoped,
    check_cancelled,
    check_hemoglobinopathy,
    check_rbc_product,
    check_request_type,
    check_status,
    is_rbc_product,
    rbc_products_in,
    resolve_anchor,
)
from bba.ingest.models import ParsedTimeOfDay


# =============================================================================
# Fixtures
# =============================================================================


def _input(**overrides: Any) -> BloodOrderInput:
    """Build a fully-valid :class:`BloodOrderInput` with sensible defaults.

    Every field is plausible by default so a test that overrides one field
    isolates the rule under test without filling 12 unrelated kwargs.
    """
    defaults: dict[str, Any] = {
        "hn": "HN-0001",
        "an": "AN-0001",
        "reqno": "REQ-0001",
        "bdvstst": "4",
        "reqtype": "P",
        "canceldate": None,
        "req_date": date(2026, 5, 1),
        "req_time": ParsedTimeOfDay(hour=8, minute=30, second=0),
        "bdvst_date": date(2026, 5, 1),
        "bdvst_time": ParsedTimeOfDay(hour=8, minute=45, second=0),
        "products": ("LPRC",),
        "diagnosis_codes": ("I50.9",),  # heart failure — not excluded
    }
    defaults.update(overrides)
    return BloodOrderInput(**defaults)


@pytest.fixture
def config() -> AuditOrdersConfig:
    return AuditOrdersConfig(code_version="v0.1.0-test")


# =============================================================================
# AC: Implementation in src/bba/audit_orders/ — public surface
#
# The imports at the top of this file fully exercise the public API. Collection
# failure means the module is mis-scaffolded; behavioral tests below confirm
# the surface is wired to real behavior, not just present as stubs.
# =============================================================================


# =============================================================================
# AC: Golden fixture per excluded subgroup
# =============================================================================


class TestHappyPathInclusion:
    """A clean order that passes every gate must land in ``included``.

    This is the anti-bug-class test: if the filter ever flips to "fail
    closed" mode, this fixture will reveal it before any of the per-rule
    fixtures even matter.
    """

    def test_clean_record_is_included(self, config: AuditOrdersConfig) -> None:
        result = build_audit_orders([_input()], config)
        assert isinstance(result, FilterResult)
        assert len(result.included) == 1
        assert len(result.excluded) == 0
        order = result.included[0]
        assert isinstance(order, AuditOrder)
        assert order.hn == "HN-0001"
        assert order.reqno == "REQ-0001"


class TestNonRbcProductExclusion:
    """Products outside :data:`RBC_PRODUCTS` → exclusion ``not_rbc_product``."""

    @pytest.mark.parametrize("product", ["PRC", "FFP", "PLT", "CRYO", "ALBUMIN"])
    def test_non_rbc_product_excluded(
        self, config: AuditOrdersConfig, product: str
    ) -> None:
        result = build_audit_orders([_input(products=(product,))], config)
        assert len(result.included) == 0
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == "not_rbc_product"

    @pytest.mark.parametrize("product", ["LPRC", "LDPRC", "SDR"])
    def test_each_rbc_product_included(
        self, config: AuditOrdersConfig, product: str
    ) -> None:
        result = build_audit_orders([_input(products=(product,))], config)
        assert len(result.included) == 1
        assert product in result.included[0].products_ordered


class TestRefusedStatusExclusion:
    """``BDVSTST = 6`` (refused) is hard-excluded per issue #4 AC."""

    def test_refused_status_excluded(self, config: AuditOrdersConfig) -> None:
        result = build_audit_orders([_input(bdvstst="6")], config)
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == "status_not_eligible"

    @pytest.mark.parametrize("status", ["1", "2", "3", "7", "9", "0"])
    def test_other_ineligible_statuses_also_excluded(
        self, config: AuditOrdersConfig, status: str
    ) -> None:
        result = build_audit_orders([_input(bdvstst=status)], config)
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == "status_not_eligible"

    @pytest.mark.parametrize("status", ["4", "5"])
    def test_eligible_statuses_pass_the_status_gate(
        self, config: AuditOrdersConfig, status: str
    ) -> None:
        result = build_audit_orders([_input(bdvstst=status)], config)
        # Must NOT be excluded by status; some other gate may still fire
        # but the reason here must not be ``status_not_eligible``.
        assert all(r.reason != "status_not_eligible" for r in result.excluded), (
            "eligible status should not be excluded by the status gate"
        )


class TestCancelledExclusion:
    """``CANCELDATE`` non-null → hard-excluded per issue #4 AC."""

    def test_cancelled_record_excluded(self, config: AuditOrdersConfig) -> None:
        result = build_audit_orders([_input(canceldate="20260501")], config)
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == "cancelled"

    def test_empty_string_canceldate_is_NOT_cancelled(
        self, config: AuditOrdersConfig
    ) -> None:
        # PRD §1 strict-loud: an empty string in a CSV column is the
        # missing-value sentinel, not a cancellation. The filter must
        # NOT treat ``""`` as a cancel date — silent string-truthiness
        # would over-exclude. (Codex review feedback: the prior
        # assertion passed ``None`` instead of ``""``, so the empty
        # branch could regress silently.)
        result = build_audit_orders([_input(canceldate="")], config)
        assert all(r.reason != "cancelled" for r in result.excluded)

    def test_whitespace_only_canceldate_is_NOT_cancelled(
        self, config: AuditOrdersConfig
    ) -> None:
        # Same rule as empty string: whitespace-only is the missing-value
        # convention, not a real cancellation timestamp.
        result = build_audit_orders([_input(canceldate="   ")], config)
        assert all(r.reason != "cancelled" for r in result.excluded)


class TestOpdNoAnExclusion:
    """OPD orders (``AN`` is None) → excluded with reason ``no_an``."""

    def test_no_an_excluded(self, config: AuditOrdersConfig) -> None:
        result = build_audit_orders([_input(an=None)], config)
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == "no_an"

    def test_empty_an_is_treated_as_no_an(self, config: AuditOrdersConfig) -> None:
        # An empty-string AN is not a real admission number; treating it as
        # present would let an OPD order slip into the audit set.
        result = build_audit_orders([_input(an="")], config)
        assert any(r.reason == "no_an" for r in result.excluded)


class TestInterHospitalExclusion:
    """``REQTYPE='H'`` → inter-hospital referral, hard-excluded."""

    def test_inter_hospital_excluded(self, config: AuditOrdersConfig) -> None:
        result = build_audit_orders([_input(reqtype="H")], config)
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == "inter_hospital"

    def test_p_request_type_is_not_inter_hospital(
        self, config: AuditOrdersConfig
    ) -> None:
        result = build_audit_orders([_input(reqtype="P")], config)
        assert all(r.reason != "inter_hospital" for r in result.excluded)


class TestHemoglobinopathyExclusion:
    """Each of D55/D56/D57/D58 gets its own fixture per issue #4 AC.

    Round 1 B1 ("hemoglobinopathy hard-exclude") is the decision; Round 2
    discussion of G6PD/D55 is documented in the issue references. The AC
    list is the source of truth: D55 is in the hard-exclude set.
    """

    def test_d55_excluded(self, config: AuditOrdersConfig) -> None:
        result = build_audit_orders([_input(diagnosis_codes=("D55.0",))], config)
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == "hemoglobinopathy"
        assert result.excluded[0].detail == "D55.0"

    def test_d56_excluded(self, config: AuditOrdersConfig) -> None:
        result = build_audit_orders([_input(diagnosis_codes=("D56.1",))], config)
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == "hemoglobinopathy"
        assert result.excluded[0].detail == "D56.1"

    def test_d57_excluded(self, config: AuditOrdersConfig) -> None:
        result = build_audit_orders([_input(diagnosis_codes=("D57.1",))], config)
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == "hemoglobinopathy"
        assert result.excluded[0].detail == "D57.1"

    def test_d58_excluded(self, config: AuditOrdersConfig) -> None:
        result = build_audit_orders([_input(diagnosis_codes=("D58.9",))], config)
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == "hemoglobinopathy"
        assert result.excluded[0].detail == "D58.9"

    def test_no_dot_code_d55_bare(self, config: AuditOrdersConfig) -> None:
        # HOSxP sometimes exports the 3-char code without subcategory.
        # ``"D55"`` alone must still match the hard-exclusion.
        result = build_audit_orders([_input(diagnosis_codes=("D55",))], config)
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == "hemoglobinopathy"


class TestFormerlyExcludedCohortsNowInScope:
    """AIHA / TMA / obstetric were dropped from the hard-exclusion set on
    2026-05-29 — those cohorts are now auditable and must pass the filter.

    This guards the *intent* of the narrowing: an otherwise-clean order
    carrying one of these codes lands in ``included`` (not silently
    excluded for some other reason). Without this, the deletion could
    regress and the partition test would still pass.
    """

    @pytest.mark.parametrize(
        "code",
        [
            "D59.3",  # AIHA (autoimmune hemolytic anemia)
            "M31.1",  # TMA (thrombotic thrombocytopenic purpura)
            "O09.9",  # obstetric (pregnancy supervision)
            "O80",  # obstetric (single spontaneous delivery, bare 3-char)
        ],
    )
    def test_code_now_included(self, config: AuditOrdersConfig, code: str) -> None:
        result = build_audit_orders([_input(diagnosis_codes=(code,))], config)
        assert len(result.included) == 1
        assert result.included[0].reqno == "REQ-0001"
        assert result.excluded == ()


# =============================================================================
# AC: Anchor-imputation flag emitted when REQDATE/REQTIME is null
# =============================================================================


class TestAnchorPrimary:
    """When REQ pair is usable, anchor is primary and ``anchor_imputed = False``."""

    def test_primary_anchor_used_when_req_pair_complete(
        self, config: AuditOrdersConfig
    ) -> None:
        result = build_audit_orders([_input()], config)
        assert len(result.included) == 1
        order = result.included[0]
        assert order.anchor_imputed is False
        # tz-aware UTC; matches Bangkok local 2026-05-01T08:30 → UTC 01:30
        assert order.order_datetime == datetime.fromisoformat(
            "2026-05-01T01:30:00+00:00"
        )


class TestAnchorImputed:
    """When REQ pair is missing/unusable, the fallback BDVST pair supplies
    the anchor and ``anchor_imputed = True``."""

    def test_imputed_when_req_date_null(self, config: AuditOrdersConfig) -> None:
        result = build_audit_orders([_input(req_date=None)], config)
        assert len(result.included) == 1
        order = result.included[0]
        assert order.anchor_imputed is True
        # Fallback Bangkok 2026-05-01T08:45 → UTC 01:45
        assert order.order_datetime == datetime.fromisoformat(
            "2026-05-01T01:45:00+00:00"
        )

    def test_imputed_when_req_time_null(self, config: AuditOrdersConfig) -> None:
        # PRD §1 strict-loud: an unparseable HOSxP time → ParsedTimeOfDay
        # is None (parse_warning fires upstream). The audit_orders filter
        # must NOT silently substitute 00:00:00; the BDVST fallback must
        # take over and flag the imputation.
        result = build_audit_orders([_input(req_time=None)], config)
        assert len(result.included) == 1
        assert result.included[0].anchor_imputed is True

    def test_imputed_when_both_req_fields_null(self, config: AuditOrdersConfig) -> None:
        result = build_audit_orders([_input(req_date=None, req_time=None)], config)
        assert len(result.included) == 1
        assert result.included[0].anchor_imputed is True


class TestAnchorUnrecoverable:
    """Both pairs unusable → :class:`UnrecoverableAnchorError`.

    Per PRD §"Output schema", every persisted ``audit_orders`` row carries
    an ``order_datetime``. A null anchor would corrupt downstream stages
    (hb_lookup's -7d window, vitals_extractor's ±6h window). Fail loud
    rather than emit a row whose anchor would silently default to "now".
    """

    def test_both_pairs_missing_raises(self, config: AuditOrdersConfig) -> None:
        record = _input(
            req_date=None,
            req_time=None,
            bdvst_date=None,
            bdvst_time=None,
        )
        with pytest.raises(UnrecoverableAnchorError):
            build_audit_orders([record], config)

    def test_partial_fallback_bdvst_date_missing_raises(
        self, config: AuditOrdersConfig
    ) -> None:
        # Codex review feedback (NEEDS-CHANGES): a partial fallback pair
        # (date xor time) is just as unrecoverable as a fully missing
        # pair — there is no synthetic value the pipeline could invent
        # without violating the strict-loud parser contract. The REQ
        # pair is also nulled out so the resolver is forced onto the
        # fallback path.
        record = _input(
            req_date=None,
            req_time=None,
            bdvst_date=None,  # date missing
            bdvst_time=ParsedTimeOfDay(hour=10, minute=0, second=0),
        )
        with pytest.raises(UnrecoverableAnchorError):
            build_audit_orders([record], config)

    def test_partial_fallback_bdvst_time_missing_raises(
        self, config: AuditOrdersConfig
    ) -> None:
        # Mirror of the date-missing test: a fallback pair whose time
        # the strict parser refused (returned None) cannot be promoted
        # to the anchor.
        record = _input(
            req_date=None,
            req_time=None,
            bdvst_date=date(2026, 5, 1),
            bdvst_time=None,  # time unparseable upstream
        )
        with pytest.raises(UnrecoverableAnchorError):
            build_audit_orders([record], config)


# =============================================================================
# AC: Output schema matches PRD §"Output schema" identity + anchor fields
# =============================================================================


class TestOutputSchemaIdentityAndAnchor:
    """The :class:`AuditOrder` shape pins the PRD §Output schema contract.

    Identity (audit_id / hn / an / reqno) + Anchor (order_datetime,
    anchor_imputed, products_ordered) are the audit_orders-side of the
    larger PRD output schema; downstream stages add hb/vitals/cohort/etc.
    A field rename here is a breaking change for #5–#8.
    """

    def test_audit_order_has_required_fields(self, config: AuditOrdersConfig) -> None:
        result = build_audit_orders([_input()], config)
        order = result.included[0]
        # Identity
        assert hasattr(order, "audit_id")
        assert hasattr(order, "hn")
        assert hasattr(order, "an")
        assert hasattr(order, "reqno")
        # Anchor
        assert hasattr(order, "order_datetime")
        assert hasattr(order, "anchor_imputed")
        assert hasattr(order, "products_ordered")
        # Joined inputs needed by #5–#7
        assert hasattr(order, "diagnosis_codes")

    def test_order_datetime_is_tz_aware_utc(self, config: AuditOrdersConfig) -> None:
        result = build_audit_orders([_input()], config)
        dt = result.included[0].order_datetime
        # tzinfo present; offset is exactly UTC (not naive, not Bangkok)
        assert dt.tzinfo is not None
        assert dt.utcoffset() is not None
        assert dt.utcoffset().total_seconds() == 0

    def test_audit_order_is_frozen(self, config: AuditOrdersConfig) -> None:
        # Pydantic v2 ConfigDict(frozen=True) raises ValidationError on
        # post-construction mutation. Catching only the specific exception
        # type ensures this test fails-capable: a no-op setter (or a
        # different exception caused by an unrelated bug) would no longer
        # satisfy it.
        from pydantic import ValidationError

        result = build_audit_orders([_input()], config)
        order = result.included[0]
        with pytest.raises(ValidationError):
            order.audit_id = "tampered"  # type: ignore[misc]


class TestAuditIdDeterminism:
    """``audit_id`` is stable across re-runs given same ``(hn, reqno)``.

    Idempotency in #9 / #19 / #24 depends on this. The hashing formula
    is implementation, not contract — what is asserted: same input →
    same id; different inputs → different ids.
    """

    def test_same_hn_reqno_same_audit_id(self) -> None:
        assert build_audit_id("HN-001", "REQ-001") == build_audit_id(
            "HN-001", "REQ-001"
        )

    def test_different_hn_different_id(self) -> None:
        assert build_audit_id("HN-001", "REQ-001") != build_audit_id(
            "HN-002", "REQ-001"
        )

    def test_different_reqno_different_id(self) -> None:
        assert build_audit_id("HN-001", "REQ-001") != build_audit_id(
            "HN-001", "REQ-002"
        )

    def test_audit_id_is_filesystem_safe(self) -> None:
        # Downstream :class:`bba.audit_store.models.SafeId` enforces the
        # ``[A-Za-z0-9._-]+`` allow-list. audit_orders must produce ids
        # that don't trip that validator at the persistence boundary.
        import re

        aid = build_audit_id("HN-001", "REQ-001")
        assert aid
        assert re.fullmatch(r"[A-Za-z0-9._-]+", aid), (
            f"audit_id {aid!r} contains characters outside the SafeId allow-list"
        )

    def test_pipeline_uses_build_audit_id(self, config: AuditOrdersConfig) -> None:
        # The pipeline-emitted id for (hn, reqno) must equal the
        # standalone formula's output. This is the contract that lets
        # #9 re-derive the id without round-tripping the canonical table.
        rec = _input(hn="HN-XYZ", reqno="REQ-XYZ")
        result = build_audit_orders([rec], config)
        assert len(result.included) == 1
        assert result.included[0].audit_id == build_audit_id("HN-XYZ", "REQ-XYZ")


# =============================================================================
# Partition invariant — every input lands in exactly one bucket
# =============================================================================


class TestPartitionInvariant:
    """No record is silently dropped or double-counted.

    This is the foundational contract. If a future refactor flips the
    cancelled-check before the products-check, the *reasons* may shift
    but the partition must still cover every input exactly once.
    """

    def test_empty_input_empty_output(self, config: AuditOrdersConfig) -> None:
        result = build_audit_orders([], config)
        assert result.included == ()
        assert result.excluded == ()

    def test_total_coverage_on_mixed_batch(self, config: AuditOrdersConfig) -> None:
        records = [
            _input(reqno="REQ-A"),  # clean
            _input(reqno="REQ-B", products=("FFP",)),  # not_rbc_product
            _input(reqno="REQ-C", an=None),  # no_an
            _input(reqno="REQ-D", bdvstst="6"),  # status_not_eligible
            _input(reqno="REQ-E", canceldate="20260501"),  # cancelled
            _input(reqno="REQ-F", reqtype="H"),  # inter_hospital
            _input(reqno="REQ-H", diagnosis_codes=("D56.1",)),  # hemoglobinopathy
        ]
        result = build_audit_orders(records, config)
        seen = {o.reqno for o in result.included} | {e.reqno for e in result.excluded}
        assert seen == {r.reqno for r in records}, (
            f"input reqnos {sorted(r.reqno for r in records)} but "
            f"partition covered {sorted(seen)}"
        )
        assert len(result.included) + len(result.excluded) == len(records), (
            "partition double-counted or dropped a record"
        )

    def test_input_ordering_preserved_within_buckets(
        self, config: AuditOrdersConfig
    ) -> None:
        # Stable ordering matters for downstream stages that materialize
        # the table to Parquet (#19) — re-running with the same input
        # ordering must yield the same row ordering.
        records = [
            _input(reqno="REQ-A"),
            _input(reqno="REQ-B", products=("FFP",)),
            _input(reqno="REQ-C"),
        ]
        result = build_audit_orders(records, config)
        assert [o.reqno for o in result.included] == ["REQ-A", "REQ-C"]
        assert [e.reqno for e in result.excluded] == ["REQ-B"]


# =============================================================================
# Property tests (hypothesis) — deep-module invariants
# =============================================================================


class TestPropertyPartitionAndIdentity:
    """Hypothesis-driven property tests for the foundational invariants."""

    @given(
        hns=st.lists(
            st.text(
                alphabet=st.characters(min_codepoint=ord("A"), max_codepoint=ord("Z")),
                min_size=2,
                max_size=4,
            ),
            min_size=1,
            max_size=10,
            unique=True,
        ),
        reqnos=st.lists(
            st.text(
                alphabet=st.characters(min_codepoint=ord("A"), max_codepoint=ord("Z")),
                min_size=2,
                max_size=4,
            ),
            min_size=1,
            max_size=10,
            unique=True,
        ),
    )
    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_audit_id_pairwise_distinct(
        self, hns: list[str], reqnos: list[str]
    ) -> None:
        # Distinct (hn, reqno) pairs produce distinct audit_ids.
        pairs = [(h, r) for h in hns for r in reqnos]
        ids = {build_audit_id(h, r) for h, r in pairs}
        assert len(ids) == len(pairs), (
            "audit_id collided for distinct (hn, reqno) pairs — "
            "downstream idempotency would mistakenly merge two audits"
        )

    @given(st.text(min_size=1, max_size=12), st.text(min_size=1, max_size=12))
    @settings(max_examples=100)
    def test_audit_id_deterministic_per_pair(self, hn: str, reqno: str) -> None:
        # Same input → same output across calls.
        assert build_audit_id(hn, reqno) == build_audit_id(hn, reqno)

    @given(
        records_spec=st.lists(
            st.fixed_dictionaries(
                {
                    # bdvstst sampled to cover both eligible {4, 5} and
                    # ineligible status codes — exercises the status gate.
                    "bdvstst": st.sampled_from(["1", "3", "4", "5", "6", "7"]),
                    # reqtype covers in-house P + inter-hospital H + an
                    # invalid sentinel to confirm the gate generalizes.
                    "reqtype": st.sampled_from(["P", "H", "X"]),
                    # Products mix RBC and non-RBC families.
                    "product": st.sampled_from(
                        ["LPRC", "LDPRC", "SDR", "FFP", "PLT", "CRYO"]
                    ),
                    # canceldate: None, empty (missing), or real timestamp.
                    "canceldate": st.sampled_from([None, "", "20260501"]),
                    # an: None and empty both mean OPD.
                    "an": st.sampled_from([None, "", "AN-001"]),
                    # diagnosis covers a mix of clean + exclusion codes.
                    "diag": st.sampled_from(
                        [
                            "I50.9",  # clean
                            "D55.0",  # hemoglobinopathy
                            "D59.3",  # AIHA — now in-scope, passes filter
                            "M31.1",  # TMA — now in-scope, passes filter
                            "O09.9",  # obstetric — now in-scope, passes filter
                            "D550",  # malformed near-miss (must NOT match D55)
                        ]
                    ),
                }
            ),
            min_size=0,
            max_size=20,
        )
    )
    @settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_partition_invariant_holds_on_generated_records(
        self, records_spec: list[dict[str, object]]
    ) -> None:
        """For any combination of input dimensions, every record lands
        in exactly one bucket (included xor excluded) and the union
        covers the input set.

        Codex review (NEEDS-CHANGES): the prior property tests only
        exercised ``build_audit_id`` and never the pipeline's partition
        behavior. This test generates real ``BloodOrderInput`` records
        and asserts the foundational partition invariant across the
        full rule matrix.
        """
        config = AuditOrdersConfig(code_version="prop-test")
        inputs = [
            _input(
                reqno=f"REQ-{i:04d}",
                bdvstst=str(spec["bdvstst"]),
                reqtype=str(spec["reqtype"]),
                products=(str(spec["product"]),),
                canceldate=spec["canceldate"],  # type: ignore[arg-type]
                an=spec["an"],  # type: ignore[arg-type]
                diagnosis_codes=(str(spec["diag"]),),
            )
            for i, spec in enumerate(records_spec)
        ]
        result = build_audit_orders(inputs, config)
        in_reqnos = [o.reqno for o in result.included]
        ex_reqnos = [e.reqno for e in result.excluded]
        # Disjoint
        assert set(in_reqnos).isdisjoint(set(ex_reqnos)), (
            f"record landed in both buckets: {set(in_reqnos) & set(ex_reqnos)}"
        )
        # Total: input count == included + excluded
        assert len(in_reqnos) + len(ex_reqnos) == len(inputs), (
            f"partition lost {len(inputs) - len(in_reqnos) - len(ex_reqnos)} "
            f"records or double-counted"
        )
        # Coverage: every input reqno appears somewhere
        all_input_reqnos = {r.reqno for r in inputs}
        assert set(in_reqnos) | set(ex_reqnos) == all_input_reqnos, (
            "partition missed some input reqnos"
        )

    @given(
        records_spec=st.lists(
            st.fixed_dictionaries(
                {
                    "bdvstst": st.sampled_from(["4", "5"]),
                    "product": st.sampled_from(["LPRC", "LDPRC", "SDR"]),
                    "diag": st.sampled_from(["I50.9", "K92.2"]),  # clean dx
                }
            ),
            min_size=1,
            max_size=8,
            unique_by=lambda d: tuple(sorted(d.items())),
        )
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_audit_ids_unique_within_included_set(
        self, records_spec: list[dict[str, object]]
    ) -> None:
        """audit_id uniqueness in the included set follows from the
        (hn, reqno) uniqueness contract of the canonical table (one
        row per (HN, REQNO)).

        Codex review: prior property tests asserted build_audit_id
        determinism in isolation; this asserts the property *through
        the pipeline* — i.e., that the pipeline never silently emits
        two AuditOrders sharing an audit_id, which would corrupt
        downstream idempotency.
        """
        config = AuditOrdersConfig(code_version="prop-test")
        inputs = [
            _input(
                hn=f"HN-{i:04d}",
                reqno=f"REQ-{i:04d}",
                bdvstst=str(spec["bdvstst"]),
                products=(str(spec["product"]),),
                diagnosis_codes=(str(spec["diag"]),),
            )
            for i, spec in enumerate(records_spec)
        ]
        result = build_audit_orders(inputs, config)
        ids = [o.audit_id for o in result.included]
        assert len(ids) == len(set(ids)), (
            "pipeline emitted duplicate audit_id for distinct (hn, reqno) pairs"
        )


# =============================================================================
# Adversarial fixtures — Round-1/Round-2 review concerns
# =============================================================================


class TestAdversarialIcdMatching:
    """ICD-10 matching must not be evadable by formatting drift."""

    def test_d55_with_long_subcategory(self, config: AuditOrdersConfig) -> None:
        # D55.999 is a malformed long code but still starts with D55 —
        # the hard-exclusion rule is "block hemoglobinopathy", not
        # "block only the published subcodes". A long-tail subcategory
        # must still trip the exclusion.
        result = build_audit_orders([_input(diagnosis_codes=("D55.999",))], config)
        assert any(r.reason == "hemoglobinopathy" for r in result.excluded), (
            "D55.999 should hit the hemoglobinopathy block"
        )

    def test_dotless_subcategory_is_canonicalized_and_excluded(
        self, config: AuditOrdersConfig
    ) -> None:
        # Chulalongkorn exports store subcategory codes WITHOUT the dot:
        # "D550" is the dotless form of D55.0, and "D561" of D56.1
        # (beta-thalassemia) — both hemoglobinopathies. ICD-10 categories
        # are always 3 chars, so index-3 digits are the subcategory; the
        # matcher canonicalizes ("D550" -> "D55.0") and the hard-exclude
        # fires. Before this fix a dotless thalassemia code slipped past
        # the exclusion and got audited with the wrong RBC logic.
        for dotless in ("D550", "D561"):
            result = build_audit_orders(
                [_input(diagnosis_codes=(dotless,))], config
            )
            assert any(r.reason == "hemoglobinopathy" for r in result.excluded), (
                f"dotless {dotless} should hit the hemoglobinopathy exclusion"
            )

    def test_lowercase_d55_does_not_match(self, config: AuditOrdersConfig) -> None:
        # HOSxP ICD-10 codes are uppercase. A lowercase ``"d55.0"`` is
        # therefore data drift, not a real hemoglobinopathy. The matcher
        # is case-sensitive — silently upper-casing would also silently
        # upper-case future codes that happen to share a substring.
        result = build_audit_orders([_input(diagnosis_codes=("d55.0",))], config)
        assert all(r.reason != "hemoglobinopathy" for r in result.excluded), (
            "lowercase ICD-10 should not match the uppercase D55 prefix"
        )

    def test_first_diagnosis_code_match_wins_detail(
        self, config: AuditOrdersConfig
    ) -> None:
        # Multiple matching codes on the same admission — the ``detail``
        # field carries the code that fired the rule. Either the first
        # matched code wins (stable ordering) or any deterministic choice;
        # whatever it is, the reviewer must see a concrete code, not None.
        result = build_audit_orders(
            [
                _input(
                    diagnosis_codes=(
                        "I50.9",  # not excluded
                        "D55.0",  # hemoglobinopathy
                        "D56.1",  # also hemoglobinopathy
                    )
                )
            ],
            config,
        )
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == "hemoglobinopathy"
        assert result.excluded[0].detail is not None
        assert result.excluded[0].detail.startswith("D5")


class TestAdversarialAnchorTimes:
    """Anchor resolution must respect the strict-time-parser contract.

    Per PRD §1 (Round 2 fix E35), the strict parser refuses to silently
    shift. The audit_orders filter must not re-introduce a sentinel time
    when the parser already returned ``None``.
    """

    def test_unparseable_req_time_falls_through_to_bdvst(
        self, config: AuditOrdersConfig
    ) -> None:
        # Simulates: BDVST.REQTIME was "0" / "9999" / decimal hour — the
        # parser returned ParsedTimeOfDay=None. The fallback must fire
        # rather than silently anchoring at 00:00:00 of req_date.
        result = build_audit_orders([_input(req_time=None)], config)
        assert len(result.included) == 1
        assert result.included[0].anchor_imputed is True

    def test_unparseable_bdvst_time_with_clean_req_uses_primary(
        self, config: AuditOrdersConfig
    ) -> None:
        # The BDVST fallback's time being None must not affect the primary
        # path — REQ alone is sufficient.
        result = build_audit_orders([_input(bdvst_time=None)], config)
        assert len(result.included) == 1
        assert result.included[0].anchor_imputed is False

    def test_resolve_anchor_returns_none_when_both_pairs_dead(
        self,
    ) -> None:
        # Direct test of the helper — :func:`resolve_anchor` returns
        # ``AnchorResolution(anchor=None, imputed=False)`` and the
        # pipeline decides to raise. The two-step separation lets a future
        # caller (e.g., a dashboard "preview" mode) inspect rather than
        # explode.
        rec = _input(req_date=None, req_time=None, bdvst_date=None, bdvst_time=None)
        result: AnchorResolution = resolve_anchor(rec)
        assert result.anchor is None


# =============================================================================
# Per-rule predicate tests — exercise predicates outside the pipeline
# =============================================================================


class TestPerRulePredicates:
    """Each ``check_*`` returns an ExcludedRecord for the matching case and
    ``None`` for the clean case. Tests the per-rule contract directly so a
    regression in one rule fails its own test cleanly."""

    def test_check_rbc_product_clean(self) -> None:
        assert check_rbc_product(_input(products=("LPRC",))) is None

    def test_check_rbc_product_excludes_non_rbc(self) -> None:
        e = check_rbc_product(_input(products=("FFP",)))
        assert isinstance(e, ExcludedRecord)
        assert e.reason == "not_rbc_product"

    def test_check_status_clean(self) -> None:
        assert check_status(_input(bdvstst="4")) is None
        assert check_status(_input(bdvstst="5")) is None

    def test_check_status_excludes_refused(self) -> None:
        e = check_status(_input(bdvstst="6"))
        assert isinstance(e, ExcludedRecord)
        assert e.reason == "status_not_eligible"

    def test_check_cancelled_clean(self) -> None:
        assert check_cancelled(_input(canceldate=None)) is None

    def test_check_cancelled_excludes(self) -> None:
        e = check_cancelled(_input(canceldate="20260501"))
        assert isinstance(e, ExcludedRecord)
        assert e.reason == "cancelled"

    def test_check_an_scoped_clean(self) -> None:
        assert check_an_scoped(_input(an="AN-1")) is None

    def test_check_an_scoped_excludes_none(self) -> None:
        e = check_an_scoped(_input(an=None))
        assert isinstance(e, ExcludedRecord)
        assert e.reason == "no_an"

    def test_check_request_type_clean(self) -> None:
        assert check_request_type(_input(reqtype="P")) is None

    def test_check_request_type_excludes_h(self) -> None:
        e = check_request_type(_input(reqtype="H"))
        assert isinstance(e, ExcludedRecord)
        assert e.reason == "inter_hospital"

    @pytest.mark.parametrize("code", ["D55.0", "D56.0", "D57.0", "D58.0"])
    def test_check_hemoglobinopathy_excludes(self, code: str) -> None:
        e = check_hemoglobinopathy(_input(diagnosis_codes=(code,)))
        assert isinstance(e, ExcludedRecord)
        assert e.reason == "hemoglobinopathy"


class TestHelperFunctions:
    """Small helpers — these exist on the surface and tests pin them so
    they cannot be silently renamed or have their semantics changed."""

    @pytest.mark.parametrize("product", ["LPRC", "LDPRC", "SDR"])
    def test_is_rbc_product_true(self, product: str) -> None:
        assert is_rbc_product(product) is True

    @pytest.mark.parametrize("product", ["FFP", "PLT", "CRYO", "ALBUMIN", ""])
    def test_is_rbc_product_false(self, product: str) -> None:
        assert is_rbc_product(product) is False

    def test_rbc_products_in_filters_and_preserves_order(self) -> None:
        out = rbc_products_in(("FFP", "LPRC", "PLT", "SDR"))
        assert out == ("LPRC", "SDR")

    def test_rbc_products_in_empty(self) -> None:
        assert rbc_products_in(("FFP", "PLT")) == ()


# =============================================================================
# A1 — Platelet order admission + component tagging
# =============================================================================


class TestPlateletOrderAdmission:
    """Platelet-only orders are admitted as component="platelet"; RBC path unchanged.

    WHY: Phase 2 requires platelet appropriateness auditing. Without
    admitting platelet-only orders at intake, every platelet transfusion
    is silently discarded and never evaluated. Simultaneously, the RBC
    path must remain byte-identical — adding a platelet route must not
    change the outcome for any existing RBC order.
    """

    def test_platelet_only_order_admitted(self, config: AuditOrdersConfig) -> None:
        """A platelet-only order (products all in PLATELET_PRODUCTS) is included."""
        result = build_audit_orders([_input(products=("PC",))], config)
        assert len(result.included) == 1
        assert len(result.excluded) == 0

    def test_platelet_order_has_component_platelet(
        self, config: AuditOrdersConfig
    ) -> None:
        """Platelet orders carry component="platelet" so downstream dispatch can route
        them to the platelet auditor rather than the RBC auditor."""
        result = build_audit_orders([_input(products=("PC",))], config)
        assert result.included[0].component == "platelet"

    def test_platelet_order_products_preserved_in_products_ordered(
        self, config: AuditOrdersConfig
    ) -> None:
        """Platelet product codes are carried verbatim in products_ordered.

        Unlike RBC mixed orders where rbc_products_in strips non-RBC codes,
        platelet-only orders must retain their platelet codes — the downstream
        auditor needs them to determine which platelet product was issued.
        """
        result = build_audit_orders([_input(products=("LDPPC", "LDPPCI"))], config)
        order = result.included[0]
        assert "LDPPC" in order.products_ordered
        assert "LDPPCI" in order.products_ordered

    def test_irradiated_platelet_product_admitted(
        self, config: AuditOrdersConfig
    ) -> None:
        """Irradiated leukodepleted platelet (LDPPCI) is in PLATELET_PRODUCTS and
        must be admitted. Irradiation changes processing, not the clinical
        appropriateness obligation to audit the transfusion."""
        result = build_audit_orders([_input(products=("LDPPCI",))], config)
        assert len(result.included) == 1
        assert result.included[0].component == "platelet"

    def test_ffp_only_order_still_rejected(self, config: AuditOrdersConfig) -> None:
        """FFP is neither RBC nor platelet — must remain excluded with
        reason not_rbc_product. Only platelet codes widen the admission gate."""
        result = build_audit_orders([_input(products=("FFP",))], config)
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == "not_rbc_product"

    def test_cryo_only_order_still_rejected(self, config: AuditOrdersConfig) -> None:
        """Cryoprecipitate is out of audit scope (docs plan §6) and must remain
        excluded. Only platelets are added to the admission gate in Stage A."""
        result = build_audit_orders([_input(products=("CPP",))], config)
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == "not_rbc_product"

    def test_rbc_only_order_component_is_red_cell(
        self, config: AuditOrdersConfig
    ) -> None:
        """RBC-only orders carry component="red_cell" — byte-identical to
        pre-Phase-2 behavior. Adding the platelet path must not change the
        component on any order that only contains RBC products."""
        result = build_audit_orders([_input(products=("LPRC",))], config)
        assert result.included[0].component == "red_cell"

    def test_rbc_only_products_ordered_unchanged(
        self, config: AuditOrdersConfig
    ) -> None:
        """RBC-only products_ordered is byte-identical to pre-Phase-2. The tuple
        value must be exactly ("LPRC",), not wrapped or reordered."""
        result = build_audit_orders([_input(products=("LPRC",))], config)
        assert result.included[0].products_ordered == ("LPRC",)

    def test_mixed_rbc_platelet_order_admitted_as_red_cell(
        self, config: AuditOrdersConfig
    ) -> None:
        """A mixed RBC+platelet order is admitted as red_cell with the platelet
        code stripped — exactly as today. Mixed orders already pass check_rbc_product
        (they have at least one RBC product) and rbc_products_in drops non-RBC codes."""
        result = build_audit_orders([_input(products=("LPRC", "PC"))], config)
        assert result.included[0].component == "red_cell"
        assert result.included[0].products_ordered == ("LPRC",)

    def test_empty_products_still_rejected(self, config: AuditOrdersConfig) -> None:
        """An order with no products at all must be rejected. The vacuous-truth
        risk of `all(is_platelet_product(p) for p in ())` returning True on an
        empty tuple is a safety hazard — the empty case must not slip through."""
        result = build_audit_orders([_input(products=())], config)
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == "not_rbc_product"
