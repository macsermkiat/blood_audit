"""Recovery must preserve the same decisions and provenance as a live run."""

from __future__ import annotations

import copy

import pytest

from bba.audit_pipeline import (
    BatchRunState,
    InMemoryBatchRunStore,
    apply_batch_results,
    resume_on_startup,
    run_pipeline,
)
from bba.audit_pipeline.replay import _build_llm_call, default_verifier
from bba.audit_pipeline.resume import _result_from_cached_call
from bba.audit_store import AuditStore, AuditStoreConfig
from bba.audit_store.models import _deep_thaw
from bba.llm_client.models import RawBatchResponse
from test_audit_pipeline import (
    _batch_run,
    _cassette_for_contexts,
    _llm_config,
    _orphan_llm_call,
    _pipeline_config,
    _row_context,
    _RUN_TS,
)


def _store(path, version="v0.1.0+test"):
    return AuditStore(AuditStoreConfig(root_dir=path, code_version=version))


def _submitted(ctx, version="v0.1.0+test"):
    batches = InMemoryBatchRunStore()
    batches.create(
        _batch_run(
            batch_id="batch-recovery",
            state=BatchRunState.SUBMITTED,
            audit_ids=(ctx.order.audit_id,),
            run_id="run-integrity",
            code_version=version,
            anthropic_batch_id="remote",
            submitted_at=_RUN_TS,
        )
    )
    return batches


def test_default_verifier_rejects_fabricated_quote():
    ctx = _row_context(audit_id="audit-integrity")
    result = _result_from_cached_call(_orphan_llm_call(ctx, run_id="run-integrity"))
    payload = copy.deepcopy(_deep_thaw(result.raw_response_json))
    payload["content"][0]["input"]["indications"] = [
        {
            "code": "ACS",
            "quote": "Fabricated chest pain that does not occur in the note",
            "source_id": "E1",
            "confidence": 0.95,
        }
    ]
    assert not default_verifier(
        result.model_copy(update={"raw_response_json": payload}), ctx
    )


def test_recovery_preserves_verifier_attempt_order_and_original_calls(tmp_path):
    ctx = _row_context(audit_id="audit-4")
    seed = _result_from_cached_call(_orphan_llm_call(ctx, run_id="run-integrity"))
    attempts = []
    for index, label in enumerate(("INSUFFICIENT_EVIDENCE", "INAPPROPRIATE")):
        payload = copy.deepcopy(_deep_thaw(seed.raw_response_json))
        payload["content"][0]["input"].update(
            classification=label, reasoning_summary_en=f"attempt-{index}"
        )
        attempts.append(seed.model_copy(update={"raw_response_json": payload}))
    normal = _store(tmp_path / "normal")
    recovered = _store(tmp_path / "recovered")
    calls = tuple(
        _build_llm_call(result, attempt_index=i, run_id="run-integrity", context=ctx)
        for i, result in enumerate(attempts)
    )
    recovered._persist_llm_calls(calls)
    apply_batch_results(
        RawBatchResponse(batch_id="remote", results=tuple(attempts)),
        audit_store=normal,
        run_id="run-integrity",
        contexts={ctx.order.audit_id: ctx},
        verifier=lambda *_: False,
    )
    report = resume_on_startup(
        batch_run_store=_submitted(ctx),
        audit_store=recovered,
        contexts={ctx.order.audit_id: ctx},
        verifier=lambda *_: False,
    )
    assert report.completed_audit_ids == (ctx.order.audit_id,)
    assert recovered.read_audit_results() == normal.read_audit_results()
    assert recovered.read_llm_calls() == normal.read_llm_calls()
    assert {c.call_id for c in recovered.read_llm_calls()} == {c.call_id for c in calls}


def test_prior_code_version_cannot_complete_current_batch(tmp_path):
    ctx = _row_context(audit_id="audit-version")
    old = _store(tmp_path)
    result = _result_from_cached_call(_orphan_llm_call(ctx, run_id="run-integrity"))
    apply_batch_results(
        RawBatchResponse(batch_id="remote", results=(result,)),
        audit_store=old,
        run_id="run-integrity",
        contexts={ctx.order.audit_id: ctx},
        verifier=lambda *_: True,
    )
    current = _store(tmp_path, "v2")
    batches = _submitted(ctx, "v2")
    report = resume_on_startup(
        batch_run_store=batches, audit_store=current, contexts={ctx.order.audit_id: ctx}
    )
    assert report.completed_audit_ids == ()
    assert report.failed_audit_ids == (ctx.order.audit_id,)
    assert batches.get("batch-recovery").state != BatchRunState.COMPLETE


def test_completed_run_is_safe_to_repeat(tmp_path):
    ctx = _row_context(audit_id="audit-repeat")
    store = _store(tmp_path)
    batches = InMemoryBatchRunStore()
    options = dict(
        transport=_cassette_for_contexts((ctx,)),
        audit_store=store,
        batch_run_store=batches,
        llm_config=_llm_config(),
        pipeline_config=_pipeline_config(),
        run_id="run-integrity",
    )
    first = run_pipeline((ctx,), **options)
    second = run_pipeline((ctx,), **options)
    assert first.orphan_audit_ids == second.orphan_audit_ids == ()
    assert second.audit_ids_persisted == (ctx.order.audit_id,)
    assert len(store.read_audit_results()) == 1
    assert len(batches.list_all()) == 1


def test_existing_call_cannot_be_overwritten_with_different_response(tmp_path):
    ctx = _row_context(audit_id="audit-immutable")
    store = _store(tmp_path)
    call = _orphan_llm_call(ctx, run_id="run-integrity")
    store._persist_llm_calls((call,))
    changed = call.model_copy(update={"response_json": {"changed": True}})
    with pytest.raises(ValueError, match="immutable"):
        store._persist_llm_calls((changed,))
    assert store.read_llm_calls() == (call,)


@pytest.mark.parametrize("verdict", [True, False])
@pytest.mark.parametrize("global_defer", [False, True])
def test_crash_during_call_writes_recovers_complete_ordered_attempt_set(
    tmp_path, monkeypatch, verdict, global_defer
):
    ctx = _row_context(audit_id="audit-crash-sequence")
    seed = _result_from_cached_call(_orphan_llm_call(ctx, run_id="run-integrity"))
    attempts = []
    for index, label in enumerate(("INSUFFICIENT_EVIDENCE", "INAPPROPRIATE")):
        payload = _deep_thaw(seed.raw_response_json)
        payload["content"][0]["input"].update(
            classification=label, reasoning_summary_en=f"attempt-{index}"
        )
        attempts.append(seed.model_copy(update={"raw_response_json": payload}))
    response = RawBatchResponse(batch_id="remote", results=tuple(attempts))
    normal = _store(tmp_path / "normal")
    recovered = _store(tmp_path / "recovered")

    def verify(*_):
        return verdict

    apply_batch_results(
        response,
        audit_store=normal,
        run_id="run-integrity",
        contexts={ctx.order.audit_id: ctx},
        verifier=verify,
    )
    write_call = recovered._persist_call_record
    count = 0

    def crash_before_second_call(call):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("synthetic crash between call files")
        write_call(call)

    monkeypatch.setattr(recovered, "_persist_call_record", crash_before_second_call)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        apply_batch_results(
            response,
            audit_store=recovered,
            run_id="run-integrity",
            contexts={ctx.order.audit_id: ctx},
            verifier=verify,
        )
    assert len(recovered.read_llm_calls()) == 1
    monkeypatch.setattr(recovered, "_persist_call_record", write_call)
    batches = _submitted(ctx)
    report = resume_on_startup(
        batch_run_store=batches,
        audit_store=recovered,
        contexts={ctx.order.audit_id: ctx},
        verifier=verify,
        pipeline_config=_pipeline_config(enable_missing_platelet_defer=global_defer),
    )
    assert report.failed_audit_ids == ()
    assert recovered.read_audit_results() == normal.read_audit_results()
    assert recovered.read_llm_calls() == normal.read_llm_calls()
    before = {
        p.name: p.read_bytes()
        for p in (tmp_path / "recovered" / "llm_calls").glob("*.parquet")
    }
    assert (
        resume_on_startup(
            batch_run_store=batches,
            audit_store=recovered,
            contexts={ctx.order.audit_id: ctx},
            verifier=verify,
        ).completed_audit_ids
        == ()
    )
    assert before == {
        p.name: p.read_bytes()
        for p in (tmp_path / "recovered" / "llm_calls").glob("*.parquet")
    }


@pytest.mark.parametrize("global_defer", [False, True])
def test_poll_crash_reuses_original_requests_and_verifier(tmp_path, global_defer):
    ctx = _row_context(audit_id="audit-poll-crash")
    store = _store(tmp_path)
    batches = InMemoryBatchRunStore()
    cassette = _cassette_for_contexts((ctx,))

    class InterruptedTransport:
        submits = 0
        fetches = 0
        requests = None

        def submit_batch_only(self, **options):
            self.submits += 1
            self.requests = options["requests"]
            return cassette.submit_batch_only(**options)

        def fetch_batch_results(self, batch_id, **options):
            self.fetches += 1
            assert options["requests"] == self.requests
            if self.fetches == 1:
                raise RuntimeError("synthetic poll crash")
            return cassette.fetch_batch_results(batch_id, **options)

    transport = InterruptedTransport()
    options = dict(
        transport=transport,
        audit_store=store,
        batch_run_store=batches,
        llm_config=_llm_config(),
        pipeline_config=_pipeline_config(enable_missing_platelet_defer=global_defer),
        run_id="run-integrity",
        verifier=lambda *_: False,
    )
    with pytest.raises(RuntimeError, match="poll crash"):
        run_pipeline((ctx,), **options)
    assert batches.list_all()[0].state == BatchRunState.SUBMITTED
    outcome = run_pipeline((ctx,), **options)
    assert transport.submits == 1
    assert outcome.orphan_audit_ids == ()
    row = store.read_audit_results()[0]
    assert row.verifier_pass is False
    assert row.review_reason == "hallucination_suspect"
    assert batches.list_all()[0].state == BatchRunState.COMPLETE


def test_pending_crash_never_resubmits_uncertain_remote_creation(tmp_path):
    from bba.audit_pipeline import ResumeReconciliationError

    ctx = _row_context(audit_id="audit-create-crash")
    store = _store(tmp_path)
    batches = InMemoryBatchRunStore()

    class UncertainTransport:
        submits = 0

        def submit_batch_only(self, **options):
            self.submits += 1
            raise RuntimeError("remote may have accepted")

    transport = UncertainTransport()
    options = dict(
        transport=transport,
        audit_store=store,
        batch_run_store=batches,
        llm_config=_llm_config(),
        pipeline_config=_pipeline_config(),
        run_id="run-integrity",
    )
    with pytest.raises(RuntimeError, match="may have accepted"):
        run_pipeline((ctx,), **options)
    with pytest.raises(ResumeReconciliationError, match="uncertain"):
        run_pipeline((ctx,), **options)
    assert transport.submits == 1
    assert store.read_audit_results() == ()


def test_repeated_deterministic_run_has_no_orphans_and_rejects_changed_input(tmp_path):
    from bba.audit_pipeline import ResumeReconciliationError

    ctx = _row_context(audit_id="audit-deterministic", classification="APPROPRIATE")
    store = _store(tmp_path)
    options = dict(
        transport=_cassette_for_contexts((ctx,)),
        audit_store=store,
        batch_run_store=InMemoryBatchRunStore(),
        llm_config=_llm_config(),
        pipeline_config=_pipeline_config(),
        run_id="run-integrity",
    )
    run_pipeline((ctx,), **options)
    repeated = run_pipeline((ctx,), **options)
    assert repeated.orphan_audit_ids == ()
    assert repeated.audit_ids_persisted == (ctx.order.audit_id,)
    with pytest.raises(ResumeReconciliationError, match="context"):
        run_pipeline(
            (ctx.model_copy(update={"evidence_bundle_hash": "changed"}),), **options
        )


@pytest.mark.parametrize("component", ["red_cell", "platelet"])
def test_unused_redaction_claim_is_not_stamped_on_deterministic_rows(
    tmp_path, component
):
    ctx = _row_context(audit_id="audit-unused-proof", classification="APPROPRIATE")
    ctx = ctx.model_copy(update={"component": component})
    store = _store(tmp_path)
    run_pipeline(
        (ctx,),
        transport=_cassette_for_contexts((ctx,)),
        audit_store=store,
        batch_run_store=InMemoryBatchRunStore(),
        llm_config=_llm_config(),
        pipeline_config=_pipeline_config(),
        run_id="run-integrity",
    )
    (row,) = store.read_audit_results()
    assert row.model_id == "deterministic"
    assert (row.redactor_version, row.redactor_model_sha) == (
        "not-applied",
        "unavailable",
    )


def test_recovery_refuses_changed_evidence_and_unknown_legacy_attempt_order(tmp_path):
    ctx = _row_context(audit_id="audit-unproven")
    store = _store(tmp_path)
    call = _orphan_llm_call(ctx, run_id="run-integrity").model_copy(
        update={"attempt_index": None, "context_hash": None}
    )
    store._persist_llm_calls((call,))
    batches = _submitted(ctx)
    report = resume_on_startup(
        batch_run_store=batches, audit_store=store, contexts={ctx.order.audit_id: ctx}
    )
    assert report.failed_audit_ids == (ctx.order.audit_id,)
    assert store.read_audit_results() == ()


def test_live_configuration_never_overrides_test_guard():
    from bba.audit_pipeline import LiveAnthropicApiError, assert_test_safe_transport
    from bba.llm_client import AnthropicBatchTransport

    transport = object.__new__(AnthropicBatchTransport)
    with pytest.raises(LiveAnthropicApiError):
        assert_test_safe_transport(transport, allow_live_api=True)


def test_same_audit_and_run_under_new_version_gets_separate_batch_and_rows(tmp_path):
    ctx = _row_context(audit_id="audit-version-rerun")
    batches = InMemoryBatchRunStore()
    for version in ("v1", "v2"):
        store = _store(tmp_path, version)
        result = run_pipeline(
            (ctx,),
            transport=_cassette_for_contexts((ctx,)),
            audit_store=store,
            batch_run_store=batches,
            llm_config=_llm_config().model_copy(update={"code_version": version}),
            pipeline_config=_pipeline_config(code_version=version),
            run_id="run-integrity",
        )
        assert result.orphan_audit_ids == ()
        assert (
            len(store.read_audit_results(run_id="run-integrity", code_version=version))
            == 1
        )
    assert len(batches.list_all()) == 2
    assert len({r.batch_id for r in batches.list_all()}) == 2
    assert len(store.read_audit_results(run_id="run-integrity")) == 2


def test_default_verifier_accepts_grounded_quote_and_rejects_unknown_source():
    ctx = _row_context(audit_id="audit-grounded")
    result = _result_from_cached_call(_orphan_llm_call(ctx, run_id="run-integrity"))
    payload = _deep_thaw(result.raw_response_json)
    payload["content"][0]["input"]["indications"] = [
        {
            "code": "ACS",
            "quote": ctx.evidence_chunks[0].text,
            "source_id": ctx.evidence_chunks[0].evidence_id,
            "confidence": 0.95,
        }
    ]
    assert default_verifier(
        result.model_copy(update={"raw_response_json": payload}), ctx
    )
    payload["content"][0]["input"]["indications"][0]["source_id"] = "E999"
    assert not default_verifier(
        result.model_copy(update={"raw_response_json": payload}), ctx
    )


def test_default_response_headers_are_immutable(tmp_path):
    ctx = _row_context(audit_id="audit-headers")
    call = _orphan_llm_call(ctx, run_id="run-integrity")
    with pytest.raises(TypeError):
        call.response_headers["changed"] = "untrusted"


def test_mixed_remote_and_partial_cached_recovery_keeps_original_attempt_set(
    tmp_path, monkeypatch
):
    from bba.audit_pipeline.manifest import persist_manifest
    from bba.audit_pipeline.pipeline import (
        _build_submission_requests,
        _normalize_llm_contexts,
    )

    a, b = _normalize_llm_contexts(
        (
            _row_context(audit_id="audit-mixed-a"),
            _row_context(audit_id="audit-mixed-b"),
        ),
        run_id="run-integrity",
    )
    store = _store(tmp_path)
    seed = _result_from_cached_call(_orphan_llm_call(a, run_id="run-integrity"))
    calls = []
    for index, label in enumerate(("INSUFFICIENT_EVIDENCE", "INAPPROPRIATE")):
        payload = _deep_thaw(seed.raw_response_json)
        payload["content"][0]["input"].update(
            classification=label, reasoning_summary_en=f"original-{index}"
        )
        call = _build_llm_call(
            seed.model_copy(update={"raw_response_json": payload}),
            attempt_index=index,
            run_id="run-integrity",
            context=a,
        )
        calls.append(call.model_copy(update={"call_id": f"original-call-{index}"}))
    write_call = store._persist_call_record

    def crash(call):
        if call.attempt_index == 1:
            raise RuntimeError("synthetic partial call persistence")
        write_call(call)

    monkeypatch.setattr(store, "_persist_call_record", crash)
    with pytest.raises(RuntimeError, match="partial call"):
        store._persist_llm_calls(calls)
    monkeypatch.setattr(store, "_persist_call_record", write_call)
    batches = _submitted(a)
    run = batches.get("batch-recovery").model_copy(
        update={"audit_ids": (a.order.audit_id, b.order.audit_id)}
    )
    batches.update(run)
    persist_manifest(
        run=run,
        audit_store=store,
        contexts=(a, b),
        requests=_build_submission_requests((a, b), run_id="run-integrity"),
        llm_config=_llm_config(),
        enable_missing_platelet_defer=False,
    )
    report = resume_on_startup(
        batch_run_store=batches,
        audit_store=store,
        contexts={a.order.audit_id: a, b.order.audit_id: b},
        transport=_cassette_for_contexts((a, b)),
        llm_config=_llm_config(),
        verifier=lambda *_: True,
    )
    assert report.failed_audit_ids == ()
    rows = {row.audit_id: row for row in store.read_audit_results()}
    assert rows[a.order.audit_id].reasoning_summary_en == "original-1"
    assert tuple(
        sorted(
            (
                call
                for call in store.read_llm_calls()
                if call.audit_id == a.order.audit_id
            ),
            key=lambda c: c.attempt_index,
        )
    ) == tuple(calls)


def test_privacy_rejection_is_persisted_before_any_remote_or_batch_creation(tmp_path):
    from bba.deid_redactor.admission import PrivacyAdmissionError

    ctx = _row_context(audit_id="audit-private-review")
    store = _store(tmp_path)
    batches = InMemoryBatchRunStore()

    def refuse(request):
        raise PrivacyAdmissionError("missing local trust")

    outcome = run_pipeline(
        (ctx,),
        transport=object(),
        audit_store=store,
        batch_run_store=batches,
        llm_config=_llm_config(),
        pipeline_config=_pipeline_config(),
        run_id="run-integrity",
        admission_verifier=refuse,
    )
    assert outcome.orphan_audit_ids == ()
    assert batches.list_all() == ()
    row = store.read_audit_results()[0]
    assert row.final_classification == "NEEDS_REVIEW"
    assert row.review_reason == "privacy_admission_failed"
    assert row.model_id == "privacy-filter"


def test_privacy_verification_receives_exact_requests_before_offline_submission(
    tmp_path,
):
    ctx = _row_context(audit_id="audit-approved-review")
    approved = []
    store = _store(tmp_path)
    result = run_pipeline(
        (ctx,),
        transport=_cassette_for_contexts((ctx,)),
        audit_store=store,
        batch_run_store=InMemoryBatchRunStore(),
        llm_config=_llm_config(),
        pipeline_config=_pipeline_config(),
        run_id="run-integrity",
        admission_verifier=approved.append,
    )
    assert result.orphan_audit_ids == ()
    assert len(approved) == 1
    assert approved[0].audit_id == ctx.order.audit_id
    assert approved[0].prompt.blocks


def test_completed_deterministic_run_rejects_changed_platelet_policy_before_writes(
    tmp_path, monkeypatch
):
    from bba import feature_flags
    from bba.audit_pipeline import ResumeReconciliationError
    from test_platelet_dispatch import _platelet_ctx, _evidence_chunk

    monkeypatch.setattr(feature_flags, "PLATELET_LLM_ENABLED", True)
    ctx = _platelet_ctx(
        "audit-policy-drift", platelet_count=None, evidence_chunks=_evidence_chunk()
    )
    store = _store(tmp_path)
    options = dict(
        transport=object(),
        audit_store=store,
        batch_run_store=InMemoryBatchRunStore(),
        llm_config=_llm_config(),
        run_id="run-integrity",
    )
    run_pipeline(
        (ctx,),
        pipeline_config=_pipeline_config(enable_missing_platelet_defer=False),
        **options,
    )
    before = store.read_audit_results()
    assert before[0].final_classification == "INSUFFICIENT_EVIDENCE"
    with pytest.raises(ResumeReconciliationError, match="policy"):
        run_pipeline(
            (ctx,),
            pipeline_config=_pipeline_config(enable_missing_platelet_defer=True),
            **options,
        )
    assert store.read_audit_results() == before
    assert options["batch_run_store"].list_all() == ()


def test_pipeline_row_records_actual_built_prompt_hash_and_repeat_is_stable(tmp_path):
    from bba.audit_pipeline.pipeline import _build_submission_requests

    ctx = _row_context(audit_id="audit-prompt-provenance").model_copy(
        update={"prompt_hash": "not-submitted"}
    )
    request = _build_submission_requests((ctx,), run_id="run-integrity")[0]
    store = _store(tmp_path)
    options = dict(
        transport=_cassette_for_contexts((ctx,)),
        audit_store=store,
        batch_run_store=InMemoryBatchRunStore(),
        llm_config=_llm_config(),
        pipeline_config=_pipeline_config(),
        run_id="run-integrity",
    )
    run_pipeline((ctx,), **options)
    assert store.read_audit_results()[0].prompt_hash == request.prompt.prompt_hash
    assert run_pipeline((ctx,), **options).orphan_audit_ids == ()


def test_missing_evidence_with_admission_requirement_goes_to_local_review(tmp_path):
    from bba.deid_redactor.admission import PrivacyAdmissionError

    ctx = _row_context(audit_id="audit-missing-private-evidence").model_copy(
        update={"evidence_chunks": ()}
    )
    store = _store(tmp_path)
    batches = InMemoryBatchRunStore()

    def refuse(_):
        raise PrivacyAdmissionError("trusted local evidence missing")

    result = run_pipeline(
        (ctx,),
        transport=object(),
        audit_store=store,
        batch_run_store=batches,
        llm_config=_llm_config(),
        pipeline_config=_pipeline_config(),
        run_id="run-integrity",
        admission_verifier=refuse,
    )
    assert result.orphan_audit_ids == ()
    assert batches.list_all() == ()
    assert store.read_audit_results()[0].review_reason == "privacy_admission_failed"


def test_rbc_run_with_platelet_defer_recovers_crash_before_first_call_file(
    tmp_path, monkeypatch
):
    ctx = _row_context(audit_id="audit-rbc-global-defer-crash")
    store = _store(tmp_path)
    batches = InMemoryBatchRunStore()
    write_call = store._persist_call_record

    def crash(_):
        raise RuntimeError("synthetic crash before first call file")

    options = dict(
        transport=_cassette_for_contexts((ctx,)),
        audit_store=store,
        batch_run_store=batches,
        llm_config=_llm_config(),
        pipeline_config=_pipeline_config(enable_missing_platelet_defer=True),
        run_id="run-integrity",
    )
    monkeypatch.setattr(store, "_persist_call_record", crash)
    with pytest.raises(RuntimeError, match="first call file"):
        run_pipeline((ctx,), **options)
    assert store.read_llm_calls() == ()
    assert batches.list_all()[0].state == BatchRunState.SUBMITTED
    monkeypatch.setattr(store, "_persist_call_record", write_call)
    assert run_pipeline((ctx,), **options).orphan_audit_ids == ()
    assert len(store.read_audit_results()) == 1
    assert batches.list_all()[0].state == BatchRunState.COMPLETE
