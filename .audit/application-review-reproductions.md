# Synthetic review reproductions

Run these commands from `/Users/admin/Project_Chatbot_research/blood_audit` at the reviewed revision. They use existing synthetic fixtures and temporary stores. They do not contact a model provider or read patient data. They demonstrate defects rather than assert desired behavior.

The pipeline probe uses `PipelineRowContext` values, ordered `BatchSubmissionResult` attempts, and a submitted `BatchRun`. It compares the actual persisted `AuditRow` after uninterrupted execution and recovery.

```sh
.venv/bin/python - <<'PY'
import copy
import runpy
import tempfile
from pathlib import Path

from bba.audit_pipeline import (
    BatchRunState, InMemoryBatchRunStore, apply_batch_results,
    resume_on_startup, run_pipeline,
)
from bba.audit_pipeline.replay import _build_llm_call
from bba.audit_pipeline.resume import _result_from_cached_call
from bba.audit_store import AuditStore, AuditStoreConfig
from bba.llm_client import AnthropicBatchTransport
from bba.llm_client.models import RawBatchResponse
from bba.report_generator import build_report_inputs

m = runpy.run_path('tests/unit/test_audit_pipeline.py')
root = Path(tempfile.mkdtemp(prefix='bba-review-'))

def store(name, version='v0.1.0+test'):
    return AuditStore(AuditStoreConfig(root_dir=root / name, code_version=version))

def submitted(audit_id, run_id, version):
    batches = InMemoryBatchRunStore()
    batches.create(m['_batch_run'](
        batch_id='batch-recovery', state=BatchRunState.SUBMITTED,
        audit_ids=(audit_id,), run_id=run_id, code_version=version,
        anthropic_batch_id='remote', submitted_at=m['_RUN_TS'],
    ))
    return batches

a = m['_row_context'](audit_id='audit-a', classification='APPROPRIATE')
b = m['_row_context'](audit_id='audit-b')
a, b = tuple(c.model_copy(update={
    'redactor_model_sha': 'same-sha', 'prompt_hash': 'same-prompt',
}) for c in (a, b))
s = store('mixed')
batches = InMemoryBatchRunStore()
options = dict(
    transport=m['_cassette_for_contexts']((a, b)), audit_store=s,
    batch_run_store=batches, llm_config=m['_llm_config'](),
    pipeline_config=m['_pipeline_config'](), run_id='run-review',
)
run_pipeline((a, b), **options)
try:
    build_report_inputs(
        run_id='run-review', audit_store=s, output_dir=root / 'report',
        ward_resolver=lambda _: 'ward', physician_resolver=lambda _: 'physician',
    )
except Exception as exc:
    print('MIXED_REPORT', type(exc).__name__, str(exc))

v2 = AuditStore(AuditStoreConfig(root_dir=s.config.root_dir, code_version='v2'))
v2_batches = submitted('audit-b', 'run-review', 'v2')
print('VERSION_RECOVERY', resume_on_startup(
    batch_run_store=v2_batches, audit_store=v2, contexts={'audit-b': b},
).model_dump())
print('VERSION_STATE', v2_batches.get('batch-recovery').state)
print('VERSION_ROWS', len(v2.read_audit_results('run-review', code_version='v2')))

def compare(audit_id, run_id, labels, reject):
    context = m['_row_context'](audit_id=audit_id)
    seed = m['_orphan_llm_call'](context, run_id=run_id)
    attempts = []
    for index, label in enumerate(labels):
        payload = copy.deepcopy(seed.model_dump(mode='json')['response_json'])
        payload['content'][0]['input'].update(
            classification=label, reasoning_summary_en=f'attempt-{index}',
        )
        attempts.append(_result_from_cached_call(
            seed.model_copy(update={'response_json': payload}),
        ))
    normal = store(audit_id + '-normal', 'v1')
    apply_batch_results(
        RawBatchResponse(batch_id='remote', results=tuple(attempts)),
        audit_store=normal, run_id=run_id, contexts={audit_id: context},
        **({'verifier': lambda *_: False} if reject else {}),
    )
    recovered = store(audit_id + '-recovered', 'v1')
    recovered._persist_llm_calls(tuple(
        _build_llm_call(result, attempt_index=index, run_id=run_id)
        for index, result in enumerate(attempts)
    ))
    resume_on_startup(
        batch_run_store=submitted(audit_id, run_id, 'v1'),
        audit_store=recovered, contexts={audit_id: context},
    )
    for name, target in (('NORMAL', normal), ('RECOVERED', recovered)):
        print(audit_id, name, [
            (r.final_classification, r.verifier_pass,
             r.review_reason, r.reasoning_summary_en)
            for r in target.read_audit_results()
        ])

compare('audit-recovery', 'run-recovery', ('INAPPROPRIATE',), True)
compare('audit-4', 'run-order', ('INSUFFICIENT_EVIDENCE', 'INAPPROPRIATE'), False)

try:
    run_pipeline((b,), **{
        **options, 'transport': object.__new__(AnthropicBatchTransport),
        'run_id': 'run-live',
    })
except Exception as exc:
    print('PRODUCTION_TRANSPORT', type(exc).__name__, str(exc))
try:
    run_pipeline((a, b), **options)
except Exception as exc:
    print('RERUN_BATCH', type(exc).__name__, str(exc))
print('RERUN_DETERMINISTIC', run_pipeline((a,), **options).model_dump())
PY
```

Observed results included `MixedRunMetadataError`, a completed v2 batch with zero v2 rows, a rejected verification becoming accepted after recovery, a different winning attempt after recovery, `LiveAnthropicApiError`, and a duplicate batch ID on rerun.

The privacy probe renders a synthetic progress note into the actual transport payload. The cache probe uses a fake CLI runner and changes both the model and evidence for the same batch member.

```sh
.venv/bin/python - <<'PY'
import json
import runpy
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path('scripts/pilot').resolve()))
from bba.prompt_builder import EvidenceChunk, PromptBuildRequest, build_prompt
from bba.llm_client import BatchSubmissionRequest
from bba.llm_client.transport import build_anthropic_request

m = runpy.run_path('scripts/pilot/run_llm_leg.py')
text = m['_render_payload']('IPDADMPROGRESS', {'sections': ({
    'label': 'S',
    'text': 'Synthetic patient Alice Example HN 1234567, visit 2026-10-18.',
},)})
prompt = build_prompt(PromptBuildRequest(
    task_mode='HB_7_10_REVIEW', cohort_threshold=7.0,
    evidence_chunks=(EvidenceChunk(
        evidence_id='E1', source='IPDADMPROGRESS', text=text,
    ),), few_shot_examples=(),
))
request = BatchSubmissionRequest(
    audit_id='synthetic-case', run_id='synthetic-run',
    task_mode='HB_7_10_REVIEW', prompt=prompt,
)
payload = json.dumps(build_anthropic_request(
    request, model='synthetic-model', prompt_cache_enabled=False,
))
print('PRIVACY_PAYLOAD', {
    'name_retained': 'Alice Example' in payload,
    'hn_retained': '1234567' in payload,
    'date_retained': '2026-10-18' in payload,
    'route_to_review': prompt.route_to_needs_review,
})

t = runpy.run_path('tests/unit/test_pilot_claude_cli_transport.py')
calls = []
def runner(argv, stdin):
    calls.append((argv, stdin))
    return t['_cli_success']({
        'classification': 'INAPPROPRIATE', 'reasoning': 'Synthetic response',
    })
transport = t['ClaudeCliTransport'](
    batch_root=Path(tempfile.mkdtemp(prefix='bba-cache-review-')),
    workers=1, runner=runner,
)
first = t['_request']('synthetic-cache')
batch = transport.submit_batch_only(
    model='claude-sonnet-5-5', requests=[first], prompt_cache_enabled=False,
)
transport.fetch_batch_results(
    batch, model='claude-sonnet-5-5', requests=[first],
    prompt_cache_enabled=False,
)
blocks = first.prompt.blocks[:-1] + (first.prompt.blocks[-1].model_copy(
    update={'text': 'Changed synthetic evidence version two'},
),)
changed = first.model_copy(update={'prompt': first.prompt.model_copy(
    update={'blocks': blocks, 'prompt_hash': 'changed-prompt'},
)})
result = transport.fetch_batch_results(
    batch, model='claude-opus-5-5', requests=[changed],
    prompt_cache_enabled=False,
).results[0]
print('CACHE_RESUME', {
    'runner_calls': len(calls), 'reported_model': result.model_id,
    'stored_request_model': result.request_json['model'],
    'contains_changed_evidence':
        'Changed synthetic evidence version two' in str(result.request_json),
})
PY
```

Observed privacy fields were all retained while `route_to_review` was false. The cache probe called the runner once, reported Opus, retained a Sonnet request, and omitted the changed evidence.

The offline source test command and its exclusions are recorded in [the evidence log](/Users/admin/Project_Chatbot_research/blood_audit/.audit/application-review.evidence.md).
