"""Live submission requires approval from the configured local redactor."""

from datetime import date
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bba.deid_redactor import (
    BackendRedactionResult,
    NoteInput,
    QuasiIdentifiers,
    RedactionRequest,
    RedactorVersion,
    redact_bundle,
)
from bba.deid_redactor.admission import (
    PrivacyAdmissionError,
    PrivacyAdmissionPolicy,
    attest_redaction,
)
from bba.llm_client.models import BatchSubmissionRequest
from bba.llm_client.transport import AnthropicBatchTransport
from bba.prompt_builder import EvidenceChunk, PromptBuildRequest

KEY = b"\x01" * 32
PUBLIC_KEY = Ed25519PrivateKey.from_private_bytes(KEY).public_key().public_bytes_raw()
VERSION = RedactorVersion(
    version="local-deid-1", model_sha="a" * 64, gazetteer_version="gazetteer-1"
)


class Backend:
    def redact(self, text: str) -> BackendRedactionResult:
        return BackendRedactionResult(text=text)


def admitted_request(*, k: int = 5, text: str = "Hb 8 g/dL"):
    result = redact_bundle(
        RedactionRequest(
            notes=(NoteInput(note_id="E1", text=text),),
            quasi_identifiers=QuasiIdentifiers(
                ward="W1",
                icd_3char="D62",
                age_band="60-69",
                sex="M",
                admission_month="2026-05",
            ),
            admission_date=date(2026, 5, 1),
            patient_age_years=65,
            redactor_version=VERSION,
        ),
        backend=Backend(),
        k_gate=lambda _: k,
    )
    prompt_request = PromptBuildRequest(
        task_mode="HB_7_10_REVIEW",
        cohort_threshold=7.0,
        evidence_chunks=(EvidenceChunk(evidence_id="E1", source="Lab", text=text),),
        few_shot_examples=(),
    )
    prompt, proof = attest_redaction(
        result,
        prompt_request,
        audit_id="a1",
        run_id="run1",
        key_id="local",
        signing_key=KEY,
    )
    return BatchSubmissionRequest(
        audit_id="a1",
        run_id="run1",
        task_mode=prompt.task_mode,
        prompt=prompt,
        privacy_attestation=proof,
    )


def policy():
    return PrivacyAdmissionPolicy(
        key_id="local", verification_key=PUBLIC_KEY, approved_redactor=VERSION
    )


def test_authenticated_exact_evidence_is_admitted():
    policy().verify(admitted_request())


def test_application_public_key_cannot_issue_redaction_approval():
    import hashlib
    import json

    request = admitted_request()
    changed_prompt = request.prompt.model_copy(
        update={
            "blocks": request.prompt.blocks[:-1]
            + (
                request.prompt.blocks[-1].model_copy(
                    update={"text": "Synthetic Alice Example HN 1234567"}
                ),
            )
        }
    )

    def canonical(value):
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()

    proof = request.privacy_attestation.model_copy(
        update={
            "prompt_digest": hashlib.sha256(
                canonical(changed_prompt.model_dump(mode="json"))
            ).hexdigest()
        }
    )
    attacker = Ed25519PrivateKey.from_private_bytes(PUBLIC_KEY)
    forged = proof.model_copy(
        update={
            "signature": attacker.sign(
                canonical(proof.model_dump(mode="json", exclude={"signature"}))
            ).hex()
        }
    )
    with pytest.raises(PrivacyAdmissionError, match="authentication"):
        policy().verify(
            request.model_copy(
                update={"prompt": changed_prompt, "privacy_attestation": forged}
            )
        )


def test_modified_evidence_and_self_asserted_proof_are_rejected():
    request = admitted_request()
    changed = request.model_copy(
        update={
            "prompt": request.prompt.model_copy(
                update={
                    "blocks": request.prompt.blocks[:-1]
                    + (
                        request.prompt.blocks[-1].model_copy(
                            update={"text": "Patient Alice Example HN 1234567"}
                        ),
                    )
                }
            )
        }
    )
    with pytest.raises(PrivacyAdmissionError, match="exact prompt"):
        policy().verify(changed)
    proof = request.privacy_attestation.model_copy(update={"signature": "0" * 128})
    with pytest.raises(PrivacyAdmissionError, match="authentication"):
        policy().verify(request.model_copy(update={"privacy_attestation": proof}))


@pytest.mark.parametrize("text,k", [("Hb 8", 4), ("[NURSE]" * 5, 5)])
def test_failed_privacy_checks_cannot_be_attested(text, k):
    with pytest.raises(PrivacyAdmissionError, match="privacy checks"):
        admitted_request(k=k, text=text)


def test_zero_model_hash_is_not_a_trusted_redactor():
    with pytest.raises(PrivacyAdmissionError, match="model SHA"):
        PrivacyAdmissionPolicy(
            key_id="local",
            verification_key=PUBLIC_KEY,
            approved_redactor=VERSION.model_copy(update={"model_sha": "0" * 64}),
        )


def test_live_transport_stops_before_sdk_on_missing_admission(
    tmp_path: Path, monkeypatch
):
    transport = AnthropicBatchTransport(
        api_key="synthetic-key", manifest_root=tmp_path, privacy_policy=policy()
    )
    monkeypatch.setattr(
        transport, "_client", lambda: pytest.fail("SDK must not be reached")
    )
    request = admitted_request().model_copy(update={"privacy_attestation": None})
    with pytest.raises(PrivacyAdmissionError, match="attestation"):
        transport.submit_batch_only(
            model="claude-sonnet-5-5", requests=[request], prompt_cache_enabled=False
        )


def test_attestation_cannot_move_to_another_run_or_unapproved_backend():
    request = admitted_request()
    with pytest.raises(PrivacyAdmissionError, match="exact prompt"):
        policy().verify(request.model_copy(update={"run_id": "another-run"}))
    unapproved = PrivacyAdmissionPolicy(
        key_id="local",
        verification_key=PUBLIC_KEY,
        approved_redactor=VERSION.model_copy(update={"model_sha": "b" * 64}),
    )
    with pytest.raises(PrivacyAdmissionError, match="not approved"):
        unapproved.verify(request)


def test_live_cli_without_authority_never_starts_a_process(tmp_path: Path, monkeypatch):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "pilot"))
    import _claude_cli_transport as cli

    monkeypatch.setattr(
        cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("no process")
    )
    transport = cli.ClaudeCliTransport(batch_root=tmp_path)
    with pytest.raises(PrivacyAdmissionError, match="trust root"):
        transport.submit_batch_only(
            model="claude-sonnet-5-5",
            requests=[admitted_request()],
            prompt_cache_enabled=False,
        )
    assert not list(tmp_path.iterdir())


def test_api_fetch_uses_durable_manifest_and_rejects_caller_drift(
    tmp_path: Path, monkeypatch
):
    from types import SimpleNamespace

    request = admitted_request()
    created = []
    batches = SimpleNamespace(
        create=lambda **kw: created.append(kw) or SimpleNamespace(id="remote-1"),
        retrieve=lambda _: SimpleNamespace(processing_status="ended"),
        results=lambda _: [],
    )
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    transport = AnthropicBatchTransport(
        api_key="synthetic", manifest_root=tmp_path, privacy_policy=policy()
    )

    def fake_client():
        assert list(tmp_path.glob("submission-*.json")), (
            "manifest must precede remote create"
        )
        return client

    monkeypatch.setattr(transport, "_client", fake_client)
    batch_id = transport.submit_batch_only(
        model="claude-sonnet-5-5", requests=[request], prompt_cache_enabled=False
    )
    restarted = AnthropicBatchTransport(
        api_key="synthetic", manifest_root=tmp_path, privacy_policy=policy()
    )
    monkeypatch.setattr(
        restarted, "_client", lambda: pytest.fail("drift must precede SDK")
    )
    with pytest.raises(ValueError, match="manifest drift"):
        restarted.fetch_batch_results(
            batch_id,
            model="claude-opus-5-5",
            requests=[request],
            prompt_cache_enabled=False,
        )
    assert len(created) == 1


def test_redaction_authority_verifies_actual_model_artifact(tmp_path: Path):
    import hashlib
    from bba.deid_redactor.admission import RedactionAuthority

    model_file = tmp_path / "approved-model.bin"
    model_file.write_bytes(b"synthetic local model artifact")
    with pytest.raises(PrivacyAdmissionError, match="artifact SHA mismatch"):
        RedactionAuthority(
            backend=Backend(),
            k_gate=lambda _: 5,
            model_artifact=model_file,
            version=VERSION,
            key_id="local",
            signing_key=KEY,
        )
    version = VERSION.model_copy(
        update={"model_sha": hashlib.sha256(model_file.read_bytes()).hexdigest()}
    )
    authority = RedactionAuthority(
        backend=Backend(),
        k_gate=lambda _: 5,
        model_artifact=model_file,
        version=version,
        key_id="local",
        signing_key=KEY,
    )
    artifact = authority.prepare(
        RedactionRequest(
            notes=(NoteInput(note_id="E1", text="Hb 8"),),
            quasi_identifiers=QuasiIdentifiers(
                ward="W1",
                icd_3char="D62",
                age_band="60-69",
                sex="M",
                admission_month="2026-05",
            ),
            admission_date=date(2026, 5, 1),
            patient_age_years=65,
            redactor_version=version,
        ),
        PromptBuildRequest(
            task_mode="HB_7_10_REVIEW",
            cohort_threshold=7.0,
            evidence_chunks=(
                EvidenceChunk(evidence_id="E1", source="Lab", text="Hb 8"),
            ),
            few_shot_examples=(),
        ),
        audit_id="a1",
        run_id="run1",
    )
    assert artifact.attestation.redactor_version == version
    assert artifact.evidence_chunks[0].text == "Hb 8"
