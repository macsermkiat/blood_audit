"""Authenticated approval of exact prompts by an approved local redactor.

The signing key belongs to the approved redaction service. Callers that assemble
model requests receive an attestation, never that key. A digest or caller-supplied
privacy flag without this authentication cannot admit a live submission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from bba.deid_redactor.models import (
    K_ANONYMITY_MIN,
    KAnonymityGate,
    RedactionRequest,
    RedactionResult,
    RedactorBackend,
    RedactorVersion,
)
from bba.deid_redactor.semantic import detect_semantic_degradation
from bba.prompt_builder import (
    EvidenceChunk,
    PromptBuildRequest,
    PromptBuildResult,
    build_prompt,
)

if TYPE_CHECKING:
    from bba.llm_client.models import BatchSubmissionRequest


class PrivacyAdmissionError(ValueError):
    """The evidence lacks trusted approval for a live submission."""


class PrivacyAttestation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key_id: str = Field(min_length=1)
    audit_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    prompt_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    redaction_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    redactor_version: RedactorVersion
    k_anonymity_size: int = Field(ge=K_ANONYMITY_MIN)
    semantic_checks_passed: StrictBool
    signature: str = Field(pattern=r"^[a-f0-9]{128}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _prompt_digest(prompt: PromptBuildResult) -> str:
    return hashlib.sha256(_canonical(prompt.model_dump(mode="json"))).hexdigest()


def _real_version(version: RedactorVersion) -> None:
    sha = version.model_sha
    if (
        len(sha) != 64
        or any(c not in "0123456789abcdef" for c in sha)
        or set(sha) == {"0"}
    ):
        raise PrivacyAdmissionError(
            "approved redactor requires a real nonzero model SHA-256"
        )


def _key(key: bytes) -> None:
    if len(key) != 32:
        raise PrivacyAdmissionError("Ed25519 privacy key must contain exactly 32 bytes")


def attest_redaction(
    result: RedactionResult,
    prompt_request: PromptBuildRequest,
    *,
    audit_id: str,
    run_id: str,
    key_id: str,
    signing_key: bytes,
) -> tuple[PromptBuildResult, PrivacyAttestation]:
    """Run only inside the trusted redactor after its approved backend succeeds."""
    _key(signing_key)
    result = RedactionResult.model_validate(result.model_dump(mode="json"))
    _real_version(result.redactor_version)
    if (
        not result.k_anonymity_passed
        or result.k_anonymity_size < K_ANONYMITY_MIN
        or result.route_to_needs_review
        or result.needs_review_reasons
        or any(
            n.semantic_degraded or detect_semantic_degradation(n.redacted_text)
            for n in result.notes
        )
    ):
        raise PrivacyAdmissionError(
            "redaction privacy checks failed; local review is required"
        )
    expected = tuple((n.note_id, n.redacted_text) for n in result.notes)
    actual = tuple((c.evidence_id, c.text) for c in prompt_request.evidence_chunks)
    if (
        not expected
        or expected != actual
        or len({n[0] for n in expected}) != len(expected)
    ):
        raise PrivacyAdmissionError(
            "prompt evidence does not match exact redacted notes"
        )
    if prompt_request.few_shot_examples:
        raise PrivacyAdmissionError(
            "privacy authority requires separately approved examples"
        )
    prompt = build_prompt(prompt_request)
    if prompt.route_to_needs_review:
        raise PrivacyAdmissionError("prompt requires local review")
    payload = {
        "key_id": key_id,
        "audit_id": audit_id,
        "run_id": run_id,
        "prompt_digest": _prompt_digest(prompt),
        "redaction_hash": result.redaction_hash,
        "redactor_version": result.redactor_version.model_dump(mode="json"),
        "k_anonymity_size": result.k_anonymity_size,
        "semantic_checks_passed": True,
    }
    signature = (
        Ed25519PrivateKey.from_private_bytes(signing_key)
        .sign(_canonical(payload))
        .hex()
    )
    return prompt, PrivacyAttestation.model_validate(
        {**payload, "signature": signature}
    )


class PrivacyAdmissionPolicy:
    def __init__(
        self,
        *,
        key_id: str,
        verification_key: bytes,
        approved_redactor: RedactorVersion,
    ) -> None:
        _key(verification_key)
        _real_version(approved_redactor)
        self.key_id = key_id
        self._public_key = Ed25519PublicKey.from_public_bytes(verification_key)
        self.approved_redactor = approved_redactor

    @classmethod
    def from_file(cls, path: Path) -> PrivacyAdmissionPolicy:
        """Load a trusted policy containing only the issuer's public key."""
        data = json.loads(path.read_text(encoding="utf-8"))
        key_path = Path(data["public_key_file"])
        if not key_path.is_absolute():
            key_path = path.parent / key_path
        if path.stat().st_mode & 0o022 or key_path.stat().st_mode & 0o022:
            raise PrivacyAdmissionError(
                "privacy trust configuration must not be writable by others"
            )
        return cls(
            key_id=data["key_id"],
            verification_key=key_path.read_bytes(),
            approved_redactor=RedactorVersion.model_validate(data["approved_redactor"]),
        )

    def verify(self, request: BatchSubmissionRequest) -> None:
        proof = request.privacy_attestation
        if proof is None:
            raise PrivacyAdmissionError("trusted redaction attestation is required")
        proof = PrivacyAttestation.model_validate(proof.model_dump(mode="json"))
        payload = proof.model_dump(mode="json", exclude={"signature"})
        try:
            self._public_key.verify(bytes.fromhex(proof.signature), _canonical(payload))
        except InvalidSignature as exc:
            raise PrivacyAdmissionError(
                "redaction attestation authentication failed"
            ) from exc
        if proof.key_id != self.key_id:
            raise PrivacyAdmissionError("redaction attestation authentication failed")
        if proof.redactor_version != self.approved_redactor:
            raise PrivacyAdmissionError(
                "redactor version is not approved by this trust root"
            )
        if (
            proof.audit_id != request.audit_id
            or proof.run_id != request.run_id
            or proof.prompt_digest != _prompt_digest(request.prompt)
            or request.task_mode != request.prompt.task_mode
        ):
            raise PrivacyAdmissionError(
                "attestation does not bind this exact prompt and row"
            )
        if not proof.semantic_checks_passed or request.prompt.route_to_needs_review:
            raise PrivacyAdmissionError("privacy checks require local review")


class PrivacyArtifact(BaseModel):
    """Redacted evidence and approval exported by the trusted redaction service."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_chunks: tuple[EvidenceChunk, ...]
    attestation: PrivacyAttestation


def load_privacy_artifacts(path: Path) -> dict[str, PrivacyArtifact]:
    """Read an artifact map keyed by audit ID; authenticity is checked at admission."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PrivacyAdmissionError("privacy artifacts must be an audit-ID object")
    return {
        audit_id: PrivacyArtifact.model_validate(value)
        for audit_id, value in data.items()
    }


class RedactionAuthority:
    """Issuer-side composition for an operator-approved backend and model artifact.

    The operator loads the backend from this model artifact in the isolated local
    redaction service. The request-producing application receives only exported
    PrivacyArtifacts. It must not hold the signing key or construct this service.
    """

    def __init__(
        self,
        *,
        backend: RedactorBackend,
        k_gate: KAnonymityGate,
        model_artifact: Path,
        version: RedactorVersion,
        key_id: str,
        signing_key: bytes,
    ) -> None:
        _real_version(version)
        _key(signing_key)
        with model_artifact.open("rb") as stream:
            actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_sha != version.model_sha:
            raise PrivacyAdmissionError(
                "configured redactor model artifact SHA mismatch"
            )
        self._backend = backend
        self._k_gate = k_gate
        self._version = version
        self._key_id = key_id
        self._signing_key = signing_key

    def prepare(
        self,
        request: RedactionRequest,
        prompt_request: PromptBuildRequest,
        *,
        audit_id: str,
        run_id: str,
    ) -> PrivacyArtifact:
        from bba.deid_redactor.redactor import redact_bundle

        if request.redactor_version != self._version:
            raise PrivacyAdmissionError(
                "redaction request differs from approved model version"
            )
        result = redact_bundle(request, backend=self._backend, k_gate=self._k_gate)
        notes = {note.note_id: note.redacted_text for note in result.notes}
        ids = tuple(chunk.evidence_id for chunk in prompt_request.evidence_chunks)
        if ids != tuple(note.note_id for note in result.notes):
            raise PrivacyAdmissionError("prompt membership differs from redacted notes")
        chunks = tuple(
            chunk.model_copy(update={"text": notes[chunk.evidence_id]})
            for chunk in prompt_request.evidence_chunks
        )
        _, proof = attest_redaction(
            result,
            prompt_request.model_copy(update={"evidence_chunks": chunks}),
            audit_id=audit_id,
            run_id=run_id,
            key_id=self._key_id,
            signing_key=self._signing_key,
        )
        return PrivacyArtifact(evidence_chunks=chunks, attestation=proof)
