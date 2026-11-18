"""Rootless ops verifies scoped app attestations, not Fernet turn tokens."""
from __future__ import annotations

from copy import deepcopy

import pytest

from agent_bridge.mcp_call_context import canonical_invocation_hash
from llm_bawt.ops import invocation_attestation as attestation, secret_exports
from llm_bawt.service.providers import crypto

TOKEN = "synthetic-control-token"
ARGS = {"operation": "sample.restart", "args": {"target": "fixture"}}
CALLER = {
    "session_id": "00000000-0000-4000-8000-000000000001",
    "turn_id": "turn-" + "a" * 32,
    "trigger_message_id": "00000000-0000-4000-8000-000000000002",
    "bot_id": "fixture-bot",
    "user_id": "fixture-user",
    "issued_at": 1,
    "agent_request_id": "fixture-request",
    "session_key": "fixture:operator",
    "backend": "claude-code",
    "tool_use_id": "fixture-tool",
}


def _envelope(now=100):
    return attestation.mint(ARGS, CALLER, canonical_invocation_hash("ops_run", ARGS),
                            now=now, token_source=lambda: TOKEN)


@pytest.fixture
def rootless(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("ops must not access the app master key")

    # Only synthetic values, through the real scoped export reader.
    secret_exports.export_all(tmp_path, {
        name: (lambda: TOKEN) for name in secret_exports.FILES
    })
    monkeypatch.delenv("LLM_BAWT_SECRET_KEY", raising=False)
    monkeypatch.setenv(secret_exports.READ_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(crypto, "_get_fernet", forbidden)
    monkeypatch.setattr(crypto, "derive_key", forbidden)


def test_rootless_ops_verifies_with_only_actual_scoped_exports(rootless):
    result = attestation.verify(_envelope(), now=101)
    assert result.arguments == ARGS
    assert result.caller.model_dump() == CALLER
    assert result.request_id == attestation.verify(_envelope(now=110), now=111).request_id


@pytest.mark.parametrize("field", ["arguments", "caller", "invocation_hash"])
def test_modified_invocation_is_refused(rootless, field):
    envelope = deepcopy(_envelope())
    envelope["payload"][field] = {}
    with pytest.raises(attestation.AttestationError):
        attestation.verify(envelope, now=101)


@pytest.mark.parametrize("now", [94, 221])
def test_future_or_expired_attestation_is_refused(rootless, now):
    with pytest.raises(attestation.AttestationError):
        attestation.verify(_envelope(), now=now)


@pytest.mark.parametrize("field,value", [
    ("protocol", "another-purpose/v1"),
    ("expires_at", 1000),
    ("invocation_hash", "0" * 64),
    ("unexpected", "not allowed"),
])
def test_even_signed_invalid_payload_is_refused(rootless, field, value):
    envelope = _envelope()
    envelope["payload"][field] = value
    envelope["signature"] = attestation._mac(envelope["payload"], lambda: TOKEN)
    with pytest.raises(attestation.AttestationError):
        attestation.verify(envelope, now=101)


def test_missing_export_never_falls_back_to_master_key(rootless, monkeypatch, tmp_path):
    monkeypatch.setenv(secret_exports.READ_DIR_ENV, str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="not exported"):
        attestation.verify(_envelope(), now=101)


def test_oversized_envelope_is_refused_before_key_access():
    envelope = {"payload": "x" * attestation.MAX_BODY_BYTES, "signature": "unused"}
    with pytest.raises(attestation.AttestationError):
        attestation.verify(envelope, now=101, token_source=lambda: pytest.fail("key read"))
