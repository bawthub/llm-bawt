import json
import stat
from dataclasses import replace

from llm_bawt import inspection_attestation as attestation
from llm_bawt.task_turn_context import TaskTurnContext

ORIGIN = "https://tenant.example"
TURN = TaskTurnContext(
    session_id="f5d5175f-d118-4f42-8aef-db9c681ee036",
    turn_id="turn-" + "1" * 32,
    trigger_message_id="c4e9028f-8fb8-41b6-8fd6-d84c6b685159",
    bot_id="worker-1",
    user_id="owner-1",
    issued_at=1_700_000_000,
    inspection=("account-1", ORIGIN, 1_700_000_000),
)


def test_export_and_cross_language_vector(tmp_path, monkeypatch):
    path = tmp_path / "inspection" / "key"
    monkeypatch.setattr(attestation, "_key", None)
    monkeypatch.setenv("BAWTHUB_INSPECTION_KEY_EXPORT_FILE", str(path))
    monkeypatch.delenv("BAWTHUB_INSPECTION_KEY_EXPORT_UID", raising=False)
    monkeypatch.setenv("BAWTHUB_INSPECTION_ORIGIN", ORIGIN)
    monkeypatch.setattr(attestation.secrets, "token_bytes", lambda _: bytes([42]) * 32)
    monkeypatch.setattr(attestation.secrets, "token_hex", lambda _: "0123456789abcdef" * 2)
    monkeypatch.setattr(attestation.time, "time_ns", lambda: 1_700_000_000_000_000_000)

    assert attestation.sign_inspection_tasks(TURN, "phone") is None
    attestation.export_inspection_key()
    assert path.read_bytes() == b"2a" * 32 + b"\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    signed = attestation.sign_inspection_tasks(TURN, "phone")
    assert signed is not None
    body, mac = signed
    assert json.loads(body) == {
        "op": "observe_tasks", "aud": "bawthub-internal", "iat": 1_700_000_000_000,
        "exp": 1_700_000_015_000, "nonce": "0123456789abcdef" * 2, "origin": ORIGIN,
        "accountId": "account-1", "userId": "owner-1", "botId": "worker-1", "viewport": "phone",
    }
    # Also verified against the TypeScript verifier's raw-byte vector.
    assert mac == "eb63d9c154ec9bed2fe1345d1a2c5a1ac1e05c68ec6abb67948aef38fc0b5088"
    for wrong in (replace(TURN, inspection=None), replace(TURN, turn_id="not-an-ordinary-turn"),
                  replace(TURN, inspection=("account-1", "https://other.example", TURN.issued_at))):
        assert attestation.sign_inspection_tasks(wrong, "phone") is None
    assert attestation.sign_inspection_tasks(TURN, "any-size") is None
    queued = replace(TURN, turn_id="turn-delivery-" + "1" * 32)
    for age, allowed in ((301, False), (-1, False), (300, True), (0, True)):
        original_admission = replace(queued, inspection=("account-1", ORIGIN, TURN.issued_at - age))
        assert (attestation.sign_inspection_tasks(original_admission, "phone") is not None) == allowed

    monkeypatch.setenv("BAWTHUB_INSPECTION_KEY_EXPORT_UID", "invalid")
    attestation.export_inspection_key()
    assert attestation.sign_inspection_tasks(TURN, "phone") is None
    assert not path.exists()


def test_key_is_not_imported_from_env_or_file_by_standalone_process(monkeypatch):
    monkeypatch.setattr(attestation, "_key", None)
    monkeypatch.setenv("BAWTHUB_INSPECTION_ORIGIN", ORIGIN)
    monkeypatch.setenv("BAWTHUB_INSPECTION_KEY", "fake-not-a-config-value")
    monkeypatch.delenv("BAWTHUB_INSPECTION_KEY_EXPORT_FILE", raising=False)
    attestation.export_inspection_key()
    assert attestation.sign_inspection_tasks(TURN, "desktop") is None


def test_mcp_server_process_publishes_the_signing_key(tmp_path, monkeypatch):
    """The tool signs in the MCP server process, so that process must export."""
    from llm_bawt.mcp_server import server

    path = tmp_path / "inspection" / "key"
    monkeypatch.setattr(attestation, "_key", None)
    monkeypatch.setenv("BAWTHUB_INSPECTION_KEY_EXPORT_FILE", str(path))
    monkeypatch.delenv("BAWTHUB_INSPECTION_KEY_EXPORT_UID", raising=False)
    monkeypatch.setattr("sys.argv", ["llm-mcp-server"])
    monkeypatch.setattr("llm_bawt.api_auth.outbound.install", lambda: None)
    seen = []
    monkeypatch.setattr(server.mcp, "run", lambda **kw: seen.append(attestation.inspection_key_available()))
    server.run_server(transport="stdio")
    assert seen == [True]
    assert path.exists()
