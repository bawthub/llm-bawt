"""Read-only bot projection of the installation's host-access registry."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from cryptography.fernet import Fernet
import pytest
from sqlalchemy import create_engine, event, inspect
from sqlalchemy.pool import StaticPool

from llm_bawt import task_turn_context
from llm_bawt.mcp_server import host_access_tools as tool
from llm_bawt.mcp_server.task_association import (
    reset_current_task_turn_capability,
    set_current_task_turn_capability,
)
from llm_bawt.ops import host_access as ha


@pytest.fixture
def engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    yield engine
    engine.dispose()


def _host(engine, label="Build box"):
    return ha.create_host(engine, dict(label=label, host="build.example.test", port=2222, login_user="builder"))


def _record(engine, row, state="succeeded", op="verify", *, error=None, **result):
    target = {k: row[k] for k in ("host", "port", "login_user")}
    prefix = {"verify": "host login verified: ", "remove": "bot key removed from host: "}[op]
    job = SimpleNamespace(id="private-job-id", state=state, error_text=error, output_tail=prefix + json.dumps(result))
    ha.settle(engine, job, {"spec": {"op": op}, "host_key": {**target, "host_id": row["id"]}})


def mint_turn():
    return task_turn_context.mint_for_agent_turn(
        backend="claude-code", session_id=str(uuid4()), turn_id="turn-" + uuid4().hex,
        trigger_message_id=str(uuid4()), bot_id="fresh-bot", user_id="fixture-owner",
    )


async def _call(capability):
    binding = set_current_task_turn_capability(capability)
    try:
        return await tool.host_access_list()
    finally:
        reset_current_task_turn_capability(binding)


@pytest.mark.parametrize(("config", "expected"), [
    ("installed", "recorded_installed_runtime_unknown"),
    ("not_writable", "installation_not_confirmed"),
    (None, "unknown"),
])
def test_registration_verification_and_invalidation(engine, config, expected):
    row = _host(engine)
    [entry], truncated = ha.read_host_inventory(engine)
    assert entry.login_status == "registered_only" and entry.last_checked_at is None and not truncated

    _record(engine, row, login_ok=True, bot_ssh_config=config, key_fingerprint="secret fixture", raw="operator diagnostic")
    [entry], _ = ha.read_host_inventory(engine)
    assert (entry.login_status, entry.managed_config, entry.last_check) == ("verified", expected, "verify")
    assert entry.last_checked_at and entry.last_checked_at.endswith("+00:00")
    assert set(entry.model_dump()) == {
        "label", "host", "port", "login_user", "last_checked_at", "last_check", "login_status", "managed_config",
    }
    assert "secret fixture" not in entry.model_dump_json()

    _record(engine, row, state="failed", error="bot public key changed since approval; nothing was run")
    [entry], _ = ha.read_host_inventory(engine)
    assert (entry.login_status, entry.managed_config, entry.last_check) == ("unknown", "unknown", "verify")
    assert "public key changed" not in entry.model_dump_json()
    _record(engine, row, state="failed", error="login check failed: key_refused")
    [entry], _ = ha.read_host_inventory(engine)
    assert (entry.login_status, entry.managed_config, entry.last_check) == ("known_failed", "unknown", "verify")
    _record(engine, row, op="remove", login_ok=True)
    [entry], _ = ha.read_host_inventory(engine)
    assert entry.login_status == "unknown" and entry.last_check == "remove"
    ha.update_host(engine, row["id"], {"host": "other.example.test"})
    _record(engine, row, login_ok=True, bot_ssh_config="installed")  # stale receipt
    [entry], _ = ha.read_host_inventory(engine)
    assert entry.host == "other.example.test"
    assert (entry.login_status, entry.last_checked_at, entry.last_check) == ("registered_only", None, None)


def test_inventory_bounded_and_read_only(engine, monkeypatch):
    for i in range(ha.INVENTORY_LIMIT + 2):
        _host(engine, f"Fixture {i:02}")
    before = ha.list_hosts(engine)
    monkeypatch.setattr(ha, "ensure_schema", Mock(side_effect=AssertionError("must not bootstrap")))
    for name in ("run_worker", "install_trust", "read_public_key"):
        monkeypatch.setattr(ha.host_key, name, Mock(side_effect=AssertionError("must not execute ops")))
    statements = []

    @event.listens_for(engine, "before_cursor_execute")
    def capture(_conn, _cursor, statement, parameters, _context, _executemany):
        statements.append((statement, parameters))

    entries, truncated = ha.read_host_inventory(engine)
    assert truncated and [entry.label for entry in entries] == [f"Fixture {i:02}" for i in range(ha.INVENTORY_LIMIT)]
    [(statement, parameters)] = statements
    assert statement.startswith("SELECT ") and "LIMIT" in statement and parameters[0] == ha.INVENTORY_LIMIT + 1
    assert all(name not in statement for name in ("known_host_keys_json", "last_job_id", "created_by"))
    assert len(before) == ha.INVENTORY_LIMIT + 2


@pytest.mark.asyncio
async def test_empty_vs_unavailable_and_installation_bound_scope(engine, monkeypatch):
    tenant_a, tenant_b = (Fernet(Fernet.generate_key()) for _ in range(2))
    monkeypatch.setattr(task_turn_context, "_get_fernet", lambda: tenant_a)
    authorized = mint_turn()
    other_engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    _host(other_engine, "Other installation")
    monkeypatch.setattr(task_turn_context, "_get_fernet", lambda: tenant_b)
    foreign = mint_turn()
    assert authorized and foreign
    monkeypatch.setattr(task_turn_context, "_get_fernet", lambda: tenant_a)

    engine_factory = Mock(return_value=engine)
    monkeypatch.setattr(tool, "get_shared_engine", engine_factory)
    for capability in (None, "invalid", foreign):
        result = await _call(capability)
        assert result.status == "unauthorized" and result.hosts == []
    engine_factory.assert_not_called()

    result = await _call(authorized)
    assert result.status == "unavailable" and result.hosts == []
    assert not inspect(engine).has_table("ops_host_access")
    ha.ensure_schema(engine)
    result = await _call(authorized)
    assert result.status == "ok" and result.hosts == []
    _host(engine)
    result = await _call(authorized)
    assert result.status == "ok" and [host.label for host in result.hosts] == ["Build box"]
    other_engine.dispose()
