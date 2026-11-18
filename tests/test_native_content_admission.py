"""Native policy and content checks execute before any backend mutation."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import create_engine, select

from agent_bridge.approval import ApprovalPolicy, PolicyAction, PolicyBundle
from llm_bawt.content_provenance import ContentProvenanceStore, ContentTurn, _writes
from llm_bawt.tools.approval import NativePolicyGate
from llm_bawt.tools.executor import ToolExecutor
from llm_bawt.tools.parser import ToolCall, tool_result_is_error


def call(name, **args):
    return ToolCall(name=name, arguments=args, raw_text="")


def policy_store(*policies):
    return SimpleNamespace(
        compile_bundle=lambda: PolicyBundle(1, "fixture", list(policies)),
        record_decision=Mock(),
    )


@pytest.fixture
def store():
    engine = create_engine("sqlite://")
    yield ContentProvenanceStore(engine)
    engine.dispose()


def executor(store=None, **kwargs):
    return ToolExecutor(
        user_id="owner", bot_id="reader", trigger_message_id="trigger",
        turn_id="turn", content_store=store, **kwargs,
    )


@pytest.mark.parametrize("action,held", [
    (PolicyAction.ALLOW, False), (PolicyAction.DENY, True),
    (PolicyAction.REQUIRE_APPROVAL, True),
])
def test_native_policy_actions_precede_all_handlers(action, held):
    rules = policy_store(ApprovalPolicy("rule", tool_name="Bash", action=action))
    ex = executor(policy_gate=NativePolicyGate(store=rules))
    handler = Mock(return_value="ok")
    ex._handlers["Bash"] = handler
    result = ex.execute(call("Bash", command="true"))
    assert tool_result_is_error(result) is held
    assert handler.call_count == (0 if held else 1)
    audit = rules.record_decision.call_args.kwargs
    assert audit["backend"] == "native" and audit["turn_id"] == "turn"
    assert len(audit["invocation_hash"]) == 64
    if action == PolicyAction.REQUIRE_APPROVAL:
        assert "no card" in result and "Nothing was executed" in result


def test_unmatched_policy_and_bridge_only_policy_do_not_hold_native():
    rules = policy_store(ApprovalPolicy(
        "bridge-only", backend_scope="claude-code", tool_name="Bash",
        action=PolicyAction.DENY,
    ))
    ex = executor(policy_gate=NativePolicyGate(store=rules))
    ex._handlers["Bash"] = Mock(return_value="ok")
    assert not tool_result_is_error(ex.execute(call("Bash", command="true")))


@pytest.mark.parametrize("name,handler_name,args", [
    ("store_memory", "memory", {"content": "x"}), ("memory", "memory", {"action": "search"}),
    ("time", "time", {}),
])
def test_ungated_native_tools_are_neither_evaluated_nor_recorded(name, handler_name, args):
    """TASK-2418: only shell and ops tools meet approval policy."""
    rules = policy_store(ApprovalPolicy("rule", tool_name=name, action=PolicyAction.DENY))
    rules.compile_bundle = Mock(side_effect=AssertionError("must not evaluate"))
    ex = executor(policy_gate=NativePolicyGate(store=rules))
    handler = Mock(return_value="ok")
    ex._handlers[handler_name] = handler
    assert not tool_result_is_error(ex.execute(call(name, **args)))
    handler.assert_called_once()
    rules.record_decision.assert_not_called()


@pytest.mark.parametrize("failure", ["compile", "audit"])
def test_policy_or_audit_failure_never_invokes_handler(failure):
    rules = policy_store()
    if failure == "compile":
        rules.compile_bundle = Mock(side_effect=RuntimeError("not logged"))
    else:
        rules.record_decision.side_effect = RuntimeError("not logged")
    ex = executor(policy_gate=NativePolicyGate(store=rules))
    handler = Mock()
    ex._handlers["Bash"] = handler
    assert "policies are unavailable" in ex.execute(call("Bash", command="true"))
    handler.assert_not_called()


def test_ha_native_tools_are_not_policy_evaluated():
    ha = SimpleNamespace(is_ha_tool=lambda name: True, call_tool=Mock(return_value="done"))
    rules = policy_store(ApprovalPolicy(
        "rule", tool_name="HassTurnOn", action=PolicyAction.REQUIRE_APPROVAL,
    ))
    result = executor(ha_native_client=ha, policy_gate=NativePolicyGate(store=rules)).execute(
        call("HassTurnOn", domain=["light"]),
    )
    assert "Held:" not in result
    rules.record_decision.assert_not_called()


@pytest.mark.parametrize("name,args", [
    ("memory", {"action": "store"}), ("memory", {"action": "update"}),
    ("memory", {"action": "delete"}), ("self", {"action": "set"}),
    ("self", {"action": "delete"}), ("profile", {"action": "set"}),
    ("profile", {"action": "delete"}), ("store_memory", {"content": "unsafe"}),
    ("bot_trait", {"key": "tone", "value": "unsafe"}),
])
def test_exposed_writes_have_receipts_and_zero_mutator_calls(store, name, args):
    ex = executor(store)
    mutators = {key: Mock() for key in ("memory", "self", "profile")}
    ex._handlers.update(mutators)
    store.expose(ContentTurn("reader", "owner", "turn", "trigger"))
    result = ex.execute(call(name, **args))
    assert "Held:" in result and "untrusted_seen=true" in result and "Receipt" in result
    for handler in mutators.values():
        handler.assert_not_called()
    with store.engine.connect() as conn:
        receipt = conn.execute(select(_writes)).mappings().one()
    assert receipt["source_turn"] == "turn" and receipt["decision"] == "held"


def test_external_exposure_commits_before_handler_even_when_handler_fails(store):
    ex = executor(store)

    def fail(_call):
        assert store.seen(ContentTurn("reader", "owner", "turn", "trigger"))
        raise RuntimeError("external response body")

    ex._handlers["web_fetch"] = fail
    result = ex.execute(call("web_fetch", url="https://example.test"))
    assert tool_result_is_error(result)
    assert '"untrusted_content":true' in result and '"source":"native:web_fetch"' in result
    assert ex.external_content_seen
    assert "Held:" in ex.execute(call("memory", action="store", content="do not store"))


def test_durable_exposure_survives_executor_reset_and_retry(store):
    store.expose(ContentTurn("reader", "owner", "turn", "trigger"))
    ex = executor(store)
    ex.reset_call_count()
    assert "Held:" in ex.execute(call("self", action="set", key="tone", value="unsafe"))
    retry = executor(store)
    assert "Held:" in retry.execute(call("memory", action="delete", memory_id="m"))
    fresh = ToolExecutor(
        user_id="owner", bot_id="reader", trigger_message_id="next-trigger",
        turn_id="next-turn", content_store=store,
    )
    handler = Mock(return_value="ok")
    fresh._handlers["memory"] = handler
    assert fresh.execute(call("memory", action="store", content="clean")) == "ok"
    handler.assert_called_once()


def test_provenance_outage_or_missing_identity_never_runs_mutator(store):
    for ex in (
        ToolExecutor(user_id="owner", bot_id="reader", content_store=store),
        executor(SimpleNamespace(admit=Mock(side_effect=RuntimeError("down")))),
    ):
        handler = Mock()
        ex._handlers["memory"] = handler
        assert "content provenance is unavailable" in ex.execute(call("memory", action="store"))
        handler.assert_not_called()


def test_trusted_human_fact_extraction_uses_unchanged_storage_despite_bot_exposure(store, monkeypatch):
    from llm_bawt.mcp_server.client import MemoryClient
    from llm_bawt.mcp_server import extraction
    from llm_bawt.service import dependencies

    store.expose(ContentTurn("reader", "owner", "turn", "trigger"))
    # Tool admission is not the shared storage API. Automatic extraction from
    # the human's authored message must still reach its existing storage path.
    monkeypatch.setattr(
        dependencies, "get_content_provenance_store",
        Mock(side_effect=AssertionError("background extraction reached bot tool admission")),
    )
    fact = {"id": "fact", "content": "Owner enjoys hiking", "source_message_ids": ["human-msg"]}
    monkeypatch.setattr(extraction, "extract_facts_from_messages", AsyncMock(return_value=[fact]))
    storage = SimpleNamespace(store_extracted_memory=AsyncMock(
        return_value=SimpleNamespace(to_dict=lambda: fact),
    ))
    client = MemoryClient.__new__(MemoryClient)
    client.config, client.bot_id, client.user_id = SimpleNamespace(), "reader", "owner"
    client.server_url, client._initialized, client._storage = None, True, storage
    results = client.extract_facts([{
        "id": "human-msg", "role": "user", "content": "I enjoy hiking.",
        "author_entity_type": "user", "author_entity_id": "owner",
    }], store=True, use_llm=False)
    assert results[0].id == "fact"
    storage.store_extracted_memory.assert_awaited_once()
    assert storage.store_extracted_memory.call_args.kwargs["source_message_ids"] == ["human-msg"]
