"""TASK-1203: a scheduled reset's continuity parameter drives one /new scope."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from llm_bawt.message_authorship import AuthorReference
from llm_bawt.service.job_parameters import IdleContextResetParameters, RESET_CONTINUITY_TITLES
from llm_bawt.service.session_lifecycle.execution import NewSessionExecution
from llm_bawt.service.session_lifecycle.intent import (
    Admission, Continuity, NewSessionIntent, Origin,
)
from llm_bawt.service.session_lifecycle.lifecycle import NewSessionLifecycle
from llm_bawt.setting_definitions import HISTORY_SCOPES
from llm_bawt.utils.history import effective_history_scope


def scheduled(parameters=None):
    return NewSessionIntent(
        bot_id="fixture", user_id="owner", author=AuthorReference.bot("system"),
        continuity=Continuity.CONFIGURED, admission=Admission.SKIP, origin=Origin.SCHEDULER,
        idempotency_key="key", expected_session_id="old", expected_usage_turn_id="usage",
        job_id="job", run_id="run", scheduler_parameters=parameters,
    )


def resolver(scope):
    return SimpleNamespace(resolve_config_setting=lambda key: SimpleNamespace(
        value=scope if key == "history_scope" else None,
    ))


def service_with(metadata):
    store = SimpleNamespace(get=lambda _id: None if metadata is None else SimpleNamespace(metadata=metadata))
    return SimpleNamespace(_inter_bot_dispatcher=SimpleNamespace(store=store))


def test_continuity_parameter_schema_and_override():
    schema = IdleContextResetParameters.model_json_schema()["properties"]["continuity"]
    assert schema["enum"] == ["configured", *HISTORY_SCOPES]
    assert schema["x-enum-titles"] == list(RESET_CONTINUITY_TITLES.values())
    assert schema["default"] == "configured"
    assert IdleContextResetParameters().history_scope_override is None
    for scope in HISTORY_SCOPES:
        assert IdleContextResetParameters(continuity=scope).history_scope_override == scope
    with pytest.raises(ValidationError):
        IdleContextResetParameters(continuity="bogus")


def test_old_intent_decodes_as_configured_and_roundtrips():
    old = NewSessionLifecycle._encode(scheduled(IdleContextResetParameters()))
    del old["scheduler_parameters"]["continuity"]
    assert NewSessionLifecycle._decode(old).scheduler_parameters.continuity == "configured"
    raw = NewSessionLifecycle._encode(scheduled(IdleContextResetParameters(continuity="inline")))
    assert NewSessionLifecycle._decode(raw).scheduler_parameters.history_scope_override == "inline"


def test_effective_history_scope():
    assert effective_history_scope(resolver("summaries")) == "summaries"
    assert effective_history_scope(resolver(None)) == "inline+summaries"
    assert effective_history_scope(resolver("summaries"), "none") == "none"
    assert effective_history_scope(None, "inline") == "inline"
    with pytest.raises(ValueError):
        effective_history_scope(resolver("inline"), "everything")


def test_seed_obeys_override_in_both_directions(monkeypatch):
    from llm_bawt.service.routes import history_seed

    built = Mock(return_value={"messages": [{"role": "user", "content": "carried"}]})
    monkeypatch.setattr(history_seed, "build_context_seed", built)
    binding = {"thread_session_id": "old"}

    on = SimpleNamespace(bot=SimpleNamespace(agent_backend="claude-code"),
                         config_resolver=resolver("inline+summaries"))
    assert history_seed.maybe_build_session_seed(on, "fixture", "m", "/new", None, binding,
                                                 history_scope="none") == []
    built.assert_not_called()

    off = SimpleNamespace(bot=SimpleNamespace(agent_backend="claude-code"), config_resolver=resolver("none"))
    assert history_seed.maybe_build_session_seed(off, "fixture", "m", "/new", None, binding,
                                                 history_scope="inline") == [
        {"role": "user", "content": "carried"}]
    assert built.call_args.kwargs["history_scope"] == "inline"
    assert history_seed.maybe_build_session_seed(off, "fixture", "m", "/new", None, binding) == []


def test_summarize_gate_obeys_override():
    from llm_bawt.service.session_lifecycle.bridge_reset import BridgeSessionResetMixin

    llm = SimpleNamespace(config_resolver=resolver("inline+summaries"),
                          history_manager=SimpleNamespace(_db_backend=None))
    summarize = BridgeSessionResetMixin._maybe_summarize_on_new
    # The override removes summaries before any backend is consulted.
    assert summarize(Mock(), llm, "fixture", "/new", history_scope="inline") is False
    assert summarize(Mock(), llm, "fixture", "/new", history_scope="none") is False


def test_scheduled_history_scope_reads_only_the_durable_scheduler_intent():
    raw = NewSessionLifecycle._encode(scheduled(IdleContextResetParameters(continuity="summaries")))
    read = NewSessionExecution.scheduled_history_scope
    assert read(service_with({"new_session_intent": raw}), "d") == "summaries"
    assert read(service_with({"new_session_intent": NewSessionLifecycle._encode(scheduled())}), "d") is None
    assert read(service_with({"new_session_intent": {**raw, "origin": "chat_command"}}), "d") is None
    assert read(service_with({}), "d") is None
    assert read(service_with({"new_session_intent": raw}), None) is None
    assert read(SimpleNamespace(), "d") is None
    with pytest.raises(ValueError, match="missing"):
        read(service_with(None), "d")


def counting_llm(scope):
    reads = []

    def resolve(key):
        reads.append(key)
        return SimpleNamespace(value=scope if key == "history_scope" else None)

    return SimpleNamespace(config_resolver=SimpleNamespace(resolve_config_setting=resolve)), reads


def prepare(monkeypatch, service, llm):
    from llm_bawt.service.routes import history

    seed = Mock(return_value=[])
    monkeypatch.setattr(history, "maybe_build_session_seed", seed)
    service._maybe_summarize_on_new = Mock()
    request = SimpleNamespace(inter_bot_session_policy=None, inter_bot_typed_follow_up=False)
    prepared = NewSessionExecution.prepare_agent(
        service, llm, request, bot_id="fixture", user_prompt="/new", model_alias="m",
        thread_binding={"thread_session_id": "old"}, delivery_id="d",
        backend_name="claude-code", is_agent=False,
    )
    return prepared, service._maybe_summarize_on_new.call_args.kwargs["history_scope"], \
        seed.call_args.kwargs["history_scope"]


@pytest.mark.parametrize("continuity,expected", [("none", "none"), ("configured", "summaries")])
def test_prepare_agent_resolves_one_scope_for_summary_seed_and_receipt(monkeypatch, continuity, expected):
    raw = NewSessionLifecycle._encode(scheduled(IdleContextResetParameters(continuity=continuity)))
    llm, reads = counting_llm("summaries")
    prepared, summary_scope, seed_scope = prepare(
        monkeypatch, service_with({"new_session_intent": raw}), llm)
    assert summary_scope == seed_scope == prepared.history_scope == expected
    # The bot setting is read at most once per reset, so an edit cannot split it.
    assert reads.count("history_scope") == (0 if continuity != "configured" else 1)


def test_prepare_agent_ordinary_new_uses_the_bot_scope_once(monkeypatch):
    llm, reads = counting_llm("inline")
    prepared, summary_scope, seed_scope = prepare(
        monkeypatch, service_with({"new_session_intent": {"origin": "chat_command"}}), llm)
    assert summary_scope == seed_scope == prepared.history_scope == "inline"
    assert reads.count("history_scope") == 1


def test_prepare_agent_unresolvable_bot_scope_keeps_consumer_fallbacks(monkeypatch):
    llm, _ = counting_llm("legacy-garbage")
    prepared, summary_scope, seed_scope = prepare(monkeypatch, service_with({}), llm)
    assert summary_scope is seed_scope is prepared.history_scope is None


def test_continuity_receipt_records_the_scope_actually_used(monkeypatch):
    from llm_bawt.memory.session_continuity import SessionContinuity
    from llm_bawt.service import session_continuity

    record = Mock(return_value=True)
    monkeypatch.setattr(session_continuity, "get_storage", lambda: SimpleNamespace(engine="engine"))
    monkeypatch.setattr(SessionContinuity, "record", record)
    llm = SimpleNamespace(resolved_model_alias="m", config_resolver=resolver("inline+summaries"))
    session_continuity.record_session_continuity(
        llm, {"thread_session_id": "new"}, [], bot_id="fixture", user_id="owner", history_scope="none")
    assert record.call_args.args[3]["settings"]["history_scope"] == "none"
