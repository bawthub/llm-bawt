"""TASK-1302 A0/A3: the route eligibility rule and the lists that carry it.

``route_eligibility`` is the one rule behind the bot-save gate and every route
list, so the picker can never offer a route the save would refuse.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from llm_bawt.model_catalog import AccessPath, ModelCatalog, ModelEndpoint, ModelIdentity
from llm_bawt.service.providers.registry import route_eligibility
from llm_bawt.service.routes import bot_profile_validation as validation

#: The app package; every BotProfileStore writer lives here.
SRC = Path(__file__).resolve().parents[1] / "src" / "llm_bawt"


def _connection(state, detail=None):
    return {"provider_id": "p", "label": "Provider", "state": state, "detail": detail,
            "connected": state in {"ok", "warning"}}


@pytest.mark.parametrize(("auth", "connection", "eligible", "state", "reason"), [
    ("none", None, True, "local", None),
    ("NONE", _connection("broken"), True, "local", None),
    ("api-key", _connection("ok"), True, "ok", None),
    ("oauth", _connection("warning", "expires soon"), True, "warning", None),
    ("api-key", _connection("unconfigured"), False, "unconfigured",
     "Provider is not connected. Connect it in Integrations or pick a connected route."),
    ("oauth", _connection("broken", "refresh failed"), False, "broken",
     "Provider needs reconnecting (refresh failed)."),
    ("oauth", _connection("broken"), False, "broken", "Provider needs reconnecting."),
    ("api-key", _connection("unknown", "timeout"), False, "unknown",
     "Provider status couldn't be checked (timeout)."),
    ("api-key", {"state": None}, False, "unknown", "path status couldn't be checked."),
    ("configured", None, False, "untracked",
     "BawtHub doesn't manage the credential for path; this route can't be assigned yet."),
    (None, None, False, "untracked",
     "BawtHub doesn't manage the credential for path; this route can't be assigned yet."),
])
def test_route_eligibility_table(auth, connection, eligible, state, reason):
    assert route_eligibility("path", auth, connection) == {
        "eligible": eligible, "state": state, "reason": reason,
    }


def _list_endpoints(connections, **query):
    from llm_bawt.service.routes import model_catalog

    rows = [
        {"id": i, "model_key": "m", "access_path_id": i, "access_path_key": key,
         "access_vendor": vendor, "protocol": protocol, "base_url": None,
         "auth_mechanism": auth, "engine_kind": None}
        for i, (key, vendor, protocol, auth) in enumerate((
            ("anthropic-api", "anthropic", "anthropic-messages", "api-key"),
            ("local-vllm", "local", "chat-completions", "none"),
            ("openai-api", "openai", "chat-completions", "api-key"),
            ("openai-oauth", "openai", "responses", "oauth"),
        ), start=1)
    ]
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value.execute.return_value.mappings.return_value.all.return_value = rows
    probe = MagicMock(side_effect=lambda _config, keys: {key: connections.get(key) for key in keys})
    with (
        patch.object(model_catalog, "_engine", return_value=engine),
        patch.object(model_catalog, "get_service", return_value=MagicMock()),
        patch.object(model_catalog, "access_path_connections", probe),
    ):
        result = model_catalog.list_endpoints(**query)
    return {row["access_path_key"]: row for row in result["endpoints"]}, probe


_CONNECTIONS = {
    "anthropic-api": _connection("unconfigured"),
    "openai-oauth": _connection("ok"),
    "openai-api": _connection("ok"),
}


def test_endpoint_rows_carry_connection_eligibility_and_save_harness_for_claude_code():
    rows, probe = _list_endpoints(_CONNECTIONS, agent_backend="claude-code")

    assert set(rows) == {"anthropic-api", "local-vllm", "openai-oauth"}
    probe.assert_called_once()  # one probe for the whole list, after the filter
    assert sorted(probe.call_args.args[1]) == sorted(rows)

    assert rows["anthropic-api"]["harness"] == "claude-code"
    assert rows["anthropic-api"]["eligibility"]["eligible"] is False
    assert rows["anthropic-api"]["connection"] == _CONNECTIONS["anthropic-api"]
    # The ChatGPT OAuth route runs a Claude bot through the proxy.
    assert rows["openai-oauth"]["harness"] == "claude-proxy"
    assert rows["openai-oauth"]["eligibility"] == {"eligible": True, "state": "ok", "reason": None}
    assert rows["local-vllm"]["harness"] == "claude-proxy"
    assert rows["local-vllm"]["eligibility"]["state"] == "local"


def test_endpoint_rows_for_codex_list_only_its_oauth_route():
    rows, _ = _list_endpoints(_CONNECTIONS, agent_backend="codex")
    assert set(rows) == {"openai-oauth"}
    assert rows["openai-oauth"]["harness"] == "codex"
    assert rows["openai-oauth"]["eligibility"]["eligible"] is True


def test_endpoint_rows_without_a_backend_carry_eligibility_but_no_harness():
    rows, _ = _list_endpoints(_CONNECTIONS)
    assert set(rows) == {"anthropic-api", "local-vllm", "openai-api", "openai-oauth"}
    assert all("eligibility" in row and "harness" not in row for row in rows.values())


def test_v1_models_labels_each_alias_with_its_route_and_eligibility(monkeypatch):
    """One model on API key and subscription lists as two aliases, each with its own route."""
    from llm_bawt.service.routes import models as models_route

    claude = ModelIdentity(1, "claude-x", "anthropic", "Claude X")
    local = ModelIdentity(2, "qwen", "local", "Qwen")
    api = AccessPath(10, "anthropic-api", "anthropic", "anthropic-messages", None, "api-key")
    oauth = AccessPath(11, "anthropic-oauth", "anthropic", "anthropic-messages", None, "oauth")
    vllm = AccessPath(12, "local-vllm", "local", "chat-completions", None, "none")
    catalog = ModelCatalog([
        ModelEndpoint(100, claude, api, "claude-x"),
        ModelEndpoint(101, claude, oauth, "claude-x"),
        ModelEndpoint(102, local, vllm, "qwen"),
    ])
    defined = {**catalog.compatibility_mapping(), "legacy": {"type": "openai", "model_id": "old"}}
    service = SimpleNamespace(
        config=SimpleNamespace(defined_models={"models": defined}),
        _available_models=list(defined),
    )
    connections = {
        "anthropic-api": _connection("unconfigured"),
        "anthropic-oauth": _connection("ok"),
    }
    probe = MagicMock(side_effect=lambda _config, keys: {key: connections.get(key) for key in keys})
    monkeypatch.setattr(models_route, "get_service", lambda: service)
    monkeypatch.setattr(models_route, "access_path_connections", probe)

    rows = {row["id"]: row for row in models_route.list_models().model_dump()["data"]}

    probe.assert_called_once()  # one probe for the whole list
    assert sorted(probe.call_args.args[1]) == ["anthropic-api", "anthropic-oauth", "local-vllm"]
    api_row, oauth_row = rows["claude-x@anthropic-api"], rows["claude-x@anthropic-oauth"]
    assert (api_row["endpoint_id"], api_row["access_path"]) == (100, "anthropic-api")
    assert api_row["eligibility"] == {
        "eligible": False, "state": "unconfigured",
        "reason": "Provider is not connected. Connect it in Integrations or pick a connected route.",
    }
    assert (oauth_row["endpoint_id"], oauth_row["access_path"]) == (101, "anthropic-oauth")
    assert oauth_row["eligibility"] == {"eligible": True, "state": "ok", "reason": None}
    assert rows["qwen"]["eligibility"] == {"eligible": True, "state": "local", "reason": None}
    assert rows["qwen"]["endpoint_id"] == 102
    legacy = rows["legacy"]
    assert (legacy["endpoint_id"], legacy["access_path"], legacy["eligibility"]) == (None, None, None)


def test_python_validation_stays_the_harness_authority_over_the_coarse_trigger(monkeypatch):
    """The DB trigger admits codex on any Responses path; the save's validation does not.

    The route gate judges only the credential, so this 422 is what keeps a
    codex bot off openai-responses.
    """
    model = ModelIdentity(1, "gpt-test", "openai", "GPT test")
    responses = AccessPath(2, "openai-responses", "openai", "responses", None, "api-key")
    oauth = AccessPath(3, "openai-oauth", "openai", "responses", None, "oauth")
    catalog = ModelCatalog([
        ModelEndpoint(20, model, responses, "gpt-test"),
        ModelEndpoint(30, model, oauth, "gpt-test"),
    ])
    service = SimpleNamespace(config=SimpleNamespace(ensure_model_catalog=lambda: catalog))
    monkeypatch.setattr(validation, "get_service", lambda: service)

    with pytest.raises(HTTPException) as exc:
        validation._validate_model_endpoint({"slug": "b", "harness": "codex", "endpoint_id": 20})
    assert exc.value.status_code == 422

    payload = {"slug": "b", "harness": "codex", "endpoint_id": 30}
    validation._validate_model_endpoint(payload)
    assert payload["agent_backend"] == "codex"


def _calls_named(tree, name):
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", getattr(node.func, "id", None)) == name]


def test_only_the_guarded_route_save_writes_a_bot_endpoint():
    """Every other BotProfileStore writer leaves endpoint_id alone.

    ``seed_from_yaml`` would write it unguarded, so it must stay uncalled; the
    remaining profile upserts pass payloads without an ``endpoint_id`` key.
    """
    seed_callers, endpoint_writers = [], []
    for path in SRC.rglob("*.py"):
        source = path.read_text()
        if "seed_from_yaml(" not in source and "endpoint_id" not in source:
            continue  # parse only files that could break the rule (unit time budget)
        tree = ast.parse(source, filename=str(path))
        if _calls_named(tree, "seed_from_yaml"):
            seed_callers.append(path.name)
        for call in _calls_named(tree, "upsert"):
            if any(kw.arg == "guard" for kw in call.keywords):
                continue
            for node in ast.walk(call):
                if isinstance(node, ast.Dict) and any(
                    isinstance(k, ast.Constant) and k.value == "endpoint_id" for k in node.keys
                ):
                    endpoint_writers.append(f"{path.name}:{call.lineno}")
    assert seed_callers == []
    assert endpoint_writers == []
    bots = ast.parse((SRC / "bots.py").read_text())
    assert not any(
        isinstance(node, ast.Dict)
        and any(isinstance(k, ast.Constant) and k.value == "endpoint_id" for k in node.keys)
        for node in ast.walk(bots)
    )
