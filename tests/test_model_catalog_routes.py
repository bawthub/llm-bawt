import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from unittest.mock import MagicMock, patch

from llm_bawt.service.routes.model_catalog import (
    AccessPathWrite,
    EndpointWrite,
    ModelWrite,
    _check_endpoint_cas,
    _delete_catalog_model,
    _normalized_harness,
    list_harnesses,
)


def test_model_write_requires_positive_context_window():
    # TASK-616: a windowless model must be rejected at write time so the global
    # default stays a deliberate fallback, not a substitute for missing data.
    with pytest.raises(ValidationError):
        ModelWrite(vendor="xai", display_name="Grok")  # no default_context_window
    with pytest.raises(ValidationError):
        ModelWrite(vendor="xai", display_name="Grok", default_context_window=0)
    with pytest.raises(ValidationError):
        ModelWrite(vendor="xai", display_name="Grok", default_context_window=-1)

    ok = ModelWrite(vendor="xai", display_name="Grok", default_context_window=1000000)
    assert ok.default_context_window == 1000000


def test_endpoint_serving_config_is_normalized_and_validated():
    endpoint = EndpointWrite(
        serving_config={
            "reasoning_effort": " HIGH ",
            "responses_transport": " SSE ",
            "other": True,
        }
    )
    assert endpoint.serving_config == {
        "reasoning_effort": "high",
        "responses_transport": "sse",
        "other": True,
    }

    with pytest.raises(ValidationError, match="reasoning_effort"):
        EndpointWrite(serving_config={"reasoning_effort": "minimal"})
    with pytest.raises(ValidationError, match="reasoning_effort"):
        EndpointWrite(serving_config={"reasoning_effort": 42})
    with pytest.raises(ValidationError, match="responses_transport"):
        EndpointWrite(serving_config={"responses_transport": "websocket"})
    with pytest.raises(ValidationError, match="responses_transport"):
        EndpointWrite(serving_config={"responses_transport": True})


def test_access_path_prompt_overrides_trim_drop_blanks_and_preserve_entries():
    request = AccessPathWrite(
        vendor="openai",
        protocol="responses",
        auth_mechanism="oauth",
        system_prompt_overrides={
            "claude-proxy": "  proxy instructions  ",
            "codex": " \n ",
        },
    )

    assert request.system_prompt_overrides == {
        "claude-proxy": "proxy instructions"
    }


def test_access_path_prompt_overrides_reject_unknown_harness_and_non_string():
    with pytest.raises(ValidationError, match="Unknown harness"):
        AccessPathWrite(
            vendor="openai",
            protocol="responses",
            auth_mechanism="oauth",
            system_prompt_overrides={"future-harness": "instructions"},
        )
    with pytest.raises(ValidationError):
        AccessPathWrite(
            vendor="openai",
            protocol="responses",
            auth_mechanism="oauth",
            system_prompt_overrides={"claude-proxy": 42},
        )


def test_catalog_harnesses_are_driven_by_protocol_compatibility():
    rows = {row["key"]: row for row in list_harnesses()["harnesses"]}

    assert rows["chat"]["protocol"] == "chat-completions"
    assert rows["codex"]["protocol"] == "responses"
    assert rows["claude-proxy"]["proxies"] is True


def test_unknown_harness_is_rejected_before_querying_catalog():
    with pytest.raises(HTTPException) as exc:
        _normalized_harness("not-a-harness")

    assert exc.value.status_code == 400


def test_endpoint_compare_and_set_rejects_stale_or_missing_identity():
    _check_endpoint_cas(20, None)
    _check_endpoint_cas(20, 20)

    with pytest.raises(HTTPException, match="expected 19, found 20"):
        _check_endpoint_cas(20, 19)
    with pytest.raises(HTTPException, match="no longer exists"):
        _check_endpoint_cas(None, 20)


def test_delete_catalog_model_removes_normalized_row():
    # The legacy ``model_definitions`` table is dropped, so deletion now targets
    # only the normalized ``models`` row (endpoints cascade at the DB level).
    conn = MagicMock()
    conn.execute.return_value.first.return_value = (42,)

    assert _delete_catalog_model(conn, "grok-old") == (42,)

    stmt = str(conn.execute.call_args.args[0])
    assert "DELETE FROM models" in stmt
    assert conn.execute.call_args.args[1] == {"key": "grok-old"}


# ---------------------------------------------------------------------------
# Access-path list: agent_backend filter + provider connection (TASK-1297)
# ---------------------------------------------------------------------------

#: (key, vendor, protocol, auth_mechanism, engine_kind), as on a live tenant.
_PATH_ROWS = (
    ("anthropic-api", "anthropic", "anthropic-messages", "api-key", None),
    ("anthropic-oauth", "anthropic", "anthropic-messages", "oauth", None),
    ("local-llamacpp", "local", "chat-completions", "none", "llama-cpp"),
    ("openai-api", "openai", "chat-completions", "api-key", None),
    ("openai-oauth", "openai", "responses", "oauth", None),
    ("openai-responses", "openai", "responses", "api-key", None),
    ("openrouter-anthropic", "openrouter", "anthropic-messages", "api-key", None),
    ("xai-chat", "xai", "chat-completions", "api-key", None),
    ("xai-responses", "xai", "responses", "api-key", None),
)


def _list_access_paths(**query):
    from llm_bawt.service.routes import model_catalog

    rows = [
        {"id": i, "key": key, "vendor": vendor, "protocol": protocol, "base_url": None,
         "auth_mechanism": auth, "engine_kind": engine, "system_prompt_overrides": {}}
        for i, (key, vendor, protocol, auth, engine) in enumerate(_PATH_ROWS, start=1)
    ]
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value.execute.return_value.mappings.return_value.all.return_value = rows
    service = MagicMock()
    connections = MagicMock(side_effect=lambda _config, keys: {key: {"key": key} for key in keys})
    with (
        patch.object(model_catalog, "_engine", return_value=engine),
        patch.object(model_catalog, "get_service", return_value=service),
        patch.object(model_catalog, "access_path_connections", connections),
    ):
        result = model_catalog.list_access_paths(**query)
    return result["access_paths"], connections, service


def test_access_paths_for_claude_code_backend_cover_native_and_proxy_routes():
    rows, connections, service = _list_access_paths(harness=None, agent_backend="claude-code")
    keys = [row["key"] for row in rows]
    # Everything the bridge can run natively or through its proxy; openai chat-completions it can't.
    assert keys == [key for key, *_ in _PATH_ROWS if key != "openai-api"]
    connections.assert_called_once_with(service.config, keys)
    assert all(row["connection"] == {"key": row["key"]} for row in rows)


def test_access_paths_harness_filter_still_narrows_to_one_harness():
    rows, _, _ = _list_access_paths(harness="claude-code", agent_backend=None)
    assert [row["key"] for row in rows] == ["anthropic-api", "anthropic-oauth"]


def test_access_paths_unfiltered_lists_every_row_with_a_connection():
    rows, _, _ = _list_access_paths(harness=None, agent_backend=None)
    assert [row["key"] for row in rows] == [key for key, *_ in _PATH_ROWS]
    assert all("connection" in row for row in rows)


@pytest.mark.parametrize(("harness", "agent_backend", "message"), [
    (None, "nope", "Unknown agent_backend"),
    ("claude-code", "claude-code", "not both"),
])
def test_access_path_filter_rejects_bad_combinations(harness, agent_backend, message):
    from llm_bawt.service.routes.model_catalog import _path_filter

    with pytest.raises(HTTPException, match=message) as exc:
        _path_filter(harness, agent_backend)
    assert exc.value.status_code == 400


def test_harness_for_agent_backend_matches_profile_derivation():
    from llm_bawt.model_catalog import AccessPath
    from llm_bawt.service.routes.model_catalog import harness_for_agent_backend

    def path(key, vendor, protocol):
        return AccessPath(1, key, vendor, protocol, None, "api-key")

    assert harness_for_agent_backend("claude-code", path("anthropic-api", "anthropic", "anthropic-messages")) == "claude-code"
    assert harness_for_agent_backend("claude-code", path("xai-responses", "xai", "responses")) == "claude-proxy"
    assert harness_for_agent_backend("claude-code", path("openai-api", "openai", "chat-completions")) is None
    assert harness_for_agent_backend("codex", path("openai-oauth", "openai", "responses")) == "codex"
    assert harness_for_agent_backend("codex", path("xai-responses", "xai", "responses")) is None


def test_model_write_compact_threshold_is_optional_and_bounded():
    base = {"vendor": "anthropic", "display_name": "Opus", "default_context_window": 1_000_000}
    assert ModelWrite(**base).compact_threshold is None
    assert ModelWrite(**base, compact_threshold=400_000).compact_threshold == 400_000
    for bad in (99_999, 1_000_001):
        with pytest.raises(ValidationError):
            ModelWrite(**base, compact_threshold=bad)
