"""Preview entrypoint rejects production authority and never runs the prod app."""

import asyncio
from types import SimpleNamespace
import socket
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from llm_bawt.service import preview_entrypoint as preview_module
from llm_bawt.service.preview_entrypoint import create_preview_app


@pytest.fixture(autouse=True)
def hide_pytest_process_marker(monkeypatch):
    """Pytest reinstates its own env marker after fixtures; omit it at startup."""
    original = preview_module._check_environment

    def check():
        with monkeypatch.context() as clean:
            clean.delenv("PYTEST_CURRENT_TEST", raising=False)
            return original()

    monkeypatch.setattr(preview_module, "_check_environment", check)


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    # The host has a Docker socket and repo .env; the preview container has neither.
    original_paths = preview_module._privileged_paths
    monkeypatch.setattr(preview_module, "_privileged_paths", lambda: tuple(
        path for path in original_paths()
        if path != preview_module.Path.cwd() / ".env"
    ))
    monkeypatch.setattr(preview_module, "_RUN_DIRECTORIES", (tmp_path,))
    for key in list(preview_module.os.environ):
        if key not in preview_module._ALLOWED_ENV:
            monkeypatch.delenv(key)
    monkeypatch.setenv("LLM_BAWT_PREVIEW_ENABLED", "1")
    monkeypatch.setenv("LLM_BAWT_POSTGRES_HOST", "preview-postgres")
    monkeypatch.setenv("LLM_BAWT_POSTGRES_DATABASE", "preview")
    monkeypatch.setenv("LLM_BAWT_POSTGRES_USER", "preview")
    monkeypatch.setenv("LLM_BAWT_REDIS_URL", "redis://preview-redis:6379/0")
    monkeypatch.setenv("LLM_BAWT_PREVIEW_AGENT_BACKEND", "claude-code")
    password_file = tmp_path / "preview-pg-secret"
    password_file.write_text("isolated-database-password\n", encoding="utf-8")
    monkeypatch.setenv("LLM_BAWT_POSTGRES_PASSWORD_FILE", str(password_file))
    return tmp_path


@pytest.fixture
def client(monkeypatch, isolated_env):
    service = SimpleNamespace(shutdown=AsyncMock())
    subscriber = SimpleNamespace(close=AsyncMock())

    async def start(app, config):
        assert config.POSTGRES_PASSWORD == "isolated-database-password"
        assert config.REDIS_URL == "redis://preview-redis:6379/0"
        assert config.MCP_SERVER_URL is None
        app.state.preview_ready = True
        app.state.preview_backend = "claude-code"
        return service, subscriber

    monkeypatch.setattr(preview_module, "_start_preview_runtime", start)
    with TestClient(create_preview_app()) as preview:
        yield preview
    subscriber.close.assert_awaited_once()
    service.shutdown.assert_awaited_once()
    assert "LLM_BAWT_POSTGRES_PASSWORD" not in preview_module.os.environ


def test_preview_health_identifies_only_a_ready_preview(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"service": "preview", "state": "ready", "preview": True}


def test_not_configured_process_does_not_claim_readiness():
    response = TestClient(create_preview_app()).get("/health")
    assert response.status_code == 503
    assert response.json() == {"service": "preview", "state": "not_configured", "preview": True}


@pytest.mark.parametrize("method,path", [
    ("get", "/v1/providers/claude/token"),
    ("get", "/v1/providers/openai_chatgpt/bundle"),
    ("post", "/v1/ops/jobs"),
    ("get", "/v1/bots"),
    ("post", "/v1/tasks"),
    ("get", "/mcp"),
])
def test_preview_does_not_register_production_routes(client, method, path):
    response = client.request(method, path)
    assert response.status_code == 404


def test_session_requires_exact_preview_marker_and_selected_bot(client):
    body = {"bot_id": "configured-bot", "user_id": "preview-user"}
    assert client.post("/v1/sessions", json=body).status_code == 403
    assert client.post("/v1/sessions", json=body,
                       headers={"X-BawtHub-Preview": "true"}).status_code == 403
    assert client.post("/v1/sessions", json={**body, "fork_from": {"session_id": "prod"}},
                       headers={"X-BawtHub-Preview": "1"}).status_code == 422


def test_isolated_session_and_streamed_chat_contract(client, monkeypatch):
    from llm_bawt.service import dependencies

    calls = []
    bot = SimpleNamespace(agent_backend="claude-code", endpoint_id=42)
    manager = SimpleNamespace(get_bot=lambda slug: bot if slug == "configured-bot" else None)

    class Storage:
        async def rotate_session(self, *, bot_id, user_id, working_memory_mode):
            calls.append(("rotate", bot_id, user_id, working_memory_mode))
            return "preview-session-uuid"

        async def get_session(self, session_id, *, bot_id):
            calls.append(("lookup", session_id, bot_id))
            return {"bot_id": bot_id, "user_id": "preview-user", "status": "active"}

    async def stream(_service, request, *, admission_comment, extensions):
        calls.append(("stream", request.bot_id, request.user, request.session_id,
                      request.augment_memory, request.extract_memory,
                      admission_comment, extensions))
        yield "data: [DONE]\n\n"

    with monkeypatch.context() as patch:
        patch.setattr(dependencies, "get_service", lambda: SimpleNamespace(config=object()))
        patch.setitem(sys.modules, "llm_bawt.bots", SimpleNamespace(BotManager=lambda _: manager))
        patch.setitem(sys.modules, "llm_bawt.mcp_server.storage",
                      SimpleNamespace(get_storage=Storage))
        patch.setitem(sys.modules, "llm_bawt.working_memory.session_state",
                      SimpleNamespace(mode_from_flag=lambda flag: "suppressed" if not flag else "pinned"))
        patch.setitem(sys.modules, "llm_bawt.service.chat_response_adapter",
                      SimpleNamespace(completion_stream=stream, completion_response=AsyncMock()))
        headers = {"X-BawtHub-Preview": "1", "X-LLM-Bawt-Extension-Events": "comment"}
        session = client.post("/v1/sessions", json={
            "bot_id": "configured-bot", "user_id": "preview-user",
        }, headers=headers)
        assert session.status_code == 200
        assert session.json()["session_id"] == "preview-session-uuid"
        result = client.post("/v1/chat/completions", json={
            "bot_id": "configured-bot", "user": "preview-user",
            "session_id": "preview-session-uuid", "stream": True,
            "messages": [{"role": "user", "content": "ping"}],
        }, headers=headers)
        assert result.status_code == 200
        assert result.text == "data: [DONE]\n\n"
    assert calls == [
        ("rotate", "configured-bot", "preview-user", "suppressed"),
        ("lookup", "preview-session-uuid", "configured-bot"),
        ("stream", "configured-bot", "preview-user", "preview-session-uuid",
         False, False, False, "comment"),
    ]


def test_chat_requires_exact_preview_marker_and_selected_bot(client):
    body = {"messages": [{"role": "user", "content": "ping"}], "stream": True}
    assert client.post("/v1/chat/completions", json=body).status_code == 403
    assert client.post("/v1/chat/completions", json=body,
                       headers={"X-BawtHub-Preview": "true"}).status_code == 403
    response = client.post("/v1/chat/completions", json=body,
                           headers={"X-BawtHub-Preview": "1"})
    assert response.status_code == 400
    assert "selected bot" in response.json()["detail"]


@pytest.mark.parametrize("authority", [
    {"inter_bot_delivery_id": "injected"},
    {"inter_bot_claim_token": "injected"},
    {"inter_bot_reset_only": True},
    {"parent_turn_id": "injected"},
])
def test_chat_rejects_delivery_authority(client, authority):
    body = {"messages": [{"role": "user", "content": "ping"}],
            "bot_id": "configured-bot", "user": "preview-user", **authority}
    response = client.post("/v1/chat/completions", json=body,
                           headers={"X-BawtHub-Preview": "1"})
    assert response.status_code == 403
    assert "authority" in response.json()["detail"]


def test_preview_refuses_to_start_without_explicit_role(monkeypatch):
    monkeypatch.delenv("LLM_BAWT_PREVIEW_ENABLED", raising=False)
    with pytest.raises(RuntimeError, match="explicit preview role"):
        with TestClient(create_preview_app()):
            pass


def test_preview_refuses_missing_isolation_config(monkeypatch, isolated_env):
    monkeypatch.delenv("LLM_BAWT_REDIS_URL")
    with pytest.raises(RuntimeError, match="LLM_BAWT_REDIS_URL"):
        with TestClient(create_preview_app()):
            pass


def test_preview_refuses_production_secrets(monkeypatch, isolated_env):
    monkeypatch.setenv("BRIDGE_BROKER_TOKEN_FILE", "/run/llm-bawt-bridge-tokens/broker")
    with pytest.raises(RuntimeError, match="BRIDGE_BROKER_TOKEN_FILE"):
        with TestClient(create_preview_app()):
            pass


@pytest.mark.parametrize("key", [
    "LLM_BAWT_UNREVIEWED_PROVIDER_SECRET", "XAI_API_KEY", "KIMI_API_KEY",
    "GITHUB_APP_PRIVATE_KEY_PATH", "PGPASSWORD", "UV_TOKEN", "EMPTY_UNKNOWN",
])
def test_preview_refuses_unclassified_environment(monkeypatch, isolated_env, key):
    monkeypatch.setenv(key, "" if key == "EMPTY_UNKNOWN" else "not-a-prod-secret")
    with pytest.raises(RuntimeError, match=key):
        with TestClient(create_preview_app()):
            pass


def test_preview_checks_dotenv_mount():
    assert preview_module.Path.cwd() / ".env" in preview_module._privileged_paths()


def test_preview_refuses_mounted_control_socket(monkeypatch, isolated_env):
    control_socket = socket.socket(socket.AF_UNIX)
    try:
        control_socket.bind(str(isolated_env / "host-control.sock"))
        assert preview_module._privileged_sockets() == (isolated_env / "host-control.sock",)
        with pytest.raises(RuntimeError, match="privileged mounts"):
            with TestClient(create_preview_app()):
                pass
    finally:
        control_socket.close()


def test_preview_refuses_privileged_mount(monkeypatch, isolated_env):
    (isolated_env / ".ssh").mkdir()
    with pytest.raises(RuntimeError, match="privileged mounts"):
        with TestClient(create_preview_app()):
            pass


def test_implicit_config_cannot_load_repo_dotenv(monkeypatch, isolated_env):
    from llm_bawt.utils.config import Config

    env_file = isolated_env / "unsafe.env"
    env_file.write_text("LLM_BAWT_DEFAULT_USER=prod-identity\n", encoding="utf-8")
    monkeypatch.setitem(Config.model_config, "env_file", env_file)

    async def start(app, config):
        assert Config().DEFAULT_USER is None  # callers that construct their own Config()
        assert config.DEFAULT_USER is None
        app.state.preview_ready = True
        return SimpleNamespace(shutdown=AsyncMock()), SimpleNamespace(close=AsyncMock())

    monkeypatch.setattr(preview_module, "_start_preview_runtime", start)
    with TestClient(create_preview_app()):
        assert Config.model_config["env_file"] is None
    assert Config.model_config["env_file"] == env_file


def test_startup_failure_removes_password_from_process(monkeypatch, isolated_env):
    async def fail(_app, _config):
        raise RuntimeError("isolated Redis unavailable")

    monkeypatch.setattr(preview_module, "_start_preview_runtime", fail)
    with pytest.raises(RuntimeError, match="isolated Redis unavailable"):
        with TestClient(create_preview_app()):
            pass
    assert "LLM_BAWT_POSTGRES_PASSWORD" not in preview_module.os.environ


@pytest.mark.parametrize("database,role,seed,bots,connected,construct_error,error", [
    (None, "preview", 1, True, True, False, "database unavailable"),
    ("wrong-db", "preview", 1, True, True, False, "database identity mismatch"),
    ("preview", "wrong-role", 1, True, True, False, "database identity mismatch"),
    ("preview", "preview", None, True, True, False, "seed is not complete"),
    ("preview", "preview", 1, False, True, False, "no seeded bot matches"),
    ("preview", "preview", 1, True, False, False, "Redis is unavailable"),
    ("preview", "preview", 1, True, True, True, "service init failed"),
    ("preview", "preview", 1, True, True, False, None),
])
def test_runtime_requires_seeded_db_matching_bot_and_redis(
    monkeypatch, database, role, seed, bots, connected, construct_error, error,
):
    """Exercise the actual startup admission checks, not a mocked lifespan."""
    engine = MagicMock()
    conn = engine.connect.return_value.__enter__.return_value
    conn.execute.side_effect = [
        SimpleNamespace(one=lambda: (database, role)),
        SimpleNamespace(scalar=lambda: seed),
    ]
    subscriber = SimpleNamespace(connect=AsyncMock(), close=AsyncMock(), connected=connected)
    service = SimpleNamespace(shutdown=AsyncMock())
    app = create_preview_app()
    config = SimpleNamespace(
        POSTGRES_DATABASE="preview", POSTGRES_USER="preview",
        REDIS_URL="redis://preview-redis:6379/0",
        install_model_catalog=MagicMock(),
    )
    selected = [SimpleNamespace(agent_backend="isolated-backend", endpoint_id=42)] if bots else []
    published = []
    monkeypatch.setenv("LLM_BAWT_PREVIEW_AGENT_BACKEND", "isolated-backend")
    with monkeypatch.context() as patch:
        from llm_bawt.utils import db

        patch.setattr(db, "get_shared_engine", lambda _: engine if database is not None else None)
        patch.setitem(sys.modules, "llm_bawt.bots", SimpleNamespace(
            BotManager=lambda _: SimpleNamespace(list_bots=lambda: selected),
        ))
        patch.setitem(sys.modules, "llm_bawt.model_catalog", SimpleNamespace(
            ModelCatalogStore=lambda _: SimpleNamespace(load=lambda: []),
        ))
        def start_service(_):
            if construct_error:
                raise RuntimeError("service init failed")
            return service

        patch.setitem(sys.modules, "llm_bawt.service.background_service", SimpleNamespace(
            BackgroundService=start_service,
        ))
        patch.setitem(sys.modules, "agent_bridge.subscriber", SimpleNamespace(
            RedisSubscriber=lambda _: subscriber,
        ))
        patch.setitem(sys.modules, "llm_bawt.agent_backends.agent_bridge", SimpleNamespace(
            set_agent_subscriber=lambda value: published.append(value),
        ))
        from llm_bawt.service import dependencies

        patch.setattr(dependencies, "set_service", lambda value: published.append(value))
        if error:
            with pytest.raises(RuntimeError, match=error):
                asyncio.run(preview_module._start_preview_runtime(app, config))
            assert app.state.preview_ready is False
        else:
            actual_service, actual_subscriber = asyncio.run(
                preview_module._start_preview_runtime(app, config)
            )
            assert (actual_service, actual_subscriber) == (service, subscriber)
            assert app.state.preview_ready is True
            assert published == [subscriber, service]
    if database == role == "preview" and seed == 1 and bots:
        subscriber.connect.assert_awaited_once()
        if not connected or construct_error:
            subscriber.close.assert_awaited_once()
        if construct_error:
            assert published == [None, None]
    else:
        subscriber.connect.assert_not_awaited()


@pytest.mark.real_process(reason="fresh Python process verifies the preview import boundary")
def test_import_does_not_import_production_app():
    result = subprocess.run(
        [sys.executable, "-c", "import sys, llm_bawt.service.preview_entrypoint; "
         "import llm_bawt.service.chat_response_adapter; "
         "assert 'llm_bawt.service.api' not in sys.modules; "
         "assert 'llm_bawt.service.routes' not in sys.modules"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
