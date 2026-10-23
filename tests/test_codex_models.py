import asyncio
from types import SimpleNamespace

from codex_bridge import __main__ as codex_bridge_main
from llm_bawt import model_manager
from llm_bawt.service.core import ServiceLLMBawt
from llm_bawt.service.routes.models import _public_model_type


def test_public_model_type_exposes_codex_group():
    info = {"type": "agent_backend", "backend": "codex", "model_id": "gpt-5.5"}
    assert _public_model_type(info) == "codex"


def test_fetch_codex_models_uses_live_catalog_without_fallback(monkeypatch):
    class Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "models": [
                    {
                        "id": "gpt-6-astra",
                        "description": "Astra",
                        "context_window": 1_100_000,
                    }
                ]
            }

    monkeypatch.setattr("httpx.get", lambda *_args, **_kwargs: Response())

    assert model_manager.fetch_codex_models() == (
        True,
        [
            {
                "id": "gpt-6-astra",
                "summary": "Astra",
                "context_length": 1_100_000,
            }
        ],
    )


def test_fetch_codex_models_fails_when_live_catalog_is_unavailable(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("bridge unavailable")

    monkeypatch.setattr("httpx.get", fail)

    assert model_manager.fetch_codex_models() == (False, [])


def test_fetch_codex_models_fails_when_live_catalog_is_empty(monkeypatch):
    class Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"models": []}

    monkeypatch.setattr("httpx.get", lambda *_args, **_kwargs: Response())

    assert model_manager.fetch_codex_models() == (False, [])


def test_codex_bridge_model_discovery_supplies_header_body(monkeypatch):
    observed = {}

    class Adapter:
        async def authorize(self):
            return "token", "https://chatgpt.example/backend-api/codex"

        def extra_headers(self, responses_body):
            observed["responses_body"] = responses_body
            return {"chatgpt-account-id": "acct"}

    class Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "models": [
                    {
                        "slug": "gpt-6-astra",
                        "description": "Astra",
                        "context_window": 1_100_000,
                    }
                ]
            }

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url, *, params, headers):
            observed.update(url=url, params=params, headers=headers)
            return Response()

    monkeypatch.setattr(
        "claude_code_bridge.proxy.adapters.openai_chatgpt.OpenAIChatGPTAdapter",
        Adapter,
    )
    monkeypatch.setattr("httpx.AsyncClient", Client)
    monkeypatch.setattr(codex_bridge_main, "_codex_client_version", lambda: _async_value("0.200.0"))

    assert asyncio.run(codex_bridge_main._fetch_codex_models()) == [
        {
            "id": "gpt-6-astra",
            "description": "Astra",
            "context_window": 1_100_000,
        }
    ]
    assert observed["responses_body"] == {}
    assert observed["params"] == {"client_version": "0.200.0"}
    assert observed["headers"] == {
        "Authorization": "Bearer token",
        "chatgpt-account-id": "acct",
    }


async def _async_value(value):
    return value


def test_parse_chatgpt_models_drops_hidden_and_slugless_rows():
    from agent_bridge.chatgpt_models import parse_chatgpt_models

    payload = {
        "models": [
            {"slug": "gpt-6-astra", "description": "Astra", "context_window": 1_100_000, "visibility": "list"},
            {"slug": "gpt-6-hidden", "context_window": 1, "visibility": "hide"},
            {"id": "gpt-5.4", "display_name": "GPT-5.4", "context_window": 400_000},
            {"description": "no slug"},
            "junk",
        ]
    }

    assert parse_chatgpt_models(payload) == [
        {"id": "gpt-6-astra", "context_window": 1_100_000, "description": "Astra"},
        {"id": "gpt-5.4", "context_window": 400_000, "description": "GPT-5.4"},
    ]
    assert parse_chatgpt_models({"models": "nope"}) == []
    assert parse_chatgpt_models(None) == []


def test_chatgpt_models_request_shape():
    from agent_bridge.chatgpt_models import chatgpt_models_request

    url, params, headers = chatgpt_models_request(
        base_url="https://chatgpt.com/backend-api/codex/",
        bearer="tok",
        account_id="acct",
        client_version="0.144.1",
    )

    assert url == "https://chatgpt.com/backend-api/codex/models"
    assert params == {"client_version": "0.144.1"}
    assert headers == {"Authorization": "Bearer tok", "chatgpt-account-id": "acct"}


def test_codex_bridge_fallback_version_is_the_shared_floor():
    from agent_bridge.chatgpt_models import CHATGPT_MODELS_CLIENT_VERSION

    # Pinned value: the codex CLI the agent-bridge image ships. 0.144.1 listed
    # only gpt-5.6-* and hid the gpt-6 family (TASK-1296 live check).
    assert codex_bridge_main._CODEX_CLIENT_VERSION_FALLBACK == CHATGPT_MODELS_CLIENT_VERSION == "0.157.1"


def test_service_bot_init_applies_codex_default_model(monkeypatch):
    bot = SimpleNamespace(
        agent_backend="codex",
        agent_backend_config={"session_key": "bot:user"},
        default_model="codex-gpt-5-5",
    )
    config = SimpleNamespace(
        defined_models={
            "models": {
                "codex-gpt-5-5": {
                    "type": "agent_backend",
                    "backend": "codex",
                    "model_id": "gpt-5.5",
                }
            }
        }
    )

    class DummyAgentBackendClient:
        def __init__(self):
            self._bot_config = {}

    def fake_init_bot(self, _config):
        self.bot = bot

    monkeypatch.setattr("llm_bawt.core.base.BaseLLMBawt._init_bot", fake_init_bot)
    monkeypatch.setattr(
        "llm_bawt.clients.agent_backend_client.AgentBackendClient",
        DummyAgentBackendClient,
    )

    service = ServiceLLMBawt.__new__(ServiceLLMBawt)
    service.client = DummyAgentBackendClient()
    service.bot_id = "builder"
    service.user_id = "nick"
    service.config = config

    ServiceLLMBawt._init_bot(service, config)

    assert service.client._bot_config["session_key"] == "bot:user"
    assert service.client._bot_config["bot_id"] == "builder"
    assert service.client._bot_config["user_id"] == "nick"
    assert service.client._bot_config["model"] == "gpt-5.5"
