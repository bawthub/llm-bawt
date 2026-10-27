"""Reddit and Talk provisioner credentials are record-first and fail-closed (TASK-1476).

Both live in an ``integration_connection:<id>`` record written by the
Integrations credentials form. Their legacy settings apply only while no record
exists, with one warning naming the settings and never a value. A record that
exists but can't be used never falls back to them.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from llm_bawt.search.base import SearchProvider
from llm_bawt.service import connection_secrets
from llm_bawt.service.connection_secrets import ConnectionSecret, ConnectionUnavailable
from tests.stored_provider_keys import real_credential_store

CLIENT_SECRET = "reddit-secret-do-not-echo"
TOKEN = "provisioner-token-do-not-echo"
STORED_REDDIT = {"client_id": "stored-id", "client_secret": CLIENT_SECRET}
STORED_TALK = {"url": "https://talk-provisioner.example", "token": TOKEN}
_TALK_ENV = ("LLM_BAWT_TALK_PROVISIONER_URL", "TALK_PROVISIONER_URL",
             "LLM_BAWT_TALK_PROVISIONER_TOKEN", "TALK_PROVISIONER_TOKEN")
CODES = ["not_connected", "unreadable", "store_unavailable"]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in _TALK_ENV:
        monkeypatch.delenv(name, raising=False)
    if hasattr(connection_secrets, "_legacy_warned"):
        monkeypatch.setattr(connection_secrets, "_legacy_warned", set())
    from llm_bawt.search import credentials
    monkeypatch.setattr(credentials, "_legacy_warned", set())


def _config(**fields):
    """A config with database credentials, so the record is consulted."""
    base = {"POSTGRES_PASSWORD": "synthetic", "BRAVE_API_KEY": "", "TAVILY_API_KEY": "",
            "NEWSAPI_API_KEY": "", "SEARCH_PROVIDER": None, "SEARCH_MAX_RESULTS": 5,
            "SEARCH_TIMEOUT": 10, "SEARCH_INCLUDE_ANSWER": False, "SEARCH_DEPTH": "basic",
            "BRAVE_SAFESEARCH": "moderate", "REDDIT_CLIENT_ID": "", "REDDIT_CLIENT_SECRET": "",
            "REDDIT_USER_AGENT": "llm-bawt-test/1.0", "TALK_PROVISIONER_URL": "http://localhost:8790",
            "TALK_PROVISIONER_TOKEN": None}
    return SimpleNamespace(**{**base, **fields})


def stub_integrations(monkeypatch, outcomes: dict[str, object]) -> list[str]:
    """Stub the strict reader for ``integration`` records and the provider keys.

    ``outcomes`` maps an id to its stored fields, ``None`` (no record) or a
    :class:`ConnectionUnavailable` to raise. Returns the integration ids read.
    """
    reads: list[str] = []

    def fake_read(_config, kind, connection_id, spec, **_kwargs):
        if kind == "provider":
            return None  # no Brave/Tavily key: Reddit is the only keyed search
        reads.append(connection_id)
        outcome = outcomes.get(connection_id)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            return None
        return ConnectionSecret(
            kind=kind, id=connection_id,
            meta={n: outcome[n] for n in (*spec.meta, *spec.meta_optional) if n in outcome},
            secret={n: outcome[n] for n in spec.secret},
        )

    monkeypatch.setattr("llm_bawt.service.connection_secrets.read_connection", fake_read)
    return reads


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def _every_search_available(monkeypatch):
    from llm_bawt.search import factory, multi

    monkeypatch.setattr(multi, "is_search_available", lambda provider=None: True)
    monkeypatch.setattr(factory, "is_search_available", lambda provider=None: True)


# --- definitions ------------------------------------------------------------------


def test_both_are_credential_integrations_llm_bawt_reads_in_process_only():
    from llm_bawt.integrations.credential_integrations import CREDENTIAL_INTEGRATIONS
    from llm_bawt.service.connection_secrets import BAWTHUB_READABLE, FieldSpec, integration_spec

    for integration_id in ("reddit", "talk-provisioner"):
        definition = CREDENTIAL_INTEGRATIONS[integration_id]
        assert definition.bawthub_readable is False
        assert ("integration", integration_id) not in BAWTHUB_READABLE
        # Legacy env applies while no record exists, so disconnect must leave one.
        assert definition.keeps_disconnected_record is True
    assert integration_spec("reddit") == FieldSpec(secret=("client_secret",), meta=("client_id",))
    assert integration_spec("talk-provisioner") == FieldSpec(secret=("token",), meta=("url",))


# --- Reddit -----------------------------------------------------------------------


def test_reddit_record_wins_over_the_legacy_settings(monkeypatch):
    from llm_bawt.search.credentials import reddit_credentials

    stub_integrations(monkeypatch, {"reddit": STORED_REDDIT})
    creds = reddit_credentials(_config(REDDIT_CLIENT_ID="legacy-id", REDDIT_CLIENT_SECRET="legacy-secret"))

    assert (creds.client_id, creds.client_secret, creds.problem) == ("stored-id", CLIENT_SECRET, None)
    assert CLIENT_SECRET not in repr(creds)


def test_reddit_legacy_settings_apply_only_without_a_record_and_warn_once(monkeypatch, caplog):
    from llm_bawt.search.credentials import reddit_credentials

    reads = stub_integrations(monkeypatch, {})
    config = _config(REDDIT_CLIENT_ID="legacy-id", REDDIT_CLIENT_SECRET=f" {CLIENT_SECRET} ")

    with caplog.at_level(logging.WARNING):
        first = reddit_credentials(config)
        second = reddit_credentials(config)

    assert (first.client_id, first.client_secret) == ("legacy-id", CLIENT_SECRET)
    assert first == second
    assert reads == ["reddit", "reddit"]
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "LLM_BAWT_REDDIT_CLIENT_SECRET" in warnings[0]
    assert CLIENT_SECRET not in warnings[0] and "legacy-id" not in warnings[0]


def test_reddit_half_configured_legacy_settings_are_not_credentials(monkeypatch):
    from llm_bawt.search.credentials import reddit_credentials

    stub_integrations(monkeypatch, {})

    assert not reddit_credentials(_config(REDDIT_CLIENT_ID="legacy-id"))


@pytest.mark.parametrize("code", CODES)
def test_an_unusable_reddit_record_never_falls_back_to_legacy(monkeypatch, caplog, code):
    from llm_bawt.search.credentials import reddit_credentials

    stub_integrations(monkeypatch, {"reddit": ConnectionUnavailable(code)})
    config = _config(REDDIT_CLIENT_ID="legacy-id", REDDIT_CLIENT_SECRET="legacy-secret")

    with caplog.at_level(logging.WARNING):
        first = reddit_credentials(config)
        second = reddit_credentials(config)

    assert not first and first.problem == second.problem == code
    assert first.client_secret is None
    assert len(_warnings(caplog)) == 1


def test_reddit_without_database_credentials_reads_only_legacy(monkeypatch):
    from llm_bawt.search.credentials import reddit_credentials

    reads = stub_integrations(monkeypatch, {"reddit": STORED_REDDIT})
    no_db = _config(POSTGRES_PASSWORD="", REDDIT_CLIENT_ID="legacy-id", REDDIT_CLIENT_SECRET="legacy-secret")
    duck_typed = SimpleNamespace(REDDIT_CLIENT_ID="legacy-id", REDDIT_CLIENT_SECRET="legacy-secret")

    assert reddit_credentials(no_db).client_id == "legacy-id"
    assert reddit_credentials(duck_typed).client_id == "legacy-id"
    assert reads == []


def test_search_uses_a_stored_reddit_record(monkeypatch):
    from llm_bawt.search import factory, multi
    from llm_bawt.search.reddit_client import RedditSearchClient

    _every_search_available(monkeypatch)
    stub_integrations(monkeypatch, {"reddit": STORED_REDDIT})
    config = _config()

    explicit = factory.get_search_client(config, provider=SearchProvider.REDDIT, fallback=False)
    auto = factory.get_search_client(config)

    assert isinstance(explicit, RedditSearchClient) and isinstance(auto, RedditSearchClient)
    assert (explicit._client_id, explicit._client_secret) == ("stored-id", CLIENT_SECRET)
    assert SearchProvider.REDDIT in multi.available_providers(config)


@pytest.mark.parametrize("code", CODES)
def test_search_skips_reddit_whose_record_is_unusable(monkeypatch, code):
    from llm_bawt.search import factory, multi
    from llm_bawt.search.reddit_client import RedditSearchClient

    _every_search_available(monkeypatch)
    stub_integrations(monkeypatch, {"reddit": ConnectionUnavailable(code)})
    config = _config(REDDIT_CLIENT_ID="legacy-id", REDDIT_CLIENT_SECRET="legacy-secret")

    explicit = factory.get_search_client(config, provider=SearchProvider.REDDIT, fallback=False)
    auto = factory.get_search_client(config)
    reason = factory.get_search_unavailable_reason(config, provider=SearchProvider.REDDIT)

    assert explicit is None and not isinstance(auto, RedditSearchClient)
    assert SearchProvider.REDDIT not in multi.available_providers(config)
    assert "Settings → Integrations" in reason or "credential store" in reason
    assert "legacy-secret" not in reason


def test_an_existing_executor_follows_a_reddit_disconnect_or_rotation(monkeypatch):
    # The Brave/Tavily slice's Codex r1 Blocker, for Reddit: a cached client
    # must not keep using credentials that were rotated or disconnected.
    from llm_bawt.search.ddgs_client import DuckDuckGoClient
    from llm_bawt.search.factory import get_search_client
    from llm_bawt.search.reddit_client import RedditSearchClient
    from llm_bawt.tools.executor import ToolExecutor
    from llm_bawt.tools.parser import ToolCall

    _every_search_available(monkeypatch)
    outcomes: dict[str, object] = {"reddit": STORED_REDDIT}
    stub_integrations(monkeypatch, outcomes)
    config = _config(SEARCH_PROVIDER="reddit")
    calls = []
    monkeypatch.setattr(RedditSearchClient, "search",
                        lambda self, query, **_kw: calls.append(self._client_secret) or [])
    monkeypatch.setattr(DuckDuckGoClient, "search", lambda self, query, **_kw: calls.append("ddg") or [])
    executor = ToolExecutor(user_id="synthetic", config=config,
                            search_client=get_search_client(config, provider=SearchProvider.REDDIT))

    def search():
        executor.execute(ToolCall(name="search", arguments={"query": "q"}, raw_text=""))

    search()
    outcomes["reddit"] = {"client_id": "stored-id", "client_secret": "rotated-secret"}
    search()
    outcomes["reddit"] = ConnectionUnavailable("not_connected")
    search()

    assert calls == [CLIENT_SECRET, "rotated-secret", "ddg"]


def _executor_searches(monkeypatch, config, initial):
    """An executor holding ``initial``; returns ``search()`` and the list of who served."""
    from llm_bawt.search.ddgs_client import DuckDuckGoClient
    from llm_bawt.search.reddit_client import RedditSearchClient
    from llm_bawt.tools.executor import ToolExecutor
    from llm_bawt.tools.parser import ToolCall

    calls: list[str] = []
    monkeypatch.setattr(RedditSearchClient, "search",
                        lambda self, query, **_kw: calls.append(self._client_secret) or [])
    monkeypatch.setattr(DuckDuckGoClient, "search", lambda self, query, **_kw: calls.append("ddg") or [])
    executor = ToolExecutor(user_id="synthetic", config=config, search_client=initial)

    def search():
        executor.execute(ToolCall(name="search", arguments={"query": "q"}, raw_text=""))

    return search, calls


@pytest.mark.parametrize("preferred", ["reddit", None])
def test_an_existing_executor_returns_to_reddit_when_it_is_reconnected(monkeypatch, preferred):
    # Loopy r1 B1: the DuckDuckGo stand-in was "always current", so a long-lived
    # executor kept serving DuckDuckGo after Reddit was connected again.
    from llm_bawt.search.factory import get_search_client

    _every_search_available(monkeypatch)
    outcomes: dict[str, object] = {"reddit": STORED_REDDIT}
    stub_integrations(monkeypatch, outcomes)
    config = _config(SEARCH_PROVIDER=preferred)
    search, calls = _executor_searches(monkeypatch, config, get_search_client(config))

    search()
    outcomes["reddit"] = ConnectionUnavailable("not_connected")
    search()
    outcomes["reddit"] = STORED_REDDIT
    search()

    assert calls == [CLIENT_SECRET, "ddg", CLIENT_SECRET]


def test_an_existing_executor_picks_up_a_first_reddit_connection(monkeypatch):
    from llm_bawt.search.factory import get_search_client

    _every_search_available(monkeypatch)
    outcomes: dict[str, object] = {}
    stub_integrations(monkeypatch, outcomes)
    config = _config()
    search, calls = _executor_searches(monkeypatch, config, get_search_client(config))

    search()
    outcomes["reddit"] = STORED_REDDIT
    search()

    assert calls == ["ddg", CLIENT_SECRET]


def test_an_existing_executor_follows_a_change_of_preferred_provider(monkeypatch):
    from llm_bawt.search.factory import get_search_client

    _every_search_available(monkeypatch)
    stub_integrations(monkeypatch, {"reddit": STORED_REDDIT})
    config = _config(SEARCH_PROVIDER="reddit")
    search, calls = _executor_searches(monkeypatch, config, get_search_client(config))

    search()
    config.SEARCH_PROVIDER = "duckduckgo"
    search()
    config.SEARCH_PROVIDER = "reddit"
    search()

    assert calls == [CLIENT_SECRET, "ddg", CLIENT_SECRET]


def test_a_real_reddit_record_is_read_and_a_disconnect_turns_it_off(monkeypatch):
    from llm_bawt.integrations.credential_integrations import CredentialIntegrations
    from llm_bawt.search.credentials import reddit_credentials

    config = _config(REDDIT_CLIENT_ID="legacy-id", REDDIT_CLIENT_SECRET="legacy-secret")
    with real_credential_store(monkeypatch) as (_store, rotate_root):
        manager = CredentialIntegrations(SimpleNamespace())
        manager.configure("reddit", STORED_REDDIT)
        assert reddit_credentials(config).client_secret == CLIENT_SECRET

        assert manager.disconnect("reddit") is True
        assert reddit_credentials(config).problem == "not_connected"

        manager.configure("reddit", STORED_REDDIT)
        rotate_root()
        assert reddit_credentials(config).problem == "unreadable"


# --- Talk provisioner -------------------------------------------------------------


def test_talk_record_wins_over_env_and_config(monkeypatch):
    from llm_bawt.integrations.nextcloud.provisioner import get_provisioner_client

    monkeypatch.setenv("LLM_BAWT_TALK_PROVISIONER_TOKEN", "env-token")
    stub_integrations(monkeypatch, {"talk-provisioner": STORED_TALK})

    client = get_provisioner_client(_config(TALK_PROVISIONER_TOKEN="config-token"))

    assert (client.base_url, client.token) == ("https://talk-provisioner.example", TOKEN)


def test_talk_token_is_read_per_call_not_at_import(monkeypatch):
    # The module used to read its token once at import, so a token set (or a
    # record saved) later was never seen.
    from llm_bawt.integrations.nextcloud.provisioner import get_provisioner_client

    outcomes: dict[str, object] = {}
    stub_integrations(monkeypatch, outcomes)
    config = _config()
    monkeypatch.setenv("TALK_PROVISIONER_TOKEN", "env-token")
    assert get_provisioner_client(config).token == "env-token"

    outcomes["talk-provisioner"] = STORED_TALK
    assert get_provisioner_client(config).token == TOKEN


def test_talk_legacy_settings_warn_once_naming_them_only(monkeypatch, caplog):
    from llm_bawt.integrations.nextcloud.provisioner import get_provisioner_client

    stub_integrations(monkeypatch, {})
    monkeypatch.setenv("LLM_BAWT_TALK_PROVISIONER_TOKEN", TOKEN)

    with caplog.at_level(logging.WARNING):
        first = get_provisioner_client(_config())
        second = get_provisioner_client(_config())

    assert first.token == second.token == TOKEN
    assert first.base_url == "http://localhost:8790"
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "LLM_BAWT_TALK_PROVISIONER_TOKEN" in warnings[0] and TOKEN not in warnings[0]


@pytest.mark.parametrize("code", CODES)
def test_an_unusable_talk_record_never_falls_back_to_env(monkeypatch, code):
    from llm_bawt.integrations.nextcloud.provisioner import get_provisioner_client

    monkeypatch.setenv("LLM_BAWT_TALK_PROVISIONER_TOKEN", "env-token")
    stub_integrations(monkeypatch, {"talk-provisioner": ConnectionUnavailable(code)})

    with pytest.raises(ValueError) as raised:
        get_provisioner_client(_config(TALK_PROVISIONER_TOKEN="config-token"))

    assert raised.value.code == code
    assert "env-token" not in str(raised.value) and "config-token" not in str(raised.value)


def test_talk_without_record_or_token_is_not_configured(monkeypatch):
    from llm_bawt.integrations.nextcloud.provisioner import get_provisioner_client

    stub_integrations(monkeypatch, {})

    with pytest.raises(ValueError):
        get_provisioner_client(_config())


@pytest.mark.parametrize("code", CODES)
def test_provision_route_reports_an_unusable_record_as_400(monkeypatch, code):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from llm_bawt.integrations.nextcloud import manager as nextcloud_manager
    from llm_bawt.service.routes import nextcloud as routes

    monkeypatch.setenv("LLM_BAWT_TALK_PROVISIONER_TOKEN", "env-token")
    stub_integrations(monkeypatch, {"talk-provisioner": ConnectionUnavailable(code)})
    monkeypatch.setattr(routes, "get_service", lambda: SimpleNamespace(config=_config()))
    monkeypatch.setattr(nextcloud_manager, "get_nextcloud_manager",
                        lambda: SimpleNamespace(get_bot=lambda _bot: None))
    app = FastAPI()
    app.include_router(routes.router)

    response = TestClient(app).post("/admin/nextcloud-talk/provision", json={"bot_id": "synthetic"})

    assert response.status_code == 400
    assert code in response.json()["detail"]
    assert "env-token" not in response.text


def test_a_real_talk_record_is_read_and_a_disconnect_turns_it_off(monkeypatch):
    from llm_bawt.integrations.credential_integrations import CredentialIntegrations
    from llm_bawt.integrations.nextcloud.provisioner import get_provisioner_client

    monkeypatch.setenv("LLM_BAWT_TALK_PROVISIONER_TOKEN", "env-token")
    with real_credential_store(monkeypatch) as _handles:
        manager = CredentialIntegrations(SimpleNamespace())
        manager.configure("talk-provisioner", STORED_TALK)
        assert get_provisioner_client(_config()).token == TOKEN

        assert manager.disconnect("talk-provisioner") is True
        with pytest.raises(ValueError) as raised:
            get_provisioner_client(_config())
        assert raised.value.code == "not_connected"
