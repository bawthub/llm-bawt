"""Direct voice protocol changes are live DB decisions, not process restarts."""

import asyncio

import httpx
import pytest
from sqlalchemy import text

from llm_bawt.bawthub_auth import VOICE_PROXY_TOKEN_HEADER, service_token_source, voice_backend_auth
from llm_bawt.bawthub_auth.service_token import VOICE_ORIGINS_ENV
from llm_bawt.bawthub_auth.voice_credential import (
    VOICE_AUTH_MODE_KEY,
    VoiceCredentialUnavailable,
    derive_voice_credential,
    read_voice_auth_mode,
)
from llm_bawt.runtime_settings import RuntimeSettingsStore

TOKEN = "t" * 64
URL = "http://voice.example/v1/internal/voice/sessions"


@pytest.fixture
def store(private_app_pg, monkeypatch):
    store = RuntimeSettingsStore(private_app_pg)
    assert store.engine is not None
    monkeypatch.setattr(service_token_source(), "read", lambda: TOKEN)
    monkeypatch.setenv(VOICE_ORIGINS_ENV, "http://voice.example")
    try:
        yield store
    finally:
        store.delete_value("global", "*", VOICE_AUTH_MODE_KEY)
        store.delete_value("bot", "fixture", VOICE_AUTH_MODE_KEY)


def send():
    return next(voice_backend_auth().auth_flow(httpx.Request("GET", URL)))


def test_absent_global_row_stays_legacy_after_loading_new_source(store):
    store.set_value("bot", "fixture", VOICE_AUTH_MODE_KEY, "derived-v1")
    assert read_voice_auth_mode() == "legacy"
    assert send().headers[VOICE_PROXY_TOKEN_HEADER] == TOKEN


def test_same_auth_instance_observes_mode_flip_and_rollback(store):
    for mode, expected in (
        ("legacy", TOKEN), ("derived-v1", derive_voice_credential(TOKEN)), ("legacy", TOKEN),
    ):
        store.set_value("global", "*", VOICE_AUTH_MODE_KEY, mode)
        assert send().headers[VOICE_PROXY_TOKEN_HEADER] == expected


def test_same_async_client_observes_live_setting_without_process_restart(store):
    seen = []

    def respond(request):
        seen.append(request.headers[VOICE_PROXY_TOKEN_HEADER])
        return httpx.Response(200)

    async def run():
        async with httpx.AsyncClient(
            auth=voice_backend_auth(), transport=httpx.MockTransport(respond),
        ) as client:
            for mode in ("legacy", "derived-v1", "legacy"):
                store.set_value("global", "*", VOICE_AUTH_MODE_KEY, mode)
                assert (await client.get(URL)).status_code == 200

    asyncio.run(run())
    assert seen == [TOKEN, derive_voice_credential(TOKEN), TOKEN]


@pytest.mark.parametrize("value", [None, "", "unknown", True, ["legacy"], {"mode": "legacy"}])
def test_invalid_stored_mode_never_downgrades_to_raw(store, value):
    store.set_value("global", "*", VOICE_AUTH_MODE_KEY, value)
    with pytest.raises(VoiceCredentialUnavailable):
        send()


def test_malformed_json_never_counts_as_an_absent_setting(store):
    store.set_value("global", "*", VOICE_AUTH_MODE_KEY, "derived-v1")
    with store.engine.begin() as connection:
        connection.execute(
            text("UPDATE runtime_settings SET value_json = 'bad-json' WHERE key = :key"),
            {"key": VOICE_AUTH_MODE_KEY},
        )
    with pytest.raises(VoiceCredentialUnavailable):
        send()


@pytest.mark.parametrize("failure", ["missing", "exception"])
def test_db_unavailable_never_selects_legacy_or_leaves_a_header(store, monkeypatch, failure):
    from llm_bawt.utils import db

    def unavailable(_config):
        if failure == "exception":
            raise RuntimeError("unavailable")
        return None

    monkeypatch.setattr(db, "get_shared_engine", unavailable)
    request = httpx.Request("GET", URL, headers={VOICE_PROXY_TOKEN_HEADER: TOKEN})
    with pytest.raises(VoiceCredentialUnavailable):
        next(voice_backend_auth().auth_flow(request))
    assert VOICE_PROXY_TOKEN_HEADER not in request.headers
