"""Direct voice clients support the gated legacy and derived protocols.

Every consumer uses the same auth helper; a live DB setting coordinates the
paired backend/HA migration without tying activation to an unrelated restart.
"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from llm_bawt.bawthub_auth import VOICE_PROXY_TOKEN_HEADER, service_token_source, voice_backend_auth
from llm_bawt.bawthub_auth.service_token import VOICE_ORIGINS_ENV, trusted_voice_origins
from llm_bawt.bawthub_auth.voice_credential import derive_voice_credential
from llm_bawt.integrations.home_audio import HomeAudioClient, HomeAudioSettings
from llm_bawt.media.gpu_handoff_adapter import LocalHandoffAdapter
from llm_bawt.media.gpu_preflight import GpuPreflight

TOKEN = "t" * 64
SETTINGS = HomeAudioSettings(tts_url="http://voice.example/api", media_base_url="http://media.example",
                             targets=("media_player.test",))


@pytest.fixture(params=["legacy", "derived-v1"])
def token(monkeypatch, request):
    monkeypatch.setattr(service_token_source(), "read", lambda: TOKEN)
    monkeypatch.setenv(VOICE_ORIGINS_ENV, "http://voice.example")
    monkeypatch.setattr(
        "llm_bawt.bawthub_auth.service_token.read_voice_auth_mode", lambda: request.param,
    )
    return TOKEN if request.param == "legacy" else derive_voice_credential(TOKEN)


@pytest.fixture
def voice_settings(monkeypatch):
    monkeypatch.setattr(HomeAudioSettings, "load", classmethod(lambda cls, config: SETTINGS))


def _send(request: httpx.Request) -> httpx.Request:
    flow = voice_backend_auth().auth_flow(request)
    return next(flow)


def test_adds_the_proxy_header_and_no_bearer(token):
    sent = _send(httpx.Request("GET", "http://voice.example/v1/internal/voice/sessions"))
    assert sent.headers[VOICE_PROXY_TOKEN_HEADER] == token
    assert "authorization" not in sent.headers


@pytest.mark.parametrize("url", [
    "http://receiver.example/v1/internal/voice/sessions",   # another host
    "https://voice.example/v1/voices",                      # another scheme
    "http://voice.example:8080/v1/voices",                  # another port
    "http://voice.example.evil/v1/voices",
])
def test_untrusted_destination_gets_no_token(token, url):
    assert VOICE_PROXY_TOKEN_HEADER not in _send(httpx.Request("GET", url)).headers


def test_no_declared_origin_sends_nothing(monkeypatch):
    monkeypatch.setattr(service_token_source(), "read", lambda: TOKEN)
    monkeypatch.delenv(VOICE_ORIGINS_ENV, raising=False)
    sent = _send(httpx.Request("GET", "http://voice.example/v1/voices"))
    assert VOICE_PROXY_TOKEN_HEADER not in sent.headers


def test_origin_parsing_normalizes_and_skips_junk():
    origins = trusted_voice_origins({VOICE_ORIGINS_ENV: " http://Voice.Example/api , ftp://x, ,https://a.b:8443"})
    assert origins == {("http", "voice.example", 80), ("https", "a.b", 8443)}


def test_anonymous_destination_edit_cannot_capture_the_token(token, monkeypatch):
    """Codex's TASK-2235 reproducer: an unauthenticated settings write moves
    home_audio.tts_url; the next preflight must not carry the token there."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from llm_bawt.integrations import home_audio
    from llm_bawt.service.routes import settings

    values = {}
    store = SimpleNamespace(
        engine=object(),
        set_value=lambda scope_type, scope_id, key, value: values.update({key: value}),
        get_scope_settings=lambda scope_type, scope_id: values,
    )
    monkeypatch.setattr(settings, "get_service", lambda: SimpleNamespace(config=object(), _default_bot="fixture"))
    monkeypatch.setattr(settings, "get_runtime_settings_store", lambda config: store)
    monkeypatch.setattr(settings, "_invalidate_all_instance_cache", lambda service: None)
    monkeypatch.setattr(home_audio, "RuntimeSettingsStore", lambda config: store)
    app = FastAPI()
    app.include_router(settings.router)
    response = TestClient(app).put("/v1/settings", json={
        "scope_type": "global", "key": "home_audio",
        "value": {"tts_url": "http://receiver.example", "media_base_url": "http://media.example",
                  "targets": ["media_player.test"]},
    })
    assert response.status_code == 200
    seen = []

    async def run():
        async with httpx.AsyncClient(transport=_recording(seen, SESSIONS)) as http:
            return await GpuPreflight(config=object(), http=http, ops=None).voice_status()

    voice, error = asyncio.run(run())
    assert error is None
    assert seen[0].url.host == "receiver.example"
    assert VOICE_PROXY_TOKEN_HEADER not in seen[0].headers


def test_no_token_sends_no_header():
    # conftest refuses derivation, so no token is in effect.
    sent = _send(httpx.Request("GET", "http://voice.example/v1/voices"))
    assert VOICE_PROXY_TOKEN_HEADER not in sent.headers


def _recording(seen, body):
    def respond(request):
        seen.append(request)
        return httpx.Response(200, **body)
    return httpx.MockTransport(respond)


SESSIONS = {"json": {"active_count": 0, "active_gpu_work": 0, "parked": False}}


def test_gpu_preflight_sends_the_token(token, voice_settings):
    seen = []

    async def run():
        async with httpx.AsyncClient(transport=_recording(seen, SESSIONS)) as http:
            return await GpuPreflight(config=object(), http=http, ops=None).voice_status()

    voice, error = asyncio.run(run())
    assert error is None and voice["active_count"] == 0
    assert seen[0].headers[VOICE_PROXY_TOKEN_HEADER] == token


def test_handoff_adapter_sends_the_token_on_sessions_park_and_unpark(token, voice_settings):
    seen = []

    def respond(request):
        seen.append(request)
        if request.url.path.endswith("/park"):
            return httpx.Response(200, json={"lease_id": "lease"})
        return httpx.Response(200, json=SESSIONS["json"])

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            async def none():
                return None

            video = SimpleNamespace(gpu_telemetry=none, model_status=none)
            adapter = LocalHandoffAdapter(config=object(), http=http, video=video, ops=None, policies=None)
            adapter._services = none
            assert (await adapter.observe())["voice"] == SESSIONS["json"]
            assert await adapter.park() == "lease"
            await adapter.unpark("lease")

    asyncio.run(run())
    paths = [r.url.path for r in seen]
    assert paths == ["/api/v1/internal/voice/sessions", "/api/v1/internal/voice/park",
                     "/api/v1/internal/voice/unpark"]
    assert all(r.headers[VOICE_PROXY_TOKEN_HEADER] == token for r in seen)


def test_home_audio_render_sends_the_token(token):
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\0\0" * 800)
    seen = []

    async def run():
        async with httpx.AsyncClient(transport=_recording(seen, {"content": buf.getvalue()})) as http:
            return await HomeAudioClient(SimpleNamespace(), SETTINGS, http).render("hi", "voice")

    assert asyncio.run(run()) == buf.getvalue()
    assert seen[0].url.path == "/api/v1/tts/render"
    assert seen[0].headers[VOICE_PROXY_TOKEN_HEADER] == token


def test_video_bridge_lease_check_uses_selected_credential(token, voice_settings, monkeypatch):
    from local_model_bridge import video_server

    seen = []
    transport = _recording(seen, {"json": {
        "parked": True, "park_id": "lease", "active_count": 0, "active_gpu_work": 0,
    }})
    client_type = httpx.AsyncClient
    monkeypatch.setattr(video_server.httpx, "AsyncClient", lambda: client_type(transport=transport))
    asyncio.run(video_server.verify_voice_park(SimpleNamespace(voice_lease=lambda: "lease")))
    assert seen[0].url.path == "/api/v1/internal/voice/sessions"
    assert seen[0].headers[VOICE_PROXY_TOKEN_HEADER] == token


def test_unknown_destination_strips_a_preexisting_voice_header(token):
    request = httpx.Request("GET", "http://receiver.example/v1/voices",
                            headers={VOICE_PROXY_TOKEN_HEADER: TOKEN})
    assert VOICE_PROXY_TOKEN_HEADER not in _send(request).headers


def test_seed_rotation_is_seen_by_the_same_auth_instance(token, monkeypatch):
    monkeypatch.setattr(service_token_source(), "read", lambda: "r" * 64)
    sent = _send(httpx.Request("GET", "http://voice.example/v1/voices"))
    expected = "r" * 64 if token == TOKEN else derive_voice_credential("r" * 64)
    assert sent.headers[VOICE_PROXY_TOKEN_HEADER] == expected


def test_bawthub_shared_hmac_vector():
    # Also pinned in bawthub tests/test_voice_proxy_auth.py and voiceProxyToken.test.ts.
    vector = "dd8e93cebe290185616abf0205545265abc5fdbdc317e35b628c799b518aa242"
    assert derive_voice_credential("v" * 40) == vector
    assert derive_voice_credential("  " + "v" * 40 + "\n") == vector


@pytest.mark.parametrize("seed", ["", "short", " " * 40])
def test_derived_credential_refuses_an_unusable_seed(seed):
    from llm_bawt.bawthub_auth.voice_credential import VoiceCredentialUnavailable

    with pytest.raises(VoiceCredentialUnavailable):
        derive_voice_credential(seed)
