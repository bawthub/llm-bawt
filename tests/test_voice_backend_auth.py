"""llm-bawt's direct voice-backend calls carry the voice proxy token (TASK-2235).

The voice backend can require the service token in ``x-bawthub-voice-proxy-token``
on every route except health. Each llm-bawt caller must send it, or enforcing
it would break GPU handoff, video preflight and home-audio speech.
"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from llm_bawt.bawthub_auth import VOICE_PROXY_TOKEN_HEADER, service_token_source, voice_backend_auth
from llm_bawt.integrations.home_audio import HomeAudioClient, HomeAudioSettings
from llm_bawt.media.gpu_handoff_adapter import LocalHandoffAdapter
from llm_bawt.media.gpu_preflight import GpuPreflight

TOKEN = "t" * 64
SETTINGS = HomeAudioSettings(tts_url="http://voice.example/api", media_base_url="http://media.example",
                             targets=("media_player.test",))


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setattr(service_token_source(), "read", lambda: TOKEN)


@pytest.fixture
def voice_settings(monkeypatch):
    monkeypatch.setattr(HomeAudioSettings, "load", classmethod(lambda cls, config: SETTINGS))


def _send(request: httpx.Request) -> httpx.Request:
    flow = voice_backend_auth().auth_flow(request)
    return next(flow)


def test_adds_the_proxy_header_and_no_bearer(token):
    sent = _send(httpx.Request("GET", "http://voice.example/v1/internal/voice/sessions"))
    assert sent.headers[VOICE_PROXY_TOKEN_HEADER] == TOKEN
    assert "authorization" not in sent.headers


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
    assert seen[0].headers[VOICE_PROXY_TOKEN_HEADER] == TOKEN


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
    assert all(r.headers[VOICE_PROXY_TOKEN_HEADER] == TOKEN for r in seen)


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
    assert seen[0].headers[VOICE_PROXY_TOKEN_HEADER] == TOKEN
