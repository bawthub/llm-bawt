"""A handoff preflight reports missing evidence, never invents permission."""
import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest

from llm_bawt.integrations.home_audio import HomeAudioSettings
from llm_bawt.media.gpu_preflight import GpuPreflight, MORPH_OPS


@pytest.fixture(autouse=True)
def unavailable_voice_settings(monkeypatch):
    # Tests must never query the live settings DB or voice service.
    def unavailable(config):
        raise ValueError("Settings unavailable")
    monkeypatch.setattr("llm_bawt.integrations.home_audio.RuntimeSettingsStore", unavailable)


def test_voice_status_reuses_db_speech_address_without_auth(monkeypatch):
    config = object()

    class SettingsStore:
        def __init__(self, received_config):
            assert received_config is config

        def get_scope_settings(self, scope, scope_id):
            assert (scope, scope_id) == ("global", "*")
            return {"home_audio": {"tts_url": "http://voice.example/custom/api/"}}

    monkeypatch.setattr("llm_bawt.integrations.home_audio.RuntimeSettingsStore", SettingsStore)

    async def check():
        def respond(request):
            assert str(request.url) == "http://voice.example/custom/api/v1/internal/voice/sessions"
            assert "authorization" not in request.headers
            return httpx.Response(200, json={"active_count": 0, "active_gpu_work": 0, "parked": False})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            return await GpuPreflight(config=config, http=http, ops=None).voice_status()

    voice, error = asyncio.run(check())
    assert error is None
    assert voice == {"active_count": 0, "active_gpu_work": 0, "parked": False}


def test_unreachable_voice_backend_still_blocks_handoff(monkeypatch):
    monkeypatch.setattr(HomeAudioSettings, "load", lambda config: HomeAudioSettings())

    async def check():
        def unavailable(request):
            raise httpx.ConnectError("offline", request=request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as http:
            return await GpuPreflight(config=object(), http=http, ops=None).voice_status()

    voice, error = asyncio.run(check())
    assert voice is None
    assert error == "Voice admission unavailable (ConnectError)"


def test_missing_voice_settings_and_ops_fail_closed():

    async def check():
        async with httpx.AsyncClient() as http:
            async def ops():
                return []
            return await GpuPreflight(config=object(), http=http, ops=ops).assess(
                state=SimpleNamespace(owner="voice", phase="idle"),
                gpu={"ready": True, "observed_at": datetime.now(UTC).isoformat(),
                     "total_mib": 16303, "free_mib": 3819},
            )

    result = asyncio.run(check())
    assert result["switch_ready"] is False
    assert result["voice"] is None
    assert any("runtime settings unavailable" in issue for issue in result["issues"])
    assert all(any(slug in issue for issue in result["issues"]) for slug in MORPH_OPS)
    assert any("peak VRAM" in issue for issue in result["issues"])


def test_active_voice_and_stale_gpu_fail_closed(monkeypatch):
    monkeypatch.setattr(HomeAudioSettings, "load", lambda config: HomeAudioSettings(tts_url="http://voice.example/api"))

    async def check():
        def respond(request):
            assert "authorization" not in request.headers
            assert str(request.url) == "http://voice.example/api/v1/internal/voice/sessions"
            return httpx.Response(200, json={"active_count": 1, "active_gpu_work": 0,
                                              "parked": False})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            async def ops():
                return [{"slug": slug, "enabled": True} for slug in MORPH_OPS]
            return await GpuPreflight(config=object(), http=http, ops=ops).assess(
                state=SimpleNamespace(owner="voice", phase="idle"),
                gpu={"ready": True, "observed_at": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
                     "total_mib": 16303, "free_mib": 14000},
            )

    result = asyncio.run(check())
    assert result["switch_ready"] is False
    assert result["voice"]["active_count"] == 1
    assert any("Active voice" in issue for issue in result["issues"])
    assert any("stale" in issue for issue in result["issues"])


def test_malformed_telemetry_and_missing_enablement_fail_closed(monkeypatch):

    async def check(gpu):
        async with httpx.AsyncClient() as http:
            async def ops():
                # Disabled operations and summaries without an explicit enabled
                # bit must both remain unavailable to a transition.
                return [{"slug": slug, "enabled": False} for slug in MORPH_OPS[:2]] + [
                    {"slug": slug} for slug in MORPH_OPS[2:]
                ]
            return await GpuPreflight(config=object(), http=http, ops=ops).assess(
                state=SimpleNamespace(owner="voice", phase="idle"), gpu=gpu,
            )

    for gpu in (
        {"ready": True, "observed_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
         "total_mib": 16303, "free_mib": 1000},
        {"ready": True, "observed_at": datetime.now(UTC).isoformat(),
         "total_mib": 16303, "free_mib": True},
        {"ready": True, "observed_at": datetime.now(UTC).isoformat(),
         "total_mib": 0, "free_mib": 0},
        {"ready": True, "observed_at": datetime.now(UTC).isoformat(),
         "total_mib": 16303, "free_mib": float("nan")},
        {"ready": True, "observed_at": datetime.now(UTC).isoformat(),
         "total_mib": 16303, "free_mib": -1},
    ):
        result = asyncio.run(check(gpu))
        assert result["switch_ready"] is False
        assert any("GPU telemetry" in issue for issue in result["issues"])
        assert all(any(slug in issue for issue in result["issues"]) for slug in MORPH_OPS)


def test_malformed_external_payloads_fail_closed(monkeypatch):

    async def check():
        async with httpx.AsyncClient() as http:
            async def ops():
                return {"not": "a list"}
            return await GpuPreflight(config=object(), http=http, ops=ops).assess(
                state=SimpleNamespace(owner="unknown", phase="idle"),
                gpu="not telemetry", model=["not a model"], active_video_jobs=True,
            )

    result = asyncio.run(check())
    assert result["switch_ready"] is False
    assert any("Ops catalog unavailable" in issue for issue in result["issues"])
    assert any("Durable video job claims" in issue for issue in result["issues"])
    assert any("GPU driver telemetry unavailable" in issue for issue in result["issues"])
    assert any("Wan model installation" in issue for issue in result["issues"])


def test_parked_voice_requires_lease_reconciliation(monkeypatch):
    monkeypatch.setattr(HomeAudioSettings, "load", lambda config: HomeAudioSettings(tts_url="http://voice.example/api"))

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"active_count": 0, "active_gpu_work": 0,
                                                       "parked": True})
        )) as http:
            async def ops():
                return [{"slug": slug, "enabled": True} for slug in MORPH_OPS]
            return await GpuPreflight(config=object(), http=http, ops=ops).assess(
                state=SimpleNamespace(owner="voice", phase="idle"),
                gpu={"ready": True, "observed_at": datetime.now(UTC).isoformat(),
                     "total_mib": 16303, "free_mib": 14000},
                model={"installed": True, "resident": False}, active_video_jobs=0,
            )

    result = asyncio.run(check())
    assert result["switch_ready"] is False
    assert any("already parked" in issue for issue in result["issues"])


def test_uncertain_worker_blocks_handoff(monkeypatch):

    async def check():
        async with httpx.AsyncClient() as http:
            async def ops():
                return [{"slug": slug, "enabled": True} for slug in MORPH_OPS]
            return await GpuPreflight(config=object(), http=http, ops=ops).assess(
                state=SimpleNamespace(owner="video", phase="idle"),
                gpu={"ready": True, "observed_at": datetime.now(UTC).isoformat(),
                     "total_mib": 16303, "free_mib": 14000},
                model={"installed": True, "resident": False, "worker_uncertain": True},
                active_video_jobs=0,
            )

    result = asyncio.run(check())
    assert not result["switch_ready"]
    assert any("Wan worker outcome" in issue for issue in result["issues"])


def test_bad_voice_response_does_not_allow_switch(monkeypatch):
    monkeypatch.setattr(HomeAudioSettings, "load", lambda config: HomeAudioSettings(tts_url="http://voice.example/api"))

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"active_count": True, "active_gpu_work": 0,
                                                       "parked": False})
        )) as http:
            async def ops():
                return [{"slug": slug, "enabled": True} for slug in MORPH_OPS]
            return await GpuPreflight(config=object(), http=http, ops=ops).assess(
                state=SimpleNamespace(owner="voice", phase="idle"), gpu={"ready": False},
            )

    result = asyncio.run(check())
    assert result["voice"] is None
    assert any("Voice admission unavailable" in issue for issue in result["issues"])
