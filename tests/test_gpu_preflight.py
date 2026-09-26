"""A handoff preflight reports missing evidence, never invents permission."""
import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx

from llm_bawt.media.gpu_preflight import GpuPreflight, MORPH_OPS


def test_missing_voice_credentials_and_ops_fail_closed(monkeypatch):
    monkeypatch.delenv("BAWTHUB_VOICE_ADMISSION_TOKEN", raising=False)
    monkeypatch.delenv("BAWTHUB_VOICE_ADMISSION_URL", raising=False)

    async def check():
        async with httpx.AsyncClient() as http:
            async def ops():
                return []
            return await GpuPreflight(http=http, ops=ops).assess(
                state=SimpleNamespace(owner="voice", phase="idle"),
                gpu={"ready": True, "observed_at": datetime.now(UTC).isoformat(),
                     "total_mib": 16303, "free_mib": 3819},
            )

    result = asyncio.run(check())
    assert result["switch_ready"] is False
    assert result["voice"] is None
    assert any("credentials" in issue for issue in result["issues"])
    assert all(any(slug in issue for issue in result["issues"]) for slug in MORPH_OPS)
    assert any("peak VRAM" in issue for issue in result["issues"])


def test_active_voice_and_stale_gpu_fail_closed(monkeypatch):
    monkeypatch.setenv("BAWTHUB_VOICE_ADMISSION_TOKEN", "a" * 32)
    monkeypatch.setenv("BAWTHUB_VOICE_ADMISSION_URL", "http://voice.example")

    async def check():
        def respond(request):
            assert request.headers["authorization"] == "Bearer " + "a" * 32
            return httpx.Response(200, json={"active_count": 1, "active_gpu_work": 0,
                                              "parked": False})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            async def ops():
                return [{"slug": slug, "enabled": True} for slug in MORPH_OPS]
            return await GpuPreflight(http=http, ops=ops).assess(
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
    monkeypatch.delenv("BAWTHUB_VOICE_ADMISSION_TOKEN", raising=False)

    async def check(gpu):
        async with httpx.AsyncClient() as http:
            async def ops():
                # Disabled operations and summaries without an explicit enabled
                # bit must both remain unavailable to a transition.
                return [{"slug": slug, "enabled": False} for slug in MORPH_OPS[:2]] + [
                    {"slug": slug} for slug in MORPH_OPS[2:]
                ]
            return await GpuPreflight(http=http, ops=ops).assess(
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
    monkeypatch.delenv("BAWTHUB_VOICE_ADMISSION_TOKEN", raising=False)

    async def check():
        async with httpx.AsyncClient() as http:
            async def ops():
                return {"not": "a list"}
            return await GpuPreflight(http=http, ops=ops).assess(
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
    monkeypatch.setenv("BAWTHUB_VOICE_ADMISSION_TOKEN", "a" * 32)
    monkeypatch.setenv("BAWTHUB_VOICE_ADMISSION_URL", "http://voice.example")

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"active_count": 0, "active_gpu_work": 0,
                                                       "parked": True})
        )) as http:
            async def ops():
                return [{"slug": slug, "enabled": True} for slug in MORPH_OPS]
            return await GpuPreflight(http=http, ops=ops).assess(
                state=SimpleNamespace(owner="voice", phase="idle"),
                gpu={"ready": True, "observed_at": datetime.now(UTC).isoformat(),
                     "total_mib": 16303, "free_mib": 14000},
                model={"installed": True, "resident": False}, active_video_jobs=0,
            )

    result = asyncio.run(check())
    assert result["switch_ready"] is False
    assert any("already parked" in issue for issue in result["issues"])


def test_uncertain_worker_blocks_handoff(monkeypatch):
    monkeypatch.delenv("BAWTHUB_VOICE_ADMISSION_TOKEN", raising=False)

    async def check():
        async with httpx.AsyncClient() as http:
            async def ops():
                return [{"slug": slug, "enabled": True} for slug in MORPH_OPS]
            return await GpuPreflight(http=http, ops=ops).assess(
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
    monkeypatch.setenv("BAWTHUB_VOICE_ADMISSION_TOKEN", "a" * 32)
    monkeypatch.setenv("BAWTHUB_VOICE_ADMISSION_URL", "http://voice.example")

    async def check():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"active_count": True, "active_gpu_work": 0,
                                                       "parked": False})
        )) as http:
            async def ops():
                return [{"slug": slug, "enabled": True} for slug in MORPH_OPS]
            return await GpuPreflight(http=http, ops=ops).assess(
                state=SimpleNamespace(owner="voice", phase="idle"), gpu={"ready": False},
            )

    result = asyncio.run(check())
    assert result["voice"] is None
    assert any("Voice admission unavailable" in issue for issue in result["issues"])
