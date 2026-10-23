"""Home audio contracts. No live devices, providers, or production DB writes."""
import asyncio
import io
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import text

from llm_bawt.integrations.home_audio import HomeAudioClient, HomeAudioSettings, wav_duration
from llm_bawt.integrations.home_audio_store import HomeAudioStore
from llm_bawt.integrations.home_audio_worker import HomeAudioWorker
from tests.sqlite_support import fast_sqlite_engine


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def store(tmp_path):
    return HomeAudioStore(fast_sqlite_engine(tmp_path / 'audio.db'))


def payload(text="hello", bot_id="loopy"):
    return dict(target="media_player.kitchen_display", bot_id=bot_id, text=text,
                voice="test-voice", asset_id=None, user_id="nick")


def wav(seconds=1):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(24000)
        out.writeframes(b"\0\0" * int(24000 * seconds))
    return buffer.getvalue()


def test_fifo_idempotency_and_bot_scopes(store):
    first = store.enqueue(payload(), "same")
    assert first["id"] == store.enqueue(payload(), "same")["id"]
    second = store.enqueue(payload(bot_id="nova"), "same")
    with pytest.raises(ValueError, match="Idempotency"):
        store.enqueue(payload("different"), "same")
    assert store.claim("worker1")["id"] == first["id"]
    assert store.claim("worker2") is None
    assert store.save("worker1", first["id"], "completed")
    assert store.claim("worker2")["id"] == second["id"]


def test_cancel_does_not_cancel_claimed_job(store):
    first = store.enqueue(payload(), "1")
    second = store.enqueue(payload(), "2")
    store.claim("worker")
    assert store.cancel(first["id"])["status"] == "preparing"
    assert store.cancel(second["id"])["status"] == "cancelled"


def test_expiry_and_crash_recovery_never_replay(store):
    first = store.enqueue(payload(), "1")
    store.claim("old")
    store.save("old", first["id"], "playing")
    expired = store.enqueue(payload(), "expired", ttl=-1)
    next_job = store.enqueue(payload(), "next")
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE home_audio_lane SET lease_until = 0"))
    assert store.claim("new") is None  # quarantine uncertain group playback
    assert store.get(first["id"])["status"] == "interrupted"
    assert store.claim("new") is None
    with store.engine.begin() as conn:
        conn.execute(text("UPDATE home_audio_lane SET lease_until = 0"))
    assert store.claim("new")["id"] == next_job["id"]
    assert store.get(expired["id"])["status"] == "expired"
    assert not store.renew("old")
    assert not store.save("old", next_job["id"], "completed")
    assert store.get(next_job["id"])["status"] == "preparing"


def test_persistence_across_store_instances(store):
    job = store.enqueue(payload(), "persist")
    other = HomeAudioStore(store.engine)
    assert other.claim("worker")["id"] == job["id"]


def test_wav_validation():
    assert wav_duration(wav()) == 1
    with pytest.raises(ValueError):
        wav_duration(b"not a WAV")
    with pytest.raises(ValueError, match="Truncated"):
        wav_duration(wav()[:-10])
    with pytest.raises(ValueError):
        wav_duration(wav(0))


def client(http):
    return HomeAudioClient(SimpleNamespace(HA_NATIVE_MCP_URL="http://ha/api/mcp", HA_NATIVE_MCP_TOKEN="test"),
                           HomeAudioSettings(), http)


@pytest.mark.anyio
async def test_actual_ha_play_media_contract_and_device_filter():
    calls = []
    def respond(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=[
                {"entity_id": "media_player.kitchen_display", "state": "off", "attributes": {"friendly_name": "Kitchen"}},
                {"entity_id": "media_player.unrelated_tv", "state": "on"},
            ])
        return httpx.Response(200, json=[])
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        service = client(http)
        assert len(await service.devices()) == 1
        await service.play("media_player.kitchen_display", "asset123")
        import json
        body = json.loads(calls[-1].content)
        assert calls[-1].url.path == "/api/services/media_player/play_media"
        assert body == {"entity_id": "media_player.kitchen_display",
                        "media_content_id": "http://10.0.0.101:8642/v1/uploads/asset123",
                        "media_content_type": "audio/wav"}
        with pytest.raises(ValueError, match="allowed"):
            await service.play("media_player.unrelated_tv", "asset123")


@pytest.mark.anyio
async def test_render_validates_audio_and_does_not_fallback():
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, content=wav())
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        assert await client(http).render("hello", "specific-voice") == wav()
    assert requests[0].url.path == "/api/v1/tts/render"
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503))) as http:
        with pytest.raises(httpx.HTTPStatusError):
            await client(http).render("hello", "specific-voice")


@pytest.mark.anyio
async def test_busy_device_waits_and_expires_without_playing(monkeypatch):
    import time
    service = client(None)
    service.state = AsyncMock(return_value={"state": "playing"})
    service.play = AsyncMock()
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)
    async def advance(_):
        nonlocal now
        now += 1
    monkeypatch.setattr(asyncio, "sleep", advance)
    with pytest.raises(TimeoutError):
        await service.wait_idle("media_player.kitchen_display", now + 2)
    service.play.assert_not_called()
    service.state = AsyncMock(return_value={"state": "unavailable"})
    with pytest.raises(RuntimeError, match="unavailable"):
        await service.wait_idle("media_player.kitchen_display", now + 2)


@pytest.mark.anyio
async def test_worker_failure_is_terminal_and_not_retried(store):
    job = store.enqueue(payload(), "fail")
    claimed = store.claim("owner")
    worker = HomeAudioWorker(None, store)
    worker._perform = AsyncMock(side_effect=RuntimeError("uncertain dispatch"))
    await worker._run_claim("owner", claimed)
    assert store.get(job["id"])["status"] == "failed"
    assert store.claim("next") is None
    worker._perform.assert_awaited_once()


@pytest.mark.anyio
async def test_worker_shutdown_preserves_uncertain_claim(store):
    job = store.enqueue(payload(), "shutdown")
    claimed = store.claim("owner")
    entered = asyncio.Event()
    async def perform(*args):
        entered.set()
        await asyncio.Event().wait()
    worker = HomeAudioWorker(None, store)
    worker._perform = perform
    task = asyncio.create_task(worker._run_claim("owner", claimed))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.get(job["id"])["status"] == "preparing"
    assert store.claim("next") is None


@pytest.mark.anyio
async def test_observed_playback_completion_and_early_stop(monkeypatch):
    service = client(None)
    loop = asyncio.get_running_loop()
    now = loop.time()
    monkeypatch.setattr(loop, "time", lambda: now)
    async def advance(seconds):
        nonlocal now
        now += seconds
    monkeypatch.setattr(asyncio, "sleep", advance)
    playing = {"state": "playing", "attributes": {"media_content_id": service.media_url("asset")}}
    idle = {"state": "idle", "attributes": {}}
    service.state = AsyncMock(side_effect=[playing, playing, idle])
    await service.wait_finished("media_player.kitchen_display", "asset", 1)
    service.state = AsyncMock(side_effect=[playing, idle])
    with pytest.raises(RuntimeError, match="early"):
        await service.wait_finished("media_player.kitchen_display", "asset", 10)
    service.state = AsyncMock(return_value=idle)
    with pytest.raises(TimeoutError, match="not confirmed"):
        await service.wait_finished("media_player.kitchen_display", "asset", 1)


@pytest.mark.anyio
async def test_complete_worker_path_stores_renders_and_plays_once(store, monkeypatch):
    import llm_bawt.media as media
    from unittest.mock import Mock
    job = store.enqueue(payload(), "success")
    claimed = store.claim("owner")
    fake_media = SimpleNamespace(upload=Mock(return_value=SimpleNamespace(id="asset")))
    monkeypatch.setattr(media, "get_media_store", lambda: fake_media)
    monkeypatch.setattr(HomeAudioSettings, "load", lambda _: HomeAudioSettings())
    monkeypatch.setattr(HomeAudioClient, "wait_idle", AsyncMock())
    monkeypatch.setattr(HomeAudioClient, "render", AsyncMock(return_value=wav()))
    play = AsyncMock()
    monkeypatch.setattr(HomeAudioClient, "play", play)
    monkeypatch.setattr(HomeAudioClient, "wait_finished", AsyncMock())
    worker = HomeAudioWorker(SimpleNamespace(HA_NATIVE_MCP_URL="http://ha/api/mcp", HA_NATIVE_MCP_TOKEN="test"), store)
    await worker._run_claim("owner", claimed)
    play.assert_awaited_once_with("media_player.kitchen_display", "asset", version=job["id"])
    assert store.get(job["id"])["status"] == "completed"
    assert store.get(job["id"])["asset_id"] == "asset"
    fake_media.upload.assert_called_once()
    assert fake_media.upload.call_args.kwargs["replace_key"] == "announcements/current.wav"


def test_named_write_requires_current_lane_owner(store):
    from unittest.mock import Mock
    store.enqueue(payload(), "owned")
    store.claim("owner")
    write = Mock(return_value="asset")
    with pytest.raises(RuntimeError, match="lease lost"):
        store.while_owned("stale-owner", write)
    write.assert_not_called()
    assert store.while_owned("owner", write) == "asset"
    write.assert_called_once()


def test_playback_versions_are_distinct():
    service = client(None)
    assert service.media_url("asset", "job1").endswith("/asset?v=job1")
    assert service.media_url("asset", "job1") != service.media_url("asset", "job2")
    with pytest.raises(ValueError):
        service.media_url("asset", "bad&query")


@pytest.mark.anyio
async def test_second_announcement_cannot_replace_while_first_is_playing(store, monkeypatch):
    import llm_bawt.media as media
    from unittest.mock import Mock
    first = store.enqueue(payload(), "first")
    second = store.enqueue(payload("second"), "second")
    fake_media = SimpleNamespace(upload=Mock(return_value=SimpleNamespace(id="shared-slot")))
    monkeypatch.setattr(media, "get_media_store", lambda: fake_media)
    monkeypatch.setattr(HomeAudioSettings, "load", lambda _: HomeAudioSettings())
    monkeypatch.setattr(HomeAudioClient, "wait_idle", AsyncMock())
    monkeypatch.setattr(HomeAudioClient, "render", AsyncMock(return_value=wav()))
    monkeypatch.setattr(HomeAudioClient, "play", AsyncMock())
    started = asyncio.Event()
    finish = asyncio.Event()
    async def wait_finished(*args, **kwargs):
        started.set()
        await finish.wait()
    monkeypatch.setattr(HomeAudioClient, "wait_finished", wait_finished)
    worker = HomeAudioWorker(SimpleNamespace(HA_NATIVE_MCP_URL="http://ha/api/mcp", HA_NATIVE_MCP_TOKEN="test"), store)
    work = asyncio.create_task(worker._run_claim("one", store.claim("one")))
    await started.wait()
    assert store.claim("two") is None
    assert fake_media.upload.call_count == 1
    finish.set()
    await work
    await worker._run_claim("two", store.claim("two"))
    assert fake_media.upload.call_count == 2
    assert store.get(first["id"])["asset_id"] == store.get(second["id"])["asset_id"] == "shared-slot"
    assert store.get(second["id"])["status"] == "completed"


@pytest.mark.anyio
async def test_uncertain_dispatch_holds_lane_before_failure(store, monkeypatch):
    job = store.enqueue(payload(), "uncertain")
    claimed = store.claim("owner")
    worker = HomeAudioWorker(None, store)
    async def perform(*args):
        store.save("owner", job["id"], "playing")
        raise TimeoutError("dispatch timed out")
    worker._perform = perform
    holds = []
    async def hold(seconds):
        holds.append(seconds)
        assert store.claim("other") is None
    # Avoid the real heartbeat/sleep while isolating quarantine behavior.
    async def heartbeat(*args):
        await asyncio.Event().wait()
    worker._heartbeat = heartbeat
    monkeypatch.setattr(asyncio, "sleep", hold)
    await worker._run_claim("owner", claimed)
    assert holds == [330]
    assert store.get(job["id"])["status"] == "failed"


@pytest.mark.anyio
async def test_enqueue_requires_explicit_target_and_exactly_one_input(monkeypatch):
    from llm_bawt.mcp_server import home_audio_tools as tools
    with pytest.raises(ValueError, match="exactly one"):
        await tools.home_audio_enqueue("media_player.kitchen_display", "loopy", "key")
    with pytest.raises(ValueError, match="exactly one"):
        await tools.home_audio_enqueue("media_player.kitchen_display", "loopy", "key", text="hi", asset_id="x")
    with pytest.raises(ValueError, match="blank"):
        await tools.home_audio_enqueue("media_player.kitchen_display", "loopy", "key", text=" ")
    monkeypatch.setattr(HomeAudioSettings, "load", lambda _: HomeAudioSettings())
    with pytest.raises(ValueError, match="not allowed"):
        await tools.home_audio_enqueue("media_player.unrelated", "loopy", "key", text="hi")
