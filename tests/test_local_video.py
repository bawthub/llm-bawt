from __future__ import annotations

import asyncio
import base64

import httpx
import pytest
from fastapi import HTTPException

from llm_bawt.media.clients.local_video import LocalVideoClient
from llm_bawt.media.clients.registry import media_provider_registry
from llm_bawt.media.generation_service import MediaGenerationService
from local_model_bridge.video_models import VideoModelManager
from local_model_bridge import video_server
from local_model_bridge.video_server import VideoJobs, VideoRequest
from local_model_bridge.video_worker import dimensions, load_source_image


def test_provider_is_visible_without_changing_default() -> None:
    capability = media_provider_registry.capabilities("local-video")
    assert capability.media_types == ("video",)
    assert capability.default_models["video"] == "wan2.2-ti2v-5b"
    assert MediaGenerationService().resolve_request(
        provider="local-video", media_type="video", model=None,
        aspect_ratio=None, resolution=None,
    ) == ("local-video", "wan2.2-ti2v-5b", "16:9", "480p")
    assert media_provider_registry.canonical_provider(None) == "grok"


def test_video_dimensions_and_image_input_are_bounded() -> None:
    assert dimensions("720p", "16:9") == (1280, 704)
    with pytest.raises(ValueError, match="Unsupported"):
        dimensions("4k", "16:9")
    with pytest.raises(ValueError, match="embedded image"):
        load_source_image("https://example.com/private")
    with pytest.raises(Exception):
        load_source_image("data:image/png;base64," + base64.b64encode(b"not an image").decode())


def test_video_jobs_reject_invalid_size_without_starting_process(tmp_path) -> None:
    jobs = VideoJobs(tmp_path)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(jobs.submit(VideoRequest(prompt="bird", resolution="4k")))
    assert exc.value.status_code == 400
    assert not list(tmp_path.iterdir())


def test_direct_video_request_requires_durable_owner(monkeypatch, tmp_path) -> None:
    class DeniedLedger:
        def __init__(self, engine):
            pass

        def claim_video(self, job_id):
            from llm_bawt.media.gpu_handoff_store import HandoffConflict
            raise HandoffConflict("GPU does not belong to video")

    monkeypatch.setattr(video_server.models, "installed", lambda: True)
    monkeypatch.setattr(video_server.models, "status", lambda: {"worker_ready": True})
    monkeypatch.setattr(video_server, "GpuHandoffStore", DeniedLedger)
    monkeypatch.setattr(video_server, "get_shared_engine", lambda config: object())
    jobs = VideoJobs(tmp_path)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(jobs.submit(VideoRequest(prompt="bird")))
    assert exc.value.status_code == 409
    assert not jobs.worker.running
    assert not list(tmp_path.iterdir())


def test_uncertain_spawn_keeps_video_claim_and_requires_recovery(monkeypatch, tmp_path) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from llm_bawt.media.gpu_handoff_store import GpuHandoffStore

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ledger = GpuHandoffStore(engine)
    token, offered = ledger.offer(user="nick", target="video", actions=("verify",), expected_generation=0)
    transition = ledger.confirm(user="nick", token=token, expected_generation=offered.generation)
    ledger.begin_action(generation=transition.generation, action="verify")
    ledger.finish_action(generation=transition.generation, action="verify")
    ledger.complete(generation=transition.generation, target="video")

    async def uncertain_spawn(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(video_server.models, "installed", lambda: True)
    monkeypatch.setattr(video_server.models, "status", lambda: {"worker_ready": True})
    monkeypatch.setattr(video_server, "get_shared_engine", lambda config: engine)
    monkeypatch.setattr(video_server.asyncio, "create_subprocess_exec", uncertain_spawn)
    async def exercise():
        jobs = VideoJobs(tmp_path)
        await jobs.submit(VideoRequest(prompt="bird"))
        with pytest.raises(asyncio.CancelledError):
            await jobs._drain_task
        assert jobs.status(next(iter(jobs.jobs)))["status"] == "failed"

    asyncio.run(exercise())
    assert ledger.status().phase == "recovery_required"
    with pytest.raises(Exception, match="does not belong"):
        ledger.claim_video("next-render")
    engine.dispose()


def test_queued_video_jobs_share_worker_and_release_claims(monkeypatch, tmp_path) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from llm_bawt.media.gpu_handoff_store import GpuHandoffStore

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ledger = GpuHandoffStore(engine)
    token, offered = ledger.offer(user="nick", target="video", actions=("verify",), expected_generation=0)
    transition = ledger.confirm(user="nick", token=token, expected_generation=offered.generation)
    ledger.begin_action(generation=transition.generation, action="verify")
    ledger.finish_action(generation=transition.generation, action="verify")
    ledger.complete(generation=transition.generation, target="video")
    monkeypatch.setattr(video_server.models, "installed", lambda: True)
    monkeypatch.setattr(video_server.models, "status", lambda: {"worker_ready": True})
    monkeypatch.setattr(video_server, "get_shared_engine", lambda config: engine)
    started = asyncio.Event()
    proceed = asyncio.Event()
    calls = []

    class Worker:
        uncertain = False
        resident = True
        running = True

        async def render(self, job, output):
            calls.append(output.name)
            if len(calls) == 1:
                started.set()
                await proceed.wait()
            output.write_bytes(b"video")
            return {"width": 832}

    async def exercise():
        jobs = VideoJobs(tmp_path)
        jobs.worker = Worker()
        first = await jobs.submit(VideoRequest(prompt="first"))
        await started.wait()
        second = await jobs.submit(VideoRequest(prompt="second"))
        assert ledger.active_video_jobs() == 2
        assert jobs.status(second["id"])["status"] == "pending"
        proceed.set()
        await jobs._drain_task
        assert jobs.status(first["id"])["status"] == "completed"
        assert jobs.status(second["id"])["status"] == "completed"
        assert len(calls) == 2
        assert ledger.active_video_jobs() == 0

    asyncio.run(exercise())
    engine.dispose()


def test_video_queue_limit_rejects_without_claim(monkeypatch, tmp_path) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from llm_bawt.media.gpu_handoff_store import GpuHandoffStore

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ledger = GpuHandoffStore(engine)
    token, offered = ledger.offer(user="nick", target="video", actions=("verify",), expected_generation=0)
    transition = ledger.confirm(user="nick", token=token, expected_generation=offered.generation)
    ledger.begin_action(generation=transition.generation, action="verify")
    ledger.finish_action(generation=transition.generation, action="verify")
    ledger.complete(generation=transition.generation, target="video")
    monkeypatch.setattr(video_server.models, "installed", lambda: True)
    monkeypatch.setattr(video_server.models, "status", lambda: {"worker_ready": True})
    monkeypatch.setattr(video_server, "get_shared_engine", lambda config: engine)
    monkeypatch.setattr(video_server, "MAX_PENDING_VIDEO_JOBS", 1)
    proceed = asyncio.Event()
    started = asyncio.Event()

    class Worker:
        uncertain = False
        resident = False
        running = False

        async def render(self, job, output):
            started.set()
            await proceed.wait()
            output.write_bytes(b"video")
            return {"width": 832}

    async def exercise():
        jobs = VideoJobs(tmp_path)
        jobs.worker = Worker()
        await jobs.submit(VideoRequest(prompt="first"))
        await started.wait()
        await jobs.submit(VideoRequest(prompt="second"))
        assert ledger.active_video_jobs() == 2
        with pytest.raises(HTTPException) as exc:
            await jobs.submit(VideoRequest(prompt="third"))
        assert exc.value.status_code == 429
        assert ledger.active_video_jobs() == 2
        proceed.set()
        await jobs._drain_task

    asyncio.run(exercise())
    assert ledger.active_video_jobs() == 0
    engine.dispose()


def test_worker_failure_blocks_entire_queue_and_retains_claims(monkeypatch, tmp_path) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from llm_bawt.media.gpu_handoff_store import GpuHandoffStore

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ledger = GpuHandoffStore(engine)
    token, offered = ledger.offer(user="nick", target="video", actions=("verify",), expected_generation=0)
    transition = ledger.confirm(user="nick", token=token, expected_generation=offered.generation)
    ledger.begin_action(generation=transition.generation, action="verify")
    ledger.finish_action(generation=transition.generation, action="verify")
    ledger.complete(generation=transition.generation, target="video")
    monkeypatch.setattr(video_server.models, "installed", lambda: True)
    monkeypatch.setattr(video_server.models, "status", lambda: {"worker_ready": True})
    monkeypatch.setattr(video_server, "get_shared_engine", lambda config: engine)
    started = asyncio.Event()
    proceed = asyncio.Event()
    calls = []

    class Worker:
        uncertain = False
        resident = False
        running = False

        async def render(self, job, output):
            calls.append(output.name)
            started.set()
            await proceed.wait()
            raise RuntimeError("worker failed")

    async def exercise():
        jobs = VideoJobs(tmp_path)
        jobs.worker = Worker()
        first = await jobs.submit(VideoRequest(prompt="first"))
        await started.wait()
        second = await jobs.submit(VideoRequest(prompt="second"))
        proceed.set()
        await jobs._drain_task
        assert jobs.status(first["id"])["status"] == "failed"
        assert jobs.status(second["id"])["status"] == "failed"
        assert len(calls) == 1
        assert ledger.active_video_jobs() == 2
        assert ledger.status().phase == "recovery_required"
        with pytest.raises(HTTPException, match="reconciliation"):
            await jobs.submit(VideoRequest(prompt="third"))

    asyncio.run(exercise())
    engine.dispose()


def test_model_delete_rejects_durable_video_claim(monkeypatch, tmp_path) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import StaticPool
    from llm_bawt.media.gpu_handoff_store import GpuHandoffStore

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ledger = GpuHandoffStore(engine)
    token, offered = ledger.offer(user="nick", target="video", actions=("verify",), expected_generation=0)
    transition = ledger.confirm(user="nick", token=token, expected_generation=offered.generation)
    ledger.begin_action(generation=transition.generation, action="verify")
    ledger.finish_action(generation=transition.generation, action="verify")
    ledger.complete(generation=transition.generation, target="video")
    ledger.claim_video("pending-job")
    monkeypatch.setattr(video_server, "get_shared_engine", lambda config: engine)
    monkeypatch.setattr(video_server, "jobs", VideoJobs(tmp_path))

    async def refuse(*, active=False):
        assert active is True
        raise HTTPException(status_code=409, detail="Video generation is active")

    monkeypatch.setattr(video_server.models, "remove", refuse)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(video_server.remove_model())
    assert exc.value.status_code == 409
    assert ledger.active_video_jobs() == 1
    engine.dispose()


def test_model_inventory_and_scoped_cleanup(tmp_path) -> None:
    manager = VideoModelManager(tmp_path)
    other = tmp_path / "models--other--keep"
    other.mkdir()
    (other / "weights.bin").write_bytes(b"keep")
    blobs = manager.repo_dir / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "partial.incomplete").write_bytes(b"partial")
    status = manager.status()
    assert status["size_bytes"] == 7
    assert not status["installed"]
    assert status["cache_path"] == str(manager.repo_dir)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(manager.remove(active=True))
    assert exc.value.status_code == 409
    assert manager.repo_dir.exists()
    asyncio.run(manager.remove())
    assert not manager.repo_dir.exists()
    assert (other / "weights.bin").read_bytes() == b"keep"


def test_model_cleanup_rejects_symlink(tmp_path) -> None:
    manager = VideoModelManager(tmp_path)
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    manager.repo_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(manager.remove())
    assert exc.value.status_code == 400
    assert outside.exists()
    manager.repo_dir.unlink()


def test_local_video_client_submit_poll_and_download() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"id": "job1", "status": "processing", "progress": 10})
        if request.method == "DELETE":
            return httpx.Response(200, json={"deleted": True})
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"mp4")
        return httpx.Response(200, json={"id": "job1", "status": "completed", "progress": 100, "width": 832, "height": 480})

    async def exercise() -> None:
        client = LocalVideoClient(base_url="http://local-model-bridge:8685")
        await client.close()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=client.base_url)
        try:
            generated = await client.generate("running dog", "video", "wan2.2-ti2v-5b")
            assert generated.provider_job_id == "job1"
            status = await client.poll_status("job1")
            assert status.width == 832
            assert await client.download(status.media_url or "") == b"mp4"
            await client.remove_job("job1")
            with pytest.raises(ValueError, match="Unexpected"):
                await client.download("https://example.com/steal")
        finally:
            await client.close()

    asyncio.run(exercise())
    assert [r.url.path for r in requests] == ["/videos", "/videos/job1", "/videos/job1/content", "/videos/job1"]
