"""Manual GPU claim disposition is audited and never resets live worker/calls."""
import asyncio
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.media.gpu_handoff import GpuHandoff
from llm_bawt.media.gpu_handoff_store import GpuHandoffStore, HandoffConflict
from llm_bawt.media.gpu_profile import video_profile
from llm_bawt.service.routes import media, media_handoff
from local_model_bridge.video_server import VideoRequest
from tests.sqlite_support import fast_sqlite_engine
from tests.test_gpu_handoff_executor import FakeAdapter, confirm, offer
from tests.test_gpu_handoff_independent_review import measurement


@pytest.fixture
def restored(tmp_path):
    engine = fast_sqlite_engine(tmp_path / "claims.db", check_same_thread=False)
    store, adapter = GpuHandoffStore(engine), FakeAdapter()
    controller = GpuHandoff(store, adapter, poll_interval=0, job_timeout=0.1)

    async def establish():
        await confirm(controller, await offer(controller))
        generation = store.status().generation
        store.claim_calibration("calibration", generation=generation, profile=video_profile(VideoRequest(prompt="test")))
        store.finish_calibration("calibration", measurement())
        store.claim_video("lost-render")
        store.recover_orphaned_video(worker_restarted=True)
        proposed = await controller.offer_restore(user="nick", expected_generation=store.status().generation)
        restore_generation = await controller.accept_restore(user="nick", token=proposed["token"],
            expected_generation=proposed["generation"], approved=True)
        await controller.execute_restore(user="nick", generation=restore_generation)

    asyncio.run(establish())
    assert store.status().owner == "voice" and store.active_video_jobs() == 1
    yield store, adapter, controller
    engine.dispose()


def reviewed(controller, store, **overrides):
    args = dict(user="nick", job_id="lost-render", expected_generation=store.status().generation,
                outcome="unknown", evidence="Inspected bridge job and media record; output not recoverable.",
                approved=True, media_observation={"id": "gen-1", "provider": "local-video", "status": "pending"})
    return asyncio.run(controller.resolve_video_claim(**(args | overrides)))


def test_claim_resolution_requires_consent_evidence_and_stopped_worker(restored):
    store, adapter, controller = restored
    effects = list(adapter.effects)
    with pytest.raises(HandoffConflict, match="not approved"):
        reviewed(controller, store, approved=False)
    with pytest.raises(HandoffConflict, match="artifact"):
        reviewed(controller, store, outcome="completed")
    adapter.render_active = True
    with pytest.raises(HandoffConflict, match="still running"):
        reviewed(controller, store)
    adapter.render_active = False
    adapter.voice["active_count"] = 1
    with pytest.raises(HandoffConflict, match="Active voice"):
        reviewed(controller, store)
    adapter.voice["active_count"] = 0
    assert store.active_video_jobs() == 1 and adapter.effects == effects
    result = reviewed(controller, store)
    assert result["outcome"] == "unknown"
    assert store.active_video_jobs() == 0
    archive = store.resolved_video_claim("lost-render")
    assert archive["claim_generation"] > 0
    assert archive["reviewed_by"] == "nick"
    assert archive["observations"]["media"]["id"] == "gen-1"
    assert "Inspected bridge job" in archive["evidence"]
    assert adapter.effects == effects
    with pytest.raises(HandoffConflict, match="already resolved"):
        reviewed(controller, store)


def test_resolution_is_generation_fenced_and_only_after_voice(restored):
    store, adapter, controller = restored
    with pytest.raises(HandoffConflict, match="Restore voice"):
        reviewed(controller, store, expected_generation=store.status().generation - 1)
    assert store.active_video_jobs() == 1
    with pytest.raises(HandoffConflict, match="inspection evidence"):
        reviewed(controller, store, evidence="not inspected")
    assert store.resolved_video_claim("lost-render") is None


def test_resolution_route_validates_payload_and_persists_audit(restored, monkeypatch):
    store, adapter, controller = restored

    @asynccontextmanager
    async def factory():
        yield controller

    monkeypatch.setattr(media_handoff, "_controller", factory)
    monkeypatch.setattr(media, "_get_store", lambda: type("Media", (), {
        "get_by_provider_job_id": lambda self, job: {
            "id": "gen-1", "provider": "local-video", "status": "pending", "file_path": None, "error": None}})())
    app = FastAPI()
    app.include_router(media_handoff.router)
    with TestClient(app) as client:
        path = "/v1/media/local-video/handoff/claims/resolve"
        body = {"user": "nick", "job_id": "lost-render", "expected_generation": store.status().generation,
                "outcome": "unknown", "evidence": "Inspected job output and found no recoverable video.", "approved": True}
        assert client.post(path, json=body | {"approved": False}).status_code == 409
        assert client.post(path, json=body | {"approved": "true"}).status_code == 422
        assert client.post(path, json=body).json()["job_id"] == "lost-render"
        assert client.post(path, json=body).status_code == 409
        archive = client.get("/v1/media/local-video/handoff/claims/resolved/lost-render")
        assert archive.status_code == 200 and archive.json()["observations"]["media"]["id"] == "gen-1"
    assert store.resolved_video_claim("lost-render") is not None
