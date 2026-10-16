"""Independent TASK-930 safety regressions; fake hardware and isolated SQLite only."""
import asyncio

import pytest
from sqlalchemy import create_engine

from llm_bawt.media.gpu_handoff import GpuHandoff, START_OPERATIONS
from llm_bawt.media.gpu_handoff_store import GpuHandoffStore, HandoffConflict
from llm_bawt.media.gpu_profile import video_profile
from local_model_bridge.video_server import VideoRequest
from tests.test_gpu_handoff_executor import FakeAdapter, confirm, offer
from tests.test_video_calibration import MEASUREMENT


@pytest.fixture
def review_setup(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/review.db", connect_args={"check_same_thread": False})
    store, adapter = GpuHandoffStore(engine), FakeAdapter()
    controller = GpuHandoff(store, adapter, poll_interval=0, job_timeout=0.1)
    asyncio.run(confirm(controller, asyncio.run(offer(controller))))
    yield store, adapter, controller
    engine.dispose()


def measurement():
    return MEASUREMENT | {"gpu_uuid": "GPU-one", "device_total_mib": 16303}


def test_resume_video_refuses_nonterminal_moshi_start(review_setup):
    store, adapter, controller = review_setup
    store.claim_calibration("calibration", generation=store.status().generation, profile=video_profile(VideoRequest(prompt="review")))
    store.finish_calibration("calibration", measurement())

    async def response_lost(snapshot, *, user, generation):
        slug = snapshot["operation"]["slug"]
        adapter.jobs["pending-start"] = {
            "id": "pending-start", "operation": slug, "terminal": False,
            "state": "queued", "exit_code": None,
        }
        raise RuntimeError("Response lost; the queued operation may still execute")

    async def run():
        proposed = await controller.offer_restore(user="nick", expected_generation=store.status().generation)
        generation = await controller.accept_restore(user="nick", token=proposed["token"],
            expected_generation=proposed["generation"], approved=True)
        adapter.dispatch = response_lost
        with pytest.raises(RuntimeError, match="Response lost"):
            await controller.execute_restore(user="nick", generation=generation)
        assert store.status().pending_action == "start_moshi_stt"
        assert adapter.jobs["pending-start"]["operation"] == START_OPERATIONS[0]
        with pytest.raises(HandoffConflict, match="unverified|receipt|pending"):
            await controller.resume_video(user="nick", expected_generation=store.status().generation)

    asyncio.run(run())


def test_repeated_unconfirmed_voice_offers_preserve_calibration(review_setup):
    store, adapter, controller = review_setup
    store.claim_calibration("accepted-calibration", generation=store.status().generation,
                            profile=video_profile(VideoRequest(prompt="review")))
    adapter.render_active = True

    async def run():
        for _ in range(2):
            await controller.offer_restore(user="nick", expected_generation=store.status().generation)
        return store.finish_calibration("accepted-calibration", measurement())

    evidence = asyncio.run(run())
    assert evidence["supported"]
    assert store.calibration()["job_id"] == "accepted-calibration"
    assert store.active_video_jobs() == 0


def test_public_calibration_admission_survives_repeated_voice_offers(review_setup, monkeypatch):
    from types import SimpleNamespace
    from llm_bawt.media.schemas import MediaGenerationRequest
    from llm_bawt.service.routes import media

    store, _, controller = review_setup
    reservation = store.status().generation

    class AdmissionReached(Exception):
        pass

    def stop_before_persisting(row):
        raise AdmissionReached()

    monkeypatch.setattr(media, "_get_store", lambda: SimpleNamespace(insert=stop_before_persisting))
    monkeypatch.setattr(media, "_get_storage", lambda: object())
    monkeypatch.setattr(media, "get_service", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr(media, "get_shared_engine", lambda config: object())
    monkeypatch.setattr(media, "GpuHandoffStore", lambda engine: store)

    async def run():
        for _ in range(2):
            await controller.offer_restore(user="nick", expected_generation=store.status().generation)
        with pytest.raises(AdmissionReached):
            await media.create_generation(MediaGenerationRequest(
                prompt="review", provider="local-video", media_type="video",
                calibration_generation=reservation))

    asyncio.run(run())
