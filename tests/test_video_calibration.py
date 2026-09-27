import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine

from llm_bawt.media.gpu_handoff_store import GpuHandoffStore, HandoffConflict
from llm_bawt.media.gpu_profile import CALIBRATION_PROFILE, video_profile
from local_model_bridge import video_server
from local_model_bridge.video_server import VideoJobs, VideoRequest
from local_model_bridge.video_memory import VideoMemoryMeasurement

MEASUREMENT = {"gpu_uuid": "GPU-test", "total_mib": 15500, "device_total_mib": 16000, "baseline_used_mib": 1000,
               "sampled_peak_used_mib": 11000, "peak_reserved_mib": 9500,
               "peak_allocated_mib": 9000, "worker_reserved_mib": 9500}


@pytest.fixture
def reserved(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/calibration.db", connect_args={"check_same_thread": False})
    ledger = GpuHandoffStore(engine)
    token, offer = ledger.offer(user="nick", target="video_calibration", actions=("park_voice",), expected_generation=0,
        plan={"calibration_profile": CALIBRATION_PROFILE, "observed": {"gpu": {"uuid": "GPU-test", "total_mib": 16000}}})
    transition = ledger.confirm(user="nick", token=token, expected_generation=offer.generation)
    ledger.begin_action(generation=transition.generation, action="park_voice")
    ledger.record_voice_lease(generation=transition.generation, lease_id="a" * 32)
    ledger.finish_action(generation=transition.generation, action="park_voice")
    state = ledger.complete(generation=transition.generation, target="video_calibration")
    yield ledger, state, engine
    engine.dispose()


def test_calibration_claim_is_one_use_and_binds_profile(reserved):
    ledger, state, _ = reserved
    profile = video_profile(VideoRequest(prompt="test"))
    with pytest.raises(HandoffConflict):
        ledger.claim_video("normal")
    with pytest.raises(ValueError):
        ledger.claim_calibration("bad", generation=state.generation, profile=profile | {"duration": 10})
    ledger.claim_calibration("first", generation=state.generation, profile=profile)
    with pytest.raises(HandoffConflict):
        ledger.claim_calibration("second", generation=state.generation, profile=profile)
    measured = ledger.finish_calibration("first", MEASUREMENT)
    assert measured["supported"]
    assert measured["required_mib"] == 12048
    assert ledger.status().owner == "video"
    assert ledger.active_video_jobs() == 0
    gpu = {"ready": True, "uuid": "GPU-test", "total_mib": 16000, "free_mib": 5500,
           "observed_at": datetime.now(UTC).isoformat()}
    ledger.validate_video_profile(profile, gpu, 9500)
    with pytest.raises(HandoffConflict):
        ledger.validate_video_profile(profile, gpu | {"free_mib": 1000}, 9500)
    with pytest.raises(ValueError):
        ledger.validate_video_profile(profile | {"image_conditioned": True}, gpu, 9500)


@pytest.mark.parametrize("change", [{"gpu_uuid": "GPU-other"}, {"peak_reserved_mib": -1}, {"peak_allocated_mib": True}])
def test_invalid_measurement_never_grants_video(reserved, change):
    ledger, state, _ = reserved
    ledger.claim_calibration("first", generation=state.generation, profile=video_profile(VideoRequest(prompt="test")))
    with pytest.raises(HandoffConflict):
        ledger.finish_calibration("first", MEASUREMENT | change)
    assert ledger.status().owner == "video_calibration"
    assert ledger.active_video_jobs() == 1


def test_insufficient_margin_keeps_ordinary_video_blocked(reserved):
    ledger, state, _ = reserved
    ledger.claim_calibration("first", generation=state.generation, profile=video_profile(VideoRequest(prompt="test")))
    result = ledger.finish_calibration("first", MEASUREMENT | {"sampled_peak_used_mib": 15500})
    assert not result["supported"]
    assert ledger.status().phase == "recovery_required"
    with pytest.raises(HandoffConflict):
        ledger.claim_video("normal")


def test_private_calibration_queue_records_evidence_once(reserved, monkeypatch, tmp_path):
    ledger, state, engine = reserved
    monkeypatch.setattr(video_server, "get_shared_engine", lambda config: engine)
    monkeypatch.setattr(video_server.models, "installed", lambda: True)
    monkeypatch.setattr(video_server.models, "status", lambda: {"worker_ready": True})
    async def parked(store):
        assert store.voice_lease() == "a" * 32
    monkeypatch.setattr(video_server, "verify_voice_park", parked)
    class Worker:
        uncertain = False
        async def render(self, input_path, output):
            output.write_bytes(b"test output")
            return {"gpu_measurement": MEASUREMENT, "worker_reserved_mib": 9500}
    async def run():
        jobs = VideoJobs(tmp_path)
        jobs.worker = Worker()
        result = await jobs.submit(VideoRequest(prompt="test", calibration_generation=state.generation))
        await jobs._drain_task
        assert jobs.status(result["id"])["status"] == "completed"
        with pytest.raises(HTTPException):
            await jobs.submit(VideoRequest(prompt="duplicate", calibration_generation=state.generation))
    asyncio.run(run())
    assert ledger.calibration()["supported"]


def test_memory_meter_keeps_sampled_and_allocator_peaks_distinct():
    mib = 1024 * 1024
    cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
        mem_get_info=lambda: (12000 * mib, 16000 * mib), max_memory_reserved=lambda: 9500 * mib,
        max_memory_allocated=lambda: 9000 * mib, memory_reserved=lambda: 9500 * mib)
    telemetry = SimpleNamespace(observe=lambda: {"ready": True, "uuid": "GPU-test", "total_mib": 16303})
    with VideoMemoryMeasurement(cuda=cuda, telemetry=telemetry) as meter:
        cuda.mem_get_info = lambda: (5000 * mib, 16000 * mib)
    result = meter.result()
    assert result["baseline_used_mib"] == 4000
    assert result["sampled_peak_used_mib"] == 11000
    assert result["peak_reserved_mib"] == 9500
