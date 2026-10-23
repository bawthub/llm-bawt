"""GPU status must report telemetry failure without inventing capacity."""
import asyncio
from types import SimpleNamespace

from llm_bawt.service.routes import media


def test_gpu_status_reports_unavailable_bridge(monkeypatch):
    state = SimpleNamespace(owner="unknown", phase="recovery_required", generation=3,
                            target="video", actions=("stop_moshi_stt",), next_action=0,
                            pending_action="stop_moshi_stt", last_job_id="job1",
                            last_error="Outcome uncertain")
    monkeypatch.setattr(media, "get_service", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr(media, "get_shared_engine", lambda config: object())
    monkeypatch.setattr(media, "GpuHandoffStore", lambda engine: SimpleNamespace(
        status=lambda: state, active_video_jobs=lambda: 1, video_claims=lambda: ["job1"],
        calibration=lambda: None, calibration_reservation=lambda: None))

    class FailedClient:
        async def gpu_telemetry(self):
            raise ConnectionError("driver unavailable")

    monkeypatch.setattr(media, "_get_video_client", lambda provider: FailedClient())
    result = asyncio.run(media.local_video_gpu_status())
    assert (result["owner"], result["phase"], result["last_job_id"]) == (
        "unknown", "recovery_required", "job1")
    assert result["gpu"] == {"ready": False, "error": "GPU bridge telemetry unavailable: ConnectionError"}
    assert "free_mib" not in result["gpu"]
    assert result["active_video_jobs"] == 1
    assert result["video_claims"] == ["job1"]
    assert result["model"] is None
    assert any("Active video jobs" in issue for issue in result["preflight"]["issues"])
    assert any("residency is unverified" in issue for issue in result["preflight"]["issues"])


def test_gpu_status_exposes_original_calibration_reservation_after_reoffers(monkeypatch, tmp_path):
    from llm_bawt.media.gpu_handoff import GpuHandoff
    from llm_bawt.media.gpu_handoff_store import GpuHandoffStore
    from tests.sqlite_support import fast_sqlite_engine
    from tests.test_gpu_handoff_executor import FakeAdapter, confirm, offer

    engine = fast_sqlite_engine(tmp_path / "status.db", check_same_thread=False)
    ledger, adapter = GpuHandoffStore(engine), FakeAdapter()
    controller = GpuHandoff(ledger, adapter, poll_interval=0, job_timeout=0.1)

    async def reoffer():
        await confirm(controller, await offer(controller))
        reservation = ledger.status().generation
        for _ in range(2):
            await controller.offer_restore(user="nick", expected_generation=ledger.status().generation)
        return reservation

    reservation = asyncio.run(reoffer())
    monkeypatch.setattr(media, "get_service", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr(media, "get_shared_engine", lambda config: object())
    monkeypatch.setattr(media, "GpuHandoffStore", lambda engine: ledger)

    class Client:
        async def gpu_telemetry(self):
            return (await adapter.observe())["gpu"]

        async def model_status(self):
            return (await adapter.observe())["model"]

    class Preflight:
        def __init__(self, **kwargs):
            pass

        async def assess(self, **kwargs):
            return {"switch_ready": False, "issues": [], "voice": None}

    monkeypatch.setattr(media, "_get_video_client", lambda provider: Client())
    monkeypatch.setattr(media, "GpuPreflight", Preflight)
    result = asyncio.run(media.local_video_gpu_status())
    assert result["phase"] == "offered" and result["generation"] > reservation + 1
    assert result["calibration_generation"] == reservation
    engine.dispose()
