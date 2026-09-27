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
        status=lambda: state, active_video_jobs=lambda: 1, calibration=lambda: None))

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
    assert result["model"] is None
    assert any("Active video jobs" in issue for issue in result["preflight"]["issues"])
    assert any("residency is unverified" in issue for issue in result["preflight"]["issues"])
