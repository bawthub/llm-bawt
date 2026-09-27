"""Read-only GPU handoff preflight; never equate observations with consent.

All missing signals fail closed. In particular, free VRAM is not a Wan peak
estimate, and configured Docker operations are not permission to execute them.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx

from llm_bawt.integrations.home_audio import HomeAudioSettings

MORPH_OPS = (
    "bawthub.stop-moshi-stt", "bawthub.stop-moshi-tts",
    "bawthub.start-moshi-stt", "bawthub.start-moshi-tts",
)


class GpuPreflight:
    def __init__(self, *, config, http: httpx.AsyncClient, ops):
        self.config = config
        self.http = http
        self.ops = ops

    async def voice_status(self) -> tuple[dict | None, str | None]:
        # Voice admission lives on the same backend as speech rendering. Reuse
        # its existing DB-backed address; the network boundary supplies trust.
        try:
            settings = await asyncio.to_thread(HomeAudioSettings.load, self.config)
        except Exception:
            return None, "Voice backend runtime settings unavailable"
        try:
            result = await self.http.get(settings.tts_url.rstrip("/") + "/v1/internal/voice/sessions",
                                         timeout=3)
            result.raise_for_status()
            data = result.json()
            if (not isinstance(data, dict) or type(data.get("active_count")) is not int
                    or type(data.get("active_gpu_work")) is not int
                    or type(data.get("parked")) is not bool
                    or data["active_count"] < 0 or data["active_gpu_work"] < 0):
                raise ValueError("Invalid voice admission response")
            return {"active_count": data["active_count"], "active_gpu_work": data["active_gpu_work"],
                    "parked": data["parked"]}, None
        except (httpx.HTTPError, ValueError) as exc:
            # Never return auth headers, body or the configured URL in the brief.
            return None, f"Voice admission unavailable ({type(exc).__name__})"

    async def assess(self, *, state, gpu: dict, model: dict | None = None,
                     active_video_jobs: int | None = None, calibration: dict | None = None) -> dict:
        issues: list[str] = []
        if not isinstance(model, dict) or type(model.get("installed")) is not bool or not model["installed"]:
            issues.append("Wan model installation is unavailable or incomplete")
        if not isinstance(model, dict) or type(model.get("resident")) is not bool:
            issues.append("Wan worker residency is unverified")
        if isinstance(model, dict) and model.get("worker_uncertain") is True:
            issues.append("Wan worker outcome requires manual reconciliation")
        if type(active_video_jobs) is not int or active_video_jobs < 0:
            issues.append("Durable video job claims are unavailable")
        elif active_video_jobs:
            issues.append("Active video jobs must finish before handoff")
        if state.phase not in ("idle", "offered") or state.owner == "unknown":
            issues.append("GPU owner is unknown or needs reconciliation")
        if not isinstance(gpu, dict) or gpu.get("ready") is not True:
            issues.append("GPU driver telemetry unavailable")
        else:
            try:
                observed_at = datetime.fromisoformat(gpu["observed_at"])
                if observed_at.tzinfo is None or observed_at.utcoffset() is None:
                    issues.append("GPU telemetry is stale")
                else:
                    age = (datetime.now(UTC) - observed_at.astimezone(UTC)).total_seconds()
                    if not 0 <= age <= 15:
                        issues.append("GPU telemetry is stale")
                total, free = gpu["total_mib"], gpu["free_mib"]
                if type(total) is not int or type(free) is not int or not 0 <= free <= total or total == 0:
                    issues.append("GPU telemetry values are invalid")
            except (KeyError, TypeError, ValueError, OverflowError):
                issues.append("GPU telemetry values are invalid")
        voice, voice_error = await self.voice_status()
        if voice_error:
            issues.append(voice_error)
        else:
            if voice["parked"] and state.owner not in ("video", "video_calibration"):
                issues.append("Voice admission is already parked; reconcile its lease before handoff")
            if voice["active_count"] or voice["active_gpu_work"]:
                issues.append("Active voice or Moshi speech must finish before handoff")
        try:
            operations = await self.ops()
            if not isinstance(operations, list):
                raise ValueError("Invalid ops catalog")
            enabled = {op["slug"] for op in operations if isinstance(op, dict)
                       and type(op.get("slug")) is str and op.get("enabled") is True}
            missing = [slug for slug in MORPH_OPS if slug not in enabled]
            if missing:
                issues.append("Moshi start/stop operations are not enabled: " + ", ".join(missing))
        except Exception:
            issues.append("Ops catalog unavailable")
        # A measured profile is evidence for that profile only, never a permit
        # to switch or silently extend the supported resolution/duration.
        if not calibration or calibration.get("supported") is not True:
            issues.append("Wan peak VRAM and safety margin have not been measured")
        return {"switch_ready": False, "issues": issues, "voice": voice}
