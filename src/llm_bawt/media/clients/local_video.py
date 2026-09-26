"""Media client for Wan jobs running in the existing local-model GPU bridge."""

from __future__ import annotations

import os

import httpx

from .base import GenerationResult, MediaClient


class LocalVideoClient(MediaClient):
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("LOCAL_MODEL_VIDEO_URL", "http://local-model-bridge:8685")).rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(60, connect=10))

    async def generate(
        self, prompt: str, media_type: str, model: str, *, source_image: str | None = None,
        aspect_ratio: str = "16:9", duration: float = 5, resolution: str = "480p", num_outputs: int = 1,
    ) -> GenerationResult:
        if media_type != "video" or model != "wan2.2-ti2v-5b":
            raise ValueError("Local video supports only wan2.2-ti2v-5b")
        response = await self._client.post("/videos", json={
            "prompt": prompt, "source_image": source_image, "aspect_ratio": aspect_ratio,
            "duration": duration, "resolution": resolution,
        })
        response.raise_for_status()
        data = response.json()
        return GenerationResult(provider_job_id=data["id"], status=data["status"], progress=data["progress"])

    async def poll_status(self, provider_job_id: str) -> GenerationResult:
        response = await self._client.get(f"/videos/{provider_job_id}")
        response.raise_for_status()
        data = response.json()
        return GenerationResult(
            provider_job_id=provider_job_id, status=data["status"], progress=data["progress"],
            error=data.get("error"),
            media_url=f"{self.base_url}/videos/{provider_job_id}/content" if data["status"] == "completed" else None,
            width=data.get("width"), height=data.get("height"), actual_duration=data.get("actual_duration"),
        )

    async def download(self, media_url: str) -> bytes:
        # Never fetch arbitrary URLs: completed output must be the bridge job we polled.
        prefix = f"{self.base_url}/videos/"
        job_id = media_url.removeprefix(prefix).removesuffix("/content")
        if not media_url.startswith(prefix) or not media_url.endswith("/content") or not job_id.isascii() or not job_id.isalnum():
            raise ValueError("Unexpected local video URL")
        response = await self._client.get(media_url.removeprefix(self.base_url))
        response.raise_for_status()
        data = response.content
        # Keep the temporary file until the app persists it successfully.
        # The app's poller can retry a failed storage write without losing output.
        return data

    async def model_status(self) -> dict:
        response = await self._client.get("/models/wan2.2-ti2v-5b")
        response.raise_for_status()
        return response.json()

    async def gpu_telemetry(self) -> dict:
        response = await self._client.get("/gpu/telemetry")
        response.raise_for_status()
        return response.json()

    async def install_model(self) -> dict:
        response = await self._client.post("/models/wan2.2-ti2v-5b/install")
        response.raise_for_status()
        return response.json()

    async def remove_model(self) -> dict:
        response = await self._client.delete("/models/wan2.2-ti2v-5b")
        response.raise_for_status()
        return response.json()

    async def remove_job(self, provider_job_id: str) -> None:
        response = await self._client.delete(f"/videos/{provider_job_id}")
        response.raise_for_status()

    async def close(self) -> None:
        await self._client.aclose()
