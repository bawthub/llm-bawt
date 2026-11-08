"""OpenAI Images API media client."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from urllib.parse import unquote_to_bytes

import httpx

from .base import GenerationResult, MediaClient, resolve_media_api_key

logger = logging.getLogger(__name__)

OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 180.0


@dataclass(frozen=True)
class _SourceImage:
    data: bytes
    mime_type: str
    filename: str


class OpenAIImageClient(MediaClient):
    """OpenAI image generation/edit client using the official HTTP API."""

    def __init__(self, api_key: str | None = None, base_url: str = OPENAI_BASE_URL, config=None):
        # The stored OpenAI API connection first, then OPENAI_API_KEY (TASK-1612).
        self.api_key = resolve_media_api_key(config, "openai-api", ("OPENAI_API_KEY",), explicit=api_key)
        self.base_url = base_url.rstrip("/")
        if not self.api_key:
            logger.warning("OpenAIImageClient: no OpenAI API key; connect OpenAI in Settings → Providers.")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(DEFAULT_TIMEOUT, connect=10.0),
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def generate(
        self,
        prompt: str,
        media_type: str,
        model: str,
        *,
        source_image: str | None = None,
        aspect_ratio: str = "1:1",
        duration: float = 5,
        resolution: str = "1k",
        num_outputs: int = 1,
    ) -> GenerationResult:
        if media_type != "image":
            raise ValueError("OpenAIImageClient supports image generation only")
        if not self.api_key:
            raise RuntimeError("An OpenAI API key is required for the openai image provider (connect OpenAI in Settings → Providers)")

        size = self._image_size(aspect_ratio, resolution)
        common = {
            "model": model,
            "prompt": prompt,
            "n": max(1, int(num_outputs or 1)),
            "size": size,
        }

        if source_image:
            source = await self._resolve_source_image(source_image)
            response = await self._client.post(
                "/images/edits",
                data={key: str(value) for key, value in common.items()},
                files={"image": (source.filename, source.data, source.mime_type)},
            )
        else:
            response = await self._client.post("/images/generations", json=common)

        if response.status_code >= 400:
            logger.error(
                "OpenAI image generation failed: status=%s body=%s",
                response.status_code,
                response.text[:500],
            )
        response.raise_for_status()
        payload = response.json()
        items = payload.get("data") or []
        if not items:
            return GenerationResult(
                provider_job_id="",
                status="failed",
                error="OpenAI image response contained no outputs",
            )

        urls: list[str] = []
        revised_prompt: str | None = None
        for item in items:
            revised_prompt = revised_prompt or item.get("revised_prompt")
            if item.get("b64_json"):
                urls.append(f"data:image/png;base64,{item['b64_json']}")
            elif item.get("url"):
                urls.append(item["url"])
        if not urls:
            return GenerationResult(
                provider_job_id="",
                status="failed",
                error="OpenAI image response contained neither image bytes nor URLs",
            )

        return GenerationResult(
            provider_job_id=payload.get("id", ""),
            status="completed",
            progress=100,
            media_url=urls[0],
            revised_prompt=revised_prompt,
            extra={"urls": urls, "size": size},
        )

    async def poll_status(self, provider_job_id: str) -> GenerationResult:
        raise ValueError("OpenAI image generation is synchronous and cannot be polled")

    async def download(self, media_url: str) -> bytes:
        if media_url.startswith("data:"):
            return self._decode_data_url(media_url).data
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
            response = await client.get(media_url)
            response.raise_for_status()
            return response.content

    @staticmethod
    def _image_size(aspect_ratio: str, resolution: str) -> str:
        """Map generic aspect/resolution inputs onto GPT Image supported sizes."""
        del resolution  # GPT Image's stable API currently exposes 1k dimension presets.
        if aspect_ratio in {"2:3", "3:4", "9:16"}:
            return "1024x1536"
        if aspect_ratio in {"3:2", "4:3", "16:9"}:
            return "1536x1024"
        return "1024x1024"

    async def _resolve_source_image(self, source_image: str) -> _SourceImage:
        if source_image.startswith("data:"):
            return self._decode_data_url(source_image)
        response = await self._client.get(source_image)
        response.raise_for_status()
        mime = response.headers.get("content-type", "image/png").split(";", 1)[0]
        return _SourceImage(response.content, mime, self._filename_for_mime(mime))

    @staticmethod
    def _decode_data_url(data_url: str) -> _SourceImage:
        header, encoded = data_url.split(",", 1)
        mime = header[5:].split(";", 1)[0] or "image/png"
        raw = base64.b64decode(encoded) if ";base64" in header else unquote_to_bytes(encoded)
        return _SourceImage(raw, mime, OpenAIImageClient._filename_for_mime(mime))

    @staticmethod
    def _filename_for_mime(mime: str) -> str:
        extension = {
            "image/jpeg": "jpg",
            "image/webp": "webp",
            "image/gif": "gif",
        }.get(mime, "png")
        return f"reference.{extension}"
