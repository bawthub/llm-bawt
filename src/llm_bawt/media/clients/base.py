"""Abstract base class for media generation clients."""

from __future__ import annotations

import abc
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_media_api_key(
    config,
    provider_id: str,
    env_vars: Sequence[str],
    *,
    explicit: str | None = None,
    config_attr: str | None = None,
) -> str:
    """A media key using the LLM clients' credential resolution policy.

    Explicit keys win; OpenAI/xAI otherwise require their stored connection.
    Returns ``""`` when no key is found or credential resolution fails.
    """
    if explicit:
        return explicit
    try:
        from ...service.providers.api_key import resolve_api_key  # noqa: PLC0415 — keep media imports light
        from ...utils.config import Config  # noqa: PLC0415

        return resolve_api_key(
            config if config is not None else Config(),
            provider_id,
            env_vars=env_vars,
            config_attr=config_attr,
        ) or ""
    except Exception as error:  # noqa: BLE001 — media remains unconfigured if lookup fails
        logger.debug("Media key lookup for %s failed: %s", provider_id, type(error).__name__)
        return ""


@dataclass
class GenerationResult:
    """Result returned by a media generation client."""

    provider_job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int = 0  # 0-100
    media_url: Optional[str] = None  # Temporary URL to download completed media
    revised_prompt: Optional[str] = None
    error: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    actual_duration: Optional[float] = None
    extra: dict = field(default_factory=dict)  # Provider-specific metadata


class MediaClient(abc.ABC):
    """Abstract base class for provider-specific media generation clients.

    Each provider (xAI, OpenAI, Stability, etc.) implements this interface
    to provide a unified media generation API.
    """

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        media_type: str,
        model: str,
        *,
        source_image: str | None = None,
        aspect_ratio: str = "16:9",
        duration: float = 5,
        resolution: str = "720p",
        num_outputs: int = 1,
    ) -> GenerationResult:
        """Submit a media generation request.

        Args:
            prompt: Text prompt for generation.
            media_type: 'video' or 'image'.
            model: Model identifier.
            source_image: Optional base64 data URI or URL for image-to-video.
            aspect_ratio: Desired aspect ratio.
            duration: Duration in seconds (video only).
            resolution: Desired resolution.
            num_outputs: Number of outputs (image only).

        Returns:
            GenerationResult with at least provider_job_id and status.
        """
        ...

    @abc.abstractmethod
    async def poll_status(self, provider_job_id: str) -> GenerationResult:
        """Check the status of a previously submitted generation job.

        Args:
            provider_job_id: The job ID returned by generate().

        Returns:
            Updated GenerationResult.
        """
        ...

    @abc.abstractmethod
    async def download(self, media_url: str) -> bytes:
        """Download completed media from a URL or provider data URI."""
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        """Close provider-owned network resources."""
        ...
