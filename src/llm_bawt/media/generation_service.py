"""Shared provider dispatch for route and MCP image generation surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from .clients import GenerationResult, MediaProviderRegistry, media_provider_registry


@dataclass(frozen=True)
class GeneratedImage:
    """Normalized image result after provider output has been downloaded."""

    data: bytes
    result: GenerationResult
    provider: str
    model: str


class MediaGenerationService:
    """Validate provider capabilities and normalize image generation output."""

    def __init__(self, registry: MediaProviderRegistry | None = None) -> None:
        self.registry = registry or media_provider_registry

    def resolve_request(
        self,
        *,
        provider: str | None,
        media_type: str,
        model: str | None,
        aspect_ratio: str | None,
        resolution: str | None,
        has_source_image: bool = False,
    ) -> tuple[str, str, str, str]:
        canonical = self.registry.canonical_provider(provider)
        capabilities = self.registry.capabilities(canonical)
        if media_type not in capabilities.media_types:
            raise ValueError(f"Provider {canonical!r} does not support {media_type!r}")
        resolved_model = model or self.registry.default_model(canonical, media_type)
        self.registry.validate_model(canonical, media_type, resolved_model)
        resolved_aspect = aspect_ratio or capabilities.default_aspect_ratios.get(media_type, "1:1")
        resolved_resolution = resolution or capabilities.default_resolutions.get(media_type, "1k")

        allowed_aspects = capabilities.aspect_ratios.get(media_type, ())
        if allowed_aspects and resolved_aspect not in allowed_aspects:
            raise ValueError(
                f"Provider {canonical!r} does not support aspect ratio {resolved_aspect!r} "
                f"for {media_type}; choose one of: {', '.join(allowed_aspects)}"
            )
        allowed_resolutions = capabilities.resolutions.get(media_type, ())
        if allowed_resolutions and resolved_resolution not in allowed_resolutions:
            raise ValueError(
                f"Provider {canonical!r} does not support resolution {resolved_resolution!r} "
                f"for {media_type}; choose one of: {', '.join(allowed_resolutions)}"
            )
        if has_source_image and media_type == "image" and not capabilities.image_input:
            raise ValueError(f"Provider {canonical!r} does not support reference-image generation")
        return canonical, resolved_model, resolved_aspect, resolved_resolution

    async def generate_image(
        self,
        *,
        prompt: str,
        provider: str | None = None,
        model: str | None = None,
        source_image: str | None = None,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        num_outputs: int = 1,
    ) -> GeneratedImage:
        if num_outputs != 1:
            # Only one output is persisted; refuse before paying for discarded ones.
            raise ValueError("num_outputs must be 1; multiple outputs are not supported")
        canonical, resolved_model, resolved_aspect, resolved_resolution = self.resolve_request(
            provider=provider,
            media_type="image",
            model=model,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            has_source_image=bool(source_image),
        )
        client = self.registry.create_client(canonical, "image")
        try:
            result = await client.generate(
                prompt=prompt,
                media_type="image",
                model=resolved_model,
                source_image=source_image,
                aspect_ratio=resolved_aspect,
                resolution=resolved_resolution,
                num_outputs=num_outputs,
            )
            if result.status != "completed" or not result.media_url:
                raise RuntimeError(result.error or "Image generation did not complete")
            data = await client.download(result.media_url)
            if not data:
                raise RuntimeError("Image generation returned no bytes")
            return GeneratedImage(data=data, result=result, provider=canonical, model=resolved_model)
        finally:
            await client.close()
