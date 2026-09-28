"""Registry and capability metadata for media generation providers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

from ..gpu_profile import CALIBRATION_PROFILE
from .base import MediaClient
from .grok_media import GrokMediaClient
from .local_video import LocalVideoClient
from .openai_images import OpenAIImageClient


@dataclass(frozen=True)
class MediaProviderCapabilities:
    """Provider options safe to expose to API and UI consumers."""

    provider: str
    label: str
    media_types: tuple[str, ...]
    default_models: dict[str, str]
    models: dict[str, tuple[str, ...]]
    image_input: bool = False
    aspect_ratios: dict[str, tuple[str, ...]] = field(default_factory=dict)
    resolutions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    default_aspect_ratios: dict[str, str] = field(default_factory=dict)
    default_resolutions: dict[str, str] = field(default_factory=dict)
    durations: dict[str, tuple[int, ...]] = field(default_factory=dict)
    default_durations: dict[str, int] = field(default_factory=dict)
    # Accepts `negative_prompt` (things to avoid) on generate().
    negative_prompt: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _ProviderRegistration:
    capabilities: MediaProviderCapabilities
    factory: Callable[[], MediaClient]


class MediaProviderRegistry:
    """Resolve provider clients without provider conditionals in callers."""

    def __init__(self) -> None:
        self._providers: dict[str, _ProviderRegistration] = {}
        self._aliases: dict[str, str] = {}

    def register(
        self,
        capabilities: MediaProviderCapabilities,
        factory: Callable[[], MediaClient],
        *,
        aliases: tuple[str, ...] = (),
    ) -> None:
        provider = capabilities.provider.lower()
        self._providers[provider] = _ProviderRegistration(capabilities, factory)
        for alias in aliases:
            self._aliases[alias.lower()] = provider

    def canonical_provider(self, provider: str | None) -> str:
        candidate = (provider or "grok").strip().lower()
        return self._aliases.get(candidate, candidate)

    def capabilities(self, provider: str | None = None) -> MediaProviderCapabilities:
        canonical = self.canonical_provider(provider)
        registration = self._providers.get(canonical)
        if registration is None:
            supported = ", ".join(sorted(self._providers))
            raise ValueError(f"Unknown media provider {provider!r}; choose one of: {supported}")
        return registration.capabilities

    def create_client(self, provider: str | None, media_type: str) -> MediaClient:
        canonical = self.canonical_provider(provider)
        registration = self._providers.get(canonical)
        if registration is None:
            self.capabilities(provider)  # raises the detailed error
            raise AssertionError("unreachable")
        if media_type not in registration.capabilities.media_types:
            raise ValueError(
                f"Provider {canonical!r} does not support {media_type!r}; "
                f"supported media types: {', '.join(registration.capabilities.media_types)}"
            )
        return registration.factory()

    def default_model(self, provider: str | None, media_type: str) -> str:
        capabilities = self.capabilities(provider)
        try:
            return capabilities.default_models[media_type]
        except KeyError as exc:
            raise ValueError(
                f"Provider {capabilities.provider!r} has no default {media_type} model"
            ) from exc

    def validate_model(self, provider: str | None, media_type: str, model: str) -> None:
        capabilities = self.capabilities(provider)
        allowed = capabilities.models.get(media_type, ())
        if allowed and model not in allowed:
            raise ValueError(
                f"Model {model!r} is not configured for provider {capabilities.provider!r} "
                f"and media type {media_type!r}; choose one of: {', '.join(allowed)}"
            )

    def list_capabilities(self, media_type: str | None = None) -> list[MediaProviderCapabilities]:
        values = [registration.capabilities for registration in self._providers.values()]
        if media_type:
            values = [cap for cap in values if media_type in cap.media_types]
        return sorted(values, key=lambda cap: cap.provider)


def build_default_registry() -> MediaProviderRegistry:
    registry = MediaProviderRegistry()
    registry.register(
        MediaProviderCapabilities(
            provider="grok",
            label="Grok",
            media_types=("image", "video"),
            default_models={
                "image": "grok-imagine-image",
                "video": "grok-imagine-video",
            },
            models={
                "image": ("grok-imagine-image",),
                "video": ("grok-imagine-video",),
            },
            image_input=True,
            aspect_ratios={
                "image": ("1:1", "2:3", "3:2", "9:16", "16:9", "4:3", "3:4"),
                "video": ("16:9", "9:16", "1:1", "4:3", "3:4"),
            },
            resolutions={
                "image": ("1k", "2k"),
                "video": ("480p", "720p"),
            },
            default_aspect_ratios={"image": "2:3", "video": "16:9"},
            default_resolutions={"image": "1k", "video": "720p"},
            durations={"video": (3, 5, 8, 10, 15)},
            default_durations={"video": 5},
        ),
        GrokMediaClient,
        aliases=("xai",),
    )
    registry.register(
        MediaProviderCapabilities(
            provider="local-video",
            label="Local (Wan 2.2)",
            media_types=("video",),
            default_models={"video": "wan2.2-ti2v-5b"},
            models={"video": ("wan2.2-ti2v-5b",)},
            # Wan runs only the measured calibration profile; the submit gate
            # (gpu_profile.require_calibration_profile) rejects anything else,
            # so advertise exactly that and nothing the UI could pick wrongly.
            image_input=False,
            aspect_ratios={"video": (CALIBRATION_PROFILE["aspect_ratio"],)},
            resolutions={"video": (CALIBRATION_PROFILE["resolution"],)},
            default_aspect_ratios={"video": CALIBRATION_PROFILE["aspect_ratio"]},
            default_resolutions={"video": CALIBRATION_PROFILE["resolution"]},
            durations={"video": (CALIBRATION_PROFILE["duration"],)},
            default_durations={"video": CALIBRATION_PROFILE["duration"]},
            negative_prompt=True,
        ),
        LocalVideoClient,
    )
    registry.register(
        MediaProviderCapabilities(
            provider="openai",
            label="OpenAI",
            media_types=("image",),
            default_models={"image": "gpt-image-2"},
            models={"image": ("gpt-image-2", "gpt-image-1.5", "gpt-image-1")},
            image_input=True,
            aspect_ratios={"image": ("1:1", "2:3", "3:2")},
            resolutions={"image": ("1k",)},
            default_aspect_ratios={"image": "1:1"},
            default_resolutions={"image": "1k"},
        ),
        OpenAIImageClient,
    )
    return registry


media_provider_registry = build_default_registry()
