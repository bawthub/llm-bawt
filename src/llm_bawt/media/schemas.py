"""Pydantic request/response models for the media generation API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MediaGenerationRequest(BaseModel):
    """Request body for POST /v1/media/generations."""

    prompt: str = Field(..., description="Text prompt describing the desired media")
    original_prompt: Optional[str] = Field(
        default=None,
        description="User's text before prompt expansion; set only when `prompt` was expanded",
    )
    negative_prompt: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Things to avoid; only for providers whose capabilities set negative_prompt",
    )
    provider: str = Field(
        default="grok",
        description="Stable provider ID. Defaults to grok for compatibility.",
    )
    media_type: str = Field(
        default="video",
        description="Type of media to generate: 'video' or 'image'",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model to use. Auto-selected if omitted (e.g. grok-imagine-video)",
    )
    calibration_generation: int | None = Field(default=None, ge=0, strict=True,
        description="One-use local calibration reservation generation; never an ordinary render bypass")
    source_image: Optional[str] = Field(
        default=None,
        description="Base64 data URI or URL used as a generation reference",
    )
    reference_generation_id: Optional[str] = Field(
        default=None,
        description="Existing generation whose stored output becomes the reference image",
    )
    aspect_ratio: Optional[str] = Field(
        default=None,
        description="Aspect ratio. Provider default is used when omitted.",
    )
    duration: Optional[float] = Field(
        default=5,
        ge=1,
        le=15,
        description="Video duration in seconds (1-15, default 5). Video only.",
    )
    resolution: Optional[str] = Field(
        default=None,
        description="Output resolution. Provider default is used when omitted.",
    )
    num_outputs: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of outputs to generate (image only, default 1)",
    )


class PromptExpansionRequest(BaseModel):
    """Request body for POST /v1/media/prompts/expand."""

    prompt: str = Field(..., min_length=1, description="The user's short prompt")
    media_type: str = Field(default="video", description="Selects media.prompt_expansion.<media_type>")
    duration: Optional[float] = Field(default=None, ge=1, le=15)
    aspect_ratio: Optional[str] = Field(default=None)


class PromptExpansionResponse(BaseModel):
    prompt: str = Field(..., description="Expanded, model-ready prompt")
    original_prompt: str
    model: str = Field(..., description="Model that expanded it (global maintenance_model)")
    template_key: str


class MediaOutput(BaseModel):
    """A single generated media output."""

    file_path: Optional[str] = Field(default=None, description="Relative path in blob storage")
    thumbnail_path: Optional[str] = Field(default=None, description="Relative path to poster frame")
    file_size_bytes: Optional[int] = Field(default=None, description="File size in bytes")
    mime_type: Optional[str] = Field(default=None, description="MIME type of the media file")
    content_url: Optional[str] = Field(default=None, description="URL to stream the media content")
    thumbnail_url: Optional[str] = Field(default=None, description="URL to serve the poster frame")
    width: Optional[int] = Field(default=None, description="Media width in pixels")
    height: Optional[int] = Field(default=None, description="Media height in pixels")
    actual_duration: Optional[float] = Field(default=None, description="Actual duration in seconds")


class MediaGenerationResponse(BaseModel):
    """Response for a media generation job."""

    id: str = Field(..., description="Generation ID (prefixed with 'gen_')")
    status: str = Field(..., description="Job status: pending|processing|completed|failed")
    media_type: str = Field(..., description="Type of media: video or image")
    prompt: str = Field(..., description="Prompt sent to the provider")
    original_prompt: Optional[str] = Field(default=None, description="User's text before prompt expansion")
    negative_prompt: Optional[str] = Field(default=None, description="User's things-to-avoid text, if any")
    revised_prompt: Optional[str] = Field(default=None, description="Provider-revised prompt")
    progress: int = Field(default=0, description="Progress percentage 0-100")
    outputs: list[MediaOutput] = Field(default_factory=list, description="Generated outputs (populated when completed)")
    provider: Optional[str] = Field(default=None, description="Provider name (e.g. 'xai')")
    model: Optional[str] = Field(default=None, description="Model used for generation")
    aspect_ratio: Optional[str] = Field(default=None, description="Requested aspect ratio")
    duration: Optional[float] = Field(default=None, description="Requested duration (video)")
    resolution: Optional[str] = Field(default=None, description="Requested resolution")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    created_at: Optional[str] = Field(default=None, description="ISO 8601 creation timestamp")
    completed_at: Optional[str] = Field(default=None, description="ISO 8601 completion timestamp")


class MediaGenerationListResponse(BaseModel):
    """Paginated list of media generations."""

    items: list[MediaGenerationResponse] = Field(default_factory=list)
    total: int = Field(default=0)


class MediaProviderCapabilitiesResponse(BaseModel):
    """Provider options consumed by MCP and Studio clients."""

    provider: str
    label: str
    media_types: list[str]
    default_models: dict[str, str]
    models: dict[str, list[str]]
    image_input: bool = False
    aspect_ratios: dict[str, list[str]] = Field(default_factory=dict)
    resolutions: dict[str, list[str]] = Field(default_factory=dict)
    default_aspect_ratios: dict[str, str] = Field(default_factory=dict)
    default_resolutions: dict[str, str] = Field(default_factory=dict)
    durations: dict[str, list[int]] = Field(default_factory=dict)
    default_durations: dict[str, int] = Field(default_factory=dict)
    negative_prompt: bool = False


class MediaProviderListResponse(BaseModel):
    providers: list[MediaProviderCapabilitiesResponse] = Field(default_factory=list)
    default_provider: str = "grok"
