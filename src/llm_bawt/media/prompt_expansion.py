"""LLM prompt expansion for media generation (TASK-958).

Short user prompts ("a man talking, his head morphs into an emoji") leave
text-to-video models like Wan guessing; a detailed scene description is the
single biggest quality lever. The instructions live in the prompt registry
(``media.prompt_expansion.<media_type>``) so they are editable from BawtHub's
prompts UI; this module only owns the code default and the call.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Callable

from ..service.schemas import RawCompletionRequest, RawCompletionResponse

PROMPT_EXPANSION_KEY_PREFIX = "media.prompt_expansion."

VIDEO_PROMPT_EXPANSION_TEMPLATE = """\
You rewrite short video ideas into detailed prompts for a text-to-video diffusion model.

The model renders exactly what is described and nothing more, so spell out:
- Subject: who or what, with concrete appearance (age, clothing, colors, materials).
- Setting: where, with background details and time of day.
- Action: what happens over the clip, in order. If something transforms or changes, describe \
the start state, how it changes, and the end state explicitly and visually.
- Lighting and mood.
- Camera: shot size (close-up, medium, wide), angle and movement (static, slow push-in, pan).
- Style: e.g. cinematic, photorealistic, anime, claymation. Keep any style the user asked for.

Rules:
- Keep every element of the user's idea; add detail, never change the intent.
- The clip is about {duration} seconds long with a {aspect_ratio} frame. Describe only what fits \
in that time and composes well in that frame.
- Write one flowing paragraph of 80 to 150 words in present tense.
- Output only the prompt: no title, no quotes, no preamble, no explanation.

User's idea:
{prompt}"""


class PromptExpansionError(ValueError):
    """The request cannot be expanded (unknown media type, bad template)."""


@dataclass(frozen=True)
class ExpandedPrompt:
    prompt: str
    original_prompt: str
    model: str
    template_key: str


def prompt_expansion_key(media_type: str) -> str:
    return f"{PROMPT_EXPANSION_KEY_PREFIX}{(media_type or '').strip().lower()}"


def _clean(text: str) -> str:
    """Strip wrapping quotes/whitespace models add despite instructions."""
    cleaned = text.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'“”":
        cleaned = cleaned[1:-1].strip()
    return cleaned.strip("“”").strip()


def validate_reference_image(source_image: str) -> str:
    """Only inline, bounded raster images reach the utility vision client."""
    prefix, sep, payload = source_image.partition(",")
    if not sep or prefix not in {
        "data:image/jpeg;base64", "data:image/png;base64",
        "data:image/webp;base64", "data:image/gif;base64",
    }:
        raise PromptExpansionError("Reference image must be a JPEG, PNG, WebP or GIF data URI")
    if len(payload) > 7_000_000:
        raise PromptExpansionError("Reference image is too large (5 MB maximum)")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise PromptExpansionError("Reference image is not valid base64") from exc
    if not raw or len(raw) > 5 * 1024 * 1024:
        raise PromptExpansionError("Reference image is empty or larger than 5 MB")
    signatures = {
        "data:image/jpeg;base64": raw.startswith(b"\xff\xd8\xff"),
        "data:image/png;base64": raw.startswith(b"\x89PNG\r\n\x1a\n"),
        "data:image/gif;base64": raw.startswith((b"GIF87a", b"GIF89a")),
        "data:image/webp;base64": raw.startswith(b"RIFF") and raw[8:12] == b"WEBP",
    }
    if not signatures[prefix]:
        raise PromptExpansionError("Reference image bytes do not match its image type")
    return source_image


def expand_prompt(
    *,
    prompt: str,
    media_type: str,
    duration: float | None,
    aspect_ratio: str | None,
    complete: Callable[[RawCompletionRequest], RawCompletionResponse],
    source_image: str | None = None,
) -> ExpandedPrompt:
    """Render the registry template and run it through the utility completion.

    ``complete`` is the botless utility completion (global maintenance_model);
    its HTTP errors propagate unchanged so callers surface them verbatim.
    """
    from ..prompt_registry import get_prompt_resolver

    original = (prompt or "").strip()
    if not original:
        raise PromptExpansionError("Prompt is empty")
    key = prompt_expansion_key(media_type)
    resolver = get_prompt_resolver()
    if resolver.definition_for(key) is None:
        raise PromptExpansionError(f"No prompt expansion template for media type '{media_type}'")
    try:
        rendered = resolver.render(key=key, variables={
            "prompt": original,
            "duration": f"{duration:g}" if duration else "5",
            "aspect_ratio": aspect_ratio or "16:9",
        })
    except (KeyError, IndexError, ValueError) as exc:
        raise PromptExpansionError(f"Prompt template '{key}' failed to render: {exc}") from exc

    image = validate_reference_image(source_image) if source_image else None
    result = complete(RawCompletionRequest(
        prompt=rendered, image_url=image, max_tokens=600, temperature=0.7,
    ))
    expanded = _clean(result.content)
    if not expanded:
        raise PromptExpansionError("The model returned an empty prompt")
    return ExpandedPrompt(prompt=expanded, original_prompt=original, model=result.model, template_key=key)
