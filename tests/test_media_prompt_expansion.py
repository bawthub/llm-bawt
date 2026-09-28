"""Studio prompt expansion (TASK-958): registry-owned template, maintenance model."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from llm_bawt.media.prompt_expansion import (
    PromptExpansionError,
    expand_prompt,
    prompt_expansion_key,
)
from llm_bawt.media.schemas import MediaGenerationRequest, PromptExpansionRequest
from llm_bawt.prompt_registry import PromptResolver
from llm_bawt.service.schemas import RawCompletionResponse


def _resolver() -> PromptResolver:
    resolver = PromptResolver.__new__(PromptResolver)
    resolver.store = MagicMock(engine=None)
    return resolver


def _completion(content: str):
    calls = []

    def complete(request):
        calls.append(request)
        return RawCompletionResponse(content=content, model="maint-model", elapsed_ms=1.0)

    return complete, calls


def test_video_template_is_registered_and_its_default_validates():
    resolver = _resolver()
    resolved = resolver.resolve("media.prompt_expansion.video")
    assert resolved is not None and resolved.source == "code_default"
    check = resolver.validate(key="media.prompt_expansion.video", body=resolved.body)
    assert check["valid"], check["errors"]
    assert set(check["placeholders"]) == {"prompt", "duration", "aspect_ratio"}


def test_expand_renders_registry_template_and_leaves_model_to_maintenance_setting():
    complete, calls = _completion('  "A detailed scene."  ')
    with patch("llm_bawt.prompt_registry.get_prompt_resolver", return_value=_resolver()):
        result = expand_prompt(prompt=" a man talking ", media_type="video",
                               duration=8, aspect_ratio="9:16", complete=complete)
    assert result.prompt == "A detailed scene."
    assert result.original_prompt == "a man talking"
    assert result.model == "maint-model"
    assert result.template_key == "media.prompt_expansion.video"
    sent = calls[0]
    assert sent.model is None  # inherits global maintenance_model
    assert "a man talking" in sent.prompt
    assert "8 seconds" in sent.prompt and "9:16" in sent.prompt


def test_edited_template_in_registry_changes_what_is_sent():
    resolver = _resolver()
    row = MagicMock(title=None, category=None, format=None, body="CUSTOM {prompt} {duration} {aspect_ratio}",
                    required_vars_json=None, metadata_json=None, scope_type="global", scope_id="*", updated_at=None)
    resolver.store = MagicMock(engine=object())
    resolver.store.get_exact.return_value = row
    complete, calls = _completion("expanded")
    with patch("llm_bawt.prompt_registry.get_prompt_resolver", return_value=resolver):
        expand_prompt(prompt="cat", media_type="video", duration=None, aspect_ratio=None, complete=complete)
    assert calls[0].prompt == "CUSTOM cat 5 16:9"


def test_unknown_media_type_and_empty_prompt_fail_visibly():
    complete, calls = _completion("x")
    with patch("llm_bawt.prompt_registry.get_prompt_resolver", return_value=_resolver()):
        with pytest.raises(PromptExpansionError):
            expand_prompt(prompt="cat", media_type="audio", duration=5, aspect_ratio="1:1", complete=complete)
        with pytest.raises(PromptExpansionError):
            expand_prompt(prompt="   ", media_type="video", duration=5, aspect_ratio="1:1", complete=complete)
    assert calls == []
    assert prompt_expansion_key(" Video ") == "media.prompt_expansion.video"


def test_route_maps_expansion_errors_to_400_and_passes_maintenance_errors_through():
    from llm_bawt.service.routes import media

    with patch("llm_bawt.media.prompt_expansion.expand_prompt", side_effect=PromptExpansionError("bad")), \
         pytest.raises(HTTPException) as exc:
        asyncio.run(media.expand_media_prompt(PromptExpansionRequest(prompt="cat")))
    assert exc.value.status_code == 400

    missing = HTTPException(status_code=503, detail="No maintenance model configured")
    with patch("llm_bawt.media.prompt_expansion.expand_prompt", side_effect=missing), \
         pytest.raises(HTTPException) as exc:
        asyncio.run(media.expand_media_prompt(PromptExpansionRequest(prompt="cat")))
    assert exc.value.status_code == 503


def test_original_prompt_is_kept_only_when_it_differs_from_the_sent_prompt():
    from llm_bawt.service.routes.media import _original_prompt

    assert _original_prompt(MediaGenerationRequest(prompt="long", original_prompt="short")) == "short"
    assert _original_prompt(MediaGenerationRequest(prompt="same", original_prompt=" same ")) is None
    assert _original_prompt(MediaGenerationRequest(prompt="raw")) is None


def test_local_video_capabilities_advertise_only_the_enforced_wan_profile():
    from llm_bawt.media.clients import media_provider_registry
    from llm_bawt.media.gpu_profile import CALIBRATION_PROFILE, require_calibration_profile

    caps = media_provider_registry.capabilities("local-video")
    assert caps.aspect_ratios["video"] == tuple(CALIBRATION_PROFILE["aspect_ratios"])
    assert caps.resolutions["video"] == (CALIBRATION_PROFILE["resolution"],)
    assert caps.durations["video"] == (CALIBRATION_PROFILE["duration"],)
    assert caps.image_input is True
    # Every combination the UI can offer passes the submit gate.
    for aspect in caps.aspect_ratios["video"]:
        for image_conditioned in (False, True):
            require_calibration_profile({
                "resolution": caps.default_resolutions["video"], "aspect_ratio": aspect,
                "duration": caps.default_durations["video"], "num_outputs": 1, "image_conditioned": image_conditioned,
            })
    with pytest.raises(ValueError):
        require_calibration_profile({"resolution": "720p", "aspect_ratio": "16:9", "duration": 5,
                                     "num_outputs": 1, "image_conditioned": False})
