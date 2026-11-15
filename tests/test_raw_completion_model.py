"""Model-specific raw completions never borrow an agent bot or silently fall back."""

import base64
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from llm_bawt.service import utility_completion as llm
from llm_bawt.service.routes.llm import raw_completion
from llm_bawt.service.schemas import RawCompletionRequest


def test_explicit_model_uses_direct_client_without_bot_or_history():
    client = MagicMock()
    client.query.return_value = '{"headline":"Next"}'
    service = MagicMock()
    service._client_cache = {"different-model": MagicMock()}
    service._get_background_client.return_value = (client, "grok-4.3")
    with patch("llm_bawt.service.utility_completion.get_service", return_value=service):
        result = raw_completion(RawCompletionRequest(prompt="what next?", system="curate", model="grok-4.3"))
    assert result.model == "grok-4.3"
    assert result.content == '{"headline":"Next"}'
    service._get_background_client.assert_called_once_with(
        model_override="grok-4.3", job="utility_completion", allow_fallback=False,
    )
    kwargs = client.query.call_args.kwargs
    assert kwargs["stream"] is False and kwargs["plaintext_output"] is True
    assert [(message.role, message.content) for message in kwargs["messages"]] == [
        ("system", "curate"), ("user", "what next?"),
    ]


def test_reference_image_reaches_configured_utility_client_as_image_part():
    client = MagicMock()
    client.query.return_value = "Expanded image scene"
    service = MagicMock()
    service._get_background_client.return_value = (client, "configured-maintenance")
    image = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nimage").decode()
    with patch("llm_bawt.service.utility_completion.get_service", return_value=service), \
         patch("llm_bawt.runtime_settings.resolve_job_model", return_value="configured-maintenance"):
        result = raw_completion(RawCompletionRequest(prompt="describe the subject", image_url=image))
    assert result.content == "Expanded image scene"
    sent = client.query.call_args.kwargs["messages"][-1].to_api_format()
    assert sent["content"] == [
        {"type": "text", "text": "describe the subject"},
        {"type": "image_url", "image_url": {"url": image, "detail": "low"}},
    ]


def _image(tag: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n" + tag).decode()


@pytest.mark.parametrize("legacy", [None, _image(b"legacy")])
def test_all_images_reach_configured_model_in_order(legacy):
    images = [_image(b"one"), _image(b"two")]
    client = MagicMock()
    client.query.return_value = "Plan with both attachments"
    service = MagicMock()
    service._get_background_client.return_value = (client, "configured-maintenance")
    with patch("llm_bawt.service.utility_completion.get_service", return_value=service), \
         patch("llm_bawt.runtime_settings.resolve_job_model", return_value="configured-maintenance"):
        result = raw_completion(RawCompletionRequest(
            prompt="plan", image_url=legacy, image_urls=images,
        ))
    assert result.model == "configured-maintenance"
    service._get_background_client.assert_called_once_with(
        model_override="configured-maintenance", job="utility_completion", allow_fallback=True,
    )
    sent = client.query.call_args.kwargs["messages"][-1].to_api_format()
    assert sent["content"] == [
        {"type": "text", "text": "plan"},
        *[{"type": "image_url", "image_url": {"url": image, "detail": "low"}}
          for image in ([legacy] if legacy else []) + images],
    ]
    client.query.assert_called_once()


@pytest.mark.parametrize("invalid", [
    "https://example.test/image.png",
    "data:image/png;base64,not-base64",
    "data:image/png;base64," + base64.b64encode(b"not a PNG").decode(),
    "",
])
def test_invalid_list_image_rejects_whole_request_before_model_query(invalid):
    client = MagicMock()
    client.query.return_value = "must not be called"
    service = MagicMock()
    service._get_background_client.return_value = (client, "configured-maintenance")
    with patch("llm_bawt.service.utility_completion.get_service", return_value=service), \
         patch("llm_bawt.runtime_settings.resolve_job_model", return_value="configured-maintenance"), \
         pytest.raises(HTTPException) as exc:
        raw_completion(RawCompletionRequest(prompt="plan", image_urls=[_image(b"one"), invalid]))
    assert exc.value.status_code == 400
    assert "image 2" in exc.value.detail
    client.query.assert_not_called()


@pytest.mark.parametrize("legacy", [None, _image(b"legacy")])
def test_image_count_overflow_rejects_before_model_query(legacy):
    client = MagicMock()
    service = MagicMock()
    service._get_background_client.return_value = (client, "configured-maintenance")
    images = [_image(b"one")] * (33 if legacy is None else 32)
    with patch("llm_bawt.service.utility_completion.get_service", return_value=service), \
         patch("llm_bawt.runtime_settings.resolve_job_model", return_value="configured-maintenance"), \
         pytest.raises(HTTPException) as exc:
        raw_completion(RawCompletionRequest(prompt="plan", image_url=legacy, image_urls=images))
    assert exc.value.status_code == 400
    assert "32" in exc.value.detail
    client.query.assert_not_called()


def test_total_decoded_image_bytes_include_legacy_image(monkeypatch):
    monkeypatch.setattr(llm, "MAX_UTILITY_IMAGE_BYTES", 20, raising=False)
    client = MagicMock()
    service = MagicMock()
    service._get_background_client.return_value = (client, "configured-maintenance")
    with patch("llm_bawt.service.utility_completion.get_service", return_value=service), \
         patch("llm_bawt.runtime_settings.resolve_job_model", return_value="configured-maintenance"), \
         pytest.raises(HTTPException) as exc:
        raw_completion(RawCompletionRequest(
            prompt="plan", image_url=_image(b"one"), image_urls=[_image(b"two")],
        ))
    assert exc.value.status_code == 400
    assert "total" in exc.value.detail.lower()
    client.query.assert_not_called()


def test_multi_image_model_rejection_does_not_retry_text_only():
    client = MagicMock()
    client.query.side_effect = ValueError("unsupported image input")
    service = MagicMock()
    service._get_background_client.return_value = (client, "configured-maintenance")
    with patch("llm_bawt.service.utility_completion.get_service", return_value=service), \
         patch("llm_bawt.runtime_settings.resolve_job_model", return_value="configured-maintenance"), \
         pytest.raises(HTTPException) as exc:
        raw_completion(RawCompletionRequest(prompt="plan", image_urls=[_image(b"one"), _image(b"two")]))
    assert exc.value.status_code == 422
    assert "vision-capable" in exc.value.detail
    client.query.assert_called_once()


def test_empty_image_list_keeps_text_only_request():
    client = MagicMock()
    client.query.return_value = "Text-only plan"
    service = MagicMock()
    service._get_background_client.return_value = (client, "configured-maintenance")
    with patch("llm_bawt.service.utility_completion.get_service", return_value=service), \
         patch("llm_bawt.runtime_settings.resolve_job_model", return_value="configured-maintenance"):
        raw_completion(RawCompletionRequest(prompt="plan", image_urls=[]))
    sent = client.query.call_args.kwargs["messages"][-1].to_api_format()
    assert sent["content"] == "plan"


def test_image_model_rejection_does_not_silently_retry_text_only():
    client = MagicMock()
    client.query.side_effect = ValueError("unsupported image input")
    service = MagicMock()
    service._get_background_client.return_value = (client, "configured-maintenance")
    image = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nimage").decode()
    with patch("llm_bawt.service.utility_completion.get_service", return_value=service), \
         patch("llm_bawt.runtime_settings.resolve_job_model", return_value="configured-maintenance"), \
         pytest.raises(HTTPException) as exc:
        raw_completion(RawCompletionRequest(prompt="describe", image_url=image))
    assert exc.value.status_code == 422
    assert "vision-capable" in exc.value.detail
    client.query.assert_called_once()


def test_missing_direct_model_fails_instead_of_using_loaded_client():
    service = MagicMock()
    service._client_cache = {"loaded": MagicMock()}
    service._get_background_client.return_value = (None, None)
    with patch("llm_bawt.service.utility_completion.get_service", return_value=service), pytest.raises(HTTPException) as exc:
        raw_completion(RawCompletionRequest(prompt="what next?", model="agent-only"))
    assert exc.value.status_code == 422
    service._client_cache["loaded"].query.assert_not_called()


def test_no_model_uses_global_maintenance_setting_not_loaded_client():
    client = MagicMock()
    client.query.return_value = "result"
    service = MagicMock()
    service._client_cache = {"loaded": MagicMock()}
    service._get_background_client.return_value = (client, "configured-maintenance")
    with patch("llm_bawt.service.utility_completion.get_service", return_value=service), \
         patch("llm_bawt.runtime_settings.resolve_job_model", return_value="configured-maintenance") as setting:
        result = raw_completion(RawCompletionRequest(prompt="hello"))
    assert result.model == "configured-maintenance"
    setting.assert_called_once_with(service.config, "maintenance_model")
    service._get_background_client.assert_called_once_with(
        model_override="configured-maintenance", job="utility_completion", allow_fallback=True,
    )
    service._client_cache["loaded"].query.assert_not_called()


def test_missing_maintenance_setting_does_not_fall_back_to_loaded_model():
    service = MagicMock()
    service._client_cache = {"loaded": MagicMock()}
    with patch("llm_bawt.service.utility_completion.get_service", return_value=service), \
         patch("llm_bawt.runtime_settings.resolve_job_model", return_value=None), \
         pytest.raises(HTTPException) as exc:
        raw_completion(RawCompletionRequest(prompt="hello"))
    assert exc.value.status_code == 503
    service._client_cache["loaded"].query.assert_not_called()
