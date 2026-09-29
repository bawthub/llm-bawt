"""Model-specific raw completions never borrow an agent bot or silently fall back."""

import base64
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from llm_bawt.service.routes.llm import raw_completion
from llm_bawt.service.schemas import RawCompletionRequest


def test_explicit_model_uses_direct_client_without_bot_or_history():
    client = MagicMock()
    client.query.return_value = '{"headline":"Next"}'
    service = MagicMock()
    service._client_cache = {"different-model": MagicMock()}
    service._get_background_client.return_value = (client, "grok-4.3")
    with patch("llm_bawt.service.routes.llm.get_service", return_value=service):
        result = raw_completion(RawCompletionRequest(prompt="what next?", system="curate", model="grok-4.3"))
    assert result.model == "grok-4.3"
    assert result.content == '{"headline":"Next"}'
    service._get_background_client.assert_called_once_with(model_override="grok-4.3")
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
    with patch("llm_bawt.service.routes.llm.get_service", return_value=service), \
         patch("llm_bawt.runtime_settings.resolve_job_model", return_value="configured-maintenance"):
        result = raw_completion(RawCompletionRequest(prompt="describe the subject", image_url=image))
    assert result.content == "Expanded image scene"
    sent = client.query.call_args.kwargs["messages"][-1].to_api_format()
    assert sent["content"] == [
        {"type": "text", "text": "describe the subject"},
        {"type": "image_url", "image_url": {"url": image, "detail": "low"}},
    ]


def test_image_model_rejection_does_not_silently_retry_text_only():
    client = MagicMock()
    client.query.side_effect = ValueError("unsupported image input")
    service = MagicMock()
    service._get_background_client.return_value = (client, "configured-maintenance")
    image = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nimage").decode()
    with patch("llm_bawt.service.routes.llm.get_service", return_value=service), \
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
    with patch("llm_bawt.service.routes.llm.get_service", return_value=service), pytest.raises(HTTPException) as exc:
        raw_completion(RawCompletionRequest(prompt="what next?", model="agent-only"))
    assert exc.value.status_code == 422
    service._client_cache["loaded"].query.assert_not_called()


def test_no_model_uses_global_maintenance_setting_not_loaded_client():
    client = MagicMock()
    client.query.return_value = "result"
    service = MagicMock()
    service._client_cache = {"loaded": MagicMock()}
    service._get_background_client.return_value = (client, "configured-maintenance")
    with patch("llm_bawt.service.routes.llm.get_service", return_value=service), \
         patch("llm_bawt.runtime_settings.resolve_job_model", return_value="configured-maintenance") as setting:
        result = raw_completion(RawCompletionRequest(prompt="hello"))
    assert result.model == "configured-maintenance"
    setting.assert_called_once_with(service.config, "maintenance_model")
    service._get_background_client.assert_called_once_with(model_override="configured-maintenance")
    service._client_cache["loaded"].query.assert_not_called()


def test_missing_maintenance_setting_does_not_fall_back_to_loaded_model():
    service = MagicMock()
    service._client_cache = {"loaded": MagicMock()}
    with patch("llm_bawt.service.routes.llm.get_service", return_value=service), \
         patch("llm_bawt.runtime_settings.resolve_job_model", return_value=None), \
         pytest.raises(HTTPException) as exc:
        raw_completion(RawCompletionRequest(prompt="hello"))
    assert exc.value.status_code == 503
    service._client_cache["loaded"].query.assert_not_called()
