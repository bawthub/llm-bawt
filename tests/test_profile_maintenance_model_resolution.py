"""Profile maintenance uses the global background-job model, not an agent bot model."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm_bawt.service.background_service import BackgroundService
from llm_bawt.service.tasks import Task, TaskType


def _task(model=None):
    return Task(
        task_type=TaskType.PROFILE_MAINTENANCE,
        bot_id="agent-bot",
        user_id="selected-profile",
        payload={"entity_type": "bot", "entity_id": "selected-profile", **({"model": model} if model else {})},
    )


def _service(client):
    service = BackgroundService.__new__(BackgroundService)
    service.config = MagicMock()
    service._resolve_request_model = MagicMock(side_effect=AssertionError("agent resolver must not run"))
    service._get_background_client = MagicMock(return_value=(client, "configured-model" if client else None))
    return service


def test_profile_rebuild_uses_global_job_model_without_agent_override(monkeypatch):
    monkeypatch.setattr(
        "llm_bawt.runtime_settings.resolve_job_model",
        lambda _config, key: "configured-model" if key == "maintenance_model" else None,
    )
    service = _service(MagicMock())
    with ThreadPoolExecutor(max_workers=1) as pool, \
         patch("llm_bawt.service.dependencies.get_profile_manager", return_value=MagicMock()), \
         patch("llm_bawt.memory.profile_maintenance.ProfileMaintenanceService") as maintenance:
        service._bg_executor = pool
        maintenance.return_value.run.return_value = SimpleNamespace(
            entity_id="selected-profile", attributes_before=3, attributes_after=3,
            categories_updated=["preferences"], error=None,
        )
        result = asyncio.run(service._process_profile_maintenance(_task()))

    service._resolve_request_model.assert_not_called()
    service._get_background_client.assert_called_once_with(model_override="configured-model")
    maintenance.return_value.run.assert_called_once_with("selected-profile", "bot", False)
    assert result["attributes_before"] == 3
    assert result["error"] is None


def test_profile_rebuild_reports_unusable_configured_model(monkeypatch):
    monkeypatch.setattr(
        "llm_bawt.runtime_settings.resolve_job_model",
        lambda _config, key: "unsupported-model" if key == "profile_maintenance_model" else None,
    )
    service = _service(None)
    result = asyncio.run(service._process_profile_maintenance(_task()))
    service._get_background_client.assert_called_once_with(model_override="unsupported-model")
    assert "unsupported-model" in result["error"]


def test_profile_rebuild_preserves_explicit_model_override(monkeypatch):
    monkeypatch.setattr("llm_bawt.runtime_settings.resolve_job_model", lambda *_: "configured-model")
    service = _service(None)
    asyncio.run(service._process_profile_maintenance(_task("explicit-model")))
    service._get_background_client.assert_called_once_with(model_override="explicit-model")
