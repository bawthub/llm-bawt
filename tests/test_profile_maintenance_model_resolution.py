"""Profile maintenance uses the global background-job model, not an agent bot model."""

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
    with patch("llm_bawt.service.dependencies.get_profile_manager", return_value=_profiles()), \
         patch("llm_bawt.memory.profile_maintenance.ProfileCurationService") as maintenance:
        maintenance.return_value.run.return_value = SimpleNamespace(
            entity_id="selected-profile", candidates=1, applied=[{"op": "add"}],
            rejected=[], error=None,
        )
        result = service._process_profile_maintenance(_task())

    service._resolve_request_model.assert_not_called()
    service._get_background_client.assert_called_once_with(
        model_override="configured-model", job="profile_maintenance", allow_fallback=True,
    )
    maintenance.return_value.run.assert_called_once_with("selected-profile", "bot", False)
    assert result["candidates"] == 1
    assert result["error"] is None


def _profiles(candidates=1):
    profiles = MagicMock()
    profiles.attributes_with_status.return_value = [object()] * candidates
    return profiles


def test_profile_curation_without_candidates_resolves_no_model():
    service = _service(MagicMock())
    with patch("llm_bawt.service.dependencies.get_profile_manager",
               return_value=_profiles(0)):
        result = service._process_profile_maintenance(_task())
    service._get_background_client.assert_not_called()
    assert result["candidates"] == 0 and result["error"] is None


def test_profile_rebuild_reports_unusable_configured_model(monkeypatch):
    monkeypatch.setattr(
        "llm_bawt.runtime_settings.resolve_job_model",
        lambda _config, key: "unsupported-model" if key == "profile_maintenance_model" else None,
    )
    service = _service(None)
    with patch("llm_bawt.service.dependencies.get_profile_manager", return_value=_profiles()):
        result = service._process_profile_maintenance(_task())
    service._get_background_client.assert_called_once_with(
        model_override="unsupported-model", job="profile_maintenance", allow_fallback=True,
    )
    assert "unsupported-model" in result["error"]


def test_profile_rebuild_preserves_explicit_model_override(monkeypatch):
    monkeypatch.setattr("llm_bawt.runtime_settings.resolve_job_model", lambda *_: "configured-model")
    service = _service(None)
    with patch("llm_bawt.service.dependencies.get_profile_manager", return_value=_profiles()):
        service._process_profile_maintenance(_task("explicit-model"))
    service._get_background_client.assert_called_once_with(
        model_override="explicit-model", job="profile_maintenance", allow_fallback=False,
    )
