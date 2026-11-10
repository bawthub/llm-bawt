"""Focused tests for unified profile route behavior."""

from types import SimpleNamespace

import pytest
from llm_bawt.service.routes import profiles as profile_routes
from llm_bawt.service.schemas import ProfileUpdateRequest
from pydantic import ValidationError


class _ProfileManager:
    def __init__(self) -> None:
        self.avatar: str | None = "🐕"
        self.avatar_render: str | None = "data:image/svg+xml,old"
        self.is_service = False
        self.update_calls: list[dict[str, object]] = []

    def update_profile(self, _entity_type, _entity_id, **fields):
        self.update_calls.append(fields)
        if fields.get("avatar") is not None:
            self.avatar = str(fields["avatar"]).strip() or None
        if fields.get("avatar_render") is not None:
            self.avatar_render = str(fields["avatar_render"]) or None
        if fields.get("is_service") is not None:
            self.is_service = fields["is_service"]
        return self.get_profile(_entity_type, _entity_id)

    def get_profile(self, _entity_type, _entity_id):
        return SimpleNamespace(
            email="nick@example.com",
            display_name="Nick",
            description=None,
            color="#012f7b",
            avatar=self.avatar,
            avatar_render=self.avatar_render,
            is_service=self.is_service,
            summary=None,
            summary_updated_at=None,
            created_at=None,
        )

    def get_all_attributes(self, _entity_type, _entity_id):
        return []


def test_update_typed_profile_clears_cached_avatar_render(monkeypatch) -> None:
    manager = _ProfileManager()
    monkeypatch.setattr(profile_routes, "get_service", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr(profile_routes, "get_profile_manager", lambda _config: manager)
    monkeypatch.setattr("llm_bawt.media.avatar.resolve_avatar_render", lambda _avatar: None)

    result = profile_routes.update_typed_profile(
        "user",
        "nick",
        ProfileUpdateRequest(avatar=""),
    )

    assert manager.update_calls[0]["avatar"] == ""
    assert manager.update_calls[0]["avatar_render"] == ""
    assert result.avatar is None
    assert result.avatar_render is None


def test_service_flag_round_trips_and_unrelated_edits_preserve_it(monkeypatch) -> None:
    manager = _ProfileManager()
    monkeypatch.setattr(profile_routes, "get_service", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr(profile_routes, "get_profile_manager", lambda _config: manager)

    assert profile_routes.update_typed_profile(
        "user", "automation", ProfileUpdateRequest(is_service=True),
    ).is_service is True
    assert profile_routes.update_typed_profile(
        "user", "automation", ProfileUpdateRequest(display_name="Renamed"),
    ).is_service is True
    assert profile_routes.update_typed_profile(
        "user", "automation", ProfileUpdateRequest(is_service=False),
    ).is_service is False


def test_service_flag_is_applied_on_profile_creation(monkeypatch) -> None:
    manager = _ProfileManager()
    exists = False
    update = manager.update_profile

    def update_if_exists(*args, **kwargs):
        return update(*args, **kwargs) if exists else None

    def create(*_args):
        nonlocal exists
        exists = True

    monkeypatch.setattr(manager, "update_profile", update_if_exists)
    monkeypatch.setattr(manager, "get_or_create_profile", create, raising=False)
    monkeypatch.setattr(profile_routes, "get_service", lambda: SimpleNamespace(config=object()))
    monkeypatch.setattr(profile_routes, "get_profile_manager", lambda _config: manager)
    assert profile_routes.update_typed_profile(
        "user", "automation", ProfileUpdateRequest(is_service=True),
    ).is_service is True


@pytest.mark.parametrize("value", ["true", "false", 1, 0])
def test_service_flag_rejects_coerced_boolean_values(value) -> None:
    with pytest.raises(ValidationError):
        ProfileUpdateRequest(is_service=value)
