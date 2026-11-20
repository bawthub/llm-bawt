"""_persist_bot_profile provisions storage only after a successful create."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from llm_bawt.runtime_settings import BotAlreadyExists
from llm_bawt.service.routes import settings as routes

PAYLOAD = {"slug": "fresh", "name": "Fresh", "system_prompt": "hi"}


@contextmanager
def _env(*, previous=None, create_error=None):
    profile = SimpleNamespace(slug="fresh")
    store = MagicMock(engine=object())
    store.get.return_value = previous
    store.upsert.return_value = profile
    create = MagicMock()
    if create_error:
        create.return_value.create.side_effect = create_error
    else:
        create.return_value.create.return_value = profile
    provision = AsyncMock()
    with patch.multiple(
        routes,
        get_service=MagicMock(return_value=SimpleNamespace(config=object())),
        get_bot_profile_store=MagicMock(return_value=store),
        get_runtime_settings_store=MagicMock(return_value=SimpleNamespace(engine=object())),
        _validate_profile_payload=MagicMock(),
        ProfileRouteGate=MagicMock(),
        ProfileSaveGuard=MagicMock(),
        ProfileSwitchGate=MagicMock(),
        BotCreateTransaction=create,
        _provision_bot_storage=provision,
        _reload_bot_registry=MagicMock(),
        _reload_service_model_catalog=MagicMock(),
        _invalidate_bot_instance_cache=MagicMock(),
        _clear_session_model_overrides=MagicMock(),
        _rotate_threads_on_runtime_switch=MagicMock(return_value=[]),
        _to_profile_response=MagicMock(return_value="response"),
        _effective_bot_settings=MagicMock(return_value={}),
        _profile_tags=MagicMock(return_value={}),
        _publish_bot_profile_changed=AsyncMock(),
        _publish_runtime_switch_threads=AsyncMock(),
    ):
        yield provision


def test_successful_create_provisions_the_new_bots_storage():
    with _env() as provision:
        assert asyncio.run(routes._persist_bot_profile(dict(PAYLOAD), create_only=True)) == "response"
    provision.assert_awaited_once()
    assert provision.await_args.args[1] == "fresh"


def test_profile_update_does_not_provision():
    with _env(previous=SimpleNamespace(slug="fresh", avatar=None)) as provision:
        asyncio.run(routes._persist_bot_profile(dict(PAYLOAD), create_only=False))
    provision.assert_not_awaited()


def test_refused_create_does_not_provision():
    with _env(create_error=BotAlreadyExists("fresh")) as provision:
        with pytest.raises(HTTPException) as refused:
            asyncio.run(routes._persist_bot_profile(dict(PAYLOAD), create_only=True))
    assert refused.value.status_code == 409
    provision.assert_not_awaited()
