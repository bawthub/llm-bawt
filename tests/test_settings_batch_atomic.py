"""TASK-2204: a settings batch is applied whole or not at all.

``/v1/settings/batch`` used to commit each item in its own session, so a later
item failing in the database left the earlier ones live while the client saw
an error, and cache invalidation never ran.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore
from llm_bawt.service.routes import settings as settings_routes
from llm_bawt.service.schemas import (
    RuntimeSettingBatchItem,
    RuntimeSettingBatchUpsertRequest,
    RuntimeSettingUpsertRequest,
)

LONG_KEY = "k" * 129


@pytest.fixture
def env(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine, tables=[RuntimeSetting.__table__])
    # A real database error on one key, standing in for a constraint the
    # route cannot see in advance.
    with engine.begin() as conn:
        conn.execute(text(
            "CREATE TRIGGER refuse_boom BEFORE INSERT ON runtime_settings "
            "WHEN NEW.key = 'boom' BEGIN SELECT RAISE(ABORT, 'boom refused'); END"
        ))
    store = object.__new__(RuntimeSettingsStore)
    store.config = None
    store.engine = engine
    invalidated: list[tuple[str, ...]] = []
    service = SimpleNamespace(config=None, _default_bot="examplebot")
    monkeypatch.setattr(settings_routes, "get_service", lambda: service)
    monkeypatch.setattr(settings_routes, "get_runtime_settings_store", lambda _config: store)
    monkeypatch.setattr(
        settings_routes, "_invalidate_bot_instance_cache", lambda _s, bot: invalidated.append(("bot", bot)),
    )
    monkeypatch.setattr(
        settings_routes, "_invalidate_all_instance_cache", lambda _s: invalidated.append(("all",)),
    )
    store.set_value("global", "*", "first_setting", 1)
    return SimpleNamespace(store=store, invalidated=invalidated)


def _global(env):
    return env.store.get_scope_settings("global", "*")


def _batch(*items):
    return RuntimeSettingBatchUpsertRequest(items=[RuntimeSettingBatchItem(**item) for item in items])


def test_database_failure_on_a_later_item_rolls_back_the_whole_batch(env):
    with pytest.raises(Exception):
        settings_routes.batch_upsert_runtime_settings(_batch(
            {"scope_type": "global", "key": "first_setting", "value": 2},
            {"scope_type": "global", "key": "new_setting", "value": "x"},
            {"scope_type": "global", "key": "boom", "value": True},
        ))
    assert _global(env) == {"first_setting": 1}


def test_overlong_key_is_refused_before_anything_is_written(env):
    # model_construct skips schema validation, as a non-HTTP caller would.
    request = RuntimeSettingBatchUpsertRequest.model_construct(items=[
        RuntimeSettingBatchItem(scope_type="global", key="first_setting", value=2),
        RuntimeSettingBatchItem.model_construct(scope_type="global", scope_id=None, key=LONG_KEY, value=1),
    ])
    with pytest.raises(HTTPException) as raised:
        settings_routes.batch_upsert_runtime_settings(request)
    assert raised.value.status_code == 400
    assert _global(env) == {"first_setting": 1}
    assert env.invalidated == []


def test_overlong_scope_id_is_refused(env):
    request = RuntimeSettingBatchUpsertRequest.model_construct(items=[
        RuntimeSettingBatchItem.model_construct(scope_type="bot", scope_id="b" * 129, key="x", value=1),
    ])
    with pytest.raises(HTTPException) as raised:
        settings_routes.batch_upsert_runtime_settings(request)
    assert raised.value.status_code == 400


@pytest.mark.parametrize("model", [RuntimeSettingBatchItem, RuntimeSettingUpsertRequest])
def test_request_schemas_reject_keys_the_column_cannot_hold(model):
    with pytest.raises(ValidationError):
        model(scope_type="global", key=LONG_KEY, value=1)
    model(scope_type="global", key="k" * 128, value=1)


def test_successful_batch_applies_everything_then_invalidates(env):
    result = settings_routes.batch_upsert_runtime_settings(_batch(
        {"scope_type": "global", "key": "first_setting", "value": 2},
        {"scope_type": "bot", "scope_id": "examplebot", "key": "temperature", "value": 0.5},
    ))
    assert result["applied_count"] == 2
    assert _global(env) == {"first_setting": 2}
    assert env.store.get_scope_settings("bot", "examplebot") == {"temperature": 0.5}
    assert env.invalidated == [("all",)]


def test_failed_batch_does_not_invalidate_caches(env):
    with pytest.raises(Exception):
        settings_routes.batch_upsert_runtime_settings(_batch(
            {"scope_type": "bot", "scope_id": "examplebot", "key": "temperature", "value": 0.5},
            {"scope_type": "global", "key": "boom", "value": True},
        ))
    assert env.invalidated == []
    assert env.store.get_scope_settings("bot", "examplebot") == {}
