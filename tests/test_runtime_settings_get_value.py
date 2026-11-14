"""RuntimeSettingsStore.get_value reads one key without scanning the scope."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import create_engine, event, insert

from llm_bawt.runtime_settings import RuntimeSetting, RuntimeSettingsStore


def _store():
    engine = create_engine("sqlite://")
    RuntimeSetting.__table__.create(engine)
    rows = [
        ("global", "*", "provider.wanted", json.dumps({"status": "connected"})),
        ("global", "*", "unrelated", "not json"),
        ("bot", "*", "provider.wanted", json.dumps("other scope")),
    ]
    with engine.begin() as conn:
        for scope_type, scope_id, key, value in rows:
            conn.execute(insert(RuntimeSetting.__table__).values(
                scope_type=scope_type, scope_id=scope_id, key=key,
                value_json=value, updated_at=datetime.now(timezone.utc),
            ))
    store = object.__new__(RuntimeSettingsStore)
    store.engine = engine
    return store, engine


def test_get_value_returns_only_requested_key():
    store, engine = _store()
    statements = []
    event.listen(engine, "before_cursor_execute",
                 lambda _c, _cur, sql, params, *_: statements.append(params))

    assert store.get_value("global", "*", "provider.wanted") == {"status": "connected"}
    assert len(statements) == 1
    assert "provider.wanted" in statements[0]


def test_get_value_missing_or_invalid_is_none():
    store, _ = _store()
    assert store.get_value("global", "*", "absent") is None
    assert store.get_value("global", "*", "unrelated") is None
