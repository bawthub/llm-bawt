"""Private PostgreSQL proof for working memory slice 1 (TASK-1281, docs/WORKING_MEMORY.md).

Store, trusted bot path, owner REST path and setting validation against the
real schema. Protected-bot cases use a synthetic slug, never Mira's data.
"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import text

from llm_bawt.runtime_settings import BotProfileStore, RuntimeSettingsStore
from llm_bawt.service.routes import settings as settings_routes
from llm_bawt.service.routes import working_memory as wm_routes
from llm_bawt.service.schemas import (
    RuntimeSettingBatchItem,
    RuntimeSettingBatchUpsertRequest,
    RuntimeSettingUpsertRequest,
)
from llm_bawt.task_turn_context import TaskTurnContext
from llm_bawt.utils.db import get_shared_engine
from llm_bawt.working_memory import policy as policy_module
from llm_bawt.working_memory.contracts import WorkingMemoryError
from llm_bawt.working_memory.service import WorkingMemoryService

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]

KEY = "working_memory_enabled"
# Columns a change row may hold after a hard delete (11.3), plus the key on the create row.
SURVIVING = {
    "seq", "bot_id", "user_id", "ns_version", "item_id", "revision", "kind", "actor_kind",
    "actor_id", "session_id", "turn_id", "created_at", "idempotency_key",
}


@pytest.fixture
def wm(private_app_pg, monkeypatch):
    engine = get_shared_engine(private_app_pg)
    profiles = BotProfileStore(private_app_pg)
    settings = RuntimeSettingsStore(private_app_pg)
    tag = uuid.uuid4().hex[:10]
    bots = {
        "on": f"test-wm-on-{tag}",
        "off": f"test-wm-off-{tag}",
        "codex": f"test-wm-codex-{tag}",
        "protected": f"test-wm-protected-{tag}",
    }
    for name, slug in bots.items():
        backend = "codex" if name == "codex" else "claude-code"
        profiles.upsert({"slug": slug, "bot_type": "agent", "agent_backend": backend, "harness": backend})
    settings.set_value("bot", bots["on"], KEY, True)
    settings.set_value("bot", bots["codex"], KEY, True)  # stored directly; read-time gate must still refuse
    monkeypatch.setattr(policy_module, "PROTECTED_BOT_IDS", frozenset({"mira", bots["protected"]}))

    def context(bot: str = bots["on"], user: str = f"user-{tag}") -> TaskTurnContext:
        return TaskTurnContext(
            session_id=str(uuid.uuid4()), turn_id=f"turn-{uuid.uuid4().hex}",
            trigger_message_id=str(uuid.uuid4()), bot_id=bot, user_id=user, issued_at=0,
        )

    def version(bot: str = bots["on"], user: str = f"user-{tag}"):
        with engine.connect() as conn:
            return conn.execute(text(
                "SELECT version FROM working_memory_namespaces WHERE bot_id=:b AND user_id=:u"
            ), {"b": bot, "u": user}).scalar()

    def changes(item_id: str) -> list[dict]:
        with engine.connect() as conn:
            return [dict(r) for r in conn.execute(text(
                "SELECT * FROM working_memory_changes WHERE item_id=:i ORDER BY seq"
            ), {"i": item_id}).mappings()]

    yield SimpleNamespace(
        engine=engine, service=WorkingMemoryService(engine), settings=settings, bots=bots,
        user=f"user-{tag}", context=context, version=version, changes=changes,
    )
    with engine.begin() as conn:
        for table in ("working_memory_items", "working_memory_changes", "working_memory_namespaces"):
            conn.execute(text(f"DELETE FROM {table} WHERE bot_id LIKE :p"), {"p": f"test-wm-%-{tag}"})
        conn.execute(text("DELETE FROM runtime_settings WHERE scope_id LIKE :p"), {"p": f"test-wm-%-{tag}"})
        conn.execute(text("DELETE FROM bot_profiles WHERE slug LIKE :p"), {"p": f"test-wm-%-{tag}"})


def _key() -> str:
    return str(uuid.uuid4())


def _code(call) -> str:
    with pytest.raises(WorkingMemoryError) as raised:
        call()
    return raised.value.code


def test_create_update_list_records_trusted_actor(wm):
    ctx = wm.context()
    created = wm.service.bot_create(ctx, {"objective": "Ship slice 1", "next_action": "write store"}, _key())
    item = created["item"]
    assert (created["replayed"], item["revision"], item["status"], created["version"]) == (False, 1, "planned", 1)

    updated = wm.service.bot_update(ctx, item["id"], 1, {"status": "active", "blockers": ["review"]})
    assert (updated["item"]["revision"], updated["version"], updated["changed"]) == (2, 2, True)

    listing = wm.service.bot_list(wm.context())
    assert listing["version"] == 2 and [i["id"] for i in listing["items"]] == [item["id"]]

    rows = wm.changes(item["id"])
    assert [(r["kind"], r["ns_version"], r["actor_kind"], r["actor_id"]) for r in rows] == [
        ("create", 1, "bot", wm.bots["on"]), ("update", 2, "bot", wm.bots["on"]),
    ]
    assert rows[0]["session_id"] == ctx.session_id and rows[0]["turn_id"] == ctx.turn_id

    owner = wm.service.owner_update(wm.bots["on"], wm.user, item["id"], 2, {"next_action": None})
    assert owner["item"]["next_action"] is None
    last = wm.changes(item["id"])[-1]
    assert (last["actor_kind"], last["actor_id"], last["session_id"], last["turn_id"]) == ("user", wm.user, None, None)


def test_h1_replay_returns_current_item_and_writes_nothing(wm):
    ctx, key = wm.context(), _key()
    first = wm.service.bot_create(ctx, {"objective": "A"}, key)
    wm.service.bot_update(ctx, first["item"]["id"], 1, {"objective": "B"})
    before = wm.version()

    replay = wm.service.bot_create(wm.context(), {"objective": "A"}, key)
    assert replay["replayed"] is True and replay["item"]["objective"] == "B" and replay["version"] is None
    assert wm.version() == before and len(wm.changes(first["item"]["id"])) == 2


def test_h2_payload_conflict_writes_nothing(wm):
    ctx, key = wm.context(), _key()
    first = wm.service.bot_create(ctx, {"objective": "C"}, key)
    wm.service.bot_update(ctx, first["item"]["id"], 1, {"objective": "B"})
    before = wm.version()
    assert _code(lambda: wm.service.bot_create(ctx, {"objective": "A"}, key)) == "idempotency_conflict"
    assert wm.version() == before


def test_at_cap_replay_succeeds_and_conflict_stays_conflict(wm):
    ctx = wm.context()
    keys = [_key() for _ in range(20)]
    items = [wm.service.bot_create(ctx, {"objective": f"item {n}"}, k)["item"] for n, k in enumerate(keys)]
    wm.service.bot_update(ctx, items[0]["id"], 1, {"objective": "B"})
    before = wm.version()

    assert wm.service.bot_create(ctx, {"objective": "item 0"}, keys[0])["replayed"] is True
    assert _code(lambda: wm.service.bot_create(ctx, {"objective": "other"}, keys[1])) == "idempotency_conflict"
    assert _code(lambda: wm.service.bot_create(ctx, {"objective": "21st"}, _key())) == "cap_exceeded"
    assert wm.version() == before

    # Archiving frees a slot; reopening it while full counts toward the cap.
    wm.service.bot_update(ctx, items[1]["id"], 1, {"status": "done"})
    wm.service.bot_create(ctx, {"objective": "20th again"}, _key())
    assert _code(lambda: wm.service.bot_update(ctx, items[1]["id"], 2, {"status": "active"})) == "cap_exceeded"


def test_replay_after_archive_soft_and_hard_delete(wm):
    ctx = wm.context()
    archived_key, soft_key, hard_key = _key(), _key(), _key()
    archived = wm.service.bot_create(ctx, {"objective": "archive me"}, archived_key)["item"]
    archive = wm.service.bot_update(ctx, archived["id"], 1, {"status": "cancelled"})
    assert archive["item"]["archived_at"] is not None
    assert wm.changes(archived["id"])[-1]["kind"] == "archive"
    replay = wm.service.bot_create(ctx, {"objective": "archive me"}, archived_key)
    assert replay["replayed"] and replay["item"]["status"] == "cancelled"

    soft = wm.service.bot_create(ctx, {"objective": "soft"}, soft_key)["item"]
    wm.service.owner_delete(wm.bots["on"], wm.user, soft["id"], hard=False)
    assert _code(lambda: wm.service.bot_create(ctx, {"objective": "soft"}, soft_key)) == "removed"
    assert _code(lambda: wm.service.bot_update(ctx, soft["id"], 1, {"objective": "x"})) == "removed"

    hard = wm.service.bot_create(ctx, {"objective": "hard"}, hard_key)["item"]
    wm.service.owner_delete(wm.bots["on"], wm.user, hard["id"], hard=True)
    before = wm.version()
    assert _code(lambda: wm.service.bot_create(ctx, {"objective": "hard"}, hard_key)) == "removed"
    assert wm.changes(hard["id"])[0]["create_fingerprint"] is None
    assert wm.version() == before


def test_non_uuid_key_refused_and_first_request_leaves_no_namespace(wm):
    user = f"fresh-{uuid.uuid4().hex[:8]}"
    ctx = wm.context(user=user)
    for key in ("Ship slice 1", "not-a-uuid", "", str(uuid.uuid4()).replace("-", "")):
        assert _code(lambda: wm.service.bot_create(ctx, {"objective": "Ship slice 1"}, key)) == "invalid_idempotency_key"
    assert _code(lambda: wm.service.bot_update(ctx, str(uuid.uuid4()), 1, {"objective": "x"})) == "not_found"
    assert _code(lambda: wm.service.bot_create(
        wm.context(bot=wm.bots["off"], user=user), {"objective": "x"}, _key())) == "not_enabled"
    assert wm.version(user=user) is None
    assert wm.version(bot=wm.bots["off"], user=user) is None

    upper = str(uuid.uuid4()).upper()
    created = wm.service.bot_create(ctx, {"objective": "upper"}, upper)
    assert str(wm.changes(created["item"]["id"])[0]["idempotency_key"]) == upper.lower()
    assert wm.service.bot_create(ctx, {"objective": "upper"}, upper.lower())["replayed"] is True


def test_hard_delete_leaves_only_content_free_columns(wm):
    ctx = wm.context()
    secret = f"secret-{uuid.uuid4().hex}"
    item = wm.service.bot_create(ctx, {
        "objective": f"{secret}-objective", "next_action": f"{secret}-next",
        "refs": [{"kind": "doc", "id": f"{secret}-ref", "excerpt": f"{secret}-excerpt"}],
    }, _key())["item"]
    wm.service.bot_update(ctx, item["id"], 1, {"checkpoint": f"{secret}-cp", "blockers": [f"{secret}-b"]})
    wm.service.owner_delete(wm.bots["on"], wm.user, item["id"], hard=False)
    hard = wm.service.owner_delete(wm.bots["on"], wm.user, item["id"], hard=True)
    assert hard["state"] == "purged" and hard["changed"] is True
    again = wm.service.owner_delete(wm.bots["on"], wm.user, item["id"], hard=True)
    assert again["changed"] is False

    with wm.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM working_memory_items WHERE id=:i"), {"i": item["id"]}).scalar() == 0
        dumps = conn.execute(text(
            "SELECT row_to_json(c)::text FROM working_memory_changes c WHERE item_id=:i"
        ), {"i": item["id"]}).scalars().all()
    rows = wm.changes(item["id"])
    assert [r["kind"] for r in rows] == ["create", "update", "removed", "purged"]
    for row in rows:
        assert {k for k, v in row.items() if v is not None} <= SURVIVING
    assert not any(secret in dump for dump in dumps)


def test_cas_conflict_and_noop_write_nothing(wm):
    ctx = wm.context()
    item = wm.service.bot_create(ctx, {"objective": "cas"}, _key())["item"]
    wm.service.bot_update(ctx, item["id"], 1, {"objective": "cas 2"})
    before = wm.version()
    with pytest.raises(WorkingMemoryError) as raised:
        wm.service.bot_update(ctx, item["id"], 1, {"objective": "stale"})
    assert raised.value.code == "revision_conflict"
    assert raised.value.detail["item"]["objective"] == "cas 2" and raised.value.detail["item"]["revision"] == 2

    noop = wm.service.bot_update(ctx, item["id"], 2, {"objective": "cas 2"})
    assert noop["changed"] is False and noop["version"] is None
    assert wm.version() == before and len(wm.changes(item["id"])) == 2


def test_namespace_isolation(wm):
    item = wm.service.bot_create(wm.context(), {"objective": "mine"}, _key())["item"]
    other_user = wm.context(user=f"other-{uuid.uuid4().hex[:6]}")
    assert wm.service.bot_list(other_user)["items"] == []
    assert _code(lambda: wm.service.bot_update(other_user, item["id"], 1, {"objective": "x"})) == "not_found"
    assert _code(lambda: wm.service.owner_delete(wm.bots["off"], wm.user, item["id"], hard=True)) == "not_found"


def test_concurrent_writers_get_consecutive_versions_in_commit_order(wm):
    ctx = wm.context()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda n: wm.service.bot_create(ctx, {"objective": f"c{n}"}, _key()), range(16)))
    assert sorted(r["version"] for r in results) == list(range(1, 17))
    with wm.engine.connect() as conn:
        by_seq = conn.execute(text(
            "SELECT ns_version FROM working_memory_changes WHERE bot_id=:b AND user_id=:u ORDER BY seq"
        ), {"b": wm.bots["on"], "u": wm.user}).scalars().all()
    assert by_seq == list(range(1, 17))


def test_cap_holds_under_concurrent_creates(wm):
    ctx = wm.context()

    def attempt(n):
        try:
            return wm.service.bot_create(ctx, {"objective": f"race {n}"}, _key())["item"]["id"]
        except WorkingMemoryError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=10) as pool:
        outcomes = list(pool.map(attempt, range(26)))
    assert outcomes.count("cap_exceeded") == 6
    assert len(wm.service.bot_list(ctx)["items"]) == 20 and wm.version() == 20


def test_render_snapshot_pairs_version_with_items(wm):
    ctx = wm.context()
    stop = threading.Event()
    seen: list[tuple[int, int]] = []

    def read():
        while not stop.is_set():
            listing = wm.service.bot_list(ctx)
            seen.append((listing["version"], len(listing["items"])))

    reader = threading.Thread(target=read)
    reader.start()
    try:
        for n in range(15):
            wm.service.bot_create(ctx, {"objective": f"snap {n}"}, _key())
    finally:
        stop.set()
        reader.join()
    assert seen and all(version == count for version, count in seen)


def test_policy_gates_tools_but_not_owner_cleanup(wm):
    on = wm.service.bot_create(wm.context(), {"objective": "keep"}, _key())["item"]
    assert _code(lambda: wm.service.bot_list(wm.context(bot=wm.bots["off"]))) == "not_enabled"
    assert _code(lambda: wm.service.bot_list(wm.context(bot=wm.bots["codex"]))) == "not_enabled"
    assert _code(lambda: wm.service.bot_list(wm.context(bot=wm.bots["protected"]))) == "protected_bot"
    assert _code(lambda: wm.service.owner_list(wm.bots["protected"], wm.user)) == "protected_bot"
    assert _code(lambda: wm.service.owner_delete(wm.bots["protected"], wm.user, on["id"], hard=True)) == "protected_bot"

    # Disabling stops the tools at once; Nick can still read and clean up.
    wm.settings.set_value("bot", wm.bots["on"], KEY, False)
    assert _code(lambda: wm.service.bot_list(wm.context())) == "not_enabled"
    assert [i["id"] for i in wm.service.owner_list(wm.bots["on"], wm.user)["items"]] == [on["id"]]
    assert wm.service.owner_delete(wm.bots["on"], wm.user, on["id"], hard=False)["state"] == "removed"


def test_setting_validation_single_and_batch(wm, monkeypatch):
    service = SimpleNamespace(config=wm.settings.config, _default_bot=wm.bots["on"])
    monkeypatch.setattr(settings_routes, "get_service", lambda: service)
    monkeypatch.setattr(settings_routes, "_invalidate_bot_instance_cache", lambda *_a: None)
    monkeypatch.setattr(settings_routes, "_invalidate_all_instance_cache", lambda *_a: None)

    def stored(slug):
        with wm.engine.connect() as conn:
            return conn.execute(text(
                "SELECT value_json FROM runtime_settings WHERE scope_type='bot' AND scope_id=:s AND key=:k"
            ), {"s": slug, "k": KEY}).scalar()

    refused = [
        ("global", None, True), ("global", None, False), ("bot", wm.bots["codex"], True),
        ("bot", wm.bots["protected"], True), ("bot", f"test-wm-missing-{uuid.uuid4().hex[:6]}", True),
        ("bot", wm.bots["off"], "true"),
    ]
    for scope_type, scope_id, value in refused:
        with pytest.raises(HTTPException) as raised:
            settings_routes.upsert_runtime_setting(RuntimeSettingUpsertRequest(
                scope_type=scope_type, scope_id=scope_id, key=KEY, value=value))
        assert raised.value.status_code == 400

    with pytest.raises(HTTPException):
        settings_routes.batch_upsert_runtime_settings(RuntimeSettingBatchUpsertRequest(items=[
            RuntimeSettingBatchItem(scope_type="bot", scope_id=wm.bots["off"], key=KEY, value=True),
            RuntimeSettingBatchItem(scope_type="bot", scope_id=wm.bots["codex"], key=KEY, value=True),
        ]))
    assert stored(wm.bots["off"]) is None  # nothing in the batch was applied

    settings_routes.batch_upsert_runtime_settings(RuntimeSettingBatchUpsertRequest(items=[
        RuntimeSettingBatchItem(scope_type="bot", scope_id=wm.bots["off"], key=KEY, value=True),
        RuntimeSettingBatchItem(scope_type="bot", scope_id=wm.bots["protected"], key=KEY, value=False),
    ]))
    assert stored(wm.bots["off"]) == "true"


def test_owner_rest_routes(wm, monkeypatch):
    monkeypatch.setattr(wm_routes, "get_service", lambda: SimpleNamespace(config=wm.settings.config))
    app = FastAPI()
    app.include_router(wm_routes.router)
    client = TestClient(app)
    item = wm.service.bot_create(wm.context(), {"objective": "rest", "next_action": "n", "priority": 3}, _key())["item"]
    base = f"/v1/working-memory/{wm.bots['on']}/{wm.user}"

    patched = client.patch(f"{base}/{item['id']}", json={"expected_revision": 1, "next_action": None, "status": "blocked"})
    assert patched.status_code == 200
    body = patched.json()["item"]
    assert (body["next_action"], body["status"], body["priority"], body["revision"]) == (None, "blocked", 3, 2)

    stale = client.patch(f"{base}/{item['id']}", json={"expected_revision": 1, "objective": "stale"})
    assert stale.status_code == 409 and stale.json()["item"]["revision"] == 2
    assert client.patch(f"{base}/{item['id']}", json={"objective": "no revision"}).status_code == 422

    assert client.delete(f"{base}/{item['id']}").json()["state"] == "removed"
    assert client.delete(f"{base}/{item['id']}").json()["changed"] is False
    assert client.delete(f"{base}/{item['id']}", params={"hard": "true"}).json()["state"] == "purged"
    assert [r["kind"] for r in wm.changes(item["id"])] == ["create", "update", "removed", "purged"]

    listing = client.get(base).json()
    assert listing["items"] == [] and [p["id"] for p in listing["purged"]] == [item["id"]]
    assert client.get(f"/v1/working-memory/{wm.bots['protected']}/{wm.user}").status_code == 403
    assert client.delete(f"{base}/not-a-uuid").status_code == 404
