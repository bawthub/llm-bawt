"""Private PostgreSQL proof for working memory slices 1 and 2a (TASK-1281, docs/WORKING_MEMORY.md).

Store, trusted bot path, owner REST path and setting validation against the
real schema; then the section render, change notice, session state and health
report. Protected-bot cases use a synthetic slug, never Mira's data.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import text

from llm_bawt import task_turn_context as codec
from llm_bawt.mcp_server import task_association
from llm_bawt.mcp_server import working_memory_tools as tools
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
from llm_bawt.working_memory import health as wm_health
from llm_bawt.working_memory import metrics, prompts, session_state
from llm_bawt.working_memory import notice as notice_module
from llm_bawt.working_memory.contracts import WorkingMemoryError
from llm_bawt.working_memory.notice import prepare_notice
from llm_bawt.working_memory.render import render_section
from llm_bawt.working_memory.service import WorkingMemoryService
from llm_bawt.working_memory.store import Actor
from tests.bot_tag_fixtures import PRIVATE_MODES, make_private

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
    with engine.begin() as conn:
        make_private(conn, bots["protected"], "tag")

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

    service = WorkingMemoryService(engine)
    # Create the tables up front: a test that never touches the store still runs teardown.
    service.store.ensure_schema()
    yield SimpleNamespace(
        engine=engine, service=service, settings=settings, bots=bots,
        user=f"user-{tag}", context=context, version=version, changes=changes,
    )
    with engine.begin() as conn:
        conn.execute(text(
            "DELETE FROM working_memory_task_purge_audit WHERE item_id IN"
            " (SELECT item_id FROM working_memory_changes WHERE bot_id LIKE :p)"
        ), {"p": f"test-wm-%-{tag}"})
        for table in ("working_memory_task_suppressions", "working_memory_items",
                      "working_memory_changes", "working_memory_namespaces"):
            conn.execute(text(f"DELETE FROM {table} WHERE bot_id LIKE :p"), {"p": f"test-wm-%-{tag}"})
        conn.execute(text("DELETE FROM runtime_settings WHERE scope_id LIKE :p"), {"p": f"test-wm-%-{tag}"})
        conn.execute(text("DELETE FROM bot_profiles WHERE slug LIKE :p"), {"p": f"test-wm-%-{tag}"})
        conn.execute(text("DELETE FROM bot_profile_tags WHERE bot_slug LIKE :p"), {"p": f"test-wm-%-{tag}"})


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


# Signed-capability, concurrency and stored-setting cases contributed by Loopy
# during the TASK-1281 r1 implementation review.

def _signer(monkeypatch) -> Fernet:
    signer = Fernet(Fernet.generate_key())
    monkeypatch.setattr(codec, "_get_fernet", lambda: signer)
    return signer


def _tool_as(token: str, call) -> dict:
    """Run one tool coroutine factory under ``token`` as the turn capability."""
    binding = task_association.set_current_task_turn_capability(token)
    try:
        return json.loads(asyncio.run(call()))
    finally:
        task_association.reset_current_task_turn_capability(binding)


def _mint(ctx: TaskTurnContext) -> str:
    return codec.mint_task_turn_context(
        session_id=ctx.session_id, turn_id=ctx.turn_id, trigger_message_id=ctx.trigger_message_id,
        bot_id=ctx.bot_id, user_id=ctx.user_id,
    )


def test_tools_bind_signed_identity_and_refuse_other_users_items(wm, monkeypatch):
    _signer(monkeypatch)
    monkeypatch.setattr(tools, "_service", lambda: wm.service)
    ctx = wm.context()
    token = _mint(ctx)

    item = _tool_as(token, lambda: tools.working_memory_create(objective="signed fixture", idempotency_key=_key()))["item"]
    listing = _tool_as(token, lambda: tools.working_memory_list())
    assert [row["id"] for row in listing["items"]] == [item["id"]]
    changed = _tool_as(token, lambda: tools.working_memory_update(item_id=item["id"], expected_revision=1, checkpoint="proof"))
    assert changed["item"]["revision"] == 2
    row = wm.changes(item["id"])[-1]
    assert (row["bot_id"], row["user_id"], row["session_id"], row["turn_id"]) == (
        ctx.bot_id, ctx.user_id, ctx.session_id, ctx.turn_id,
    )

    other = _mint(wm.context(user="other-user"))
    refused = _tool_as(other, lambda: tools.working_memory_update(
        item_id=item["id"], expected_revision=2, objective="intrusion"))
    assert refused["error"] == "not_found"
    assert _tool_as(other, lambda: tools.working_memory_list())["items"] == []
    assert wm.service.bot_list(ctx)["items"][0]["objective"] == "signed fixture"


class _RecordingService:
    """Stands in for the service so a test sees exactly which calls got through."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name):
        def record(*_args, **_kwargs):
            self.calls.append(name)
            return {"reached": name}
        return record


def test_invalid_or_expired_capability_never_reaches_the_service(wm, monkeypatch):
    signer = _signer(monkeypatch)
    service = _RecordingService()
    monkeypatch.setattr(tools, "_service", lambda: service)
    # A structurally valid payload, so only the Fernet TTL can refuse the expired copy.
    payload = signer.decrypt(_mint(wm.context()).encode())
    expired = signer.encrypt_at_time(payload, int(time.time()) - codec.TASK_TURN_CONTEXT_TTL_SECONDS - 1).decode()
    current = signer.encrypt_at_time(payload, int(time.time())).decode()
    calls = (
        lambda: tools.working_memory_list(),
        lambda: tools.working_memory_create(objective="x", idempotency_key=_key()),
        lambda: tools.working_memory_update(item_id=_key(), expected_revision=1),
    )
    for token in ("invalid", signer.encrypt(b"not-json").decode(), expired):
        for call in calls:
            assert _tool_as(token, call)["error"] == "unauthorized"
    assert service.calls == []

    # Positive control: the same payload signed now reaches the service.
    for call in calls:
        assert "reached" in _tool_as(current, call)
    assert service.calls == ["bot_list", "bot_create", "bot_update"]


def test_same_key_concurrent_creates_make_one_write(wm):
    ctx, key = wm.context(), _key()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: wm.service.bot_create(ctx, {"objective": "one"}, key), range(16)))
    assert sum(not result["replayed"] for result in results) == 1
    assert len({result["item"]["id"] for result in results}) == 1
    assert wm.version() == 1
    assert len(wm.changes(results[0]["item"]["id"])) == 1


@pytest.mark.parametrize("value", [1, "true", "false", {}, None])
def test_non_boolean_stored_setting_fails_closed(wm, value):
    wm.settings.set_value("bot", wm.bots["on"], KEY, value)
    assert _code(lambda: wm.service.bot_create(wm.context(), {"objective": "refuse"}, _key())) == "not_enabled"
    assert wm.version() is None


# -- slice 2a: section, notice, session state, health (8.1, 9, 13, 14) ---------

@pytest.fixture
def ses(wm):
    """Synthetic session rows for the slice-2a bots, removed afterwards."""
    made: list[str] = []

    def new(*, bot: str = wm.bots["on"], status: str = "archived", metadata: dict | None = None) -> str:
        sid = str(uuid.uuid4())
        with wm.engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO sessions (id, bot_id, user_id, status, session_metadata)"
                " VALUES (:id, :bot, :user, :status, CAST(:meta AS jsonb))"
            ), {"id": sid, "bot": bot, "user": wm.user, "status": status,
                "meta": json.dumps(metadata) if metadata is not None else None})
        made.append(sid)
        return sid

    def metadata(sid: str) -> dict:
        with wm.engine.connect() as conn:
            return conn.execute(text("SELECT session_metadata FROM sessions WHERE id=:s"), {"s": sid}).scalar() or {}

    def ctx(sid: str):
        return dataclasses.replace(wm.context(), session_id=sid)

    metrics.reset()
    yield SimpleNamespace(new=new, metadata=metadata, ctx=ctx)
    metrics.reset()
    with wm.engine.begin() as conn:
        conn.execute(text("DELETE FROM sessions WHERE id = ANY(:ids)"), {"ids": made})


def _pin(version, state="ok"):
    return {"working_memory": {"v": 1, "version": version, "state": state}}


def _decision(*, pin="pin-1", boundary=False, unverified=False, pinned=None, candidate=None):
    return SimpleNamespace(pin_turn_id=pin, is_boundary=boundary, snapshot_unverified=unverified,
                           pinned_sections=pinned, candidate_sections=candidate)


def _notice(wm, sid, decision, *, eligible=True, bot=None):
    return prepare_notice(wm.engine, bot_id=bot or wm.bots["on"], user_id=wm.user, session_id=sid,
                          decision=decision, eligible=eligible)


def test_session_state_writes_merge_siblings_under_the_row_lock(wm, ses):
    pin = {"turn_id": "t-pin", "sha256": "abc"}
    sid = ses.new(metadata={"system_prompt_pin": pin, "other": 1,
                            "working_memory": {"mode": "pinned", "notice": {"version": 2}}})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: session_state.bump_compaction_gen(wm.engine, sid), range(20)))
    meta = ses.metadata(sid)
    assert meta["system_prompt_pin"] == pin and meta["other"] == 1
    assert meta["working_memory"] == {"mode": "pinned", "notice": {"version": 2}, "compaction_gen": 20}
    assert session_state.bump_compaction_gen(wm.engine, str(uuid.uuid4())) is None  # no row, no write


def test_concurrent_notice_marks_and_bumps_keep_each_others_keys(wm, ses):
    sid = ses.new(metadata={"keep": True})
    item = wm.service.bot_create(wm.context(), {"objective": "x"}, _key())["item"]
    pending = threading.Barrier(2)

    def mark(n):
        wm.service.owner_update(wm.bots["on"], wm.user, item["id"], n + 1, {"priority": n})
        pending.wait()
        return _notice(wm, sid, _decision(pin=f"pin-{n}", pinned=_pin(0)))

    def bump(_):
        pending.wait()
        return session_state.bump_compaction_gen(wm.engine, sid)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for n in range(6):
            list(pool.map(lambda f: f(n), (mark, bump)))
    state = ses.metadata(sid)
    assert state["keep"] is True
    assert state["working_memory"]["compaction_gen"] == 6
    assert state["working_memory"]["notice"]["pin_turn_id"] == "pin-5"


def test_floor_only_write_leaves_the_notice_bytes_alone(wm, ses):
    marked = {"version": 5, "compaction_gen": 0, "pin_turn_id": "pin-1", "at": "2026-10-23T00:00:00+00:00"}
    sid = ses.new(metadata={"working_memory": {"notice": marked}})
    # Unverified re-pin: the floor changes, nothing external -> no notice.
    assert _notice(wm, sid, _decision(unverified=True, pinned=_pin(3), candidate=_pin(5))) is None
    state = ses.metadata(sid)["working_memory"]
    assert state["unverified_floor"] == {"min_version": 3, "may_be_missing": False}
    assert state["notice"] == marked
    # Ineligible and boundary turns fold too, without touching the notice.
    assert _notice(wm, sid, _decision(boundary=True, unverified=True, pinned=_pin(1), candidate=_pin(5))) is None
    state = ses.metadata(sid)["working_memory"]
    assert state["unverified_floor"]["min_version"] == 1 and state["notice"] == marked
    # A verified kept turn clears the floor; still no notice write.
    assert _notice(wm, sid, _decision(pinned=_pin(5)), eligible=False) is None
    state = ses.metadata(sid)["working_memory"]
    assert "unverified_floor" not in state and state["notice"] == marked


def test_render_states_and_order(wm, ses):
    sid = ses.new()
    on = wm.bots["on"]
    empty = render_section(wm.engine, bot_id=on, user_id=wm.user, session_id=sid)
    assert (empty.state, empty.version) == ("empty", 0)

    ctx = ses.ctx(sid)
    planned = wm.service.bot_create(ctx, {"objective": "planned one"}, _key())["item"]
    blocked = wm.service.bot_create(ctx, {"objective": "blocked one", "status": "blocked"}, _key())["item"]
    active = wm.service.bot_create(ctx, {"objective": "active one", "status": "active",
                                         "next_action": "do it"}, _key())["item"]
    wm.service.owner_update(on, wm.user, blocked["id"], 1, {"blockers": ["review"]})
    done = wm.service.bot_create(ctx, {"objective": "finished"}, _key())["item"]
    wm.service.bot_update(ctx, done["id"], 1, {"status": "done"})

    out = render_section(wm.engine, bot_id=on, user_id=wm.user, session_id=sid)
    assert (out.state, out.version) == ("ok", wm.version())
    lines = [json.loads(line) for line in out.text.split("\n") if line.startswith("{")]
    assert [line["id"] for line in lines] == [active["id"], blocked["id"], planned["id"]]
    assert lines[0]["next_action"] == "do it" and "other_session" not in lines[0]
    assert lines[1]["other_session"] is True  # last changed by the owner, outside this session
    assert "revision" not in lines[2]  # planned renders as a title

    # Same state, same bytes.
    again = render_section(wm.engine, bot_id=on, user_id=wm.user, session_id=sid)
    assert again.text == out.text

    suppressed = ses.new(metadata={"working_memory": {"mode": "suppressed"}})
    s = render_section(wm.engine, bot_id=on, user_id=wm.user, session_id=suppressed)
    assert (s.text, s.version, s.state) == (None, None, "suppressed")
    for bot in (wm.bots["off"], wm.bots["codex"], wm.bots["protected"]):
        assert render_section(wm.engine, bot_id=bot, user_id=wm.user, session_id=sid) is None


def test_notice_fires_on_external_changes_once(wm, ses):
    sid, other = ses.new(), ses.new()
    on = wm.bots["on"]
    item = wm.service.bot_create(ses.ctx(sid), {"objective": "x"}, _key())["item"]
    pinned = render_section(wm.engine, bot_id=on, user_id=wm.user, session_id=sid)
    kept = _decision(pinned={"working_memory": pinned.meta})

    assert _notice(wm, sid, kept) is None  # nothing external
    wm.service.bot_update(ses.ctx(sid), item["id"], 1, {"priority": 1})
    assert _notice(wm, sid, kept) is None  # own write
    assert "notice" not in ses.metadata(sid).get("working_memory", {})

    wm.service.owner_update(on, wm.user, item["id"], 2, {"priority": 2})  # REST edit
    assert _notice(wm, sid, kept) == prompts.DEFAULT_NOTICE
    assert _notice(wm, sid, kept) is None  # not the one after
    assert ses.metadata(sid)["working_memory"]["notice"]["version"] == wm.version()

    wm.service.bot_update(ses.ctx(other), item["id"], 3, {"priority": 3})  # another session
    assert _notice(wm, sid, kept, eligible=False) is None  # maintenance/scheduler: stays pending
    assert _notice(wm, sid, _decision(boundary=True, pinned=kept.pinned_sections)) is None
    assert _notice(wm, sid, kept) == prompts.DEFAULT_NOTICE
    assert metrics.snapshot(on)["notices_fired"] == 2


def test_notice_mark_is_best_effort_and_the_loss_window_holds(wm, ses):
    # 9.3/OI-R2: marked at prepare; if the turn dies, the notice is not re-sent
    # until the next external change (or a boundary/compaction).
    sid = ses.new()
    on = wm.bots["on"]
    item = wm.service.bot_create(wm.context(), {"objective": "x"}, _key())["item"]
    kept = _decision(pinned=_pin(0))
    assert _notice(wm, sid, kept) == prompts.DEFAULT_NOTICE
    # The turn carrying it is lost before the model sees it: nothing re-sends.
    assert _notice(wm, sid, kept) is None
    assert _notice(wm, sid, kept) is None
    wm.service.owner_update(on, wm.user, item["id"], 1, {"priority": 1})
    assert _notice(wm, sid, kept) == prompts.DEFAULT_NOTICE


def test_compaction_rearms_but_an_earlier_one_does_not(wm, ses):
    sid = ses.new()
    wm.service.bot_create(wm.context(), {"objective": "x"}, _key())
    kept = _decision(pinned=_pin(0))
    session_state.bump_compaction_gen(wm.engine, sid)  # finalized before the mark: counted in g
    assert _notice(wm, sid, kept) == prompts.DEFAULT_NOTICE
    assert ses.metadata(sid)["working_memory"]["notice"]["compaction_gen"] == 1
    assert _notice(wm, sid, kept) is None
    session_state.bump_compaction_gen(wm.engine, sid)  # compacted after the mark
    assert _notice(wm, sid, kept) == prompts.DEFAULT_NOTICE
    assert _notice(wm, sid, kept) is None
    # A new pin episode also starts over.
    assert _notice(wm, sid, _decision(pin="pin-2", pinned=_pin(0))) == prompts.DEFAULT_NOTICE


def test_missing_section_notices_once_per_episode(wm, ses):
    # Enable mid-session / legacy pin / unavailable at the boundary: one notice, no writes needed.
    sid = ses.new()
    for pinned in (None, _pin(None, "unavailable")):
        decision = _decision(pin=f"pin-{uuid.uuid4().hex[:6]}", pinned=pinned)
        assert _notice(wm, sid, decision) == prompts.DEFAULT_NOTICE
        assert _notice(wm, sid, decision) is None


def test_unverified_floor_counterexample_across_turns(wm, ses):
    # r12 S1: S absent, C1 v3, auto-compaction, C2 v5 -> P = 3, so a change at v4 notices.
    sid = ses.new()
    ctx = ses.ctx(sid)
    item = wm.service.bot_create(ctx, {"objective": "x"}, _key())["item"]  # v1, own
    wm.service.bot_update(ctx, item["id"], 1, {"priority": 1})              # v2, own
    wm.service.bot_update(ctx, item["id"], 2, {"priority": 2})              # v3, own
    first = _decision(pin="p1", unverified=True, pinned=None, candidate=_pin(3))
    assert _notice(wm, sid, first) == prompts.DEFAULT_NOTICE  # missing S -> one notice
    wm.service.owner_update(wm.bots["on"], wm.user, item["id"], 3, {"priority": 4})  # v4, external
    wm.service.bot_update(ctx, item["id"], 4, {"priority": 5})                      # v5, own
    # Same episode, N = 3: only the folded floor (3, not C2's 5) makes v4 count.
    second = _decision(pin="p1", unverified=True, pinned=None, candidate=_pin(5))
    assert _notice(wm, sid, second) == prompts.DEFAULT_NOTICE
    assert _notice(wm, sid, second) is None
    assert ses.metadata(sid)["working_memory"]["unverified_floor"] == {"min_version": 3, "may_be_missing": True}


def test_excluded_bots_and_suppressed_sessions_get_no_notice_and_no_write(wm, ses):
    for bot in (wm.bots["off"], wm.bots["codex"], wm.bots["protected"]):
        sid = ses.new(bot=bot, metadata={"x": 1})
        assert _notice(wm, sid, _decision(pinned=None), bot=bot) is None
        assert ses.metadata(sid) == {"x": 1}
    sid = ses.new(metadata={"working_memory": {"mode": "suppressed"}})
    assert _notice(wm, sid, _decision(pinned=None)) is None
    assert ses.metadata(sid) == {"working_memory": {"mode": "suppressed"}}
    assert _notice(wm, str(uuid.uuid4()), _decision(pinned=None)) is None  # no session row


def test_notice_storage_failure_logs_and_counts(wm, ses, monkeypatch):
    sid = ses.new()

    def down(*args, **kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(notice_module, "read_state", down)
    assert _notice(wm, sid, _decision(pinned=None)) is None
    assert metrics.snapshot(wm.bots["on"])["notice_failures"] == 1


def test_health_report_counts_without_item_text(wm, ses, monkeypatch):
    on = wm.bots["on"]
    ctx = wm.context()
    keep = wm.service.bot_create(ctx, {"objective": "secret objective"}, _key())["item"]
    gone = wm.service.bot_create(ctx, {"objective": "finished"}, _key())["item"]
    wm.service.bot_update(ctx, gone["id"], 1, {"status": "done"})
    soft = wm.service.bot_create(ctx, {"objective": "soft"}, _key())["item"]
    wm.service.owner_delete(on, wm.user, soft["id"], hard=False)
    version = wm.version()
    ses.new(status="active", metadata={"system_prompt_pin": {"sections": _pin(version - 2)}})
    ses.new(metadata={"system_prompt_pin": {"sections": _pin(0)}})  # archived: not counted
    metrics.record_notice(on)

    report = wm_health.health_report(wm.engine)
    assert "secret objective" not in json.dumps(report) and keep["id"] not in json.dumps(report)
    mine = {bot["bot_id"]: bot for bot in report["bots"] if bot["bot_id"] in set(wm.bots.values())}
    assert set(mine) == {on, wm.bots["codex"]}  # stored true; never the protected or off bot
    assert mine[on]["allowed"] is True and mine[wm.bots["codex"]]["allowed"] is False
    assert mine[on]["items"] == {"open": 1, "archived": 1, "removed": 1}
    assert mine[on]["max_namespace_version"] == version
    assert mine[on]["sessions"] == {"pinned_sessions": 1, "behind": 1, "oldest_lag": 2,
                                    "without_section_version": 0}
    assert mine[on]["since_start"]["notices_fired"] == 1

    monkeypatch.setattr(wm_routes, "get_service", lambda: SimpleNamespace(config=wm.settings.config))
    app = FastAPI()
    app.include_router(wm_routes.router)
    client = TestClient(app)
    served = client.get("/v1/working-memory/health").json()
    assert {bot["bot_id"] for bot in served["bots"]} >= {on, wm.bots["codex"]}

    monkeypatch.setattr(wm_health, "health_report", lambda engine: 1 / 0)
    response = client.get("/v1/working-memory/health")
    assert (response.status_code, response.json()) == (503, {"status": "unavailable", "bots": []})


@pytest.mark.parametrize("mode", PRIVATE_MODES)
def test_health_never_reads_an_opted_in_private_bot(wm, monkeypatch, mode):
    """BOT_TAGS.md §10.11: excluded in the listing statement, before any per-bot query."""
    private = f"test-wm-private-{mode}-{wm.user.removeprefix('user-')}"
    BotProfileStore(wm.settings.config).upsert(
        {"slug": private, "bot_type": "agent", "agent_backend": "claude-code", "harness": "claude-code"})
    wm.settings.set_value("bot", private, KEY, True)
    with wm.engine.begin() as conn:
        make_private(conn, private, mode)
    read: list[str] = []
    for name in ("_counts", "_behind"):
        original = getattr(wm_health, name)
        monkeypatch.setattr(wm_health, name, lambda conn, bot, _o=original: (read.append(bot), _o(conn, bot))[1])

    report = wm_health.health_report(wm.engine)

    assert private not in {bot["bot_id"] for bot in report["bots"]}
    assert wm.bots["on"] in {bot["bot_id"] for bot in report["bots"]}
    assert private not in read and wm.bots["on"] in read


# -- TASK-1530: canonical task pointers and durable suppression -----------------

def _task_links(wm, monkeypatch):
    from llm_bawt.working_memory import store as store_module
    from llm_bawt.working_memory.task_links import TaskInfo

    monkeypatch.setattr(store_module, "derive_key", lambda label: b"test-only-stable-key")
    task = TaskInfo("task-" + uuid.uuid4().hex, "TASK-1530", "Task title", "IN_PROGRESS", wm.bots["on"])

    class Catalog:
        def get(self, ref):
            if ref not in (task.id, task.short_id):
                raise WorkingMemoryError("task_unavailable", "missing task")
            return task

    catalog = Catalog()
    wm.service.catalog = catalog
    return task, catalog


@pytest.mark.parametrize("first", ["manual", "auto"])
def test_link_canonicalizes_both_manual_and_association_orderings(wm, monkeypatch, first):
    task, _ = _task_links(wm, monkeypatch)
    ctx = wm.context()

    def manual():
        return wm.service.bot_create(ctx, {"kind": "initiative_link", "objective": "My pointer",
                                            "initiative_task_id": task.short_id}, _key())

    def auto():
        return wm.service.link_associated(ctx, task.id)

    if first == "manual":
        original = manual()
        assert auto()["item"]["id"] == original["item"]["id"]
    else:
        original = auto()
        assert _code(manual) == "already_linked"
    rows = wm.service.bot_list(ctx)["items"]
    assert len(rows) == 1 and rows[0]["canonical_task_id"] == task.id
    assert rows[0]["task_short_id"] == task.short_id


def test_private_and_off_bots_never_resolve_or_touch_task_links(wm, monkeypatch):
    class ForbiddenCatalog:
        def get(self, ref):
            pytest.fail("private/off bots must not read tasks")

    wm.service.catalog = ForbiddenCatalog()
    for name, code in (("protected", "protected_bot"), ("off", "not_enabled"),
                       ("codex", "not_enabled")):
        ctx = wm.context(bot=wm.bots[name])
        assert _code(lambda: wm.service.link_associated(ctx, "task-id")) == code
        with wm.engine.connect() as conn:
            assert conn.execute(text(
                "SELECT count(*) FROM working_memory_namespaces WHERE bot_id=:bot AND user_id=:user"
            ), {"bot": wm.bots[name], "user": wm.user}).scalar_one() == 0


def test_manual_task_lookup_must_verify_identity_before_writing(wm, monkeypatch):
    task, catalog = _task_links(wm, monkeypatch)
    ctx = wm.context()
    fields = {"kind": "initiative_link", "objective": "pointer", "initiative_task_id": task.short_id}
    catalog.get = lambda ref: (_ for _ in ()).throw(WorkingMemoryError("task_unavailable", "missing"))
    assert _code(lambda: wm.service.bot_create(ctx, fields, _key())) == "task_unavailable"
    assert wm.version() is None


def test_task_pointer_renders_full_and_compact_json_fields(wm, monkeypatch):
    from datetime import datetime, timezone
    from llm_bawt.working_memory.render import full_line, title_line

    task, _ = _task_links(wm, monkeypatch)
    item = wm.service.link_associated(wm.context(), task.id)["item"]
    for line in (full_line(item, None, datetime.now(timezone.utc)),
                 title_line(item, None, datetime.now(timezone.utc))):
        pointer = json.loads(line)
        assert pointer["initiative_task_id"] == task.id
        assert (pointer["task_short_id"], pointer["task_title"], pointer["task_status"]) == (
            task.short_id, task.title, task.status)


def test_lone_preexisting_removed_link_is_backfilled_as_suppressed(wm, monkeypatch):
    task, _ = _task_links(wm, monkeypatch)
    ctx = wm.context()
    old = wm.service.store.create(wm.bots["on"], wm.user, {
        "kind": "initiative_link", "initiative_task_id": task.short_id, "objective": "Old item",
    }, idempotency_key=_key(), actor=Actor("bot", wm.bots["on"]))["item"]
    with wm.engine.begin() as conn:
        conn.execute(text("UPDATE working_memory_items SET removed_at=now() WHERE id=:id"), {"id": old["id"]})
    assert _code(lambda: wm.service.link_associated(ctx, task.id)) == "removed"
    assert _code(lambda: wm.service.bot_create(ctx, {
        "kind": "initiative_link", "initiative_task_id": task.id, "objective": "again",
    }, _key())) == "removed"
    with wm.engine.connect() as conn:
        assert conn.execute(text(
            "SELECT count(*) FROM working_memory_task_suppressions WHERE bot_id=:bot AND user_id=:user"
        ), {"bot": wm.bots["on"], "user": wm.user}).scalar_one() == 1
        assert conn.execute(text(
            "SELECT count(*) FROM working_memory_items WHERE bot_id=:bot AND user_id=:user"
            " AND kind='initiative_link' AND removed_at IS NULL"
        ), {"bot": wm.bots["on"], "user": wm.user}).scalar_one() == 0


def test_hard_purge_retains_opaque_suppression_without_item_text(wm, monkeypatch):
    task, _ = _task_links(wm, monkeypatch)
    ctx = wm.context()
    item = wm.service.link_associated(ctx, task.id)["item"]
    assert wm.service.owner_delete(wm.bots["on"], wm.user, item["id"], hard=True)["state"] == "purged"
    assert _code(lambda: wm.service.link_associated(ctx, task.id)) == "removed"
    with wm.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM working_memory_items WHERE id=:id"), {"id": item["id"]}).scalar_one() == 0
        rows = conn.execute(text(
            "SELECT row_to_json(s)::text FROM working_memory_task_suppressions s"
            " WHERE bot_id=:bot AND user_id=:user"
        ), {"bot": wm.bots["on"], "user": wm.user}).scalars().all()
    assert len(rows) == 1 and task.id not in rows[0] and task.title not in rows[0]


def test_link_audit_failure_refuses_all_creates_before_index_or_backfill(wm, monkeypatch):
    task, _ = _task_links(wm, monkeypatch)
    ctx = wm.context()
    # Schema guard has already run; a missing index must still fail closed.
    from llm_bawt.working_memory.store import _TASK_RELATIONS
    from llm_bawt.utils.schema import ensure_relations

    with wm.engine.begin() as conn:
        conn.execute(text("DROP INDEX ux_working_memory_live_task_links"))
    try:
        assert _code(lambda: wm.service.link_associated(ctx, task.id)) == "task_link_audit_pending"
        assert _code(lambda: wm.service.bot_create(ctx, {
            "kind": "initiative_link", "initiative_task_id": task.short_id, "objective": "new",
        }, _key())) == "task_link_audit_pending"
        with wm.engine.connect() as conn:
            assert conn.execute(text(
                "SELECT count(*) FROM working_memory_items WHERE bot_id=:bot AND user_id=:user"
            ), {"bot": wm.bots["on"], "user": wm.user}).scalar_one() == 0
    finally:
        with wm.engine.begin() as conn:
            ensure_relations(conn, _TASK_RELATIONS)


def test_concurrent_manual_and_associated_cuids_use_one_link(wm, monkeypatch):
    task, _ = _task_links(wm, monkeypatch)
    ctx = wm.context()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(
            wm.service.bot_create, ctx,
            {"kind": "initiative_link", "objective": f"manual {n}",
             "initiative_task_id": task.short_id if n % 2 else task.id}, _key(),
        ) for n in range(4)]
        futures += [pool.submit(wm.service.link_associated, ctx, task.id) for _ in range(4)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result()["item"]["id"])
            except WorkingMemoryError as exc:
                assert exc.code == "already_linked"
    assert len(set(outcomes)) == 1
    assert len(wm.service.store.list_items(wm.bots["on"], wm.user)["items"]) == 1


def test_legacy_duplicate_prefers_owner_edited_row_and_retains_the_other(wm, monkeypatch):
    task, _ = _task_links(wm, monkeypatch)
    first = wm.service.store.create(wm.bots["on"], wm.user, {
        "kind": "initiative_link", "initiative_task_id": task.short_id,
        "objective": "oldest content", "refs": [{"kind": "doc", "id": "keep this"}],
    }, idempotency_key=_key(), actor=Actor("bot", wm.bots["on"]))["item"]
    second_key = _key()
    second = wm.service.store.create(wm.bots["on"], wm.user, {
        "kind": "initiative_link", "initiative_task_id": task.id, "objective": "owner content",
    }, idempotency_key=second_key, actor=Actor("bot", wm.bots["on"]))["item"]
    wm.service.owner_update(wm.bots["on"], wm.user, second["id"], 1, {"checkpoint": "owner note"})
    result = wm.service.link_associated(wm.context(), task.id)
    assert result["item"]["id"] == second["id"]
    rows = wm.service.owner_list(wm.bots["on"], wm.user)["items"]
    assert {row["id"] for row in rows} == {first["id"], second["id"]}
    loser = next(row for row in rows if row["id"] == first["id"])
    assert loser["status"] == "cancelled" and loser["refs"][0]["id"] == "keep this"
    assert _code(lambda: wm.service.bot_update(wm.context(), first["id"], loser["revision"],
                                                {"status": "active"})) == "already_linked"
    assert _code(lambda: wm.service.owner_update(wm.bots["on"], wm.user, first["id"],
                                                  loser["revision"], {"status": "active"})) == "already_linked"


def test_soft_then_hard_task_removal_blocks_manual_and_auto_recreation(wm, monkeypatch):
    task, _ = _task_links(wm, monkeypatch)
    ctx = wm.context()
    fields = {"kind": "initiative_link", "initiative_task_id": task.short_id, "objective": "pointer"}
    key = _key()
    item = wm.service.bot_create(ctx, fields, key)["item"]
    wm.service.owner_delete(wm.bots["on"], wm.user, item["id"], hard=False)
    assert _code(lambda: wm.service.bot_create(ctx, fields, key)) == "removed"
    assert _code(lambda: wm.service.bot_create(ctx, fields, _key())) == "removed"
    assert _code(lambda: wm.service.link_associated(ctx, task.id)) == "removed"
    wm.service.owner_delete(wm.bots["on"], wm.user, item["id"], hard=True)
    assert _code(lambda: wm.service.bot_create(ctx, fields, key)) == "removed"
    assert _code(lambda: wm.service.bot_create(ctx, fields, _key())) == "removed"
    assert _code(lambda: wm.service.link_associated(ctx, task.id)) == "removed"


def test_purge_key_rotation_uses_one_snapshot_and_fences_recreation(wm, monkeypatch):
    from llm_bawt.working_memory import store as store_module

    task, _ = _task_links(wm, monkeypatch)
    ctx = wm.context()
    item = wm.service.link_associated(ctx, task.id)["item"]
    calls = []

    def rotating_key(label):
        calls.append(label)
        return b"test-only-stable-key" if len(calls) == 1 else b"rotated-test-key"

    monkeypatch.setattr(store_module, "derive_key", rotating_key)
    assert wm.service.owner_delete(wm.bots["on"], wm.user, item["id"], hard=True)["state"] == "purged"
    assert len(calls) == 1  # digest and key sentinel came from this single snapshot
    with wm.engine.connect() as conn:
        assert conn.execute(text(
            "SELECT count(*) FROM working_memory_task_suppressions"
            " WHERE bot_id=:bot AND user_id=:user"
        ), {"bot": wm.bots["on"], "user": wm.user}).scalar_one() == 1
    assert _code(lambda: wm.service.link_associated(ctx, task.id)) == "task_link_audit_pending"
    assert len(calls) >= 2
    assert wm.service.store.list_items(wm.bots["on"], wm.user)["items"] == []


def test_rotated_suppression_key_fails_closed_for_links_and_purge(wm, monkeypatch):
    from llm_bawt.working_memory import store as store_module

    task, _ = _task_links(wm, monkeypatch)
    ctx = wm.context()
    item = wm.service.link_associated(ctx, task.id)["item"]
    wm.service.owner_delete(wm.bots["on"], wm.user, item["id"], hard=False)
    monkeypatch.setattr(store_module, "derive_key", lambda label: b"different-persisted-key")
    assert _code(lambda: wm.service.link_associated(ctx, task.id)) == "task_link_audit_pending"
    assert _code(lambda: wm.service.owner_delete(wm.bots["on"], wm.user, item["id"],
                                                      hard=True)) == "task_link_audit_pending"
    with wm.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM working_memory_items WHERE id=:id"),
                            {"id": item["id"]}).scalar_one() == 1


@pytest.mark.parametrize("hard", [False, True])
def test_disabled_owner_removes_resolvable_legacy_link(wm, monkeypatch, hard):
    task, _ = _task_links(wm, monkeypatch)
    item = wm.service.store.create(wm.bots["on"], wm.user, {
        "kind": "initiative_link", "initiative_task_id": task.short_id,
        "objective": "legacy item content",
    }, idempotency_key=_key(), actor=Actor("bot", wm.bots["on"]))["item"]
    wm.settings.set_value("bot", wm.bots["on"], KEY, False)
    assert wm.service.owner_delete(wm.bots["on"], wm.user, item["id"], hard=hard)["state"] == (
        "purged" if hard else "removed")
    assert wm.service.owner_delete(wm.bots["on"], wm.user, item["id"], hard=hard)["changed"] is False
    with wm.engine.connect() as conn:
        assert conn.execute(text(
            "SELECT count(*) FROM working_memory_task_suppressions"
            " WHERE bot_id=:bot AND user_id=:user"
        ), {"bot": wm.bots["on"], "user": wm.user}).scalar_one() == 1
    wm.settings.set_value("bot", wm.bots["on"], KEY, True)
    assert _code(lambda: wm.service.link_associated(wm.context(), task.id)) == "removed"


@pytest.mark.parametrize("hard", [False, True])
def test_disabled_owner_cleans_unresolvable_legacy_link(wm, monkeypatch, hard):
    task, catalog = _task_links(wm, monkeypatch)
    item = wm.service.store.create(wm.bots["on"], wm.user, {
        "kind": "initiative_link", "initiative_task_id": task.short_id,
        "objective": "legacy item content",
    }, idempotency_key=_key(), actor=Actor("bot", wm.bots["on"]))["item"]
    resolve = catalog.get
    catalog.get = lambda _ref: (_ for _ in ()).throw(WorkingMemoryError("task_unavailable", "missing"))
    wm.settings.set_value("bot", wm.bots["on"], KEY, False)
    assert wm.service.owner_delete(wm.bots["on"], wm.user, item["id"], hard=hard)["state"] == (
        "purged" if hard else "removed")
    wm.settings.set_value("bot", wm.bots["on"], KEY, True)
    catalog.get = resolve
    # Soft removal can be resolved and suppressed by the later audit; a hard
    # purge without identity cannot prove which task to fence, so gate all links.
    assert _code(lambda: wm.service.link_associated(wm.context(), task.id)) == (
        "task_link_audit_pending" if hard else "removed")
    with wm.engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM working_memory_items WHERE id=:id"),
                            {"id": item["id"]}).scalar_one() == (0 if hard else 1)


def test_protected_owner_cannot_resolve_legacy_link(wm, monkeypatch):
    class ForbiddenCatalog:
        def get(self, _ref):
            pytest.fail("protected owner must not read task catalog")

    wm.service.catalog = ForbiddenCatalog()
    assert _code(lambda: wm.service.owner_delete(
        wm.bots["protected"], wm.user, str(uuid.uuid4()), hard=True)) == "protected_bot"


def test_historical_unidentifiable_purge_gates_new_task_links(wm, monkeypatch):
    task, _ = _task_links(wm, monkeypatch)
    old = wm.service.bot_create(wm.context(), {"objective": "unidentifiable"}, _key())["item"]
    wm.service.owner_delete(wm.bots["on"], wm.user, old["id"], hard=True)
    with wm.engine.begin() as conn:
        conn.execute(text("DELETE FROM working_memory_task_purge_audit WHERE item_id=:id"), {"id": old["id"]})
    assert _code(lambda: wm.service.link_associated(wm.context(), task.id)) == "task_link_audit_pending"
    assert wm.service.store.list_items(wm.bots["on"], wm.user)["items"] == []


def test_reconciliation_updates_task_pointer_and_archives_without_reopening(wm, monkeypatch):
    task, catalog = _task_links(wm, monkeypatch)
    ctx = wm.context()
    item = wm.service.link_associated(ctx, task.id)["item"]
    assert wm.service.bot_list(ctx)["version"] == 1  # unchanged task never bumps
    catalog.get = lambda ref: dataclasses.replace(task, title="New title", status="BLOCKED")
    listed = wm.service.bot_list(ctx)
    assert listed["version"] == 2 and listed["items"][0]["task_title"] == "New title"
    assert "New title" in render_section(wm.engine, bot_id=wm.bots["on"],
                                         user_id=wm.user, session_id=None).text
    catalog.get = lambda ref: dataclasses.replace(task, status="COMPLETED")
    wm.service.bot_list(ctx)
    assert wm.service.store.list_items(wm.bots["on"], wm.user)["items"] == []
    assert wm.service.owner_list(wm.bots["on"], wm.user)["items"][0]["status"] == "done"
    catalog.get = lambda ref: task
    assert _code(lambda: wm.service.link_associated(ctx, task.id)) == "already_linked"


@pytest.mark.parametrize("ending", ["CANCELLED", "reassigned"])
def test_reconciliation_cancels_unavailable_task_ownership(wm, monkeypatch, ending):
    task, catalog = _task_links(wm, monkeypatch)
    ctx = wm.context()
    item = wm.service.link_associated(ctx, task.id)["item"]
    catalog.get = lambda ref: (dataclasses.replace(task, status="CANCELLED") if ending == "CANCELLED"
                               else dataclasses.replace(task, assignee="different-bot"))
    wm.service.bot_list(ctx)
    assert wm.service.owner_list(wm.bots["on"], wm.user)["items"][0]["status"] == "cancelled"
    assert wm.changes(item["id"])[-1]["session_id"] is None


def test_task_pointer_actual_sent_prompt_keeps_pin_until_boundary(wm, ses, monkeypatch):
    from llm_bawt.models.message import Message
    from llm_bawt.service.system_prompt_pin import SystemPromptPinStore, apply_session_pin
    from llm_bawt.service.turn_logs import TurnLogStore
    from llm_bawt.working_memory.task_links import TaskCatalog

    task, catalog = _task_links(wm, monkeypatch)
    monkeypatch.setattr(TaskCatalog, "get", catalog.get)
    TurnLogStore(wm.settings.config)
    sid = ses.new()
    turns = []
    client = SimpleNamespace(
        _backend=SimpleNamespace(pins_system_prompt=True),
        compose_system_prompt=lambda base: base,
    )

    def sent(section, *, cold):
        tid = "turn-" + uuid.uuid4().hex
        turns.append(tid)
        with wm.engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status, user_prompt)"
                " VALUES (:id, now(), '/test', true, :bot, :user, 'ok', 'hi')"
            ), {"id": tid, "bot": wm.bots["on"], "user": wm.user})
        ctx = SimpleNamespace(
            svc=SimpleNamespace(_turn_log_store=SimpleNamespace(engine=wm.engine)),
            bot_id=wm.bots["on"], user_prompt="hi", bridge_reset_action=None,
            thread_binding={"thread_session_id": sid,
                            **({} if cold else {"thread_resume_id": "existing-sdk-session"})},
        )
        messages, decision = apply_session_pin(
            ctx, client, [Message("system", section.text), Message("user", "hi")],
            turn_id=tid, sections={"working_memory": section.meta},
        )
        assert decision is not None
        if decision.is_boundary:
            assert SystemPromptPinStore(wm.engine).confirm(session_id=sid, turn_id=tid)
        return messages[0].content, decision

    try:
        original = render_section(wm.engine, bot_id=wm.bots["on"], user_id=wm.user, session_id=sid)
        first, _ = sent(original, cold=True)
        assert task.short_id not in first
        item = wm.service.link_associated(ses.ctx(sid), task.id)["item"]
        fresh = render_section(wm.engine, bot_id=wm.bots["on"], user_id=wm.user, session_id=sid)
        assert task.short_id in fresh.text and task.id in fresh.text
        kept, decision = sent(fresh, cold=False)
        assert kept == first and task.short_id not in kept
        assert _notice(wm, sid, decision) == prompts.DEFAULT_NOTICE
        assert wm.service.bot_list(ses.ctx(sid))["items"][0]["id"] == item["id"]
        new, boundary = sent(fresh, cold=True)
        assert boundary.is_boundary and task.short_id in new and task.id in new
    finally:
        with wm.engine.begin() as conn:
            conn.execute(text("DELETE FROM turn_logs WHERE id = ANY(:ids)"), {"ids": turns})


# -- Loopy 2a r1 regressions (written by Loopy, adapted) -----------------------

def test_disabled_fallback_then_compaction_notices_on_reenable(wm, ses):
    """B1: a sectionless fallback leaves no candidate_sections, yet the harness may adopt it."""
    from llm_bawt.service.system_prompt_pin import SystemPromptPinStore
    from llm_bawt.service.turn_logs import TurnLogStore
    from llm_bawt.working_memory.turn import WorkingMemoryTurn

    TurnLogStore(wm.settings.config)
    sid = ses.new()
    store = SystemPromptPinStore(wm.engine)
    turns: list[str] = []

    def resolve(body, sections, cold=False):
        tid = "turn-" + uuid.uuid4().hex
        turns.append(tid)
        with wm.engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO turn_logs (id, created_at, path, stream, bot_id, user_id, status, user_prompt)"
                " VALUES (:id, now(), '/test', true, :bot, :user, 'ok', 'hi')"
            ), {"id": tid, "bot": wm.bots["on"], "user": wm.user})
        decision = store.resolve(session_id=sid, turn_id=tid, storage_bot=wm.bots["on"],
                                 rendered=body, cold_start=cold, sections=sections)
        if decision.is_boundary:
            store.confirm(session_id=sid, turn_id=tid)
        return decision

    try:
        original = resolve("WITH WORKING MEMORY", _pin(0, "empty"), cold=True)
        wm.settings.set_value("bot", wm.bots["on"], KEY, False)
        with wm.engine.begin() as conn:
            conn.execute(text("UPDATE turn_logs SET request_json='{}' WHERE id=:id"), {"id": original.pin_turn_id})
        ctx = SimpleNamespace(
            is_agent_backend=True, bridge_reset_action=None, bot_id=wm.bots["on"], user_id=wm.user,
            thread_binding={"thread_session_id": sid}, request=SimpleNamespace(), user_prompt="hi",
            llm_bawt=SimpleNamespace(client=SimpleNamespace(_backend=SimpleNamespace(pins_system_prompt=True))),
            svc=SimpleNamespace(_turn_log_store=SimpleNamespace(engine=wm.engine)),
        )
        disabled = WorkingMemoryTurn.begin(ctx)
        assert disabled.sections is None
        fallback = resolve("NO WORKING MEMORY", disabled.sections)
        assert fallback.snapshot_unverified
        # The CLI's auto-compaction adopts the fallback text, which has no section.
        session_state.bump_compaction_gen(wm.engine, sid)
        wm.settings.set_value("bot", wm.bots["on"], KEY, True)
        kept = resolve("WITH WORKING MEMORY AGAIN", _pin(0, "empty"))
        assert not kept.is_boundary and kept.text == "NO WORKING MEMORY"
        assert _notice(wm, sid, kept) == prompts.DEFAULT_NOTICE
    finally:
        wm.settings.set_value("bot", wm.bots["on"], KEY, True)
        with wm.engine.begin() as conn:
            conn.execute(text("DELETE FROM turn_logs WHERE id = ANY(:ids)"), {"ids": turns})


def test_concurrent_marks_do_not_exhaust_the_connection_pool(wm, ses, monkeypatch):
    """S1: a mark holding its session row reads external changes on its own connection."""
    from sqlalchemy import create_engine

    from llm_bawt.utils.db import DEFAULT_MAX_OVERFLOW, DEFAULT_POOL_SIZE
    from llm_bawt.working_memory.store import WorkingMemoryStore

    count = DEFAULT_POOL_SIZE + DEFAULT_MAX_OVERFLOW
    engine = create_engine(wm.engine.url, pool_size=DEFAULT_POOL_SIZE,
                           max_overflow=DEFAULT_MAX_OVERFLOW, pool_timeout=0.3)
    WorkingMemoryStore(engine).ensure_schema()
    sessions = [ses.new() for _ in range(count)]
    wm.service.bot_create(wm.context(), {"objective": "external change"}, _key())
    original = WorkingMemoryStore.external_changes
    all_locked = threading.Barrier(count)
    local = threading.local()

    def synchronized_read(self, *args, **kwargs):
        local.calls = getattr(local, "calls", 0) + 1
        if local.calls == 2:  # pre-check done; every worker now holds its session row
            all_locked.wait(timeout=5)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(WorkingMemoryStore, "external_changes", synchronized_read)
    monkeypatch.setattr(notice_module, "notice_text", lambda bot: prompts.DEFAULT_NOTICE)
    try:
        with ThreadPoolExecutor(max_workers=count) as pool:
            results = list(pool.map(lambda sid: prepare_notice(
                engine, bot_id=wm.bots["on"], user_id=wm.user, session_id=sid,
                decision=_decision(pinned=_pin(0)), eligible=True), sessions))
        assert results == [prompts.DEFAULT_NOTICE] * count
        assert metrics.snapshot(wm.bots["on"])["notice_failures"] == 0
    finally:
        engine.dispose()
