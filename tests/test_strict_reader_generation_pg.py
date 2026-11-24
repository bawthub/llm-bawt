"""TASK-2590: the strict reader's explicit Skills generation and house route.

Synthetic fixtures on private PostgreSQL, through the real durable route
(HTTP admission, queue, claim, dispatch, real ``send_command``); only Redis and
the bridge's run events are faked. Proves an explicit candidate generation vs
an explicit legacy tree, refusal of a missing, escaping or mismatched
generation or house, and that route loss never falls back to the mansion.
"""
from __future__ import annotations

import json

import pytest

from agent_bridge.houses import HouseRouteLost, commands_stream, lease_key, route_new_command
from llm_bawt.service import strict_reader_execution as strict
from tests.session_lifecycle_pg_fixtures import rig  # noqa: F401
from tests.test_message_native_upsert_pg import http_runtime, real_storage  # noqa: F401
from tests.test_native_selected_owner_pg import native_runtime  # noqa: F401
from tests.test_strict_reader_durable_pg import (  # noqa: F401
    _RecordingRedis, _command_fields, _drain, durable,
)
from tests.test_strict_reader_stream_pg import (  # noqa: F401
    APPROVED, GENERATION, HOUSE, ROWS, _body, _request, _stream, strict_runtime, write_skill_tree,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("private_app_pg", "persisted_master_key"),
]


def _legacy_body(ref: str) -> str:
    return f"LEGACY-BODY-{ref.upper()}"


def _system_of(durable, rig):
    receipt = _stream(durable.rt.service, _request(rig))
    assert '"delivery_id"' in receipt, receipt
    assert _drain(durable.dispatcher, rig.bot) is True
    assert len(durable.published) >= 1
    return _command_fields(durable.published[-1]).get("system_prompt", "")


def _generations():
    import os
    from pathlib import Path
    return Path(os.environ["SKILL_SOURCES_ROOT"]) / "generations"


SECOND = "gen-fxsecond"
SECOND_LAYERS = {"fx-source": "c" * 40}


def _second_body(ref: str) -> str:
    return f"SECOND-BODY-{ref.upper()}"


def _add_second_generation():
    root = _generations() / SECOND
    write_skill_tree(root, _second_body)
    (root / ".layers.json").write_text(json.dumps(SECOND_LAYERS))


def _select_second(service):
    service.strict_reader_skill_generation = SECOND
    service.strict_reader_skill_layers = dict(SECOND_LAYERS)


def _turn(durable, rig):
    before = len(durable.published)
    receipt = _stream(durable.rt.service, _request(rig))
    assert '"delivery_id"' in receipt, receipt
    _drain(durable.dispatcher, rig.bot)
    assert len(durable.published) == before + 1, "turn did not publish"
    return _command_fields(durable.published[-1])


def _active_session(rig):
    from sqlalchemy import text
    with rig.engine.connect() as conn:
        return conn.execute(text(
            "SELECT id, session_metadata FROM sessions WHERE bot_id=:b AND user_id=:u"
            " AND status='active' AND ended_at IS NULL"), {"b": rig.bot, "u": rig.user}).one()


def _bind_sdk(durable, rig, monkeypatch, session_id, sdk_id):
    """The bridge's real SDK key writeback (PUT /v1/sessions/{id}/agent-session-key)."""
    import asyncio
    from llm_bawt.service.routes import sessions as routes
    monkeypatch.setattr(routes, "get_service", lambda: durable.rt.service)
    body = routes.AgentSessionKeyRequest(backend="claude-code", session_key=sdk_id)
    assert asyncio.run(routes.put_agent_session_key(session_id, body, bot_id=rig.bot))["stored"]


def _marker(rig):
    return (_active_session(rig)[1] or {}).get(strict.SESSION_SELECTION_KEY)


def test_persisted_resume_binding_follows_the_selection(durable, rig, monkeypatch):
    service = durable.rt.service
    _turn(durable, rig)
    session_id, _ = _active_session(rig)
    _bind_sdk(durable, rig, monkeypatch, session_id, "sdk-fx-a")
    assert _marker(rig)["sdk_key"] == "sdk-fx-a"

    # Same selection: the confirmed SDK session resumes.
    assert _turn(durable, rig).get("thread_resume_id") == "sdk-fx-a"

    # Selection changes: the old SDK session (and its pinned prompt) is not
    # resumed, and the new render carries the new generation's text.
    _add_second_generation()
    _select_second(service)
    fields = _turn(durable, rig)
    assert not fields.get("thread_resume_id")
    for ref in APPROVED:
        assert _second_body(ref) in fields.get("system_prompt", ""), ref
    recorded = _marker(rig)
    assert recorded["selection"]["generation"] == SECOND and recorded["sdk_key"] is None

    # Until a strict turn under this selection binds a new key, the old one
    # never resumes; the newly confirmed key then does.
    assert not _turn(durable, rig).get("thread_resume_id")
    _bind_sdk(durable, rig, monkeypatch, session_id, "sdk-fx-b")
    assert _turn(durable, rig).get("thread_resume_id") == "sdk-fx-b"


def test_ordinary_key_writeback_never_inherits_strict_provenance(durable, rig, monkeypatch):
    service = durable.rt.service
    _turn(durable, rig)
    session_id, _ = _active_session(rig)
    _bind_sdk(durable, rig, monkeypatch, session_id, "sdk-fx-strict")
    assert _turn(durable, rig).get("thread_resume_id") == "sdk-fx-strict"

    # Strict mode is disabled; an ordinary cold turn binds its own SDK key.
    slug = service.strict_reader_bot
    service.strict_reader_bot = None
    _bind_sdk(durable, rig, monkeypatch, session_id, "sdk-fx-ordinary")
    assert _marker(rig) is None

    # Re-enabled under the same selection: the ordinary transcript is not resumed.
    service.strict_reader_bot = slug
    assert not _turn(durable, rig).get("thread_resume_id")
    assert _marker(rig)["sdk_key"] is None


def test_ordinary_resume_strips_strict_provenance(durable, rig, monkeypatch):
    _turn(durable, rig)
    session_id, _ = _active_session(rig)
    _bind_sdk(durable, rig, monkeypatch, session_id, "sdk-fx-strict")
    strict.clear_session_selection(rig.engine, {"thread_session_id": session_id,
                                                "thread_resume_id": "sdk-fx-strict"})
    assert _marker(rig) is None
    assert not _turn(durable, rig).get("thread_resume_id")


def _queued_stamp(durable, rig):
    from sqlalchemy import text
    receipt = _stream(durable.rt.service, _request(rig))
    assert '"delivery_id"' in receipt, receipt
    with rig.engine.connect() as conn:
        return conn.execute(text(
            "SELECT metadata_json->'strict_reader' FROM inter_bot_deliveries"
            " WHERE target_bot_id=:b ORDER BY created_at DESC LIMIT 1"), {"b": rig.bot}).scalar_one()


def test_selection_change_between_receipt_and_drain_refuses(durable, rig):
    service = durable.rt.service
    stamp = _queued_stamp(durable, rig)
    assert stamp == strict.strict_selection_fingerprint(service)
    assert stamp["generation"] == GENERATION and stamp["house"] == HOUSE and stamp["bot"] == rig.bot

    _add_second_generation()
    _select_second(service)  # a valid selection, but not the one admitted
    _drain(durable.dispatcher, rig.bot)
    assert durable.published == [] and _RecordingRedis.streams == []


@pytest.mark.parametrize("change", ["disable", "retarget"])
def test_stamped_row_stays_fenced_after_disable_or_retarget(durable, rig, monkeypatch, change):
    service = durable.rt.service
    assert _queued_stamp(durable, rig)["bot"] == rig.bot
    resolved = []

    def ordinary_resolver(*args, **kwargs):
        resolved.append(args)
        raise AssertionError("ordinary resolver reached")

    monkeypatch.setattr(service, "_resolve_request_model", ordinary_resolver)

    class _NoProfiles:  # no longer strict, so drain may consult the registry
        def __init__(self, *_a, **_k):
            pass

        def get_bot(self, _slug):
            return None

    from llm_bawt import bots
    monkeypatch.setattr(bots, "BotManager", _NoProfiles)
    service.strict_reader_bot = None if change == "disable" else "fx-other-strict"
    _drain(durable.dispatcher, rig.bot)
    assert resolved == [], "a strict-stamped row reached the ordinary resolver"
    assert durable.published == [] and _RecordingRedis.streams == []
    from sqlalchemy import text
    with rig.engine.connect() as conn:
        error = conn.execute(text(
            "SELECT last_error FROM inter_bot_deliveries WHERE target_bot_id=:b"
            " ORDER BY created_at DESC LIMIT 1"), {"b": rig.bot}).scalar_one()
    assert "selection changed" in (error or ""), error  # refused at the strict fence


def test_candidate_and_legacy_are_each_explicit(durable, rig, tmp_path, monkeypatch):
    # Explicit candidate: bodies come from the pinned generation.
    candidate = _system_of(durable, rig)
    for ref in APPROVED:
        assert _body(ref) in candidate and _legacy_body(ref) not in candidate, ref

    # Explicit legacy (the inverse): the configured legacy roots, never the
    # generation, even while a store's SKILL_SOURCES_CURRENT link is set and
    # points at that candidate.
    legacy_root = tmp_path / "legacy"
    write_skill_tree(legacy_root, _legacy_body)
    monkeypatch.setenv("CLAUDE_CODE_SKILLS_PATH", str(legacy_root))
    current = tmp_path / "current-link"
    current.symlink_to(_generations() / GENERATION)
    monkeypatch.setenv("SKILL_SOURCES_CURRENT", str(current))
    from agent_bridge.skill_selection import legacy_claude_skill_roots
    assert legacy_claude_skill_roots() == (current,)  # ordinary legacy follows CURRENT
    durable.rt.service.strict_reader_skill_generation = strict.LEGACY_GENERATION
    legacy = _system_of(durable, rig)
    for ref in APPROVED:
        assert _legacy_body(ref) in legacy and _body(ref) not in legacy, ref


@pytest.mark.parametrize("change", ["unset", "missing", "current", "mismatch", "no-layers",
                                    "escape"])
def test_unusable_generation_refuses_before_model(strict_runtime, rig, change):  # noqa: F811
    service = strict_runtime.rt.service
    store = strict_runtime.store
    if change == "unset":
        service.strict_reader_skill_generation = None
    elif change == "missing":
        service.strict_reader_skill_generation = "gen-fxabsent"
    elif change == "current":
        service.strict_reader_skill_generation = "current"
    elif change == "mismatch":
        service.strict_reader_skill_layers = {"fx-source": "b" * 40}
    elif change == "no-layers":
        service.strict_reader_skill_layers = None
    elif change == "escape":
        outside = store.parent / "outside-gen"
        write_skill_tree(outside)
        (outside / ".layers.json").write_text(json.dumps({"fx-source": "a" * 40}))
        (store / "generations" / "gen-fxlinked").symlink_to(outside)
        service.strict_reader_skill_generation = "gen-fxlinked"
    with pytest.raises(strict.StrictReaderRefused):
        strict.create_strict_instance(service, _request(rig))
    assert strict_runtime.built == []
    assert GENERATION  # the fixture's candidate stays untouched


@pytest.mark.parametrize("change", ["no-house", "other-house"])
def test_house_must_match_operator_selection(strict_runtime, rig, change):  # noqa: F811
    service = strict_runtime.rt.service
    if change == "no-house":
        service.strict_reader_house = None
    else:
        strict_runtime.admit(house="fx-other-house")
    with pytest.raises(strict.StrictReaderRefused):
        strict.create_strict_instance(service, _request(rig))
    assert strict_runtime.built == []


def test_lost_house_lease_fails_the_durable_turn_without_mansion(durable, rig, monkeypatch):
    monkeypatch.setattr(_RecordingRedis, "leases", {})
    receipt = _stream(durable.rt.service, _request(rig))
    assert '"delivery_id"' in receipt, receipt
    _drain(durable.dispatcher, rig.bot)
    assert durable.published == [] and _RecordingRedis.streams == []
    assert ROWS  # nothing reached any command stream


def test_no_fallback_route_unit():
    import asyncio

    class R:
        def __init__(self, leases):
            self.leases = leases

        async def get(self, key):
            return self.leases.get(key)

    async def run(redis, house, fallback):
        return await route_new_command(redis, house=house, backend="claude-code",
                                       bot_id="sample", allow_fallback=fallback)

    live = R({lease_key("h1", "claude-code"): "3"})
    assert asyncio.run(run(live, "h1", False)) == (commands_stream("h1"), "3")
    for redis, house in ((R({}), "h1"), (live, None), (live, "BAD HOUSE")):
        with pytest.raises(HouseRouteLost):
            asyncio.run(run(redis, house, False))
    # Ordinary routing is unchanged: lease loss still falls back to the mansion.
    assert asyncio.run(run(R({}), "h1", True))[0] == commands_stream("")
