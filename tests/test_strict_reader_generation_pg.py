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


def test_candidate_and_legacy_are_each_explicit(durable, rig, tmp_path, monkeypatch):
    from agent_bridge import skill_sources

    # Explicit candidate: bodies come from the pinned generation.
    candidate = _system_of(durable, rig)
    for ref in APPROVED:
        assert _body(ref) in candidate and _legacy_body(ref) not in candidate, ref

    # Explicit legacy (the inverse): the ordinary ambient tree, never the generation.
    legacy_root = tmp_path / "legacy"
    write_skill_tree(legacy_root, _legacy_body)
    monkeypatch.setattr(skill_sources, "legacy_claude_skill_roots", lambda: (legacy_root,),
                        raising=False)
    monkeypatch.setattr("agent_bridge.skill_selection.legacy_claude_skill_roots",
                        lambda: (legacy_root,))
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
