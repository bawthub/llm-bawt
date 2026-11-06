"""TASK-2120: Campus location/house events and snapshot shaping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from llm_bawt import campus_events as ce
from llm_bawt.house_watcher import (DOWN, RECREATING, UNKNOWN, UP, HouseStates, frames_for,
                                    merge_flags)
from llm_bawt.service.routes.campus import house_rows

AGENT = {"bot_type": "agent", "agent_backend": "claude-code"}
LEASE_H1 = "agent:house:h1:lease:claude-code"


def fields(house=None, pinned=False, **over):
    return {**AGENT, "house": house, "house_pinned": pinned, **over}


# --- assignment frames -------------------------------------------------------

def test_move_in_and_out_report_routes_and_reasons():
    live = {LEASE_H1: True}
    move_in = ce.location_frame("b", fields(), fields("h1"), "", live)
    assert move_in == {"_type": "bot_location_changed", "kind": "assignment", "bot_id": "b",
                       "from": {"runtime": "claude-code-bridge"},
                       "to": {"runtime": "house", "house": "h1"}, "reason": "move_in"}
    move_out = ce.location_frame("b", fields("h1"), fields(), "", live)
    assert move_out["reason"] == "move_out" and move_out["to"] == {"runtime": "claude-code-bridge"}


def test_named_reason_wins_and_route_reasons_are_not_accepted_for_writes():
    assert ce.location_frame("b", fields("h1"), fields(), "idle_evict", {})["reason"] == "idle_evict"
    assert ce.location_frame("b", fields("h1"), fields(), "house_up", {})["reason"] == "move_out"


def test_pin_only_is_pinned_change_with_same_house():
    frame = ce.location_frame("b", fields("h1"), fields("h1", pinned=True), "", {LEASE_H1: True})
    assert frame["reason"] == "pinned_change"
    assert frame["from"]["house"] == frame["to"]["house"] == "h1"


def test_assignment_to_dead_house_keeps_bridge_route_but_records_house():
    frame = ce.location_frame("b", fields(), fields("h1"), "", {LEASE_H1: False})
    assert frame["to"] == {"runtime": "claude-code-bridge", "house": "h1", "house_live": False}


def test_no_location_change_no_frame():
    assert ce.location_frame("b", fields(), fields(), "", {}) is None
    # Chat bots stay in the app whatever their house field says.
    chat = {"bot_type": "chat", "agent_backend": None, "house_pinned": False}
    assert ce.location_frame("d", {**chat, "house": None}, {**chat, "house": "h1"}, "", {}) is None


def test_create_and_delete_only_for_housed_profiles():
    assert ce.location_frame("b", {}, fields(), "", {}) is None
    created = ce.location_frame("b", {}, fields("h1"), "", {LEASE_H1: True})
    assert created["from"] is None and created["reason"] == "move_in"
    assert ce.location_frame("b", fields(), {}, "move_out", {}) is None
    deleted = ce.location_frame("b", fields("h1"), {}, "move_out", {LEASE_H1: True})
    assert deleted["to"] is None and deleted["reason"] == "move_out"


# --- commit binding ----------------------------------------------------------

@pytest.fixture
def sent(monkeypatch):
    out: list = []
    monkeypatch.setattr(ce, "emit_tenancy_changes", lambda pending, config=None: out.append(pending))
    return out


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    with Session(engine) as s:
        s.execute(text("select 1"))
        yield s


def test_frame_sent_only_after_commit(session, sent):
    ce.note_tenancy_change(session, "b", fields(), fields("h1"), None)
    assert sent == []
    session.commit()
    assert [p[0] for p in sent[0]] == ["b"]
    session.commit()  # listener removed: a later commit of this session sends nothing more
    assert len(sent) == 1


def test_rollback_and_noop_send_nothing(session, sent):
    ce.note_tenancy_change(session, "b", fields(), fields("h1"), None)
    session.rollback()
    session.commit()
    ce.note_tenancy_change(session, "b", fields("h1"), fields("h1"), None)
    session.commit()
    assert sent == []


def test_borrowed_session_batches_every_write_until_its_commit(session, sent):
    ce.note_tenancy_change(session, "a", fields(), fields("h1"), "move_in")
    ce.note_tenancy_change(session, "b", fields("h2"), fields(), "move_out")
    session.commit()
    assert [(p[0], p[3]) for p in sent[0]] == [("a", "move_in"), ("b", "move_out")]


def test_publish_stamps_shared_seq_identity_and_ts():
    class FakeRedis:
        def __init__(self):
            self.n, self.added = 0, []

        def incr(self, key):
            assert key == ce.SEQ_KEY
            self.n += 1
            return self.n

        def xadd(self, stream, data, **_):
            self.added.append((stream, data))

    client = FakeRedis()
    ce.publish(client, [{"_type": "x"}, {"_type": "y"}])
    import json
    frames = [json.loads(d["payload"]) for _, d in client.added]
    assert [f["seq"] for f in frames] == [1, 2]
    assert len({f["event_id"] for f in frames}) == 2 and all(f["ts"] for f in frames)
    assert {s for s, _ in client.added} == {"events:system:__config__"}


# --- house transitions -------------------------------------------------------

def test_lease_lifecycle_transitions():
    s = HouseStates()
    assert s.on_set("h1", "claude-code", "7") == [("h1", None, UP)]
    assert s.on_set("h1", "claude-code", "7") == []  # heartbeat
    assert s.on_gone("h1", "claude-code", released=True) == [("h1", UP, RECREATING)]
    assert s.on_set("h1", "claude-code", "8") == [("h1", RECREATING, UP)]
    assert s.on_gone("h1", "claude-code", released=False) == [("h1", UP, DOWN)]
    assert s.on_gone("h1", "claude-code", released=False) == []


def test_restart_between_heartbeats_is_recreated():
    s = HouseStates()
    s.on_set("h1", "claude-code", "7")
    assert s.on_set("h1", "claude-code", "9") == [("h1", UP, RECREATING), ("h1", RECREATING, UP)]


def test_reseed_reports_what_changed_while_disconnected():
    s = HouseStates()
    s.reseed({("h1", "claude-code"): "1", ("h2", "claude-code"): "1"})
    assert s.reseed({("h2", "claude-code"): "2"}) == [
        ("h1", UP, DOWN), ("h2", UP, RECREATING), ("h2", RECREATING, UP)]


def test_house_frames_move_tenant_routes_without_profile_write():
    tenant = SimpleNamespace(slug="b", house="h1", house_pinned=True, **AGENT)
    frames = frames_for([("h1", UP, DOWN)], lambda house: [tenant])
    assert frames[0] == {"_type": "house_state_changed", "house_id": "h1", "from": UP,
                         "to": DOWN, "tenants": ["b"]}
    assert frames[1]["kind"] == "route" and frames[1]["reason"] == "house_down_fallback"
    assert frames[1]["to"] == {"runtime": "claude-code-bridge", "house": "h1",
                               "house_live": False, "pinned": True}
    back = frames_for([("h1", DOWN, UP)], lambda house: [tenant])[1]
    assert back["reason"] == "house_up" and back["to"]["runtime"] == "house"
    # recreating -> down is not a route change: both are "no live lease".
    assert len(frames_for([("h1", RECREATING, DOWN)], lambda house: [tenant])) == 1


def test_first_sighting_announces_house_only():
    calls = []
    frames = frames_for([("h1", None, UP)], lambda house: calls.append(house) or [])
    assert [f["_type"] for f in frames] == ["house_state_changed"] and calls == []


def test_merge_flags_keeps_existing():
    assert set(merge_flags("")) == set("K$gx")
    assert set(merge_flags("Elg")) == set("Elg") | set("K$x")
    assert merge_flags("KA") == "AK"


# --- snapshot houses ---------------------------------------------------------

POOL = [{"id": "h1", "harness": "claude-code", "preview_host": "https://dev1.example.test"},
        {"id": "h2", "harness": "claude-code", "preview_host": "https://dev2.example.test"}]


def test_configured_empty_and_down_houses_are_listed():
    rows = house_rows(POOL, {"h1": {"claude-code"}}, {}, {})
    assert [(r["id"], r["state"], r["configured"]) for r in rows] == [
        ("h1", UP, True), ("h2", DOWN, True)]
    assert rows[1]["preview_host"] == "https://dev2.example.test"


def test_watcher_distinguishes_recycle_from_death():
    rows = house_rows(POOL, {}, {"h1": RECREATING}, {})
    assert rows[0]["state"] == RECREATING and rows[1]["state"] == DOWN


def test_unreadable_redis_is_unknown_not_down():
    rows = house_rows(POOL, None, {}, {"h1": ["b"]})
    assert {r["state"] for r in rows} == {UNKNOWN}
    assert rows[0]["tenants"] == ["b"]


def test_unconfigured_observed_house_is_flagged():
    rows = house_rows(None, {"h9": {"codex"}}, {}, {"h3": ["b"]})
    assert [(r["id"], r["configured"], r["state"], r["harness"]) for r in rows] == [
        ("h3", False, DOWN, None), ("h9", False, UP, "codex")]


def test_pool_file_exposes_only_id_harness_and_origin(tmp_path, monkeypatch):
    import json

    from llm_bawt.service.routes.campus import POOL_FILE_ENV, configured_pool

    monkeypatch.delenv(POOL_FILE_ENV, raising=False)
    assert configured_pool() is None
    empty = tmp_path / "empty.json"
    empty.write_text("")
    monkeypatch.setenv(POOL_FILE_ENV, str(empty))
    assert configured_pool() is None  # compose mounts /dev/null when unset
    pool = tmp_path / "houses.json"
    pool.write_text(json.dumps({
        "image": "img", "base_image": "base", "preview_network": "net",
        "service_token_volume": "vol", "frontend_env_file": "/secret/frontend.env",
        "houses": [{"id": "h1", "harness": "claude-code", "origin": "https://dev1.example.test",
                    "memory": "4g"}]}))
    monkeypatch.setenv(POOL_FILE_ENV, str(pool))
    assert configured_pool() == [{"id": "h1", "harness": "claude-code",
                                  "preview_host": "https://dev1.example.test"}]
    pool.write_text("{not json")
    assert configured_pool() is None
