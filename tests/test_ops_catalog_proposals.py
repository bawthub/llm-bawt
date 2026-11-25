"""TASK-2630: bots propose ops catalog edits; a person applies a batch at once."""

import json

import pytest
from sqlmodel import create_engine
from sqlalchemy.pool import StaticPool

from llm_bawt.ops import catalog_proposals as cp
from llm_bawt.ops.store import OpsStore


@pytest.fixture
def store(monkeypatch):
    s = object.__new__(OpsStore)
    s.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    s.events = None
    s._ensure_tables_exist()
    bots = {"vex"}

    def refuse(engine, actor):
        if not actor or actor in bots:
            raise PermissionError("a bot cannot decide catalog edits")
    monkeypatch.setattr(cp, "_refuse_bot", refuse)
    for slug in ("test.one", "test.two"):
        s.create_operation(dict(
            slug=slug, title=slug, description="d", enabled=True,
            command_script='{"action":"restart","container_name":"old-repo"}',
            args_schema_json='{"type":"object","additionalProperties":false,"properties":{}}',
            args_defaults_json="{}", timeout_seconds=60, max_output_bytes=1000, risk_level="low"),
            actor="seed")
    return s


def _edit(slug, version=1, name="new-repo"):
    return {"slug": slug, "expect_version": version,
            "changes": {"command_script": json.dumps({"action": "restart", "container_name": name}, separators=(",", ":"))}}


def test_batch_approves_with_one_decision_and_records_revisions(store):
    batch = cp.propose(store, items=[_edit("test.one"), _edit("test.two")],
                       reason="rename repo", proposed_by="vex")
    pending = cp.list_batches(store)
    assert [len(b["items"]) for b in pending] == [2]
    item = pending[0]["items"][0]
    assert item["before"] == {"command_script": '{"action":"restart","container_name":"old-repo"}'}
    out = cp.decide(store, batch_id=batch["batch_id"], decided_by="person", approve=True)
    assert out == {"batch_id": batch["batch_id"], "status": "approved",
                   "applied": {"test.one": 2, "test.two": 2}}
    for slug in ("test.one", "test.two"):
        row = store.get_operation_by_slug(slug)
        assert "new-repo" in row.command_script and row.updated_by == "person"
        revisions, total = store.list_revisions(slug)
        assert total == 2 and revisions[0].actor == "person"
    assert cp.list_batches(store) == []
    with pytest.raises(cp.ProposalError):
        cp.decide(store, batch_id=batch["batch_id"], decided_by="person", approve=True)


def test_moved_operation_makes_whole_batch_stale(store):
    batch = cp.propose(store, items=[_edit("test.one"), _edit("test.two")],
                       reason="rename repo", proposed_by="vex")
    store.update_operation("test.two", {"title": "changed"}, actor="person")
    out = cp.decide(store, batch_id=batch["batch_id"], decided_by="person", approve=True)
    assert out["status"] == "stale" and out["moved"] == ["test.two"]
    assert "old-repo" in store.get_operation_by_slug("test.one").command_script
    assert cp.list_batches(store, status="stale")[0]["status"] == "stale"


def test_bot_cannot_decide_and_reject_applies_nothing(store):
    batch = cp.propose(store, items=[_edit("test.one")], reason="r", proposed_by="vex")
    with pytest.raises(PermissionError):
        cp.decide(store, batch_id=batch["batch_id"], decided_by="vex", approve=True)
    assert cp.decide(store, batch_id=batch["batch_id"], decided_by="person",
                     approve=False)["status"] == "rejected"
    assert store.get_operation_by_slug("test.one").version == 1


@pytest.mark.parametrize("item, message", [
    (_edit("test.one", version=3), "not 3"),
    ({"slug": "test.one", "expect_version": 1, "changes": {"slug": "x"}}, "not editable"),
    (_edit("test.one", name="old-repo"), "match the current"),
    ({"slug": "test.one", "expect_version": 1, "changes": {"timeout_seconds": 0}}, "timeout_seconds"),
    (_edit("test.missing"), "no such operation"),
])
def test_invalid_proposals_are_refused(store, item, message):
    with pytest.raises(cp.ProposalError, match=message):
        cp.propose(store, items=[item], reason="r", proposed_by="vex")
    assert cp.list_batches(store) == []
