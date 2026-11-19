"""Private-PG proof of the guidance approval read/decide surface (TASK-893 slice 5)."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from llm_bawt.bot_tags import catalog
from llm_bawt.memory import guidance_view as gv
from llm_bawt.service.routes import guidance as route
from llm_bawt.service.routes.service_auth import require_service_token
from tests.test_guidance_pg import engine  # noqa: F401  (module fixture)
from tests.test_guidance_run_pg import rig  # noqa: F401  (fixture)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("private_app_pg")]


@pytest.fixture
def client(rig, monkeypatch):  # noqa: F811
    app = FastAPI()
    app.include_router(route.router)
    app.dependency_overrides[require_service_token] = lambda: None
    monkeypatch.setattr(route, "_engine", lambda: rig.engine)
    monkeypatch.setattr(route, "get_service", lambda: None)

    async def _no_publish(*a, **k):
        return True
    monkeypatch.setattr(route.state_events, "publish_state_changed", _no_publish)
    return TestClient(app)


def _pending(rig, statement="short answers"):  # noqa: F811
    ev = rig.msg("I really prefer short answers")
    return rig.propose(rig.doc(rig.pref("p1", [ev], statement=statement))), ev


def test_service_token_is_required(rig):  # noqa: F811
    app = FastAPI()
    app.include_router(route.router)
    r = TestClient(app).get("/v1/guidance", params={"user_id": rig.user})
    assert r.status_code in (401, 403, 503)


def test_view_shows_pending_change_with_excerpt(rig, client):  # noqa: F811
    pid, ev = _pending(rig)
    body = client.get("/v1/guidance", params={"user_id": rig.user}).json()
    assert body["current"] is None and [p["id"] for p in body["pending"]] == [pid]
    ch = body["pending"][0]["changes"]
    assert ch[0]["kind"] == "added" and ch[0]["after"] == "short answers"
    assert ch[0]["evidence"][0]["text"] == "I really prefer short answers"
    assert ch[0]["evidence"][0]["unavailable"] is False


def test_view_is_empty_for_a_user_without_guidance(rig, client):  # noqa: F811
    body = client.get("/v1/guidance", params={"user_id": rig.other}).json()
    assert body["pending"] == [] and body["history"] == [] and body["current"] is None


def test_approve_derives_the_decider_from_user_id(rig, client):  # noqa: F811
    pid, _ = _pending(rig)
    r = client.post(f"/v1/guidance/proposals/{pid}/decide",
                    json={"user_id": rig.user, "approve": True})
    assert r.status_code == 200 and r.json()["status"] == "approved"
    with rig.engine.connect() as c:
        assert c.execute(text("SELECT decided_by FROM guidance_proposals WHERE id=:i"),
                         {"i": pid}).scalar() == rig.user
    body = client.get("/v1/guidance", params={"user_id": rig.user}).json()
    assert body["current"]["version"] == 1 and body["history"][0]["status"] == "approved"
    again = client.post(f"/v1/guidance/proposals/{pid}/decide",
                        json={"user_id": rig.user, "approve": False})
    assert again.status_code == 409


def test_conflicting_decided_by_is_rejected_by_schema(rig, client):  # noqa: F811
    pid, _ = _pending(rig)
    r = client.post(f"/v1/guidance/proposals/{pid}/decide",
                    json={"user_id": rig.user, "decided_by": rig.other, "approve": True})
    assert r.status_code == 422 and rig.status(pid) == "pending"


@pytest.mark.parametrize("who", ["other", "", "unknown", "bot:al", "bot"])
def test_only_the_owner_person_can_decide(rig, client, who):  # noqa: F811
    pid, _ = _pending(rig)
    user = {"other": rig.other, "bot": rig.bot}.get(who, who)
    r = client.post(f"/v1/guidance/proposals/{pid}/decide", json={"user_id": user, "approve": True})
    assert r.status_code in (403, 422) and rig.status(pid) == "pending"


def test_stale_proposal_reports_stale(rig, client):  # noqa: F811
    first, ev = _pending(rig)
    second = rig.propose(rig.doc(rig.pref("p1", [ev], statement="very short answers")))
    rig.approve(first)
    r = client.post(f"/v1/guidance/proposals/{second}/decide",
                    json={"user_id": rig.user, "approve": True})
    assert r.json()["status"] == "stale" and rig.status(second) == "stale"


def test_forgotten_evidence_is_unavailable(rig, client):  # noqa: F811
    pid, ev = _pending(rig)
    with rig.engine.begin() as c:
        c.execute(text("DELETE FROM messages WHERE bot_id=:b AND id=:m"),
                  {"b": rig.bot, "m": ev["message_id"]})
    e = client.get("/v1/guidance", params={"user_id": rig.user}).json()["pending"][0]["changes"][0]["evidence"][0]
    assert e["unavailable"] is True and "text" not in e


def test_newly_protected_source_is_unavailable(rig, client):  # noqa: F811
    _pending(rig)
    with rig.engine.begin() as c:
        catalog.assign(c, rig.bot, ["private"], "test")
    e = client.get("/v1/guidance", params={"user_id": rig.user}).json()["pending"][0]["changes"][0]["evidence"][0]
    assert e["unavailable"] is True and "text" not in e


def test_protection_refuses_before_any_history_query(rig):  # noqa: F811
    class NoSQL:
        def execute(self, *a, **k):
            raise AssertionError("history queried for a protected source")
    out = gv.excerpt(NoSQL(), user_id=rig.user, ref={"bot_id": rig.prot, "message_id": "m"},
                     protected=[rig.prot])
    assert out["unavailable"] is True


def test_view_runs_no_ddl_when_tables_are_missing():
    class Conn:
        def __init__(self):
            self.sql = []

        def execute(self, stmt, params=None):
            self.sql.append(str(stmt))

            class R:
                def scalar(self):
                    return None
            return R()
    conn = Conn()
    out = gv.view(conn, user_id="sample-user", protected=["p"])
    assert out["pending"] == [] and out["current"] is None
    assert not any("CREATE" in s.upper() for s in conn.sql)


def test_changes_classifies_rewords_retirements_and_new_evidence():
    import json
    ev1, ev2 = {"bot_id": "b", "message_id": "1"}, {"bot_id": "b", "message_id": "2"}
    base = {"v": 1, "mandate": "", "preferences": [
        {"id": "a", "statement": "x", "status": "active", "evidence": [ev1]},
        {"id": "b", "statement": "y", "status": "active", "evidence": [ev1]},
        {"id": "c", "statement": "z", "status": "active", "evidence": [ev1]},
        {"id": "d", "statement": "same", "status": "active", "evidence": [ev1]}]}
    new = {"v": 1, "mandate": "", "preferences": [
        {"id": "a", "statement": "x2", "status": "active", "evidence": [ev2]},
        {"id": "b", "statement": "y", "status": "superseded", "evidence": [ev1]},
        {"id": "c", "statement": "z", "status": "active", "evidence": [ev1, ev2]},
        {"id": "d", "statement": "same", "status": "active", "evidence": [ev1]}]}
    ch = {c["id"]: c for c in gv.changes(json.dumps(base), json.dumps(new))}
    assert {k: c["kind"] for k, c in ch.items()} == {
        "a": "reworded", "b": "retired", "c": "evidence_changed"}
    assert [(r["message_id"], r["new"]) for r in ch["c"]["evidence"]] == [("1", False), ("2", True)]


def test_citation_only_removal_is_shown(rig, client):  # noqa: F811
    a, b = rig.msg("I prefer short answers"), rig.msg("Really, keep it short")
    rig.approve(rig.propose(rig.doc(rig.pref("p1", [a, b]))))
    pid = rig.propose(rig.doc(rig.pref("p1", [a])))
    ch = client.get("/v1/guidance", params={"user_id": rig.user}).json()["pending"][0]
    assert ch["id"] == pid
    (c,) = ch["changes"]
    assert c["kind"] == "evidence_changed"
    assert [e["message_id"] for e in c["removed_evidence"]] == [b["message_id"]]
    assert c["removed_evidence"][0]["text"] == "Really, keep it short"


def test_rewrite_shows_retained_citations(rig, client):  # noqa: F811
    a, b = rig.msg("I prefer short answers"), rig.msg("Lists help me")
    rig.approve(rig.propose(rig.doc(rig.pref("p1", [a]))))
    rig.propose(rig.doc(rig.pref("p1", [a, b], statement="short answers, in lists")))
    (c,) = client.get("/v1/guidance", params={"user_id": rig.user}).json()["pending"][0]["changes"]
    assert c["kind"] == "reworded"
    assert [(e["message_id"], e["new"]) for e in c["evidence"]] == [
        (a["message_id"], False), (b["message_id"], True)]
    assert c["removed_evidence"] == []


def test_unnamed_differences_are_reported_as_changed():
    import json
    ev = [{"bot_id": "b", "message_id": "1"}]
    p = {"id": "a", "statement": "x", "status": "active", "evidence": ev}
    base = {"v": 1, "mandate": "", "preferences": [p]}
    new = {"v": 1, "mandate": "", "preferences": [dict(p, first_seen_ts=5.0)]}
    (c,) = gv.changes(json.dumps(base), json.dumps(new))
    assert c["kind"] == "changed"


# -- operator run route (slice 7a) ---------------------------------------------

def test_run_route_previews_then_commits_a_pending_proposal(rig, client, monkeypatch):  # noqa: F811
    from llm_bawt.memory import guidance_run as gr

    ev = rig.msg("I prefer short answers")

    def drafter(base, evidence):
        return {"preferences": [{"id": "p1", "statement": "short answers", "status": "active",
                                 "evidence": [{"bot_id": ev["bot_id"],
                                               "message_id": ev["message_id"]}]}],
                "reason": "stated"}
    monkeypatch.setattr(gr, "draft_with_utility", drafter)
    pre = client.post("/v1/guidance/runs", json={"user_id": rig.user, "bot_id": rig.bot})
    assert pre.status_code == 200 and pre.json()["status"] == "preview_only"
    r = client.post("/v1/guidance/runs",
                    json={"user_id": rig.user, "bot_id": rig.bot, "commit": True})
    assert r.status_code == 200 and r.json()["proposal"]["status"] == "pending"
    body = client.get("/v1/guidance", params={"user_id": rig.user}).json()
    assert len(body["pending"]) == 1


@pytest.mark.parametrize("payload,code", [
    ({"user_id": "unknown", "bot_id": "x", "commit": True}, 403),
    ({"user_id": "bot:snark", "bot_id": "x", "commit": True}, 403),
    ({"user_id": "u", "bot_id": "x", "commit": True, "extra": 1}, 422),
    ({"user_id": "u", "bot_id": "x", "max_evidence": 0}, 422),
])
def test_run_route_refuses_bots_sentinels_and_bad_bodies(client, payload, code):
    assert client.post("/v1/guidance/runs", json=payload).status_code == code


def test_run_route_refuses_a_protected_bot(rig, client):  # noqa: F811
    r = client.post("/v1/guidance/runs",
                    json={"user_id": rig.user, "bot_id": rig.prot, "commit": True})
    assert r.status_code == 409
