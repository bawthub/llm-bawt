"""Management API integration with disposable SQLite and stubbed catalog only."""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from llm_bawt.service.prompt_schedule_management import PromptManagement
from llm_bawt.service.routes import prompt_schedules as routes
from llm_bawt.service.scheduler import ScheduledJob, create_scheduler_tables


@pytest.fixture
def client(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    create_scheduler_tables(engine)
    monkeypatch.setattr(routes, "management", lambda: PromptManagement(engine))
    monkeypatch.setattr(routes, "validate_target", lambda *args: None)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        yield client
    engine.dispose()


def payload(**kw):
    return dict(name="Schedule", description="A description", prompt="Test prompt", bot_id="test",
                timing={"kind": "interval", "interval_seconds": 60,
                        "anchor_at": (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()},
                idempotency_key=str(uuid4()), **kw)


def create(client, body=None):
    response = client.post("/v1/prompt-schedules?user=test", json=body or payload())
    assert response.status_code == 201, response.text
    return response.json()


def test_round_trip_revision_pause_resume_cancel(client):
    item = create(client)
    url = f"/v1/prompt-schedules/{item['id']}"
    assert client.get("/v1/prompt-schedules?user=test").json()["total_count"] == 1
    response = client.patch(url+"?user=test", json={"expected_revision": 1, "name": "Updated"})
    assert response.status_code == 200, response.text
    assert response.json()["revision"] == 2
    assert client.patch(url+"?user=test", json={"expected_revision": 1, "name": "Stale"}).status_code == 409
    assert item["next_run_at"] is not None
    paused = client.post(url+"/pause?user=test").json()
    assert paused["lifecycle"] == "paused" and paused["next_run_at"] is None
    assert client.get(url+"?user=test").json()["next_run_at"] is None
    with Session(routes.management().engine) as session:
        assert session.get(ScheduledJob, item["id"]).next_run_at is None
    resumed = client.post(url+"/resume?user=test").json()
    assert resumed["enabled"] is True and resumed["next_run_at"] is not None
    cancelled = client.delete(url+"?user=test").json()
    assert cancelled["lifecycle"] == "cancelled" and cancelled["next_run_at"] is None
    assert client.get(url+"?user=test").json()["next_run_at"] is None
    with Session(routes.management().engine) as session:
        assert session.get(ScheduledJob, item["id"]).next_run_at is None
    assert client.post(url+"/resume?user=test").status_code == 409


def test_disabled_create_and_update_clear_due_pointer(client):
    paused = create(client, payload(enabled=False))
    assert paused["lifecycle"] == "paused" and paused["next_run_at"] is None
    item = create(client)
    url = f"/v1/prompt-schedules/{item['id']}?user=test"
    stopped = client.patch(url, json={"expected_revision": 1, "enabled": False}).json()
    assert stopped["lifecycle"] == "paused" and stopped["next_run_at"] is None
    assert client.get("/v1/prompt-schedules?user=test&lifecycle=paused").json()["total_count"] == 2
    resumed = client.patch(url, json={"expected_revision": 2, "enabled": True}).json()
    assert resumed["lifecycle"] == "active" and resumed["next_run_at"] is not None

def test_create_idempotency_and_payload_conflict(client):
    body = payload()
    first = create(client, body)
    assert create(client, body)["id"] == first["id"]
    body["prompt"] = "Different"
    assert client.post("/v1/prompt-schedules?user=test", json=body).status_code == 409


def test_manual_idempotency_cadence_and_busy_edit(client):
    item = create(client, payload(enabled=False))
    url = f"/v1/prompt-schedules/{item['id']}"
    key = {"idempotency_key": str(uuid4())}
    response = client.post(url+"/run-now?user=test", json=key)
    assert response.status_code == 202, response.text
    run = response.json()
    assert client.post(url+"/run-now?user=test", json=key).json()["id"] == run["id"]
    assert client.get(url+"?user=test").json()["next_run_at"] == item["next_run_at"]
    assert client.post(url+"/run-now?user=test", json={"idempotency_key": str(uuid4())}).status_code == 409
    assert client.patch(url+"?user=test", json={"expected_revision": 1, "prompt": "new"}).status_code == 409
    assert client.get(url+"/runs?user=test").json()["runs"][0]["state"] == "pending"
    assert client.post(url+f"/runs/{run['id']}/cancel?user=test").status_code == 202


def test_wrong_owner_all_read_and_mutation_surfaces(client):
    item = create(client)
    url = f"/v1/prompt-schedules/{item['id']}"
    assert client.get("/v1/prompt-schedules?user=other").json()["schedules"] == []
    for method, path, data in [("get", "", None), ("get", "/runs", None),
        ("patch", "", {"expected_revision": 1, "name": "steal"}), ("delete", "", None),
        ("post", "/pause", None), ("post", "/resume", None),
        ("post", "/run-now", {"idempotency_key": str(uuid4())}),
        ("post", "/runs/fake/cancel", None)]:
        response = client.request(method, url+path+"?user=other", json=data)
        assert response.status_code == 404, response.text


def test_preview_and_timezone_resolution(client):
    response = client.post("/v1/prompt-schedules/preview?user=test", json={
        "timing": {"kind": "cron", "cron_expression": "30 1 * * *", "timezone": "America/New_York"},
        "after": "2026-11-01T04:00:00Z"})
    assert response.status_code == 200
    assert response.json()["occurrences"][0]["utc"] == "2026-11-01T05:30:00+00:00"
    response = client.post("/v1/prompt-schedules/resolve-local?user=test", json={"local": "2026-11-01T01:30", "timezone": "America/New_York"})
    assert response.status_code == 422
    assert response.json()["detail"]["offset_choices"] == [-300, -240]
    response = client.post("/v1/prompt-schedules/resolve-local?user=test", json={"local": "2026-11-01T01:30", "timezone": "America/New_York", "offset_minutes": -300})
    assert response.json()["utc"] == "2026-11-01T06:30:00+00:00"


def test_merged_patch_validation_and_unknown_fields(client):
    item = create(client)
    url = f"/v1/prompt-schedules/{item['id']}?user=test"
    for data in [{"prompt": ""}, {"timing": {"kind": "cron", "cron_expression": "0 0 31 2 *"}}, {"owner_user_id": "other"}, {"enabled": None}]:
        response = client.patch(url, json={"expected_revision": 1, **data})
        assert response.status_code == 422, response.text
    assert client.get(url).json()["revision"] == 1


def test_list_filter_and_last_run(client):
    item = create(client)
    client.post(f"/v1/prompt-schedules/{item['id']}/run-now?user=test", json={"idempotency_key": str(uuid4())})
    assert client.get("/v1/prompt-schedules?user=test&search=nope").json()["total_count"] == 0
    listed = client.get("/v1/prompt-schedules?user=test").json()["schedules"][0]
    assert listed["last_run"]["state"] == "pending"


def test_wakeup_creator_is_proven_once_at_create_and_backfilled(client):
    """TASK-1790: only a name that matches its derived id proves a bot creator."""
    from sqlalchemy import text

    from llm_bawt.service.prompt_schedule_management import upgrade_prompt_schedules, wakeup_create_key

    nonce = "0123456789abcdef"
    name = f"wakeup:snark:{nonce}:x"
    real = create(client, payload() | {"name": name, "idempotency_key": str(wakeup_create_key("snark", nonce))})
    forged = create(client, payload() | {"name": name})
    plain = create(client)
    assert (real["creator_bot_id"], forged["creator_bot_id"], plain["creator_bot_id"]) == ("snark", None, None)
    url = f"/v1/prompt-schedules/{real['id']}?user=test"
    renamed = client.patch(url, json={"expected_revision": real["revision"], "name": "wakeup:al:x:y"})
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["creator_bot_id"] == "snark"

    # Rows from before the column existed: the backfill re-proves from the name.
    engine = routes.management().engine
    with engine.begin() as conn:
        conn.execute(text("UPDATE prompt_schedules SET creator_bot_id=NULL"))
        conn.execute(text("UPDATE prompt_schedules SET name=:name WHERE job_id=:id"),
                     {"name": f"wakeup:snark:{nonce}:x", "id": real["id"]})
        upgrade_prompt_schedules(conn, engine.dialect.name)
        upgrade_prompt_schedules(conn, engine.dialect.name)  # idempotent
        creators = dict(conn.execute(text("SELECT job_id, creator_bot_id FROM prompt_schedules")).all())
    assert creators == {real["id"]: "snark", forged["id"]: None, plain["id"]: None}
