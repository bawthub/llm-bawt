"""Management API integration with disposable SQLite and stubbed catalog only."""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from llm_bawt.bot_access import service as service_codec
from llm_bawt.bot_access.service import SERVICE_HEADER, ServicePrincipal
from llm_bawt.service.prompt_schedule_identity import CREATE_PURPOSE, create_invocation
from llm_bawt.service.prompt_schedule_management import PromptManagement
from llm_bawt.service.routes import prompt_schedules as routes
from llm_bawt.service.scheduler import ScheduledJob, create_scheduler_tables


@pytest.fixture
def client(monkeypatch):
    signer = Fernet(Fernet.generate_key())
    monkeypatch.setattr(service_codec, "_get_fernet", lambda: signer)
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
    item = create(client, payload(enabled=False, task_id="TASK-2019"))
    assert item["task_id"] == "TASK-2019"
    url = f"/v1/prompt-schedules/{item['id']}"
    key = {"idempotency_key": str(uuid4())}
    response = client.post(url+"/run-now?user=test", json=key)
    assert response.status_code == 202, response.text
    run = response.json()
    assert run["snapshot"]["task_id"] == "TASK-2019"
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


def proof(body, bot="al", user="test", *, purpose=CREATE_PURPOSE):
    return {SERVICE_HEADER: ServicePrincipal(bot, user).mint(purpose, create_invocation("test", body))}


def signed_create(client, body, bot="al"):
    response = client.post("/v1/prompt-schedules?user=test", json=body, headers=proof(body, bot))
    assert response.status_code == 201, response.text
    return response.json()


def test_signed_creator_ignores_name_and_is_immutable(client):
    # No wakeup name or special create key is required: caller proof alone wins.
    body = payload() | {"name": " Ordinary name ", "bot_id": " TEST "}
    real = signed_create(client, body)
    assert real["creator_bot_id"] == "al" and real["name"] == "Ordinary name"
    url = f"/v1/prompt-schedules/{real['id']}?user=test"
    renamed = client.patch(url, json={"expected_revision": 1, "name": "wakeup:other:x:y"})
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["creator_bot_id"] == "al"
    assert client.patch(url, json={"expected_revision": 2, "creator_bot_id": "other"}).status_code == 422
    assert client.post("/v1/prompt-schedules?user=test",
                       json=payload(creator_bot_id="al")).status_code == 422
    assert client.get(url).json()["creator_bot_id"] == "al"


def test_forged_public_key_name_and_plain_identity_header_cannot_claim_creator(client):
    from llm_bawt.service.prompt_schedule_management import wakeup_create_key, upgrade_prompt_schedules

    body = payload() | {"name": "wakeup:al:public-nonce:x",
                        "idempotency_key": wakeup_create_key("al", "public-nonce")}
    response = client.post("/v1/prompt-schedules?user=test", json=body, headers={"X-Agent-Bot-Id": "al"})
    assert response.status_code == 201
    assert response.json()["creator_bot_id"] is None
    engine = routes.management().engine
    with engine.begin() as conn:
        upgrade_prompt_schedules(conn, engine.dialect.name)
        upgrade_prompt_schedules(conn, engine.dialect.name)
    assert client.get(f"/v1/prompt-schedules/{response.json()['id']}?user=test").json()["creator_bot_id"] is None


@pytest.mark.parametrize("case", ["blank", "duplicate", "malformed", "tampered", "owner", "no-user", "tool", "expired"])
def test_supplied_invalid_creator_proof_is_never_anonymous_fallback(client, case):
    body = payload()
    headers = proof(body)
    if case == "blank":
        headers = {SERVICE_HEADER: ""}
    elif case == "duplicate":
        headers = [(SERVICE_HEADER, headers[SERVICE_HEADER]), (SERVICE_HEADER, headers[SERVICE_HEADER])]
    elif case == "malformed":
        headers = {SERVICE_HEADER: "not-a-capability"}
    elif case == "tampered":
        body["prompt"] += "tampered"
    elif case == "owner":
        headers = proof(body, user="other")
    elif case == "no-user":
        headers = proof(body, user=None)
    elif case == "tool":
        headers = proof(body, purpose="schedule_wakeup")
    elif case == "expired":
        import time
        token = headers[SERVICE_HEADER].encode()
        signer = service_codec._get_fernet()
        headers = {SERVICE_HEADER: signer.encrypt_at_time(signer.decrypt(token), int(time.time()) - 121).decode()}
    response = client.post("/v1/prompt-schedules?user=test", json=body, headers=headers)
    assert response.status_code == 403, response.text
    assert client.get("/v1/prompt-schedules?user=test").json()["total_count"] == 0


def test_creator_proof_binds_raw_json_and_normalized_owner(client):
    body = payload() | {"name": " Schedule "}
    headers = proof(body)
    # Model-normalization would erase this change; raw JSON verification won't.
    normalized = body | {"name": "Schedule"}
    assert client.post("/v1/prompt-schedules?user=test", json=normalized, headers=headers).status_code == 403
    response = client.post("/v1/prompt-schedules", params={"user": " TEST "}, json=body, headers=headers)
    assert response.status_code == 201, response.text
    assert response.json()["creator_bot_id"] == "al"


@pytest.mark.parametrize("initial_creator", [None, "al", "byte"])
def test_create_replay_cannot_adopt_or_change_creator(client, initial_creator):
    body = payload()
    first = signed_create(client, body, initial_creator) if initial_creator else create(client, body)
    for replay_creator in [None, "al", "byte"]:
        response = client.post("/v1/prompt-schedules?user=test", json=body,
                               headers=proof(body, replay_creator) if replay_creator else {})
        assert response.status_code == (201 if initial_creator == replay_creator else 409)
    stored = client.get(f"/v1/prompt-schedules/{first['id']}?user=test").json()
    assert stored["creator_bot_id"] == initial_creator
    assert (stored["id"], stored["revision"], stored["next_run_at"]) == (
        first["id"], first["revision"], first["next_run_at"])


def test_wakeup_sql_filter_keeps_renamed_real_rows_but_not_handmade_targets(client):
    real = signed_create(client, payload(), "al")
    targeted = signed_create(client, payload(), "byte")
    unrelated = signed_create(client, payload() | {"bot_id": "other"}, "byte")
    handmade = create(client)
    forged = create(client, payload() | {"name": "wakeup:al:fake:x"})
    url = f"/v1/prompt-schedules/{real['id']}?user=test"
    assert client.patch(url, json={"expected_revision": 1, "name": "Renamed real wakeup"}).status_code == 200
    query = "/v1/prompt-schedules?user=test&wakeup_bot_id=al"
    result = client.get(query).json()
    assert [s["id"] for s in result["schedules"]] == [real["id"]]
    targeted_list = client.get("/v1/prompt-schedules?user=test&wakeup_bot_id=test").json()
    assert {s["id"] for s in targeted_list["schedules"]} == {real["id"], targeted["id"]}
    assert targeted_list["total_count"] == 2
    assert all(s["id"] not in {handmade["id"], forged["id"], unrelated["id"]}
               for s in targeted_list["schedules"])
    assert client.get("/v1/prompt-schedules?user=other&wakeup_bot_id=al").json()["total_count"] == 0
    page = client.get(query + "&limit=1&offset=1").json()
    assert page["total_count"] == 1 and page["schedules"] == []


@pytest.mark.asyncio
async def test_real_tool_http_transport_sets_creator_and_lists_renames(client, monkeypatch):
    """The real tool, HTTP sender, verifier and SQL list work together, offline."""
    import json
    from types import SimpleNamespace

    import httpx
    from llm_bawt.mcp_server import schedule_tools

    original_client = httpx.AsyncClient

    def transport(request):
        response = client.request(
            request.method, f"{request.url.path}?{request.url.query.decode()}",
            content=request.content, headers=request.headers)
        return httpx.Response(response.status_code, content=response.content, headers=response.headers)

    monkeypatch.setattr(httpx, "AsyncClient",
                        lambda **kwargs: original_client(transport=httpx.MockTransport(transport), **kwargs))
    monkeypatch.setattr(schedule_tools, "_caller", lambda args: SimpleNamespace(bot_id="al", user_id="test"))
    monkeypatch.setattr(schedule_tools, "_target_refusal", lambda *args: None)
    monkeypatch.setattr(schedule_tools, "current_turn_task", lambda: None)
    created = json.loads(await schedule_tools.schedule_wakeup(message="Offline callback", in_minutes=60))
    assert "error" not in created, created
    identifier = created["id"]
    url = f"/v1/prompt-schedules/{identifier}?user=test"
    assert client.get(url).json()["creator_bot_id"] == "al"
    assert client.patch(url, json={"expected_revision": 1, "name": "No prefix anymore"}).status_code == 200
    handmade = create(client, payload() | {"name": "wakeup:al:forged:handmade", "bot_id": "al"})
    listed = json.loads(await schedule_tools.schedule_wakeup(action="list"))
    assert [s["id"] for s in listed["wakeups"]] == [identifier]
    assert handmade["id"] not in [s["id"] for s in listed["wakeups"]]
