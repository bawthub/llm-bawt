"""TASK-908 Keep notes management contracts against a mocked Google API."""
import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.integrations.connections import IntegrationError
from llm_bawt.integrations.keep_notes import KeepNotes, build_body, normalize
from llm_bawt.service.routes import integrations as routes
from tests.test_integration_connections import manager as integration_manager
from tests.test_keep_workspace import configure, google, key  # noqa: F401 -- fixtures

manager = pytest.fixture(name="manager")(integration_manager.__wrapped__)  # noqa: F401

LIST_NOTE = {
    "name": "notes/abc", "title": "Groceries", "createTime": "2026-09-01T00:00:00Z", "updateTime": "2026-09-02T00:00:00Z",
    "body": {"list": {"listItems": [
        {"text": {"text": "milk"}, "checked": False},
        {"text": {"text": "baking"}, "checked": False, "childListItems": [{"text": {"text": "flour"}, "checked": True}]},
    ]}},
    "permissions": [
        {"name": "notes/abc/permissions/o1", "role": "OWNER", "email": "nick@example.com"},
        {"name": "notes/abc/permissions/w1", "role": "WRITER", "user": {"email": "pal@example.com"}},
    ],
    "attachments": [{"name": "notes/abc/attachments/att1", "mimeType": ["image/png", "bogus mime"]}],
}


@pytest.fixture
def api(manager, key):
    configure(manager, key)
    google(manager, key)
    manager.verify_workspace("google-keep")
    state = {"notes": {"abc": json.loads(json.dumps(LIST_NOTE))}, "calls": [], "fail": {}}

    def handle(request: httpx.Request):
        if request.url.path == "/token":
            return httpx.Response(200, json={"access_token": "tok", "expires_in": 3600})
        assert request.headers["authorization"] == "Bearer tok"
        path, method = request.url.path, request.method
        state["calls"].append((method, path, request.content.decode() if request.content else ""))
        if (method, path) in state.get("raise", {}):
            raise httpx.ReadTimeout("response lost")
        if (method, path) in state["fail"]:
            return httpx.Response(state["fail"][(method, path)], json={"error": {"message": "NO LEAK"}})
        if path == "/v1/notes" and method == "GET":
            f = request.url.params.get("filter")
            assert f in (None, "trashed OR -trashed")
            notes = list(state["notes"].values())
            return httpx.Response(200, json={"notes": notes if f else [n for n in notes if not n.get("trashed")]})
        if path == "/v1/notes" and method == "POST":
            body = json.loads(request.content)
            nid = f"new{len(state['notes'])}"
            state["notes"][nid] = {"name": f"notes/{nid}", **body,
                                   "permissions": [{"name": f"notes/{nid}/permissions/o", "role": "OWNER", "email": "nick@example.com"}]}
            return httpx.Response(200, json=state["notes"][nid])
        nid = path.split("/")[3]
        if nid not in state["notes"]:
            return httpx.Response(404, json={})
        if path.endswith("permissions:batchCreate"):
            for r in json.loads(request.content)["requests"]:
                assert r["permission"]["role"] == "WRITER" and r["parent"] == f"notes/{nid}"
                state["notes"][nid]["permissions"].append({"name": f"notes/{nid}/permissions/{r['permission']['email'].split('@')[0]}", "role": "WRITER", "email": r["permission"]["email"]})
            return httpx.Response(200, json={})
        if path.endswith("permissions:batchDelete"):
            names = set(json.loads(request.content)["names"])
            state["notes"][nid]["permissions"] = [p for p in state["notes"][nid]["permissions"] if p["name"] not in names]
            return httpx.Response(200, json={})
        if "/attachments/" in path:
            assert request.url.params["alt"] == "media"
            return httpx.Response(200, content=b"\x89PNG", headers={"content-type": request.url.params["mimeType"]})
        if method == "DELETE":
            del state["notes"][nid]
            return httpx.Response(200, json={})
        return httpx.Response(200, json=state["notes"][nid])

    manager.client_factory = lambda **kw: httpx.Client(transport=httpx.MockTransport(handle), **kw)
    return KeepNotes(manager), state


def test_normalize_full_note_shape():
    note = normalize(LIST_NOTE)
    assert note["type"] == "list" and note["id"] == "abc"
    assert note["items"][1]["children"][0] == {"text": "flour", "checked": True, "children": []}
    assert (note["item_count"], note["unchecked_count"]) == (3, 2)
    assert note["attachments"] == [{"id": "att1", "mime_types": ["image/png"]}]
    assert note["collaborators"][1]["email"] == "pal@example.com"


def test_list_active_and_trashed(api):
    keep, state = api
    state["notes"]["t"] = {"name": "notes/t", "title": "old", "trashed": True, "body": {"text": {"text": "x"}}}
    assert [n["id"] for n in keep.list()["notes"]] == ["abc"]
    assert [n["id"] for n in keep.list(trashed=True)["notes"]] == ["t"]


def test_create_checklist_with_nesting_and_collaborators(api):
    keep, state = api
    out = keep.create("Hardware", items=[{"text": "screws", "children": [{"text": "m3", "checked": True}]}],
                      collaborators=["Pal@Example.com"])
    assert out["note"]["items"][0]["children"][0]["checked"] is True
    assert "pal@example.com" in [c["email"] for c in out["note"]["collaborators"]]


@pytest.mark.parametrize("kwargs,code", [
    ({"title": "x", "text": None, "items": None}, "keep_invalid_content"),
    ({"title": "", "text": " ", "items": None}, "keep_empty_note"),
    ({"title": "x", "text": None, "items": [{"text": "a", "children": [{"text": "b", "children": [{"text": "c"}]}]}]}, "keep_nesting_too_deep"),
    ({"title": "x", "text": None, "items": [{"text": " "}]}, "keep_empty_item"),
    ({"title": "x" * 1000, "text": "y", "items": None}, "keep_title_too_long"),
])
def test_body_validation(kwargs, code):
    with pytest.raises(IntegrationError, match=code):
        build_body(**kwargs)


def test_share_unshare_and_validation(api):
    keep, _ = api
    assert "new@example.com" in [c["email"] for c in keep.share("abc", ["new@example.com"])["note"]["collaborators"]]
    left = keep.unshare("abc", ["notes/abc/permissions/w1"])["note"]["collaborators"]
    assert "pal@example.com" not in [c["email"] for c in left]
    with pytest.raises(IntegrationError, match="invalid_permission"):
        keep.unshare("abc", ["notes/other/permissions/w1"])
    with pytest.raises(IntegrationError, match="invalid_collaborator_email"):
        keep.share("abc", ["not-an-email"])


def test_delete_and_not_found(api):
    keep, state = api
    keep.delete("abc")
    assert "abc" not in state["notes"]
    with pytest.raises(IntegrationError, match="keep_note_not_found"):
        keep.get("abc")
    with pytest.raises(IntegrationError, match="invalid_keep_note"):
        keep.get("../etc")


def test_attachment_download(api):
    keep, _ = api
    assert keep.attachment("abc", "att1", "image/png") == (b"\x89PNG", "image/png")
    with pytest.raises(IntegrationError, match="invalid_mime_type"):
        keep.attachment("abc", "att1", "text/html; x")


def test_replace_copies_collaborators_then_deletes_original(api):
    keep, state = api
    out = keep.replace("abc", "Groceries", items=[{"text": "milk"}, {"text": "eggs"}])
    assert out["original_deleted"] is True and "abc" not in state["notes"]
    assert "pal@example.com" in [c["email"] for c in out["note"]["collaborators"]]
    methods = [(m, p) for m, p, _ in state["calls"]]
    assert methods.index(("POST", "/v1/notes")) < methods.index(("DELETE", "/v1/notes/abc"))


def test_replace_keeps_original_when_sharing_fails(api):
    keep, state = api
    state["fail"][("POST", "/v1/notes/new1/permissions:batchCreate")] = 500
    out = keep.replace("abc", "Groceries", text="hi")
    assert out == {"note": out["note"], "original_deleted": False, "warning": "collaborators_not_copied"}
    assert "abc" in state["notes"]


def test_routes_and_bounded_errors(api, manager, monkeypatch):
    keep, state = api
    monkeypatch.setattr(routes, "manager", lambda _id: manager)
    app = FastAPI(); app.include_router(routes.router)
    client = TestClient(app)
    assert client.get("/v1/integrations/google-keep/keep/notes").json()["total"] == 1
    created = client.post("/v1/integrations/google-keep/keep/notes", json={"title": "T", "items": [{"text": "a"}]})
    assert created.status_code == 200 and created.json()["note"]["type"] == "list"
    img = client.get("/v1/integrations/google-keep/keep/notes/abc/attachments/att1", params={"mime_type": "image/png"})
    assert img.content == b"\x89PNG" and img.headers["content-disposition"].startswith("inline")
    state["fail"][("GET", "/v1/notes/abc")] = 403
    denied = client.get("/v1/integrations/google-keep/keep/notes/abc")
    assert denied.status_code == 400 and "NO LEAK" not in denied.text
    assert client.get("/v1/integrations/other/keep/notes").status_code == 404


def test_replace_reports_unknown_delete_after_transport_failure(api):
    """TASK-2260: a timeout during DELETE keeps the new note and says the outcome is unknown."""
    keep, state = api
    state.setdefault("raise", {})[("DELETE", "/v1/notes/abc")] = True
    out = keep.replace("abc", "Groceries", text="hi")
    assert out["original_deleted"] is None
    assert out["warning"] == "original_delete_unknown"
    assert out["note"]["id"].startswith("new")


def test_create_keeps_note_when_sharing_times_out(api):
    """TASK-2260: a timeout while sharing returns the created note, not a generic error."""
    keep, state = api
    state.setdefault("raise", {})[("POST", "/v1/notes/new1/permissions:batchCreate")] = True
    out = keep.create("Hi", text="x", collaborators=["pal@example.com"])
    assert out["warning"] == "collaborators_unknown"
    assert out["note"]["id"] == "new1"
