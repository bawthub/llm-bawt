"""GET /v1/admin/auth-observations (TASK-2380)."""
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_bawt.api_auth import wiring
from llm_bawt.api_auth.middleware import Observations
from llm_bawt.bawthub_auth.service_token import TokenCheck
from llm_bawt.service.routes import auth_observations, service_auth


def test_requires_service_bearer_and_returns_all_rows(monkeypatch):
    app = FastAPI()
    app.include_router(auth_observations.router)
    client = TestClient(app)
    obs = Observations()
    for i in range(3):
        obs.record(("anon", f"10.0.0.{i}", "curl", "GET", "/v1/bots"))
    monkeypatch.setattr(wiring, "OBSERVATIONS", obs)
    monkeypatch.setattr(wiring, "RESOLVER", SimpleNamespace(mode=lambda: "log"))

    monkeypatch.setattr(service_auth, "service_token_source",
                        lambda: SimpleNamespace(check=lambda header: TokenCheck.INVALID))
    assert client.get("/v1/admin/auth-observations").status_code == 401

    monkeypatch.setattr(service_auth, "service_token_source",
                        lambda: SimpleNamespace(check=lambda header: TokenCheck.OK))
    res = client.get("/v1/admin/auth-observations", headers={"Authorization": "Bearer x"})
    assert res.status_code == 200
    assert res.headers["cache-control"] == "no-store"
    body = res.json()
    assert body["mode"] == "log"
    assert sorted(r["key"][1] for r in body["rows"]) == ["10.0.0.0", "10.0.0.1", "10.0.0.2"]
    assert all(r["count"] == 1 for r in body["rows"])
