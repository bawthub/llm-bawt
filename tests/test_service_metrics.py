"""Prometheus scrape contract: stock HTTP metrics without caller-chosen labels."""

import asyncio

import pytest
from fastapi import APIRouter, FastAPI
from httpx import ASGITransport, AsyncClient
from prometheus_client.parser import text_string_to_metric_families

from llm_bawt.api_auth import wiring
from llm_bawt.api_auth.mode import ModeResolver
from llm_bawt.service.metrics import PROMETHEUS_METRICS_ENV, install
from llm_bawt.service.request_guard import GuardPolicy, RequestGuardMiddleware


@pytest.fixture(autouse=True)
def prometheus_on(monkeypatch):
    monkeypatch.setenv(PROMETHEUS_METRICS_ENV, "1")


@pytest.mark.asyncio
async def test_http_metrics_keep_only_bounded_labels():
    app = FastAPI()
    outer, router = APIRouter(), APIRouter()
    entered, release = asyncio.Event(), asyncio.Event()

    @router.get("/items/{item_id}")
    async def get_item(item_id: str):
        if item_id == "fixture-held":
            entered.set()
            await release.wait()
        return {"id": item_id}

    outer.include_router(router, prefix="/nested")
    app.include_router(outer, prefix="/v1")
    install(app)
    template = "/v1/nested/items/{item_id}"

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        active = asyncio.create_task(client.get("/v1/nested/items/fixture-held?token=fixture-query"))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            response = await client.get("/metrics")
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/plain; version=")
            current = [sample for metric in text_string_to_metric_families(response.text)
                       for sample in metric.samples if sample.name == "http_requests_inprogress"]
            assert len(current) == 1
            assert current[0].labels == {"handler": template, "method": "GET"}
            assert current[0].value == 1
        finally:
            release.set()
            assert (await active).status_code == 200

        for ident in range(5):
            assert (await client.get(f"/v1/nested/items/id-{ident}?name=fixture-query")).status_code == 200
            assert (await client.get(f"/v1/nested/items/redirect-{ident}/")).status_code == 307
            assert (await client.get(f"/fixture-unmatched-{ident}")).status_code == 404
            assert (await client.request(f"FIXTURE-METHOD-{ident}", f"/v1/nested/items/{ident}")).status_code == 405
        body = (await client.get("/metrics")).text

    families = list(text_string_to_metric_families(body))
    samples = [sample for family in families for sample in family.samples]
    assert all(value not in body for value in (
        "fixture-held", "fixture-query", "id-0", "redirect-0", "fixture-unmatched", "FIXTURE-METHOD", 'handler="/metrics',
    ))
    handlers = {sample.labels["handler"] for sample in samples if "handler" in sample.labels}
    assert handlers == {template}
    assert {sample.labels["method"] for sample in samples if "method" in sample.labels} == {"GET"}
    assert any(f.name == "http_request_duration_seconds" and f.type == "histogram" for f in families)
    for name, value in (("http_requests_total", 6), ("http_request_duration_seconds_count", 6),
                        ("http_requests_inprogress", 0)):
        assert any(sample.name == name and sample.labels.get("handler") == template
                   and sample.value == value for sample in samples), name
    assert {"process_cpu_seconds_total", "process_resident_memory_bytes",
            "python_gc_collections_total"} <= {s.name for s in samples}


@pytest.mark.asyncio
async def test_metrics_exemption_is_exact_and_keeps_host_guard(monkeypatch):
    from llm_bawt.service.api import app as production_app

    assert [r.path for r in production_app.routes if getattr(r, "path", None) == "/metrics"] == ["/metrics"]
    assert "/metrics" not in production_app.openapi()["paths"]

    reads = 0

    def settings():
        nonlocal reads
        reads += 1
        return {"api_auth_mode": "enforce"}

    monkeypatch.setattr(wiring, "RESOLVER", ModeResolver(settings, environ={}))
    app = FastAPI()
    install(app)
    app.add_middleware(RequestGuardMiddleware, policy=GuardPolicy.from_env({
        "LLM_BAWT_HOST_GUARD": "enforce", "LLM_BAWT_ALLOWED_HOSTS": "app.lan.example",
    }))
    wiring.install(app, prometheus_metrics=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://app.lan.example") as client:
        assert (await client.get("/metrics")).status_code == 200
        assert (await client.get("/metrics", headers={"Host": "else.example"})).status_code == 421
        assert reads == 0  # Scrapes don't need the rollout mode or its DB.
        assert (await client.get("/metrics/anything")).status_code == 401
        assert (await client.get("/v1/private")).status_code == 401
        assert reads == 1


@pytest.mark.asyncio
async def test_unset_switch_preserves_existing_json_and_auth(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv(PROMETHEUS_METRICS_ENV)
    app = FastAPI()
    snapshot = {"event_loop_lag_max_ms": 15, "event_loop_lag_p99_ms": 5,
                "event_loop_stalls_total": 0, "event_loop_window_seconds": 60}
    app.state.loop_watchdog = SimpleNamespace(snapshot=lambda: snapshot)
    assert not install(app)
    assert [r.path for r in app.routes if getattr(r, "path", None) == "/metrics"] == ["/metrics"]
    values = {"api_auth_mode": "log"}
    monkeypatch.setattr(wiring, "RESOLVER", ModeResolver(lambda: values, environ={}, ttl=0))
    wiring.install(app)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        legacy = await client.get("/metrics")
        assert legacy.status_code == 200
        assert legacy.headers["cache-control"] == "no-store"
        assert legacy.json() == snapshot
        values["api_auth_mode"] = "enforce"
        assert (await client.get("/metrics")).status_code == 401


@pytest.mark.parametrize("setting", ["PROMETHEUS_MULTIPROC_DIR", "prometheus_multiproc_dir"])
def test_multiprocess_mode_is_not_silently_accepted(monkeypatch, setting):
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    monkeypatch.delenv("prometheus_multiproc_dir", raising=False)
    monkeypatch.setenv(setting, "/not-used")
    with pytest.raises(RuntimeError, match=f"single-process serving; unset {setting}"):
        install(FastAPI())

    monkeypatch.delenv(PROMETHEUS_METRICS_ENV)
    assert not install(FastAPI())
