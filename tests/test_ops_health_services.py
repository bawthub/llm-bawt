"""TASK-1169 S1c: /v1/health/services[/{id}] over typed Compose identity.

The service set is what enabled catalog ops target; instances come from the
cached inventory with an opaque instance id; nothing reaches Docker or the DB
on a GET.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from llm_bawt.ops.health_collector import (
    MAX_CAPABILITIES,
    MAX_OPERATIONS,
    OpsHealthCollector,
    catalog_targets,
    instance_id,
    observe_container,
)
from llm_bawt.ops.health_http import OpsHealthServer
from llm_bawt.ops.health_state import CatalogSample, OpsHealthState
from llm_bawt.ops.seeds import SEEDS

PROJECT = "com.docker.compose.project"
SERVICE = "com.docker.compose.service"


def request(server, path, *, method="GET"):
    url = f"http://127.0.0.1:{server.address[1]}{path}"
    try:
        response = urlopen(Request(url, method=method), timeout=2)
    except HTTPError as exc:
        response = exc
    with response:
        return response.status, json.loads(response.read())


class Clock:
    def __init__(self):
        self.elapsed = 100.0
        self.wall = datetime(2026, 10, 28, tzinfo=UTC)

    def advance(self, seconds):
        self.elapsed += seconds
        self.wall += timedelta(seconds=seconds)

    def state(self):
        return OpsHealthState(clock=lambda: self.elapsed, utcnow=lambda: self.wall)


def container(name, *, cid=None, project="bawthub", service="tts", status="running", extra_labels=None):
    labels = {"private.label": "secret-label", **(extra_labels or {})}
    if project is not None:
        labels[PROJECT] = project
    if service is not None:
        labels[SERVICE] = service
    return {"Name": "/" + name, "Id": cid or ("c" * 63 + name[-1]),
            "State": {"Status": status, "StartedAt": "2026-10-28T00:00:00Z"},
            "Config": {"Labels": labels, "Env": ["SECRET=password"]}}


def op(slug, script, schema=None):
    return (slug, json.dumps(script), json.dumps(schema or {"type": "object", "properties": {}}))


def enum_schema(arg, values):
    return {"type": "object", "properties": {arg: {"type": "string", "enum": values}}}


CATALOG = [
    op("bawthub.restart-tts", {"action": "restart", "compose_project": "bawthub", "compose_service": "tts"}),
    op("bawthub.stop-tts", {"action": "stop", "compose_project": "bawthub", "compose_service": "tts"}),
    op("llm-bawt.restart-aux", {"action": "restart", "compose_project": "llm-bawt",
                                "compose_service_from_arg": "service"},
       enum_schema("service", ["crawl4ai", "playwright-mcp"])),
]


def docker_client(rows):
    api = SimpleNamespace(containers=Mock(return_value=[{"Id": str(i)} for i in range(len(rows))]),
                          inspect_container=Mock(side_effect=lambda i: rows[int(i)]))
    return SimpleNamespace(api=api, close=Mock())


@pytest.fixture
def world(monkeypatch):
    monkeypatch.setattr("llm_bawt.ops.health_collector.DockerExecutor.readiness", Mock(return_value=None))
    clock = Clock()
    state = clock.state()
    rows = [container("bawthub-tts-1"), container("loose", project=None, service=None)]
    client = docker_client(rows)
    probe = OpsHealthCollector(state, lambda: None, client_factory=lambda: client)
    probe.collect_docker()
    state.publish_catalog(CatalogSample(state.sample_time(), "ok", services=catalog_targets(CATALOG)))
    return clock, state, client, rows, probe


# --- collector projection -------------------------------------------------

def test_observation_types_compose_identity_and_hashes_the_id():
    row = container("bawthub-tts-1", cid="a" * 64)
    observed = observe_container(row)
    assert (observed.project, observed.service) == ("bawthub", "tts")
    assert observed.instance_id == hashlib.sha256(("a" * 64).encode()).hexdigest()[:16]
    assert observed.instance_api() == {"instanceId": observed.instance_id, "name": "bawthub-tts-1",
                                       "state": "running", "health": None,
                                       "startedAt": "2026-10-28T00:00:00+00:00"}
    # The /containers row is unchanged: no identity fields leak into the compat shape.
    assert observed.to_api() == {"name": "bawthub-tts-1", "state": "running", "health": None,
                                 "startedAt": "2026-10-28T00:00:00+00:00"}
    assert "a" * 64 not in json.dumps(observed.instance_api())


@pytest.mark.parametrize("labels", [
    {"project": None, "service": None},
    {"project": "bawthub", "service": None},
    {"project": "Bad Project", "service": "tts"},
    {"project": "bawthub", "service": "../tts"},
])
def test_missing_or_invalid_labels_leave_the_container_unlabeled(labels):
    observed = observe_container(container("x1", **labels))
    assert observed.project is None and observed.service is None
    assert observed.instance_id is not None


def test_non_dict_labels_are_unlabeled_and_bad_id_fails_closed():
    row = container("x1")
    row["Config"]["Labels"] = ["not", "a", "dict"]
    assert observe_container(row).project is None
    for bad in (None, "", "x" * 129, 7):
        with pytest.raises(ValueError):
            instance_id(bad)


# --- catalog targets -------------------------------------------------------

def test_catalog_targets_fixed_and_bounded_enum_selectors():
    targets = {t.service_id: t for t in catalog_targets(CATALOG)}
    assert list(targets) == ["bawthub.tts", "llm-bawt.crawl4ai", "llm-bawt.playwright-mcp"]
    assert [(c.operation, c.action, c.args) for c in targets["bawthub.tts"].capabilities] == [
        ("bawthub.restart-tts", "restart", ()), ("bawthub.stop-tts", "stop", ())]
    assert [c.to_api() for c in targets["llm-bawt.crawl4ai"].capabilities] == [
        {"operation": "llm-bawt.restart-aux", "action": "restart", "args": {"service": "crawl4ai"}}]


@pytest.mark.parametrize("row", [
    # Dynamic selector without a bounded enum: never "all services".
    op("x.any", {"action": "restart", "compose_project": "p", "compose_service_from_arg": "s"},
       {"type": "object", "properties": {"s": {"type": "string"}}}),
    op("x.empty", {"action": "restart", "compose_project": "p", "compose_service_from_arg": "s"},
       enum_schema("s", [])),
    op("x.huge", {"action": "restart", "compose_project": "p", "compose_service_from_arg": "s"},
       enum_schema("s", [f"s{i}" for i in range(65)])),
    op("x.nonstr", {"action": "restart", "compose_project": "p", "compose_service_from_arg": "s"},
       enum_schema("s", ["a", 1])),
    op("x.name", {"action": "restart", "container_name": "llm-bawt-app"}),
    op("x.release", {"action": "release_orchestrate", "github_repository": "o/r"}),
    op("x.badproject", {"action": "restart", "compose_project": "Bad", "compose_service": "tts"}),
    op("x.badaction", {"action": "Restart; rm", "compose_project": "p", "compose_service": "tts"}),
    op("x.both", {"action": "restart", "compose_project": "p", "compose_service": "a",
                  "compose_service_from_arg": "s"}, enum_schema("s", ["b"])),
    ("bad slug!", json.dumps({"action": "restart", "compose_project": "p", "compose_service": "a"}), "{}"),
    ("x.json", "{not json", "{}"),
    ("x.list", "[]", "{}"),
])
def test_unusable_ops_contribute_nothing_and_do_not_poison_others(row):
    assert [t.service_id for t in catalog_targets([row, *CATALOG])] == [
        "bawthub.tts", "llm-bawt.crawl4ai", "llm-bawt.playwright-mcp"]


def test_catalog_bounds_fail_closed():
    with pytest.raises(ValueError):
        catalog_targets([CATALOG[0]] * (MAX_OPERATIONS + 1))
    too_many = [op(f"x.{i}", {"action": "restart", "compose_project": "p", "compose_service_from_arg": "s"},
                   enum_schema("s", [f"s{j}" for j in range(64)])) for i in range(MAX_CAPABILITIES // 64 + 1)]
    with pytest.raises(ValueError):
        catalog_targets(too_many)


def test_shipped_seeds_produce_service_targets():
    rows = [(s["slug"], s["command_script"], s["args_schema_json"]) for s in SEEDS]
    ids = {t.service_id for t in catalog_targets(rows)}
    assert {"bawthub.tts", "bawthub.stt", "bawthub.backend", "llm-bawt.crawl4ai",
            "llm-bawt.claude-code-bridge"} <= ids
    assert all("." in i for i in ids)


def test_catalog_collection_is_read_only_enabled_only_and_sanitized():
    state = Clock().state()
    conn = Mock()
    conn.execute.return_value.all.return_value = CATALOG
    context = Mock(__enter__=Mock(return_value=conn), __exit__=Mock(return_value=False))
    engine = SimpleNamespace(connect=lambda: context, dialect=SimpleNamespace(name="postgresql"))
    probe = OpsHealthCollector(state, lambda: engine)
    probe.collect_catalog()
    statements = [str(call.args[0]) for call in conn.execute.call_args_list]
    assert statements[:3] == ["SET TRANSACTION READ ONLY", "SET LOCAL statement_timeout = '2000ms'",
                              "SET LOCAL lock_timeout = '2000ms'"]
    assert "WHERE enabled IS TRUE AND soft_deleted_at IS NULL" in statements[3]
    assert conn.execute.call_args_list[3].args[1] == {"limit": MAX_OPERATIONS + 1}
    assert state._catalog.status == "ok" and len(state._catalog.services) == 3
    conn.execute.side_effect = RuntimeError("postgres://private-password")
    probe.collect_catalog()
    assert state._catalog.status == "down"
    assert "private" not in json.dumps(state._catalog.reason)
    OpsHealthCollector(state, lambda: None).collect_catalog()
    assert state._catalog.reason == "operation catalog unavailable"


def test_storage_tick_runs_probe_then_catalog():
    probe = OpsHealthCollector(Clock().state(), lambda: None)
    calls = []
    probe.collect_database = lambda: calls.append("database")
    probe.collect_catalog = lambda: calls.append("catalog")
    probe.collect_storage()
    assert calls == ["database", "catalog"]


# --- response contract ----------------------------------------------------

def test_fresh_services_list_instances_zero_and_unlabeled(world):
    _, state, _, rows, _ = world
    code, body = state.services()
    assert code == 200 and body["fresh"] is True and body["reason"] is None
    assert body["schemaVersion"] == 1 and body["unlabeledCount"] == 1
    tts, crawl, _ = body["services"]
    assert tts["id"] == "bawthub.tts" and tts["project"] == "bawthub" and tts["service"] == "tts"
    assert [i["name"] for i in tts["instances"]] == ["bawthub-tts-1"]
    assert tts["instances"][0]["instanceId"] == instance_id(rows[0]["Id"])
    assert crawl["instances"] == []  # catalogued, observed, none running
    dumped = json.dumps(body)
    assert "loose" not in dumped and "secret" not in dumped and rows[0]["Id"] not in dumped


def test_replacement_changes_instance_id(world):
    _, state, _, rows, probe = world
    before = state.services("bawthub.tts")[1]["service"]["instances"][0]["instanceId"]
    rows[0] = container("bawthub-tts-1", cid="d" * 64)
    probe.collect_docker()
    after = state.services("bawthub.tts")[1]["service"]["instances"][0]["instanceId"]
    assert before != after


def test_many_instances_are_all_listed(world):
    _, state, client, rows, probe = world
    rows.append(container("bawthub-tts-2"))
    client.api.containers.return_value = [{"Id": str(i)} for i in range(len(rows))]
    probe.collect_docker()
    assert len(state.services("bawthub.tts")[1]["service"]["instances"]) == 2


def test_stale_inventory_nulls_instances_but_keeps_catalog(world):
    clock, state, _, _, _ = world
    clock.advance(20)
    state.publish_catalog(CatalogSample(state.sample_time(), "ok", services=catalog_targets(CATALOG)))
    clock.advance(11)  # inventory now 31s old, catalog 11s
    code, body = state.services()
    assert code == 503 and body["fresh"] is False and "stale" in body["reason"]
    assert body["unlabeledCount"] is None
    assert [s["instances"] for s in body["services"]] == [None, None, None]
    assert body["services"][0]["capabilities"]
    assert body["observedAt"] == datetime(2026, 10, 28, tzinfo=UTC).isoformat()


def test_down_inventory_nulls_instances(world):
    _, state, client, _, probe = world
    client.api.inspect_container.side_effect = RuntimeError("secret")
    probe.collect_docker()
    code, body = state.services("bawthub.tts")
    assert code == 503 and body["service"]["instances"] is None
    assert body["reason"] == "container observation unavailable"


def test_unavailable_catalog_nulls_services(world):
    _, state, _, _, _ = world
    state.publish_catalog(CatalogSample(state.sample_time(), "down", "operation catalog unavailable"))
    code, body = state.services()
    assert code == 503 and body["services"] is None and body["reason"] == "operation catalog unavailable"
    assert body["unlabeledCount"] == 1  # inventory itself is still fresh
    code, body = state.services("bawthub.tts")
    assert code == 503 and body["service"] is None


def test_initial_state_is_never_fresh():
    state = Clock().state()
    code, body = state.services()
    assert code == 503 and body["services"] is None and body["unlabeledCount"] is None


def test_shutdown_is_never_fresh(world):
    _, state, _, _, _ = world
    state.stop()
    code, body = state.services()
    assert code == 503 and body["reason"] == "ops reconciler is shutting down"


def test_single_service_and_unknown_id(world):
    _, state, _, _, _ = world
    code, body = state.services("llm-bawt.crawl4ai")
    assert code == 200 and body["service"]["capabilities"][0]["args"] == {"service": "crawl4ai"}
    assert set(body) == {"schemaVersion", "observedAt", "fresh", "reason", "unlabeledCount", "service"}
    assert state.services("bawthub.frontend") == (404, {"error": "not found"})


def test_maximum_catalog_and_inventory_fit_consumers_body_limit(monkeypatch):
    monkeypatch.setattr("llm_bawt.ops.health_collector.DockerExecutor.readiness", Mock(return_value=None))
    state = Clock().state()
    services = [f"s{i:03d}" for i in range(64)]
    catalog = [op(f"x.op{i}", {"action": "restart", "compose_project": "p", "compose_service_from_arg": "s"},
                  enum_schema("s", services)) for i in range(MAX_CAPABILITIES // 64)]
    rows = [container(f"{i:04d}".zfill(255), cid=f"{i:064d}", project="p", service=services[i % 64])
            for i in range(1000)]
    OpsHealthCollector(state, lambda: None, client_factory=lambda: docker_client(rows)).collect_docker()
    state.publish_catalog(CatalogSample(state.sample_time(), "ok", services=catalog_targets(catalog)))
    code, body = state.services()
    assert code == 200 and sum(len(s["instances"]) for s in body["services"]) == 1000
    assert sum(len(s["capabilities"]) for s in body["services"]) == MAX_CAPABILITIES
    assert len(json.dumps(body, separators=(",", ":")).encode()) < 512 * 1024


# --- HTTP surface ----------------------------------------------------------

def test_http_routes_are_exact_cache_only_and_get_only(world):
    _, state, client, _, _ = world
    server = OpsHealthServer(state, ("127.0.0.1", 0))
    server.start()
    try:
        for _ in range(3):
            assert request(server, "/v1/health/services")[0] == 200
            code, body = request(server, "/v1/health/services/bawthub.tts")
            assert code == 200 and body["service"]["id"] == "bawthub.tts"
        assert client.api.containers.call_count == 1
        assert request(server, "/v1/health/services/bawthub.nope") == (404, {"error": "not found"})
        for path in ("/v1/health/services/", "/v1/health/services?id=bawthub.tts",
                     "/v1/health/services/bawthub.tts?x=1", "/v1/health/services/bawthub.tts/",
                     "/v1/health/services/bawthub", "/v1/health/services/%2e%2e",
                     "/v1/health/services/Bad.tts"):
            assert request(server, path) == (404, {"error": "not found"}), path
        for method in ("POST", "PUT", "PATCH", "DELETE"):
            assert request(server, "/v1/health/services/bawthub.tts", method=method)[0] == 405
    finally:
        server.close()
