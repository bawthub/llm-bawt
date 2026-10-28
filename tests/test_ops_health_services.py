"""TASK-1169 S1c/S2a: /v1/health/services[/{id}] over typed Compose identity.

The service set is what enabled catalog ops target together with every
observed Compose service (S2a); instances come from the
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
    MAX_CATALOG_BYTES,
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
                                       "startedAt": "2026-10-28T00:00:00+00:00", "restartCount": None}
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


@pytest.mark.parametrize("count", [0, 3, 2**31 - 1])
def test_restart_count_is_reported_on_the_instance_only(count):
    # TASK-1171: an API restart resets RestartCount to 0 and a policy restart
    # increments it; the handoff uses it to tell the two apart.
    row = {**container("bawthub-tts-1"), "RestartCount": count}
    observed = observe_container(row)
    assert observed.restart_count == count and observed.instance_api()["restartCount"] == count
    assert "restartCount" not in observed.to_api()


@pytest.mark.parametrize("bad", [True, False, -1, 2**31, 1.0, "1", [1]])
def test_ill_typed_restart_count_fails_closed(bad):
    with pytest.raises(ValueError):
        observe_container({**container("bawthub-tts-1"), "RestartCount": bad})


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
    # Codex r1 B1: unhashable selector / non-mapping properties or schema.
    op("x.listarg", {"action": "restart", "compose_project": "p", "compose_service_from_arg": []},
       {"type": "object", "properties": {}}),
    op("x.propslist", {"action": "restart", "compose_project": "p", "compose_service_from_arg": "s"},
       {"type": "object", "properties": ["not a map"]}),
    op("x.schemalist", {"action": "restart", "compose_project": "p", "compose_service_from_arg": "s"},
       ["not a map"]),
    # Codex r1 B2: an over-long selector name never reaches the response.
    op("x.widearg", {"action": "restart", "compose_project": "p", "compose_service_from_arg": "s" * 65},
       enum_schema("s" * 65, ["a"])),
    ("x.deep", json.dumps({"action": "restart"}), "[" * 100000 + "]" * 100000),
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
    assert (body["inventoryFresh"], body["catalogFresh"]) == (False, True)
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


def test_unavailable_catalog_keeps_observed_services_with_unknown_capabilities(world):
    # S2a: a stale catalog leaves the inventory half usable. Observed services
    # are listed with real instances and capabilities null (actions unknown);
    # catalog-only services (no instances) are not knowable and not listed.
    _, state, _, _, _ = world
    state.publish_catalog(CatalogSample(state.sample_time(), "down", "operation catalog unavailable"))
    code, body = state.services()
    assert code == 503 and body["fresh"] is False and body["reason"] == "operation catalog unavailable"
    assert (body["inventoryFresh"], body["catalogFresh"]) == (True, False)
    assert body["unlabeledCount"] == 1  # inventory itself is still fresh
    assert [(s["id"], s["capabilities"]) for s in body["services"]] == [("bawthub.tts", None)]
    assert [i["name"] for i in body["services"][0]["instances"]] == ["bawthub-tts-1"]
    code, body = state.services("bawthub.tts")
    assert code == 503 and body["service"]["capabilities"] is None
    assert len(body["service"]["instances"]) == 1
    # Catalogued but unobserved, or never seen: unknown, never a 404.
    for unknown in ("llm-bawt.crawl4ai", "bawthub.frontend"):
        assert state.services(unknown) == (503, {**body, "service": None})


def test_aged_catalog_does_not_date_a_fresh_inventory(world):
    # Codex S2a r1 B1: catalog 31s old, inventory resampled now. The inventory
    # half is usable and carries its own fresh timestamp; observedAt alone
    # (the older start) would make a consumer discard real instances.
    clock, state, _, _, probe = world
    start = clock.wall
    clock.advance(31)
    probe.collect_docker()
    code, body = state.services()
    assert code == 503 and (body["inventoryFresh"], body["catalogFresh"]) == (True, False)
    assert body["inventoryObservedAt"] == clock.wall.isoformat()
    assert body["catalogObservedAt"] == start.isoformat()
    assert body["observedAt"] == start.isoformat()
    assert [i["name"] for i in body["services"][0]["instances"]] == ["bawthub-tts-1"]


def test_both_halves_unavailable_null_services(world):
    clock, state, _, _, _ = world
    clock.advance(31)
    code, body = state.services()
    assert code == 503 and body["services"] is None
    assert (body["inventoryFresh"], body["catalogFresh"]) == (False, False)
    assert state.services("bawthub.tts")[1]["service"] is None


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
    assert set(body) == {"schemaVersion", "observedAt", "inventoryObservedAt", "catalogObservedAt",
                         "fresh", "inventoryFresh", "catalogFresh", "reason", "unlabeledCount", "service"}
    assert state.services("bawthub.frontend") == (404, {"error": "not found"})


# --- S2a: catalog targets union observed Compose services ------------------

def test_observed_services_without_ops_are_listed_with_no_capabilities(world):
    _, state, client, rows, probe = world
    rows += [container("bawthub-backend-1", service="backend"),
             container("other-db-1", project="other", service="db"),
             container("other-db-2", project="other", service="db")]
    client.api.containers.return_value = [{"Id": str(i)} for i in range(len(rows))]
    probe.collect_docker()
    code, body = state.services()
    assert code == 200 and (body["inventoryFresh"], body["catalogFresh"]) == (True, True)
    by_id = {s["id"]: s for s in body["services"]}
    assert list(by_id) == sorted(by_id) == ["bawthub.backend", "bawthub.tts", "llm-bawt.crawl4ai",
                                            "llm-bawt.playwright-mcp", "other.db"]
    assert by_id["bawthub.backend"]["capabilities"] == [] and by_id["other.db"]["capabilities"] == []
    assert [i["name"] for i in by_id["other.db"]["instances"]] == ["other-db-1", "other-db-2"]
    assert by_id["bawthub.tts"]["capabilities"]  # catalogued caps unchanged
    assert by_id["llm-bawt.crawl4ai"]["instances"] == []  # catalogued, none present
    assert body["unlabeledCount"] == 1
    code, body = state.services("other.db")
    assert code == 200 and body["service"]["project"] == "other" and body["service"]["capabilities"] == []


def test_stale_inventory_single_lookup_is_unknown_not_absent(world):
    clock, state, _, _, _ = world
    clock.advance(20)
    state.publish_catalog(CatalogSample(state.sample_time(), "ok", services=catalog_targets(CATALOG)))
    clock.advance(11)
    code, body = state.services("bawthub.tts")
    assert code == 503 and body["service"]["instances"] is None and body["service"]["capabilities"]
    assert (body["inventoryFresh"], body["catalogFresh"]) == (False, True)
    # An uncatalogued id may be an observed service: unknown (503), not 404.
    code, body = state.services("bawthub.backend")
    assert code == 503 and body["service"] is None


def test_union_past_the_service_bound_fails_the_inventory_half_closed(world, monkeypatch):
    _, state, client, rows, probe = world
    monkeypatch.setattr("llm_bawt.ops.health_state.MAX_SERVICES", 4)
    rows += [container(f"x-s{i}-1", project="x", service=f"s{i}") for i in range(2)]
    client.api.containers.return_value = [{"Id": str(i)} for i in range(len(rows))]
    probe.collect_docker()
    code, body = state.services()
    assert code == 503 and body["reason"] == "service inventory exceeds bounded snapshot"
    assert (body["inventoryFresh"], body["catalogFresh"]) == (False, True)
    assert body["unlabeledCount"] is None
    # Never a truncated union: the catalogued services alone, instances unknown.
    assert [(s["id"], s["instances"]) for s in body["services"]] == [
        ("bawthub.tts", None), ("llm-bawt.crawl4ai", None), ("llm-bawt.playwright-mcp", None)]
    assert state.services("x.s0")[0:1] == (503,)


def test_union_past_the_byte_budget_fails_the_inventory_half_closed(world, monkeypatch):
    _, state, _, _, _ = world
    assert state.services()[0] == 200
    monkeypatch.setattr("llm_bawt.ops.health_state.MAX_SERVICES_BODY_BYTES", 100)
    code, body = state.services()
    assert code == 503 and body["reason"] == "service inventory exceeds bounded snapshot"
    assert [s["instances"] for s in body["services"]] == [None, None, None]


def test_widest_union_never_exceeds_consumers_body_limit(monkeypatch):
    # The largest admitted catalog next to a 1000-row max-width inventory
    # spread over the most distinct services: either it fits or the
    # inventory half fails closed. The body stays under 512 KiB either way.
    monkeypatch.setattr("llm_bawt.ops.health_collector.DockerExecutor.readiness", Mock(return_value=None))
    n = max(i for i in range(1, MAX_CAPABILITIES // 64 + 1) if admits(max_width_catalog(i)))
    targets = catalog_targets(max_width_catalog(n))
    for distinct in (64, 255, 256, 257, 1000):
        rows = []
        for i in range(1000):
            row = container(f"{i:04d}".zfill(255), cid=f"{i:064d}", project="q" * 63,
                            service=f"{i % distinct:04d}" + "s" * 59, status="restarting")
            row["State"].update(StartedAt="2026-10-28T05:56:33.196123+14:00", Health={"Status": "unhealthy"})
            rows.append(row)
        state = Clock().state()
        OpsHealthCollector(state, lambda: None, client_factory=lambda: docker_client(rows)).collect_docker()
        state.publish_catalog(CatalogSample(state.sample_time(), "ok", services=targets))
        code, body = state.services()
        assert len(json.dumps(body, separators=(",", ":"), allow_nan=False).encode()) < 512 * 1024
        if code == 200:
            assert sum(len(s["instances"]) for s in body["services"]) == 1000
        else:
            assert body["reason"] == "service inventory exceeds bounded snapshot"


WIDE_PROJECT = "p" * 63
WIDE_SERVICES = [f"{j:02d}" + "s" * 61 for j in range(64)]


def max_width_catalog(n_ops):
    """``n_ops`` ops of 64 capabilities each, every emitted field at its max width."""
    arg = "a" * 64
    return [op(f"{i:03d}" + "o" * 125, {"action": "r" * 64, "compose_project": WIDE_PROJECT,
                                        "compose_service_from_arg": arg}, enum_schema(arg, WIDE_SERVICES))
            for i in range(n_ops)]


def admits(rows):
    try:
        catalog_targets(rows)
    except ValueError:
        return False
    return True


def test_rendered_catalog_byte_budget_fails_closed():
    # 512 max-width capabilities on 64 services pass every count bound, but
    # not the rendered-byte bound (Codex r1 B2).
    rows = max_width_catalog(MAX_CAPABILITIES // 64)
    assert not admits(rows)
    assert admits(max_width_catalog(1))


def test_maximum_catalog_and_inventory_fit_consumers_body_limit(monkeypatch):
    monkeypatch.setattr("llm_bawt.ops.health_collector.DockerExecutor.readiness", Mock(return_value=None))
    # The largest max-width catalog the byte budget admits, next to a
    # 1000-row inventory with every field at its widest.
    n = max(i for i in range(1, MAX_CAPABILITIES // 64 + 1) if admits(max_width_catalog(i)))
    targets = catalog_targets(max_width_catalog(n))
    rendered = json.dumps([t.to_api([]) for t in targets], separators=(",", ":")).encode()
    assert len(rendered) > MAX_CATALOG_BYTES * 3 // 4
    rows = []
    for i in range(1000):
        row = container(f"{i:04d}".zfill(255), cid=f"{i:064d}", project=WIDE_PROJECT,
                        service=WIDE_SERVICES[i % 64], status="restarting")
        row["State"].update(StartedAt="2026-10-28T05:56:33.196123+14:00", Health={"Status": "unhealthy"})
        rows.append(row)
    state = Clock().state()
    OpsHealthCollector(state, lambda: None, client_factory=lambda: docker_client(rows)).collect_docker()
    state.publish_catalog(CatalogSample(state.sample_time(), "ok", services=targets))
    code, body = state.services()
    assert code == 200 and sum(len(s["instances"]) for s in body["services"]) == 1000
    assert len(json.dumps(body, separators=(",", ":"), allow_nan=False).encode()) < 512 * 1024


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
