"""TASK-1802: the live preview probe's pure logic, on fake ``docker inspect`` data."""

from __future__ import annotations

import pytest

from llm_bawt.preview import probes, topology
from llm_bawt.preview.probes import inventory_findings, parse_ports, parse_targets, prod_targets


def _container(service, networks, *, running=True, ports=None, exposed=None, ip="172.30.0.5"):
    return {
        "Name": f"/llm-bawt-{service}-1",
        "Config": {"Labels": {probes.SERVICE_LABEL: service}, "ExposedPorts": exposed or {}, "Env": []},
        "State": {"Running": running},
        "NetworkSettings": {"Networks": {n: {"IPAddress": ip} for n in networks}, "Ports": ports or {}},
        "HostConfig": {},
        "Mounts": [],
    }


def _healthy():
    name = topology.NETWORK_NAMES
    return [
        _container("preview-postgres", [name["preview"]]),
        _container("preview-redis", [name["preview"]]),
        _container("preview-credential-gateway", [name["preview-broker-net"], name["preview-credential-net"]]),
        _container("app", ["llm-bawt_default", name["preview-broker-net"]], exposed={"8642/tcp": {}}),
    ]


def _failures(containers):
    return [label for ok, label in inventory_findings(containers) if not ok]


def test_healthy_topology_passes():
    assert _failures(_healthy()) == []


def test_gateway_on_wrong_network_fails():
    containers = _healthy()
    containers[2]["NetworkSettings"]["Networks"][topology.NETWORK_NAMES["preview"]] = {"IPAddress": "x"}
    assert any("preview-credential-gateway" in f and "networks" in f for f in _failures(containers))


def test_service_drifted_off_preview_network_is_still_inspected():
    containers = _healthy()
    containers[1]["NetworkSettings"]["Networks"] = {"llm-bawt_default": {"IPAddress": "x"}}
    assert any("preview-redis" in f and "networks" in f for f in _failures(containers))


def test_missing_required_service_fails():
    containers = [c for c in _healthy() if _service(c) != "preview-redis"]
    assert any("required preview services" in f for f in _failures(containers))


def test_prod_service_on_preview_network_fails():
    containers = _healthy()
    containers.append(_container("redis", ["llm-bawt_default", topology.NETWORK_NAMES["preview"]]))
    assert any(f.startswith("redis: off preview networks") for f in _failures(containers))


def test_parse_targets():
    assert parse_targets("db=10.1.2.3:5432; app=host:8642;") == [("db", "10.1.2.3", 5432), ("app", "host", 8642)]
    assert parse_targets(None) == []
    with pytest.raises(ValueError):
        parse_targets("db=nohost")


def test_prod_targets_cover_non_preview_addresses_and_ports():
    targets, uncovered = prod_targets(_healthy())
    assert targets and {label for label, _, _ in targets} == {"prod app"} and uncovered == []
    assert {port for _, _, port in targets} == {8642}


def test_undeclared_bridge_port_is_still_probed():
    # Codex r2 cmvvdv0kl00xj0mpd3nanq9ui: the bridges EXPOSE nothing.
    containers = [*_healthy(), _container("claude-code-bridge", ["llm-bawt_default"], ip="172.30.0.9")]
    targets, uncovered = prod_targets(containers, {8681})
    assert ("prod claude-code-bridge", "172.30.0.9", 8681) in targets
    assert ("prod claude-code-bridge", "172.30.0.9", 8642) in targets  # any project port, any prod address
    assert uncovered == []


def test_prod_container_without_any_probe_port_is_reported():
    containers = [_container("claude-code-bridge", ["llm-bawt_default"])]
    assert prod_targets(containers) == ([], ["claude-code-bridge"])


def test_bound_ports_count():
    c = _container("app", ["llm-bawt_default"])
    c["HostConfig"]["PortBindings"] = {"8001/tcp": [{"HostPort": "8001"}]}
    assert {p for _, _, p in prod_targets([c])[0]} == {8001}


def test_parse_ports():
    assert parse_ports("8681, 8682") == {8681, 8682}
    assert parse_ports(None) == set()
    with pytest.raises(ValueError):
        parse_ports("86x1")


def _service(c):
    return c["Config"]["Labels"][probes.SERVICE_LABEL]
