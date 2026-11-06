"""TASK-2088: ``init`` and the memory cap on the built and mutable-tag kinds
(the agent bridges and playwright-mcp, TASK-2072). Pure functions, no Docker."""
from __future__ import annotations

import pytest

from llm_bawt.ops.compose_config import (
    BUILT_PROFILE, DEV_PROFILE, HARDENED_PROFILE, PINNED_PROCESS_PROFILE, apply_desired, canonical_hash, drift,
    needs_init_memory_worker, normalized, running, translate,
)
from llm_bawt.ops.compose_render import ComposeError

PROJECT, SERVICE, NAME = "llm-bawt", "codex-bridge", "llm-bawt-codex-bridge"
SPEC = {"compose_project": PROJECT, "compose_service": SERVICE, "container_name": NAME}
GIB6 = 6 * 1024 ** 3
IMAGE = {"Config": {"Env": ["PATH=/usr/bin"], "Labels": {}, "Cmd": None, "Entrypoint": None}}


def _rendered(**changes):
    # The shape `docker compose config --format json` gives the codex bridge.
    service = {"build": {"context": "/srv/llm-bawt", "dockerfile": "Dockerfile.agent-bridge"},
               "image": "llm-bawt-agent-bridge", "container_name": NAME,
               "command": ["python", "-m", "codex_bridge"], "environment": {"A": "1"},
               "networks": {"default": None}, "restart": "unless-stopped",
               "init": True, "mem_limit": str(GIB6)}
    for key, value in changes.items():
        if value is None:
            service.pop(key, None)
        else:
            service[key] = value
    return {"services": {SERVICE: service}}


def _container(**host):
    return {"Config": {"Env": ["PATH=/usr/bin", "A=1"], "Labels": {}, "Cmd": ["python", "-m", "codex_bridge"],
                       "Entrypoint": None, "OpenStdin": False, "Tty": False},
            "HostConfig": {"RestartPolicy": {"Name": "unless-stopped"}, "Init": True,
                           "Memory": GIB6, "MemorySwap": 2 * GIB6, **host},
            "NetworkSettings": {"Networks": {f"{PROJECT}_default": {}}}}


def _drift(rendered, container, profile=BUILT_PROFILE):
    desired = translate(rendered, SPEC, "unused", profile)
    return drift(normalized(desired, IMAGE), normalized(running(container), IMAGE))


def test_the_live_bridge_shape_admits_and_matches_the_container_compose_made():
    desired = translate(_rendered(), SPEC, "unused", BUILT_PROFILE)
    assert desired["init"] is True and desired["memory"] == {"Memory": GIB6, "MemorySwap": 0}
    # Compose sends MemorySwap 0 for a lone mem_limit; the daemon sets twice the limit.
    assert _drift(_rendered(), _container()) == []


@pytest.mark.parametrize("host, name", [
    ({"Init": None}, "init:~"), ({"Init": False}, "init:~"),
    ({"Memory": GIB6 // 2, "MemorySwap": GIB6}, "memory:~"), ({"MemorySwap": -1}, "memory:~"),
    ({"Memory": 0, "MemorySwap": 0}, "memory:~"),
])
def test_init_and_memory_drift_is_detected_by_name(host, name):
    assert _drift(_rendered(), _container(**host)) == [name]


def test_an_undeclared_init_on_the_container_is_drift_and_is_removed():
    rendered = _rendered(init=None)
    assert _drift(rendered, _container()) == ["init:~"]
    assert _drift(rendered, _container(Init=None)) == []


def test_no_memory_env_renders_no_cap_and_keeps_the_old_desired_configuration():
    # TASK-2072 renders `${..._MEM_LIMIT:-0}` as no mem_limit at all.
    rendered = _rendered(mem_limit=None, init=None)
    desired = translate(rendered, SPEC, "unused", BUILT_PROFILE)
    assert "memory" not in desired and "init" not in desired
    before = {k: v for k, v in translate(_rendered(), SPEC, "unused", BUILT_PROFILE).items()
              if k not in ("memory", "init")}
    assert canonical_hash(desired) == canonical_hash(before)
    assert _drift(rendered, _container(Init=None, Memory=0, MemorySwap=0)) == []


def test_the_recreate_applies_init_and_the_cap():
    desired = translate(_rendered(), SPEC, "unused", BUILT_PROFILE)
    old = _container(Init=None, Memory=0, MemorySwap=0)
    body = {"Labels": {}, "HostConfig": dict(old["HostConfig"])}
    out = apply_desired(body, old, desired, container_name=NAME, service=SERVICE, rendered_hash="h", image=IMAGE)
    host = out["HostConfig"]
    assert host["Init"] is True and host["Memory"] == GIB6 and host["MemorySwap"] == 0
    undeclared = translate(_rendered(init=None, mem_limit=None), SPEC, "unused", BUILT_PROFILE)
    out = apply_desired({"Labels": {}, "HostConfig": dict(_container()["HostConfig"])}, old, undeclared,
                        container_name=NAME, service=SERVICE, rendered_hash="h", image=IMAGE)
    assert out["HostConfig"]["Init"] is None  # the daemon default; the cloned cap is carried as before


@pytest.mark.parametrize("changes, names", [
    ({"init": "yes"}, ["init"]), ({"mem_limit": "0"}, ["mem_limit"]), ({"mem_limit": "6g"}, ["mem_limit"]),
    ({"mem_limit": None, "memswap_limit": str(GIB6)}, ["mem_limit"]),
    ({"memswap_limit": str(GIB6 - 1)}, ["memswap_limit"]),
])
def test_malformed_init_or_memory_refuses_by_name(changes, names):
    with pytest.raises(ComposeError) as caught:
        translate(_rendered(**changes), SPEC, "unused", BUILT_PROFILE)
    assert caught.value.code == "unsupported_compose_key" and caught.value.names == names


def test_the_mutable_tag_kind_admits_init_and_the_cap():
    # playwright-mcp's base service: `mcp/playwright`, init, lone mem_limit.
    rendered = {"services": {SERVICE: {"image": "mcp/playwright", "container_name": NAME,
                                       "networks": {"default": None}, "init": True, "mem_limit": str(GIB6)}}}
    desired = translate(rendered, SPEC, "mcp/playwright", PINNED_PROCESS_PROFILE)
    assert desired["init"] is True and desired["memory"] == {"Memory": GIB6, "MemorySwap": 0}


def test_the_worker_gate_covers_only_what_an_older_worker_refuses():
    built = translate(_rendered(), SPEC, "unused", BUILT_PROFILE)
    assert needs_init_memory_worker(built, BUILT_PROFILE)
    assert needs_init_memory_worker({"memory": {"Memory": GIB6, "MemorySwap": GIB6}}, BUILT_PROFILE)
    assert not needs_init_memory_worker(translate(_rendered(init=None, mem_limit=None), SPEC, "unused",
                                                  BUILT_PROFILE), BUILT_PROFILE)
    # The dev/hardened kinds already modelled a declared pair (TASK-1649/1801).
    for profile in (DEV_PROFILE, HARDENED_PROFILE):
        assert not needs_init_memory_worker({"memory": {"Memory": GIB6, "MemorySwap": GIB6}}, profile)
        assert not needs_init_memory_worker({"memory": {"Memory": 0, "MemorySwap": 0}}, profile)
        assert needs_init_memory_worker({"memory": {"Memory": GIB6, "MemorySwap": 0}}, profile)
