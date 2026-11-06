"""TASK-2076: optional ``cgroup_parent`` (a host slice such as agents.slice) on
the built and mutable-tag kinds. Pure functions, no Docker."""
from __future__ import annotations

import pytest

from llm_bawt.ops.compose_config import (
    BUILT_PROFILE, PINNED_PROCESS_PROFILE, PINNED_PROFILE, apply_desired, canonical_hash, needs_cgroup_parent_worker,
    translate,
)
from llm_bawt.ops.compose_render import ComposeError
from tests.test_ops_init_memory_profile import IMAGE, NAME, SERVICE, SPEC, _container, _drift, _rendered

SLICE = "agents.slice"


def test_unset_parent_keeps_the_desired_configuration_and_its_hash():
    # Compose omits an empty `${AGENTS_CGROUP_PARENT:-}` entirely.
    desired = translate(_rendered(), SPEC, "unused", BUILT_PROFILE)
    assert "cgroup_parent" not in desired
    assert canonical_hash(desired) == canonical_hash(translate(_rendered(cgroup_parent=""), SPEC, "unused",
                                                                BUILT_PROFILE))
    assert _drift(_rendered(), _container()) == []
    assert not needs_cgroup_parent_worker(desired)


def test_a_declared_parent_is_admitted_compared_and_applied():
    rendered = _rendered(cgroup_parent=SLICE)
    desired = translate(rendered, SPEC, "unused", BUILT_PROFILE)
    assert desired["cgroup_parent"] == SLICE and needs_cgroup_parent_worker(desired)
    assert _drift(rendered, _container()) == ["cgroup_parent:~"]
    assert _drift(rendered, _container(CgroupParent=SLICE)) == []
    old = _container()
    out = apply_desired({"Labels": {}, "HostConfig": dict(old["HostConfig"])}, old, desired,
                        container_name=NAME, service=SERVICE, rendered_hash="h", image=IMAGE)
    assert out["HostConfig"]["CgroupParent"] == SLICE


def test_an_undeclared_parent_on_the_container_is_drift_and_is_removed():
    assert _drift(_rendered(), _container(CgroupParent=SLICE)) == ["cgroup_parent:~"]
    assert needs_cgroup_parent_worker({}, ["cgroup_parent:~"])
    old = _container(CgroupParent=SLICE)
    out = apply_desired({"Labels": {}, "HostConfig": dict(old["HostConfig"])}, old,
                        translate(_rendered(), SPEC, "unused", BUILT_PROFILE),
                        container_name=NAME, service=SERVICE, rendered_hash="h", image=IMAGE)
    assert out["HostConfig"]["CgroupParent"] == ""


def test_the_mutable_tag_kind_admits_a_parent_and_the_pinned_kind_still_refuses_it():
    rendered = {"services": {SERVICE: {"image": "mcp/playwright", "container_name": NAME,
                                       "networks": {"default": None}, "restart": "unless-stopped",
                                       "cgroup_parent": SLICE}}}
    assert translate(rendered, SPEC, "mcp/playwright", PINNED_PROCESS_PROFILE)["cgroup_parent"] == SLICE
    with pytest.raises(ComposeError) as caught:
        translate(rendered, SPEC, "mcp/playwright", PINNED_PROFILE)
    assert caught.value.code == "unsupported_compose_key" and "cgroup_parent" in caught.value.names
