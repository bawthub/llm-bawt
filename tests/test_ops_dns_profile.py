"""TASK-2171: optional ordered ``dns`` upstreams on the built kind (the agent
bridges). Pure functions, no Docker."""
from __future__ import annotations

import pytest

from llm_bawt.ops.compose_config import (
    BUILT_PROFILE, PINNED_PROFILE, apply_desired, canonical_hash, needs_dns_worker, translate,
)
from llm_bawt.ops.compose_render import ComposeError
from tests.test_ops_init_memory_profile import IMAGE, NAME, SERVICE, SPEC, _container, _drift, _rendered

DNS = ["10.0.0.3", "10.0.0.1"]


def _apply(old, desired):
    return apply_desired({"Labels": {}, "HostConfig": dict(old["HostConfig"])}, old, desired,
                         container_name=NAME, service=SERVICE, rendered_hash="h", image=IMAGE)


def test_unset_dns_keeps_the_desired_configuration_and_its_hash():
    desired = translate(_rendered(), SPEC, "unused", BUILT_PROFILE)
    assert "dns" not in desired and not needs_dns_worker(desired)
    assert canonical_hash(desired) == canonical_hash(translate(_rendered(dns=[]), SPEC, "unused", BUILT_PROFILE))
    assert _drift(_rendered(), _container()) == []
    assert _apply(_container(), desired)["HostConfig"]["Dns"] is None


def test_declared_dns_is_admitted_in_order_compared_and_applied():
    rendered = _rendered(dns=DNS)
    desired = translate(rendered, SPEC, "unused", BUILT_PROFILE)
    assert desired["dns"] == DNS and needs_dns_worker(desired)
    assert _drift(rendered, _container()) == ["dns:~"]
    assert _drift(rendered, _container(Dns=list(reversed(DNS)))) == ["dns:~"]  # order matters
    assert _drift(rendered, _container(Dns=DNS)) == []
    assert _apply(_container(), desired)["HostConfig"]["Dns"] == DNS


def test_undeclared_dns_on_the_container_is_drift_and_is_removed():
    assert _drift(_rendered(), _container(Dns=DNS)) == ["dns:~"]
    assert needs_dns_worker({}, ["dns:~"])
    desired = translate(_rendered(), SPEC, "unused", BUILT_PROFILE)
    assert _apply(_container(Dns=DNS), desired)["HostConfig"]["Dns"] is None


@pytest.mark.parametrize("bad", ["resolver.lan", ["10.0.0.3", "10.0.0.3"], ["1.1.1.1", "1.0.0.1", "8.8.8.8", "9.9.9.9"],
                                 [53], {"a": "10.0.0.3"}])
def test_invalid_dns_is_refused(bad):
    with pytest.raises(ComposeError) as caught:
        translate(_rendered(dns=bad), SPEC, "unused", BUILT_PROFILE)
    assert caught.value.code == "unsupported_compose_key" and "dns" in caught.value.names


def test_the_pinned_kind_still_refuses_dns():
    rendered = {"services": {SERVICE: {"image": "x@sha256:" + "0" * 64, "container_name": NAME,
                                       "networks": {"default": None}, "restart": "unless-stopped", "dns": DNS}}}
    with pytest.raises(ComposeError) as caught:
        translate(rendered, SPEC, "x@sha256:" + "0" * 64, PINNED_PROFILE)
    assert caught.value.code == "unsupported_compose_key" and "dns" in caught.value.names
