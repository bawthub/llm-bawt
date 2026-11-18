"""TASK-2469: the claude-code-bridge rows classify every mount the bridge's
Compose service declares, including the ${HOME}/dev identical-path view."""
from __future__ import annotations

import copy
import json

import pytest

from llm_bawt.ops.source_state import (SourceStateError, SourceStateRefusals, resolved_data_mounts,
                                       source_plan, valid_classification, validate_source_spec)
from tests.ops_install_rows import SEEDS

HOME = "/home/operator"
PROJECT = f"{HOME}/dev/llm-bawt"
SLUGS = ("llm-bawt.reconcile-claude-code-bridge-config", "llm-bawt.deploy-claude-code-bridge-image",
         "llm-bawt.rollback-claude-code-bridge-image")


def _spec(slug, home=HOME):
    spec = json.loads(next(s for s in SEEDS if s["slug"] == slug)["command_script"])
    if home is not None:
        spec["compose_config"] = {**spec["compose_config"], "interpolation_env": {"HOME": home}}
    return spec


def _codes(spec, destinations):
    desired = {"mounts": {d: {"type": "bind", "source": f"{PROJECT}/src" if d == "/app/src" else "/srv/x",
                              "read_only": False} for d in destinations},
               "env": {}, "tmpfs": {}}
    try:
        source_plan(spec, {"project_dir": PROJECT, "compose_files": [f"{PROJECT}/docker-compose.yml"]}, {}, {}, desired)
    except SourceStateRefusals as caught:
        return {r.code: r.names for r in caught.refusals}
    except SourceStateError as caught:
        return {caught.code: caught.names}
    return {}


# The bridge service's destinations in the product's docker-compose files.
LIVE = ["/app/src", "/app/.logs", "/home/bridge/dev", f"{HOME}/dev", "/home/bridge/.claude",
        "/home/bridge/.claude.json", "/home/bridge/.config/bawthub-git", "/home/bridge/.ssh", "/etc/ssh/ssh_config",
        "/opt/agent-skills/bin", "/run/llm-bawt-bridge-tokens", "/run/llm-bawt-api-key"]


@pytest.mark.parametrize("slug", SLUGS)
def test_bridge_rows_classify_every_declared_mount(slug):
    spec = _spec(slug)
    validate_source_spec(spec)
    codes = _codes(spec, LIVE)
    assert "source_mount_unclassified" not in codes and "source_mount_shadowed" not in codes


def test_home_mount_resolves_from_interpolation_env_only():
    assert f"{HOME}/dev" in resolved_data_mounts(_spec(SLUGS[0]))
    assert "${HOME}/dev" in _spec(SLUGS[0])["data_mounts"]  # the stored row stays host-neutral


def test_without_declared_home_the_seed_validates_but_the_mount_is_refused():
    spec = _spec(SLUGS[1], home=None)
    validate_source_spec(spec)
    assert _codes(spec, LIVE).get("source_mount_unclassified") == [f"{HOME}/dev"]


def test_a_different_home_does_not_classify_this_hosts_mount():
    assert _codes(_spec(SLUGS[1], home="/home/other"), LIVE).get("source_mount_unclassified") == [f"{HOME}/dev"]


def test_classification_accepts_the_resolved_destination():
    spec = _spec(SLUGS[0])
    value = {"source_mounts": ["/app/src"], "data_mounts": sorted([f"{HOME}/dev", "/app/.logs"]), "opaque_env": []}
    assert valid_classification(spec, value)
    stale = copy.deepcopy(value)
    stale["data_mounts"] = sorted(["/home/elsewhere/dev", "/app/.logs"])
    assert not valid_classification(spec, stale)
