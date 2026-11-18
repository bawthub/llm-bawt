"""The agent bridge image's baked preload env is admitted exactly (TASK-2097)."""
import pytest

from llm_bawt.ops.source_state import (
    BRIDGE_NODE_OPTIONS, BRIDGE_NODE_PRELOAD, SourceStateError, _classify_env, reserved_env_name)

MOUNTS = {"/app/src": False, "/opt/agent-skills/bin": True}
BAKED = {"BAWT_AGENT_BRIDGE": "1", "NODE_OPTIONS": BRIDGE_NODE_OPTIONS}


def test_exact_bridge_pairs_admitted():
    assert _classify_env(dict(BAKED), MOUNTS, set()) == []


@pytest.mark.parametrize(("name", "value"), [
    ("BAWT_AGENT_BRIDGE", "0"),
    ("BAWT_AGENT_BRIDGE", "true"),
    ("NODE_OPTIONS", "--require=/tmp/evil.cjs"),
    ("NODE_OPTIONS", BRIDGE_NODE_OPTIONS + " --require=/tmp/evil.cjs"),
    ("NODE_OPTIONS", f"-r {BRIDGE_NODE_PRELOAD}"),
    ("NODE_OPTIONS", "--inspect=0.0.0.0:9229"),
])
def test_other_values_refused(name, value):
    with pytest.raises(SourceStateError) as err:
        _classify_env({**BAKED, name: value}, MOUNTS, set())
    assert err.value.code == "source_env_unsafe"


def test_preload_under_a_mount_refused():
    with pytest.raises(SourceStateError):
        _classify_env(dict(BAKED), {**MOUNTS, "/usr/local/libexec": True}, set())


@pytest.mark.parametrize(("name", "code"), [
    ("NODE_PATH", "source_env_unclassified"),
    ("NODE_EXTRA_CA_CERTS", "source_env_unclassified"),
    ("NODE_ENV", "source_env_unsafe"),
])
def test_generic_node_names_stay_refused(name, code):
    with pytest.raises(SourceStateError) as err:
        _classify_env({**BAKED, name: "x"}, MOUNTS, set())
    assert err.value.code == code
    assert reserved_env_name(name)  # and cannot be listed as app data either
