"""The agent bridge's uv cache env is admitted exactly (TASK-2134)."""
import pytest

from llm_bawt.ops.source_state import BRIDGE_UV_CACHE_DIR, SourceStateError, _classify_env

MOUNTS = {"/home/bridge/dev": False, "/opt/agent-skills/bin": True}
BAKED = {"UV_CACHE_DIR": BRIDGE_UV_CACHE_DIR, "UV_LINK_MODE": "clone"}


def test_exact_pair_admitted_even_under_the_dev_mount():
    assert _classify_env(dict(BAKED), MOUNTS, set()) == []


@pytest.mark.parametrize(("name", "value"), [
    ("UV_CACHE_DIR", "/home/bridge/dev/.uv-cache/"),
    ("UV_CACHE_DIR", "/home/bridge/dev/llm-bawt/.uv-cache"),
    ("UV_CACHE_DIR", "/home/bridge/dev/../dev/.uv-cache"),
    ("UV_CACHE_DIR", "/tmp/uv"),
    ("UV_CACHE_DIR", ""),
    ("UV_LINK_MODE", "copy"),
    ("UV_LINK_MODE", "symlink"),
    ("UV_LINK_MODE", ""),
])
def test_other_values_refused(name, value):
    with pytest.raises(SourceStateError) as err:
        _classify_env({**BAKED, name: value}, MOUNTS, set())
    assert err.value.code == "source_env_unsafe"
