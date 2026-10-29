"""Access path -> provider connection ownership (TASK-1297).

The adapter registry is the one place that knows which provider credential
authorizes which ``access_paths`` row; the admin catalog and the setup wizard
both read it through :func:`access_path_connections`.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_bawt.memory.model_catalog_migration import STANDARD_ACCESS_PATHS
from llm_bawt.service.providers import registry
from llm_bawt.service.providers.base import ProviderAdapter

#: Stated independently of the adapters: access path -> owning provider id.
EXPECTED_OWNERS = {
    "anthropic-oauth": "claude",
    "anthropic-api": "anthropic-api",
    "openai-oauth": "codex",
    "openai-api": "openai-api",
    "openai-responses": "openai-api",
    "xai-chat": "xai",
    "xai-responses": "xai",
    "openrouter-anthropic": "openrouter",
    "openrouter-chat": "openrouter",
    # TASK-1476: bridge-proxy API keys moved into the encrypted store.
    "zai-anthropic": "zai",
    "kimi-coding-chat": "kimi-coding",
    "moonshot-anthropic": "moonshot",
}


def test_registry_owner_map_is_exactly_the_expected_one():
    assert {key: cls.id for key, cls in registry._ACCESS_PATH_OWNERS.items()} == EXPECTED_OWNERS


def test_every_owned_access_path_is_a_standard_seeded_path():
    seeded = {seed.key for seed in STANDARD_ACCESS_PATHS}
    # openrouter-anthropic is seeded by TASK-1295 B1c; once it lands nothing may be missing.
    assert set(registry._ACCESS_PATH_OWNERS) - seeded <= {"openrouter-anthropic"}


def test_unauthenticated_standard_paths_have_no_owner():
    for seed in STANDARD_ACCESS_PATHS:
        if seed.auth_mechanism == "none":
            assert seed.key not in registry._ACCESS_PATH_OWNERS, seed.key


def test_two_adapters_claiming_one_path_fail_at_import():
    class A(ProviderAdapter):
        id = "a"
        access_path_keys = ("shared",)

    class B(ProviderAdapter):
        id = "b"
        access_path_keys = ("shared",)

    with patch.dict(registry._ADAPTER_CLASSES, {"a": A, "b": B}, clear=True):
        with pytest.raises(RuntimeError, match="'shared' is claimed by both 'a' and 'b'"):
            registry._access_path_owners()


def _owner(provider_id: str, health):
    calls = []

    class Owner(ProviderAdapter):
        id = provider_id
        label = provider_id.upper()

        def health(self):
            calls.append(self.id)
            if isinstance(health, Exception):
                raise health
            return health

    return Owner, calls


def test_connections_report_state_once_per_adapter_and_none_for_unowned():
    ok, ok_calls = _owner("one", {"state": "ok", "detail": None})
    warn, _ = _owner("two", {"state": "warning", "detail": "refresh failing"})
    off, _ = _owner("three", {"state": "unconfigured", "detail": "not connected"})
    owners = {"p1": ok, "p2": ok, "p3": warn, "p4": off}
    with patch.object(registry, "_ACCESS_PATH_OWNERS", owners):
        result = registry.access_path_connections(MagicMock(), ["p1", "p2", "p3", "p4", "local"])

    assert result == {
        "p1": {"provider_id": "one", "label": "ONE", "state": "ok", "detail": None, "connected": True},
        "p2": {"provider_id": "one", "label": "ONE", "state": "ok", "detail": None, "connected": True},
        "p3": {"provider_id": "two", "label": "TWO", "state": "warning",
               "detail": "refresh failing", "connected": True},
        "p4": {"provider_id": "three", "label": "THREE", "state": "unconfigured",
               "detail": "not connected", "connected": False},
        "local": None,
    }
    assert ok_calls == ["one"]


def test_one_failing_health_probe_does_not_hide_the_rest():
    broken, _ = _owner("broken", RuntimeError("boom"))
    fine, _ = _owner("fine", {"state": "ok"})
    with patch.object(registry, "_ACCESS_PATH_OWNERS", {"b": broken, "f": fine}):
        result = registry.access_path_connections(MagicMock(), ["b", "f"])

    assert result["b"] == {"provider_id": "broken", "label": "BROKEN", "state": "unknown",
                           "detail": "boom", "connected": False}
    assert result["f"]["connected"] is True
